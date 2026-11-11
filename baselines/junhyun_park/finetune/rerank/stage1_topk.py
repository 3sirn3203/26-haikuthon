"""재정렬기 학습·검증에 쓸 1단계(FG-CLIP) 후보 목록을 만든다.

FG-CLIP 원격코드는 transformers==4.46.3 이 필요하고 재정렬기(Qwen3.5)는 5.x 가 필요하다. 그래서 이 스크립트는
4.46.3 을 별도 폴더에 설치해 PYTHONPATH 로 앞세워 따로 실행한다 (노트북 참고).

  PYTHONPATH=/content/tf446 python stage1_topk.py --stage1 <FT 폴더 | HF repo> --data /content/data/train_v1 \
      --val_ids val_ids.txt --out stage1_topk.json

출력 JSON
  train_topk: {학습 프롬프트 id: [학습 원본 이미지 id x k_train]}  재정렬기 학습용 '1단계가 헷갈린 오답' 후보
  val_topk:   {val 프롬프트 id: [val 후보 풀 이미지 id x k_val]}   재정렬 대상 (풀 = val 원본 500 + val HN)
  val_gt_rank:{val 프롬프트 id: 1단계 정답 순위(1부터, 동점 비관적)}
  stage1_val: 1단계 단독 val 점수, R@5/R@10
이미지 id 는 패키지 파일명(원본 '0005', HN '4005')이다.

test 모드 (--test, val_ids 불필요): 데이터 = test_v1, 후보 풀 = 전체 15,000장(원본 + HN), 질의 = 원본 프롬프트 8,000.
  test_topk:        {test 프롬프트 id: [이미지 id x k_test]}   재정렬 대상
  test_topk_scores: {test 프롬프트 id: [코사인 유사도 x k_test]}
  stage1_test:      1단계 단독 test 점수 (정답 = 같은 id, 서버 채점과 같은 규칙), R@5/R@10
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class ImgDS(Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.proc(images=im.convert("RGB"), return_tensors="pt")["pixel_values"][0]


def ensure_remote_code(stage1, base):
    """파인튜닝 폴더에 원격코드(*.py)가 없으면 원본 repo 에서 복사한다."""
    d = Path(stage1)
    if d.is_dir() and not list(d.glob("*.py")):
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(base, allow_patterns=["*.py"]))
        for f in snap.glob("*.py"):
            (d / f.name).write_bytes(f.read_bytes())
        print(f"[stage1] {base} 원격코드를 {d} 로 복사")


def metrics(ranks):
    n = len(ranks)
    r1 = sum(r == 1 for r in ranks) / n
    nd = sum(1 / math.log2(r + 1) for r in ranks if r <= 5) / n
    return {"n": n, "recall_at_1": round(r1, 4), "ndcg_at_5": round(nd, 4), "score": round((r1 + nd) / 2, 4),
            "recall_at_5": round(sum(r <= 5 for r in ranks) / n, 4),
            "recall_at_10": round(sum(r <= 10 for r in ranks) / n, 4)}


def encode(model, tok, proc, dev, max_len, walk, img_paths, texts):
    """이미지·텍스트 정규화 임베딩 (float32, GPU)."""
    dl = DataLoader(ImgDS(img_paths, proc), batch_size=128, num_workers=4)
    feats, tfe = [], []
    with torch.inference_mode():
        for px in dl:
            feats.append(F.normalize(model.get_image_features(px.to(dev, torch.bfloat16)).float(), dim=-1))
        for s in range(0, len(texts), 256):
            t = tok(texts[s:s + 256], padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")["input_ids"].to(dev)
            tfe.append(F.normalize(model.get_text_features(t, walk_short_pos=walk).float(), dim=-1))
    return torch.cat(feats), torch.cat(tfe)


def run_test(a, model, tok, proc, dev, max_len, walk, text, orig, hn, files):
    import time
    ids = orig + hn
    print(f"[stage1-test] {a.stage1} | 질의 {len(orig)} / 후보 풀 {len(ids)}")
    t0 = time.time()
    img, txt = encode(model, tok, proc, dev, max_len, walk, [files[i] for i in ids], [text[i] for i in orig])
    t_enc = time.time() - t0
    S = txt @ img.T                                            # (질의, 풀)
    top_s, top_i = S.topk(a.k_test, dim=1)
    pos = {i: j for j, i in enumerate(ids)}
    gt = S[torch.arange(len(orig)), torch.tensor([pos[q] for q in orig])]
    ranks = (S >= gt[:, None]).sum(1).tolist()                 # 정답 = 같은 id, 비관적 순위
    res = {"stage1": a.stage1, "mode": "test", "k_test": a.k_test,
           "test_topk": {q: [ids[j] for j in row] for q, row in zip(orig, top_i.tolist())},
           "test_topk_scores": {q: [round(x, 6) for x in row] for q, row in zip(orig, top_s.tolist())},
           "stage1_test": metrics(ranks), "encode_sec": round(t_enc, 1)}
    Path(a.out).write_text(json.dumps(res))
    print("[stage1-test] 1단계 단독 test:", res["stage1_test"], f"| encode {t_enc:.0f}s")
    print("->", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True, help="FG-CLIP 파인튜닝 폴더 또는 HF repo id")
    ap.add_argument("--base", default="qihoo360/fg-clip-base")
    ap.add_argument("--data", required=True, help="train_v1 폴더 (prompts.jsonl, images/)")
    ap.add_argument("--val_ids", help="train 모드에서 필수")
    ap.add_argument("--test", action="store_true", help="test_v1 전체 풀에서 원본 프롬프트별 top-k")
    ap.add_argument("--k_test", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k_train", type=int, default=20)
    ap.add_argument("--k_val", type=int, default=20)
    ap.add_argument("--long", action="store_true", help="248토큰 모드")
    a = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
    ensure_remote_code(a.stage1, a.base)
    dev = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(a.stage1, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
    tok = AutoTokenizer.from_pretrained(a.stage1, trust_remote_code=True)
    proc = AutoImageProcessor.from_pretrained(a.stage1, trust_remote_code=True)
    max_len, walk = (248, False) if a.long else (77, True)

    data = Path(a.data)
    rows = [json.loads(l) for l in open(data / "prompts.jsonl", encoding="utf-8") if l.strip()]
    text = {r["id"]: r.get("prompts", r.get("prompt")) for r in rows}
    orig = sorted(i for i, t in text.items() if t != "hard_negative")
    hn = sorted(i for i, t in text.items() if t == "hard_negative")
    width = len(orig[0])
    hn_src = {h: f"{int(h) - len(orig):0{width}d}" for h in hn}       # chan_lee pair_hn.py 와 같은 순서 규칙
    files = {p.stem: p for p in (data / "images").rglob("*.png")}
    if a.test:
        return run_test(a, model, tok, proc, dev, max_len, walk, text, orig, hn, files)
    val = {f"{int(l):0{width}d}" for l in open(a.val_ids) if l.strip()}
    fit = [i for i in orig if i not in val]
    val = sorted(val)
    val_pool = val + [h for h in hn if hn_src[h] in set(val)]
    print(f"[stage1] {a.stage1} | fit {len(fit)} / val {len(val)} / val pool {len(val_pool)}")

    ids = orig + hn
    img_t, txt_t = encode(model, tok, proc, dev, max_len, walk, [files[i] for i in ids], [text[i] for i in orig])
    img = dict(zip(ids, img_t))
    txt = dict(zip(orig, txt_t))

    def topk(queries, pool, k):
        P = torch.stack([img[i] for i in pool])
        out, ranks = {}, {}
        for q in queries:
            s = P @ txt[q]
            out[q] = [pool[j] for j in s.topk(min(k, len(pool))).indices.tolist()]
            if q in pool:
                ranks[q] = int((s >= s[pool.index(q)]).sum().item())   # 정답 이상 점수 개수 = 비관적 순위
        return out, ranks

    train_topk, _ = topk(fit, fit, a.k_train)
    val_topk, val_rank = topk(val, val_pool, a.k_val)
    res = {"stage1": a.stage1, "k_train": a.k_train, "k_val": a.k_val, "val_pool": val_pool,
           "train_topk": train_topk, "val_topk": val_topk, "val_gt_rank": val_rank,
           "stage1_val": metrics([val_rank[q] for q in val])}
    Path(a.out).write_text(json.dumps(res))
    print("[stage1] val (1단계 단독):", res["stage1_val"])
    print("->", a.out)


if __name__ == "__main__":
    main()
