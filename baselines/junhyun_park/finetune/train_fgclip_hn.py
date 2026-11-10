"""FG-CLIP v1 파인튜닝: in-batch 대조학습 (+ 선택적으로 text hard negative). Colab Pro 용.

--hn 0  : 대조군 (hard negative 없음)
--hn 2  : 프롬프트마다 규칙 기반 text HN 최대 2개를 image->text 로짓의 추가 negative 로 사용 (NegCLIP 방식)

예 (Colab):
  pip install -q "transformers==4.46.3" einops
  python train_fgclip_hn.py --prompts train_prompts_v2.jsonl --images /content/train_v1/images \
      --repo qihoo360/fg-clip-base --hn 2 --out runs/hn2
결과: runs/<name>/ (safetensors + 원격코드 + 토크나이저) -> build_fgclip_submission.py --repo runs/<name> 로 zip 생성.
검증 점수는 train 내부 holdout 이라 같은 묶음(bundle) 문항이 섞여 과대평가될 수 있다. 최종 판단은 서버/테스트셋 점수로.
"""
import argparse, json, math, random, re, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hn_rules import make_hard_negatives


def load_rows(prompts_path, images_dir):
    files = {p.stem: p for p in Path(images_dir).rglob("*.png")}
    rows = []
    for line in open(prompts_path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        text = r.get("prompt", r.get("prompts"))
        if not text or text == "hard_negative" or r["id"] not in files:
            continue
        rows.append({"id": r["id"], "prompt": text, "path": files[r["id"]]})
    return rows


class DS(Dataset):
    def __init__(self, rows, proc):
        self.rows, self.proc = rows, proc

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = Image.open(self.rows[i]["path"]).convert("RGB")
        return self.proc(images=img, return_tensors="pt")["pixel_values"][0], self.rows[i]["prompt"]


def collate(b):
    return torch.stack([x[0] for x in b]), [x[1] for x in b]


def set_trainable(model, last_n):
    """마지막 last_n 블록(비전/텍스트 각각) + 투영/정규화 층만 학습. 이름 패턴 기반이라 학습 파라미터 수를 출력해 확인."""
    for p in model.parameters():
        p.requires_grad = False
    layer_re = re.compile(r"^(.*?)\.layers\.(\d+)\.")
    maxidx = {}
    for n, _ in model.named_parameters():
        m = layer_re.match(n)
        if m:
            maxidx[m.group(1)] = max(maxidx.get(m.group(1), -1), int(m.group(2)))
    for n, p in model.named_parameters():
        m = layer_re.match(n)
        if m and int(m.group(2)) > maxidx[m.group(1)] - last_n:
            p.requires_grad = True
        elif re.search(r"(projection|post_layernorm|final_layer_norm|ln_post|ln_final|visual_projection|text_projection)", n):
            p.requires_grad = True
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params {tr / 1e6:.1f}M / {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M; towers={maxidx}")


def encode_text(model, tok, texts, dev, max_len, walk):
    ids = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")["input_ids"].to(dev)
    return F.normalize(model.get_text_features(ids, walk_short_pos=walk).float(), dim=-1)


@torch.no_grad()
def evaluate(model, tok, val_rows, val_px, dev, max_len, walk, amp):
    model.eval()
    with torch.autocast(dev.type, dtype=amp, enabled=dev.type == "cuda"):
        img = torch.cat([F.normalize(model.get_image_features(val_px[i:i + 64].to(dev)).float(), dim=-1)
                         for i in range(0, len(val_px), 64)])
        txt = torch.cat([encode_text(model, tok, [r["prompt"] for r in val_rows[i:i + 256]], dev, max_len, walk)
                         for i in range(0, len(val_rows), 256)])
    sim = txt @ img.T
    gt = sim.diag()
    rank = (sim > gt[:, None]).sum(1).cpu().numpy()  # 정답보다 점수가 높은 이미지 수
    r1 = float((rank == 0).mean())
    nd = float(np.where(rank < 5, 1.0 / np.log2(rank + 2), 0.0).mean())
    return {"recall@1": round(r1, 4), "ndcg@5": round(nd, 4), "score": round((r1 + nd) / 2, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--repo", default="qihoo360/fg-clip-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hn", type=int, default=0, help="프롬프트당 text HN 개수 (0=대조군)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--last_n", type=int, default=4)
    ap.add_argument("--val", type=int, default=500)
    ap.add_argument("--val_ids", default="", help="번들 단위 val id 목록 (make_val_ids.py 출력). 없으면 무작위 분할(번들 누수 가능)")
    ap.add_argument("--long", action="store_true", help="긴 캡션 모드(max_length 248, walk_short_pos=False)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if (dev.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    max_len, walk = (248, False) if a.long else (77, True)

    rows = load_rows(a.prompts, a.images)
    if a.val_ids:
        ids = {l.strip() for l in open(a.val_ids) if l.strip()}
        norm = lambda x: str(int(x))
        ids = {norm(i) for i in ids}
        val_rows = [r for r in rows if norm(r["id"]) in ids]
        tr_rows = [r for r in rows if norm(r["id"]) not in ids]
        random.Random(a.seed).shuffle(tr_rows)
    else:
        random.Random(a.seed).shuffle(rows)
        val_rows, tr_rows = rows[:a.val], rows[a.val:]
    print(f"[data] train {len(tr_rows)} / val {len(val_rows)} | amp={amp} max_len={max_len} hn={a.hn}")

    model = AutoModelForCausalLM.from_pretrained(a.repo, trust_remote_code=True, torch_dtype=torch.float32).to(dev)
    tok = AutoTokenizer.from_pretrained(a.repo, trust_remote_code=True)
    proc = AutoImageProcessor.from_pretrained(a.repo, trust_remote_code=True)
    set_trainable(model, a.last_n)
    scale = model.logit_scale.exp().detach() if hasattr(model, "logit_scale") else torch.tensor(100.0, device=dev)
    scale = scale.clamp(max=100).to(dev)

    val_px = torch.stack([DS(val_rows, proc)[i][0] for i in range(len(val_rows))]).half()
    log = {"args": vars(a) | {"images": str(a.images)}, "epochs": []}
    log["zero_shot_val"] = evaluate(model, tok, val_rows, val_px, dev, max_len, walk, amp)
    print("[val] zero-shot", log["zero_shot_val"])

    dl = DataLoader(DS(tr_rows, proc), batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    collate_fn=collate, drop_last=True, pin_memory=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * total))) * 0.5 * (1 + math.cos(math.pi * s / total)))
    scaler = torch.amp.GradScaler(enabled=(dev.type == "cuda" and amp == torch.float16))
    rng = random.Random(a.seed)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        run = []
        for step, (px, prompts) in enumerate(dl):
            hn_txt = [t for p in prompts for _, t in make_hard_negatives(p, a.hn, rng)] if a.hn else []
            with torch.autocast(dev.type, dtype=amp, enabled=dev.type == "cuda"):
                img = F.normalize(model.get_image_features(px.to(dev)).float(), dim=-1)
                txt = encode_text(model, tok, prompts + hn_txt, dev, max_len, walk)
            b = len(prompts)
            tgt = torch.arange(b, device=dev)
            loss_i2t = F.cross_entropy(scale * img @ txt.T, tgt)       # HN 텍스트는 추가 negative 로만 들어감
            loss_t2i = F.cross_entropy(scale * txt[:b] @ img.T, tgt)
            loss = (loss_i2t + loss_t2i) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run.append(loss.item())
            if step % 20 == 0:
                print(f"[ep {ep}] step {step}/{len(dl)} loss {np.mean(run[-20:]):.4f}  {time.time() - t0:.0f}s", flush=True)
        v = evaluate(model, tok, val_rows, val_px, dev, max_len, walk, amp)
        print(f"[val] epoch {ep}", v)
        log["epochs"].append({"epoch": ep, "loss": float(np.mean(run)), "val": v})

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.to(torch.bfloat16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out); proc.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps(log, indent=2, default=str))
    print("saved", out)


if __name__ == "__main__":
    main()
