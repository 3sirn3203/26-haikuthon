"""FG-CLIP(1단계) + Qwen3.5 VLM Yes/No 재정렬(2단계) 제출용 model.py (서버 계약: load_model / predict).

zip 구성 (scripts/modal_build_rerank.py 가 만든다)
  model.py
  fgclip.safetensors        FG-CLIP 파인튜닝 가중치 (표준 CLIPModel 키만, 서버 규칙상 zip 루트에 safetensors 필요)
  fgclip/                   CLIP config.json + 토크나이저
  qwen/                     Qwen3.5 config·토크나이저·프로세서 + model.safetensors (LoRA 병합본, MTP 제외)
  rerank_meta.json          K, 시간 예산, 배치, 이미지 전처리 값
  requirements.txt          transformers 5.x (Qwen3.5 용)

1단계: FG-CLIP 짧은 모드(77토큰, walk_short_pos)는 구조가 표준 CLIP 과 같아서 transformers 의 CLIPModel 로 돌린다.
       FG-CLIP 원격코드(transformers 4.46 필요)를 쓰지 않으므로 Qwen3.5(transformers 5.x)와 한 환경에서 돈다.
2단계: 문항마다 1단계 top-K 를 "이 그림이 설명과 맞나? Yes/No" 점수(logit Yes − logit No)로 다시 정렬한다.
       쌍마다 독립 채점이고 다른 문항 정보는 쓰지 않는다. 동점은 1단계 순서를 따른다.
시간 예산: load_model 시작부터 budget_sec 가 지나면 남은 문항은 재정렬하지 않고 1단계 순서를 그대로 낸다.
       환경변수 HAIKU_RERANK_BUDGET 로 덮어쓸 수 있다 (0 이하 = 제한 없음).
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

QUESTION = ('Does this image accurately match the following description, including every '
            'object count, color, material, pattern, pose and spatial relation?\n'
            'Description: "{prompt}"\nAnswer with Yes or No.')


def _log(msg):
    print(f"[rerank] {msg}", flush=True)


class _Bundle:
    pass


def load_model(model_dir, device="cuda:0"):
    from transformers import (AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, CLIPConfig, CLIPModel)
    from safetensors.torch import load_file

    t0 = time.monotonic()
    d = Path(model_dir).resolve()
    meta = json.loads((d / "rerank_meta.json").read_text())
    b = _Bundle()
    b.meta, b.device, b.t_start = meta, device, t0
    env = os.environ.get("HAIKU_RERANK_BUDGET")
    b.budget = float(env) if env not in (None, "") else float(meta.get("budget_sec", 600))

    # --- 1단계 FG-CLIP (표준 CLIP)
    cfg = CLIPConfig.from_pretrained(str(d / "fgclip"))
    clip = CLIPModel(cfg)
    missing, unexpected = clip.load_state_dict(load_file(str(d / "fgclip.safetensors")), strict=False)
    missing = [k for k in missing if not k.endswith("position_ids")]
    if missing or unexpected:
        raise RuntimeError(f"FG-CLIP 가중치 불일치: missing {missing[:5]} unexpected {unexpected[:5]}")
    b.clip = clip.to(device, torch.bfloat16).eval()
    b.clip_tok = AutoTokenizer.from_pretrained(str(d / "fgclip"))

    # --- 2단계 Qwen3.5 (LoRA 병합본)
    q = str(d / "qwen")
    b.vlm = AutoModelForImageTextToText.from_pretrained(q, dtype=torch.bfloat16).to(device).eval()
    b.proc = AutoProcessor.from_pretrained(q, min_pixels=int(meta.get("min_pixels", 64 * 64)),
                                           max_pixels=int(meta.get("max_pixels", 448 * 448)))
    b.proc.tokenizer.padding_side = "left"
    tok = b.proc.tokenizer
    b.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes", " Yes")]
    b.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no", " No")]
    _log(f"load {time.monotonic() - t0:.1f}s | K={meta.get('k', 5)} budget={b.budget:.0f}s")
    return b


class _ClipImages(Dataset):
    """CLIPImageProcessor(shortest_edge 224, bicubic, center crop 224, CLIP mean/std) 와 같은 전처리."""

    def __init__(self, paths, meta):
        self.paths = paths
        self.size, self.crop = int(meta.get("clip_size", 224)), int(meta.get("clip_crop", 224))
        self.mean = np.array(meta.get("clip_mean", [0.48145466, 0.4578275, 0.40821073]), np.float32)
        self.std = np.array(meta.get("clip_std", [0.26862954, 0.26130258, 0.27577711]), np.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            w, h = im.size
            s = self.size
            nw, nh = (s, int(s * h / w)) if w <= h else (int(s * w / h), s)
            im = im.resize((nw, nh), Image.BICUBIC)
            c = self.crop
            top, left = (nh - c) // 2, (nw - c) // 2
            im = im.crop((left, top, left + c, top + c))
            x = (np.asarray(im, np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


def _feat(out):
    """transformers 버전에 따라 get_*_features 가 텐서 또는 출력 객체를 돌려준다."""
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output if getattr(out, "pooler_output", None) is not None else out[0]


@torch.inference_mode()
def _stage1(b, queries, candidates, n_keep):
    meta, dev = b.meta, b.device
    t0 = time.time()
    paths = [c["image_path"] for c in candidates]
    ids = [str(c["image_id"]) for c in candidates]
    nw = min(8, os.cpu_count() or 2)
    dl = DataLoader(_ClipImages(paths, meta), batch_size=int(meta.get("image_batch", 128)),
                    num_workers=nw, pin_memory=True, prefetch_factor=4 if nw else None)
    feats = []
    for px in dl:
        f = _feat(b.clip.get_image_features(pixel_values=px.to(dev, torch.bfloat16)))
        feats.append(F.normalize(f.float(), dim=-1).to(torch.bfloat16))
    img = torch.cat(feats)
    t1 = time.time()
    _log(f"stage1 image encode {len(ids)}: {t1 - t0:.1f}s")
    prompts = [q["prompt"] for q in queries]
    max_len = int(meta.get("clip_max_length", 77))
    txt = []
    for s in range(0, len(prompts), 512):
        enc = b.clip_tok(prompts[s:s + 512], padding="max_length", truncation=True, max_length=max_len,
                         return_tensors="pt")
        f = _feat(b.clip.get_text_features(input_ids=enc["input_ids"].to(dev)))
        txt.append(F.normalize(f.float(), dim=-1).to(torch.bfloat16))
    txt = torch.cat(txt)
    top = []
    for s in range(0, len(prompts), 1000):
        top += (txt[s:s + 1000] @ img.T).float().topk(n_keep, dim=1).indices.cpu().tolist()
    _log(f"stage1 text+search: {time.time() - t1:.1f}s")
    return [[ids[j] for j in row] for row in top], dict(zip(ids, paths))


def _chat(b, prompt):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTION.format(prompt=prompt)}]}]
    return b.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def _load_rgb(p):
    with Image.open(p) as im:
        return im.convert("RGB")


@torch.inference_mode()
def _margins(b, texts, images):
    inputs = b.proc(text=texts, images=images, return_tensors="pt", padding=True).to(b.device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(b.device).startswith("cuda")):
        logits = b.vlm(**inputs, logits_to_keep=1).logits[:, -1].float()
    return (torch.logsumexp(logits[:, b.yes], -1) - torch.logsumexp(logits[:, b.no], -1)).tolist()


def predict(model, queries, candidates, top_k=5):
    b = model
    k = int(b.meta.get("k", 5))
    s1, path_of = _stage1(b, queries, candidates, max(k, top_k))
    out = {str(q["prompt_id"]): row[:top_k] for q, row in zip(queries, s1)}
    if k <= 1 or not queries:
        return out

    deadline = b.t_start + b.budget if b.budget > 0 else float("inf")
    q_per = max(1, int(b.meta.get("pairs_per_batch", 20)) // k)
    chunks = [list(range(i, min(i + q_per, len(queries)))) for i in range(0, len(queries), q_per)]
    pool = ThreadPoolExecutor(8)
    load = lambda ch: [pool.submit(_load_rgb, path_of[c]) for i in ch for c in s1[i][:k]]
    t0, done, nxt = time.monotonic(), 0, load(chunks[0])
    log_every = int(b.meta.get("log_every", 1000))
    for ci, ch in enumerate(chunks):
        if time.monotonic() > deadline:
            _log(f"시간 예산 도달: {done}/{len(queries)} 문항 재정렬, 나머지는 1단계 순서")
            break
        ims = [f.result() for f in nxt]
        if ci + 1 < len(chunks):
            nxt = load(chunks[ci + 1])
        texts = [_chat(b, queries[i]["prompt"]) for i in ch for _ in range(k)]
        sc = _margins(b, texts, ims)
        for j, i in enumerate(ch):
            g = sc[j * k:(j + 1) * k]
            order = sorted(range(k), key=lambda t: (-g[t], t))
            row = s1[i]
            out[str(queries[i]["prompt_id"])] = ([row[t] for t in order] + row[k:])[:top_k]
        prev, done = done, done + len(ch)
        if done // log_every > prev // log_every:
            el = time.monotonic() - t0
            _log(f"rerank {done}/{len(queries)}  {el:.0f}s ({el / (done * k):.4f}s/pair)")
    pool.shutdown(wait=False, cancel_futures=True)
    el = time.monotonic() - t0
    _log(f"rerank 끝: {done}/{len(queries)} 문항, {el:.1f}s"
         + (f" | peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else ""))
    return out
