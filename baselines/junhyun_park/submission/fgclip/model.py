"""FG-CLIP zero-shot 제출용 model.py (서버 계약: load_model / predict).

- load_model(model_dir, device): zip 루트에 번들된 FG-CLIP 가중치/원격코드를 로컬에서만 로드(네트워크 X).
- predict(model, queries, candidates, top_k=5): 8,000 쿼리 x 15,000 후보를 한 번에 받아
  {prompt_id: [image_id x top_k]} 반환. 이미지 임베딩은 이 호출 안에서 계산(캐시 없음).
fgclip_meta.json 으로 v1/v2 차이를 조절한다 (v2 경로는 실제 API 확인 전이라 미검증).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class _Bundle:
    def __init__(self, model, tokenizer, image_processor, meta, device):
        self.model, self.tokenizer, self.image_processor = model, tokenizer, image_processor
        self.meta, self.device = meta, device


def load_model(model_dir, device="cuda:0"):
    from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoTokenizer

    d = str(Path(model_dir).resolve())
    meta_path = Path(d) / "fgclip_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    model = AutoModelForCausalLM.from_pretrained(
        d, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(d, local_files_only=True, trust_remote_code=True)
    ip = AutoImageProcessor.from_pretrained(d, local_files_only=True, trust_remote_code=True)
    return _Bundle(model, tok, ip, meta, device)


class _ImgDS(Dataset):
    def __init__(self, paths, processor, patches=0):
        self.paths, self.processor, self.patches = paths, processor, patches

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        kw = {"max_num_patches": self.patches} if self.patches else {}
        out = self.processor(images=img, return_tensors="pt", **kw)
        return {k: v[0] for k, v in out.items()}


def _log(msg):
    print(f"[fgclip] {msg}", flush=True)


@torch.inference_mode()
def predict(model, queries, candidates, top_k=5):
    b, dev = model.model, model.device
    meta = model.meta
    max_len = int(meta.get("max_length", 77))
    walk = bool(meta.get("walk_short_pos", True))
    bs_img = int(meta.get("image_batch", 128))
    bs_txt = int(meta.get("text_batch", 512))

    t0 = time.time()
    paths = [c["image_path"] for c in candidates]
    ids = [str(c["image_id"]) for c in candidates]
    nw = min(8, os.cpu_count() or 2)
    v2 = int(meta.get("version", 1)) == 2
    patches = int(meta.get("max_num_patches", 256)) if v2 else 0
    dl = DataLoader(_ImgDS(paths, model.image_processor, patches), batch_size=bs_img,
                    num_workers=nw, pin_memory=True, prefetch_factor=4 if nw else None)
    feats = []
    for batch in dl:
        batch = {k: (v.to(dev, dtype=torch.bfloat16) if v.is_floating_point() else v.to(dev))
                 for k, v in batch.items()}
        f = b.get_image_features(**batch) if v2 else b.get_image_features(batch["pixel_values"])
        feats.append(torch.nn.functional.normalize(f.float(), dim=-1).to(torch.bfloat16))
    img = torch.cat(feats)
    t1 = time.time()
    _log(f"image encode {len(ids)} imgs: {t1 - t0:.1f}s")

    prompts = [q["prompt"] for q in queries]
    pids = [str(q["prompt_id"]) for q in queries]
    txt = []
    for s in range(0, len(prompts), bs_txt):
        enc = model.tokenizer([p.lower() for p in prompts[s:s + bs_txt]] if v2 else prompts[s:s + bs_txt],
                              padding="max_length", truncation=True,
                              max_length=max_len, return_tensors="pt").to(dev)
        if v2:
            f = b.get_text_features(**enc, walk_type=meta.get("walk_type", "short"))
        else:
            f = b.get_text_features(enc["input_ids"], walk_short_pos=walk)
        txt.append(torch.nn.functional.normalize(f.float(), dim=-1).to(torch.bfloat16))
    txt = torch.cat(txt)
    t2 = time.time()
    _log(f"text encode {len(pids)} prompts: {t2 - t1:.1f}s")

    out = {}
    for s in range(0, len(pids), 1000):
        sim = txt[s:s + 1000] @ img.T
        top = sim.float().topk(top_k, dim=1).indices.cpu().tolist()
        for pid, row in zip(pids[s:s + 1000], top):
            out[pid] = [ids[j] for j in row]
    _log(f"search: {time.time() - t2:.1f}s | peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f}GB"
         if torch.cuda.is_available() else "search done")
    return out
