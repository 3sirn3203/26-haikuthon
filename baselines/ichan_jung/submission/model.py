"""HAIKUTHON 제출용 model.py — CLIP 계열(PickScore / CLIP) 순수 torch 구현.

외부 패키지 불필요(torch, safetensors, numpy, pillow만 사용 = 서버 기본 이미지).
ZIP 최상위 구성:
  model.py, model.safetensors (HF CLIPModel 키 이름, fp16),
  clip_config.json, vocab.json, merges.txt, requirements.txt(비어 있음)
"""
import html
import json
import os
import re
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ============================================================ tokenizer
@lru_cache()
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class CLIPTokenizer:
    # 원본 패턴의 \p{L}, \p{N}을 표준 re로 근사
    PAT = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+",
                     re.IGNORECASE)

    def __init__(self, vocab_path, merges_path, context_length=77):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            merges = [tuple(l.split()) for l in f.read().split("\n") if l and not l.startswith("#version")]
        self.ranks = {m: i for i, m in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.bos, self.eos = self.encoder["<|startoftext|>"], self.encoder["<|endoftext|>"]
        self.ctx = context_length
        self.cache = {}

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        while len(word) > 1:
            pairs = {(word[i], word[i + 1]) for i in range(len(word) - 1)}
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            a, b = best
            out, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = tuple(out)
        self.cache[token] = word
        return word

    def encode(self, text):
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text).strip().lower()
        ids = []
        for tok in self.PAT.findall(text):
            tok = "".join(self.byte_encoder[b] for b in tok.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self._bpe(tok))
        return [self.bos] + ids[:self.ctx - 2] + [self.eos]

    def __call__(self, texts):
        out = torch.zeros(len(texts), self.ctx, dtype=torch.long)
        eos_pos = torch.zeros(len(texts), dtype=torch.long)
        for i, t in enumerate(texts):
            ids = self.encode(t)
            out[i, :len(ids)] = torch.tensor(ids)
            eos_pos[i] = len(ids) - 1
        return out, eos_pos


# ============================================================ model (HF CLIPModel 키 이름과 동일)
class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x, causal):
        b, n, d = x.shape
        q, k, v = (p(x).view(b, n, self.heads, -1).transpose(1, 2) for p in (self.q_proj, self.k_proj, self.v_proj))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.out_proj(o.transpose(1, 2).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, d, hidden, act):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, hidden), nn.Linear(hidden, d)
        self.act = act

    def forward(self, x):
        x = self.fc1(x)
        x = x * torch.sigmoid(1.702 * x) if self.act == "quick_gelu" else F.gelu(x)
        return self.fc2(x)


class Layer(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        self.self_attn = Attention(d, c["num_attention_heads"])
        self.layer_norm1 = nn.LayerNorm(d, eps=c["layer_norm_eps"])
        self.mlp = MLP(d, c["intermediate_size"], c["hidden_act"])
        self.layer_norm2 = nn.LayerNorm(d, eps=c["layer_norm_eps"])

    def forward(self, x, causal):
        x = x + self.self_attn(self.layer_norm1(x), causal)
        return x + self.mlp(self.layer_norm2(x))


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.ModuleList(Layer(c) for _ in range(c["num_hidden_layers"]))

    def forward(self, x, causal):
        for layer in self.layers:
            x = layer(x, causal)
        return x


class TextEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.token_embedding = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.position_embedding = nn.Embedding(c["max_position_embeddings"], c["hidden_size"])


class TextModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.embeddings = TextEmbeddings(c)
        self.encoder = Encoder(c)
        self.final_layer_norm = nn.LayerNorm(c["hidden_size"], eps=c["layer_norm_eps"])

    def forward(self, ids, eos_pos):
        x = self.embeddings.token_embedding(ids) + self.embeddings.position_embedding.weight[:ids.shape[1]]
        x = self.final_layer_norm(self.encoder(x, causal=True))
        return x[torch.arange(len(ids), device=ids.device), eos_pos]


class VisionEmbeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        d, p = c["hidden_size"], c["patch_size"]
        self.class_embedding = nn.Parameter(torch.zeros(d))
        self.patch_embedding = nn.Conv2d(3, d, p, p, bias=False)
        self.position_embedding = nn.Embedding((c["image_size"] // p) ** 2 + 1, d)


class VisionModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["hidden_size"]
        self.embeddings = VisionEmbeddings(c)
        self.pre_layrnorm = nn.LayerNorm(d, eps=c["layer_norm_eps"])
        self.encoder = Encoder(c)
        self.post_layernorm = nn.LayerNorm(d, eps=c["layer_norm_eps"])

    def forward(self, px):
        e = self.embeddings
        x = e.patch_embedding(px).flatten(2).transpose(1, 2)
        x = torch.cat([e.class_embedding.expand(len(x), 1, -1).to(x.dtype), x], 1) + e.position_embedding.weight
        x = self.encoder(self.pre_layrnorm(x), causal=False)
        return self.post_layernorm(x[:, 0])


class CLIP(nn.Module):
    def __init__(self, c):
        super().__init__()
        tc, vc = c["text_config"], c["vision_config"]
        self.text_model, self.vision_model = TextModel(tc), VisionModel(vc)
        self.text_projection = nn.Linear(tc["hidden_size"], c["projection_dim"], bias=False)
        self.visual_projection = nn.Linear(vc["hidden_size"], c["projection_dim"], bias=False)
        self.logit_scale = nn.Parameter(torch.zeros(()))

    def encode_image(self, px):
        return F.normalize(self.visual_projection(self.vision_model(px)).float(), dim=-1)

    def encode_text(self, ids, eos_pos):
        return F.normalize(self.text_projection(self.text_model(ids, eos_pos)).float(), dim=-1)


# ============================================================ 이미지 전처리 (CLIPImageProcessor 동일 규칙)
class Preprocess:
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def __call__(self, img):
        w, h = img.size
        s = self.size
        if w <= h:
            nw, nh = s, int(s * h / w)
        else:
            nw, nh = int(s * w / h), s
        img = img.resize((nw, nh), Image.BICUBIC)
        left, top = (nw - s) // 2, (nh - s) // 2
        img = img.crop((left, top, left + s, top + s))
        x = (np.asarray(img, np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(x.transpose(2, 0, 1).copy())


class _Images(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


# ============================================================ 제출 계약
def load_model(model_dir: str, device: str):
    from safetensors.torch import load_file
    with open(os.path.join(model_dir, "clip_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    model = CLIP(cfg)
    state = {k: v for k, v in load_file(os.path.join(model_dir, "model.safetensors")).items() if "position_ids" not in k}
    model.load_state_dict(state, strict=True)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype).eval()
    tok = CLIPTokenizer(os.path.join(model_dir, "vocab.json"), os.path.join(model_dir, "merges.txt"),
                        cfg["text_config"]["max_position_embeddings"])
    tf = Preprocess(cfg["vision_config"]["image_size"], cfg["image_mean"], cfg["image_std"])
    return {"model": model, "tok": tok, "tf": tf, "device": device, "dtype": dtype, "cfg": cfg}


@torch.inference_mode()
def predict(model, queries, candidates, top_k=5):
    m, dev, dt, cfg = model["model"], model["device"], model["dtype"], model["cfg"]
    workers = cfg.get("num_workers", 4) if os.name != "nt" else 0  # Windows 로컬 테스트는 0
    loader = DataLoader(_Images([c["image_path"] for c in candidates], model["tf"]),
                        batch_size=cfg.get("batch_size", 64), num_workers=workers,
                        pin_memory=dev.startswith("cuda"))
    img = torch.cat([m.encode_image(x.to(dev, dt, non_blocking=True)) for x in loader])

    texts = [q["prompt"] for q in queries]
    txt = []
    for i in range(0, len(texts), 256):
        ids, eos = model["tok"](texts[i:i + 256])
        txt.append(m.encode_text(ids.to(dev), eos.to(dev)))
    sim = torch.cat(txt) @ img.T

    if cfg.get("matching", "none") == "dsm":
        # 규정 허점 실험 전용: 이미지 축 softmax가 다른 질의를 참조 (질의 독립 규정 위반 소지)
        s = sim / cfg.get("temp", 0.02)
        sim = torch.softmax(s, dim=1) * torch.softmax(s, dim=0)
    top = sim.topk(top_k, dim=1).indices.cpu().tolist()
    ids = [c["image_id"] for c in candidates]
    return {q["prompt_id"]: [ids[j] for j in row] for q, row in zip(queries, top)}
