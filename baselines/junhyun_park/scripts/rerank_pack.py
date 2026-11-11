"""FG-CLIP + Qwen3.5 재정렬 제출 zip 조립 함수 (scripts/modal_build_rerank.py 가 Modal 안에서 쓴다).

- fgclip_to_clip: FG-CLIP 파인튜닝 zip -> 표준 CLIPModel 가중치(fgclip.safetensors) + fgclip/ (config, 토크나이저)
- merge_lora:     Qwen3.5 원본 safetensors + PEFT LoRA 어댑터 -> 병합 가중치 (MTP 층 제외)
- pack_zip:       스테이징 폴더 -> zip (deflate) + 크기 검사
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

ZIP_LIMIT = 4 * 1024 ** 3          # 서버 MAX_UPLOAD_BYTES 기본값
TOKENIZER_FILES = ("vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
# FG-CLIP 에만 있고 짧은 모드(walk_short_pos=True)에서 쓰지 않는 가중치
FGCLIP_ONLY = re.compile(r"(position_embedding_res|position_embedding_ori|text_filip_projection|"
                         r"logit_scale_finegraind|logit_scale_hardneg|position_ids)")


def clip_config(base_cfg: dict | None = None) -> dict:
    """fg-clip-base 와 같은 크기의 표준 CLIP config. FG-CLIP 처럼 eos(=pad)=49407 로 풀링 위치를 정한다."""
    t = {"hidden_size": 512, "intermediate_size": 2048, "num_attention_heads": 8, "num_hidden_layers": 12,
         "max_position_embeddings": 77, "vocab_size": 49408, "hidden_act": "quick_gelu", "layer_norm_eps": 1e-5,
         "projection_dim": 512, "bos_token_id": 49406, "eos_token_id": 49407, "pad_token_id": 49407}
    v = {"hidden_size": 768, "intermediate_size": 3072, "num_attention_heads": 12, "num_hidden_layers": 12,
         "image_size": 224, "patch_size": 16, "hidden_act": "quick_gelu", "layer_norm_eps": 1e-5, "projection_dim": 512,
         "num_channels": 3}
    return {"architectures": ["CLIPModel"], "model_type": "clip", "projection_dim": 512,
            "logit_scale_init_value": 2.6592, "text_config": t, "vision_config": v}


def fgclip_to_clip(fgclip_zip: str | Path, stage: str | Path) -> dict:
    """FG-CLIP zip(빌드된 제출 zip 또는 save_pretrained 폴더를 묶은 것) -> stage/fgclip.safetensors, stage/fgclip/."""
    stage = Path(stage)
    (stage / "fgclip").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(fgclip_zip) as z:
        names = set(z.namelist())
        for f in TOKENIZER_FILES:
            if f in names:
                (stage / "fgclip" / f).write_bytes(z.read(f))
        pre = json.loads(z.read("preprocessor_config.json")) if "preprocessor_config.json" in names else {}
        tmp = stage / "_fg.safetensors"
        tmp.write_bytes(z.read("model.safetensors"))
    sd = load_file(tmp)
    tmp.unlink()
    keep = {k: v.to(torch.bfloat16).contiguous() for k, v in sd.items() if not FGCLIP_ONLY.search(k)}
    dropped = sorted(k for k in sd if FGCLIP_ONLY.search(k))
    save_file(keep, stage / "fgclip.safetensors", metadata={"format": "pt"})
    (stage / "fgclip" / "config.json").write_text(json.dumps(clip_config(), indent=1))
    print(f"[fgclip] 유지 {len(keep)} / 제외 {len(dropped)}: {dropped}")
    return {"clip_size": pre.get("size", {}).get("shortest_edge", 224),
            "clip_crop": pre.get("crop_size", {}).get("height", 224),
            "clip_mean": pre.get("image_mean", [0.48145466, 0.4578275, 0.40821073]),
            "clip_std": pre.get("image_std", [0.26862954, 0.26130258, 0.27577711])}


def merge_lora(base_files: list[str | Path], adapter_dir: str | Path, out_path: str | Path, drop=("mtp.",)) -> dict:
    """W' = W + (alpha/r) * B @ A  (fp32 로 계산 후 원래 dtype). PEFT 키 'base_model.model.<모듈>.lora_A.weight'."""
    adapter_dir = Path(adapter_dir)
    acfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    r, alpha = acfg["r"], acfg["lora_alpha"]
    scale = alpha / (r ** 0.5) if acfg.get("use_rslora") else alpha / r
    lora = load_file(adapter_dir / "adapter_model.safetensors")
    pairs = {}
    for k, v in lora.items():
        m = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])(?:\.default)?\.weight", k)
        if not m:
            raise ValueError(f"알 수 없는 어댑터 키: {k}")
        pairs.setdefault(m.group(1) + ".weight", {})[m.group(2)] = v
    out, merged, dropped = {}, 0, 0
    for f in base_files:
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if any(k.startswith(p) for p in drop):
                    dropped += 1
                    continue
                w = h.get_tensor(k)
                if k in pairs:
                    a, bm = pairs[k]["A"].float(), pairs[k]["B"].float()
                    w = (w.float() + scale * (bm @ a)).to(w.dtype)
                    merged += 1
                out[k] = w.contiguous()
    missing = sorted(set(pairs) - set(out))
    if missing:
        raise ValueError(f"어댑터 대상이 원본에 없음 ({len(missing)}): {missing[:3]}")
    save_file(out, str(out_path), metadata={"format": "pt"})
    info = {"lora_r": r, "lora_alpha": alpha, "scale": scale, "merged_tensors": merged, "dropped_tensors": dropped,
            "n_tensors": len(out)}
    print(f"[merge] {info}")
    return info


def pack_zip(stage: str | Path, out_zip: str | Path, level: int = 6) -> int:
    stage, out_zip = Path(stage), Path(out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as z:
        for p in files:
            z.write(p, p.relative_to(stage).as_posix())
    size = out_zip.stat().st_size
    print(f"[zip] {out_zip} {size / 1e9:.3f} GB ({len(files)} files, 제한 {ZIP_LIMIT / 1e9:.3f} GB)")
    if size > ZIP_LIMIT:
        raise RuntimeError("zip 이 서버 업로드 제한(4 GiB)을 넘음")
    return size
