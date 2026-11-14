"""HF CLIP 계열 모델 → 서버에 바로 올릴 수 있는 submission.zip 생성 + 원본(transformers)과 출력 일치 검증.

  python export_submission.py --model pickscore --out build/pickscore

만든 ZIP 최상위: model.py, model.safetensors, clip_config.json, vocab.json, merges.txt
(빌드에만 transformers 필요. 제출 ZIP은 torch/safetensors/numpy/pillow만 사용)
"""
import argparse
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

PRESETS = {
    "pickscore": ("yuvalkirstain/PickScore_v1", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
    "clip-b32": ("openai/clip-vit-base-patch32", None),  # 빌드 파이프라인 빠른 동작 확인용 (제출 대상 아님)
}
LIMIT = 4 * 2**30
TEXT_KEYS = ["vocab_size", "hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers",
             "hidden_act", "layer_norm_eps", "max_position_embeddings"]
VISION_KEYS = ["hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers",
               "hidden_act", "layer_norm_eps", "image_size", "patch_size"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=PRESETS, required=True)
    ap.add_argument("--matching", default="none", choices=["none", "dsm"])
    ap.add_argument("--temp", type=float, default=0.02)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    sys.dont_write_bytecode = True
    import torch
    from safetensors.torch import save_file
    from transformers import CLIPModel, CLIPProcessor

    model_id, proc_id = PRESETS[args.model]
    out = Path(args.out)
    pkg = out / "pkg"
    if pkg.exists():
        shutil.rmtree(pkg)
    pkg.mkdir(parents=True)

    print(f"download/load {model_id}")
    hf = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    proc = CLIPProcessor.from_pretrained(proc_id or model_id)

    c = hf.config
    cfg = {
        "source": model_id,
        "projection_dim": c.projection_dim,
        "text_config": {k: getattr(c.text_config, k) for k in TEXT_KEYS},
        "vision_config": {k: getattr(c.vision_config, k) for k in VISION_KEYS},
        "image_mean": list(proc.image_processor.image_mean),
        "image_std": list(proc.image_processor.image_std),
        "matching": args.matching, "temp": args.temp, "batch_size": 64, "num_workers": 4,
    }
    (pkg / "clip_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    state = {k: v.detach().half().contiguous() for k, v in hf.state_dict().items() if "position_ids" not in k}
    save_file(state, str(pkg / "model.safetensors"))
    from huggingface_hub import hf_hub_download
    for name in ("vocab.json", "merges.txt"):
        shutil.copy(hf_hub_download(proc_id or model_id, name), pkg / name)
    shutil.copy(Path(__file__).parent / "submission" / "model.py", pkg / "model.py")
    # requirements.txt는 선택 사항. 추가 패키지가 없으므로 넣지 않아 서버의 의존성 설치 단계를 건너뛴다.
    keep = {"model.py", "model.safetensors", "clip_config.json", "vocab.json", "merges.txt"}
    for f in pkg.iterdir():
        if f.name not in keep:
            f.unlink()

    if not args.no_verify:
        verify(hf, proc, pkg)

    zpath = out / "submission.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED, allowZip64=True) as z:  # 가중치는 압축 효과 없음
        for name in sorted(keep):  # 서버는 __pycache__ 등 불필요 항목을 거부
            z.write(pkg / name, name)
    size = zpath.stat().st_size
    print(f"\n{zpath.resolve()}\n  {size / 2**30:.2f} GiB  {'OK' if size <= LIMIT else '4GiB 초과!'}")
    print("  내용:", ", ".join(sorted(zipfile.ZipFile(zpath).namelist())))


def verify(hf, proc, pkg):
    """순수 torch 구현이 transformers와 같은 임베딩을 내는지 (CPU, fp32) 확인."""
    import numpy as np
    import torch
    from PIL import Image
    spec = importlib.util.spec_from_file_location("_sub_model", pkg / "model.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["_sub_model"] = mod
    spec.loader.exec_module(mod)
    mine = mod.load_model(str(pkg), "cpu")
    texts = ["a red car parked on a rainy street at night", "Two cats sleeping on a sofa, 4k, highly detailed!",
             "A_robot's painting: \"sunset\" & mountains (1990s style) -- 3 birds"]
    rng = np.random.default_rng(0)
    imgs = [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for h, w in [(512, 512), (768, 512), (400, 600)]]
    with torch.no_grad():
        tok = proc.tokenizer(texts, padding=True, return_tensors="pt")
        same_tok = all(mine["tok"].encode(t) == proc.tokenizer(t)["input_ids"] for t in texts)
        def feat(x):
            return x if isinstance(x, torch.Tensor) else x.pooler_output
        t_ref = torch.nn.functional.normalize(feat(hf.get_text_features(**tok)), dim=-1)
        ids, eos = mine["tok"](texts)
        t_my = mine["model"].encode_text(ids, eos)
        px_ref = proc.image_processor(images=imgs, return_tensors="pt")["pixel_values"]
        i_ref = torch.nn.functional.normalize(feat(hf.get_image_features(pixel_values=px_ref)), dim=-1)
        px_my = torch.stack([mine["tf"](im) for im in imgs])
        i_my = mine["model"].encode_image(px_my)
    print(f"verify: tokenizer 동일={same_tok}  text cos={(t_ref * t_my).sum(-1).min():.5f}  "
          f"image cos={(i_ref * i_my).sum(-1).min():.5f}  pixel max diff={(px_ref - px_my).abs().max():.4f}")


if __name__ == "__main__":
    main()
