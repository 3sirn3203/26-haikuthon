"""FG-CLIP(1단계) + Qwen3.5 LoRA 재정렬(2단계) 제출 zip 을 Modal 안에서 만든다 (4GB zip 을 Mac 에서 올리지 않기 위해).

입력 (볼륨 jp-fgclip-zip, modal_run_fgclip.py 와 같은 볼륨)
  /<fgclip zip>      예: fgclip_ft_bundle_bval_3.zip  (이미 올라가 있음)
  /<adapter 폴더>     예: adapter_R1/ (Colab Drive haikuthon_runs/rerank/adapter_R1 을 받아서 올림)
출력
  /<out>.zip         서버 형식 제출 zip (model.py + fgclip.safetensors + fgclip/ + qwen/ + rerank_meta.json + requirements.txt)

빌드 중 검증 (L4)
  1) LoRA 병합본 vs 원본+PEFT 어댑터: 같은 쌍에서 Yes-No 점수 비교
  2) model.py load_model/predict 를 합성 데이터(이미지 40장, 문항 8개)로 끝까지 실행
  3) 재정렬 쌍당 시간: 배치 10/20/40 쌍
  4) 서버 prepare_submission 으로 zip 규칙 검사 (크기 4 GiB 이하 포함)

실행:
  modal volume put jp-fgclip-zip <로컬 adapter_R1 폴더> /adapter_R1
  modal run baselines/junhyun_park/scripts/modal_build_rerank.py --adapter adapter_R1 --out fgclip_qwen2b_r1_k5.zip
  modal run baselines/junhyun_park/scripts/modal_run_fgclip.py --zip fgclip_qwen2b_r1_k5.zip --budget 0
"""
import json
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
SERVER = Path(os.environ.get("HAIKUTHON_SERVER_REPO", "~/Desktop/2026-HAIKUTHON")).expanduser()
REQUIREMENTS = ["transformers==5.22.2", "torchvision==0.29.2", "flash-linear-attention==0.6.0", "fla-core==0.6.0",
                "einops==0.8.1"]

app = modal.App("jp-rerank-build")
zip_vol = modal.Volume.from_name("jp-fgclip-zip", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(SERVER / "evaluation" / "gpu-requirements.txt"))
    .pip_install(*REQUIREMENTS, "peft==0.21.2", "accelerate==1.15.0")
    .env({"PYTHONPATH": "/opt/haikuthon:/opt/jp"})
    .add_local_dir(str(SERVER / "evaluation"), "/opt/haikuthon/evaluation", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_file(str(HERE / "rerank_pack.py"), "/opt/jp/rerank_pack.py")
    .add_local_file(str(HERE.parent / "submission/rerank/model.py"), "/opt/jp/submission_model.py")
)


def _synthetic(root, n_img=40, n_q=8, seed=0):
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cands = []
    for i in range(n_img):
        x = (np.linspace(0, 255, 1024)[None, :, None] * np.ones((1024, 1, 3)))
        x[..., i % 3] = rng.integers(0, 255, (1024, 1024))
        p = root / f"{i:05d}.png"
        Image.fromarray(x.astype("uint8")).save(p)
        cands.append({"image_id": f"{i:05d}", "image_path": str(p)})
    words = ["red", "blue", "green", "two", "three", "cat", "dog", "mug", "left of", "on top of"]
    qs = [{"prompt_id": f"{i:05d}", "prompt": " ".join(rng.choice(words, 8))} for i in range(n_q)]
    return qs, cands


@app.function(image=image, gpu="L4", cpu=8, memory=49152, timeout=3600, ephemeral_disk=100 * 1024,
              volumes={"/zips": zip_vol})
def build(fgclip_zip: str, adapter: str, out: str, reranker: str, k: int, budget: float, pairs_per_batch: int,
          max_pixels: int, verify: bool = True) -> dict:
    import gc
    import importlib.util
    import shutil
    import sys
    import tempfile
    import time

    import torch
    from huggingface_hub import snapshot_download

    import rerank_pack as P

    zip_vol.reload()
    res = {"fgclip_zip": fgclip_zip, "adapter": adapter, "reranker": reranker, "k": k}
    stage = Path("/tmp/stage")
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()

    # 1) FG-CLIP -> 표준 CLIP
    meta = P.fgclip_to_clip(f"/zips/{fgclip_zip}", stage)

    # 2) Qwen 원본 + LoRA 병합
    base = Path(snapshot_download(reranker))
    (stage / "qwen").mkdir()
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and not f.name.endswith(".safetensors.index.json") \
                and not f.name.startswith("."):
            shutil.copy(f.resolve(), stage / "qwen" / f.name)
    shards = sorted(base.glob("*.safetensors"))
    res["merge"] = P.merge_lora(shards, f"/zips/{adapter}", stage / "qwen" / "model.safetensors")
    ad_cfg = json.loads(Path(f"/zips/{adapter}/adapter_config.json").read_text())

    meta.update({"k": k, "budget_sec": budget, "pairs_per_batch": pairs_per_batch, "max_pixels": max_pixels,
                 "min_pixels": 64 * 64, "clip_max_length": 77, "image_batch": 128,
                 "reranker": reranker, "adapter": adapter, "stage1": fgclip_zip,
                 "lora": {kk: ad_cfg.get(kk) for kk in ("r", "lora_alpha", "lora_dropout")}})
    (stage / "rerank_meta.json").write_text(json.dumps(meta, indent=1))
    (stage / "requirements.txt").write_text("\n".join(REQUIREMENTS) + "\n")
    shutil.copy("/opt/jp/submission_model.py", stage / "model.py")
    res["meta"] = meta

    if verify:
        from PIL import Image
        spec = importlib.util.spec_from_file_location("sub_model", stage / "model.py")
        M = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(M)
        tmp = Path(tempfile.mkdtemp())
        qs, cands = _synthetic(tmp / "imgs")

        # 2-1) 병합본 vs 원본 + PEFT 어댑터
        b = M.load_model(str(stage), "cuda:0")
        ims = [Image.open(c["image_path"]).convert("RGB") for c in cands[:10]]
        texts = [M._chat(b, qs[i % len(qs)]["prompt"]) for i in range(10)]
        m_merged = M._margins(b, texts, ims)
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText
        vlm_merged = b.vlm
        b.vlm = PeftModel.from_pretrained(
            AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16).to("cuda:0"),
            f"/zips/{adapter}").eval()
        m_peft = M._margins(b, texts, ims)
        diff = [abs(x - y) for x, y in zip(m_merged, m_peft)]
        res["merge_check"] = {"merged": [round(x, 3) for x in m_merged], "peft": [round(x, 3) for x in m_peft],
                              "max_abs_diff": round(max(diff), 4)}
        print("[verify] 병합 검사", res["merge_check"], flush=True)
        del b.vlm
        b.vlm = vlm_merged
        gc.collect(); torch.cuda.empty_cache()

        # 2-2) 쌍당 시간
        bench = {}
        big = [Image.open(c["image_path"]).convert("RGB") for c in cands[:40]]
        for n in (10, 20, 40):
            tx = [M._chat(b, qs[i % len(qs)]["prompt"]) for i in range(n)]
            M._margins(b, tx, big[:n]); torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                M._margins(b, tx, big[:n])
            torch.cuda.synchronize()
            bench[n] = round((time.time() - t0) / (3 * n), 4)
        res["sec_per_pair_by_batch"] = bench
        print("[verify] 쌍당 초", bench, flush=True)

        # 2-3) predict 끝까지 (합성 데이터)
        b.t_start = time.monotonic()
        out_pred = M.predict(b, qs, cands, top_k=5)
        assert set(out_pred) == {q["prompt_id"] for q in qs} and all(len(v) == 5 for v in out_pred.values())
        res["predict_ok"] = True
        res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        del b
        gc.collect(); torch.cuda.empty_cache()

    # 3) zip + 서버 규칙 검사
    out_zip = Path(f"/zips/{out}")
    res["zip_bytes"] = P.pack_zip(stage, out_zip)
    from evaluation.model_loader import prepare_submission
    with tempfile.TemporaryDirectory() as t:
        prepare_submission(out_zip, Path(t) / "x")
    res["prepare_submission"] = "ok"
    zip_vol.commit()
    return res


@app.local_entrypoint()
def main(adapter: str = "adapter_R1", out: str = "fgclip_qwen2b_r1_k5.zip",
         fgclip_zip: str = "fgclip_ft_bundle_bval_3.zip", reranker: str = "Qwen/Qwen3.5-2B", k: int = 5,
         budget: float = 660, pairs_per_batch: int = 20, max_pixels: int = 448 * 448, verify: bool = True):
    print(json.dumps(build.remote(fgclip_zip, adapter, out, reranker, k, budget, pairs_per_batch, max_pixels, verify),
                     indent=2, ensure_ascii=False))
