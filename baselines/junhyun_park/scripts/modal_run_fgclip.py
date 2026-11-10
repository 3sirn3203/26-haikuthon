"""웹사이트 없이 서버의 evaluation 코드를 Modal L4에서 직접 실행 (end-to-end 시간/VRAM 측정).

서버(2026-HAIKUTHON)의 gpu_evaluate 와 같은 이미지/데이터/호출(evaluation.runner.evaluate)을 쓴다.
- 데이터: 서버의 숨김 테스트셋 볼륨 `<APP>-dataset` (읽기 전용, /dataset/test/test.csv)
- 제출 zip: 별도 볼륨 `jp-fgclip-zip` 에 올려 둔다 (서버 DB/큐/제출 횟수와 무관)

준비/실행:
  export HAIKUTHON_SERVER_REPO=~/Desktop/2026-HAIKUTHON
  modal volume ls haikuthon-dev-dataset            # test/ 가 있는지 먼저 확인
  modal volume create jp-fgclip-zip
  modal volume put jp-fgclip-zip baselines/junhyun_park/build/fgclip_base.zip /fgclip_base.zip
  modal run baselines/junhyun_park/scripts/modal_run_fgclip.py --zip fgclip_base.zip
  modal volume get jp-fgclip-zip /results/fgclip_base.csv baselines/junhyun_park/build/   # 결과 CSV
  python baselines/junhyun_park/scripts/score_csv.py baselines/junhyun_park/build/fgclip_base.csv
"""
import json
import os
from pathlib import Path

import modal

SERVER = Path(os.environ.get("HAIKUTHON_SERVER_REPO", "~/Desktop/2026-HAIKUTHON")).expanduser()
SERVER_APP = os.environ.get("HAIKUTHON_MODAL_APP", "haikuthon-dev")
app = modal.App("jp-fgclip-test")
zip_vol = modal.Volume.from_name("jp-fgclip-zip", create_if_missing=True)
dataset_vol = modal.Volume.from_name(f"{SERVER_APP}-dataset")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(SERVER / "evaluation" / "gpu-requirements.txt"))
    .env({"PYTHONPATH": "/opt/haikuthon"})
    .add_local_dir(str(SERVER / "evaluation"), "/opt/haikuthon/evaluation",
                   ignore=["**/__pycache__/**", "**/*.pyc"])
)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=1800,
              volumes={"/dataset": dataset_vol.read_only(), "/zips": zip_vol})
def run(zip_name: str, limit_note: str = "") -> dict:
    import time
    import traceback

    import torch
    from evaluation.runner import evaluate

    zip_vol.reload()
    out = Path("/tmp/result.csv")
    t0 = time.monotonic()
    res = {"zip": zip_name, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    torch.cuda.reset_peak_memory_stats()
    try:
        r = evaluate(f"/zips/{zip_name}", "/dataset/test/test.csv", out, image_root="/dataset/test",
                     device="cuda:0", top_k=5, expected_queries=8000, expected_images=15000,
                     sample_submission=(Path("/dataset/test/sample_submission.csv")
                                        if Path("/dataset/test/sample_submission.csv").is_file() else None))
        res.update(status="COMPLETED", evaluate=r)
        stem = Path(zip_name).stem
        Path("/zips/results").mkdir(exist_ok=True)
        Path(f"/zips/results/{stem}.csv").write_bytes(out.read_bytes())
        zip_vol.commit()
    except Exception as exc:  # 실제 원인 확인용 (내 테스트라 메시지 그대로 노출)
        res.update(status="FAILED", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc()[-1500:])
    res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
    res["wall_s"] = round(time.monotonic() - t0, 1)
    return res


@app.function(image=image)
def info() -> dict:
    """이미지에 설치된 패키지 버전 (requirements.txt 고정값 결정용)."""
    import importlib.metadata as md
    names = ["torch", "numpy", "pillow", "safetensors", "packaging"]
    return {n: md.version(n) for n in names}


@app.local_entrypoint()
def main(zip: str = "fgclip_base.zip", show_info: bool = False):
    if show_info:
        print(json.dumps(info.remote(), indent=2))
        return
    print(json.dumps(run.remote(zip), indent=2, ensure_ascii=False))
