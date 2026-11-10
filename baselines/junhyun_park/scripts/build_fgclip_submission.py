"""FG-CLIP HF 스냅샷 -> 서버 형식 submission.zip 빌드 + 서버 규칙 검증.

사용 (Colab/Mac, HF 접근 가능한 곳):
  pip install huggingface_hub
  python scripts/build_fgclip_submission.py --repo qihoo360/fg-clip-large --out build/fgclip_large.zip --walk-short-pos 1
레포 id는 HF에서 정확한 이름을 확인해 넣을 것 (v2는 --walk-short-pos 0 등 API 확인 필요).
zip 루트에 model.py + *.safetensors + config/tokenizer/preprocessor + 원격코드(*.py) 포함.
"""
import argparse, json, shutil, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SKIP_SUFFIX = {".bin", ".h5", ".msgpack", ".onnx", ".gguf"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--walk-short-pos", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=77)
    ap.add_argument("--version", type=int, default=1, help="1=fg-clip, 2=fg-clip2")
    ap.add_argument("--walk-type", default="short", help="v2: short(max_len 64) / long(196)")
    ap.add_argument("--max-num-patches", type=int, default=256, help="v2 이미지 패치 수 (128/256/576/784/1024)")
    ap.add_argument("--req", action="append", default=[], help="requirements.txt 줄 추가 (여러 번 가능, 예: --req einops==0.8.1)")
    ap.add_argument("--transformers-pin", default="", help="예: transformers==4.46.3 (모델 카드 권장 버전)")
    a = ap.parse_args()

    if Path(a.repo).is_dir():   # 파인튜닝 결과 폴더 (save_pretrained 출력)
        snap = Path(a.repo)
    else:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(a.repo))
    stage = Path(a.out).with_suffix("")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for p in snap.iterdir():
        if p.is_file() and p.name != "model.py" and p.suffix not in SKIP_SUFFIX and not p.name.startswith("."):
            shutil.copy(p.resolve(), stage / p.name)  # symlink 해제 복사
    # 원격코드가 model.py 를 덮지 않도록: 원본 repo의 modeling 파일은 그대로 두고 제출 model.py만 추가
    shutil.copy(HERE / "submission/fgclip/model.py", stage / "model.py")
    (stage / "fgclip_meta.json").write_text(json.dumps(
        {"repo": a.repo, "version": a.version, "walk_short_pos": bool(a.walk_short_pos),
         "max_length": a.max_length, "walk_type": a.walk_type, "max_num_patches": a.max_num_patches}))
    reqs = ([a.transformers_pin] if a.transformers_pin else []) + a.req
    if reqs:
        (stage / "requirements.txt").write_text("\n".join(reqs) + "\n")

    if not list(stage.glob("*.safetensors")):
        sys.exit("safetensors 가중치가 없음 (repo가 .bin만 제공?) -> safetensors로 변환 필요")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(stage.iterdir()):
            z.write(p, p.name)
    print(f"zip: {out} ({out.stat().st_size / 1e9:.2f} GB)")
    shutil.rmtree(stage, ignore_errors=True)  # 스테이징 복사본 삭제 (디스크 절약)

    # 서버 규칙 검증 (2026-HAIKUTHON 의 evaluation/model_loader.py 재사용)
    srv = Path(__import__("os").environ.get("HAIKUTHON_SERVER_REPO", ""))
    if (srv / "evaluation/model_loader.py").exists():
        sys.path.insert(0, str(srv))
        from evaluation.model_loader import prepare_submission
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            prepare_submission(out, Path(t) / "x")
        print("서버 prepare_submission 검증 통과")
    else:
        print("HAIKUTHON_SERVER_REPO 미설정 -> 서버 규칙 검증 생략 (2026-HAIKUTHON 경로를 지정하면 검증)")


if __name__ == "__main__":
    main()
