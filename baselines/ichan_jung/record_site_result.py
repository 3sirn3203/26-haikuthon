"""HAIKUTHON 제출 사이트(L4 24GB)에서 받은 결과를 RESULTS_SCHEMA 형식 metrics.json으로 기록.

사이트가 보여 주는 값만 넣고, 보고되지 않는 값은 생략하면 null로 남는다(추정치로 채우지 않음).

  python record_site_result.py --exp b1_pickscore_zs --score 0.xx --total-sec 123.4
  python record_site_result.py --exp b1_pickscore_zs --score 0.xx --r1 0.xx --ndcg5 0.xx \
      --public-score 0.xx --private-score 0.xx --total-sec 123.4 --eval-split test_full
"""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

AUTHOR = "ichan_jung"
# 제출 ZIP 사양 (export_submission.py --model pickscore 결과)
PRESET = {
    "b1_pickscore_zs": {
        "method": "B1 zero-shot dual encoder: PickScore v1 (CLIP-H/14 fine-tuned on Pick-a-Pic "
                  "prompt-generated image preferences); cosine similarity over the full pool; "
                  "pure-torch model.py, fp16, evaluated on the HAIKUTHON submission site",
        "stage1_model": "yuvalkirstain/PickScore_v1",
        "params_b": 0.986109441,
        "artifact_size_gb": 1.973729186,  # 실제 submission.zip 크기 (bytes / 1e9)
    },
}


def split_scores(score, r1, ndcg5):
    if score is None and r1 is None and ndcg5 is None:
        return None
    if score is None and r1 is not None and ndcg5 is not None:
        score = (r1 + ndcg5) / 2
    return {"recall_at_1": r1, "ndcg_at_5": ndcg5, "score": score}


def git_commit():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "--", "."]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="b1_pickscore_zs", choices=PRESET)
    ap.add_argument("--eval-split", default="test_full",
                    choices=["test_full", "test_public", "test_private", "val"])
    ap.add_argument("--score", type=float)
    ap.add_argument("--r1", type=float)
    ap.add_argument("--ndcg5", type=float)
    ap.add_argument("--public-score", type=float)
    ap.add_argument("--public-r1", type=float)
    ap.add_argument("--public-ndcg5", type=float)
    ap.add_argument("--private-score", type=float)
    ap.add_argument("--private-r1", type=float)
    ap.add_argument("--private-ndcg5", type=float)
    ap.add_argument("--total-sec", type=float, help="사이트에 표시된 추론 시간 (load_model~predict)")
    ap.add_argument("--peak-vram-gb", type=float, help="사이트/운영진이 알려 준 경우에만")
    ap.add_argument("--submission-id", help="사이트 제출 ID (추적용)")
    ap.add_argument("--n-queries", type=int, default=8000)
    ap.add_argument("--pool-size", type=int, default=15000)
    ap.add_argument("--n-hn", type=int, default=7000)
    args = ap.parse_args()

    p = PRESET[args.exp]
    metrics = {
        "exp_id": f"{AUTHOR}/{args.exp}",
        "author": AUTHOR,
        "category": "honest",
        "method": p["method"],
        "stage1_model": p["stage1_model"],
        "stage2_model": None,
        "top_k": None,
        "uses_train_finetune": False,
        "uses_train_hard_negatives": False,
        "violates_cross_query_rule": False,
        "eval_split": args.eval_split,
        "n_queries": args.n_queries,
        "pool_size": args.pool_size,
        "n_hn_in_pool": args.n_hn,
        "scores": split_scores(args.score, args.r1, args.ndcg5),
        "scores_by_split": {
            "public": split_scores(args.public_score, args.public_r1, args.public_ndcg5),
            "private": split_scores(args.private_score, args.private_r1, args.private_ndcg5),
        },
        "timing": {
            "device": "cuda:L4",
            "image_embeddings_cached": False,
            "model_load_sec": None,
            "image_encode_sec": None,
            "image_io_wait_sec": None,
            "text_encode_sec": None,
            "stage1_search_sec": None,
            "rerank_sec": 0.0,
            "total_sec": args.total_sec,
            "note": "HAIKUTHON 제출 사이트 실측(L4 24GB, CPU 4코어, RAM 16GiB). 사이트는 "
                    "load_model~predict 전체 시간만 보고하므로 구간별 시간은 null",
        },
        "peak_vram_gb": args.peak_vram_gb,
        "params_b": p["params_b"],
        "artifact_size_gb": p["artifact_size_gb"],
        "git_commit": git_commit(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "result_source": "haikuthon_submission_site",
        "site_submission_id": args.submission_id,
    }
    out = Path(__file__).parent / "results" / args.exp / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
