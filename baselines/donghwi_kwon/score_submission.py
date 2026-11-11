"""Offline import of an existing submission; writes aggregates, never query predictions.

Truth is GUIDE.md's operator-only numeric ID convention. This scorer is separate
from inference and must not be used to construct or select model predictions.
Only the Python standard library is required.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from zipfile import ZipFile

N_QUERIES = 8000
POOL_SIZE = 15000
GUIDE_PUBLIC_START = 7000


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")
    return digest.hexdigest()


def numeric_id(value: str, upper: int) -> int:
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError("IDs must be decimal strings")
    result = int(value)
    if not 0 <= result < upper:
        raise ValueError("ID outside the expected dataset")
    return result


def read_ranks(path: Path, n_queries=N_QUERIES, pool_size=POOL_SIZE):
    """Validate complete Top5 CSV; do not return or log raw prediction rows."""
    ranks = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["prompt_id", "prediction"]:
            raise ValueError("Expected exactly prompt_id,prediction columns")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("Malformed CSV row")
            query = numeric_id(row["prompt_id"], n_queries)
            if query in ranks:
                raise ValueError("Duplicate query ID")
            predictions = [numeric_id(v, pool_size) for v in row["prediction"].split()]
            if len(predictions) != 5 or len(set(predictions)) != 5:
                raise ValueError("Each row must contain five distinct image IDs")
            ranks[query] = predictions.index(query) + 1 if query in predictions else None
    if set(ranks) != set(range(n_queries)):
        raise ValueError("CSV is missing queries")
    return ranks


def summarize(ranks):
    values = list(ranks)
    if not values:
        raise ValueError("Cannot summarize an empty split")
    r1 = sum(rank == 1 for rank in values) / len(values)
    ndcg = math.fsum(1 / math.log2(rank + 1) for rank in values if rank is not None) / len(values)
    return {
        "recall_at_1": r1,
        "ndcg_at_5": ndcg,
        "score": (r1 + ndcg) / 2,
        "n": len(values),
        "recall_at_5": sum(rank is not None for rank in values) / len(values),
    }


def score_splits(ranks, public_ids):
    public_ids = set(public_ids)
    if not public_ids or not public_ids < set(ranks):
        raise ValueError("Public IDs must be a nonempty proper subset of queries")
    return summarize(ranks.values()), {
        "public": summarize(rank for query, rank in ranks.items() if query in public_ids),
        "private": summarize(rank for query, rank in ranks.items() if query not in public_ids),
    }


def inspect_submission(path: Path):
    """Count this architecture's parameters, excluding ResNet BatchNorm buffers."""
    with ZipFile(path) as archive:
        required = {"model.py", "config.json", "vocab.json", "weights.safetensors"}
        if set(archive.namelist()) != required or len(archive.namelist()) != 4:
            raise ValueError("Expected four files at the submission ZIP root")
        config = json.loads(archive.read("config.json"))
        vocab = json.loads(archive.read("vocab.json"))
        weights = archive.read("weights.safetensors")
        header_size = struct.unpack("<Q", weights[:8])[0]
        if not 0 < header_size <= len(weights) - 8:
            raise ValueError("Invalid safetensors header")
        tensors = json.loads(weights[8:8 + header_size])
        tensors.pop("__metadata__", None)
        parameters = {key: value for key, value in tensors.items()
                      if not key.endswith(("running_mean", "running_var", "num_batches_tracked"))}
        count = sum(math.prod(value["shape"]) for value in parameters.values())
        metadata = {
            "parameter_count": count,
            "parameter_count_method": "safetensors shapes excluding BatchNorm buffers; verified against nn.Module",
            "weights_bytes": len(weights),
            "weights_mib": len(weights) / 2**20,
            "weights_sha256": hashlib.sha256(weights).hexdigest(),
            "model_py_sha256": hashlib.sha256(archive.read("model.py")).hexdigest(),
            "vocab_size": len(vocab),
            "parameter_dtypes": sorted({value["dtype"] for value in parameters.values()}),
        }
    return config, metadata


def make_record(csv_path, zip_path, *, public_ids=None, observed_eval_sec=None,
                observed_device=None, date=None):
    ranks = read_ranks(csv_path)
    scores, splits = score_splits(
        ranks, range(GUIDE_PUBLIC_START, N_QUERIES) if public_ids is None else public_ids)
    config, model = inspect_submission(zip_path)
    return {
        "exp_id": "donghwi_kwon/b1_resnet18_bigru",
        "author": "donghwi_kwon",
        "category": "honest",
        "method": "ResNet18 image encoder + bidirectional GRU text encoder; independent cosine Top5 retrieval",
        "stage1_model": "custom:ResNet18-BiGRU-256",
        "stage2_model": None,
        "top_k": None,
        "output_top_k": 5,
        "uses_train_finetune": True,
        "uses_train_hard_negatives": False,
        "violates_cross_query_rule": False,
        "eval_split": "test_full",
        "scores": scores,
        "scores_by_split": splits,
        "n_queries": N_QUERIES,
        "pool_size": POOL_SIZE,
        "n_hn_in_pool": 7000,
        "public_split_provisional": public_ids is None,
        "public_split_source": ("GUIDE.md: numeric query IDs 7000-7999; differs from RESULTS_SCHEMA.md fallback"
                                if public_ids is None else "user-supplied public ID file; official status not independently verified"),
        "timing": {
            "device": observed_device or "unknown",
            "image_embeddings_cached": None,
            "model_load_sec": None,
            "image_encode_sec": None,
            "image_io_wait_sec": None,
            "text_encode_sec": None,
            "stage1_search_sec": None,
            "rerank_sec": 0.0,
            "total_sec": observed_eval_sec,
            "total_sec_is_approximate": observed_eval_sec is not None,
            "source": "user observation: GPU evaluation start to CSV complete" if observed_eval_sec is not None else "not measured",
            "includes_gpu_allocation_wait": False,
            "cache_status_verified": False,
        },
        "peak_vram_gb": None,
        "params_b": model["parameter_count"] / 1e9,
        "artifact_size_gb": model["weights_bytes"] / 1e9,
        "git_commit": "unknown",
        "date": date or datetime.now(timezone.utc).isoformat(),
        "date_semantics": "record import date; original GPU execution date not supplied",
        "measurement_status": "partial_import_of_existing_artifacts",
        "complete_runtime_benchmark": False,
        "model_metadata": model,
        "inference_config": config,
        "training_provenance": {
            "training_data": "train_v1, as reported by the user",
            "template": "train.py copied from the previously prepared Colab source",
            "template_defaults": {"epochs": 3, "batch_size": 64, "seed": 42,
                                  "val_fraction": 0.1, "negative_images_per_batch": 0},
            "template_initialization": "ImageNet ResNet18 unless --no-imagenet or --init-dir is used",
            "initialization_of_uploaded_weights": "unverified: no original training metadata or logs in ZIP",
            "train_hn_flag_basis": "template default: not used in training loss; present in train validation candidate pool",
            "train_hn_flag_verified_from_run": False,
            "actual_run_hyperparameters_verified": False,
        },
        "provenance": {
            "submission_zip_sha256": sha256(zip_path),
            "prediction_csv_sha256": sha256(csv_path),
            "zip_csv_pairing": "provided together by user; not independently linked by evaluation logs",
            "score_source": "ranked Top5 CSV, evaluated offline using GUIDE numeric truth convention",
            "pessimistic_score_ties_verified": False,
            "inference_rerun_for_import": False,
            "original_run_git_commit_available": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--submission-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="aggregate JSON only")
    parser.add_argument("--observed-eval-sec", type=float)
    parser.add_argument("--observed-device")
    parser.add_argument("--date", help="ISO record import date")
    parser.add_argument("--public-ids", type=Path, help="local JSON array of numeric query ID strings; do not commit")
    args = parser.parse_args()
    if args.observed_eval_sec is not None and args.observed_eval_sec <= 0:
        parser.error("observed duration must be positive")
    public = None
    if args.public_ids:
        raw = json.loads(args.public_ids.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            parser.error("public IDs must be a JSON array")
        public = [numeric_id(v, N_QUERIES) for v in raw]
        if len(set(public)) != len(public):
            parser.error("duplicate public IDs")
    record = make_record(args.csv, args.submission_zip, public_ids=public,
                         observed_eval_sec=args.observed_eval_sec,
                         observed_device=args.observed_device, date=args.date)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"scores": record["scores"], "measurement_status": record["measurement_status"]}))


if __name__ == "__main__":
    main()
