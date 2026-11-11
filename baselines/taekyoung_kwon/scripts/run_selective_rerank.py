#!/usr/bin/env python3
"""Calibrate a low-margin gate on held-out train bundles, then selectively rerank test queries."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml


AUTHOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUTHOR_ROOT.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "baselines" / "chan_lee" / "src"))

from haiku_bl import data as D, metrics as M, pipeline, rerankers, retrievers  # noqa: E402
from haiku_bl.profiling import Profiler, pick_device  # noqa: E402


def _embedding_cache(cfg, es, paths):
    ret = cfg["retriever"]
    name = ret["model"] if ret.get("pretrained") is None else f"{ret['model']}:{ret['pretrained']}"
    key = retrievers.cache_key(name, ret.get("checkpoint"))
    ids = "\n".join(im.iid for im in es.pool)
    digest = hashlib.sha1(ids.encode()).hexdigest()[:10]
    return paths["cache_dir"] / "emb" / key / f"{es.name}_{digest}.npy"


def _stage1(cfg, split, paths, device, k):
    es = D.build_eval_set(split, paths)
    missing = es.missing_images()
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing images, e.g. {missing[:3]}")
    prof = Profiler(device)
    s, s1, info = pipeline.score(
        cfg,
        es.prompts,
        [im.path for im in es.pool],
        prof,
        device,
        emb_cache=_embedding_cache(cfg, es, paths),
        eval_set=es,
    )
    del s
    # Best-first candidates and the stage-1 top-1/top-2 score margin.
    top = M.topk_from_scores(s1, k)
    top_scores = np.take_along_axis(s1, top[:, :2], axis=1)
    gap = top_scores[:, 0] - top_scores[:, 1]
    result = (es, s1, top, gap, prof.report(), info)
    del top_scores
    return result


def _resume_path(paths, name, split):
    p = paths["cache_dir"] / "rerank" / f"{name}_{split}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _score_rows(rr, es, top, q_indices, store, prof=None, batch_flush_sec=60):
    done = {}
    pair_sec = []
    if store.exists():
        with np.load(store, allow_pickle=False) as z:
            done = {int(q): row for q, row in zip(z["qi"], z["scores"])}
            pair_sec = z["pair_sec"].tolist()
        print(f"resume {store.name}: {len(done)}/{len(q_indices)} queries", flush=True)

    paths_pool = [im.path for im in es.pool]
    started = time.time()
    t_last = started
    for qi in q_indices:
        qi = int(qi)
        if qi in done:
            continue
        t0 = time.perf_counter()
        sc = rr.score(es.queries[qi].prompt, [paths_pool[j] for j in top[qi]], prof)
        pair_sec.append((time.perf_counter() - t0) / top.shape[1])
        done[qi] = np.asarray(sc, dtype=np.float32)
        if time.time() - t_last >= batch_flush_sec or len(done) == len(q_indices):
            keys = np.array(sorted(done), dtype=np.int32)
            np.savez_compressed(
                store,
                qi=keys,
                scores=np.stack([done[k] for k in keys]),
                pair_sec=np.asarray(pair_sec, dtype=np.float32),
            )
            t_last = time.time()
            print(
                f"{store.stem}: {len(done)}/{len(q_indices)} queries, "
                f"median {np.median(pair_sec):.3f}s/pair, elapsed {(time.time()-started)/60:.1f} min",
                flush=True,
            )
    keys = np.array(sorted(done), dtype=np.int32)
    np.savez_compressed(
        store,
        qi=keys,
        scores=np.stack([done[k] for k in keys]),
        pair_sec=np.asarray(pair_sec, dtype=np.float32),
    )
    if len(done) != len(q_indices):
        raise RuntimeError(f"Only scored {len(done)} of {len(q_indices)} selected queries")
    return done, pair_sec


def _candidate_metrics(es, s1, top, rr_by_q, selected, k):
    final = s1.copy()
    for qi in selected:
        final[qi, top[qi, :k]] = 1e3 + rr_by_q[int(qi)][:k]
    ranks = M.ranks_from_scores(final, es.gt_index)
    masks = es.masks()
    by = {name: dict(M.summarize(ranks, mask), ci95=M.bootstrap_ci(ranks, mask))
          for name, mask in masks.items()}
    return ranks, by


def _calibrate(es, s1, top, gap, rr_by_q, cfg):
    rows = []
    n = len(gap)
    for rate in cfg["validation_gate_rates"]:
        count = min(n, int(round(rate * n)))
        selected = np.argsort(gap, kind="stable")[:count]
        ranks, by = _candidate_metrics(es, s1, top, rr_by_q, selected, cfg["top_k"])
        rows.append({
            "gate_rate": float(rate),
            "selected_queries": int(count),
            "score": M.summarize(ranks)["score"],
            "scores": M.summarize(ranks),
            "val_ci95": M.bootstrap_ci(ranks),
            "scores_by_type": {k: by[k] for k in ("easy", "bundle") if k in by},
        })
    cap = cfg.get("max_validation_gate_rate", 0.08)
    eligible = [r for r in rows if r["gate_rate"] <= cap]
    # Prefer lower inference cost on a score tie.
    chosen = max(eligible, key=lambda r: (r["score"], -r["gate_rate"]))
    n_selected = chosen["selected_queries"]
    cutoff = float(np.sort(gap, kind="stable")[max(0, n_selected - 1)]) if n_selected else None
    return {"candidates": rows, "chosen": chosen, "gate_fraction": chosen["gate_rate"],
            "margin_cutoff_val": cutoff, "top_k": cfg["top_k"]}


def _top1_breakdown(top1, es):
    bundle_of = {q.pid: q.bundle for q in es.queries}
    cats = dict.fromkeys(("gt", "own_hn", "bundle_sibling", "bundle_sibling_hn", "other"), 0)
    for q, j in zip(es.queries, top1):
        im = es.pool[j]
        if im.iid == q.gt:
            cats["gt"] += 1
        elif im.kind == "hn" and im.source_pid == q.pid:
            cats["own_hn"] += 1
        elif q.bundle is not None and bundle_of.get(im.source_pid) == q.bundle:
            cats["bundle_sibling_hn" if im.kind == "hn" else "bundle_sibling"] += 1
        else:
            cats["other"] += 1
    return {k: v / len(es.queries) for k, v in cats.items()}


def _artifact_size_gb(paths, checkpoint, fallback_params_b):
    hub = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    groups = (
        hub / "models--timm--ViT-SO400M-16-SigLIP2-384" / "blobs",
        hub / "models--Qwen--Qwen3.5-2B" / "blobs",
    )
    measured = []
    for root in groups:
        files = [p for p in root.iterdir() if p.is_file()] if root.exists() else []
        files = [p for p in files if not p.name.endswith(".incomplete")]
        if not files:
            return fallback_params_b * 2, "estimated_fp16"
        measured.append(max(p.stat().st_size for p in files))
    if not checkpoint.exists():
        return fallback_params_b * 2, "estimated_fp16"
    measured.append(checkpoint.stat().st_size)
    return sum(measured) / 1e9, "measured_local_weight_files_decimal_GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    paths = D.load_paths(args.paths or AUTHOR_ROOT / "configs" / "paths.yaml")
    name = cfg["exp_id"].split("/", 1)[1]
    s1cfg = yaml.safe_load(Path(cfg["stage1_config"]).read_text(encoding="utf-8"))
    ckpt = Path(cfg["stage1_checkpoint"])
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    s1cfg["retriever"]["checkpoint"] = str(ckpt)

    device = pick_device(args.device or "auto")
    val_es, val_s1, val_top, val_gap, val_timing, val_info = _stage1(
        s1cfg, "val", paths, device, cfg["top_k"]
    )
    test_es, test_s1, test_top, test_gap, test_timing, test_info = _stage1(
        s1cfg, "test", paths, device, cfg["top_k"]
    )

    import torch
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    rr_prof = Profiler(device, 24)
    with rr_prof.stage("model_load"):
        rr = rerankers.build(cfg["reranker"], device)

    val_store = _resume_path(paths, name, "val")
    val_rows, val_pair_sec = _score_rows(
        rr, val_es, val_top, np.arange(len(val_es.queries)), val_store, rr_prof
    )
    calibration = _calibrate(val_es, val_s1, val_top, val_gap, val_rows, cfg)
    cal_path = paths["cache_dir"] / "rerank" / f"{name}_calibration.json"
    cal_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    chosen = calibration["chosen"]
    gate_rate = calibration["gate_fraction"]

    # The gate ratio was selected on validation labels. On test, use only the score margin
    # distribution (no test labels) to select the same fixed fraction of low-margin queries.
    n_test_selected = int(round(gate_rate * len(test_gap)))
    test_selected = np.argsort(test_gap, kind="stable")[:n_test_selected]
    test_store = _resume_path(paths, name, "test")
    test_rows, test_pair_sec = _score_rows(rr, test_es, test_top, test_selected, test_store, rr_prof)
    ranks, by = _candidate_metrics(test_es, test_s1, test_top, test_rows, test_selected, cfg["top_k"])
    masks = test_es.masks()
    summaries = {k: dict(M.summarize(ranks, m), ci95=M.bootstrap_ci(ranks, m)) for k, m in masks.items()}
    base_ranks = M.ranks_from_scores(test_s1, test_es.gt_index)
    base = M.summarize(base_ranks)
    stage1_result_path = AUTHOR_ROOT / "results" / s1cfg["exp_id"].split("/", 1)[1]
    # Trainer writes the full-test directory without a suffix; use it for stage-1 timing.
    b3_metrics = json.loads((stage1_result_path / "metrics.json").read_text(encoding="utf-8"))
    rr_time = float(sum(test_pair_sec) * cfg["top_k"])
    rr_load_time = float(rr_prof.times.get("model_load", 0.0))
    artifact_gb, artifact_size_method = _artifact_size_gb(paths, ckpt, b3_metrics["params_b"] + rr.params_b)
    timing = {
        "device": rr_prof.report()["device"],
        "model_load_sec": b3_metrics["timing"].get("model_load_sec", 0) + rr_load_time,
        "image_encode_sec": b3_metrics["timing"].get("image_encode_sec"),
        "text_encode_sec": b3_metrics["timing"].get("text_encode_sec"),
        "stage1_search_sec": b3_metrics["timing"].get("stage1_search_sec"),
        "rerank_sec": rr_time,
        "rerank_sec_is_measured": True,
        "reranked_queries": int(len(test_selected)),
        "reranked_pairs": int(len(test_selected) * cfg["top_k"]),
        "total_sec": (b3_metrics["timing"].get("total_sec", 0) + rr_time + rr_load_time),
    }
    result = {
        "exp_id": cfg["exp_id"], "author": cfg["exp_id"].split("/")[0], "category": "honest",
        "method": cfg["method"],
        "stage1_model": b3_metrics["stage1_model"], "stage2_model": cfg["reranker"]["model"],
        "top_k": cfg["top_k"], "uses_train_finetune": True, "uses_train_hard_negatives": True,
        "violates_cross_query_rule": False, "eval_split": "test_full",
        "n_queries": len(test_es.queries), "pool_size": len(test_es.pool),
        "n_hn_in_pool": sum(im.kind == "hn" for im in test_es.pool),
        "scores": dict(M.summarize(ranks), ci95=M.bootstrap_ci(ranks)),
        "scores_by_split": {k: summaries[k] for k in ("public", "private") if k in summaries},
        "scores_by_type": {k: summaries[k] for k in ("easy", "bundle") if k in summaries},
        "public_split_provisional": test_es.public_provisional,
        "public_split_source": test_es.public_source,
        "gate": {"selection": "lowest stage-1 top1-top2 margin", "fraction_from_validation": gate_rate,
                 "validation_margin_cutoff": calibration["margin_cutoff_val"],
                 "test_selected_queries": int(len(test_selected)), "validation_candidates": calibration["candidates"],
                 "chosen_validation_candidate": chosen},
        "stage1_scores_same_test": base,
        "diagnostics": {"top1_breakdown": _top1_breakdown(M.topk_from_scores(
            _scored_matrix(test_s1, test_top, test_rows, test_selected, cfg["top_k"]), 1)[:, 0], test_es)},
        "timing": timing,
        "peak_vram_gb": max(float(b3_metrics.get("peak_vram_gb") or 0), rr_prof.peak_gb),
        "params_b": b3_metrics["params_b"] + rr.params_b,
        "artifact_size_gb": artifact_gb, "artifact_size_method": artifact_size_method,
        "git_commit": "local-dirty", "date": dt.datetime.now().isoformat(timespec="seconds"),
    }
    out = AUTHOR_ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    note = (
        f"# {cfg['exp_id']}\n\n{cfg['method']}\n\n"
        f"- Validation chose a {gate_rate:.1%} low-margin gate and top-{cfg['top_k']} reranking.\n"
        f"- Test reranked {len(test_selected)}/{len(test_es.queries)} queries.\n"
        f"- Stage-1 test score: {base['score']:.5f}; selective result: {result['scores']['score']:.5f}.\n"
        f"- Test rerank time: {rr_time:.1f}s on {timing['device']}; this local GPU time is not an L4 estimate.\n"
        "- Public/private split is provisional until official IDs are supplied.\n"
    )
    (out / "notes.md").write_text(note, encoding="utf-8")
    print(json.dumps({"calibration": calibration, "result": result["scores"], "stage1": base,
                      "reranked_queries": len(test_selected), "rerank_sec": rr_time,
                      "result_dir": str(out)}, indent=2), flush=True)


def _scored_matrix(s1, top, rr_by_q, selected, k):
    final = s1.copy()
    for qi in selected:
        final[qi, top[qi, :k]] = 1e3 + rr_by_q[int(qi)][:k]
    return final


if __name__ == "__main__":
    main()
