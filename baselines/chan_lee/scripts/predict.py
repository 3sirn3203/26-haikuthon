#!/usr/bin/env python
"""Scoring-server style inference: no ground truth, no caches, timed from process start.

    python scripts/predict.py --config configs/exp/<exp>.yaml \
        --images <dir of test images> --prompts <prompts .json/.jsonl with id,prompt> \
        --out submission.csv [--timing-out timing.json] [--top-k 10] [--vram-budget-gb 24]

Image ids are file names without extension (recursively under --images). Each query is
ranked independently (cross-query post-processing is refused here: it is against the rules).
"""
import argparse
import json
import sys
import time
from pathlib import Path

T0 = time.perf_counter()                      # include import/startup time in the total
CHAN = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHAN / 'src'))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from haiku_bl import pipeline  # noqa: E402
from haiku_bl.metrics import topk_from_scores  # noqa: E402
from haiku_bl.profiling import Profiler, pick_device  # noqa: E402
from haiku_bl.submission import write_submission  # noqa: E402

EXTS = {'.png', '.jpg', '.jpeg', '.webp'}


def read_prompts(path):
    text = Path(path).read_text(encoding='utf-8')
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(l) for l in text.splitlines() if l.strip()]
    return [r['id'] for r in rows], [r['prompt'] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--images', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--timing-out', default=None)
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--vram-budget-gb', type=float, default=24)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if cfg.get('postprocess'):
        sys.exit('predict.py refuses cross-query post-processing (category=loophole)')

    device = pick_device(a.device)
    prof = Profiler(device, a.vram_budget_gb)
    with prof.stage('list_inputs'):
        paths = sorted(p for p in Path(a.images).rglob('*') if p.suffix.lower() in EXTS)
        image_ids = [p.stem for p in paths]
        if len(set(image_ids)) != len(image_ids):
            sys.exit('duplicate image ids')
        qids, prompts = read_prompts(a.prompts)
    scores, _, info = pipeline.score(cfg, prompts, paths, prof, device, emb_cache=None)
    with prof.stage('write_submission'):
        top = topk_from_scores(scores, a.top_k)
        write_submission(a.out, qids, [[image_ids[j] for j in row] for row in top], top_k=a.top_k)
    rep = prof.report()
    rep['total_sec_including_startup'] = time.perf_counter() - T0
    rep.update(n_queries=len(qids), n_images=len(paths), params_b=info['params_b'],
               stage1_model=info['stage1_model'], config=a.config)
    print(json.dumps(rep, indent=2))
    if a.timing_out:
        Path(a.timing_out).write_text(json.dumps(rep, indent=2))


if __name__ == '__main__':
    main()
