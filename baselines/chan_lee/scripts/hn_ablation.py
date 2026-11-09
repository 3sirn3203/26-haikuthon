#!/usr/bin/env python
"""Question 4 (difficulty): how the score changes with the number of test HN images in the pool.

Uses cached stage-1 embeddings from a B1 run (run run_exp.py first). For n_hn in a grid, samples
that many HN images (3 seeds), and reports the score of all queries and of queries whose own HN is
in the pool. No images are re-encoded.

    python scripts/hn_ablation.py --config configs/exp/b1_siglip2_so400m384.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

CHAN = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHAN / 'src'))
sys.path.insert(0, str(CHAN / 'scripts'))
from haiku_bl import data as D, metrics as M  # noqa: E402
from run_exp import emb_cache_path  # noqa: E402

GRID = (0, 1000, 1750, 3500, 5250, 7000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    paths = D.load_paths()
    es = D.build_eval_set('test', paths)
    f = emb_cache_path(cfg, es, paths)
    img = np.load(f).astype(np.float32)
    txt = np.load(f.with_name(f.stem + f'__text{len(es.queries)}.npy')).astype(np.float32)
    S = txt @ img.T
    is_hn = np.array([im.kind == 'hn' for im in es.pool])
    orig_idx, hn_idx = np.where(~is_hn)[0], np.where(is_hn)[0]
    hn_src = np.array([es.pool[j].source_pid for j in hn_idx])
    pid = np.array([q.pid for q in es.queries])
    rows = []
    for n in GRID:
        for seed in range(a.seeds if 0 < n < len(hn_idx) else 1):
            rng = np.random.default_rng(seed)
            pick = rng.choice(len(hn_idx), n, replace=False) if n else np.array([], int)
            cols = np.concatenate([orig_idx, hn_idx[pick]])
            ranks = M.ranks_from_scores(S[:, cols], es.gt_index)   # GT columns are orig_idx == 0..7999
            own = np.isin(pid, hn_src[pick])
            rows.append({'n_hn': int(n), 'seed': seed, 'all': M.summarize(ranks)['score'],
                         'own_hn_in_pool': M.summarize(ranks, own)['score'] if own.any() else None,
                         'own_hn_absent': M.summarize(ranks, ~own)['score'] if (~own).any() else None})
    agg = {}
    for n in GRID:
        rs = [r for r in rows if r['n_hn'] == n]
        agg[n] = {k: float(np.mean([r[k] for r in rs])) if rs[0][k] is not None else None
                  for k in ('all', 'own_hn_in_pool', 'own_hn_absent')}
    name = cfg['exp_id'].split('/', 1)[1]
    out = CHAN / 'results' / 'a1_hn_ablation'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{name}.json').write_text(json.dumps({'model': cfg['exp_id'], 'grid': agg, 'runs': rows}, indent=1))
    for n, v in agg.items():
        print(n, {k: (round(x, 4) if x is not None else None) for k, x in v.items()})


if __name__ == '__main__':
    main()
