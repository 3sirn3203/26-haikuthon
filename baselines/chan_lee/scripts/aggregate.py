#!/usr/bin/env python
"""Flatten results/*/metrics.json into results/leaderboard.csv (one row per experiment).

    python scripts/aggregate.py                 # this author's results
    python scripts/aggregate.py --all-authors   # baselines/*/results/** -> one combined table
"""
import argparse
import json
from pathlib import Path

import pandas as pd

CHAN = Path(__file__).resolve().parents[1]


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


FRONT = ['exp_id', 'category', 'eval_split', 'scores.score', 'scores.recall_at_1', 'scores.ndcg_at_5',
         'scores_by_split.public.score', 'scores_by_split.private.score', 'timing.device',
         'timing.total_sec', 'peak_vram_gb', 'params_b', 'stage1_model', 'stage2_model', 'top_k',
         'uses_train_finetune', 'uses_train_hard_negatives', 'violates_cross_query_rule']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all-authors', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    roots = sorted(CHAN.parent.glob('*/results')) if a.all_authors else [CHAN / 'results']
    rows = [flatten(json.loads(f.read_text())) for r in roots for f in sorted(r.glob('*/metrics.json'))]
    df = pd.DataFrame(rows)
    if df.empty:
        print('no results'); return
    cols = [c for c in FRONT if c in df] + sorted(c for c in df if c not in FRONT)
    df = df[cols].sort_values(['eval_split', 'category', 'scores.score'], ascending=[True, True, False])
    out = Path(a.out) if a.out else (CHAN.parent / 'leaderboard_all.csv' if a.all_authors
                                     else CHAN / 'results' / 'leaderboard.csv')
    df.to_csv(out, index=False, float_format='%.5f')
    print(df[[c for c in FRONT[:9] if c in df]].to_string(index=False))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
