#!/usr/bin/env python
"""Organizer tool: recover which original each hard-negative image was perturbed from.

The data packages label HN rows only as "hard_negative". HN images are generated from a prompt
that differs from the source prompt by one slot; if they reuse the source's seed, the two images
share layout. Pixel NN on thumbnails confirmed the convention HN file id = n_orig + k <-> prompt k
(test: rank 1 for 82%, median sim 0.83 vs 0.04 random), so the mapping uses that rule and
stores the pixel evidence next to it.

    python scripts/pair_hn.py --split test      # -> cache/meta/test_hn_sources.json (git-ignored)

Output maps HN file id -> {source pid (order rule), pixel rank of that source, pixel NN}. A low
pixel rank confirms the rule; the rule itself is a leak (HN order mirrors source order).
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

CHAN = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(CHAN / 'src'))
from haiku_bl import data as D  # noqa: E402

SIDE = 32


def thumb(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert('L').resize((SIDE, SIDE), Image.BILINEAR), dtype=np.float32).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    a = ap.parse_args()
    cfg = D.load_paths()
    pkg = D.package(a.split, cfg)
    orig = sorted(pkg['orig'].items())          # (pid, path)
    hn = sorted(pkg['hn'].items())              # (file id, path)
    with ThreadPoolExecutor(12) as ex:
        O = np.stack(list(ex.map(thumb, [p for _, p in orig])))
        H = np.stack(list(ex.map(thumb, [p for _, p in hn])))
    sim = H @ O.T                               # cosine of mean-centred thumbnails
    top2 = np.argsort(-sim, axis=1)[:, :2]
    best = sim[np.arange(len(H)), top2[:, 0]]
    second = sim[np.arange(len(H)), top2[:, 1]]
    dist_ratio = (1 - best) / np.maximum(1 - second, 1e-6)
    # Generation convention (verified below): HN file id = n_orig + k is the variant of prompt k.
    width = len(orig[0][0])
    pid_index = {p: i for i, (p, _) in enumerate(orig)}
    out = {}
    for row, ((fid, _), j, b, r) in enumerate(zip(hn, top2[:, 0], best, dist_ratio)):
        k = f'{int(fid) - len(orig):0{width}d}'
        s_k = float(sim[row, pid_index[k]])
        out[fid] = {'source': k, 'sim_to_source': s_k,
                    'pixel_rank_of_source': int((sim[row] > s_k).sum()) + 1,
                    'pixel_nn': orig[j][0], 'pixel_nn_sim': float(b), 'dist_ratio': float(r)}
    f = cfg['cache_dir'] / 'meta' / f'{a.split}_hn_sources.json'
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=0))

    ranks = np.array([v['pixel_rank_of_source'] for v in out.values()])
    src = [int(v['pixel_nn']) for v in out.values()]
    fids = [int(k) for k in out]
    offsets = np.array(fids) - np.array(src)
    vals, cnt = np.unique(offsets, return_counts=True)
    r = np.array([v['dist_ratio'] for v in out.values()])
    print(json.dumps({
        'split': a.split, 'n_hn': len(hn), 'n_orig': len(orig),
        'distinct_sources': len(set(src)),
        'dist_ratio_quantiles': {q: float(np.quantile(r, q)) for q in (0.5, 0.9, 0.99, 1.0)},
        'confident(<0.5)': int((r < 0.5).sum()),
        'best_sim_quantiles': {q: float(np.quantile([v['pixel_nn_sim'] for v in out.values()], q)) for q in (0.0, 0.01, 0.5)},
        'most_common_fileid_minus_source': [(int(v), int(c)) for v, c in
                                            sorted(zip(vals, cnt), key=lambda x: -x[1])[:3]],
        'order_rule_pixel_rank': {'1': int((ranks == 1).sum()), '<=3': int((ranks <= 3).sum()),
                                  'median': float(np.median(ranks))},
        'sources_without_hn': len(set(p for p, _ in orig) - {v['source'] for v in out.values()}),
    }, indent=1))
    print(f'-> {f}')


if __name__ == '__main__':
    main()
