"""Category ``loophole``: methods the rules forbid or that sidestep the task's intent.

They exist only to measure how much they would gain, so prohibitions and detection can be
designed. Every function takes the honest (Q, N) score matrix and returns a modified one;
queries are coupled, so all of them set ``violates_cross_query_rule``.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from .metrics import topk_from_scores

BONUS = 1e3   # pushes the assigned image to rank 1 while keeping the honest order behind it


def _apply(scores, rows, cols):
    out = scores.astype(np.float32, copy=True)
    out[rows, cols] += BONUS
    return out


def greedy_one_to_one(scores, top_k: int = 50):
    """Greedy 1:1 assignment: highest-scoring (query, image) pair first, both then taken."""
    top = topk_from_scores(scores, top_k)
    q = np.repeat(np.arange(len(scores)), top.shape[1])
    i = top.ravel()
    order = np.argsort(-scores[q, i], kind='stable')
    used_q, used_i, rows, cols = set(), set(), [], []
    for o in order:
        if q[o] in used_q or i[o] in used_i:
            continue
        used_q.add(q[o]); used_i.add(i[o]); rows.append(q[o]); cols.append(i[o])
    return _apply(scores, np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))


def hungarian(scores, top_k: int = 50):
    """Global max-weight 1:1 matching (queries -> images) restricted to each query's top-k.

    Falls back to greedy for queries left unmatched if the sparse graph has no full matching.
    """
    Q, N = scores.shape
    top = topk_from_scores(scores, top_k)
    rows = np.repeat(np.arange(Q), top.shape[1])
    cols = top.ravel()
    # min-weight matching needs positive weights; shift so the best pair costs least.
    w = scores[rows, cols].astype(np.float64)
    cost = (w.max() - w) + 1e-6
    try:
        r, c = min_weight_full_bipartite_matching(csr_matrix((cost, (rows, cols)), shape=(Q, N)))
        return _apply(scores, r, c)
    except ValueError:
        return greedy_one_to_one(scores, top_k)


def group_hungarian(scores, groups: list[list[int]], top_k: int = 8):
    """1:1 matching inside each group of queries (e.g. a stage1 bundle of 8 prompts).

    Candidates are the union of the group's per-query top-k. Uses the oracle stage1 bundle
    grouping; participants would have to recover it from prompt text (same background,
    medium, viewpoint), so this is an upper bound on what the trick gains.
    """
    top = topk_from_scores(scores, top_k)
    rows, cols = [], []
    for g in groups:
        if len(g) < 2:
            continue
        cand = np.unique(top[g].ravel())
        sub = scores[np.ix_(g, cand)]
        r, c = linear_sum_assignment(-sub)
        rows += [g[x] for x in r]
        cols += cand[c].tolist()
    return _apply(scores, np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64))


def bundle_groups(queries) -> list[list[int]]:
    d = defaultdict(list)
    for i, q in enumerate(queries):
        if q.bundle is not None:
            d[q.bundle].append(i)
    return list(d.values())


def apply(cfg: dict, scores, eval_set):
    kind = cfg['type']
    if kind == 'greedy':
        return greedy_one_to_one(scores, cfg.get('top_k', 50))
    if kind == 'hungarian':
        return hungarian(scores, cfg.get('top_k', 50))
    if kind == 'bundle_hungarian':          # oracle stage1 bundles (upper bound)
        return group_hungarian(scores, bundle_groups(eval_set.queries), cfg.get('top_k', 8))
    if kind == 'text_group_hungarian':      # groups a participant can recover from prompt text alone
        return group_hungarian(scores, text_groups(eval_set.queries), cfg.get('top_k', 8))
    raise ValueError(f'unknown postprocess {kind!r}')


# ------------------------------------------------- bundle recovery from prompt text only
# A participant can read the 8,000 test prompts and list the ~40 places, 12 media and 4 viewpoints;
# grouping by (place, medium, viewpoint) recovers a superset of stage1's 8-prompt bundles.
_LOCATIONS = sorted(['airport apron', 'art studio', 'bakery', 'beach', 'bedroom', 'cafe', 'campsite', 'classroom',
                     'clothing store', 'dining room', 'farm', 'forest', 'garage', 'garden', 'greenhouse', 'harbor',
                     'home office', 'kitchen', 'lakeside', 'library', 'living room', 'meadow', 'office', 'orchard',
                     'park', 'parking lot', 'pasture', 'patio', 'playground', 'riverside', 'schoolyard',
                     'snow field', 'sports field', 'sports hall', 'street', 'studio', 'train platform', 'workshop'],
                    key=len, reverse=True)
_STYLES = [('low_poly_3d', r'low[- ]poly'), ('clay_render', r'\bclay'), ('colored_pencil', r'colou?red[- ]pencil'),
           ('pencil_sketch', r'pencil sketch|graphite'), ('flat_illustration', r'flat (vector )?illustration'),
           ('gouache', r'gouache'), ('ink_drawing', r'\bink\b'), ('oil_painting', r'oil painting|\boil\b'),
           ('paper_cut', r'paper[- ]?cut'), ('watercolor', r'watercolou?r'), ('render_3d', r'3d|three-dimensional'),
           ('photograph', r'photo')]
_VIEWS = [('overhead', r'overhead|top-down|bird'), ('oblique', r'oblique|angled|three-quarter'),
          ('side', r'\bside view|from the side|side-on|\bside\b'), ('front', r'front')]


def text_bundle_key(prompt: str):
    import re
    t = prompt.lower()
    loc = next((l for l in _LOCATIONS if l in t), None)
    sty = next((k for k, rx in _STYLES if re.search(rx, t)), None)
    view = next((k for k, rx in _VIEWS if re.search(rx, t)), None)
    return loc, sty, view


def text_groups(queries) -> list[list[int]]:
    d = defaultdict(list)
    for i, q in enumerate(queries):
        d[text_bundle_key(q.prompt)].append(i)
    return list(d.values())
