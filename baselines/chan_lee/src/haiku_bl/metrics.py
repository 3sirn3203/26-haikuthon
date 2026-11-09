"""Competition metric: per-query score = (Recall@1 + NDCG@5) / 2, averaged over queries.

With one relevant image at 1-based rank r: Recall@1 = [r == 1],
NDCG@5 = 1 / log2(r + 1) if r <= 5 else 0 (ideal DCG is 1).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

NDCG_K = 5
# Rank used for a ground truth that is absent from a (truncated) ranked list.
MISSING = np.iinfo(np.int64).max


def ndcg5_from_rank(ranks: np.ndarray) -> np.ndarray:
    ranks = np.asarray(ranks, dtype=np.int64)
    out = np.zeros(ranks.shape, dtype=np.float64)
    hit = (ranks >= 1) & (ranks <= NDCG_K)
    out[hit] = 1.0 / np.log2(ranks[hit] + 1.0)
    return out


def per_query(ranks) -> dict[str, np.ndarray]:
    ranks = np.asarray(ranks, dtype=np.int64)
    r1 = (ranks == 1).astype(np.float64)
    nd = ndcg5_from_rank(ranks)
    return {'recall_at_1': r1, 'ndcg_at_5': nd, 'score': (r1 + nd) / 2}


def summarize(ranks, mask: np.ndarray | None = None) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    if mask is not None:
        ranks = ranks[np.asarray(mask, dtype=bool)]
    if ranks.size == 0:
        return {}
    pq = per_query(ranks)
    out = {k: float(v.mean()) for k, v in pq.items()}
    finite = ranks[ranks != MISSING]
    out.update(n=int(ranks.size),
               recall_at_5=float((ranks <= 5).mean()),
               recall_at_10=float((ranks <= 10).mean()),
               median_rank=float(np.median(finite)) if finite.size else math.inf)
    return out


def ranks_from_scores(scores, gt_index, *, chunk: int = 1024) -> np.ndarray:
    """1-based rank of the ground truth in each row of a (Q, N) score matrix.

    Ties are broken pessimistically (an equal-scoring distractor ranks above the GT), so a
    constant scorer cannot look good. Works for numpy arrays and torch tensors.
    """
    gt_index = np.asarray(gt_index, dtype=np.int64)
    try:
        import torch
        is_torch = isinstance(scores, torch.Tensor)
    except ImportError:  # pragma: no cover
        is_torch = False
    out = np.empty(len(gt_index), dtype=np.int64)
    for s in range(0, len(gt_index), chunk):
        block = scores[s:s + chunk]
        idx = gt_index[s:s + chunk]
        if is_torch:
            import torch
            g = block[torch.arange(len(idx)), torch.as_tensor(idx, device=block.device)][:, None]
            better = (block >= g).sum(1).cpu().numpy()
        else:
            block = np.asarray(block)
            g = block[np.arange(len(idx)), idx][:, None]
            better = (block >= g).sum(1)
        out[s:s + chunk] = better          # count includes the GT itself -> 1-based rank
    return out


def ranks_from_rankings(rankings: Sequence[Sequence], gts: Iterable) -> np.ndarray:
    """Rank of each GT inside an explicit ranked list (MISSING if absent)."""
    out = []
    for lst, gt in zip(rankings, gts):
        try:
            out.append(list(lst).index(gt) + 1)
        except ValueError:
            out.append(MISSING)
    return np.asarray(out, dtype=np.int64)


def expected_random(pool_size: int) -> dict[str, float]:
    """Expected metric for a uniformly random ranking over ``pool_size`` images."""
    p = 1.0 / pool_size
    nd = sum(p / math.log2(r + 1) for r in range(1, min(NDCG_K, pool_size) + 1))
    return {'recall_at_1': p, 'ndcg_at_5': nd, 'score': (p + nd) / 2}


def topk_from_scores(scores, k: int):
    """Indices of the top-k columns per row, best first (numpy)."""
    scores = np.asarray(scores)
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1, kind='stable')
    return np.take_along_axis(part, order, 1)
