"""Ranking -> submission CSV.

Assumed Kaggle format (to be confirmed with HAI's scoring server):
    id,image_ids
    0000,img_a img_b img_c ...
``id`` is the test prompt id and ``image_ids`` is a space-separated ranked list, best
first. The metric only looks at the top 5, so rows are truncated to ``top_k`` (default 10)
instead of listing all 15,000 images.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .metrics import ranks_from_rankings, summarize

HEADER = ('id', 'image_ids')
DEFAULT_TOP_K = 10


def write_submission(path, query_ids: Sequence[str], rankings: Iterable[Sequence[str]],
                     *, top_k: int = DEFAULT_TOP_K, image_ids: set | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid, ranked in zip(query_ids, rankings, strict=True):
        ranked = list(ranked)[:top_k]
        validate_row(qid, ranked, image_ids)
        rows.append((qid, ' '.join(ranked)))
    if len({q for q, _ in rows}) != len(rows):
        raise ValueError('duplicate query id in submission')
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    return path


def validate_row(qid, ranked, image_ids=None):
    if not ranked:
        raise ValueError(f'{qid}: empty ranking')
    if len(set(ranked)) != len(ranked):
        raise ValueError(f'{qid}: duplicate image id in ranking')
    if any((' ' in r or ',' in r) for r in ranked):
        raise ValueError(f'{qid}: image id contains a separator')
    if image_ids is not None:
        unknown = [r for r in ranked if r not in image_ids]
        if unknown:
            raise ValueError(f'{qid}: unknown image ids {unknown[:3]}')


def read_submission(path) -> dict[str, list[str]]:
    with Path(path).open(encoding='utf-8') as f:
        r = csv.reader(f)
        header = tuple(next(r))
        if header != HEADER:
            raise ValueError(f'bad header {header}')
        out = {}
        for qid, ids in r:
            if qid in out:
                raise ValueError(f'duplicate query id {qid}')
            out[qid] = ids.split()
    return out


def score_submission(path, ground_truth: dict[str, str], masks: dict[str, Sequence[str]] | None = None):
    """Organizer-side scoring. ground_truth: query id -> GT image id. Missing rows score 0."""
    sub = read_submission(path)
    qids = sorted(ground_truth)
    ranks = ranks_from_rankings([sub.get(q, []) for q in qids], [ground_truth[q] for q in qids])
    out = {'all': summarize(ranks)}
    for name, ids in (masks or {}).items():
        ids = set(ids)
        out[name] = summarize(ranks, np.array([q in ids for q in qids]))
    return out
