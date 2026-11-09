import math

import numpy as np
import pytest

from haiku_bl import metrics as M


def test_ndcg_values():
    r = np.array([1, 2, 3, 5, 6, 100])
    want = [1.0, 1 / math.log2(3), 0.5, 1 / math.log2(6), 0.0, 0.0]
    assert np.allclose(M.ndcg5_from_rank(r), want)


def test_score_is_mean_of_r1_and_ndcg():
    pq = M.per_query([1, 2, 7])
    assert np.allclose(pq['score'], [1.0, (0 + 1 / math.log2(3)) / 2, 0.0])
    s = M.summarize([1, 2, 7])
    assert s['score'] == pytest.approx(pq['score'].mean())
    assert s['recall_at_1'] == pytest.approx(1 / 3)


def test_missing_gt_scores_zero():
    ranks = M.ranks_from_rankings([['a', 'b'], ['x']], ['b', 'zz'])
    assert ranks[0] == 2 and ranks[1] == M.MISSING
    assert M.per_query(ranks)['score'][1] == 0.0


def test_ranks_from_scores_matches_argsort_and_ties_are_pessimistic():
    rng = np.random.default_rng(0)
    s = rng.random((50, 200))
    gt = rng.integers(0, 200, 50)
    want = np.array([list(np.argsort(-row)).index(g) + 1 for row, g in zip(s, gt)])
    assert (M.ranks_from_scores(s, gt, chunk=7) == want).all()
    assert M.ranks_from_scores(np.zeros((1, 10)), [3])[0] == 10   # constant scorer = last


def test_ranks_from_scores_torch():
    torch = pytest.importorskip('torch')
    s = torch.tensor([[0.1, 0.9, 0.5], [0.3, 0.2, 0.1]])
    assert M.ranks_from_scores(s, [2, 0]).tolist() == [2, 1]


def test_random_ranking_matches_expectation():
    N, Q = 100, 200_000          # SE of R@1 ~ 2% of the mean
    rng = np.random.default_rng(1)
    ranks = rng.integers(1, N + 1, Q)          # uniform rank of the single GT
    got, exp = M.summarize(ranks), M.expected_random(N)
    for k in ('recall_at_1', 'ndcg_at_5', 'score'):
        assert got[k] == pytest.approx(exp[k], rel=0.1)


def test_topk_order():
    s = np.array([[0.1, 0.7, 0.3, 0.9]])
    assert M.topk_from_scores(s, 3).tolist() == [[3, 1, 2]]
