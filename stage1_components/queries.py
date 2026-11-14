"""Select evaluation queries without removing candidates or changing group membership."""
from collections import defaultdict
import math
import random

from .groups import partition
from .holdouts import quotas
from .model import require, total


def query_fraction(policy):
    fraction = policy.get('test_query_fraction')
    if fraction is None:
        return None
    require(type(fraction) in (int, float) and math.isfinite(fraction) and 0 < fraction <= 1,
            'test_query_fraction must be a finite number in (0, 1]')
    return fraction


def query_count(amount, fraction):
    exact = amount * fraction
    require(math.isclose(exact, round(exact), rel_tol=0, abs_tol=1e-9),
            f'test_query_fraction must select an integer number of queries from {amount} rows')
    return round(exact)


def select_query_ids(rows, policy, catalog, holds, seed):
    """Sample each small group, then stratify singleton baselines by count/novelty."""
    fraction = query_fraction(policy)
    if fraction is None:
        return None
    target = query_count(len(rows), fraction)
    identifiers = [row['prompt_id'] for row in rows]
    require(len(set(identifiers)) == len(rows), 'Query sampling requires unique prompt_ids')
    rng = random.Random(f'{seed}/test-query-selection')
    easy, groups = partition(rows, catalog)
    selected = set()
    for group in sorted(groups, key=lambda items: min(int(s['prompt_id']) for s in items)):
        ids = sorted((s['prompt_id'] for s in group), key=int)
        selected.update(rng.sample(ids, query_count(len(ids), fraction)))
    buckets = defaultdict(list)
    for row in easy:
        buckets[total(row), holds.kind(row)].append(row['prompt_id'])
    remaining = target - len(selected)
    if buckets:
        weights = {key: len(buckets[key]) for key in sorted(buckets)}
        counts = quotas(remaining, weights)
        for key, amount in counts.items():
            selected.update(rng.sample(sorted(buckets[key], key=int), amount))
    require(len(selected) == target, 'Query selection does not match the requested total')
    return selected


def mark_queries(rows, policy, catalog, holds, seed):
    selected = select_query_ids(rows, policy, catalog, holds, seed)
    if selected is not None:
        for row in rows:
            row['is_query'] = row['prompt_id'] in selected


def validate_queries(rows, policy, catalog, holds, seed):
    selected = select_query_ids(rows, policy, catalog, holds, seed)
    if selected is None:
        require(not any('is_query' in row for row in rows),
                'is_query metadata requires test_query_fraction in the policy')
        return {}
    require(all(type(row.get('is_query')) is bool for row in rows),
            'Every test candidate must have a boolean is_query')
    actual = {row['prompt_id'] for row in rows if row['is_query']}
    require(actual == selected, 'Query selection does not match the configured seed and per-group sampling')
    return {'image_candidates': len(rows), 'queries': len(selected), 'candidate_only': len(rows) - len(selected)}
