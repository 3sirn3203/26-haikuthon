"""Deterministic group-size allocation and explicit group membership."""
from collections import defaultdict
import math

from .holdouts import quotas
from .model import context_key, require


GROUP_SIZES = (4, 8, 16)


def size_weights(policy, split=None):
    weights = policy.get('group_size_weights', {8: 1})
    if isinstance(weights, dict) and any(k in weights for k in ('train', 'test')):
        require(set(weights) == {'train', 'test'}, 'group_size_weights requires both train and test')
        require(split in ('train', 'test'), 'Specify split for split-specific group_size_weights')
        weights = weights[split]
    require(isinstance(weights, dict) and bool(weights), 'group_size_weights must be a nonempty mapping')
    require(all(type(k) is int and k in GROUP_SIZES for k in weights),
            'group_size_weights supports only integer keys 4, 8, 16')
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in weights.values()),
            'group_size_weights must be finite nonnegative numbers')
    require(math.isfinite(sum(weights.values())) and sum(weights.values()) > 0,
            'group_size_weights must have a positive finite total weight')
    return {k: weights.get(k, 0) / sum(weights.values()) for k in GROUP_SIZES}


def allocate_sizes(amount, weights):
    """Closest group-count proportions subject to an exact number of rows.

    Zero-weight sizes are never used. Small bins are rounded independently, so
    requested proportions are targets rather than exact fractional group counts.
    """
    require(type(amount) is int and amount >= 0, 'Contrast row counts must be nonnegative integers')
    if not amount:
        return {k: 0 for k in GROUP_SIZES}
    best = None
    for sixteen in range(amount // 16 + 1) if weights[16] else (0,):
        remaining = amount - 16 * sixteen
        for eight in range(remaining // 8 + 1) if weights[8] else (0,):
            remainder = remaining - 8 * eight
            if remainder % 4 or (remainder and not weights[4]):
                continue
            counts = (remainder // 4, eight, sixteen)
            groups = sum(counts)
            error = sum((n / groups - weights[k]) ** 2 for k, n in zip(GROUP_SIZES, counts))
            score = (error, abs(groups - amount / sum(k * weights[k] for k in GROUP_SIZES)), counts)
            if best is None or score < best:
                best = score
    require(best is not None,
            f'Cannot allocate {amount} contrast rows using enabled group sizes '
            f'{[k for k in GROUP_SIZES if weights[k]]}; adjust row counts or group_size_weights')
    return dict(zip(GROUP_SIZES, best[2]))


def contrast_plan(policy, split):
    weights = size_weights(policy, split)
    amounts = (policy['test_contrast_counts'] if split == 'test' else
               {n: count - policy.get('train_baseline_counts', {}).get(n, 0)
                for n, count in policy['train_counts'].items()})
    result = []
    for n, amount in amounts.items():
        kinds = quotas(amount, policy['novelty_weights']) if split == 'test' else {'observed': amount}
        for kind, rows in kinds.items():
            for size, count in allocate_sizes(rows, weights).items():
                result.extend([(n, kind, size)] * count)
    return result


def cluster_limit(policy):
    """A missing/null entry retains independent groups for legacy policies."""
    config = policy.get('test_clusters')
    if config is None:
        return None
    require(isinstance(config, dict) and set(config) == {'max_size'},
            'test_clusters requires only max_size (or use null to disable)')
    size = config['max_size']
    largest = max(k for k, v in size_weights(policy, 'test').items() if v)
    require(type(size) is int and size % 4 == 0 and size >= 2 * largest,
            'test_clusters.max_size must be a multiple of 4 and fit at least two largest small groups')
    return size


def cluster_plan(policy, schedule=None):
    """Pack whole small groups within each count/novelty bin, never exceeding the cap.

    Partial final clusters preserve exact row/novelty budgets. Rebalance a lone
    final group so every cluster contains at least two independently varied groups.
    """
    limit = cluster_limit(policy)
    if limit is None:
        return []
    bins = defaultdict(list)
    for n, kind, size in contrast_plan(policy, 'test') if schedule is None else schedule:
        bins[n, kind].append(size)
    result = []
    for (n, kind), sizes in bins.items():
        packs = [[] for _ in range(math.ceil(sum(sizes) / limit))]
        for size in sorted(sizes, reverse=True):
            fitting = [p for p in packs if sum(p) + size <= limit]
            if fitting:
                min(fitting, key=sum).append(size)
            else:
                packs.append([size])
        for pack in packs:
            if len(pack) > 1:
                continue
            donors = [(s, donor) for donor in packs if len(donor) > 2 for s in donor
                      if donor is not pack and sum(pack) + s <= limit]
            require(donors, f'Test {n}/{kind} cannot form clusters with at least two small groups; increase row quotas')
            size, donor = min(donors, key=lambda item: item[0])
            donor.remove(size)
            pack.append(size)
        result.extend((n, kind, tuple(pack)) for pack in packs)
    return result


def inventory_limit(policy, split):
    cap = policy['ranking']['max_inventory_rows']
    if isinstance(cap, dict):
        require(set(cap) == {'train', 'test'}, 'max_inventory_rows requires train and test')
        cap = cap[split]
    require(type(cap) is int and cap > 0, 'max_inventory_rows must be a positive integer')
    return cap


def repeated_indices(schedule, fraction, rng):
    """Choose whole eligible groups while preserving the repeated-category row quota."""
    target = round(sum(size for _, _, size in schedule) * fraction)
    candidates = [i for i, (n, _, _) in enumerate(schedule) if n >= 3]
    rng.shuffle(candidates)
    paths = {0: None}
    for i in candidates:
        if target in paths:
            break
        size = schedule[i][2]
        for previous in sorted(paths, reverse=True):
            value = previous + size
            if value <= target and value not in paths:
                paths[value] = previous, i
    require(target in paths, f'Cannot allocate {target} repeated-category rows with the configured group sizes')
    result = set()
    while target:
        target, i = paths[target]
        result.add(i)
    return result


def partition(rows, catalog=None):
    """Use explicit IDs when available; retain support for legacy component files."""
    explicit = ['group_id' in row for row in rows]
    require(not any(explicit) or all(explicit), 'Mixed missing/present group_id fields')
    explicit = any(explicit)
    buckets = defaultdict(list)
    for row in rows:
        context = context_key(row, catalog)
        key = row['group_id'] if explicit else context
        if explicit:
            require(type(key) is int and key >= 0, 'group_id must be a nonnegative integer')
        buckets[key].append(row)
    easy, groups = [], []
    for items in buckets.values():
        require(len({context_key(row, catalog) for row in items}) == 1, 'group_id combines different contexts')
        if len(items) == 1:
            easy.extend(items)
        else:
            groups.append(items)
    return easy, groups


def assign_group_ids(rows, catalog, start=0):
    """Renumber explicit generation units in first-appearance order; infer legacy units."""
    buckets = defaultdict(list)
    explicit = ['group_id' in row for row in rows]
    require(not any(explicit) or all(explicit), 'Mixed missing/present group_id fields')
    explicit = all(explicit)
    for row in rows:
        buckets[row['group_id'] if explicit else context_key(row, catalog)].append(row)
    for identifier, group in enumerate(buckets.values(), start):
        for row in group:
            row['group_id'] = identifier
    return start + len(buckets)


def assign_cluster_ids(rows):
    ids = {}
    for row in rows:
        if row.get('cluster_id') is not None:
            old = row['cluster_id']
            if old not in ids:
                ids[old] = len(ids)
            row['cluster_id'] = ids[old]
