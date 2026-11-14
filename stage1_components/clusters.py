"""Test clusters reuse object grids under distinct small-group visual contexts."""
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations, product
import math

from .groups import cluster_limit, cluster_plan, repeated_indices
from .model import canonical, inventory, normalized_edges, require, total
from .ranking import check_group, visual_context
from .scenes import compose


def nested_grids(master, sizes, catalog, holds):
    """Find nested complete grids, retaining the existing counterfactual checks.

    Slice on a single attribute, count or relation state. Equal-sized siblings
    reuse exactly the same grid; smaller siblings take a checked subset of it.
    """
    result = {len(master): master}
    current = master
    for size in sorted(set(sizes) - {len(master)}, reverse=True):
        visited = set()

        def search(rows):
            key = frozenset(canonical(s) for s in rows)
            if key in visited:
                return None
            visited.add(key)
            if len(rows) == size:
                try:
                    check_group(rows, catalog, holds)
                    return rows
                except ValueError:
                    return None
            columns = []
            for slot in range(len(rows[0]['entities'])):
                entities = [s['entities'][slot] for s in rows]
                for axis in sorted(set().union(*(e['attributes'] for e in entities))):
                    columns.append([e['attributes'].get(axis) for e in entities])
                columns.append([e['count'] for e in entities])
            columns.append([tuple(sorted(normalized_edges(s))) for s in rows])
            for column in columns:
                levels = list(dict.fromkeys(column))
                # A four-level attribute can contribute two levels to a smaller grid.
                for width in (1, 2) if len(levels) == 4 else (1,):
                    for values in combinations(levels, width):
                        subset = [row for row, v in zip(rows, column) if v in values]
                        if size <= len(subset) < len(rows) and len(subset) % size == 0:
                            found = search(subset)
                            if found is not None:
                                return found
            return None

        current = search(current)
        if current is None:
            return None
        result[size] = current
    return result


def context_options(g, master):
    """Only contexts that can faithfully render every variant of the template."""
    cats = inventory(master[0])
    places = set.intersection(*(set(g.catalog.contexts(c)) for c in cats))
    predicates = {r['predicate'] for s in master for r in s['relations']}
    if 'riding' in predicates:
        places.intersection_update(g.catalog.rules['riding_backgrounds'])
    styles = [style for style, weight in g.policy['style_weights'].items() if weight > 0 and
              all(value in g.catalog.attributes(e['category'], style).get(axis, ())
                  for s in master for e in s['entities'] for axis, value in e['attributes'].items())]
    views = ('overhead',) if 'inside' in predicates else g.catalog.active['viewpoints']
    return [(place, style, view, light) for place, style, view in product(sorted(places), styles, views)
            for light in g.lighting_weights if light in g.catalog.lighting_options(place)]


def with_context(g, rows, options, used):
    """Prefer changes on each available context axis; never reuse a context tuple."""
    available = [context for context in options if context not in used]
    if not available:
        return None
    novelty = {context: sum(all(context[i] != old[i] for old in used) for i in range(4))
               for context in available}
    best = max(novelty.values())
    available = [context for context in available if novelty[context] == best]
    styles = dict(zip(g.policy['style_weights'], g.weights(g.policy['style_weights'], g.style_freq)))
    lights = dict(zip(g.lighting_weights, g.weights(g.lighting_weights, g.lighting_freq)))
    weights = [styles[style] * lights[light] * (.3 if place == 'studio' else 1)
               for place, style, view, light in available]
    place, style, view, light = g.rng.choices(available, weights)[0]
    result = deepcopy(rows)
    for row in result:
        row['background'] = {'location': place}
        row['rendering'] = {'style': style, 'viewpoint': view, 'lighting': light}
    return result


def build_cluster(g, n, kind, sizes, repeated, identifier, focus=None, variation=None):
    """Attempt one template and all its rendered sibling groups transactionally."""
    g.group_size = sum(sizes)
    layout = g.layout(n, kind, repeated, focus)
    if layout is None:
        g.rejects['cluster layout unavailable'] += 1
        return None
    g.group_size = max(sizes)
    master = compose(g, *layout, n, kind, repeated, focus, max(sizes), variation=variation)
    if not master:
        g.rejects['cluster template unavailable'] += 1
        return None
    try:
        check_group(master, g.catalog, g.holdouts)
    except ValueError as exc:
        g.rejects[f'cluster template: {exc}'] += 1
        return None
    grids = nested_grids(master, sizes, g.catalog, g.holdouts)
    if grids is None:
        g.rejects['cluster nested grids unavailable'] += 1
        return None
    options = context_options(g, master)
    if len(options) < len(sizes):
        g.rejects['cluster context options insufficient'] += 1
        return None
    accepted, used = [], set()
    for size in sorted(sizes, reverse=True):
        g.group_size = size
        selected = None
        for _ in range(min(40, len(options))):
            group = with_context(g, grids[size], options, used)
            if group is None:
                break
            # Failed contexts are also excluded for this attempt.
            used.add(visual_context(group[0]))
            for row in group:
                row['cluster_id'] = identifier
            if g.accept(group, kind, True):
                accepted.append(group)
                selected = group
                break
        if selected is None:
            for previous in reversed(accepted):
                g.record(previous, True, add=False)
            return None
    return accepted


def generate_clusters(g):
    schedule = list(g.clusters)
    g.rng.shuffle(schedule)
    schedule.sort(key=lambda item: item[0])
    repeated = repeated_indices([(n, kind, sum(sizes)) for n, kind, sizes in schedule],
                                g.policy['repeated_category_fraction'], g.rng)
    rows = []
    for index, (n, kind, sizes) in enumerate(schedule):
        previous = g.rejects.copy()
        for attempt in range(g.policy['attempt_limit']):
            missing = [c for c in g.cats if g.freq[c] < g.low]
            focus = None
            if missing and index not in repeated and attempt % 6 != 5:
                shortlist = g.rng.sample(missing, min(5, len(missing)))
                focus = min(shortlist, key=lambda c: (g.freq[c], len(g.catalog.contexts(c))))
            groups = build_cluster(g, n, kind, sizes, index in repeated, index, focus)
            if groups is not None:
                rows.extend(s for group in groups for s in group)
                break
        else:
            raise ValueError(f'Cannot allocate test cluster {index} ({n}/{kind}, sizes={sizes}): '
                             f'{(g.rejects - previous).most_common(8)}')
        if (index + 1) % 10 == 0 or index + 1 == len(schedule):
            print(f'test: {index + 1}/{len(schedule)} clusters; {len(rows)} contrast scenes; '
                  f'{len(g.inventory_freq)} inventories', flush=True)
    repair_clusters(g, rows)
    return rows


def replace_cluster(g, donors, predicate, focus=None, variation=None):
    """Replace the entire template so sibling grids cannot drift independently."""
    first = donors[0][0]
    sizes = [len(group) for group in donors]
    required = {c: min(g.low, g.freq[c]) for group in donors for s in group for c in inventory(s)}
    for group in donors:
        g.record(group, True, add=False)
    accepted = False
    try:
        for _ in range(96):
            candidate = build_cluster(g, total(first), g.holdouts.kind(first), sizes,
                                      len(inventory(first)) < len(first['entities']),
                                      first['cluster_id'], focus, variation)
            if candidate is None:
                continue
            stats = Counter()
            for group in candidate:
                stats.update(check_group(group, g.catalog, g.holdouts))
            if all(g.freq[c] >= minimum for c, minimum in required.items()) and predicate(candidate, stats):
                accepted = True
                return candidate
            for group in reversed(candidate):
                g.record(group, True, add=False)
    finally:
        if not accepted:
            for group in donors:
                g.record(group, True, add=True)
    return None


def repair_clusters(g, rows):
    slots = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows):
        slots[row['cluster_id']][row['group_id']].append(i)
    positions = [sorted(groups.values(), key=len, reverse=True) for groups in slots.values()]
    metrics = ('count_variation', 'nonspatial_relation_variation', 'attribute_swap')
    count = sum(len(groups) for groups in positions)
    targets = {m: math.ceil(count * g.rules['minimum_' + m]) for m in metrics}
    simple = 'one_object_attribute_and_direction'
    maximum = math.floor(count * g.rules['max_' + simple])

    def cluster_stats(groups):
        stats = Counter()
        for pp in groups:
            stats.update(check_group([rows[i] for i in pp], g.catalog, g.holdouts))
        return stats

    stats = [cluster_stats(groups) for groups in positions]
    totals = sum(stats, Counter())
    while True:
        missing = [c for c in g.cats if g.freq[c] < g.low]
        deficits = {m: target - totals[m] for m, target in targets.items() if totals[m] < target}
        if totals[simple] > maximum:
            deficits[simple] = totals[simple] - maximum
        if not missing and not deficits:
            return
        focus = min(missing, key=lambda c: (g.freq[c], len(g.catalog.contexts(c)))) if missing else None
        target = max(deficits, key=deficits.get) if not focus else None
        candidates = list(range(len(positions)))
        g.rng.shuffle(candidates)
        for index in candidates:
            donors = [[rows[i] for i in pp] for pp in positions[index]]
            if focus and focus in inventory(donors[0][0]):
                continue
            if target == 'count_variation' and total(donors[0][0]) == 2:
                continue
            old = stats[index]
            if target and ((target == simple and not old[simple]) or
                           (target != simple and old[target] == len(donors))):
                continue

            def improves(groups, new):
                if focus and focus not in inventory(groups[0][0]):
                    return False
                if target and not ((new[target] < old[target]) if target == simple else (new[target] > old[target])):
                    return False
                return (all(totals[m] - old[m] + new[m] >= min(targets[m], totals[m]) for m in metrics)
                        and totals[simple] - old[simple] + new[simple] <= max(maximum, totals[simple]))

            candidate = replace_cluster(g, donors, improves, focus,
                                        'count_variation' if target == simple else target)
            if candidate is None:
                continue
            for pp, group in zip(positions[index], candidate):
                for i, row in zip(pp, group):
                    rows[i] = row
            new = cluster_stats(positions[index])
            totals.subtract(old)
            totals.update(new)
            stats[index] = new
            print(f'test: supplemented {focus or target} by replacing cluster {candidate[0][0]["cluster_id"]}', flush=True)
            break
        else:
            raise ValueError(f'Cannot repair test {focus or target}: missing={missing}, variations={dict(totals)}')


def check_clusters(rows, groups, policy, catalog, holds):
    """Reconstruct context separation and nested content grids from saved data."""
    limit = cluster_limit(policy)
    if limit is None:
        require(all(s.get('cluster_id') is None for s in rows), 'Clusters are disabled by policy')
        return {}
    clustered = defaultdict(list)
    for group in groups:
        ids = {s.get('cluster_id') for s in group}
        require(len(ids) == 1, 'Small group crosses cluster_ids')
        identifier = ids.pop()
        require(identifier is not None, 'Test contrast group is missing cluster_id')
        clustered[identifier].append(group)
    hard_ids = {s['group_id'] for g in groups for s in g}
    require(all(s.get('cluster_id') is None for s in rows if s.get('group_id') not in hard_ids),
            'Baseline scene cannot belong to a cluster')
    require(set(clustered) == set(range(len(clustered))), 'cluster_ids must be consecutive starting at 0')
    expected = Counter((n, kind, tuple(sorted(sizes))) for n, kind, sizes in cluster_plan(policy))
    actual = Counter()
    for identifier, members in clustered.items():
        scenes = [s for group in members for s in group]
        require(len(members) >= 2 and len(scenes) <= limit, 'Invalid cluster size')
        contexts, grids = set(), {}
        for group in members:
            context = {visual_context(s) for s in group}
            require(len(context) == 1, 'Small group rendering/background differs')
            context = context.pop()
            require(context not in contexts, 'Sibling groups must have distinct visual contexts')
            contexts.add(context)
            core = frozenset(canonical(s) for s in group)
            require(len(group) not in grids or grids[len(group)] == core,
                    'Equal-sized siblings must share the same object variants')
            grids[len(group)] = core
        sizes = sorted(grids)
        require(all(grids[a] <= grids[b] for a, b in zip(sizes, sizes[1:])),
                'Smaller sibling grid must be a subset of the larger grid')
        require(len({tuple(sorted(e['category'] for e in s['entities'])) for s in scenes}) == 1,
                'Object categories must stay fixed throughout a cluster')
        require(len({total(s) for s in scenes}) == 1, 'Cluster total object count differs')
        require(len({holds.kind(s) for s in scenes}) == 1, 'Cluster novelty types differ')
        actual[total(scenes[0]), holds.kind(scenes[0]), tuple(sorted(map(len, members)))] += 1
    require(actual == expected, 'Cluster allocation differs from configured plan')
    return {'clusters': len(clustered),
            'cluster_sizes': dict(sorted(Counter(sum(map(len, gs)) for gs in clustered.values()).items())),
            'small_groups_per_cluster': dict(sorted(Counter(map(len, clustered.values())).items()))}
