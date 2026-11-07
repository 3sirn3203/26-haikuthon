"""Retrieval checks on visible facts. All keys ignore prompt IDs and array order."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations

from .model import canonical, context_key, default_catalog, edge, normalized_edges, require

SPATIAL = {'left_of', 'right_of', 'above', 'below'}
CONTACT = {'inside', 'on', 'holding', 'wearing', 'riding'}
REVERSE = {'left_of': 'right_of', 'above': 'below', 'next_to': 'next_to',
           **{p: 'inverse_' + p for p in CONTACT}}


def masked_key(scene, holds, axis):
    entities = []
    for e in scene['entities']:
        attrs = {a: v for a, v in e['attributes'].items()
                 if axis != 'binding' or not holds.binding(e['category'], a, v)}
        entities.append(dict(e, attributes=attrs))
    relations = [r for r in scene['relations'] if axis != 'relation' or not holds.relational(
        entities[r['subject']]['category'], r['predicate'], entities[r['object']]['category'])]
    return canonical(dict(scene, entities=entities, relations=relations))


def visual_context(s):
    return (s['background']['location'], *sorted(s['rendering'].items()))


def primitive_bag(s):
    """The words and numbers before assigning attributes, counts or relation roles."""
    es = s['entities']
    return (tuple(sorted(e['category'] for e in es)), tuple(sorted(e['count'] for e in es)),
            tuple(sorted(pair for e in es for pair in e['attributes'].items())),
            tuple(sorted(r['predicate'] for r in s['relations'])))


def gallery_signature(s, catalog):
    counts = Counter()
    for e in s['entities']:
        counts[e['category']] += e['count']
    # For hierarchical terms, defer exact multiplicities to the instance matcher.
    return frozenset((catalog.ancestors(c)[-1], 0 if c in catalog.hierarchical else n)
                     for c, n in counts.items())


@lru_cache(maxsize=512)
def incompatible(first, second):
    contact = CONTACT | {'inverse_' + p for p in CONTACT}
    outside = SPATIAL | {'next_to'}
    if first & contact and second & outside or second & contact and first & outside:
        return True
    opposites = [('left_of', 'right_of'), ('above', 'below'), ('holding', 'wearing'),
                 ('inverse_holding', 'inverse_wearing'), ('on', 'inside'), ('inverse_on', 'inverse_inside')]
    opposites += [(p, REVERSE[p]) for p in CONTACT]
    return any((p in first and q in second) or (q in first and p in second) for p, q in opposites)


class Description:
    """A possible image consistent with the explicit group counts, attributes and relations."""
    def __init__(self, scene, catalog=None):
        self.catalog = catalog or default_catalog()
        self.counts = Counter()
        self.entities = scene['entities']
        self.instances = [i for i, e in enumerate(self.entities) for _ in range(e['count'])]
        for e in self.entities:
            self.counts[e['category']] += e['count']
        self.roles = defaultdict(set)
        for a, p, b in normalized_edges(scene):
            self.roles[a, b].add(p)
            self.roles[b, a].add(REVERSE[p])
        self.roles = {k: frozenset(v) for k, v in self.roles.items()}

    def could_match(self, image):
        # Extra categories in the candidate do not by themselves contradict a query.
        for c, n in self.counts.items():
            definite = sum(k for other, k in image.counts.items() if c in self.catalog.ancestors(other))
            possible = sum(k for other, k in image.counts.items() if other in self.catalog.ancestors(c)[1:])
            if not definite <= n <= definite + possible:
                return False
        options = []
        for i in self.instances:
            e = self.entities[i]
            candidates = [j for j, f in enumerate(image.entities)
                          if (e['category'] in self.catalog.ancestors(f['category']) or
                              f['category'] in self.catalog.ancestors(e['category']))
                          and all(a not in f['attributes'] or f['attributes'][a] == v
                                  for a, v in e['attributes'].items())]
            if not candidates:
                return False
            options.append(candidates)
        remaining = [e['count'] for e in image.entities]
        assigned = []
        def match(k):
            if k == len(options):
                return True
            for b in options[k]:
                if not remaining[b]:
                    continue
                if any(incompatible(self.roles.get((self.instances[k], self.instances[j]), frozenset()),
                                    image.roles.get((b, target), frozenset()))
                       for j, target in enumerate(assigned)):
                    continue
                remaining[b] -= 1
                assigned.append(b)
                if match(k + 1):
                    return True
                assigned.pop()
                remaining[b] += 1
            return False
        return match(0)


def aligned(group):
    """Align semantic slots using category and a permanent visible marker for duplicates."""
    counts = [Counter(e['category'] for e in s['entities']) for s in group]
    common = counts[0].copy()
    for other in counts[1:]:
        common &= other
    size = len(group[0]['entities'])
    require(all(len(s['entities']) == size for s in group) and sum(common.values()) >= size - 1,
            'Neighbors replace more than one object category')
    markers = {}
    for c, n in common.items():
        if n == 1:
            continue
        by_row = [[e for e in s['entities'] if e['category'] == c] for s in group]
        for axis in sorted(by_row[0][0]['attributes']):
            palettes = [frozenset(e['attributes'].get(axis) for e in es) for es in by_row]
            if len(set(palettes)) == 1 and len(palettes[0]) == n and None not in palettes[0]:
                markers[c] = axis
                break
        require(c in markers, 'Repeated groups need stable, visible identifying attributes')
    result = []
    for s in group:
        def slot(i):
            e = s['entities'][i]
            c = e['category']
            return (0, c, e['attributes'].get(markers.get(c), '')) if c in common else (1, '', '')
        order = sorted(range(size), key=slot)
        lookup = {old: new for new, old in enumerate(order)}
        edges = frozenset(edge(lookup[a], p, lookup[b]) for a, p, b in normalized_edges(s))
        result.append(([s['entities'][i] for i in order], edges))
    return result


def group_features(group, records):
    cols = []
    changed_bindings = []
    categories = []
    counts = []
    per_slot = []
    for i in range(len(records[0][0])):
        entities = [es[i] for es, _ in records]
        cats = tuple(e['category'] for e in entities)
        ns = tuple(e['count'] for e in entities)
        categories.append(cats)
        counts.append(ns)
        cols.extend((cats, ns))
        fields = {}
        for a in sorted(set().union(*(e['attributes'] for e in entities))):
            values = tuple(e['attributes'].get(a) for e in entities)
            fields[a] = values
            cols.append(values)
        per_slot.append(fields)
        changed_bindings.append(any(len(set(v)) > 1 for v in fields.values()))
    edges = [rr for _, rr in records]
    for r in set.union(*(set(rr) for rr in edges)):
        cols.append(tuple(r in rr for rr in edges))
    # A four-level binding may be formed by two binary attributes.
    cols.extend(tuple(canonical(s, a) for s in group) for a in ('binding', 'count', 'relation'))
    partitions = {tuple(tuple(col).index(v) for v in col) for col in cols}
    twos = [v for v in partitions if sorted(Counter(v).values()) == [4, 4]]
    fours = [v for v in partitions if sorted(Counter(v).values()) == [2, 2, 2, 2]]
    factorial = (any(len(set(zip(a, b))) == 8 for a in twos for b in fours) or
                 any(len(set(zip(a, b, c))) == 8 for a, b, c in combinations(twos, 3)))
    require(factorial, 'Eight scenes must form a complete 2x4 or 2x2x2 crossing')
    category_var = any(len(set(col)) > 1 for col in categories)
    count_var = any(len(set(col)) > 1 for col in counts)
    contact_var = len({frozenset(r for r in rr if r[1] in CONTACT) for rr in edges}) > 1
    swap = False
    for a, b in combinations(per_slot, 2):
        for axis in a.keys() & b.keys():
            pairs = set(zip(a[axis], b[axis]))
            if any(x is not None and y is not None and x != y and (y, x) in pairs for x, y in pairs):
                swap = True
    return {'category_variation': category_var, 'count_variation': count_var,
            'nonspatial_relation_variation': contact_var, 'attribute_swap': swap,
            'one_object_attribute_and_direction': not (category_var or count_var or contact_var)
                                                 and sum(changed_bindings) <= 1}


def check_group(group, catalog=None, holds=None):
    require(len(group) == 8 and len({context_key(s, catalog) for s in group}) == 1,
            'Expected eight neighbors with the same context and total')
    require(len({canonical(s) for s in group}) == 8, 'Duplicate contrast scene')
    for axis in ('category', 'count', 'binding', 'relation'):
        require(min(Counter(canonical(s, axis) for s in group).values()) >= 2,
                f'Single-axis identification: {axis}')
    records = aligned(group)
    stats = group_features(group, records)
    axes = [(tuple(e['category'] for e in es), tuple(e['count'] for e in es),
             tuple(e['attributes'] for e in es), rs) for es, rs in records]
    differences = [0] * len(group)
    for i, j in combinations(range(8), 2):
        if sum(a != b for a, b in zip(axes[i], axes[j])) >= 2:
            differences[i] += 1
            differences[j] += 1
    require(min(differences) >= 3, 'Fewer than three neighbors differ in two core axes')
    if holds:
        kinds = {holds.kind(s) for s in group}
        require(len(kinds) == 1 and 'observed' not in kinds, 'Mixed or missing novelty in a test group')
        kind = kinds.pop()
        for axis in ('binding', 'relation'):
            if kind in (axis, 'both'):
                multiplicity = Counter(masked_key(s, holds, axis) for s in group)
                require(min(multiplicity.values()) >= 2, f'Masking held-out {axis} still identifies a query')
    require(min(Counter(primitive_bag(s) for s in group).values()) >= 2,
            'Unbound vocabulary identifies a query')
    return stats


def check_gallery(rows, catalog=None):
    """No other same-context candidate may satisfy all explicit facts of a query."""
    catalog = catalog or default_catalog()
    by_render = defaultdict(lambda: defaultdict(list))
    for s in rows:
        signature = gallery_signature(s, catalog)
        by_render[visual_context(s)][signature].append((s['prompt_id'], Description(s, catalog)))
    for inventories in by_render.values():
        for counts, queries in inventories.items():
            candidates = [item for other, items in inventories.items() if counts <= other for item in items]
            for query_id, query in queries:
                for candidate_id, candidate in candidates:
                    if query_id != candidate_id:
                        require(not query.could_match(candidate), f'Query {query_id} also matches candidate {candidate_id}')
