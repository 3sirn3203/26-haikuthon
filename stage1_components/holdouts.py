"""Deterministic evaluation vocabulary, including alternatives for retrieval foils."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
import random

from .model import edge, require


def quotas(n, weights):
    scale = sum(weights.values())
    exact = {k: n * w / scale for k, w in weights.items()}
    result = {k: int(v) for k, v in exact.items()}
    remainder = sorted(exact, key=lambda k: -(exact[k] - result[k]))
    for k in remainder[:n - sum(result.values())]:
        result[k] += 1
    return result


# Each pair makes incompatible claims about the same two visible objects.
ALTERNATIVE = {'left_of': 'right_of', 'above': 'below', 'on': 'next_to',
               'inside': 'next_to', 'holding': 'next_to', 'wearing': 'holding',
               'riding': 'next_to'}


def legal_pairs(catalog):
    candidates = defaultdict(list)
    for a in catalog.entities:
        for b in catalog.entities:
            if a == b or not catalog.compatible((a, b)):
                continue
            for p in (*ALTERNATIVE, 'next_to'):
                if catalog.legal_relation(a, p, b):
                    candidates[p].append((a, b))
    return candidates


class Holdouts:
    def __init__(self, catalog, policy, seed, candidates=None):
        self.catalog = catalog
        rng = random.Random(seed + 101)
        self.bindings = set()
        self.by_category = defaultdict(list)
        self.binding_pairs = {}
        self._choose_bindings(policy['binding_holdouts'], rng)
        self.relations = set()
        self.relation_pairs = []
        self._choose_relations(policy['relation_holdouts'], candidates or legal_pairs(catalog), rng)

    def _choose_bindings(self, size, rng):
        require(size % 2 == 0, 'Binding holdouts must form pairs')
        targets = quotas(size // 2, dict(color=12, material=6, pose=4, state=4, pattern=3, shape=3))
        catalog = self.catalog
        domains = Counter()
        for axis, amount in targets.items():
            last = None
            for i in range(amount):
                choices = []
                for cat, entity in catalog.entities.items():
                    values = catalog.attributes(cat).get(axis, ())
                    if cat in self.binding_pairs or cat == 'person' or len(values) < (2 if axis == 'state' else 3):
                        continue
                    if axis == 'state' and (cat in catalog.rules['inside_targets'] or cat == 'desk_lamp'):
                        continue
                    for pair in combinations(values, 2):
                        if axis == 'pose' and 'rideable' in entity['capabilities'] and catalog.rules['ridden_animal_pose'] in pair:
                            continue
                        # At least one other object can teach each primitive.
                        if all(any(other != cat and v in catalog.attributes(other).get(axis, ())
                                   and not self.binding(other, axis, v)
                                   and cat not in catalog.ancestors(other)
                                   for other in catalog.entities) for v in pair):
                            choices.append((cat, pair))
                require(choices, f'Insufficient binding alternatives for {axis}')
                rng.shuffle(choices)
                # Sharing palettes across some categories supports attribute swaps.
                cat, pair = min(choices, key=lambda x: (
                    int(i % 2 == 1 and set(x[1]) != set(last or ())),
                    domains[catalog.entities[x[0]]['domain']]))
                for value in pair:
                    self.bindings.add((cat, axis, value))
                    self.by_category[cat].append((axis, value))
                self.binding_pairs[cat] = (axis, pair)
                domains[catalog.entities[cat]['domain']] += 1
                last = pair
                self.binding.cache_clear()
        require(len(self.bindings) == size, 'Binding holdout cardinality')

    def _choose_relations(self, size, candidates, rng):
        used_pairs = set()
        frequency = Counter()
        paired = {'riding': 2, 'wearing': 2, 'holding': 6, 'inside': 8,
                  'on': 12, 'above': 20, 'left_of': 30}
        single = quotas(size - 2 * sum(paired.values()),
                        {'inside': 16, 'on': 36, 'holding': 8, 'next_to': 22, 'above': 120, 'left_of': 150})
        require(min(single.values()) >= 0, 'Too few relation holdouts for the contrast pairs')
        for grouped, targets in [(True, paired), (False, single)]:
            for pred, amount in targets.items():
                available = [(a, b) for a, b in candidates[pred] if frozenset((a, b)) not in used_pairs]
                for i in range(amount):
                    require(available, f'Insufficient relation holdouts for {pred}')
                    # A useful fraction must support BOTH novelty bins, including two-object queries.
                    pool = available
                    if pred == 'riding':
                        # Hold out one animal and one vehicle, leaving examples
                        # of both kinds of riding for the training gallery.
                        pool = [x for x in pool if (self.catalog.entities[x[1]]['kind'] == 'animal') == (i % 2 == 0)]
                    anchored = [x for x in pool if any(c in self.binding_pairs for c in x)]
                    if anchored and i % 4 != 3:
                        pool = anchored
                    require(pool, f'No eligible holdout pair for {pred}')
                    sample = rng.sample(pool, min(60, len(pool)))
                    a, b = min(sample, key=lambda x: (
                        frequency[x[0]] + frequency[x[1]],
                        self.catalog.contexts(x[0]) & self.catalog.contexts(x[1]) == {'studio'}))
                    self.relations.add(edge(a, pred, b))
                    if grouped:
                        other = ALTERNATIVE[pred]
                        self.relations.add(edge(a, other, b))
                        self.relation_pairs.append((a, b, pred, other))
                    used_pairs.add(frozenset((a, b)))
                    available = [x for x in available if x != (a, b) and x != (b, a)]
                    frequency.update((a, b))
        require(len(self.relations) == size, 'Relation holdout cardinality')

    @lru_cache(maxsize=None)
    def binding(self, c, a, v):
        return any((parent, a, v) in self.bindings for parent in self.catalog.ancestors(c))

    @lru_cache(maxsize=None)
    def relational(self, a, p, b):
        return any(edge(x, p, y) in self.relations for x in self.catalog.ancestors(a)
                   for y in self.catalog.ancestors(b))

    def kind(self, scene):
        bindings, relations = self.catalog.expanded_atoms(scene)
        has_binding = bool(bindings & self.bindings)
        has_relation = bool(relations & self.relations)
        if has_binding and has_relation:
            return 'both'
        if has_binding:
            return 'binding'
        return 'relation' if has_relation else 'observed'
