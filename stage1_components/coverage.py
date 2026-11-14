"""Train coverage measured in independent scenes and explicit relation partners."""
from collections import Counter, defaultdict
from copy import copy
from itertools import chain
import random

from .holdouts import ALTERNATIVE
from .model import context_key, family, inventory, require, total
from .scenes import compose


def category_pairs(scene):
    cats = [e['category'] for e in scene['entities']]
    return {tuple(sorted((cats[r['subject']], cats[r['object']]))) for r in scene['relations']
            if cats[r['subject']] != cats[r['object']]}


def training_exposure(units):
    counts, pair_counts = Counter(), Counter()
    partners = defaultdict(set)
    for unit in units:
        counts.update(set().union(*(inventory(s) for s in unit)))
        pairs = set().union(*(category_pairs(s) for s in unit))
        pair_counts.update(pairs)
        for a, b in pairs:
            partners[a].add(b)
            partners[b].add(a)
    return counts, partners, pair_counts


def check_shared_categories(train, test, active):
    require(not (test - train), f'Test categories absent from train: {sorted(test - train)}')
    require(not (train - test), f'Train categories absent from test: {sorted(train - test)}')
    require(not (active - train), f'Active categories missing from both splits: {sorted(active - train)}')
    require(not (train - active), f'Categories outside the active catalog: {sorted(train - active)}')


def unit_atoms(catalog, rows):
    facts = set()
    for s in rows:
        bindings, relations = catalog.atoms(s)
        facts.update(('binding', b) for b in bindings)
        facts.update(('relation', r) for r in relations)
    return facts


def primitives(facts):
    return {(kind, axis, value) if kind == 'binding' else (kind, axis)
            for kind, (_, axis, value) in facts}


def replacement(g, a, b, donor):
    n = total(donor[0])
    hard = len(donor) > 1
    preferred = family(donor[0]['relations'][0]['predicate'])
    choices = []
    for x, y in ((a, b), (b, a)):
        predicates = ALTERNATIVE if hard else g.catalog.active['relations']
        for p in predicates:
            checks = (p, ALTERNATIVE[p]) if hard else (p,)
            if all(g.catalog.legal_relation(x, pred, y) and g.common_relation(x, pred, y, False) for pred in checks):
                choices.append((x, p, y))
    if not choices:
        return None
    matched = [v for v in choices if family(v[1]) == preferred]
    x, p, y = g.rng.choice(matched or choices)
    if hard:
        edges = (((0, p, 1),), ((0, ALTERNATIVE[p], 1),))
        return compose(g, (x, y), edges, n, 'observed', False, group_size=len(donor))
    row = g.baseline_scene((x, y), p, n, 'observed')
    return [row] if row else None


def fit_replacement(g, focus, partners, donor):
    g.group_size = len(donor)
    for partner in partners:
        if g.freq[partner] + len(donor) > g.max_count(partner):
            continue
        for _ in range(4):
            candidate = replacement(g, focus, partner, donor)
            if candidate and g.accept(candidate, 'observed', len(donor) > 1):
                return candidate
    return None


def diversify_train(generator, rows):
    """Rebalance surplus train slots, preferring slots with redundant supervision.

    The evaluation bank is the intersection of witnessed facts before and after
    balancing. A separate random stream keeps the evaluation stable when that
    bank is preserved. No test scenes or output files are inputs to this step.
    """
    g = copy(generator)
    g.rng = random.Random(f'{g.seed}/train-diversity')
    policy = g.policy['train_diversity']
    min_units, min_partners = policy['minimum_units'], policy['minimum_relation_partners']
    slots = defaultdict(list)
    for i, s in enumerate(rows):
        slots[s.get('group_id', context_key(s, g.catalog))].append(i)
    indices = list(slots.values())
    units = [[rows[i] for i in group] for group in indices]
    facts = [unit_atoms(g.catalog, unit) for unit in units]
    witnesses = Counter(chain.from_iterable(facts))
    initial = set(witnesses)
    primitive_witnesses = Counter(p for f in facts for p in primitives(f))
    donors = [i for i, unit in enumerate(units)
              if all(len(inventory(s)) == len(s['entities']) for s in unit)]
    g.rng.shuffle(donors)
    donors.sort(key=lambda i: len(units[i]))

    while True:
        counts, partners, pair_counts = training_exposure(units)
        def need(c):
            return max(0, min_units - counts[c]) + max(0, min_partners - len(partners[c]))
        missing = [c for c in g.cats if need(c)]
        if not missing:
            retained = {entry for entry in initial if witnesses[entry] > 0}
            return ({atom for kind, atom in retained if kind == 'binding'},
                    {atom for kind, atom in retained if kind == 'relation'})
        focus = max(missing, key=lambda c: (need(c), -len(g.catalog.contexts(c))))
        others = [c for c in g.cats if c != focus and c not in partners[focus]
                  and g.catalog.compatible((focus, c))]
        g.rng.shuffle(others)
        others.sort(key=need, reverse=True)
        for idx in sorted(donors, key=lambda i: sum(witnesses[f] == 1 for f in facts[i])):
            donor = units[idx]
            if any(primitive_witnesses[p] == 1 for p in primitives(facts[idx])):
                continue
            losses = Counter(c for s in donor for c in inventory(s))
            if any(counts[c] <= min_units or g.freq[c] - n < g.low for c, n in losses.items()):
                continue
            pairs = set().union(*(category_pairs(s) for s in donor))
            lost = Counter(c for pair in pairs if pair_counts[pair] == 1 for c in pair)
            if any(len(partners[c]) - n < min_partners for c, n in lost.items()):
                continue
            new = fit_replacement(g, focus, others, donor)
            if not new:
                continue
            g.record(donor, len(donor) > 1, add=False)
            new_facts = unit_atoms(g.catalog, new)
            witnesses.subtract(facts[idx])
            witnesses.update(new_facts)
            primitive_witnesses.subtract(primitives(facts[idx]))
            primitive_witnesses.update(primitives(new_facts))
            units[idx] = new
            for pos, s in zip(indices[idx], new):
                rows[pos] = s
            donors.remove(idx)
            break
        else:
            raise ValueError(f'Insufficient train diversity for {focus}: {counts[focus]} units, '
                             f'{len(partners[focus])} partners; remaining: {missing}')
