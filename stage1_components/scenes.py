"""Compose controlled alternatives while keeping all nuisance factors fixed."""
from itertools import combinations, product

from .model import relation


def make_scene(cats, counts, attrs, edges, context):
    style, place, view, lighting = context
    return {'entities': [{'category': c, 'count': n, 'attributes': dict(a)}
                         for c, n, a in zip(cats, counts, attrs)],
            'relations': [relation(*r) for r in edges],
            'background': {'location': place},
            'rendering': {'style': style, 'viewpoint': view, 'lighting': lighting}}


def required(catalog, cats, edges):
    attrs = [{} for _ in cats]
    locked = set()
    for i, p, j in edges:
        if p in ('holding', 'wearing', 'riding'):
            locked.update((i, j))
        if p in ('inside', 'on'):
            locked.add(j)
        if p == 'riding':
            attrs[i]['pose'] = 'sitting'
            if catalog.entities[cats[j]]['kind'] == 'animal':
                attrs[j]['pose'] = catalog.rules['ridden_animal_pose']
        if p == 'on' and catalog.entities[cats[i]]['kind'] == 'human':
            attrs[i]['pose'] = 'standing' if cats[j] == 'skateboard' else 'lying' if cats[j] == 'bed' else 'sitting'
            locked.add(i)
        if p == 'inside' and 'openable' in catalog.entities[cats[j]]['capabilities']:
            attrs[j]['state'] = 'open'
    return attrs, locked


def count_options(catalog, variants, n, locked):
    options = []
    for cuts in combinations(range(1, n), len(variants[0]) - 1):
        ends = (0, *cuts, n)
        counts = tuple(b-a for a, b in zip(ends, ends[1:]))
        if any(counts[i] != 1 for i in locked):
            continue
        if any(v > catalog.rules['large_subject_max_count'] and 'large_subject' in catalog.entities[c]['capabilities']
               for cats in variants for c, v in zip(cats, counts)):
            continue
        options.append(counts)
    return options


def palettes(g, cats, fixed, blocked, style, test):
    return {(i, a): g.common_values(c, a, style, test)
            for i, c in enumerate(cats) if i not in blocked and len(fixed[i]) < 3
            for a in g.catalog.attributes(c, style) if a not in fixed[i]}


def binding_factor(g, cats, fixed, slots, blocked, style, test, held, used_slots, levels=2):
    if held:
        values = {(i, a): [x for x in pair if x in g.catalog.attributes(c, style).get(a, ())]
                  for i, c in enumerate(cats) if i not in blocked and c in g.holdouts.binding_pairs
                  for a, pair in [g.holdouts.binding_pairs[c]] if a not in fixed[i]}
    else:
        values = palettes(g, cats, fixed, blocked, style, test)
    available = {field: vs for field, vs in values.items()
                 if field not in slots and len(vs) >= levels
                 and len(fixed[field[0]]) + sum(i == field[0] for i, a in slots) < 3}
    if not available:
        return None
    if levels == 2:
        swaps = [(f, k, sorted(set(available[f]) & set(available[k]))) for f, k in combinations(available, 2)
                 if f[1] == k[1] and f[0] != k[0]]
        swaps = [s for s in swaps if len(s[2]) >= 2]
        if swaps and (held or g.rng.random() < .60):
            f, k, shared = g.rng.choice(swaps)
            x, y = g.rng.sample(shared, 2)
            return [{f: x, k: y}, {f: y, k: x}]
    fresh = [field for field in available if field[0] not in used_slots]
    field = g.rng.choice(fresh or list(available))
    return [{field: v} for v in g.rng.sample(available[field], levels)]


def distinguish_repeats(g, cats, fixed, style, test, novel_field):
    for i, j in combinations(range(len(cats)), 2):
        if cats[i] != cats[j]:
            continue
        c = cats[i]
        choices = [(a, g.common_values(c, a, style, test)) for a in g.catalog.attributes(c, style)
                   if a not in fixed[i] and a not in fixed[j] and (c, a) != novel_field]
        choices = [(a, vs) for a, vs in choices if len(vs) >= 2]
        if not choices:
            return False
        axis, values = g.rng.choice(choices)
        fixed[i][axis], fixed[j][axis] = g.rng.sample(values, 2)
    return True


def compose(g, base_cats, relations, n, kind, repeated, focus=None, group_size=8, context=None, variation=None):
    """Cross attribute, quantity and relation factors; keep object categories fixed."""
    test = kind != 'observed'
    need_binding = kind in ('binding', 'both')
    cats = list(base_cats)
    if repeated and len(cats) == len(set(cats)):
        if len(cats) >= n:
            return None
        cats.append(cats[1])
    if (len(set(cats)) < len(cats)) != repeated:
        return None
    if need_binding and not any(c in g.holdouts.binding_pairs for c in cats):
        if len(cats) >= min(4, n):
            return None
        anchor = g.extra(cats, g.holdouts.binding_pairs)
        if not anchor:
            return None
        cats.append(anchor)
    if focus is not None and focus not in cats:
        if len(cats) >= min(n, 4) or not g.catalog.compatible((*cats, focus)):
            return None
        cats.append(focus)
    all_edges = set(r for variant in relations for r in variant)
    fixed, locked = required(g.catalog, cats, all_edges)
    # Optional objects belong to the shared scene template, never a variation axis.
    room = len(cats) < min(n, 4)
    base_counts = count_options(g.catalog, [cats], n, locked)
    prop_probability = .55 if n == 3 and len(base_counts) >= 2 else .85
    if variation == 'count_variation' and n == 3 and len(base_counts) >= 2:
        prop_probability = 0
    if room and (not base_counts or g.rng.random() < prop_probability):
        prop = None
        if need_binding and g.rng.random() < .36:
            # A second held category with the same palette lets both sides of
            # a binding swap be novel while preserving every attribute word.
            signatures = {(axis, frozenset(pair)) for c in cats if c in g.holdouts.binding_pairs
                          for axis, pair in [g.holdouts.binding_pairs[c]]}
            partners = [c for c, (axis, pair) in g.holdouts.binding_pairs.items()
                        if (axis, frozenset(pair)) in signatures]
            prop = g.extra(cats, partners)
        if prop is None:
            prop = g.extra(cats)
        if prop:
            cats.append(prop)
            fixed.append({})
    if len(cats) > n or len(cats) > 4:
        return None
    variants = [tuple(cats)]
    profiles = count_options(g.catalog, variants, n, locked)
    if not profiles:
        return None
    relation_bits = {1: 0, 2: 1, 4: 2}[len(relations)]
    spare = {4: 2, 8: 3, 16: 4}[group_size] - relation_bits - int(need_binding)
    if spare < 0:
        return None
    # Give each row both a foil for its novelty and foils varying other core axes.
    use_count = spare > 0 and len(profiles) >= 2 and (not relation_bits or g.rng.random() < .93)
    if not relation_bits and not use_count:
        return None
    if use_count:
        permutations = [(a, b) for a, b in combinations(profiles, 2) if sorted(a) == sorted(b)]
        profiles = g.rng.choice(permutations) if permutations else g.rng.sample(profiles, 2)
    else:
        profiles = [g.rng.choice(profiles)]
    common_bits = spare - int(use_count)
    if common_bits < 0:
        return None
    styles = None
    if need_binding:
        styles = [style for style in g.policy['style_weights']
                  if any(c in g.holdouts.binding_pairs and
                         g.holdouts.binding_pairs[c][0] in g.catalog.attributes(c, style)
                         for c in cats)]
    context = context or g.context(set().union(*map(set, variants)), all_edges, styles)
    if not context:
        return None
    style = context[0]
    variable_slots = set()
    eligible = [(c, g.holdouts.binding_pairs[c][0]) for i, c in enumerate(cats)
                if i not in variable_slots and c in g.holdouts.binding_pairs]
    novel_field = g.rng.choice(eligible) if need_binding and eligible else None
    if not distinguish_repeats(g, cats, fixed, style, test, novel_field):
        return None
    factors = []
    fields = set()
    used_slots = set()
    if need_binding:
        factor = binding_factor(g, cats, fixed, fields, variable_slots, style, test, True, used_slots)
        if not factor:
            return None
        factors.append(factor)
        fields.update(factor[0])
        used_slots.update(i for i, a in fields)
    def common_factors(bits, occupied, used):
        if not bits:
            return []
        for width, levels in ((1, 2), (2, 4)):
            if width > bits:
                continue
            factor = binding_factor(g, cats, fixed, occupied, variable_slots, style, test, False, used, levels)
            if factor:
                rest = common_factors(bits - width, occupied | factor[0].keys(), used | {i for i, _ in factor[0]})
                if rest is not None:
                    return [factor] + rest
        return None

    common = common_factors(common_bits, fields, used_slots)
    if common is None:
        return None
    factors.extend(common)
    for factor in common:
        fields.update(factor[0])
    # Supply useful common primitive observations even for train grids with count factors.
    if (not test and g.rng.random() < .83) or (test and not fields):
        choices = [((i, a), vs) for (i, a), vs in palettes(g, cats, fixed, variable_slots, style, test).items()
                   if vs and (i, a) not in fields and len(fixed[i]) + sum(j == i for j, _ in fields) < 3]
        if choices:
            (i, a), vs = g.rng.choice(choices)
            fixed[i][a] = g.rng.choice(vs)
    states = []
    for values in product(*factors):
        attrs = [dict(a) for a in fixed]
        for assignment in values:
            for (i, axis), val in assignment.items():
                attrs[i][axis] = val
        states.append(attrs)
    # Use the same direction word on both sides of a role reversal. This
    # preserves the raw predicate vocabulary as well as the depicted meaning.
    inverse = {'left_of': 'right_of', 'right_of': 'left_of', 'above': 'below', 'below': 'above'}
    preferred = {g.rng.choice(('left_of', 'right_of')), g.rng.choice(('above', 'below'))}
    relations = [tuple((j, inverse[p], i) if p in inverse and p not in preferred else (i, p, j)
                       for i, p, j in edges) for edges in relations]
    return [make_scene(cs, ns, attrs, rs, context)
            for cs, ns, attrs, rs in product(variants, profiles, states, relations)]
