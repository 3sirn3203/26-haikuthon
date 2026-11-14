"""Generate matched text-to-image retrieval scenes using a shared, seeded registry."""
from collections import Counter, defaultdict
from itertools import combinations
import random

from .coverage import diversify_train
from .groups import (assign_cluster_ids, assign_group_ids, cluster_plan, contrast_plan,
                     inventory_limit, repeated_indices)
from .holdouts import ALTERNATIVE, Holdouts, legal_pairs, quotas
from .lighting import lighting_weights
from .model import ROOT, Catalog, edge, family, inventory, read_yaml, require, scene_key, total
from .ranking import Description, check_group, gallery_signature, visual_context
from .scenes import compose, count_options, make_scene, required
from .queries import mark_queries, query_count, query_fraction


class Generator:
    def __init__(self, seed=None, catalog=None, policy=None):
        self.catalog = catalog or Catalog()
        self.policy = policy or read_yaml(ROOT / 'policy.yaml')
        self.lighting_weights = lighting_weights(self.policy, self.catalog)
        self.seed = self.policy['seed'] if seed is None else seed
        # Reject impossible allocations before doing any scene generation.
        self.plans = {split: contrast_plan(self.policy, split) for split in ('train', 'test')}
        self.clusters = cluster_plan(self.policy, self.plans['test'])
        fraction = query_fraction(self.policy)
        if fraction is not None:
            for size in {size for _, _, size in self.plans['test']}:
                query_count(size, fraction)
        for split in ('train', 'test'):
            inventory_limit(self.policy, split)
        test_rows = sum(self.policy.get('test_baseline_counts', {}).values()) + sum(self.policy['test_contrast_counts'].values())
        if fraction is not None:
            query_count(test_rows, fraction)
        for n, _, sizes in self.clusters:
            require(sum(sizes) <= int(test_rows * self.policy['ranking']['max_category_fraction']),
                    'Test cluster exceeds the per-category frequency cap; reduce max_size or adjust the cap')
            require(sum(sizes) <= inventory_limit(self.policy, 'test'),
                    'Test cluster exceeds max_inventory_rows.test')
        if self.clusters:
            require(self.policy['ranking']['minimum_distinct_test_inventories'] <= len(self.clusters),
                    'minimum_distinct_test_inventories exceeds the number of test clusters')
        self.rng = random.Random(self.seed)
        self.cats = tuple(self.catalog.entities)
        self.domains = defaultdict(list)
        for c in self.cats:
            self.domains[self.catalog.entities[c]['domain']].append(c)
        self.legal = legal_pairs(self.catalog)
        self.holdouts = Holdouts(self.catalog, self.policy, self.seed, self.legal)
        self.seen = set()
        self.train_bindings = set()
        self.train_relations = set()
        self.train_categories = set()
        self.baselines = set()
        self.baseline_supersets = set()
        self.reset('train')

    def reset(self, split):
        self.split = split
        self.size = (sum(self.policy['train_counts'].values()) if split == 'train'
                     else sum(self.policy.get('test_baseline_counts', {}).values()) + sum(self.policy['test_contrast_counts'].values()))
        self.freq = Counter()
        self.domain_freq = Counter()
        self.style_freq = Counter()
        self.lighting_freq = Counter()
        self.lighting_rng = random.Random(f'{self.seed}/{split}/lighting')
        self.family_freq = Counter()
        self.inventory_freq = Counter()
        self.train_group_metrics = Counter()
        self.next_unit_id = 0
        self.gallery = defaultdict(dict)
        self.rejects = Counter()
        self.low = self.policy['minimum_category_scenes'][split]
        self.rules = self.policy['ranking']
        self.group_size = 1

    def weights(self, targets, observed):
        count = sum(observed.values()) + 48
        scale = sum(targets.values())
        return [max(.3, count * targets[k] / scale - observed[k]) for k in targets]

    def target(self, targets, observed):
        return self.rng.choices(list(targets), self.weights(targets, observed))[0]

    def max_count(self, c):
        key = 'max_person_fraction' if c == 'person' else 'max_category_fraction'
        return int(self.size * self.rules[key])

    def extra(self, cats, pool=None):
        options = [c for c in (self.cats if pool is None else pool) if c not in cats
                   and self.freq[c] + self.group_size <= self.max_count(c) and self.catalog.compatible((*cats, c))]
        if not options:
            return None
        domain_targets = self.policy['domain_weights']
        deficits = dict(zip(domain_targets, self.weights(domain_targets, self.domain_freq)))
        sample = self.rng.sample(options, min(18, len(options)))
        def score(c):
            d = self.catalog.entities[c]['domain']
            locations = self.catalog.contexts(c).intersection(*(self.catalog.contexts(x) for x in cats))
            return (1 + max(0, self.low - self.freq[c])) * (1 + deficits[d] ** .5) * (
                4 if locations - {'studio'} else 1) / (10 + self.freq[c])
        return self.rng.choices(sample, [score(c) for c in sample])[0]

    def context(self, cats, edges, allowed_styles=None):
        common = set.intersection(*(set(self.catalog.contexts(c)) for c in cats))
        if any(p == 'riding' for a, p, b in edges):
            common.intersection_update(self.catalog.rules['riding_backgrounds'])
        places = sorted(p for p in common if any(name in self.lighting_weights
                                                for name in self.catalog.lighting_options(p)))
        if not places:
            return None
        targets = self.policy['style_weights']
        weights = dict(zip(targets, self.weights(targets, self.style_freq)))
        styles = [s for s in targets if allowed_styles is None or s in allowed_styles]
        if not styles:
            return None
        style = self.rng.choices(styles, [weights[s] for s in styles])[0]
        location = self.rng.choices(places, [.3 if p == 'studio' else 1 for p in places])[0]
        viewpoint = 'overhead' if any(p == 'inside' for a, p, b in edges) else self.rng.choice(self.catalog.active['viewpoints'])
        available = [name for name in self.lighting_weights if name in self.catalog.lighting_options(location)]
        light_weights = dict(zip(self.lighting_weights, self.weights(self.lighting_weights, self.lighting_freq)))
        lighting = self.lighting_rng.choices(available, [light_weights[name] for name in available])[0]
        return style, location, viewpoint, lighting

    def common_values(self, c, axis, style, test):
        return [v for v in self.catalog.attributes(c, style).get(axis, ())
                if not self.holdouts.binding(c, axis, v) and (not test or (c, axis, v) in self.train_bindings)]

    def common_relation(self, a, p, b, test):
        return not self.holdouts.relational(a, p, b) and (not test or edge(a, p, b) in self.train_relations)

    def footprints(self, cats):
        expanded = sorted({parent for c in cats for parent in self.catalog.ancestors(c)})
        return {frozenset(cs) for n in (2, 3) for cs in combinations(expanded, n)}

    def accept(self, rows, kind, hard):
        test = self.split == 'test'
        occurrences = Counter(c for s in rows for c in inventory(s))
        inventories = Counter(inventory(s) for s in rows)
        try:
            require(all(self.freq[c] + n <= self.max_count(c) for c, n in occurrences.items()), 'category cap')
            if test:
                require(occurrences.keys() <= self.train_categories, 'test category absent from train')
                require(all(self.baselines.isdisjoint(self.footprints(cs)) for cs in inventories), 'reserved easy inventory')
            if hard:
                cap = inventory_limit(self.policy, self.split)
                require(all(self.inventory_freq[cs] + n <= cap for cs, n in inventories.items()), 'inventory cap')
            keys = set()
            for s in rows:
                self.catalog.validate(s, 6 if test else 4)
                require(self.holdouts.kind(s) == kind, 'wrong novelty type')
                if test:
                    bs, rs = self.catalog.atoms(s)
                    require(all(self.holdouts.binding(*b) or b in self.train_bindings for b in bs), 'unobserved common binding')
                    require(all(self.holdouts.relational(*r) or r in self.train_relations for r in rs), 'unobserved common relation')
                k = scene_key(s)
                require(k not in keys and k not in self.seen, 'duplicate scene')
                keys.add(k)
            descriptions = [Description(s, self.catalog) for s in rows]
            # Shared inventories are allowed, but every explicit query must
            # contradict every other candidate in at least one visible fact.
            for s, description in zip(rows, descriptions):
                signature = gallery_signature(s, self.catalog)
                for entries in self.gallery[visual_context(s)].values():
                    for other_signature, other in entries:
                        if signature <= other_signature or other_signature <= signature:
                            require(not (description.could_match(other) or other.could_match(description)),
                                    'gallery semantic collision')
            if hard:
                metrics = check_group(rows, self.catalog, self.holdouts if test else None)
                if not test:
                    simple = 'one_object_attribute_and_direction'
                    maximum = int(len(self.plans['train']) * self.rules['max_' + simple])
                    require(self.train_group_metrics[simple] + metrics[simple] <= maximum,
                            'train simple-group cap')
                require(not any(a.could_match(b) or b.could_match(a) for a, b in combinations(descriptions, 2)),
                        'two descriptions can match the same image')
        except ValueError as exc:
            self.rejects[str(exc)] += 1
            return False
        self.record(rows, hard, add=True)
        return True

    def record(self, rows, hard, add):
        render = visual_context(rows[0])
        keys = {scene_key(s) for s in rows}
        if add:
            identifier = self.next_unit_id
            self.next_unit_id += 1
            for row in rows:
                row['group_id'] = identifier
            self.seen.update(keys)
            self.gallery[render][identifier] = [(gallery_signature(s, self.catalog), Description(s, self.catalog)) for s in rows]
        else:
            self.seen.difference_update(keys)
            del self.gallery[render][rows[0]['group_id']]
        sign = 1 if add else -1
        if hard and self.split == 'train':
            for name, value in check_group(rows, self.catalog).items():
                self.train_group_metrics[name] += sign * value
        occurrences = Counter(c for s in rows for c in inventory(s))
        for c, n in occurrences.items():
            self.freq[c] += sign * n
            self.domain_freq[self.catalog.entities[c]['domain']] += sign * n
        for s in rows:
            if hard:
                key = inventory(s)
                self.inventory_freq[key] += sign
                if not self.inventory_freq[key]:
                    del self.inventory_freq[key]
            self.style_freq[s['rendering']['style']] += sign
            self.lighting_freq[s['rendering']['lighting']] += sign
            self.family_freq[family(s['relations'][0]['predicate'])] += sign

    def prepare_bank(self, test):
        self.fixed = defaultdict(list)
        self.variable = defaultdict(list)
        self.targets = defaultdict(list)
        self.directions = defaultdict(list)
        for p, pairs in self.legal.items():
            for a, b in pairs:
                if not self.common_relation(a, p, b, test):
                    continue
                self.fixed[family(p)].append((a, b, p, None))
                q = ALTERNATIVE.get(p)
                if q and self.common_relation(a, q, b, test):
                    self.variable[family(p)].append((a, b, p, q))
                    self.targets[a, p, q].append(b)
                    if p in ('left_of', 'above'):
                        self.directions[a].append((b, p))
        self.novel = defaultdict(list)
        self.held_targets = defaultdict(list)
        for a, b, p, q in self.holdouts.relation_pairs:
            self.novel[family(p)].append((a, b, p, q))
            self.held_targets[a, p, q].append(b)
        self.anchored = {}
        for bank_name, bank in [('known', self.variable), ('novel', self.novel)]:
            for f, pairs in bank.items():
                self.anchored[bank_name, f] = [r for r in pairs if any(c in self.holdouts.binding_pairs for c in r[:2])]
        self.focus_options = {}

    def choose_pair(self, options, focus):
        if not options:
            return None
        if focus is not None:
            key = id(options), focus
            if key not in self.focus_options:
                direct = [r for r in options if focus in r[:2]]
                compatible = [r for r in options if self.catalog.contexts(focus) &
                              self.catalog.contexts(r[0]) & self.catalog.contexts(r[1])]
                self.focus_options[key] = direct, compatible
            direct, compatible = self.focus_options[key]
            options = direct if direct and self.rng.random() < .75 else compatible
            if not options:
                return None
        # Large candidate pools can exhaust the useful grids of the highest-
        # priority pair. Explore alternatives instead of repeatedly selecting
        # a rare pair that cannot supply enough attribute levels for 16 rows.
        if self.split == 'test' and self.rng.random() < .15:
            return self.rng.choice(options)
        sample = self.rng.sample(options, min(16, len(options)))
        def priority(r):
            a, b = r[:2]
            return (int(focus is not None and focus in (a, b)),
                    -max(0, self.freq[a] + self.group_size - self.max_count(a), self.freq[b] + self.group_size - self.max_count(b)),
                    max(0, self.low - self.freq[a]) + max(0, self.low - self.freq[b]),
                    -self.inventory_freq[frozenset((a, b))] if self.rng.random() < .7 else 0)
        return max(sample, key=priority)

    def layout(self, n, kind, repeated, focus):
        test = kind != 'observed'
        held_r = kind in ('relation', 'both')
        families = self.policy['relation_weights']
        f = self.target(families, self.family_freq)
        if self.rng.random() < .14:
            f = self.rng.choice(list(families))
        bank = self.novel if held_r else self.variable
        fixed = not held_r and n >= 3 and self.rng.random() < (.70 if f not in ('horizontal', 'vertical') else .25)
        if f == 'next_to' and (held_r or n == 2):
            f = self.rng.choice(['on', 'inside', 'holding', 'riding'])
        options = self.fixed[f] if fixed else bank[f]
        needs_anchor = kind in ('binding', 'both') and (
            n == 2 or (n == 3 and (repeated or (focus is not None and focus not in self.holdouts.binding_pairs))))
        teach_anchor = not test and n == 2 and self.rng.random() < .85
        if (needs_anchor or teach_anchor) and not fixed:
            options = self.anchored.get(('novel' if held_r else 'known', f), ())
        choice = self.choose_pair(options, focus)
        if not choice:
            return None
        a, b, p, q = choice
        cats = (a, b)
        if fixed:
            rs = (((0, p, 1),),)
            if self.rng.random() < .65:
                candidates = [(c, axis) for c, axis in self.directions[a] if c not in cats and self.catalog.compatible((a, b, c))]
                if repeated:
                    candidates = [(c, axis) for c, axis in self.directions[a] if c == b and family(axis) != family(p)]
                if candidates:
                    c, axis = self.rng.choice(candidates)
                    flip = ALTERNATIVE[axis]
                    return (a, b, c), (((0, p, 1), (0, axis, 2)), ((0, p, 1), (0, flip, 2)))
            return cats, rs
        if repeated and n >= 3 and p in ('left_of', 'above') and self.rng.random() < .75:
            return (a, b, b), tuple(((0, first, 1), (0, second, 2))
                                    for first in (p, q) for second in (p, q))
        targets = self.held_targets if held_r else self.targets
        if n >= 3 and p not in ('left_of', 'above') and (repeated or self.rng.random() < .60):
            others = [c for c in targets[a, p, q] if c != b and self.catalog.compatible((a, b, c))]
            if repeated or others:
                c = b if repeated else self.rng.choice(others)
                # Specify the complementary role too: an image holding both items cannot serve as either foil.
                return (a, b, c), (((0, p, 1), (0, q, 2)), ((0, p, 2), (0, q, 1)))
        return cats, (((0, p, 1),), ((0, q, 1),))

    def easy_train(self, n, primitive=None):
        if primitive:
            axis, value = primitive
            options = [c for c in self.cats if value in self.catalog.attributes(c).get(axis, ())
                       and not self.holdouts.binding(c, axis, value)]
            a = self.rng.choice(options)
            b = self.extra((a,), self.holdouts.binding_pairs)
        else:
            a = self.rng.choice(list(self.holdouts.binding_pairs))
            b = self.extra((a,))
        if not b:
            return None
        cats = [a, b]
        if n > 2 and self.rng.random() < .65:
            c = self.extra(cats)
            if c:
                cats.append(c)
        pred = self.rng.choice(['left_of', 'right_of', 'above', 'below', 'next_to'])
        if not self.common_relation(a, pred, b, False):
            return None
        return self.baseline_scene(cats, pred, n, 'observed', primitive)

    def baseline_scene(self, cats, p, n, kind, primitive=None):
        edges = ((0, p, 1),)
        attrs, locked = required(self.catalog, cats, edges)
        profiles = count_options(self.catalog, [cats], n, locked)
        if not profiles:
            return None
        forced = primitive
        index = 0
        if kind in ('binding', 'both'):
            options = [(i, a, v) for i, c in enumerate(cats) if not attrs[i]
                       for a, v in self.holdouts.by_category.get(c, ())]
            if not options:
                return None
            index, axis, value = self.rng.choice(options)
            forced = (axis, value)
        styles = None
        if forced:
            axis, value = forced
            attrs[index][axis] = value
            styles = [s for s in self.policy['style_weights'] if value in self.catalog.attributes(cats[index], s).get(axis, ())]
        context = self.context(cats, edges, styles)
        if not context:
            return None
        for i, c in enumerate(cats):
            if attrs[i] or self.rng.random() < .22:
                continue
            options = [(a, v) for a in self.catalog.attributes(c, context[0])
                       for v in self.common_values(c, a, context[0], kind != 'observed')]
            if options:
                attrs[i] = dict([self.rng.choice(options)])
        return make_scene(cats, self.rng.choice(profiles), attrs, edges, context)

    def easy_test(self, n, kind):
        pool = self.easy_held if kind in ('both', 'relation') else self.easy_common
        a, p, b = self.rng.choice(pool)
        cats = [a, b]
        if n > 2:
            other = self.extra(cats)
            if not other:
                return None
            cats.append(other)
        inv = frozenset(cats)
        if inv in self.protected or inv in self.baseline_supersets or not self.baselines.isdisjoint(self.footprints(cats)):
            return None
        return self.baseline_scene(cats, p, n, kind)

    def baselines_for_split(self, test):
        rows = []
        # Old policies remain readable; current policies generate no singletons.
        config = self.policy.get(self.split + '_baseline_counts', {})
        if not any(config.values()):
            return rows
        primitives = sorted({(a, v) for c, a, v in self.holdouts.bindings})
        if test:
            novel_pairs = {frozenset((a, b)) for a, b, p, q in self.holdouts.relation_pairs}
            known_anchors = sorted({tuple(sorted((a, b))) for pairs in self.variable.values() for a, b, p, q in pairs
                                    if a in self.holdouts.binding_pairs or b in self.holdouts.binding_pairs})
            self.protected = set()
            for pair in list(novel_pairs) + known_anchors:
                self.protected.update(self.footprints(pair))
            self.easy_held = sorted((a, p, b) for a, p, b in self.holdouts.relations if frozenset((a, b)) not in self.protected)
            self.easy_common = sorted((a, p, b) for a, p, b in self.train_relations if a != b and frozenset((a, b)) not in self.protected)
        for n, amount in config.items():
            amounts = quotas(amount, self.policy['novelty_weights']) if test else {'observed': amount}
            for kind in (('both', 'binding', 'relation') if test else ('observed',)):
                for _ in range(amounts[kind]):
                    for _attempt in range(self.policy['attempt_limit']):
                        s = self.easy_test(n, kind) if test else self.easy_train(n, primitives[-1] if primitives else None)
                        if s and self.accept([s], kind, False):
                            rows.append(s)
                            if test:
                                self.baselines.add(inventory(s))
                                self.baseline_supersets.update(self.footprints(inventory(s)))
                            elif primitives:
                                primitives.pop()
                            break
                    else:
                        raise ValueError(f'Cannot allocate {self.split} baseline {n}/{kind}, row {len(rows)}: {self.rejects.most_common(4)}')
        return rows

    def contrast(self, test):
        if test and self.clusters:
            from .clusters import generate_clusters
            return generate_clusters(self)
        schedule = list(self.plans[self.split])
        self.rng.shuffle(schedule)
        # Keep small-count scenes first, as in the original generator.
        schedule.sort(key=lambda item: item[0])
        repeats = repeated_indices(schedule, self.policy['repeated_category_fraction'], self.rng)
        rows = []
        for i, (n, kind, group_size) in enumerate(schedule):
            self.group_size = group_size
            repeated = i in repeats
            previous = self.rejects.copy()
            missing = [c for c in self.cats if self.freq[c] < self.low]
            for attempt in range(self.policy['attempt_limit']):
                focus = None
                if missing and not repeated and attempt % 6 != 5:
                    shortlist = self.rng.sample(missing, min(5, len(missing)))
                    focus = min(shortlist, key=lambda c: (self.freq[c], len(self.catalog.contexts(c))))
                layout = self.layout(n, kind, repeated, focus)
                if layout is None:
                    continue
                group = compose(self, *layout, n, kind, repeated, focus, group_size)
                if group and self.accept(group, kind, True):
                    rows.extend(group)
                    break
            else:
                raise ValueError(f'Cannot allocate {self.split} grid {i} (n={n}, size={group_size}, {kind}, repeat={repeated}): {(self.rejects-previous).most_common(8)}')
            if (i+1) % 100 == 0 or i+1 == len(schedule):
                print(f'{self.split}: {i+1}/{len(schedule)} contrast groups; {len(self.inventory_freq)} distinct inventories', flush=True)
        missing = {c: self.freq[c] for c in self.cats if self.freq[c] < self.low}
        require(not missing, f'{self.split} category coverage below {self.low}: {missing}')
        return rows

    def run(self):
        splits = []
        next_group_id = 0
        for split in ('train', 'test'):
            test = split == 'test'
            self.reset(split)
            self.prepare_bank(test)
            rows = self.baselines_for_split(test)
            if rows:
                print(f'{split}: {len(rows)} baseline scenes', flush=True)
            rows.extend(self.contrast(test))
            if not test:
                self.train_bindings, self.train_relations = diversify_train(self, rows)
                self.train_categories = set().union(*(inventory(s) for s in rows))
            self.rng.shuffle(rows)
            next_group_id = assign_group_ids(rows, self.catalog, next_group_id)
            assign_cluster_ids(rows)
            for i, row in enumerate(rows):
                row['prompt_id'] = f'{i:04d}'
                row.setdefault('cluster_id', None)
            if test:
                mark_queries(rows, self.policy, self.catalog, self.holdouts, self.seed)
            splits.append(rows)
        return tuple(splits)
