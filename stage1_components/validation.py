"""Independent reconstruction of all constraints from the two component files."""
from collections import Counter, defaultdict
import json

from .coverage import check_shared_categories, training_exposure
from .groups import contrast_plan, inventory_limit, partition
from .clusters import check_clusters
from .model import ROOT, Catalog, family, inventory, normalize_scene, read_yaml, require, scene_key, total
from .lighting import lighting_weights
from .ranking import check_gallery, check_group
from .queries import validate_queries


def read_jsonl(path):
    def unique_pairs(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, f'Duplicate JSON key {k}')
            out[k] = v
        return out
    rows = []
    with open(path, encoding='utf-8') as stream:
        for i, line in enumerate(stream):
            require(line.strip(), f'{path}:{i+1}: blank line')
            try:
                rows.append(normalize_scene(json.loads(line, object_pairs_hook=unique_pairs)))
            except (ValueError, TypeError) as e:
                raise ValueError(f'{path}:{i+1}: {e}') from e
    return rows


def distributions(rows, catalog):
    result = {k: Counter() for k in ('count', 'domain', 'category', 'style', 'background', 'viewpoint', 'lighting', 'attributes', 'relations', 'primary_relation')}
    for s in rows:
        result['count'][total(s)] += 1
        cats = inventory(s)
        for c in sorted(cats):
            result['domain'][catalog.entities[c]['domain']] += 1 / len(cats)
            result['category'][c] += 1 / len(cats)
        result['style'][s['rendering']['style']] += 1
        result['background'][s['background']['location']] += 1
        result['viewpoint'][s['rendering']['viewpoint']] += 1
        result['lighting'][s['rendering']['lighting']] += 1
        result['attributes'][sum(len(e['attributes']) for e in s['entities'])] += 1
        result['relations'][len(s['relations'])] += 1
        result['primary_relation'][family(s['relations'][0]['predicate'])] += 1
    return result


def validate(train, test, catalog=None, policy=None, seed=None, verbose=True):
    from .holdouts import Holdouts, quotas
    catalog = catalog or Catalog(); policy = policy or read_yaml(ROOT / 'policy.yaml')
    lights = lighting_weights(policy, catalog)
    holds = Holdouts(catalog, policy, policy['seed'] if seed is None else seed)
    require(len(train) == sum(policy['train_counts'].values()), 'Train row count')
    require(len(test) == sum(policy.get('test_baseline_counts', {}).values()) + sum(policy['test_contrast_counts'].values()), 'Test row count')
    all_keys = set(); observed_b, observed_r = set(), set()
    summary = {}
    categories = {}
    has_ids = ['group_id' in row for row in train + test]
    require(not any(has_ids) or all(has_ids), 'Mixed missing/present group_id fields across splits')
    next_group_id = 0
    for split, rows in [('train', train), ('test', test)]:
        coverage = Counter(); kinds = Counter()
        for i, s in enumerate(rows):
            try:
                require(s.get('prompt_id') == f'{i:04d}', 'prompt_id is not the row index')
                catalog.validate(s, 4 if split == 'train' else 6)
                require(s['rendering']['lighting'] in lights, 'Lighting disabled by policy')
                k = scene_key(s); require(k not in all_keys, 'Duplicate scene within/across splits'); all_keys.add(k)
                kind = holds.kind(s); kinds[kind] += 1
                b, r = catalog.atoms(s)
                if split == 'train':
                    require(kind == 'observed', 'Train holdout leakage')
                    observed_b.update(b); observed_r.update(r)
                else:
                    require(kind != 'observed', 'Test has no held-out core composition')
                    require(all(holds.binding(*v) or v in observed_b for v in b), 'Unobserved nonholdout binding')
                    require(all(holds.relational(*v) or v in observed_r for v in r), 'Unobserved nonholdout relation')
                coverage.update(inventory(s))
            except (ValueError, KeyError, TypeError) as e:
                raise ValueError(f'{split}[{i:04d}]: {e}') from e
        categories[split] = set(coverage)
        if split == 'test':
            check_shared_categories(categories['train'], categories['test'], set(catalog.entities))
        minimum = policy['minimum_category_scenes'][split]
        missing = {c: coverage[c] for c in catalog.entities if coverage[c] < minimum}
        require(not missing, f'{split} insufficient category coverage: {missing}')
        mounts = {s['entities'][r['object']]['category'] for s in rows for r in s['relations'] if r['predicate'] == 'riding'}
        require({catalog.entities[c]['kind'] for c in mounts} == {'animal', 'object'},
                f'{split} must depict riding both animals and vehicles')
        easy, groups = partition(rows, catalog)
        cluster_summary = {}
        if split == 'train':
            require(all(s.get('cluster_id') is None for s in rows), 'Train must not contain clusters')
            require(all('is_query' not in s for s in rows), 'Query selection applies only to test')
        else:
            cluster_summary = check_clusters(rows, groups, policy, catalog, holds)
            cluster_summary.update(validate_queries(rows, policy, catalog, holds,
                                                    policy['seed'] if seed is None else seed))
        if all(has_ids):
            ids = {s['group_id'] for s in rows}
            require(ids == set(range(next_group_id, next_group_id + len(ids))),
                    f'{split} group_ids must be consecutive and unique across splits, starting at {next_group_id}')
            next_group_id += len(ids)
        planned = Counter(contrast_plan(policy, split))
        actual = Counter((total(g[0]), holds.kind(g[0]), len(g)) for g in groups)
        require(actual == planned, f'{split} group size allocation differs from group_size_weights: {actual}')
        diversity = {}
        if split == 'train':
            units, partners, _ = training_exposure([[s] for s in easy] + groups)
            targets = policy['train_diversity']
            for name, sizes in [('minimum_units', units), ('minimum_relation_partners', {c: len(v) for c, v in partners.items()})]:
                below = {c: sizes.get(c, 0) for c in catalog.entities if sizes.get(c, 0) < targets[name]}
                require(not below, f'Train {name} below {targets[name]}: {below}')
                diversity[name] = min(sizes.get(c, 0) for c in catalog.entities)
        group_metrics = Counter()
        for group in groups:
            group_metrics.update(check_group(group, catalog, holds if split == 'test' else None))
        check_gallery(rows, catalog)
        rules = policy['ranking']
        inv_rows = Counter(inventory(s) for group in groups for s in group)
        max_inventory_rows = max(inv_rows.values())
        require(max_inventory_rows <= inventory_limit(policy, split), f'{split} repeated inventory: {inv_rows.most_common(1)}')
        for c, amount in coverage.items():
            limit = rules['max_person_fraction' if c == 'person' else 'max_category_fraction']
            require(amount <= len(rows) * limit, f'{split} excessive {c} frequency: {amount}')
        if split == 'test':
            require(len(inv_rows) >= rules['minimum_distinct_test_inventories'], 'Too few distinct test inventories')
        for metric in ('count_variation', 'nonspatial_relation_variation', 'attribute_swap'):
            require(group_metrics[metric] >= len(groups) * rules['minimum_' + metric],
                    f'{split} insufficient {metric}: {group_metrics[metric]}/{len(groups)}; all metrics: {dict(group_metrics)}')
        require(group_metrics['one_object_attribute_and_direction'] <= len(groups) * rules['max_one_object_attribute_and_direction'],
                f'{split} excessive one-object attribute/direction grids: {dict(group_metrics)}')
        expected_easy = policy.get(f'{split}_baseline_counts', {})
        expected_hard = policy['test_contrast_counts'] if split == 'test' else {n: v - expected_easy.get(n, 0) for n, v in policy['train_counts'].items()}
        require(Counter(total(s) for s in easy) == Counter(expected_easy), f'{split} baseline count quotas')
        hard = [s for g in groups for s in g]
        require(Counter(total(s) for s in hard) == Counter(expected_hard), f'{split} contrast count quotas')
        repeats = sum(len(inventory(s)) < len(s['entities']) for s in hard)
        require(repeats == round(len(hard) * policy['repeated_category_fraction']), f'{split} repeated-category quota: {repeats}')
        for s in easy:
            require(2 <= len(inventory(s)) <= 3 and len(inventory(s)) == len(s['entities']), 'Easy distinct category count')
            require(total(s) <= 4 and len(s['relations']) == 1 and all(len(e['attributes']) <= 1 for e in s['entities']), 'Easy complexity')
        if split == 'test':
            # Count scene occurrence, not object multiplicity, including taxonomy.
            postings = defaultdict(set)
            for i, s in enumerate(rows):
                for c in inventory(s):
                    for parent in catalog.ancestors(c): postings[parent].add(i)
            for s in easy:
                matches = set.intersection(*(postings[c] for c in inventory(s)))
                require(len(matches) == 1, f'Easy category set is not unique: {sorted(inventory(s))}')
            for label, subset, expected in [('baseline', easy, expected_easy), ('contrast', hard, expected_hard)]:
                for n, amount in expected.items():
                    counts = Counter(holds.kind(s) for s in subset if total(s) == n)
                    require(counts == Counter(quotas(amount, policy['novelty_weights'])), f'Test {label}/{n} novelty quotas: {counts}')
            primitive_attributes = {(a, v) for c, a, v in observed_b}
            used_b, used_r = set(), set()
            for s in rows:
                b, r = catalog.atoms(s); used_b.update(b); used_r.update(r)
            unseen = sorted({(a, v) for c, a, v in used_b} - primitive_attributes)
            require(not unseen, f'Test introduces unseen attribute primitives: {unseen}')
            require({p for a, p, b in used_r} <= {p for a, p, b in observed_r}, 'Test introduces an unseen relation primitive')
        summary[split] = {'rows': len(rows), 'baseline': len(easy), 'contrast_groups': len(groups),
                          'group_sizes': dict(sorted(Counter(len(g) for g in groups).items())), **cluster_summary,
                          'distinct_categories': len(categories[split]), **diversity,
                          'repeated_category_scenes': repeats, 'minimum_category_coverage': min(coverage.values()),
                          'novelty': dict(kinds), 'distinct_contrast_inventories': len(inv_rows), 'max_rows_per_inventory': max_inventory_rows,
                          'max_nonhuman_category_scenes': max(n for c, n in coverage.items() if c != 'person'),
                          'contrast_variations': dict(group_metrics), 'distributions': distributions(rows, catalog)}
    if verbose:
        for split, data in summary.items():
            print(split, {k: v for k, v in data.items() if k != 'distributions'}, flush=True)
            for name, counts in data['distributions'].items():
                if name == 'category': continue
                print(f'  {name}: {dict((k, round(v, 2)) for k, v in sorted(counts.items()))}', flush=True)
            for name, target in [('domain', policy['domain_weights']), ('style', policy['style_weights']),
                                 ('lighting', lights), ('primary_relation', policy['relation_weights'])]:
                actual = data['distributions'][name]; z = sum(target.values())
                delta = {k: round(100 * (actual[k] / data['rows'] - target[k] / z), 2) for k in target}
                print(f'  {name} target difference (percentage points): {delta}', flush=True)
        for axis in summary['train']['distributions']:
            a = summary['train']['distributions'][axis]; b = summary['test']['distributions'][axis]
            tv = .5 * sum(abs(a[k] / len(train) - b[k] / len(test)) for k in a.keys() | b.keys())
            print(f'  train/test {axis} total variation: {tv:.3f}', flush=True)
        print(f'Validated: {len(catalog.entities)} categories, {len(holds.bindings)} binding / {len(holds.relations)} relation holdouts.', flush=True)
    return summary
