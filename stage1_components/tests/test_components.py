from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import io
from itertools import product
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from stage1_components import __main__ as cli, files
from stage1_components.coverage import check_shared_categories, training_exposure
from stage1_components.generator import Generator, Holdouts, quotas
from stage1_components.clusters import (build_cluster, check_clusters, context_options, nested_grids,
                                       repair_clusters, replace_cluster, with_context)
from stage1_components.groups import (allocate_sizes, assign_cluster_ids, assign_group_ids, cluster_limit,
                                      cluster_plan, contrast_plan, partition, repeated_indices, size_weights)
from stage1_components.model import ROOT, Catalog, canonical, edge, normalize_scene, read_yaml, relation, total
from stage1_components.lighting import lighting_weights
from stage1_components.ranking import Description, check_gallery, masked_key, primitive_bag, visual_context
from stage1_components.validation import check_group, read_jsonl, validate
from stage1_components.queries import mark_queries, query_fraction, select_query_ids, validate_queries


def scene():
    return {'entities': [{'category': 'mug', 'count': 1, 'attributes': {'color': 'red'}},
                         {'category': 'bowl', 'count': 1, 'attributes': {'color': 'blue'}}],
            'relations': [relation(0, 'left_of', 1)], 'background': {'location': 'studio'},
            'rendering': {'style': 'photograph', 'viewpoint': 'front',
                          'lighting': 'soft_diffuse'}}


class SemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()

    def test_shared_catalog_and_active_axes(self):
        self.assertEqual(len(self.catalog.entities), 173)
        self.assertNotIn('mirror', self.catalog.entities)
        self.assertEqual(len(self.catalog.active['attributes']), 6)
        self.assertEqual(len(self.catalog.active['relations']), 10)
        self.assertNotIn('flying', self.catalog.attributes('bird').get('pose', ()))
        self.assertEqual(self.catalog.aliases['bike'], 'bicycle')

    def test_entity_order_and_inverse_are_equivalent(self):
        a = scene(); b = deepcopy(a)
        b['entities'].reverse(); b['relations'] = [relation(0, 'right_of', 1)]
        self.assertEqual(canonical(a), canonical(b))

    def test_symmetric_relation_is_equivalent(self):
        a = scene(); a['relations'] = [relation(0, 'next_to', 1)]
        b = deepcopy(a); b['relations'] = [relation(1, 'next_to', 0)]
        self.assertEqual(canonical(a), canonical(b))

    def test_transitive_relation_is_not_a_new_scene(self):
        a = scene(); a['entities'].append({'category': 'plate', 'count': 1, 'attributes': {}})
        a['relations'].append(relation(1, 'left_of', 2))
        b = deepcopy(a); b['relations'].append(relation(0, 'left_of', 2))
        self.assertEqual(canonical(a), canonical(b))
        self.assertIn(('mug', 'left_of', 'plate'), self.catalog.atoms(a)[1])

    def test_cycles_and_duplicate_inverse_edges_are_rejected(self):
        for second in (relation(1, 'left_of', 0), relation(1, 'right_of', 0)):
            a = scene(); a['relations'].append(second)
            with self.assertRaises(ValueError): self.catalog.validate(a)

    def test_projection_does_not_leak_attribute_order(self):
        a = scene(); a['entities'][1]['category'] = 'mug'
        b = deepcopy(a)
        b['entities'][0]['attributes']['color'] = 'white'
        b['entities'][1]['attributes']['color'] = 'black'
        self.assertEqual(canonical(a, 'relation'), canonical(b, 'relation'))

    def test_all_declared_objects_count(self):
        a = scene(); a['entities'][0]['count'] = 3; a['entities'][1]['count'] = 2
        self.assertEqual(total(a), 5)
        self.catalog.validate(a, 6)
        with self.assertRaises(ValueError): self.catalog.validate(a, 4)

    def test_metadata_ids_and_invalid_references_are_rejected(self):
        for mutate in (lambda s: s.update(group_id='anything'),
                       lambda s: s['entities'][0].update(entity_id='anything'),
                       lambda s: s['relations'][0].update(object=2),
                       lambda s: s['entities'][0].update(count=True)):
            a = scene(); mutate(a)
            with self.assertRaises(ValueError): self.catalog.validate(a)

    def test_group_id_is_an_integer_and_not_part_of_scene_meaning(self):
        a = scene(); b = deepcopy(a); b['group_id'] = 0
        self.catalog.validate(b)
        self.assertEqual(canonical(a), canonical(b))
        for invalid in (True, -1, 1.5, '0', None):
            b['group_id'] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.catalog.validate(b)

    def test_monochrome_and_material_readability(self):
        a = scene(); a['rendering']['style'] = 'ink_drawing'
        with self.assertRaises(ValueError): self.catalog.validate(a)
        a = scene(); a['rendering']['style'] = 'watercolor'; a['entities'][0]['attributes'] = {'material': 'glass'}
        with self.assertRaises(ValueError): self.catalog.validate(a)

    def test_retired_fields_are_removed_on_legacy_reads_but_not_new_schema(self):
        original = scene()
        original['rendering']['framing'] = 'unused legacy framing'
        original['relations'][0].update(scope='unused legacy scope', reference_frame='unused legacy frame')
        clean = normalize_scene(original)
        self.catalog.validate(clean)
        self.assertEqual(clean, scene())
        self.assertIn('framing', original['rendering'])
        with self.assertRaisesRegex(ValueError, 'Rendering fields'):
            self.catalog.validate(original)
        with tempfile.TemporaryDirectory() as directory:
            import json
            path = Path(directory) / 'legacy.jsonl'
            path.write_text(json.dumps(original) + '\n')
            self.assertEqual(read_jsonl(path), [clean])

    def test_four_lights_and_location_compatibility(self):
        self.assertEqual(set(self.catalog.active['lighting']),
                         {'soft_diffuse', 'direct_sunlight', 'side_light', 'warm_indoor'})
        indoor = scene()
        outdoor = scene()
        outdoor['entities'] = [{'category': c, 'count': 1, 'attributes': {}} for c in ('horse', 'cow')]
        outdoor['background']['location'] = 'farm'
        for row, choices in ((indoor, ('soft_diffuse', 'side_light', 'warm_indoor')),
                             (outdoor, ('soft_diffuse', 'side_light', 'direct_sunlight'))):
            for lighting in choices:
                row['rendering']['lighting'] = lighting
                self.catalog.validate(row)
        for row, lighting in ((indoor, 'direct_sunlight'), (outdoor, 'warm_indoor')):
            row['rendering']['lighting'] = lighting
            with self.assertRaisesRegex(ValueError, 'Lighting/background'):
                self.catalog.validate(row)

    def test_lighting_weight_validation_and_context_selection(self):
        self.assertEqual(lighting_weights({}, self.catalog), {'soft_diffuse': 1})
        for weights in ({}, {'unknown': 1}, {'side_light': -1}, {'side_light': 0},
                        {'side_light': True}, {'side_light': float('nan')}, {'side_light': float('inf')}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                lighting_weights({'lighting_weights': weights}, self.catalog)
        policy = read_yaml(ROOT / 'policy.yaml')
        policy['lighting_weights'] = {'soft_diffuse': 0, 'side_light': 1}
        g = Generator(policy=policy)
        for _ in range(10):
            self.assertEqual(g.context({'mug', 'bowl'}, ((0, 'left_of', 1),))[-1], 'side_light')

    def test_same_category_needs_disjoint_attributes(self):
        a = scene(); a['entities'][1] = deepcopy(a['entities'][0])
        with self.assertRaises(ValueError): self.catalog.validate(a)
        a['entities'][1]['attributes']['color'] = 'blue'; self.catalog.validate(a)

    def test_inside_visibility_and_role_fit(self):
        a = scene(); a['entities'][0]['category'] = 'apple'
        a['entities'][1]['category'] = 'box'; a['relations'] = [relation(0, 'inside', 1)]
        with self.assertRaises(ValueError): self.catalog.validate(a)
        a['entities'][1]['attributes'] = {'state': 'open'}; a['rendering']['viewpoint'] = 'overhead'
        self.catalog.validate(a)
        a['relations'].append(relation(0, 'left_of', 1))
        with self.assertRaisesRegex(ValueError, 'Overlapping'): self.catalog.validate(a)
        a['relations'].pop()
        a['entities'][0]['category'] = 'elephant'; a['entities'][0]['attributes'] = {}
        with self.assertRaises(ValueError): self.catalog.validate(a)

    def test_large_animal_cannot_use_small_support(self):
        a = scene(); a['entities'][0] = {'category': 'horse', 'count': 1, 'attributes': {}}
        a['entities'][1]['category'] = 'tray'; a['background']['location'] = 'farm'
        a['relations'] = [relation(0, 'on', 1)]
        with self.assertRaisesRegex(ValueError, 'Illegal relation roles'): self.catalog.validate(a)

    def test_human_support_requires_explicit_posture(self):
        a = scene(); a['entities'][0] = {'category': 'person', 'count': 1, 'attributes': {}}
        a['entities'][1]['category'] = 'chair'; a['relations'] = [relation(0, 'on', 1)]
        with self.assertRaisesRegex(ValueError, 'posture'): self.catalog.validate(a)
        a['entities'][0]['attributes']['pose'] = 'sitting'; self.catalog.validate(a)

    def test_mount_and_background_interactions(self):
        a = scene()
        a['entities'] = [{'category': 'person', 'count': 1, 'attributes': {'pose': 'sitting'}},
                         {'category': 'horse', 'count': 1, 'attributes': {'pose': 'lying'}}]
        a['relations'] = [relation(0, 'riding', 1)]
        a['background']['location'] = 'farm'
        with self.assertRaisesRegex(ValueError, 'Ridden animal'): self.catalog.validate(a)
        a['entities'][1]['attributes']['pose'] = 'standing'
        self.catalog.validate(a)
        b = scene()
        b['entities'][0]['category'] = 'coat_rack'
        b['entities'][1]['category'] = 'frying_pan'
        b['background']['location'] = 'playground'
        with self.assertRaisesRegex(ValueError, 'Background'): self.catalog.validate(b)
        self.assertEqual(self.catalog.contexts('helicopter'), {'airport_apron'})

    def test_real_factorial_group_and_single_axis_shortcut(self):
        rows = []
        for x, y, left in product(['red', 'green'], ['blue', 'yellow'], [0, 1]):
            a = scene(); a['entities'][0]['attributes']['color'] = x; a['entities'][1]['attributes']['color'] = y
            a['relations'] = [relation(left, 'left_of', 1 - left)]; rows.append(a)
        check_group(rows)
        rows = []
        for x in ['red', 'green', 'blue', 'yellow', 'black', 'white', 'gray', 'brown']:
            a = scene(); a['entities'][0]['attributes']['color'] = x; rows.append(a)
        with self.assertRaisesRegex(ValueError, 'Single-axis'): check_group(rows)

    def test_json_duplicate_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'data.jsonl'; p.write_text('{"prompt_id":"0000","prompt_id":"0001"}\n')
            with self.assertRaises(ValueError): read_jsonl(p)


class CoverageTests(unittest.TestCase):
    def test_eight_variants_count_once_and_only_linked_categories_are_partners(self):
        first = scene()
        first['entities'].append({'category': 'spoon', 'count': 1, 'attributes': {}})
        second = deepcopy(first)
        second['relations'] = [relation(2, 'next_to', 0)]
        third = deepcopy(first)
        third['entities'][2]['category'] = 'bowl'
        third['relations'] = [relation(2, 'left_of', 1)]
        units, partners, _ = training_exposure([[first] * 8, [second], [third]])
        self.assertEqual(units, {'mug': 3, 'bowl': 3, 'spoon': 2})
        self.assertEqual(partners['mug'], {'bowl', 'spoon'})
        self.assertEqual(partners['bowl'], {'mug'})
        self.assertEqual(partners['spoon'], {'mug'})

    def test_shared_categories_must_be_observed_in_each_split(self):
        active = {'bowl', 'mug'}
        check_shared_categories(active, active, active)
        with self.assertRaisesRegex(ValueError, 'absent from train.*mug'):
            check_shared_categories({'bowl'}, active, active)
        with self.assertRaisesRegex(ValueError, 'absent from test.*mug'):
            check_shared_categories(active, {'bowl'}, active)
        with self.assertRaisesRegex(ValueError, 'missing from both.*mug'):
            check_shared_categories({'bowl'}, {'bowl'}, active)


class HoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog(); cls.policy = read_yaml(ROOT / 'policy.yaml')
        cls.holds = Holdouts(cls.catalog, cls.policy, cls.policy['seed'])

    def test_agreed_cardinality_and_axis_spread(self):
        self.assertEqual(len(self.holds.bindings), 64)
        self.assertEqual(len(self.holds.relations), 512)
        self.assertEqual({b[1] for b in self.holds.bindings}, set(self.catalog.active['attributes']))

    def test_held_out_values_have_observable_alternatives(self):
        for category, axis, value in self.holds.bindings:
            self.assertTrue(any(value in self.catalog.attributes(c).get(axis, ()) and not self.holds.binding(c, axis, value)
                                for c in self.catalog.entities), (category, axis, value))

    def test_riding_holdouts_leave_an_animal_and_a_vehicle_to_learn_from(self):
        by_split = {True: set(), False: set()}
        for c, entry in self.catalog.entities.items():
            if self.catalog.legal_relation('person', 'riding', c):
                by_split[self.holds.relational('person', 'riding', c)].add(entry['kind'])
        self.assertEqual(by_split[True], {'animal', 'object'})
        self.assertEqual(by_split[False], {'animal', 'object'})

    def test_holdouts_reproduce(self):
        other = Holdouts(self.catalog, self.policy, self.policy['seed'])
        self.assertEqual(self.holds.bindings, other.bindings)
        self.assertEqual(self.holds.relations, other.relations)

    def test_quantity_and_style_alone_are_not_novelty(self):
        a = scene(); kind = self.holds.kind(a)
        a['entities'][0]['count'] = 5; a['rendering']['style'] = 'watercolor'
        self.assertEqual(self.holds.kind(a), kind)

    def test_taxonomy_cannot_bypass_holdout(self):
        holds = deepcopy(self.holds)
        parent_child = next((c, e['broader_category']) for c, e in self.catalog.entities.items() if e.get('broader_category'))
        child, parent = parent_child
        holds.bindings.add((parent, 'color', 'red'))
        self.assertTrue(holds.binding(child, 'color', 'red'))

    def test_integer_allocations_preserve_totals(self):
        for n in (480, 240, 80, 165, 375, 260, 75, 25):
            self.assertEqual(sum(quotas(n, self.policy['novelty_weights']).values()), n)


class RankingTests(unittest.TestCase):
    def grid(self):
        result = []
        for mug, bowl, left in product(('red', 'green'), ('blue', 'yellow'), (0, 1)):
            s = scene()
            s['entities'][0]['attributes']['color'] = mug
            s['entities'][1]['attributes']['color'] = bowl
            s['relations'] = [relation(left, 'left_of', 1 - left)]
            result.append(s)
        return result

    def holdouts(self, bindings):
        h = Holdouts.__new__(Holdouts)
        h.catalog = Catalog()
        h.bindings = set(bindings)
        h.relations = {edge('mug', p, 'bowl') for p in ('left_of', 'right_of')}
        return h

    def test_both_novelties_each_need_a_counterfactual(self):
        rows = self.grid()
        h = self.holdouts([('mug', 'color', v) for v in ('red', 'green')])
        check_group(rows, h.catalog, h)
        self.assertEqual(len({masked_key(s, h, 'binding') for s in rows}), 4)
        self.assertEqual(len({masked_key(s, h, 'relation') for s in rows}), 4)
        # Removing just relations is enough to create ties. But the *other*
        # claimed novelty is now constant and unnecessary for every query.
        h = self.holdouts([('mug', 'material', 'glass')])
        for s in rows:
            s['entities'][0]['attributes']['material'] = 'glass'
        with self.assertRaisesRegex(ValueError, 'Masking held-out binding'):
            check_group(rows, h.catalog, h)

    def test_four_and_sixteen_variant_groups_preserve_counterfactuals(self):
        for size in (4, 8, 16):
            rows = []
            bowl_colors = ('blue', 'yellow') if size >= 8 else ('blue',)
            patterns = ('solid', 'polka_dot') if size == 16 else ('solid',)
            for mug, bowl, pattern, left in product(('red', 'green'), bowl_colors, patterns, (0, 1)):
                row = scene()
                row['entities'][0]['attributes'] = {'color': mug, 'pattern': pattern}
                row['entities'][1]['attributes']['color'] = bowl
                row['relations'] = [relation(left, 'left_of', 1 - left)]
                rows.append(row)
            holds = self.holdouts([('mug', 'color', v) for v in ('red', 'green')])
            with self.subTest(size=size):
                self.assertEqual(len(rows), size)
                check_group(rows, holds.catalog, holds)
                broken = deepcopy(rows); broken[-1] = deepcopy(broken[0])
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    check_group(broken)

    def test_false_negative_with_extra_object_and_unspecified_color(self):
        query = scene()
        query['prompt_id'] = '0000'
        query['entities'][0]['attributes'] = {}
        image = deepcopy(query)
        image['prompt_id'] = '0001'
        image['entities'][0]['attributes'] = {'color': 'red'}
        image['entities'].append({'category': 'apple', 'count': 1, 'attributes': {}})
        with self.assertRaisesRegex(ValueError, 'also matches'): check_gallery([query, image])
        image['relations'][0] = relation(0, 'right_of', 1)
        check_gallery([query, image])

    def test_unbound_words_must_have_a_matched_foil(self):
        rows = self.grid()
        self.assertEqual(primitive_bag(rows[0]), primitive_bag(rows[1]))
        for s in rows[1::2]:
            s['relations'] = [relation(0, 'right_of', 1)]
        with self.assertRaisesRegex(ValueError, 'Unbound vocabulary'): check_group(rows)

    def test_gallery_accounts_for_generic_and_specific_categories(self):
        query = scene()
        query['prompt_id'] = '0000'
        query['entities'][0]['category'] = 'chair'
        image = deepcopy(query)
        image['prompt_id'] = '0001'
        image['entities'][0]['category'] = 'armchair'
        image['entities'].append({'category': 'hat', 'count': 1, 'attributes': {}})
        with self.assertRaisesRegex(ValueError, 'also matches'): check_gallery([query, image])
        image['entities'][0]['count'] = 2
        check_gallery([query, image])

    def test_same_category_matching_is_independent_of_group_order(self):
        a = scene()
        a['entities'][1]['category'] = 'mug'
        b = deepcopy(a)
        b['entities'].reverse()
        b['relations'] = [relation(1, 'left_of', 0)]
        self.assertTrue(Description(a).could_match(Description(b)))
        b['relations'] = [relation(0, 'left_of', 1)]
        self.assertFalse(Description(a).could_match(Description(b)))


class GroupPlanTests(unittest.TestCase):
    def test_exact_row_budget_and_group_count_ratios(self):
        weights = size_weights({'group_size_weights': {4: 1, 8: 1, 16: 1}})
        self.assertEqual(allocate_sizes(1008, weights), {4: 36, 8: 36, 16: 36})
        policy = read_yaml(ROOT / 'policy.yaml')
        for split in ('train', 'test'):
            plan = contrast_plan(policy, split)
            self.assertEqual({size for _, _, size in plan}, {4, 8} if split == 'train' else {4, 8, 16})
            expected = (policy['test_contrast_counts'] if split == 'test' else
                        {n: rows - policy.get('train_baseline_counts', {}).get(n, 0)
                         for n, rows in policy['train_counts'].items()})
            for n, rows in expected.items():
                self.assertEqual(sum(size for count, _, size in plan if count == n), rows)
                if split == 'test':
                    for kind, amount in quotas(rows, policy['novelty_weights']).items():
                        self.assertEqual(sum(size for count, k, size in plan if count == n and k == kind), amount)
            first = repeated_indices(plan, policy['repeated_category_fraction'], random.Random(42))
            self.assertEqual(first, repeated_indices(plan, policy['repeated_category_fraction'], random.Random(42)))
            self.assertEqual(sum(plan[i][2] for i in first), round(sum(expected.values()) * .1))
            self.assertTrue(all(plan[i][0] >= 3 for i in first))

    def test_disabled_sizes_and_impossible_budgets(self):
        weights = size_weights({'group_size_weights': {4: 0, 8: 1, 16: 0}})
        self.assertEqual(allocate_sizes(24, weights), {4: 0, 8: 3, 16: 0})
        with self.assertRaisesRegex(ValueError, 'Cannot allocate'):
            allocate_sizes(12, weights)
        for weights in ({}, {3: 1}, {'4': 1}, {4: -1}, {4: 0}, {4: True}, {4: float('nan')}, {4: float('inf')}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                size_weights({'group_size_weights': weights})

    def test_split_ratios_and_legacy_weights(self):
        policy = read_yaml(ROOT / 'policy.yaml')
        self.assertEqual(size_weights(policy, 'train'), {4: .5, 8: .5, 16: 0})
        self.assertEqual(size_weights(policy, 'test'), {4: .25, 8: .5, 16: .25})
        self.assertEqual(Counter(s for _, _, s in contrast_plan(policy, 'train')), {4: 334, 8: 333})
        for split in ('train', 'test'):
            self.assertEqual(size_weights({'group_size_weights': {8: 1}}, split), {4: 0, 8: 1, 16: 0})
        with self.assertRaisesRegex(ValueError, 'both train and test'):
            size_weights({'group_size_weights': {'train': {4: 1}}}, 'train')

    def test_impossible_cluster_frequency_limits_fail_before_generation(self):
        policy = read_yaml(ROOT / 'policy.yaml')
        policy['ranking']['max_inventory_rows']['test'] = 24
        with self.assertRaisesRegex(ValueError, 'exceeds max_inventory_rows'):
            Generator(policy=policy)

    def test_cluster_packing_preserves_quotas_and_configured_cap(self):
        policy = read_yaml(ROOT / 'policy.yaml')
        expected = Counter(contrast_plan(policy, 'test'))
        for cap in (32, 64, 96):
            policy['test_clusters']['max_size'] = cap
            plan = cluster_plan(policy)
            self.assertEqual(Counter((n, kind, s) for n, kind, sizes in plan for s in sizes), expected)
            self.assertTrue(all(len(sizes) >= 2 and sum(sizes) <= cap for _, _, sizes in plan))
        policy['test_clusters'] = None
        self.assertEqual(cluster_plan(policy), [])
        for cap in (True, 16, 30, '64', 64.0):
            policy['test_clusters'] = {'max_size': cap}
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                cluster_limit(policy)

    def test_explicit_groups_can_share_context_and_keep_ids_after_shuffle(self):
        first, second = RankingTests().grid(), RankingTests().grid()
        for group, identifier in ((first, 22), (second, 33)):
            for row in group:
                row['group_id'] = identifier
                row['cluster_id'] = 7
        for row in second:
            row['entities'][0]['attributes']['material'] = 'glass'
        rows = first + second
        random.Random(42).shuffle(rows)
        self.assertEqual(assign_group_ids(rows, Catalog(), 10), 12)
        assign_cluster_ids(rows)
        self.assertEqual(sorted(len(g) for g in partition(rows)[1]), [8, 8])
        self.assertEqual({r['group_id'] for r in first}, {first[0]['group_id']})
        self.assertNotEqual(first[0]['group_id'], second[0]['group_id'])
        self.assertEqual({r['cluster_id'] for r in rows}, {0})

    def test_cluster_integrity_and_metadata_validation(self):
        policy = {'group_size_weights': {8: 1}, 'test_clusters': {'max_size': 32},
                  'test_contrast_counts': {2: 16}, 'novelty_weights': {'binding': 1}}
        groups = [RankingTests().grid(), RankingTests().grid()]
        for identifier, group in enumerate(groups):
            for row in group:
                row.update(group_id=identifier, cluster_id=0)
                row['rendering']['lighting'] = 'soft_diffuse' if identifier == 0 else 'side_light'
        rows = [s for g in groups for s in g]
        holds = SimpleNamespace(kind=lambda s: 'binding')
        summary = check_clusters(rows, groups, policy, Catalog(), holds)
        self.assertEqual(summary['cluster_sizes'], {16: 1})
        for mutate, message in (
                (lambda: rows[0].update(cluster_id=1), 'crosses'),
                (lambda: rows[0]['background'].update(location='kitchen'), 'background'),
                (lambda: [s['rendering'].update(lighting='soft_diffuse') for s in groups[1]], 'distinct visual'),
                (lambda: [s['entities'][0]['attributes'].update(material='glass') for s in groups[1]], 'same object variants'),
                (lambda: [s.update(cluster_id=4) for s in rows], 'consecutive')):
            saved = deepcopy(rows)
            mutate()
            with self.assertRaisesRegex(ValueError, message):
                check_clusters(rows, groups, policy, Catalog(), holds)
            for row, original in zip(rows, saved):
                row.clear(); row.update(original)
        for value in (True, -1, '0', 1.5):
            s = scene(); s['cluster_id'] = value
            with self.assertRaisesRegex(ValueError, 'cluster_id'):
                Catalog().validate(s)
        s = scene(); s['cluster_id'] = None
        Catalog().validate(s)

    def test_shared_inventory_accepts_conflicts_but_rejects_ambiguous_queries(self):
        g = Generator()
        g.holdouts = SimpleNamespace(kind=lambda s: 'observed')
        first = scene()
        self.assertTrue(g.accept([first], 'observed', False))
        second = scene(); second['entities'][0]['attributes']['color'] = 'green'
        self.assertTrue(g.accept([second], 'observed', False))
        ambiguous = scene(); ambiguous['entities'][0]['attributes'] = {}
        self.assertFalse(g.accept([ambiguous], 'observed', False))
        self.assertEqual(g.rejects['gallery semantic collision'], 1)
        g.record([first], False, add=False)
        remaining = list(g.gallery.values())[0]
        self.assertEqual(set(remaining), {second['group_id']})

    def test_coverage_repair_cannot_repeatedly_replace_an_existing_occurrence(self):
        rows = RankingTests().grid()
        for row in rows:
            row.update(group_id=0, cluster_id=0)
        g = SimpleNamespace(cats=['mug'], freq=Counter(mug=8), low=16, catalog=Catalog(),
                            rng=random.Random(42), rejects=Counter(), holdouts=None,
                            rules=read_yaml(ROOT / 'policy.yaml')['ranking'])
        with patch('sys.stdout', new=io.StringIO()), self.assertRaisesRegex(ValueError, 'Cannot repair test mug'):
            repair_clusters(g, rows)

    def test_failed_cluster_replacement_restores_semantics_and_frequency_registry(self):
        g = Generator()
        g.holdouts = SimpleNamespace(kind=lambda s: 'observed')
        donor = RankingTests().grid()
        self.assertTrue(g.accept(donor, 'observed', True))
        for s in donor:
            s['cluster_id'] = 0
        before = (set(g.seen), g.freq.copy(), g.inventory_freq.copy())
        with patch('stage1_components.clusters.build_cluster', return_value=None):
            self.assertIsNone(replace_cluster(g, [donor], lambda rows, stats: True))
        self.assertEqual((g.seen, g.freq, g.inventory_freq), before)
        self.assertEqual(sum(map(len, g.gallery.values())), 1)
        self.assertEqual(len({s['group_id'] for s in donor}), 1)

    def test_sequential_global_ids_survive_shuffling_and_reject_false_membership(self):
        first = RankingTests().grid()[:4]
        second = RankingTests().grid()
        for row in second:
            row['background']['location'] = 'kitchen'
        train = first + second
        next_id = assign_group_ids(train, Catalog())
        self.assertEqual(next_id, 2)
        test = [scene()]
        self.assertEqual(assign_group_ids(test, Catalog(), next_id), 3)
        self.assertEqual(test[0]['group_id'], 2)
        random.Random(42).shuffle(train)
        easy, groups = partition(train, Catalog())
        self.assertFalse(easy)
        self.assertEqual(sorted(len(g) for g in groups), [4, 8])
        for row in train:
            row['group_id'] = 0
        with self.assertRaisesRegex(ValueError, 'different contexts'):
            partition(train, Catalog())


class GroupRoleTests(unittest.TestCase):
    def template(self):
        rows = []
        for pattern in ('solid', 'polka_dot'):
            grid = RankingTests().grid()
            for row in grid:
                row['entities'][0]['attributes']['pattern'] = pattern
            rows.extend(grid)
        return rows

    def generator(self):
        g = Generator()
        g.reset('test')
        g.holdouts = RankingTests().holdouts([('mug', 'color', v) for v in ('red', 'green')])
        g.train_categories = {'mug', 'bowl'}
        for row in self.template():
            bindings, relations = g.catalog.atoms(row)
            g.train_bindings.update(bindings)
            g.train_relations.update(relations)
        return g

    def test_nested_grids_preserve_novelty_and_complete_crossings(self):
        g = self.generator()
        master = self.template()
        original = deepcopy(master)
        grids = nested_grids(master, (4, 8, 16), g.catalog, g.holdouts)
        for size, grid in grids.items():
            self.assertEqual(len(grid), size)
            self.assertFalse(check_group(grid, g.catalog, g.holdouts)['category_variation'])
        self.assertLess({canonical(s) for s in grids[4]}, {canonical(s) for s in grids[8]})
        self.assertLess({canonical(s) for s in grids[8]}, {canonical(s) for s in grids[16]})
        self.assertEqual(master, original)

    def test_four_level_attribute_can_be_sliced_to_two_levels(self):
        master = []
        for color, left in product(('red', 'green', 'yellow', 'black'), (0, 1)):
            row = scene()
            row['entities'][0]['attributes']['color'] = color
            row['relations'] = [relation(left, 'left_of', 1 - left)]
            master.append(row)
        grids = nested_grids(master, (4, 8), Catalog(), None)
        self.assertEqual(len(grids[4]), 4)
        self.assertEqual(len({s['entities'][0]['attributes']['color'] for s in grids[4]}), 2)
        check_group(grids[4])

    def test_context_candidates_preserve_attribute_visibility_and_relation_requirements(self):
        g = self.generator()
        master = self.template()
        for row in master:
            row['entities'][1]['attributes']['material'] = 'glass'
        options = context_options(g, master)
        self.assertTrue(options)
        for place, style, view, light in options:
            self.assertNotIn(style, g.catalog.data['style_groups']['monochrome'])
            self.assertIn(style, g.catalog.data['style_groups']['material_readable_core'])
            self.assertIn(light, g.catalog.lighting_options(place))
        for row in master:
            row['relations'] = [relation(0, 'inside', 1)]
        self.assertEqual({view for _, _, view, _ in context_options(g, master)}, {'overhead'})

    def test_context_selection_prefers_changes_on_all_available_axes(self):
        g = self.generator()
        master = self.template()
        options = [('studio', 'photograph', 'front', 'soft_diffuse'),
                   ('kitchen', 'photograph', 'front', 'soft_diffuse'),
                   ('kitchen', 'colored_pencil', 'side', 'warm_indoor')]
        rendered = with_context(g, master, options, {options[0]})
        self.assertEqual(visual_context(rendered[0]), options[2])
        self.assertEqual([canonical(s) for s in master], [canonical(s) for s in rendered])
        self.assertEqual(len({visual_context(s) for s in rendered}), 1)
        self.assertIsNone(with_context(g, master, options, set(options)))

    def test_rendered_siblings_share_grids_and_validate(self):
        g = self.generator()
        with patch.object(g, 'layout', return_value=(('mug', 'bowl'), ())), \
                patch('stage1_components.clusters.compose', return_value=self.template()):
            groups = build_cluster(g, 2, 'both', (4, 8, 8, 16), False, 0)
        self.assertIsNotNone(groups)
        self.assertEqual(sorted(map(len, groups)), [4, 8, 8, 16])
        self.assertEqual(len({visual_context(group[0]) for group in groups}), 4)
        rows = [s for group in groups for s in group]
        policy = {'group_size_weights': {4: 1, 8: 2, 16: 1}, 'test_clusters': {'max_size': 64},
                  'test_contrast_counts': {2: 36}, 'novelty_weights': {'both': 1}}
        summary = check_clusters(rows, groups, policy, g.catalog, g.holdouts)
        self.assertEqual(summary['cluster_sizes'], {36: 1})
        groups[-1][0]['entities'][0]['attributes']['color'] = 'black'
        with self.assertRaisesRegex(ValueError, 'subset'):
            check_clusters(rows, groups, policy, g.catalog, g.holdouts)

    def test_failed_sibling_creation_rolls_back_all_registered_scenes(self):
        g = self.generator()
        original_accept = g.accept
        calls = []
        def accept_once(*args):
            calls.append(1)
            return original_accept(*args) if len(calls) == 1 else False
        with patch.object(g, 'layout', return_value=(('mug', 'bowl'), ())), \
                patch('stage1_components.clusters.compose', return_value=self.template()), \
                patch.object(g, 'accept', side_effect=accept_once):
            self.assertIsNone(build_cluster(g, 2, 'both', (8, 16), False, 0))
        self.assertGreater(len(calls), 1)
        self.assertFalse(g.seen)
        for counts in (g.freq, g.inventory_freq, g.style_freq, g.lighting_freq):
            self.assertEqual(sum(counts.values()), 0)
        self.assertEqual(sum(map(len, g.gallery.values())), 0)

    def test_same_object_scene_is_allowed_only_in_a_different_visual_context(self):
        g = Generator()
        g.holdouts = SimpleNamespace(kind=lambda s: 'observed')
        first = scene()
        second = deepcopy(first)
        second['background']['location'] = 'kitchen'
        self.assertTrue(g.accept([first], 'observed', False))
        self.assertTrue(g.accept([second], 'observed', False))
        self.assertFalse(g.accept([deepcopy(second)], 'observed', False))

    def test_train_simple_group_cap_survives_group_removal(self):
        g = Generator()
        g.holdouts = SimpleNamespace(kind=lambda s: 'observed')
        g.rules = {**g.rules, 'max_one_object_attribute_and_direction': 1 / len(g.plans['train'])}
        rows = RankingTests().grid()[:2] + RankingTests().grid()[4:6]
        self.assertTrue(check_group(rows)['one_object_attribute_and_direction'])
        self.assertTrue(g.accept(rows, 'observed', True))
        other = deepcopy(rows)
        for row in other:
            row['background']['location'] = 'kitchen'
        self.assertFalse(g.accept(other, 'observed', True))
        self.assertEqual(g.rejects['train simple-group cap'], 1)
        g.record(rows, True, add=False)
        self.assertTrue(g.accept(other, 'observed', True))


class QuerySelectionTests(unittest.TestCase):
    def rows(self):
        catalog = Catalog()
        grids = nested_grids(GroupRoleTests().template(), (4, 8, 16), catalog, None)
        rows = []
        for group_id, size in enumerate((4, 8, 16)):
            for row in deepcopy(grids[size]):
                row.update(group_id=group_id, cluster_id=0)
                rows.append(row)
        for group_id in range(3, 15):
            row = scene()
            row.update(group_id=group_id, cluster_id=None)
            rows.append(row)
        for index, row in enumerate(rows, start=9980):
            row['prompt_id'] = f'{index:04d}'
        return rows

    def test_half_of_each_group_and_stratified_baselines_are_reproducible(self):
        rows = self.rows()
        original = deepcopy(rows)
        catalog = Catalog()
        holds = SimpleNamespace(kind=lambda s: 'binding' if int(s['prompt_id']) % 2 else 'relation')
        policy = {'test_query_fraction': .5}
        mark_queries(rows, policy, catalog, holds, 42)
        self.assertEqual(sum(row['is_query'] for row in rows), 20)
        easy, groups = partition(rows, catalog)
        self.assertEqual(sum(row['is_query'] for row in easy), 6)
        self.assertEqual(Counter(holds.kind(s) for s in easy if s['is_query']), {'binding': 3, 'relation': 3})
        for group in groups:
            self.assertEqual(sum(row['is_query'] for row in group), len(group) // 2)
        self.assertEqual([{k: v for k, v in row.items() if k != 'is_query'} for row in rows], original)
        self.assertEqual(select_query_ids(rows[::-1], policy, catalog, holds, 42),
                         {row['prompt_id'] for row in rows if row['is_query']})
        self.assertNotEqual(select_query_ids(rows, policy, catalog, holds, 43),
                            select_query_ids(rows, policy, catalog, holds, 42))
        self.assertEqual(validate_queries(rows, policy, catalog, holds, 42),
                         {'image_candidates': 40, 'queries': 20, 'candidate_only': 20})
        rows[0]['is_query'] = not rows[0]['is_query']
        with self.assertRaisesRegex(ValueError, 'Query selection'):
            validate_queries(rows, policy, catalog, holds, 42)

    def test_sampling_policy_and_metadata_validation(self):
        for value in (0, -1, 1.1, True, '.5', float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                query_fraction({'test_query_fraction': value})
        policy = read_yaml(ROOT / 'policy.yaml')
        with self.assertRaisesRegex(ValueError, 'integer number'):
            Generator(policy={**policy, 'test_query_fraction': .3})
        self.assertIsNone(select_query_ids([], {}, Catalog(), None, 42))
        for value in (0, 1, 'true', None):
            row = scene()
            row['is_query'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                Catalog().validate(row)

    def test_default_pool_and_query_budgets(self):
        policy = read_yaml(ROOT / 'policy.yaml')
        plan = contrast_plan(policy, 'test')
        self.assertNotIn('train_baseline_counts', policy)
        self.assertNotIn('test_baseline_counts', policy)
        self.assertEqual(sum(size for _, _, size in plan), 16000)
        self.assertEqual(sum(size // 2 for _, _, size in plan), 8000)
        self.assertEqual(Counter(size for _, _, size in plan), {4: 444, 8: 888, 16: 445})

    def test_no_baseline_policy_skips_singleton_generation_for_both_splits(self):
        g = Generator()
        with patch.object(g, 'easy_train', side_effect=AssertionError('Unexpected singleton generation')), \
                patch.object(g, 'easy_test', side_effect=AssertionError('Unexpected singleton generation')):
            self.assertEqual(g.baselines_for_split(False), [])
            g.reset('test')
            self.assertEqual(g.baselines_for_split(True), [])

    def test_prompt_ids_support_pool_sizes_above_ten_thousand_without_aliases(self):
        for identifier in ('0000', '9999', '10000', '15999', '100000'):
            self.assertTrue(files.valid_prompt_id(identifier))
        for identifier in ('00000', '0010000', '999', '1.0', '../10000', 10000, None):
            self.assertFalse(files.valid_prompt_id(identifier))


class FilenameCliTests(unittest.TestCase):
    def setUp(self):
        self.train = [{'prompt_id': '0000', 'group_id': 0}]
        self.test = [{'prompt_id': '0000', 'group_id': 1}]

    def invoke(self, *arguments, rows=None):
        # Scene generation/validation is covered independently; exercise the
        # actual CLI parsing, filename selection, reads, writes and byte checks.
        with patch('sys.argv', ['stage1', *arguments]), \
                patch.object(cli, 'Generator') as generator, patch.object(cli, 'validate') as validator, \
                patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=io.StringIO()):
            generator.return_value.run.return_value = rows or (self.train, self.test)
            cli.main()
            return generator, validator

    def test_kst_clock_rolls_over_date_and_year_independently_of_host_timezone(self):
        for instant, expected in [
                (datetime(2026, 11, 14, 14, 59, 59, tzinfo=timezone.utc), '202611142359'),
                (datetime(2026, 11, 14, 15, 0, tzinfo=timezone.utc), '202611150000'),
                (datetime(2026, 12, 31, 15, 4, 59, tzinfo=timezone.utc), '202701010004')]:
            with self.subTest(instant=instant), patch.object(files, 'datetime') as clock:
                clock.now.side_effect = lambda tz: instant.astimezone(tz)
                self.assertEqual(files.default_output_id(), expected)

    def test_omitted_id_is_captured_once_for_both_output_files(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cli, 'default_output_id', return_value='202611150000') as clock:
            self.invoke('generate', '--output-dir', directory)
            clock.assert_called_once_with()
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {'train_components_202611150000.jsonl', 'test_components_202611150000.jsonl'})

    def test_named_runs_coexist_with_legacy_files_and_validation_selects_requested_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in files.component_names():
                (root / name).write_text('legacy data\n')
            with patch.object(cli, 'default_output_id') as clock:
                self.invoke('generate', '--id', '001', '--output-dir', directory)
                clock.assert_not_called()
            first_bytes = {name: (root / name).read_bytes() for name in files.component_names('001')}
            other = ([{'prompt_id': '0000', 'group_id': 99}], self.test)
            self.invoke('generate', '--id', 'trial-v2.1', '--output-dir', directory, rows=other)
            self.assertEqual(len(list(root.iterdir())), 6)
            for name, content in first_bytes.items():
                self.assertEqual((root / name).read_bytes(), content)
            generator, validator = self.invoke('validate', '--id', '001', '--output-dir', directory,
                                               '--check-reproducibility')
            self.assertEqual(validator.call_args.args, (self.train, self.test))
            generator.return_value.run.assert_called_once_with()
            with self.assertRaises(SystemExit) as failure:
                self.invoke('validate', '--id', '001', '--output-dir', directory,
                            '--check-reproducibility', rows=other)
            self.assertEqual(failure.exception.code, 1)
            for name, content in first_bytes.items():
                self.assertEqual((root / name).read_bytes(), content)

    def test_invalid_ids_and_unrelated_files_fail_without_generation_or_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for identifier in ('', '../outside', 'a/b', 'a\\b', 'a b', 'a\nb'):
                with self.subTest(identifier=identifier), self.assertRaises(SystemExit) as failure:
                    self.invoke('generate', '--id', identifier, '--output-dir', str(root / 'output'))
                self.assertEqual(failure.exception.code, 1)
                self.assertFalse((root / 'output').exists())
            (root / 'notes.txt').write_text('keep this')
            with self.assertRaises(SystemExit):
                self.invoke('generate', '--id', 'batch', '--output-dir', directory)
            self.assertEqual({p.name for p in root.iterdir()}, {'notes.txt'})
            self.assertEqual((root / 'notes.txt').read_text(), 'keep this')


class GeneratedDataTests(unittest.TestCase):
    @unittest.skipUnless((ROOT.parent / 'outputs_stage1/train_components.jsonl').exists(), 'Run generate for the full dataset check')
    def test_full_dataset_including_legacy_eight_variant_outputs(self):
        directory = ROOT.parent / 'outputs_stage1'
        self.assertTrue({'train_components.jsonl', 'test_components.jsonl'} <= {p.name for p in directory.iterdir()})
        rows = [read_jsonl(directory / (split + '_components.jsonl')) for split in ('train', 'test')]
        policy = read_yaml(ROOT / 'policy.yaml')
        if 'group_id' not in rows[0][0]:
            policy['group_size_weights'] = {8: 1}
            policy['test_clusters'] = None
            policy['train_baseline_counts'] = {2: 240, 3: 120, 4: 40}
            policy['test_contrast_counts'] = {2: 1320, 3: 3000, 4: 2080, 5: 600, 6: 200}
            policy['test_baseline_counts'] = {2: 480, 3: 240, 4: 80}
            policy.pop('test_query_fraction', None)
        summary = validate(*rows, policy=policy, verbose=False)
        self.assertEqual(summary['train']['distinct_categories'], 173)
        self.assertEqual(summary['test']['distinct_categories'], 173)
        self.assertGreaterEqual(summary['train']['minimum_units'], 5)
        self.assertGreaterEqual(summary['train']['minimum_relation_partners'], 4)
        self.assertEqual(summary['test']['baseline'], 800)
        self.assertEqual(summary['test']['contrast_groups'], len(contrast_plan(policy, 'test')))


if __name__ == '__main__':
    unittest.main()
