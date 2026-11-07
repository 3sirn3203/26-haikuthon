from copy import deepcopy
from itertools import product
from pathlib import Path
import tempfile
import unittest

from stage1_components.coverage import check_shared_categories, training_exposure
from stage1_components.generator import Holdouts, quotas
from stage1_components.model import ROOT, Catalog, canonical, edge, read_yaml, relation, total
from stage1_components.ranking import Description, check_gallery, masked_key, primitive_bag
from stage1_components.validation import check_group, read_jsonl, validate


def scene():
    return {'entities': [{'category': 'mug', 'count': 1, 'attributes': {'color': 'red'}},
                         {'category': 'bowl', 'count': 1, 'attributes': {'color': 'blue'}}],
            'relations': [relation(0, 'left_of', 1)], 'background': {'location': 'studio'},
            'rendering': {'style': 'photograph', 'viewpoint': 'front',
                          'framing': 'all_target_objects_fully_visible', 'lighting': 'soft_diffuse'}}


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

    def test_monochrome_and_material_readability(self):
        a = scene(); a['rendering']['style'] = 'ink_drawing'
        with self.assertRaises(ValueError): self.catalog.validate(a)
        a = scene(); a['rendering']['style'] = 'watercolor'; a['entities'][0]['attributes'] = {'material': 'glass'}
        with self.assertRaises(ValueError): self.catalog.validate(a)

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


class GeneratedDataTests(unittest.TestCase):
    @unittest.skipUnless((ROOT.parent / 'outputs_stage1/train_components.jsonl').exists(), 'Run generate for the full dataset check')
    def test_full_dataset_without_metadata(self):
        directory = ROOT.parent / 'outputs_stage1'
        self.assertEqual({p.name for p in directory.iterdir()}, {'train_components.jsonl', 'test_components.jsonl'})
        rows = [read_jsonl(directory / (split + '_components.jsonl')) for split in ('train', 'test')]
        summary = validate(*rows, verbose=False)
        self.assertEqual(summary['train']['distinct_categories'], 173)
        self.assertEqual(summary['test']['distinct_categories'], 173)
        self.assertGreaterEqual(summary['train']['minimum_units'], 5)
        self.assertGreaterEqual(summary['train']['minimum_relation_partners'], 4)
        self.assertEqual(summary['test']['baseline'], 800)
        self.assertEqual(summary['test']['contrast_groups'], 900)


if __name__ == '__main__':
    unittest.main()
