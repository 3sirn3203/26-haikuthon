"""Offline checks of style balance, reproducibility, and scene preservation."""

from collections import Counter
from copy import deepcopy
from itertools import product
import json
import unittest

from stage1_components.model import Catalog

from .batch import build_batch_requests
from .prompts import (
    SINGLE_RELATION_SYNTAX, STYLE_AXES, SYSTEM_PROMPT, assign_style_profiles,
    build_prompt, style_profile_id,
)


class StyleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()
        cls.component = {
            'prompt_id': '0000',
            'entities': [
                {'category': 'ball', 'count': 1, 'attributes': {'color': 'purple'}},
                {'category': 'cabinet', 'count': 1,
                 'attributes': {'state': 'open', 'color': 'yellow'}},
                {'category': 'suitcase', 'count': 1,
                 'attributes': {'state': 'open', 'color': 'pink'}},
            ],
            'relations': [
                {'subject': 0, 'predicate': 'inside', 'object': 2},
                {'subject': 0, 'predicate': 'next_to', 'object': 1},
            ],
            'background': {'location': 'living_room'},
            'rendering': {'style': 'render_3d', 'viewpoint': 'overhead',
                          'lighting': 'warm_indoor'},
        }

    def cluster(self):
        rows = []
        for group, size in enumerate((4, 8, 16, 4, 16, 16)):
            for _ in range(size):
                rows.append({**deepcopy(self.component), 'prompt_id': f'{len(rows):04d}',
                             'group_id': group, 'cluster_id': 0, 'is_query': len(rows) % 2 == 0})
        return rows

    def test_complete_cluster_covers_64_profiles_and_balances_each_group(self):
        rows = self.cluster()
        original = deepcopy(rows)
        expected = set(product(range(4), range(4), range(2), range(2)))
        for seed in (0, 1, 42):
            with self.subTest(seed=seed):
                profiles = assign_style_profiles(rows, seed=seed)
                self.assertEqual(set(profiles.values()), expected)
                for group in range(6):
                    selected = [profiles[row['prompt_id']] for row in rows if row['group_id'] == group]
                    self.assertEqual(len(set(selected)), len(selected))
                    for axis, size in enumerate((4, 4, 2, 2)):
                        self.assertEqual(Counter(p[axis] for p in selected),
                                         {index: len(selected) // size for index in range(size)})
        self.assertEqual(rows, original)

    def test_order_scene_values_and_query_flags_do_not_determine_style(self):
        rows = self.cluster()
        expected = assign_style_profiles(rows)
        modified = deepcopy(rows)
        for row in modified:
            row['is_query'] = not row['is_query']
            row['entities'] = []  # Allocation needs only IDs and group membership.
            row['rendering'] = {}
        self.assertEqual(assign_style_profiles(list(reversed(modified))), expected)
        self.assertNotEqual(assign_style_profiles(rows, seed=1), expected)
        extra = [{**row, 'prompt_id': str(int(row['prompt_id']) + 10000),
                  'group_id': row['group_id'] + 100, 'cluster_id': 1} for row in rows]
        extended = assign_style_profiles(extra + rows)
        self.assertEqual({key: extended[key] for key in expected}, expected)

    def test_unclustered_train_groups_are_balanced(self):
        rows = self.cluster()[:12]
        for row in rows:
            row['cluster_id'] = None
        profiles = assign_style_profiles(rows, 'train')
        for group in (0, 1):
            selected = [profiles[row['prompt_id']] for row in rows if row['group_id'] == group]
            self.assertEqual(len(set(selected)), len(selected))
            for axis, size in enumerate((4, 4, 2, 2)):
                self.assertEqual(Counter(p[axis] for p in selected),
                                 {index: len(selected) // size for index in range(size)})

    def test_all_64_prompts_preserve_scene_and_assemble_four_preferences(self):
        original = deepcopy(self.component)
        baseline = build_prompt(self.component, catalog=self.catalog)['input']
        instructions = set()
        profile_ids = set()
        for profile in product(range(4), range(4), range(2), range(2)):
            prompt = build_prompt(self.component, catalog=self.catalog, style_profile=profile)
            self.assertEqual(prompt['input'], baseline)
            self.assertTrue(prompt['instructions'].startswith(SYSTEM_PROMPT))
            preferences = prompt['instructions'].split('\nWriting preferences:\n')[1].splitlines()
            self.assertEqual(len(preferences), 4)
            for axis, index in enumerate(profile):
                self.assertEqual(preferences[axis], '- ' + STYLE_AXES[axis][index])
            instructions.add(prompt['instructions'])
            profile_ids.add(style_profile_id(profile))
        self.assertEqual(len(instructions), 64)
        self.assertEqual(len(profile_ids), 64)
        self.assertEqual(self.component, original)
        self.assertEqual(set(json.loads(baseline)), {'scene'})

    def test_batch_subset_uses_complete_collection_assignment(self):
        rows = self.cluster()[:8]
        complete = build_batch_requests(rows, 'test')
        selected = build_batch_requests([rows[3]], 'test', style_components=rows)
        self.assertEqual(selected, [complete[3]])
        with self.assertRaisesRegex(ValueError, 'Missing style allocation'):
            build_batch_requests([rows[3]], 'test', style_components=rows[:2])

    def test_single_relation_gets_only_the_applicable_syntax_instruction(self):
        component = deepcopy(self.component)
        component['relations'] = component['relations'][:1]
        original = deepcopy(component)
        baseline = build_prompt(component, catalog=self.catalog)['input']
        for syntax, instruction in enumerate(SINGLE_RELATION_SYNTAX):
            with self.subTest(syntax=syntax):
                profile = (0, syntax, 0, 1)
                prompt = build_prompt(component, catalog=self.catalog, style_profile=profile)
                selected = prompt['instructions'].split('\nWriting preferences:\n')[1].splitlines()
                self.assertEqual(len(selected), 4)
                self.assertEqual(selected[1], '- ' + instruction)
                self.assertEqual(selected[3], '- ' + STYLE_AXES[3][1])
                self.assertEqual(prompt['input'], baseline)
                if syntax:
                    self.assertIn('setting', selected[1])
                    self.assertNotIn(STYLE_AXES[1][syntax], selected[1])
        self.assertEqual(component, original)

    def test_attribute_free_scene_omits_d_without_changing_counts_or_scene(self):
        component = {
            'prompt_id': '0000',
            'entities': [
                {'category': 'ball', 'count': 1, 'attributes': {}},
                {'category': 'cabinet', 'count': 1, 'attributes': {}},
            ],
            'relations': [{'subject': 0, 'predicate': 'next_to', 'object': 1}],
            'background': {'location': 'living_room'},
            'rendering': {'style': 'render_3d', 'viewpoint': 'front', 'lighting': 'soft_diffuse'},
        }
        original = deepcopy(component)
        first = build_prompt(component, catalog=self.catalog, style_profile=(0, 0, 0, 0))
        second = build_prompt(component, catalog=self.catalog, style_profile=(0, 0, 0, 1))
        self.assertEqual(first, second)
        selected = first['instructions'].split('\nWriting preferences:\n')[1].splitlines()
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(text not in first['instructions'] for text in STYLE_AXES[3]))
        scene = json.loads(first['input'])['scene']
        self.assertEqual([e['count'] for e in scene['entities']], [1, 1])
        self.assertTrue(all(not e['attributes'] for e in scene['entities']))
        self.assertEqual(component, original)

        component['entities'][0]['attributes'] = {'color': 'purple'}
        prompt = build_prompt(component, catalog=self.catalog, style_profile=(0, 0, 0, 1))
        selected = prompt['instructions'].split('\nWriting preferences:\n')[1].splitlines()
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[-1], '- ' + STYLE_AXES[3][1])

    def test_invalid_profiles_and_group_metadata_are_rejected(self):
        for profile in ((0, 0, 0), (4, 0, 0, 0), (0, 0, 2, 0), (True, 0, 0, 0)):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                build_prompt(self.component, catalog=self.catalog, style_profile=profile)
        rows = self.cluster()
        for bad_rows in ([rows[0], rows[0]], [{**rows[0], 'group_id': -1}],
                         [rows[0], {**rows[1], 'cluster_id': 99}]):
            with self.subTest(rows=bad_rows), self.assertRaises(ValueError):
                assign_style_profiles(bad_rows)


if __name__ == '__main__':
    unittest.main()
