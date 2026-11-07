from collections import Counter
from copy import deepcopy
import unittest

from stage1_components.model import Catalog
from stage1_components.validation import partition, read_jsonl
from stage2_prompt_generation.config import PROJECT_ROOT, default_config, validate_config
from stage2_prompt_generation.inputs import assign_family_presentations, load_component_splits
from stage2_prompt_generation.semantics import PRESENTATIONS


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = default_config()
        cls.catalog = Catalog()
        cls.prepared, _ = load_component_splits(cls.config)
        cls.sources = {split: read_jsonl(PROJECT_ROOT / f'outputs_stage1/{split}_components.jsonl')
                       for split in ['train', 'test']}

    def test_all_families_get_distinct_balanced_guides_and_unchanged_facts(self):
        self.assertEqual(len(PRESENTATIONS), 20)
        self.assertEqual(len(set(PRESENTATIONS)), 20)
        for split, size in [('train', 4000), ('test', 8000)]:
            cases = self.prepared[split]
            source = self.sources[split]
            self.assertEqual(len(cases), size)
            self.assertEqual([k for k, _ in cases], [r['prompt_id'] for r in source])
            by_id = dict(cases)
            _, groups = partition(source, self.catalog)
            for group in groups:
                guides = [by_id[r['prompt_id']]['presentation_id'] for r in group]
                self.assertEqual(len(guides), 8)
                self.assertEqual(len(set(guides)), 8)
                self.assertEqual(sorted(Counter(g // 4 for g in guides).values()), [1, 1, 2, 2, 2])
                self.assertEqual(sorted(Counter(g % 4 for g in guides).values()), [2, 2, 2, 2])
            for original, (identifier, scene) in zip(source, cases):
                restored = deepcopy(scene)
                restored.pop('presentation_id')
                for entity in restored['entities']:
                    entity.pop('id')
                for relation in restored['relations']:
                    for role in ['subject', 'object']:
                        relation[role] = int(relation[role][1:]) - 1
                self.assertEqual(restored, {k: original[k] for k in restored})

    def test_family_assignment_is_reproducible_independent_of_input_order(self):
        for split, cases in self.prepared.items():
            reordered = deepcopy(cases[::-1])
            assign_family_presentations(self.sources[split][::-1], reordered, self.catalog,
                                        self.config['seed'], split)
            self.assertEqual(dict(reordered), dict(cases))

    def test_invalid_phase_reasoning_settings_fail_before_generation(self):
        for override in [{'enable_thinking': False}, {'reasoning_effort': 'invalid'},
                         {'enable_thinking': 'false'}, {'min_p': -1}, {'presence_penalty': float('nan')}]:
            config = deepcopy(self.config)
            config['generation'].update(override)
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_config(config)


if __name__ == '__main__':
    unittest.main()
