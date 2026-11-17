from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from stage3_image_generation.config import load_config, validate_config
from stage3_image_generation.pipeline import generate, read_prompts


class InputSelectionTests(unittest.TestCase):
    def test_stage2_metadata_preserves_all_candidates_and_prompt_bytes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(load_config())
            config.update(input_dir=str(root), output_dir=str(root/'output'))
            train = [{'id': '0000', 'prompt': 'Train prompt.\n', 'group_id': 0}]
            test = [{'id': str(10000+i), 'prompt': 'Test prompt.\nSecond sentence.',
                     'group_id': 7, 'cluster_id': 0, 'is_query': flag}
                    for i, flag in enumerate((True, False))]
            for split, rows in [('train', train), ('test', test)]:
                path = root/config['prompt_files'][split]
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                self.assertEqual(read_prompts(path), rows)
            captured = []
            def fake_generate(split, rows, config, output):
                captured.extend((split, row) for row in rows)
                return len(rows), 0, False
            with patch('stage3_image_generation.pipeline.check_server'), \
                    patch('stage3_image_generation.pipeline.generate_pending', side_effect=fake_generate), \
                    patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(generate(config), 0)
            self.assertEqual(captured, [('train', train[0]), *[('test', row) for row in test]])
            for change in ({'is_query': 'false'}, {'cluster_id': None}, {'group_id': True}, {'extra': 1}):
                path = root/'invalid.jsonl'
                path.write_text(json.dumps({**test[0], **change})+'\n')
                with self.subTest(change=change), self.assertRaises(ValueError):
                    read_prompts(path)

    def test_default_v3_selected_when_old_and_new_files_both_exist(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(load_config())
            config.update(input_dir=str(root/'input'), output_dir=str(root/'output'))
            (root/'input').mkdir()
            for split in ['train', 'test']:
                for suffix, text in [('', 'Old prompt.'), ('_v2', 'V2 prompt.'), ('_v3', 'New prompt.')]:
                    (root/'input'/f'{split}_prompts{suffix}.jsonl').write_text(json.dumps({'id':'0000','prompt':text})+'\n')
            captured = []
            def fake_generate(split, rows, config, output):
                captured.extend((split, row['prompt']) for row in rows)
                return len(rows), 0, False
            with patch('stage3_image_generation.pipeline.check_server'), \
                 patch('stage3_image_generation.pipeline.generate_pending',side_effect=fake_generate), \
                 patch('sys.stdout',new=io.StringIO()):
                self.assertEqual(generate(config), 0)
            self.assertEqual(captured,[('train','New prompt.'),('test','New prompt.')])

    def test_invalid_input_fails_before_any_server_request(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(load_config())
            config.update(input_dir=str(root), output_dir=str(root/'output'))
            (root/config['prompt_files']['train']).write_text('{"id":"0000","prompt":"Valid."}\n')
            (root/config['prompt_files']['test']).write_text('{"id":"0000","prompt":""}\n')
            with patch('stage3_image_generation.pipeline.check_server') as check:
                with self.assertRaises(ValueError):generate(config)
                check.assert_not_called()
            self.assertFalse((root/'output').exists())

    def test_invalid_filename_mapping_rejected(self):
        for value in [None,{}, {'train':'a'}, {'train':'','test':'b'}]:
            config=deepcopy(load_config());config['prompt_files']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_config(config)

    def test_default_test_input_uses_full_candidate_pool_and_accepts_five_digit_ids(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(load_config())
            config.update(input_dir=str(root), output_dir=str(root/'output'))
            pool = [{'id': identifier, 'prompt': 'An image.'} for identifier in ('9999', '10000', '15999')]
            (root/config['prompt_files']['test']).write_text(''.join(json.dumps(row)+'\n' for row in pool))
            (root/'test_queries.jsonl').write_text(json.dumps(pool[0])+'\n')
            captured = []
            def fake_generate(split, rows, config, output):
                captured.extend(row['id'] for row in rows)
                return len(rows), 0, False
            with patch('stage3_image_generation.pipeline.check_server'), \
                    patch('stage3_image_generation.pipeline.generate_pending', side_effect=fake_generate), \
                    patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(generate(config, split='test'), 0)
            self.assertEqual(captured, ['9999', '10000', '15999'])
            for invalid in ('00000', '999', '../10000', '10000/part'):
                path = root/'invalid.jsonl'
                path.write_text(json.dumps({'id': invalid, 'prompt': 'An image.'})+'\n')
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    read_prompts(path)


if __name__=='__main__':unittest.main()
