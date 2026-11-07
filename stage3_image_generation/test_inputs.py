from copy import deepcopy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from stage3_image_generation.config import load_config, validate_config
from stage3_image_generation.pipeline import generate


class InputSelectionTests(unittest.TestCase):
    def test_v2_selected_when_old_and_new_files_both_exist(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(load_config())
            config.update(input_dir=str(root/'input'), output_dir=str(root/'output'))
            (root/'input').mkdir()
            for split in ['train', 'test']:
                for suffix, text in [('', 'Old prompt.'), ('_v2', 'New prompt.')]:
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

    def test_invalid_v2_input_fails_before_any_server_request(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = deepcopy(load_config())
            config.update(input_dir=str(root), output_dir=str(root/'output'))
            (root/'train_prompts_v2.jsonl').write_text('{"id":"0000","prompt":"Valid."}\n')
            (root/'test_prompts_v2.jsonl').write_text('{"id":"0000","prompt":""}\n')
            with patch('stage3_image_generation.pipeline.check_server') as check:
                with self.assertRaises(ValueError):generate(config)
                check.assert_not_called()
            self.assertFalse((root/'output').exists())

    def test_invalid_filename_mapping_rejected(self):
        for value in [None,{}, {'train':'a'}, {'train':'','test':'b'}]:
            config=deepcopy(load_config());config['prompt_files']=value
            with self.subTest(value=value),self.assertRaises(ValueError):validate_config(config)


if __name__=='__main__':unittest.main()
