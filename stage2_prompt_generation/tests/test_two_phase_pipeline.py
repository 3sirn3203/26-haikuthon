import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from stage2_prompt_generation import pipeline
from stage2_prompt_generation.__main__ import main
from stage2_prompt_generation.client import CRITIC_SCHEMA, PROMPT_SCHEMA, VLLMClient, json_grammar
from stage2_prompt_generation.config import default_config
from stage2_prompt_generation.semantics import PRESENTATIONS


def cases(size):
    for n in range(size):
        yield f'{n:04}', {'presentation_id': n % len(PRESENTATIONS), 'entities': [], 'relations': [],
                         'background': {}, 'rendering': {}}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class TwoPhaseTests(unittest.TestCase):
    def test_12000_cases_use_exactly_24000_requests_with_all_generation_first(self):
        test = self
        config = default_config()
        failures = ['{"issues":["wrong count"]}', 'not JSON', '{"issues":"none"}', '{}']

        class FakeClient:
            generated = 0
            verified = 0
            phases = []

            def generate(self, conversations, settings, seed):
                sampled = settings.get('do_sample', True)
                self.phases.extend(['generate' if sampled else 'verify'] * len(conversations))
                if sampled:
                    test.assertEqual(self.verified, 0)
                    # Previous batches must already be visible on disk.
                    split = 'train' if self.generated < 4000 else 'test'
                    expected_rows = self.generated if split == 'train' else self.generated - 4000
                    test.assertEqual(len(read_jsonl(directory / f'{split}_prompts.jsonl')), expected_rows)
                    self.generated += len(conversations)
                    # Duplicates and short prompts are saved, with no pre-review.
                    return ['{"prompt_text":"A cup."}'] * len(conversations)
                test.assertEqual(self.generated, 12000)
                test.assertEqual(settings['json_schema'], CRITIC_SCHEMA)
                if self.verified == 0:
                    for split, size in [('train', 4000), ('test', 8000)]:
                        rows = read_jsonl(directory / f'{split}_prompts.jsonl')
                        test.assertEqual(rows, [{'id': f'{n:04}', 'prompt': 'A cup.'} for n in range(size)])
                replies = [failures[n] if n < len(failures) else '{"issues":[]}'
                           for n in range(self.verified, self.verified + len(conversations))]
                self.verified += len(conversations)
                return replies

        client = FakeClient()
        with TemporaryDirectory() as temporary, patch('sys.stdout', new=io.StringIO()):
            directory = Path(temporary)
            with patch.object(pipeline, 'OUTPUT_DIR', directory):
                summaries = pipeline.generate_train_test(cases(4000), cases(8000), {}, client, config=config)
            self.assertEqual(client.phases, ['generate'] * 12000 + ['verify'] * 12000)
            self.assertEqual(sorted(p.name for p in directory.iterdir()), sorted(
                f'{split}_{suffix}' for split in ['train', 'test']
                for suffix in ['prompts.jsonl', 'verification.jsonl', 'summary.json']))
            for split, size, failed in [('train', 4000, 4), ('test', 8000, 0)]:
                rows = read_jsonl(directory / f'{split}_verification.jsonl')
                self.assertEqual([row['id'] for row in rows], [f'{n:04}' for n in range(size)])
                self.assertTrue(all(set(row) == {'id', 'ok', 'reply'} for row in rows))
                self.assertTrue(all(type(row['ok']) is bool and isinstance(row['reply'], str) for row in rows))
                self.assertEqual(sum(not row['ok'] for row in rows), failed)
                expected = {'total': size, 'passed': size - failed, 'failed': failed,
                            'pass_rate': (size - failed) / size}
                self.assertEqual(summaries[split], expected)
                self.assertEqual(json.loads((directory / f'{split}_summary.json').read_text()), expected)
                # Critic failures never remove or modify original prompts.
                self.assertEqual(len(read_jsonl(directory / f'{split}_prompts.jsonl')), size)
            self.assertEqual([r['reply'] for r in read_jsonl(directory / 'train_verification.jsonl')[:4]], failures)

    def test_unparseable_generation_is_kept_and_verified_once(self):
        raw = ['truncated {"prompt_text":', '', '{"prompt_text":123}', '{"prompt_text":"short"}']
        test = self

        class FakeClient:
            generated = verified = 0

            def generate(self, conversations, settings, seed):
                if settings.get('do_sample', True):
                    self.generated += len(conversations)
                    return raw
                test.assertEqual(self.generated, 4)
                prompts = [json.loads(messages[1]['content'])['prompt'] for messages in conversations]
                test.assertEqual(prompts, raw[:3] + ['short'])
                self.verified += len(conversations)
                return ['{"issues":["invalid prompt"]}'] * len(conversations)

        client = FakeClient()
        with TemporaryDirectory() as temporary, patch.object(pipeline, 'OUTPUT_DIR', Path(temporary)), \
                patch('sys.stdout', new=io.StringIO()):
            # Stale selected-split summaries must be replaced; other splits stay intact.
            (Path(temporary) / 'train_summary.json').write_text('{"old":true}')
            (Path(temporary) / 'test_summary.json').write_text('{"untouched":true}')
            summary = pipeline.generate_split('train', cases(4), {}, client)
            self.assertEqual(summary, {'total': 4, 'passed': 0, 'failed': 4, 'pass_rate': 0.0})
            self.assertEqual(client.generated, 4)
            self.assertEqual(client.verified, 4)
            self.assertEqual([r['prompt'] for r in read_jsonl(Path(temporary)/'train_prompts.jsonl')], raw[:3]+['short'])
            self.assertEqual(json.loads((Path(temporary)/'test_summary.json').read_text()), {'untouched': True})

    def test_single_prompt_has_no_verification_or_writes(self):
        class FakeClient:
            calls = 0

            def generate(self, conversations, settings, seed):
                self.calls += len(conversations)
                return ['{"prompt_text":""}']

        client = FakeClient()
        identifier, scene = next(cases(1))
        with TemporaryDirectory() as temporary, patch.object(pipeline, 'OUTPUT_DIR', Path(temporary)):
            self.assertEqual(pipeline.generate_prompt(identifier, scene, {}, client), {'id': '0000', 'prompt': ''})
            self.assertEqual(client.calls, 1)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_bad_critic_response_is_false_without_retry(self):
        for reply in ['invalid', '[]', '{}', '{"issues":null}', '{"issues":false}',
                      '{"issues":{}}', '{"issues":[],"extra":true}', '{"issues":[1]}']:
            with self.subTest(reply=reply):
                self.assertFalse(pipeline._critic_ok(reply))
        self.assertTrue(pipeline._critic_ok(' {"issues": []}\n'))

    def test_cli_completes_when_verification_fails(self):
        prepared = {'train': list(cases(1)), 'test': list(cases(1))}
        result = {split: {'total': 1, 'passed': 0, 'failed': 1, 'pass_rate': 0.0} for split in prepared}
        with patch('sys.argv', ['stage2', 'generate', '--external-server']), \
                patch('sys.stdout', new=io.StringIO()), \
                patch('stage2_prompt_generation.__main__.load_component_splits', return_value=(prepared, {})), \
                patch('stage2_prompt_generation.__main__.managed_server') as server, \
                patch('stage2_prompt_generation.__main__.VLLMClient'), \
                patch('stage2_prompt_generation.__main__.generate_train_test', return_value=result):
            self.assertEqual(main(), 0)
            self.assertTrue(server.call_args.kwargs['external'])


class ClientTests(unittest.TestCase):
    def test_grammars_allow_whitespace_and_enforce_both_schemas(self):
        import xgrammar

        vocabulary = [bytes([n]) for n in range(128)] + [b'<eos>']
        compiler = xgrammar.GrammarCompiler(xgrammar.TokenizerInfo(vocabulary, stop_token_ids=[128]))
        examples = [
            (PROMPT_SCHEMA, {'prompt_text': 'A red cup sits beside a bowl.'},
             ['{"prompt_text": 1}', '{"prompt_text":"x","extra":true}', '{"issues":[]}']),
            (CRITIC_SCHEMA, {'issues': ['Incorrect count', 'Extra object']},
             ['{"issues": "none"}', '{"issues":[false]}', '{"issues":[],"extra":true}']),
        ]
        for schema, value, invalid in examples:
            grammar = compiler.compile_grammar(json_grammar(json.dumps(schema)))
            valid = [json.dumps(value), json.dumps(value, indent=2), json.dumps(value, separators=(',', ':'))]
            if schema == CRITIC_SCHEMA:
                valid.append('{"issues": []}')
            for text in valid:
                matcher = xgrammar.GrammarMatcher(grammar)
                with self.subTest(text=text):
                    self.assertTrue(matcher.accept_string(text))
                    self.assertTrue(matcher.is_completed())
            for text in invalid:
                matcher = xgrammar.GrammarMatcher(grammar)
                with self.subTest(text=text):
                    self.assertFalse(matcher.accept_string(text))

    def make_client(self):
        client = VLLMClient.__new__(VLLMClient)
        client.url = 'http://localhost'
        client.model = 'mock'
        client.enable_thinking = False
        client.calls = []
        return client

    def test_transport_errors_never_retry(self):
        config = default_config()
        errors = [HTTPError('http://localhost', code, 'error', {}, io.BytesIO(b'failed'))
                  for code in [429, 500, 502, 503, 504]] + [URLError('offline'), TimeoutError('timeout')]
        for error in errors:
            with self.subTest(error=error), patch('stage2_prompt_generation.client.urlopen', side_effect=error) as send:
                with self.assertRaises((RuntimeError, URLError, TimeoutError)):
                    self.make_client()._request([], config['generation'], 1)
                self.assertEqual(send.call_count, 1)

    def test_http_requests_use_only_prompt_and_critic_schemas(self):
        captured = []

        def send(request, timeout):
            body = json.loads(request.data)
            captured.append(body)
            result = {'choices': [{'message': {'content': '{"issues":[]}'}, 'finish_reason': 'stop'}],
                      'usage': {'completion_tokens': 5}}
            return io.BytesIO(json.dumps(result).encode())

        config = default_config()
        client = self.make_client()
        with patch('stage2_prompt_generation.client.urlopen', side_effect=send):
            client.generate([[], []], config['generation'], 10)
            client.generate([[], []], config['critic'], 20)
        for body in captured:
            sampled = body['seed'] < 20
            self.assertNotIn('response_format', body)
            self.assertEqual(body['structured_outputs']['grammar'], json_grammar(json.dumps(
                PROMPT_SCHEMA if sampled else CRITIC_SCHEMA)))
            self.assertEqual(body['temperature'], 1.0 if sampled else 0.0)
            self.assertEqual(body['chat_template_kwargs']['enable_thinking'], sampled)
            self.assertEqual(body['max_tokens'], 2048 if sampled else 384)
            if sampled:
                self.assertEqual(body['reasoning_effort'], 'low')
                self.assertEqual(body['top_p'], .95)
                self.assertEqual(body['top_k'], 20)
                self.assertEqual(body['min_p'], 0.0)
                self.assertEqual(body['presence_penalty'], 0.0)
                self.assertEqual(body['repetition_penalty'], 1.0)
            else:
                self.assertNotIn('reasoning_effort', body)
        self.assertEqual(sorted(body['seed'] for body in captured), [10, 11, 20, 21])
        self.assertEqual(sum(call['batch_size'] for call in client.calls), 4)


if __name__ == '__main__':
    unittest.main()
