"""Offline tests of Batch transport, result association, recovery, and prompts."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from email.parser import BytesParser
from email.policy import default
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import OpenAI

from .pipeline import main
from .batch import batch_jsonl, build_batch_requests, delete_input_file
from .prompts import build_prompt
from .runner import build_request, load_component, load_components, project_path, save_results


def success(identifier, text=None):
    return {
        'custom_id': identifier, 'error': None,
        'response': {'status_code': 200, 'body': {
            'status': 'completed', 'output': [
                {'type': 'reasoning', 'summary': []},
                {'type': 'message', 'role': 'assistant', 'content': [
                    {'type': 'output_text', 'text': text or f'Prompt for {identifier}.'},
                ]},
            ],
        }},
    }


class BatchAPI:
    """Exercise the real SDK against mocked Files and Batches HTTP endpoints."""

    def __init__(self):
        self.calls = []
        self.requests = []
        self.metadata = {}
        self.statuses = ['completed']
        self.submit_status = 'validating'
        self.error_rows = []
        self.output_rows = None
        self.failure_endpoint = None
        self.input_deleted = False
        self.on_delete = None
        self.factory = Mock(side_effect=self.client)

    def client(self, **kwargs):
        return OpenAI(api_key='test-key', base_url='https://example.invalid/v1',
                      http_client=httpx.Client(transport=httpx.MockTransport(self.respond)), **kwargs)

    def batch(self, status):
        return {
            'id': 'batch_test', 'object': 'batch', 'endpoint': '/v1/responses',
            'input_file_id': 'file_input', 'completion_window': '24h',
            'status': status, 'created_at': 0, 'metadata': self.metadata, 'errors': None,
            'request_counts': {'total': len(self.requests), 'completed': len(self.requests), 'failed': 0},
            'output_file_id': 'file_output' if status in ('completed', 'expired', 'cancelled') else None,
            'error_file_id': 'file_errors' if self.error_rows else None,
        }

    def respond(self, request):
        endpoint = (request.method, request.url.path)
        self.calls.append(endpoint)
        if endpoint == self.failure_endpoint:
            return httpx.Response(503, json={'error': {
                'message': 'Temporary overload', 'type': 'server_error', 'code': 'server_error',
            }})
        if endpoint == ('POST', '/v1/files'):
            self.input_deleted = False
            headers = b'Content-Type: ' + request.headers['content-type'].encode() + b'\r\n\r\n'
            message = BytesParser(policy=default).parsebytes(headers + request.content)
            fields = {part.get_param('name', header='content-disposition'):
                      part.get_payload(decode=True) for part in message.iter_parts()}
            assert fields['purpose'] == b'batch'
            self.input_text = fields['file'].decode()
            self.requests = [json.loads(line) for line in self.input_text.splitlines()]
            return httpx.Response(200, json={
                'id': 'file_input', 'object': 'file', 'bytes': len(fields['file']),
                'created_at': 0, 'filename': 'requests.jsonl', 'purpose': 'batch',
            })
        if endpoint == ('POST', '/v1/batches'):
            body = json.loads(request.content)
            assert body['endpoint'] == '/v1/responses'
            assert body['completion_window'] == '24h'
            assert body['input_file_id'] == 'file_input'
            self.metadata = body['metadata']
            return httpx.Response(200, json=self.batch(self.submit_status))
        if endpoint == ('GET', '/v1/batches/batch_test'):
            status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
            return httpx.Response(200, json=self.batch(status))
        if endpoint == ('DELETE', '/v1/files/file_input'):
            if self.input_deleted:
                return httpx.Response(404, json={'error': {
                    'message': 'Input file was deleted', 'type': 'invalid_request_error', 'code': 'file_not_found',
                }})
            if self.on_delete is not None:
                self.on_delete()
            self.input_deleted = True
            return httpx.Response(200, json={'id': 'file_input', 'object': 'file', 'deleted': True})
        if request.method == 'GET' and request.url.path.startswith('/v1/files/'):
            file_id = request.url.path.split('/')[3]
            if file_id == 'file_input':
                if self.input_deleted:
                    return httpx.Response(404, json={'error': {
                        'message': 'Input file was deleted', 'type': 'invalid_request_error', 'code': 'file_not_found',
                    }})
                return httpx.Response(200, text=self.input_text)
            rows = (self.error_rows if file_id == 'file_errors' else
                    self.output_rows if self.output_rows is not None else
                    [success(row['custom_id']) for row in reversed(self.requests)])
            return httpx.Response(200, text=''.join(json.dumps(row) + '\n' for row in rows))
        raise AssertionError(f'Unexpected endpoint (direct inference is forbidden): {endpoint}')


class OpenAIExperimentTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
        environment.start()
        self.addCleanup(environment.stop)
        dotenv = patch('dotenv.load_dotenv')
        dotenv.start()
        self.addCleanup(dotenv.stop)
        self.components = load_components(project_path('outputs_stage1/test_components.jsonl'), limit=2)
        for i, component in enumerate(self.components):
            component.update(group_id=i, cluster_id=0, is_query=i == 0)
        self.component = self.components[0]
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.source = self.directory / 'components.jsonl'
        self.write_source(self.components)
        self.target = self.directory / 'result.jsonl'

    def expected_rows(self, components, split='test'):
        return [dict(id=c['prompt_id'], prompt=f'Prompt for {split}-{c["prompt_id"]}.',
                     group_id=c['group_id'], **({} if split == 'train' else
                     {'cluster_id': c['cluster_id'], 'is_query': c['is_query']})) for c in components]

    def write_source(self, components):
        self.source.write_text(''.join(json.dumps(row) + '\n' for row in components), encoding='utf-8')

    def grouped_components(self):
        return [
            {**self.components[0], 'group_id': 0, 'cluster_id': 0, 'is_query': True},
            {**self.components[1], 'group_id': 1, 'cluster_id': 1, 'is_query': True},
            {**self.components[0], 'prompt_id': '15999', 'group_id': 0,
             'cluster_id': 0, 'is_query': False},
            {**self.components[1], 'prompt_id': '16000', 'group_id': 2,
             'cluster_id': 0, 'is_query': False},
        ]

    def run_cli(self, api, *args):
        stderr = io.StringIO()
        with patch('openai.OpenAI', api.factory), redirect_stdout(io.StringIO()), redirect_stderr(stderr):
            code = main(['--input', str(self.source), '--output', str(self.target), *args])
        return code, stderr.getvalue()

    def test_selection_all_limit_multiple_ids_and_single_json(self):
        self.assertEqual(load_components(self.source), self.components)
        self.assertEqual(load_components(self.source, limit=1), self.components[:1])
        self.assertEqual(load_components(self.source, ['0001', '0000']), self.components)
        self.assertEqual(load_component(self.source, '0001'), self.components[1])
        single = self.directory / 'component.json'
        wide_id = {**self.component, 'prompt_id': '15999'}
        single.write_text(json.dumps(wide_id), encoding='utf-8')
        self.assertEqual(load_components(single), [wide_id])
        for ids in (['42'], ['9999'], ['0000', '0000']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                load_components(self.source, ids)
        with self.assertRaises(ValueError):
            load_components(single, ['0000'])
        self.write_source([self.component, self.component])
        with self.assertRaises(ValueError):
            load_components(self.source)

    def test_prompt_preserves_original_and_resolves_relationships(self):
        original = deepcopy(self.component)
        scene = json.loads(build_prompt(self.component)['input'])['scene']
        self.assertEqual(self.component, original)
        self.assertEqual([entity['count'] for entity in scene['entities']],
                         [entity['count'] for entity in original['entities']])
        for before, after in zip(original['relations'], scene['relationships']):
            subject = scene['entities'][before['subject']]['category']
            target = scene['entities'][before['object']]['category']
            self.assertEqual(after, f'{subject} to the right of {target}')
        self.assertIn('glass', scene['entities'][0]['attributes']['material'])
        self.assertIn('metal', scene['entities'][2]['attributes']['material'])
        self.assertIn('studio', scene['background']['location'])
        self.assertNotIn('framing', scene['rendering'])

    def test_repeated_categories_use_enough_attributes_to_resolve_each_reference(self):
        component = {
            'prompt_id': '0000',
            'entities': [
                {'category': 'bowl', 'count': 2,
                 'attributes': {'color': 'red', 'pattern': 'striped'}},
                {'category': 'bowl', 'count': 1,
                 'attributes': {'color': 'red', 'pattern': 'polka_dot'}},
                {'category': 'bowl', 'count': 1,
                 'attributes': {'color': 'blue', 'pattern': 'striped'}},
            ],
            'relations': [
                {'subject': 0, 'predicate': 'next_to', 'object': 1},
                {'subject': 2, 'predicate': 'left_of', 'object': 0},
            ],
            'background': {'location': 'studio'},
            'rendering': {'style': 'render_3d', 'viewpoint': 'overhead',
                          'lighting': 'soft_diffuse'},
        }
        original = deepcopy(component)
        scene = json.loads(build_prompt(component)['input'])['scene']
        self.assertEqual(scene['relationships'], [
            'red striped bowl next to polka-dotted bowl',
            'blue bowl to the left of red striped bowl',
        ])
        self.assertEqual(component, original)
        self.assertEqual([e['count'] for e in scene['entities']], [2, 1, 1])

    def test_complete_group_and_cluster_selection_includes_scattered_nonquery_candidates(self):
        rows = self.grouped_components()
        self.write_source(rows)
        self.assertEqual(load_components(self.source, group_id=0), [rows[0], rows[2]])
        self.assertEqual(load_components(self.source, cluster_id=0), [rows[0], rows[2], rows[3]])
        for selection in ({'group_id': 999}, {'cluster_id': -1}, {'group_id': True},
                          {'group_id': 0, 'limit': 1}, {'group_id': 0, 'cluster_id': 0}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                load_components(self.source, **selection)
        self.write_source([{k: v for k, v in c.items() if k != 'group_id'}
                           for c in self.components])  # Missing grouping cannot select all rows.
        with self.assertRaisesRegex(ValueError, 'no valid group_id'):
            load_components(self.source, group_id=0)

    def test_offset_selects_input_rows_and_rejects_invalid_ranges(self):
        rows = self.grouped_components()
        self.write_source(rows)
        self.assertEqual(load_components(self.source, offset=1, limit=2), rows[1:3])
        self.assertEqual(load_components(self.source, offset=2), rows[2:])
        for selection in ({'offset': -1}, {'offset': True}, {'offset': 1.5},
                          {'offset': 4}, {'offset': 1, 'prompt_ids': ['0000']},
                          {'offset': 1, 'group_id': 0}, {'offset': 1, 'cluster_id': 0}):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                load_components(self.source, **selection)

    def test_cluster_submission_and_later_collection_preserve_every_selected_candidate(self):
        rows = self.grouped_components()
        self.write_source(rows)
        api = BatchAPI()
        code, stderr = self.run_cli(api, '--cluster-id', '0')
        self.assertEqual(code, 0, stderr)
        self.assertEqual([row['custom_id'] for row in api.requests],
                         ['test-0000', 'test-15999', 'test-16000'])
        code, stderr = self.run_cli(api, '--batch-id', 'batch_test')
        self.assertEqual(code, 0, stderr)
        saved = [json.loads(line) for line in self.target.read_text().splitlines()]
        self.assertEqual(saved, self.expected_rows([rows[0], rows[2], rows[3]]))
        self.assertEqual(api.calls.count(('POST', '/v1/batches')), 1)

    def test_retired_constraints_never_reach_instructions_or_payload(self):
        component = deepcopy(self.component)
        component['rendering']['framing'] = 'RETIRED_FRAMING'
        for rel in component['relations']:
            rel.update(scope='RETIRED_SCOPE', reference_frame='RETIRED_REFERENCE')
        request_text = json.dumps(build_prompt(component)).lower()
        for text in ('framing', 'reference_frame', 'all_pairs', 'scope', 'viewer', 'fully visible', 'retired_'):
            self.assertNotIn(text, request_text)
        component['rendering']['lighting'] = 'side_light'
        scene = json.loads(build_prompt(component)['input'])['scene']
        self.assertEqual(scene['rendering']['lighting'], 'side lighting')

    def test_submit_wait_and_unordered_results_use_only_batch_endpoints(self):
        api = BatchAPI()
        api.statuses = ['in_progress', 'completed']
        api.on_delete = lambda: self.assertTrue(self.target.exists(), 'Save results before deleting the input')
        with patch('time.sleep') as sleep:
            code, stderr = self.run_cli(api, '--wait', '--poll-interval', '1',
                                        '--model', 'gpt-6.1-terra', '--reasoning-effort', 'low',
                                        '--temperature', 'none')
        self.assertEqual(code, 0, stderr)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(api.calls, [
            ('POST', '/v1/files'), ('POST', '/v1/batches'),
            ('GET', '/v1/batches/batch_test'), ('GET', '/v1/batches/batch_test'),
            ('GET', '/v1/files/file_output/content'),
            ('DELETE', '/v1/files/file_input'),
        ])
        self.assertEqual(api.factory.call_args.kwargs, {'max_retries': 0, 'timeout': 120.0})
        rows = [json.loads(line) for line in self.target.read_text().splitlines()]
        self.assertEqual(rows, self.expected_rows(self.components))
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), ['components.jsonl', 'result.jsonl'])
        for row in api.requests:
            self.assertEqual(row['body']['model'], 'gpt-6.1-terra')
            self.assertEqual(row['body']['reasoning'], {'effort': 'low'})
            self.assertNotIn('temperature', row['body'])
            self.assertFalse(row['body']['store'])

    def test_submit_returns_id_then_collect_restores_original_selection_without_resubmitting(self):
        api = BatchAPI()
        for component in self.components:
            component.pop('is_query')
            component['cluster_id'] = None
        self.write_source(self.components)
        code, stderr = self.run_cli(api, '--split', 'train', '--id', '0001')
        self.assertEqual(code, 0, stderr)
        self.assertIn('batch_test', stderr)
        self.assertFalse(self.target.exists())
        self.assertFalse(api.input_deleted)
        code, stderr = self.run_cli(api, '--batch-id', 'batch_test')
        self.assertEqual(code, 0, stderr)
        self.assertEqual(api.calls.count(('POST', '/v1/batches')), 1)
        self.assertEqual(api.requests[0]['custom_id'], 'train-0001')
        rows = [json.loads(line) for line in self.target.read_text().splitlines()]
        self.assertEqual(rows, self.expected_rows(self.components[1:], 'train'))
        self.assertIn(('GET', '/v1/files/file_input/content'), api.calls)
        self.assertTrue(api.input_deleted)

    def test_collect_next_range_appends_and_recollection_is_idempotent(self):
        save_results(self.target, [{'input_component': self.component,
                                   'output': 'Prompt for test-0000.'}], 'test')
        api = BatchAPI()
        code, stderr = self.run_cli(api, '--offset', '1', '--limit', '1', '--append')
        self.assertEqual(code, 0, stderr)
        self.assertEqual([r['custom_id'] for r in api.requests], ['test-0001'])
        self.assertEqual(len(self.target.read_text().splitlines()), 1)
        code, stderr = self.run_cli(api, '--batch-id', 'batch_test', '--append')
        self.assertEqual(code, 0, stderr)
        self.assertEqual([json.loads(line) for line in self.target.read_text().splitlines()],
                         self.expected_rows(self.components))
        original = self.target.read_bytes()
        code, stderr = self.run_cli(api, '--batch-id', 'batch_test', '--append')
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(api.calls.count(('POST', '/v1/batches')), 1)
        self.assertTrue(api.input_deleted)
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), ['components.jsonl', 'result.jsonl'])

    def test_append_conflict_keeps_existing_results_and_remote_input(self):
        api = BatchAPI()
        self.assertEqual(self.run_cli(api, '--id', '0001')[0], 0)
        save_results(self.target, [{'input_component': self.components[1],
                                   'output': 'An existing different prompt.'}], 'test')
        original = self.target.read_bytes()
        code, stderr = self.run_cli(api, '--batch-id', 'batch_test', '--append')
        self.assertEqual(code, 1)
        self.assertIn('Conflicting saved result for ID 0001', stderr)
        self.assertEqual(self.target.read_bytes(), original)
        self.assertFalse(api.input_deleted)
        self.assertEqual(api.calls.count(('POST', '/v1/batches')), 1)

    def test_save_failure_preserves_remote_input_for_later_collection(self):
        api = BatchAPI()
        api.submit_status = 'completed'
        with patch('stage2_prompt_generation.pipeline.save_results',
                   side_effect=OSError('Disk full')):
            code, stderr = self.run_cli(api)
        self.assertEqual(code, 1)
        self.assertIn('Disk full', stderr)
        self.assertFalse(api.input_deleted)
        self.assertNotIn(('DELETE', '/v1/files/file_input'), api.calls)
        self.assertFalse(self.target.exists())

    def test_cleanup_failure_keeps_saved_results_and_does_not_resubmit(self):
        api = BatchAPI()
        api.submit_status = 'completed'
        api.failure_endpoint = ('DELETE', '/v1/files/file_input')
        code, stderr = self.run_cli(api)
        self.assertEqual(code, 1)
        self.assertTrue(self.target.exists())
        self.assertFalse(api.input_deleted)
        self.assertEqual(api.calls.count(api.failure_endpoint), 1)
        self.assertEqual(api.calls.count(('POST', '/v1/batches')), 1)

    def test_collection_still_works_after_input_cleanup(self):
        api = BatchAPI()
        api.submit_status = 'completed'
        self.assertEqual(self.run_cli(api, '--id', '0001')[0], 0)
        first_result = self.target.read_text()
        self.assertTrue(api.input_deleted)
        self.target.unlink()
        code, stderr = self.run_cli(api, '--batch-id', 'batch_test')
        self.assertEqual(code, 0, stderr)
        self.assertEqual(self.target.read_text(), first_result)
        self.assertEqual(api.calls.count(('POST', '/v1/batches')), 1)
        self.assertTrue(self.source.exists())

    def test_cleanup_rejects_batches_that_are_still_running(self):
        api = BatchAPI()
        self.assertEqual(self.run_cli(api)[0], 0)
        with api.client() as client:
            batch = client.batches.retrieve('batch_test')
            batch.status = 'in_progress'
            with self.assertRaises(ValueError):
                delete_input_file(client, batch)
        self.assertFalse(api.input_deleted)
        self.assertNotIn(('DELETE', '/v1/files/file_input'), api.calls)

    def test_collect_rejects_changed_component_snapshot(self):
        api = BatchAPI()
        self.assertEqual(self.run_cli(api)[0], 0)
        modified = deepcopy(self.components)
        modified[0]['entities'][0]['count'] += 1
        self.write_source(modified)
        code, stderr = self.run_cli(api, '--batch-id', 'batch_test')
        self.assertEqual(code, 1)
        self.assertIn('changed after submission', stderr)
        self.assertFalse(self.target.exists())
        self.assertEqual(api.calls.count(('POST', '/v1/batches')), 1)

    def test_expired_batch_keeps_successes_and_reports_error_file(self):
        api = BatchAPI()
        api.statuses = ['expired']
        api.output_rows = [success('test-0001')]
        api.error_rows = [{'custom_id': 'test-0000', 'response': None,
                           'error': {'code': 'batch_expired', 'message': 'Expired'}}]
        with patch('time.sleep'):
            code, stderr = self.run_cli(api, '--wait')
        self.assertEqual(code, 1)
        self.assertIn('test-0000: batch_expired', stderr)
        rows = [json.loads(line) for line in self.target.read_text().splitlines()]
        self.assertEqual(rows, self.expected_rows(self.components[1:]))

    def test_incomplete_refused_empty_and_http_error_results_are_not_saved(self):
        incomplete = success('test-0000')
        incomplete['response']['body'].update(status='incomplete',
                                               incomplete_details={'reason': 'max_output_tokens'})
        refusal = success('test-0000')
        refusal['response']['body']['output'] = [{'type': 'message', 'content': [
            {'type': 'refusal', 'refusal': 'Cannot comply'},
        ]}]
        empty = success('test-0000')
        empty['response']['body']['output'] = []
        http_error = {'custom_id': 'test-0000', 'response': {'status_code': 400}, 'error': None}
        for row in (incomplete, refusal, empty, http_error):
            with self.subTest(row=row):
                api = BatchAPI()
                api.submit_status = 'completed'
                api.output_rows = [row]
                code, stderr = self.run_cli(api, '--id', '0000')
                self.assertEqual(code, 1, stderr)
                self.assertFalse(self.target.exists())

    def test_missing_duplicate_and_foreign_result_ids_are_detected(self):
        api = BatchAPI()
        api.submit_status = 'completed'
        api.output_rows = [success('test-0000')]
        code, stderr = self.run_cli(api)
        self.assertEqual(code, 1)
        self.assertIn('test-0001: missing_result', stderr)
        self.target.unlink()
        for rows in ([success('test-0000'), success('test-0000')], [success('test-9999')]):
            api.output_rows = rows
            code, stderr = self.run_cli(api)
            self.assertEqual(code, 1)
            self.assertIn('Unexpected or duplicate', stderr)
            self.assertFalse(self.target.exists())

    def test_api_failure_has_no_automatic_resubmission_or_output(self):
        for endpoint in (('POST', '/v1/files'), ('POST', '/v1/batches')):
            with self.subTest(endpoint=endpoint):
                api = BatchAPI()
                api.failure_endpoint = endpoint
                code, stderr = self.run_cli(api)
                self.assertEqual(code, 1, stderr)
                self.assertEqual(api.calls.count(endpoint), 1)
                self.assertFalse(self.target.exists())

    def test_dry_run_emits_multiple_jsonl_requests_without_api_or_files(self):
        stdout = io.StringIO()
        with patch('openai.OpenAI') as factory, redirect_stdout(stdout):
            code = main(['--input', str(self.source), '--limit', '2', '--dry-run'])
        self.assertEqual(code, 0)
        factory.assert_not_called()
        self.assertEqual(len(stdout.getvalue().splitlines()), 2)
        self.assertEqual(list(self.directory.iterdir()), [self.source])

    def test_dry_run_style_is_stable_across_selection_and_never_constructs_client(self):
        rows = [{**deepcopy(self.component), 'prompt_id': f'{index:04d}',
                 'group_id': 0, 'cluster_id': 0} for index in range(8)]
        self.write_source(rows)

        def preview(*selection):
            stdout = io.StringIO()
            with patch('openai.OpenAI', side_effect=AssertionError('API client forbidden')), redirect_stdout(stdout):
                code = main(['--input', str(self.source), '--dry-run', *selection])
            self.assertEqual(code, 0)
            return [json.loads(line) for line in stdout.getvalue().splitlines()]

        complete = preview()
        self.assertEqual(preview('--id', '0003'), [complete[3]])
        self.assertEqual(preview('--limit', '4'), complete[:4])
        self.assertEqual(preview('--offset', '4', '--limit', '4'), complete[4:8])
        self.assertEqual(preview('--group-id', '0'), complete)
        self.assertEqual(preview('--cluster-id', '0'), complete)
        self.assertNotEqual(preview('--style-seed', '1'), complete)
        self.assertTrue(all('\nWriting preferences:\n' in row['body']['instructions']
                            for row in complete))
        self.assertEqual(list(self.directory.iterdir()), [self.source])

    def test_invalid_component_and_existing_output_fail_before_upload(self):
        invalid = deepcopy(self.components[1])
        invalid['relations'][0]['object'] = 99
        self.write_source([self.component, invalid])
        api = BatchAPI()
        self.assertEqual(self.run_cli(api)[0], 1)
        api.factory.assert_not_called()
        self.write_source(self.components)
        self.target.write_text('existing result', encoding='utf-8')
        self.assertEqual(self.run_cli(api)[0], 1)
        api.factory.assert_not_called()
        self.assertEqual(self.target.read_text(), 'existing result')

    def test_batch_size_and_file_size_limits_fail_locally(self):
        with patch('stage2_prompt_generation.batch.MAX_REQUESTS', 1):
            with self.assertRaises(ValueError):
                build_batch_requests(self.components, 'test')
        with patch('stage2_prompt_generation.batch.MAX_FILE_BYTES', 1):
            with self.assertRaises(ValueError):
                batch_jsonl([{'body': 'too large'}])
        with self.assertRaises(ValueError):
            build_batch_requests([self.component, self.component], 'test')

    def test_metadata_rejected_before_paid_submission(self):
        for field, value in [('group_id', None), ('group_id', True), ('cluster_id', None),
                             ('cluster_id', -1), ('is_query', 'false')]:
            with self.subTest(field=field, value=value):
                rows = deepcopy(self.components)
                rows[0][field] = value
                self.write_source(rows)
                api = BatchAPI()
                self.assertEqual(self.run_cli(api)[0], 1)
                api.factory.assert_not_called()
                self.assertFalse(self.target.exists())

    def test_atomic_save_rejects_overwrite_and_invalid_result_without_partial_file(self):
        results = [{'input_component': c, 'output': 'A prompt.\nTwo sentences.'}
                   for c in self.components]
        save_results(self.target, results, 'test')
        original = self.target.read_bytes()
        with self.assertRaises(FileExistsError):
            save_results(self.target, results, 'test')
        self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), ['components.jsonl', 'result.jsonl'])
        self.target.unlink()
        results[1]['input_component'] = {**self.components[1], 'is_query': None}
        with self.assertRaises(ValueError):
            save_results(self.target, results, 'test')
        self.assertFalse(self.target.exists())

    def test_append_rejects_saved_ids_or_malformed_output_before_upload(self):
        save_results(self.target, [{'input_component': self.component, 'output': 'Existing.'}], 'test')
        api = BatchAPI()
        code, stderr = self.run_cli(api, '--append')
        self.assertEqual(code, 1)
        self.assertIn('already saved', stderr)
        api.factory.assert_not_called()
        valid = self.target.read_text()
        for invalid in ['not json', valid + valid,
                        json.dumps({'id': '0000', 'prompt': 'Missing metadata'}) + '\n']:
            self.target.write_text(invalid)
            code, stderr = self.run_cli(api, '--offset', '1', '--limit', '1', '--append')
            self.assertEqual(code, 1, stderr)
            api.factory.assert_not_called()
            self.assertEqual(self.target.read_text(), invalid)

    def test_interrupted_append_and_concurrent_collector_preserve_original(self):
        results = [{'input_component': c, 'output': f'Prompt {c["prompt_id"]}'}
                   for c in self.components]
        save_results(self.target, results[:1], 'test', append=True)
        original = self.target.read_bytes()
        with patch('stage2_prompt_generation.runner.os.replace', side_effect=OSError('Disk full')):
            with self.assertRaisesRegex(OSError, 'Disk full'):
                save_results(self.target, results[1:], 'test', append=True)
        self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), ['components.jsonl', 'result.jsonl'])
        lock = self.target.with_name(f'.{self.target.name}.lock')
        lock.write_text('another collector')
        with self.assertRaisesRegex(FileExistsError, 'Append is locked'):
            save_results(self.target, results[1:], 'test', append=True)
        self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(lock.read_text(), 'another collector')

    def test_default_stage2_settings_and_versioned_paths(self):
        from .config import default_config, input_path
        from .runner import default_output_path
        config = default_config()
        self.assertEqual(input_path(config, 'train').name, 'train_components_v5.jsonl')
        self.assertEqual(default_output_path('test').name, 'test_prompts_v5.jsonl')
        api = BatchAPI()
        code, stderr = self.run_cli(api)
        self.assertEqual(code, 0, stderr)
        for row in api.requests:
            self.assertEqual(row['body']['model'], 'gpt-6.1-terra')
            self.assertEqual(row['body']['reasoning'], {'effort': 'none'})
            self.assertEqual(row['body']['max_output_tokens'], 4096)
            self.assertEqual(row['body']['temperature'], 1.4)
        self.assertEqual(api.calls, [('POST', '/v1/files'), ('POST', '/v1/batches')])
        self.assertFalse(self.target.exists())

    def test_config_paths_and_train_split_are_restored_on_collect(self):
        from .config import default_config
        config = default_config()
        config.update(input_dir=str(self.directory), output_dir=str(self.directory/'output'),
                      component_id='preview', model='gpt-6-luna', reasoning_effort='medium',
                      temperature=None)
        config_path = self.directory/'config.yaml'
        config_path.write_text(json.dumps(config))
        train = [{k: v for k, v in c.items() if k not in ('is_query', 'cluster_id')}
                 for c in self.components]
        (self.directory/'train_components_preview.jsonl').write_text(
            ''.join(json.dumps(c)+'\n' for c in train))
        api = BatchAPI()
        with patch('openai.OpenAI', api.factory), redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--config', str(config_path), '--split', 'train']), 0)
            self.assertFalse((self.directory/'output').exists())
            self.assertEqual(main(['--config', str(config_path), '--batch-id', 'batch_test']), 0)
        saved = self.directory/'output'/'train_prompts_preview.jsonl'
        self.assertEqual([json.loads(line) for line in saved.read_text().splitlines()],
                         self.expected_rows(train, 'train'))
        self.assertTrue(all(r['body']['model'] == 'gpt-6-luna' and
                            r['body']['reasoning'] == {'effort': 'medium'} for r in api.requests))
        self.assertEqual(api.calls.count(('POST', '/v1/batches')), 1)

    def test_temperature_cli_override_and_omission(self):
        for value, expected in [('1.2', 1.2), ('0', 0.0), ('none', None), ('null', None)]:
            with self.subTest(value=value):
                api = BatchAPI()
                code, stderr = self.run_cli(api, '--temperature', value)
                self.assertEqual(code, 0, stderr)
                for row in api.requests:
                    if expected is None:
                        self.assertNotIn('temperature', row['body'])
                    else:
                        self.assertEqual(row['body']['temperature'], expected)

    def test_invalid_sampling_fails_before_upload(self):
        for args in [('--reasoning-effort', 'low'),
                     *[('--temperature', value) for value in ('-1', '2.1', 'nan', 'inf')]]:
            with self.subTest(args=args):
                api = BatchAPI()
                code, stderr = self.run_cli(api, *args)
                self.assertEqual(code, 1, stderr)
                api.factory.assert_not_called()

    def test_invalid_output_limits_fail_before_request(self):
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                build_request({}, max_output_tokens=limit)


if __name__ == '__main__':
    unittest.main()
