"""OpenAI Batch submission and ID-based collection; no local job/request files."""

import hashlib
import json
import time

from stage1_components.files import valid_prompt_id
from stage1_components.model import Catalog

from .prompts import STYLE_SEED, assign_style_profiles, build_prompt
from .runner import (
    build_request, component_metadata, load_components, MAX_OUTPUT_TOKENS, MODEL, REASONING_EFFORT,
    TEMPERATURE,
)


ENDPOINT = '/v1/responses'
TERMINAL_STATUSES = {'completed', 'failed', 'expired', 'cancelled'}
MAX_REQUESTS = 50_000
MAX_FILE_BYTES = 200_000_000


def custom_id(split, component):
    return f'{split}-{component["prompt_id"]}'


def components_digest(components):
    data = json.dumps(components, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def build_batch_requests(components, split, *, max_output_tokens=MAX_OUTPUT_TOKENS, style_components=None,
                         style_seed=STYLE_SEED, model=MODEL, reasoning_effort=REASONING_EFFORT,
                         temperature=TEMPERATURE):
    if not 1 <= len(components) <= MAX_REQUESTS:
        raise ValueError(f'A batch must contain 1–{MAX_REQUESTS} components; use --limit or split the input')
    catalog = Catalog()
    profiles = assign_style_profiles(
        components if style_components is None else style_components, split, seed=style_seed,
    )
    requests = []
    seen = set()
    for component in components:
        component_metadata(component, split)
        if component['prompt_id'] not in profiles:
            raise ValueError(f'Missing style allocation for {component["prompt_id"]}')
        prompt = build_prompt(component, split, catalog=catalog,
                              style_profile=profiles[component['prompt_id']])
        identifier = custom_id(split, component)
        if identifier in seen:
            raise ValueError(f'Duplicate batch custom_id: {identifier}')
        seen.add(identifier)
        requests.append({
            'custom_id': identifier, 'method': 'POST', 'url': ENDPOINT,
            'body': build_request(prompt, max_output_tokens=max_output_tokens,
                                  model=model, reasoning_effort=reasoning_effort,
                                  temperature=temperature),
        })
    return requests


def batch_jsonl(requests):
    data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in requests).encode('utf-8')
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('Batch input exceeds 200 MB; use --limit or split the input')
    return data


def submit_batch(client, components, split, data):
    uploaded = client.files.create(file=('requests.jsonl', data, 'application/jsonl'), purpose='batch')
    return client.batches.create(
        input_file_id=uploaded.id, endpoint=ENDPOINT, completion_window='24h',
        metadata={'split': split, 'components_sha256': components_digest(components)},
    )


def wait_for_batch(client, batch, poll_interval=30, on_status=None):
    if (type(poll_interval) not in (int, float) or not 1 <= poll_interval <= 60):
        raise ValueError('--poll-interval must be between 1 and 60 seconds')
    while batch.status not in TERMINAL_STATUSES:
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)
        if on_status is not None:
            on_status(batch)
    return batch


def batch_split(batch, requested_split=None):
    metadata = batch.metadata or {}
    split = metadata.get('split')
    if (batch.endpoint != ENDPOINT or split not in ('train', 'test')
            or not metadata.get('components_sha256')):
        raise ValueError('This batch was not submitted by the Stage2 Batch runner')
    if requested_split is not None and requested_split != split:
        raise ValueError(f'Batch split is {split}, but --split is {requested_split}')
    return split


def delete_input_file(client, batch):
    """Remove the uploaded request file only after the batch stops using it."""
    from openai import NotFoundError

    if batch.status not in TERMINAL_STATUSES:
        raise ValueError(f'Cannot delete the input file while batch is {batch.status}')
    try:
        deleted = client.files.delete(batch.input_file_id)
    except NotFoundError:
        return  # A previous collection already removed the uploaded file.
    if not deleted.deleted:
        raise RuntimeError(f'OpenAI did not confirm input file deletion: {batch.input_file_id}')


def _jsonl_rows(text, label):
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f'{label}:{number}: invalid JSON') from exc
        if not isinstance(row, dict):
            raise ValueError(f'{label}:{number}: expected an object')
        yield row


def _component_ids(rows, split):
    prefix = f'{split}-'
    identifiers = []
    for row in rows:
        identifier = row.get('custom_id', '')
        if (not isinstance(identifier, str) or not identifier.startswith(prefix)
                or not valid_prompt_id(identifier[len(prefix):])):
            raise ValueError(f'Unexpected batch input custom_id: {identifier!r}')
        identifiers.append(identifier[len(prefix):])
    return identifiers


def restore_components(client, batch, source, split):
    """Recover selection, including after cleanup, and reject edited sources."""
    from openai import NotFoundError

    try:
        contents = client.files.content(batch.input_file_id).text
    except NotFoundError:
        # Successful collections remove the input file. Retained result files
        # still identify the selected components, in the original source order.
        identifiers = []
        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id:
                contents = client.files.content(file_id).text
                identifiers.extend(_component_ids(_jsonl_rows(contents, file_id), split))
        if not identifiers:
            raise ValueError('Batch input was already deleted and no result IDs are available')
        components = load_components(source, identifiers)
    else:
        identifiers = _component_ids(_jsonl_rows(contents, 'batch input'), split)
        selected = load_components(source, identifiers)
        by_id = {row['prompt_id']: row for row in selected}
        components = [by_id[identifier] for identifier in identifiers]
    if components_digest(components) != batch.metadata['components_sha256']:
        raise ValueError('Input components changed after submission; use the original input file')
    return components


def _output_text(row):
    if row.get('error'):
        return None, row['error'].get('code') or 'request_error'
    response = row.get('response') or {}
    if response.get('status_code') != 200:
        return None, f'HTTP {response.get("status_code", "missing_response")}'
    body = response.get('body') or {}
    if body.get('error'):
        return None, body['error'].get('code') or 'response_error'
    if body.get('status') != 'completed':
        reason = (body.get('incomplete_details') or {}).get('reason')
        return None, reason or body.get('status') or 'missing_status'
    texts = []
    for item in body.get('output') or []:
        if item.get('type') != 'message':
            continue
        for content in item.get('content') or []:
            if content.get('type') == 'refusal':
                return None, 'refusal'
            if content.get('type') == 'output_text' and isinstance(content.get('text'), str):
                texts.append(content['text'])
    text = ''.join(texts).strip()
    return (text, None) if text else (None, 'empty_response')


def collect_results(client, batch, components, split):
    """Join unordered output/error files to inputs and retain only valid text."""
    if batch.status not in TERMINAL_STATUSES:
        raise ValueError(f'Batch is still {batch.status}')
    if components_digest(components) != batch.metadata['components_sha256']:
        raise ValueError('Input components do not match the submitted batch')
    expected = {custom_id(split, component) for component in components}
    outputs = {}
    failures = {}
    seen = set()
    for file_id in (batch.output_file_id, batch.error_file_id):
        if not file_id:
            continue
        for row in _jsonl_rows(client.files.content(file_id).text, file_id):
            identifier = row.get('custom_id')
            if identifier not in expected or identifier in seen:
                raise ValueError(f'Unexpected or duplicate batch result custom_id: {identifier!r}')
            seen.add(identifier)
            output, error = _output_text(row)
            if error is None:
                outputs[identifier] = output
            else:
                failures[identifier] = error
    for identifier in sorted(expected - seen):
        failures[identifier] = 'missing_result'
    results = [
        {'input_component': component, 'output': outputs[custom_id(split, component)]}
        for component in components if custom_id(split, component) in outputs
    ]
    return results, failures
