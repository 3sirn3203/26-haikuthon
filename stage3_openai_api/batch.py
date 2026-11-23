"""Image batches addressed by OpenAI Batch ID; only PNG/prompt pairs saved locally."""
from copy import deepcopy
import hashlib
import json
import time

from stage1_components.files import valid_prompt_id
from stage3_image_generation.pipeline import SplitOutput, output_lock, read_prompts, unique_object

from .config import render_settings, validate_config
from .images import decode_image, saved_image_config


ENDPOINT = '/v1/images/generations'
RUNNER = 'stage3_openai_api_v2'
TERMINAL = {'completed', 'failed', 'expired', 'cancelled'}
MAX_INPUT_BYTES = 200_000_000


def request_body(row, config):
    g = config['generation']
    if len(row['prompt']) > 32_000:
        raise ValueError(f'{row["id"]}: prompt exceeds 32000 characters')
    body = {
        'model': config['model'], 'prompt': row['prompt'], 'n': 1,
        'quality': g['quality'], 'size': f'{g["width"]}x{g["height"]}',
        'background': g['background'], 'output_format': g['output_format'],
        'moderation': g['moderation'],
    }
    if g['output_compression'] is not None:
        body['output_compression'] = g['output_compression']
    return body


def build_jsonl(rows, split, config):
    if not 1 <= len(rows) <= config['batch']['max_requests']:
        raise ValueError('Selected pending requests exceed batch.max_requests; use --offset/--limit')
    data = ''.join(json.dumps({
        'custom_id': f'{split}-{row["id"]}', 'method': 'POST', 'url': ENDPOINT,
        'body': request_body(row, config),
    }, ensure_ascii=False) + '\n' for row in rows).encode('utf-8')
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError('Batch input exceeds 200 MB; use a smaller --limit')
    return data


def select_rows(rows, *, offset=0, limit=None, ids=None):
    if type(offset) is not int or offset < 0:
        raise ValueError('--offset must be a nonnegative integer')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('--limit must be positive')
    if ids is not None:
        if offset or limit is not None:
            raise ValueError('--id cannot be combined with --offset/--limit')
        if not ids or len(set(ids)) != len(ids) or any(not valid_prompt_id(i) for i in ids):
            raise ValueError('--id values must be unique canonical numeric IDs')
        wanted = set(ids)
        selected = [row for row in rows if row['id'] in wanted]
        if {row['id'] for row in selected} != wanted:
            raise ValueError('Some requested IDs are absent from the input')
    else:
        selected = rows[offset:None if limit is None else offset+limit]
    if not selected:
        raise ValueError('No prompts selected')
    return selected


def pending_rows(rows, split, config, output):
    layout = SplitOutput(output, split)
    image_config = saved_image_config(config)
    return [row for row in rows if not layout.completed(row, image_config)]


def selection_digest(identifiers):
    data = json.dumps(sorted(identifiers), separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def submit(client, rows, split, source, output, config):
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        pending = pending_rows(rows, split, config, output)
        if not pending:
            print('All selected images are already saved; no API request submitted.', flush=True)
            return None
        data = build_jsonl(pending, split, config)
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        original_rows = select_rows(read_prompts(source), ids=[row['id'] for row in pending])
        if original_rows != pending or hashlib.sha256(source.read_bytes()).hexdigest() != source_digest:
            raise ValueError('Source prompts changed while preparing submission; retry after resolving edits')
        # Remote-only settings/hashes; no UUID, local job record, or prompt copy.
        metadata = {
            'runner': RUNNER, 'split': split,
            'source_sha256': source_digest,
            'selection_sha256': selection_digest(r['id'] for r in pending),
            'render': json.dumps(render_settings(config), separators=(',', ':')),
        }
        if any(len(value) > 512 for value in metadata.values()):
            raise ValueError('Batch metadata value exceeds 512 characters')
        uploaded = client.files.create(file=('image_requests.jsonl', data, 'application/jsonl'), purpose='batch')
        print(f'Uploaded requests: {uploaded.id}', flush=True)
        batch = client.batches.create(
            input_file_id=uploaded.id, endpoint=ENDPOINT,
            completion_window=config['batch']['completion_window'],
            metadata=metadata,
        )
        print(f'Created batch: {batch.id}', flush=True)
        return batch


def check_batch(batch):
    meta = batch.metadata or {}
    if (batch.endpoint != ENDPOINT or meta.get('runner') != RUNNER
            or meta.get('split') not in ('train', 'test')
            or not isinstance(meta.get('render'), str)):
        raise ValueError('This is not a Batch submitted by stage3_openai_api')
    for key in ('source_sha256', 'selection_sha256'):
        value = meta.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Invalid Batch {key}')
    return meta['split']


def report_status(batch):
    counts = batch.request_counts
    suffix = (f', completed={counts.completed}/{counts.total}, failed={counts.failed}'
              if counts is not None else '')
    print(f'Batch: {batch.id}, status={batch.status}{suffix}', flush=True)
    if batch.errors:
        for error in batch.errors.data or []:
            print(f'Batch error: {error.code}: {error.message}', flush=True)


def wait_for_batch(client, batch, interval):
    while batch.status not in TERMINAL:
        time.sleep(interval)
        batch = client.batches.retrieve(batch.id)
        check_batch(batch)
        report_status(batch)
    return batch


def result_lines(client, file_id):
    import httpx
    from openai import APIError, APITimeoutError, NotFoundError

    # Read chunks so even a large, incomplete image line shows download progress.
    # Retain at most one JSONL record plus one chunk, never the whole output file.
    started = last_report = time.monotonic()
    received = number = records = 0
    buffer = bytearray()
    phase = 'waiting for response headers'
    print(f'Downloading {file_id}: {phase}...', flush=True)

    def parse(line):
        nonlocal number, records
        number += 1
        if not line.strip():
            return None
        row = json.loads(line, object_pairs_hook=unique_object)
        if not isinstance(row, dict):
            raise ValueError(f'{file_id}:{number}: expected a result object')
        records += 1
        return row

    def report(label):
        print(f'{label} {file_id}: {received / 1_000_000:.2f} MB received, '
              f'{records} records, {time.monotonic() - started:.1f}s elapsed', flush=True)

    try:
        with client.files.with_streaming_response.content(file_id) as response:
            phase = 'reading response body'
            print(f'Download started {file_id}: headers received after '
                  f'{time.monotonic() - started:.1f}s.', flush=True)
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                received += len(chunk)
                # Previously buffered bytes have no newline, so search only the new chunk.
                search_from = len(buffer)
                buffer.extend(chunk)
                consumed = 0
                end = buffer.find(b'\n', search_from)
                while end != -1:
                    row = parse(buffer[consumed:end])
                    consumed = end + 1
                    if row is not None:
                        yield row
                    end = buffer.find(b'\n', consumed)
                if consumed:
                    del buffer[:consumed]
                if time.monotonic() - last_report >= 5:
                    report('Download progress')
                    last_report = time.monotonic()
            if buffer:
                row = parse(buffer)
                if row is not None:
                    yield row
        report('Download finished')
    except NotFoundError:
        raise  # restore_rows can recover the selection after input cleanup.
    except (APIError, httpx.HTTPError) as exc:
        detail = f': {exc}' if isinstance(exc, httpx.TransportError) and str(exc) else ''
        timeout_hint = (' For a read timeout, increase collect --timeout-seconds.'
                        if isinstance(exc, (APITimeoutError, httpx.ReadTimeout)) else '')
        raise RuntimeError(
            f'Download failed for {file_id} while {phase}: {type(exc).__name__}{detail}; '
            f'{received / 1_000_000:.2f} MB received, {records} records read. '
            'Saved images are retained. Retry collect with the same Batch ID; '
            f'do not submit again.{timeout_hint}'
        ) from exc


def restore_rows(client, batch, source, config):
    """Recover selected IDs from remote requests, or complete retained result files."""
    from openai import NotFoundError

    split = check_batch(batch)
    if hashlib.sha256(source.read_bytes()).hexdigest() != batch.metadata['source_sha256']:
        raise ValueError('Source prompts changed after submission; restore the original source file')
    by_id = {row['id']: row for row in read_prompts(source)}
    identifiers = []
    print('Restoring submitted IDs and checking the original prompts...', flush=True)

    def add_id(record):
        identifier = record.get('custom_id', '')
        prefix = f'{split}-'
        if (not isinstance(identifier, str) or not identifier.startswith(prefix)
                or identifier[len(prefix):] not in by_id):
            raise ValueError(f'Unexpected result/input ID: {identifier!r}')
        identifier = identifier[len(prefix):]
        identifiers.append(identifier)
        return identifier

    try:
        for request in result_lines(client, batch.input_file_id):
            identifier = add_id(request)
            if (request.get('method') != 'POST' or request.get('url') != ENDPOINT
                    or request.get('body') != request_body(by_id[identifier], config)):
                raise ValueError('Uploaded request does not match the original prompts/settings')
    except NotFoundError:
        identifiers.clear()
        print('Remote input was deleted; scanning result files to restore submitted IDs '
              'before saving images. Large files require another download for collection.', flush=True)
        # Re-collect after successful input cleanup. The first pass only retains
        # IDs, verifying the entire selection before any image is written.
        for file_id in (batch.output_file_id, batch.error_file_id):
            if file_id:
                for record in result_lines(client, file_id):
                    add_id(record)
    if (not identifiers or len(set(identifiers)) != len(identifiers)
            or selection_digest(identifiers) != batch.metadata['selection_sha256']):
        raise ValueError('Remote files do not identify the full original Batch selection')
    if hashlib.sha256(source.read_bytes()).hexdigest() != batch.metadata['source_sha256']:
        raise ValueError('Source prompts changed while restoring the Batch')
    selected = set(identifiers)
    return [row for identifier, row in by_id.items() if identifier in selected]


def collect(client, batch, source, output, config):
    split = check_batch(batch)
    if batch.status not in TERMINAL:
        raise ValueError(f'Batch is still {batch.status}; collect later or use --wait')
    settings = json.loads(batch.metadata['render'], object_pairs_hook=unique_object)
    if not isinstance(settings, dict) or set(settings) != {'model', 'generation', 'save_size'}:
        raise ValueError('Invalid Batch render settings')
    config = validate_config({**deepcopy(config), **settings})
    rows = restore_rows(client, batch, source, config)
    expected = {f'{split}-{row["id"]}': row for row in rows}
    print(f'Collecting {len(expected)} requests into {output / split}', flush=True)
    failures, seen, saved, skipped = {}, set(), 0, 0
    usage = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        layout = SplitOutput(output, split)
        image_config = saved_image_config(config)
        for file_id in (batch.output_file_id, batch.error_file_id):
            if not file_id:
                continue
            for result in result_lines(client, file_id):
                identifier = result.get('custom_id')
                if not isinstance(identifier, str) or identifier not in expected or identifier in seen:
                    raise ValueError(f'Unexpected or duplicate result ID: {identifier!r}')
                seen.add(identifier)
                row = expected[identifier]
                response = result.get('response') or {}
                if not isinstance(response, dict):
                    failures[identifier] = 'invalid_response'
                    continue
                body = response.get('body') or {}
                if not isinstance(body, dict):
                    failures[identifier] = 'invalid_response_body'
                    continue
                if result.get('error') or response.get('status_code') != 200 or body.get('error'):
                    error = result.get('error') or body.get('error') or {}
                    failures[identifier] = ((error.get('code') if isinstance(error, dict) else 'request_error')
                                            or f'HTTP {response.get("status_code")}')
                    continue
                for key in usage:
                    value = (body.get('usage') or {}).get(key)
                    if type(value) is int and value >= 0:
                        usage[key] += value
                if layout.completed(row, image_config):
                    skipped += 1
                    continue
                try:
                    image = decode_image(body, config)
                except (ValueError, OSError, SyntaxError) as exc:
                    failures[identifier] = f'invalid_image: {exc}'
                    continue
                # Disk errors must abort collection and preserve the remote input;
                # do not mistake them for failed generation or resubmit paid work.
                layout.save(row, image)
                saved += 1
                print(f'Saved {split}/{row["id"]} [{len(seen)}/{len(expected)} results processed]', flush=True)
        missing = set(expected)-seen
        failures.update({key: 'missing_result' for key in sorted(missing)})
        # Missing result IDs cannot be recovered from outputs after input removal.
        # Keep the remote input in that case; nothing is recorded on local disk.
        if config['batch']['delete_input_after_collect'] and not missing:
            from openai import NotFoundError
            try:
                deleted = client.files.delete(batch.input_file_id)
            except NotFoundError:
                pass  # Re-collecting a previously collected Batch is safe.
            else:
                if not deleted.deleted:
                    raise RuntimeError('OpenAI did not confirm input file deletion')
        elif missing:
            print('Remote input retained because some result IDs are missing.', flush=True)
    for identifier, error in failures.items():
        print(f'FAILED {identifier}: {error}', flush=True)
    print(f'Collected: saved={saved}, already_saved={skipped}, failed={len(failures)}; '
          f'usage={json.dumps(usage)}', flush=True)
    return 1 if failures or batch.status == 'failed' else 0
