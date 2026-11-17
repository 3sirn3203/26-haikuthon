"""Generate resumable PNG/prompt pairs without metadata or history files."""
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import project_path, validate_config
from stage1_components.files import valid_prompt_id


SPLITS = ('train', 'test')
TEMP_NAMES = ('.image.png.tmp', '.prompt.txt.tmp')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key!r}')
        result[key] = value
    return result


def read_prompts(path):
    """Validate the entire split, keeping original IDs, order and prompt bytes."""
    rows, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=unique_object)
            schemas = ({'id', 'prompt'}, {'id', 'prompt', 'group_id'},
                       {'id', 'prompt', 'group_id', 'cluster_id', 'is_query'})
            if not isinstance(row, dict) or set(row) not in schemas:
                raise ValueError('Expected id/prompt, optionally with train or test grouping metadata')
            for field in ('group_id', 'cluster_id'):
                if field in row and (type(row[field]) is not int or row[field] < 0):
                    raise ValueError(f'{field} must be a nonnegative integer')
            if 'is_query' in row and type(row['is_query']) is not bool:
                raise ValueError('is_query must be a boolean')
            if not valid_prompt_id(row['id']):
                raise ValueError('id must be a canonical numeric string with at least four digits')
            if row['id'] in seen:
                raise ValueError(f'Duplicate id: {row["id"]}')
            if not isinstance(row['prompt'], str) or not row['prompt'].strip():
                raise ValueError('prompt must be a nonempty string')
        except ValueError as exc:
            raise ValueError(f'{path}:{number}: {exc}') from exc
        seen.add(row['id'])
        rows.append(row)
    if not rows:
        raise ValueError(f'Input has no prompts: {path}')
    return rows


def request_body(split, row, config):
    generation = config['generation']
    identifier = f'{split}:{row["id"]}'
    seed = (generation['base_seed'] + int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)) % 2**32
    return {
        'model': config['model']['id'], 'prompt': row['prompt'],
        'negative_prompt': generation['negative_prompt'], 'n': 1,
        'size': f'{generation["width"]}x{generation["height"]}',
        'num_inference_steps': generation['num_inference_steps'],
        'true_cfg_scale': generation['true_cfg_scale'], 'seed': seed,
        'generator_device': 'cuda', 'response_format': 'b64_json',
        'output_format': 'png', 'output_compression': 70,
    }


def validate_image(image_bytes, config):
    from PIL import Image
    expected = (config['generation']['width'], config['generation']['height'])
    with Image.open(io.BytesIO(image_bytes)) as image:
        if image.format != 'PNG' or image.size != expected:
            raise ValueError(f'Expected a {expected[0]}x{expected[1]} PNG')
        image.verify()
    # verify checks container integrity; load also rejects corrupt pixel data.
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()


def completed(directory, row, config):
    prompt_path, image_path = directory / 'prompt.txt', directory / 'image.png'
    if prompt_path.exists():
        try:
            saved_prompt = prompt_path.read_bytes().decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError(f'Invalid UTF-8 in {prompt_path}; use --overwrite') from exc
        if saved_prompt != row['prompt']:
            raise ValueError(f'Prompt differs for {directory}; use --overwrite')
    if not prompt_path.is_file() or not image_path.is_file():
        return False
    try:
        validate_image(image_path.read_bytes(), config)
    except (OSError, ValueError, SyntaxError):
        return False
    return True


@contextmanager
def output_lock(directory):
    """Lock the directory inode itself, leaving no .lock file behind."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Another generation process is using {directory}') from exc
        yield
    finally:
        os.close(descriptor)


def clear_temporary_files(directory):
    for name in TEMP_NAMES:
        (directory / name).unlink(missing_ok=True)


def save_pair(directory, image_bytes, prompt):
    """Publish prompt.txt last so it cannot certify an old/new mixed pair."""
    directory.mkdir(parents=True, exist_ok=True)
    image_tmp, prompt_tmp = (directory / name for name in TEMP_NAMES)
    try:
        for path, content in ((image_tmp, image_bytes), (prompt_tmp, prompt.encode('utf-8'))):
            with path.open('wb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        # Remove the old completion marker before replacing the image, even on
        # --overwrite. A crash at either replacement then leaves a pending pair.
        (directory / 'prompt.txt').unlink(missing_ok=True)
        image_tmp.replace(directory / 'image.png')
        prompt_tmp.replace(directory / 'prompt.txt')
    finally:
        clear_temporary_files(directory)


def server_url(config):
    return f'http://{config["server"]["host"]}:{config["server"]["port"]}'


def check_server(config):
    with urlopen(f'{server_url(config)}/v1/models', timeout=15) as response:
        models = json.load(response)
    if config['model']['id'] not in {model['id'] for model in models['data']}:
        raise ValueError('The expected image model is not served; start serve with the same --config')


def generate_one(split, row, config, output):
    body = request_body(split, row, config)
    client = config['client']
    started = time.monotonic()
    for attempt in range(1, client['max_attempts'] + 1):
        try:
            request = Request(f'{server_url(config)}/v1/images/generations',
                              data=json.dumps(body).encode('utf-8'),
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=client['timeout_seconds']) as response:
                result = json.load(response)
            if len(result['data']) != 1:
                raise ValueError('Expected exactly one image in the response')
            image_bytes = base64.b64decode(result['data'][0]['b64_json'], validate=True)
            validate_image(image_bytes, config)
            break
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            detail = exc.read().decode(errors='replace')[:1000] if isinstance(exc, HTTPError) else str(exc)
            terminal_http = isinstance(exc, HTTPError) and exc.code < 500 and exc.code != 429
            if attempt == client['max_attempts'] or terminal_http:
                raise RuntimeError(f'Image request failed: {detail}') from exc
            print(f'RETRY {split}/{row["id"]} ({attempt}/{client["max_attempts"]}): {detail}', flush=True)
            time.sleep(2)
    save_pair(output / split / row['id'], image_bytes, row['prompt'])
    return time.monotonic() - started


def generate_pending(split, rows, config, output):
    """Keep at most concurrency jobs in flight; drain them on the first Ctrl-C."""
    pool = ThreadPoolExecutor(max_workers=config['client']['concurrency'])
    remaining, futures = iter(rows), {}
    generated, failed, interrupted = 0, 0, False

    def submit_next():
        row = next(remaining, None)
        if row is not None:
            futures[pool.submit(generate_one, split, row, config, output)] = row

    def collect(future, row):
        nonlocal generated, failed
        if future.cancelled():
            return
        try:
            seconds = future.result()
            generated += 1
            print(f'Saved {split}/{row["id"]} ({seconds:.1f}s)', flush=True)
        except Exception as exc:
            failed += 1
            print(f'FAILED {split}/{row["id"]}: {exc}', flush=True)

    try:
        for _ in range(config['client']['concurrency']):
            submit_next()
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                collect(future, futures.pop(future))
                submit_next()
    except KeyboardInterrupt:
        interrupted = True
        print('Stopping new requests; waiting for in-flight images to be saved.', flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for future, row in futures.items():
        collect(future, row)
    return generated, failed, interrupted


def generate(config, *, split='all', limit=None, overwrite=False):
    validate_config(config)
    if split not in (*SPLITS, 'all'):
        raise ValueError('split must be train, test or all')
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    splits = SPLITS if split == 'all' else (split,)
    # Read/validate every requested split before any network call or generation.
    selected = {}
    for name in splits:
        filename = config.get('prompt_files', {}).get(name, f'{name}_prompts.jsonl')
        rows = read_prompts(project_path(config['input_dir']) / filename)
        selected[name] = rows if limit is None else rows[:limit]
    output = project_path(config['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        pending, skipped = {}, 0
        for name, rows in selected.items():
            pending[name] = []
            for row in rows:
                directory = output / name / row['id']
                clear_temporary_files(directory)
                if not overwrite and completed(directory, row, config):
                    skipped += 1
                else:
                    pending[name].append(row)
        total = sum(map(len, selected.values()))
        print(f'{total} selected; {skipped} already saved; {total - skipped} to generate; '
              f'concurrency={config["client"]["concurrency"]}', flush=True)
        if any(pending.values()):
            check_server(config)
        generated, failed, interrupted = 0, 0, False
        for name, rows in pending.items():
            if not rows:
                continue
            saved, errors, interrupted = generate_pending(name, rows, config, output)
            generated += saved
            failed += errors
            if interrupted:
                break
        print(f'Finished: {skipped + generated}/{total} saved; {generated} generated; '
              f'{failed} request/save failures.', flush=True)
        return 130 if interrupted else int(bool(failed))
