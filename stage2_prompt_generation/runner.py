"""Component selection, Responses request settings, and minimal JSONL output."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
import tempfile

from stage2_prompt_generation.config import default_config, output_path, project_path, validate_sampling
from stage1_components.files import valid_prompt_id


MODEL = 'gpt-6.1-terra'
REASONING_EFFORT = 'none'
TEMPERATURE = 1.4
# Reasoning and the final prompt share this output budget.
MAX_OUTPUT_TOKENS = 4096
OUTPUT_DIR = project_path('outputs_stage2')


def _read_components(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        component = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(component, dict):
            raise ValueError('The input JSON must contain exactly one component object')
        yield component
        return
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                component = json.loads(line)
            except ValueError as exc:
                raise ValueError(f'{path}:{number}: invalid JSON') from exc
            if not isinstance(component, dict):
                raise ValueError(f'{path}:{number}: expected a component object')
            yield component


def load_components(path, prompt_ids=None, limit=None, *, group_id=None, cluster_id=None, offset=0):
    """Select rows, an offset/limit slice, IDs, or a complete group/cluster."""
    if sum(value is not None for value in (prompt_ids, limit, group_id, cluster_id)) > 1:
        raise ValueError('Use only one of --id, --limit, --group-id, or --cluster-id')
    if type(offset) is not int or offset < 0:
        raise ValueError('--offset must be a nonnegative integer')
    if offset and any(value is not None for value in (prompt_ids, group_id, cluster_id)):
        raise ValueError('--offset can only be used with --limit or without another selection')
    grouping = None
    for field, value in (('group_id', group_id), ('cluster_id', cluster_id)):
        if value is not None:
            if type(value) is not int or value < 0:
                raise ValueError(f'--{field.replace("_", "-")} must be a nonnegative integer')
            grouping = (field, value)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('--limit must be a positive integer')
    if prompt_ids is not None:
        if (not prompt_ids or any(not valid_prompt_id(value) for value in prompt_ids)
                or len(set(prompt_ids)) != len(prompt_ids)):
            raise ValueError('--id values must be unique canonical numeric IDs, e.g. 0000 or 15999')
    wanted = set(prompt_ids) if prompt_ids is not None else None
    components = []
    seen = set()
    for index, component in enumerate(_read_components(path)):
        identifier = component.get('prompt_id')
        if not valid_prompt_id(identifier):
            raise ValueError(f'{path}: invalid prompt_id: {identifier!r}')
        if index < offset:
            continue
        if grouping is not None:
            field, value = grouping
            if type(component.get(field)) is not int or component[field] < 0:
                raise ValueError(f'{path}: component {identifier} has no valid {field}')
            if component[field] != value:
                continue
        if wanted is not None and identifier not in wanted:
            continue
        if identifier in seen:
            raise ValueError(f'{path}: duplicate prompt_id: {identifier}')
        seen.add(identifier)
        components.append(component)
        if ((limit is not None and len(components) == limit)
                or (wanted is not None and seen == wanted)):
            break
    if wanted is not None and wanted != seen:
        raise ValueError(f'{path}: missing component IDs: {sorted(wanted - seen)}')
    if not components:
        detail = f' for {grouping[0]}={grouping[1]}' if grouping is not None else ''
        raise ValueError(f'{path}: no components found{detail}')
    return components


def load_component(path, prompt_id=None):
    """Compatibility helper for selecting one component without API calls."""
    return load_components(path, [prompt_id] if prompt_id is not None else None,
                           limit=1 if prompt_id is None else None)[0]


def build_request(prompt, *, max_output_tokens=MAX_OUTPUT_TOKENS, model=MODEL,
                  reasoning_effort=REASONING_EFFORT, temperature=TEMPERATURE):
    if type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError('max_output_tokens must be a positive integer')
    validate_sampling(reasoning_effort, temperature)
    request = {
        'model': model,
        **prompt,
        'reasoning': {'effort': reasoning_effort},
        'max_output_tokens': max_output_tokens,
        'store': False,
    }
    if temperature is not None:
        request['temperature'] = temperature
    return request


def default_output_path(split, config=None):
    return output_path(default_config() if config is None else config, split)


def component_metadata(component, split):
    """Validate the required Stage1 metadata before submitting paid requests."""
    if split not in ('train', 'test'):
        raise ValueError('split must be train or test')
    if not valid_prompt_id(component.get('prompt_id')):
        raise ValueError('Invalid prompt_id')
    metadata = {}
    for field in ('group_id', 'cluster_id') if split == 'test' else ('group_id',):
        value = component.get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f'{component["prompt_id"]}: {field} must be a nonnegative integer')
        metadata[field] = value
    if split == 'test':
        if type(component.get('is_query')) is not bool:
            raise ValueError(f'{component["prompt_id"]}: is_query must be a boolean')
        metadata['is_query'] = component['is_query']
    elif 'is_query' in component:
        raise ValueError('is_query applies only to test candidates')
    return metadata


def result_row(result, split):
    component, prompt = result['input_component'], result['output']
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('prompt must be a nonempty string')
    return {'id': component['prompt_id'], 'prompt': prompt,
            **component_metadata(component, split)}


def load_saved_rows(path, split):
    """Reject malformed, duplicate, or wrong-split rows before merging."""
    fields = {'id', 'prompt', 'group_id'}
    if split == 'test':
        fields |= {'cluster_id', 'is_query'}
    rows, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError(f'{path}:{number}: unexpected result fields for {split}')
        component_metadata({**row, 'prompt_id': row['id']}, split)
        if not isinstance(row['prompt'], str) or not row['prompt'].strip():
            raise ValueError(f'{path}:{number}: prompt must be a nonempty string')
        if row['id'] in seen:
            raise ValueError(f'{path}:{number}: duplicate result ID {row["id"]}')
        seen.add(row['id'])
        rows.append(row)
    return rows


@contextmanager
def _append_lock(path):
    """Fail rather than let concurrent collectors lose each other's additions."""
    lock = path.with_name(f'.{path.name}.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f'Append is locked: {lock}. If a collector crashed, '
                              'remove the stale lock only after confirming it stopped.') from exc
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink()


def _write_rows(path, rows, *, replace=False):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # New outputs still refuse overwrite, including concurrent creation.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_results(path, results, split, *, append=False):
    """Atomically save or append; identical IDs are idempotent, conflicts fail."""
    rows = [result_row(result, split) for result in results]
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Results must be nonempty and have unique IDs')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not append:
        _write_rows(path, rows)
        return path
    with _append_lock(path):
        existed = path.exists()
        existing = load_saved_rows(path, split) if existed else []
        by_id = {row['id']: row for row in existing}
        additions = []
        for row in rows:
            if row['id'] in by_id:
                if by_id[row['id']] != row:
                    raise ValueError(f'Conflicting saved result for ID {row["id"]}; '
                                     'existing output was not changed')
            else:
                additions.append(row)
        if additions:
            _write_rows(path, [*existing, *additions], replace=existed)
    return path
