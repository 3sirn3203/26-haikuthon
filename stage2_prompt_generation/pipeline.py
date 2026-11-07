"""Generate every prompt first, then verify each once with the source critic.

The CLI validates Stage1 inputs and manages the server. No inference, server
startup, or output writes happen on import. Completed runs use exactly two LLM
requests per input row, irrespective of verification results.
"""

from collections import Counter
from itertools import islice
import json
from pathlib import Path

from .client import CRITIC_SCHEMA
from .config import default_config, generation_settings
from .semantics import PRESENTATIONS, english_scene, parse_json_output


OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'outputs_stage2'
OUTPUT_NAMES = {'train': 'train_prompts.jsonl', 'test': 'test_prompts.jsonl'}


def load_instructions():
    directory = Path(__file__).resolve().parent / 'prompts'
    return {
        name: (directory / f'{name}_train_scene.txt').read_text(encoding='utf-8')
        for name in ['realize', 'critic']
    }


def generation_messages(scene, vocab, instructions):
    payload = {
        'scene': english_scene(scene, vocab),
        'presentation': PRESENTATIONS[scene['presentation_id'] % len(PRESENTATIONS)],
    }
    return [
        {'role': 'system', 'content': instructions},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
    ]


def _batches(values, size):
    if type(size) is not int or size < 1:
        raise ValueError('batch_size must be a positive integer')
    source = iter(values)
    while batch := list(islice(source, size)):
        yield batch


def _generate_once(client, conversations, settings, seed):
    replies = client.generate(conversations, settings, seed)
    if len(replies) != len(conversations) or any(not isinstance(reply, str) for reply in replies):
        raise ValueError('Expected one text response per LLM request')
    return replies


def _prompt_text(reply):
    """Unwrap structured output without filtering, repairing, or retrying it.

    Preserve an unparseable response verbatim so the later critic can inspect it;
    malformed/empty responses must not silently remove an input from the run.
    """
    try:
        parsed = parse_json_output(reply)
        if set(parsed) == {'prompt_text'} and isinstance(parsed['prompt_text'], str):
            return parsed['prompt_text']
    except (ValueError, TypeError):
        pass
    return reply


def _critic_ok(reply):
    try:
        parsed = parse_json_output(reply)
        return set(parsed) == {'issues'} and isinstance(parsed['issues'], list) and not parsed['issues']
    except (ValueError, TypeError):
        return False


def generate_prompt(identifier, scene, vocab, client, *, config=None,
                    instructions=None):
    """Generate one {id, prompt} without verification, retries, or output writes."""
    config = default_config() if config is None else config
    instructions = load_instructions() if instructions is None else instructions
    reply, = _generate_once(client, [generation_messages(scene, vocab, instructions['realize'])],
                            generation_settings(config['generation']), config['seed'])
    return {'id': identifier, 'prompt': _prompt_text(reply)}


def _write_prompts(split, cases, vocab, client, config, instructions, offset):
    settings = generation_settings(config['generation'])
    generated = 0
    # Line buffering makes each completed row visible even on the network FS.
    with (OUTPUT_DIR / OUTPUT_NAMES[split]).open('w', encoding='utf-8', buffering=1) as stream:
        for batch in _batches(cases, settings['batch_size']):
            replies = _generate_once(client, [
                generation_messages(scene, vocab, instructions['realize']) for _, scene in batch
            ], settings, config['seed'] + offset + generated)
            for (identifier, _), reply in zip(batch, replies):
                stream.write(json.dumps({'id': identifier, 'prompt': _prompt_text(reply)}, ensure_ascii=False) + '\n')
            generated += len(batch)
            calls = getattr(client, 'calls', None)
            detail = ''
            if calls:
                call = calls[-1]
                detail = (f', finish_reasons={dict(Counter(call["finish_reasons"]))}'
                          f', completion_tokens={call["completion_tokens"]}')
            print(f'{split}: generated={generated}/{len(cases)}{detail}', flush=True)
    return generated


def _verify_prompts(split, cases, vocab, client, config, instructions, offset):
    settings = {**config['critic'], 'schema_name': 'critic', 'json_schema': CRITIC_SCHEMA}
    total = passed = 0
    prompt_path = OUTPUT_DIR / OUTPUT_NAMES[split]
    verification_path = OUTPUT_DIR / f'{split}_verification.jsonl'
    with prompt_path.open(encoding='utf-8') as prompts, \
            verification_path.open('w', encoding='utf-8', buffering=1) as stream:
        for batch in _batches(cases, settings['batch_size']):
            conversations = []
            for identifier, scene in batch:
                row = json.loads(next(prompts, 'null'))
                if (not isinstance(row, dict) or set(row) != {'id', 'prompt'}
                        or row['id'] != identifier or not isinstance(row['prompt'], str)):
                    raise ValueError(f'{prompt_path}: expected prompt for id {identifier}')
                conversations.append([
                    {'role': 'system', 'content': instructions['critic']},
                    {'role': 'user', 'content': json.dumps({
                        'source_scene': english_scene(scene, vocab), 'prompt': row['prompt'],
                    }, ensure_ascii=False)},
                ])
            replies = _generate_once(client, conversations, settings,
                                     config['seed'] + 2000000 + offset + total)
            for (identifier, _), reply in zip(batch, replies):
                ok = _critic_ok(reply)
                stream.write(json.dumps({'id': identifier, 'ok': ok, 'reply': reply}, ensure_ascii=False) + '\n')
                passed += int(ok)
            total += len(batch)
            print(f'{split}: verified={total}/{len(cases)}, passed={passed}, failed={total - passed}', flush=True)
        if next(prompts, None) is not None:
            raise ValueError(f'{prompt_path}: unexpected extra prompt rows')
    summary = {'total': total, 'passed': passed, 'failed': total - passed,
               'pass_rate': passed / total if total else 0.0}
    (OUTPUT_DIR / f'{split}_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
    )
    return summary


def _run_splits(splits, vocab, client, config, instructions):
    prepared = {}
    for split, cases in splits.items():
        if split not in OUTPUT_NAMES:
            raise ValueError('split must be train or test')
        prepared[split] = list(cases)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # A new run invalidates previous outputs for the selected splits, including
    # verification summaries which would otherwise describe the old prompts.
    for split in prepared:
        for name in [OUTPUT_NAMES[split], f'{split}_verification.jsonl', f'{split}_summary.json']:
            (OUTPUT_DIR / name).unlink(missing_ok=True)

    expected = sum(len(cases) for cases in prepared.values())
    print(f'Stage2: generate {expected} prompts, then verify {expected} prompts; '
          f'LLM requests={2 * expected}, retries=0', flush=True)
    offset = 0
    for split, cases in prepared.items():
        offset += _write_prompts(split, cases, vocab, client, config, instructions, offset)

    # Both train and test files are complete and closed before any critic call.
    print('Stage2: all prompts saved; starting one-pass critic verification.', flush=True)
    summaries = {}
    offset = 0
    for split, cases in prepared.items():
        summaries[split] = _verify_prompts(split, cases, vocab, client, config, instructions, offset)
        offset += len(cases)
    return summaries


def generate_split(split, cases, vocab, client, *, config=None, instructions=None):
    """Generate a whole split, verify it once, and return its summary."""
    config = default_config() if config is None else config
    instructions = load_instructions() if instructions is None else instructions
    return _run_splits({split: cases}, vocab, client, config, instructions)[split]


def generate_train_test(train_cases, test_cases, vocab, client, *,
                        config=None, instructions=None):
    """Save ALL train/test prompts before verifying either split; return summaries.

    Preserve every input ID and its order, including prompts that fail the critic
    or duplicate another prompt. Verification never rewrites the prompt files.
    """
    config = default_config() if config is None else config
    instructions = load_instructions() if instructions is None else instructions
    return _run_splits({'train': train_cases, 'test': test_cases}, vocab, client, config, instructions)
