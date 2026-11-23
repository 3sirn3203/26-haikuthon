"""Submit component prompts to OpenAI Batch API, or collect a submitted batch."""

import argparse
import os
from pathlib import Path
import sys

from .batch import (
    TERMINAL_STATUSES, batch_jsonl, batch_split, build_batch_requests, collect_results,
    delete_input_file, restore_components, submit_batch, wait_for_batch,
)
from .runner import (
    MODEL, REASONING_EFFORT, default_output_path, load_components, load_saved_rows,
    project_path, save_results,
)
from .config import DEFAULT_CONFIG, input_path, load_config, validate_config


def _report_status(batch):
    counts = batch.request_counts
    detail = (f', completed={counts.completed}/{counts.total}, failed={counts.failed}'
              if counts is not None else '')
    print(f'Batch: {batch.id}, status={batch.status}{detail}', file=sys.stderr, flush=True)


def _temperature(value):
    if value.lower() in ('none', 'null'):
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('temperature must be a number or none') from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--split', choices=['train', 'test'],
                        help='default: test for submission, original split for --batch-id')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--id', dest='prompt_ids', action='append',
                           help='select a prompt_id; repeat for multiple IDs; omit to select all rows')
    selection.add_argument('--limit', type=int, help='submit up to N components after --offset')
    selection.add_argument('--group-id', type=int, help='submit every component in one small group')
    selection.add_argument('--cluster-id', type=int, help='submit every component in one large test cluster')
    parser.add_argument('--offset', type=int,
                        help='skip N input components before --limit (row offset, not an ID)')
    parser.add_argument('--input', type=Path,
                        help='component .json or Stage1 .jsonl; paths are relative to the project root')
    parser.add_argument('--output', type=Path,
                        help='new result JSONL path; default config uses outputs_stage2/<split>_prompts.jsonl')
    parser.add_argument('--append', action='store_true',
                        help='merge collected results into --output; skip identical IDs, reject conflicts')
    parser.add_argument('--max-output-tokens', type=int, help='override config output token budget')
    parser.add_argument('--model', help=f'override config model (default config: {MODEL})')
    parser.add_argument('--reasoning-effort',
                        choices=['none', 'low', 'medium', 'high', 'xhigh', 'max'],
                        help=f'model-supported reasoning effort (default: {REASONING_EFFORT})')
    parser.add_argument('--temperature', type=_temperature, default=argparse.SUPPRESS,
                        help='override config temperature (0–2); use none to omit it')
    parser.add_argument('--style-seed', type=int,
                        help='deterministic writing-style allocation seed (default: 0)')
    parser.add_argument('--batch-id', help='retrieve/collect an existing batch without resubmission')
    parser.add_argument('--wait', action='store_true', help='poll until the batch reaches a terminal status')
    parser.add_argument('--poll-interval', type=float, default=30,
                        help='seconds between status checks, from 1 to 60 (default: 30)')
    parser.add_argument('--dry-run', action='store_true',
                        help='print Batch input JSONL without calling OpenAI or saving files')
    args = parser.parse_args(argv)
    if args.batch_id and (any(value is not None for value in (
            args.prompt_ids, args.limit, args.group_id, args.cluster_id, args.offset)) or args.dry_run):
        parser.error('--batch-id cannot be combined with selection options or --dry-run')
    if args.offset is not None and any(value is not None for value in (
            args.prompt_ids, args.group_id, args.cluster_id)):
        parser.error('--offset can only be combined with --limit')
    if not 1 <= args.poll_interval <= 60:
        parser.error('--poll-interval must be between 1 and 60 seconds')

    try:
        config = load_config(args.config)
        for key in ('model', 'reasoning_effort', 'max_output_tokens', 'style_seed'):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        if hasattr(args, 'temperature'):
            config['temperature'] = args.temperature
        config = validate_config(config)
        split = args.split or 'test'
        components = None
        if not args.batch_id:
            source = project_path(args.input) if args.input else input_path(config, split)
            components = load_components(source, args.prompt_ids, args.limit,
                                         group_id=args.group_id, cluster_id=args.cluster_id,
                                         offset=args.offset if args.offset is not None else 0)
            # Allocate from the full source so --id/--limit previews use exactly
            # the same writing preferences as a complete group or split run.
            style_components = load_components(source)
            data = batch_jsonl(build_batch_requests(
                components, split, max_output_tokens=config['max_output_tokens'],
                style_components=style_components, style_seed=config['style_seed'],
                model=config['model'], reasoning_effort=config['reasoning_effort'],
                temperature=config['temperature'],
            ))
            if args.dry_run:
                sys.stdout.write(data.decode('utf-8'))
                return 0
            print(f'Selected {len(components)} components from {source}', file=sys.stderr)

        output_path = (project_path(args.output) if args.output else
                       default_output_path(split, config) if not args.batch_id else None)
        if output_path is not None and output_path.exists():
            if not args.append:
                raise FileExistsError(f'Result already exists; choose a new --output path '
                                      f'or use --append: {output_path}')
            if not args.batch_id:
                existing_ids = {row['id'] for row in load_saved_rows(output_path, split)}
                overlap = existing_ids & {c['prompt_id'] for c in components}
                if overlap:
                    raise ValueError(f'{len(overlap)} selected IDs are already saved in {output_path}; '
                                     'choose an uncollected range instead of resubmitting')

        from dotenv import load_dotenv
        from openai import APIError, OpenAI

        load_dotenv(project_path('.env'), override=False)
        if not os.environ.get('OPENAI_API_KEY', '').strip():
            raise ValueError('Set OPENAI_API_KEY in the environment or the project root .env')
        with OpenAI(max_retries=0, timeout=120.0) as client:
            try:
                if args.batch_id:
                    batch = client.batches.retrieve(args.batch_id)
                else:
                    batch = submit_batch(client, components, split, data)
                    del data  # Release the request JSONL as soon as submission succeeds.
                _report_status(batch)
                split = batch_split(batch, args.split)
                if output_path is None:
                    output_path = default_output_path(split, config)
                    if output_path.exists() and not args.append:
                        raise FileExistsError(f'Result already exists; choose a new --output path '
                                              f'or use --append: {output_path}')
                if args.wait:
                    batch = wait_for_batch(client, batch, args.poll_interval, _report_status)
                if batch.status not in TERMINAL_STATUSES:
                    print(f'Run again with --batch-id {batch.id} to collect; add --wait to poll. '
                          'For custom inputs, reuse the same --input file.', file=sys.stderr)
                    return 0
                if components is None:
                    source = project_path(args.input) if args.input else input_path(config, split)
                    components = restore_components(client, batch, source, split)
                results, failures = collect_results(client, batch, components, split)
                if results:
                    saved = save_results(output_path, results, split, append=args.append)
                    verb = 'Merged' if args.append else 'Saved'
                    print(f'{verb}: {saved} ({len(results)} batch results)', file=sys.stderr)
                # Persist results before removing the remote input. A failed
                # save leaves the upload available for a later collection.
                delete_input_file(client, batch)
                print(f'Input file cleaned up: {batch.input_file_id}', file=sys.stderr)
            except APIError as exc:
                status = getattr(exc, 'status_code', None)
                code = getattr(exc, 'code', None)
                raise RuntimeError(f'OpenAI request failed ({type(exc).__name__}, '
                                   f'status={status}, code={code})') from exc

        for identifier, error in failures.items():
            print(f'{identifier}: {error}', file=sys.stderr)
        if batch.status == 'failed' and batch.errors:
            for error in batch.errors.data or []:
                print(f'Batch validation error: code={error.code}, line={error.line}', file=sys.stderr)
        print(f'Batch {batch.id}: status={batch.status}, saved={len(results)}, '
              f'failed={len(failures)}', file=sys.stderr)
        return 1 if failures or batch.status != 'completed' else 0
    except KeyboardInterrupt:
        print('Interrupted. If a batch ID was printed, the batch continues; '
              'use --batch-id to collect later.', file=sys.stderr)
        return 130
    except ImportError as exc:
        print(f'Missing dependency: {exc}. From the project root, run '
              'python -m pip install -r requirements.txt.', file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, AttributeError) as exc:
        print(f'Stage2 Batch failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
