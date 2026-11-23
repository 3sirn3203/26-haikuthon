"""Submit, inspect, cancel, or collect OpenAI image batches."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys

import yaml

from stage3_image_generation.pipeline import read_prompts

from .batch import (TERMINAL, build_jsonl, check_batch, collect, pending_rows,
                    report_status, select_rows, submit, wait_for_batch)
from .config import DEFAULT_CONFIG, load_config, project_path


def create_client(config):
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(project_path('.env'), override=False)
    if not os.environ.get('OPENAI_API_KEY', '').strip():
        raise ValueError('Set OPENAI_API_KEY in the environment or project-root .env')
    # Never retry a paid submission automatically after an ambiguous timeout.
    return OpenAI(max_retries=0, timeout=config['client']['timeout_seconds'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    submission = commands.add_parser('submit', help='Submit one Batch; defaults to train only')
    submission.add_argument('--split', choices=('train', 'test'))
    submission.add_argument('--input', type=Path)
    submission.add_argument('--offset', type=int, default=0, help='Source row offset, not prompt ID')
    selection = submission.add_mutually_exclusive_group()
    selection.add_argument('--limit', type=int)
    selection.add_argument('--id', dest='ids', action='append', help='Select an ID; repeat as needed')
    submission.add_argument('--dry-run', action='store_true', help='Print JSONL; no API, key, or file writes')
    status = commands.add_parser('status', help='Read status only; never collect or resubmit')
    collection = commands.add_parser('collect', help='Save successful results, including partial terminal batches')
    collection.add_argument('--wait', action='store_true', help='Poll until terminal, then collect')
    collection.add_argument('--input', type=Path, help='Original prompts file used for submission')
    collection.add_argument('--timeout-seconds', type=float,
                            help='Override the HTTP timeout for slow result downloads (default: config)')
    cancellation = commands.add_parser('cancel', help='Request cancellation; collect partial results afterwards')
    for command in (submission, status, collection, cancellation):
        command.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        command.add_argument('--output', type=Path, help='Output root containing the train and test splits')
    for command in (status, collection, cancellation):
        command.add_argument('--batch-id', required=True)
    args = parser.parse_args(argv)
    try:
        config = deepcopy(load_config(args.config))
        if args.command == 'collect' and args.timeout_seconds is not None:
            from .config import validate_config
            config['client']['timeout_seconds'] = args.timeout_seconds
            validate_config(config)
        output = project_path(args.output or config['output_dir']).resolve()
        if args.command == 'submit':
            split = args.split or config['default_split']
            source = (project_path(args.input) if args.input else
                      project_path(config['input_dir'])/config['prompt_files'][split]).resolve()
            rows = select_rows(read_prompts(source), offset=args.offset, limit=args.limit, ids=args.ids)
            pending = pending_rows(rows, split, config, output)
            if not pending:
                print('All selected images are already saved; no API request submitted.')
                return 0
            data = build_jsonl(pending, split, config)  # Validate before opening the client.
            print(f'Selected {len(rows)}; already saved {len(rows)-len(pending)}; '
                  f'pending {len(pending)}; JSONL {len(data)} bytes.', file=sys.stderr)
            if args.dry_run:
                sys.stdout.write(data.decode('utf-8'))
                return 0
            del data
        from openai import APIError
        try:
            with create_client(config) as client:
                if args.command == 'submit':
                    batch = submit(client, rows, split, source, output, config)
                    if batch is not None:
                        report_status(batch)
                        print('Submission only. Keep the Batch ID; collect with the original --input '
                              'and desired --output.')
                    return 0
                batch = client.batches.retrieve(args.batch_id)
                split = check_batch(batch)
                report_status(batch)
                if args.command == 'status':
                    return 0
                if args.command == 'cancel':
                    if batch.status not in TERMINAL and batch.status != 'cancelling':
                        batch = client.batches.cancel(batch.id)
                        report_status(batch)
                    return 0
                if args.wait:
                    batch = wait_for_batch(client, batch, config['batch']['poll_interval_seconds'])
                if batch.status not in TERMINAL:
                    print('Not ready to collect; try later or add --wait.')
                    return 0
                source = (project_path(args.input) if args.input else
                          project_path(config['input_dir'])/config['prompt_files'][split]).resolve()
                return collect(client, batch, source, output, config)
        except APIError as exc:
            raise RuntimeError(f'OpenAI request failed: {type(exc).__name__}, '
                               f'status={getattr(exc, "status_code", None)}, '
                               f'code={getattr(exc, "code", None)}. '
                               'No request was automatically retried. Check the printed Batch/input file ID '
                               'in OpenAI before resubmitting.') from exc
    except KeyboardInterrupt:
        print('Interrupted. Submitted batches continue remotely; use status/collect.', file=sys.stderr)
        return 130
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f'Stage3 OpenAI failed: {exc}', file=sys.stderr)
        return 1
