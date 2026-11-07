"""Generate all Stage1 prompts first, then verify each once with the critic."""

import argparse
import json
import sys

from .client import VLLMClient
from .config import DEFAULT_CONFIG, load_config
from .inputs import load_component_splits
from .pipeline import generate_split, generate_train_test, load_instructions
from .server import build_server_command, managed_server, serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    serve_parser = commands.add_parser('serve', help='run the vLLM server in the foreground')
    generate_parser = commands.add_parser('generate', help='generate all prompts, then verify each once without retries')
    for command in [serve_parser, generate_parser]:
        command.add_argument('--config', type=str, default=str(DEFAULT_CONFIG))
        command.add_argument('--vllm-executable', default='vllm', help='vLLM command or path in the prepared environment')
    serve_parser.add_argument('--dry-run', action='store_true', help='print the server command without launching it')
    generate_parser.add_argument('--split', choices=['all', 'train', 'test'], default='all')
    generate_parser.add_argument('--limit', type=int, help='generate only the first N rows of each selected split')
    generate_parser.add_argument('--external-server', action='store_true', help='connect to an existing server without starting or stopping it')
    args = parser.parse_args()
    if args.command == 'generate' and args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    try:
        config = load_config(args.config)
        if args.command == 'serve':
            if args.dry_run:
                print(json.dumps(build_server_command(config, args.vllm_executable), indent=2))
                return 0
            return serve(config, args.vllm_executable)

        splits = ['train', 'test'] if args.split == 'all' else [args.split]
        # Read/validate all selected input rows before GPU startup or output
        # truncation, including rows outside the requested preview limit.
        prepared, vocab = load_component_splits(config, splits)
        instructions = load_instructions()
        if args.limit is not None:
            prepared = {split: rows[:args.limit] for split, rows in prepared.items()}
        selected = {split: len(rows) for split, rows in prepared.items()}
        with managed_server(config, external=args.external_server, executable=args.vllm_executable):
            client = VLLMClient(config)
            if args.split == 'all':
                summaries = generate_train_test(
                    prepared['train'], prepared['test'], vocab, client,
                    config=config, instructions=instructions,
                )
            else:
                summaries = {args.split: generate_split(
                    args.split, prepared[args.split], vocab, client,
                    config=config, instructions=instructions,
                )}
        for split in splits:
            summary = summaries[split]
            print(f'{split}: saved={selected[split]}, verified={summary["total"]}, '
                  f'passed={summary["passed"]}, failed={summary["failed"]}', flush=True)
        # A negative verification is data, not an execution failure.
        return 0
    except KeyboardInterrupt:
        print('Stage2 interrupted.', file=sys.stderr)
        return 130
    except ImportError as exc:
        print(
            f'Stage2 dependency is missing: {exc}. Prepare the environment with '
            'stage2_prompt_generation/requirements.txt.', file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f'Stage2 failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
