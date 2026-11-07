"""python -m stage3_image_generation {serve,generate}."""
import argparse
import sys

import yaml

from .config import DEFAULT_CONFIG, load_config


def positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    server = commands.add_parser('serve', help='Run the pinned Qwen-Image/Lightning server')
    server.add_argument('--dry-run', action='store_true', help='Print command without downloading or using GPU')
    generation = commands.add_parser('generate', help='Generate or resume PNG/prompt pairs')
    generation.add_argument('--split', choices=('train', 'test', 'all'), default='all')
    generation.add_argument('--limit', type=positive_integer, help='Select the first N rows of each split')
    generation.add_argument('--overwrite', action='store_true', help='Regenerate every selected sample')
    for command in (server, generation):
        command.add_argument('--config', default=str(DEFAULT_CONFIG), help='YAML configuration path')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == 'serve':
            from .server import serve
            return serve(config, dry_run=args.dry_run)
        from .pipeline import generate
        return generate(config, split=args.split, limit=args.limit, overwrite=args.overwrite)
    except KeyboardInterrupt:
        print('Stage3 interrupted.', file=sys.stderr, flush=True)
        return 130
    except (ValueError, OSError, RuntimeError, yaml.YAMLError) as exc:
        print(f'Stage3 failed: {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
