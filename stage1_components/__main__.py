"""python3 -m stage1_components {generate,validate} [--id ID] [--seed N]."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from .generator import Generator
from .files import check_output_directory, component_names
from .model import ROOT, read_yaml, require
from .validation import read_jsonl, validate


def encode(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows).encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'validate'))
    parser.add_argument('--config', type=Path, default=ROOT / 'policy.yaml', help='Stage1 policy YAML')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--id', dest='output_id',
                        help='optional filename suffix; explicit IDs may replace existing files')
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent / 'outputs_stage1')
    parser.add_argument('--check-reproducibility', action='store_true', help='validate: regenerate in memory and compare both JSONL byte streams')
    args = parser.parse_args()
    try:
        identifier = args.output_id
        names = component_names(identifier)
        policy = read_yaml(args.config)
        seed = policy['seed'] if args.seed is None else args.seed
        print(f'Output id: {identifier if identifier is not None else "(default)"}', flush=True)
        if args.output_dir.exists():
            check_output_directory(args.output_dir)
        if args.command == 'generate':
            if identifier is None and any((args.output_dir / name).exists() for name in names):
                raise FileExistsError('Default component output already exists; use --output-dir or --id')
            generator = Generator(seed, policy=policy)
            train, test = generator.run()
            validate(train, test, generator.catalog, generator.policy, seed)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            check_output_directory(args.output_dir)
            # Stage both validated byte streams outside outputs_stage1. A failure
            # before replacement cannot leave a partial JSONL in that directory.
            temporary = []
            try:
                for name, rows in zip(names, (train, test)):
                    with tempfile.NamedTemporaryFile(dir=args.output_dir.parent, delete=False) as f:
                        temporary.append(Path(f.name)); f.write(encode(rows)); f.flush(); os.fsync(f.fileno())
                publish = os.link if identifier is None else os.replace
                for temp, name in zip(temporary, names):
                    publish(temp, args.output_dir / name)
            finally:
                for temp in temporary: temp.unlink(missing_ok=True)
            print(f'Saved {len(train)} train + {len(test)} test scenes to {args.output_dir}')
            for name in names:
                print(f'  {args.output_dir / name}')
        else:
            require(all((args.output_dir / name).is_file() for name in names),
                    f'Expected both component files: {", ".join(names)}')
            validate(*(read_jsonl(args.output_dir / name) for name in names), policy=policy, seed=seed)
            if args.check_reproducibility:
                rows = Generator(seed, policy=policy).run()
                for name, data in zip(names, rows):
                    require(encode(data) == (args.output_dir / name).read_bytes(), f'Non-reproducible output: {name}')
                print('Reproducibility passed: both JSONL byte streams are identical.')
    except (ValueError, OSError) as e:
        parser.exit(1, f'Stage 1 failed: {e}\n')


if __name__ == '__main__':
    main()
