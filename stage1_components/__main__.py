"""python3 -m stage1_components {generate,validate} [--seed N]."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from .generator import Generator
from .model import ROOT, read_yaml, require
from .validation import read_jsonl, validate


def encode(rows):
    return ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in rows).encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('generate', 'validate'))
    parser.add_argument('--seed', type=int, default=read_yaml(ROOT / 'policy.yaml')['seed'])
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent / 'outputs_stage1')
    parser.add_argument('--check-reproducibility', action='store_true', help='validate: regenerate in memory and compare both JSONL byte streams')
    args = parser.parse_args()
    try:
        names = ('train_components.jsonl', 'test_components.jsonl')
        if args.command == 'generate':
            generator = Generator(args.seed)
            train, test = generator.run()
            validate(train, test, generator.catalog, generator.policy, args.seed)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            require(not (set(p.name for p in args.output_dir.iterdir()) - set(names)), 'Output directory contains unexpected files')
            # Stage both validated byte streams outside outputs_stage1. A failure
            # before replacement cannot leave a partial JSONL in that directory.
            temporary = []
            try:
                for name, rows in zip(names, (train, test)):
                    with tempfile.NamedTemporaryFile(dir=args.output_dir.parent, delete=False) as f:
                        temporary.append(Path(f.name)); f.write(encode(rows)); f.flush(); os.fsync(f.fileno())
                for temp, name in zip(temporary, names): os.replace(temp, args.output_dir / name)
            finally:
                for temp in temporary: temp.unlink(missing_ok=True)
            print(f'Saved {len(train)} train + {len(test)} test scenes to {args.output_dir}')
        else:
            require(set(p.name for p in args.output_dir.iterdir()) == set(names), 'Expected exactly the two JSONL outputs')
            validate(*(read_jsonl(args.output_dir / name) for name in names), seed=args.seed)
            if args.check_reproducibility:
                rows = Generator(args.seed).run()
                for name, data in zip(names, rows):
                    require(encode(data) == (args.output_dir / name).read_bytes(), f'Non-reproducible output: {name}')
                print('Reproducibility passed: both JSONL byte streams are identical.')
    except (ValueError, OSError) as e:
        parser.exit(1, f'Stage 1 failed: {e}\n')


if __name__ == '__main__':
    main()
