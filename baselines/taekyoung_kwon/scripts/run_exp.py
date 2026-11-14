#!/usr/bin/env python3
"""Run the shared organizer evaluator while writing artifacts to this author folder."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


AUTHOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUTHOR_ROOT.parents[1]
SHARED_RUNNER = REPO_ROOT / 'baselines' / 'chan_lee' / 'scripts' / 'run_exp.py'


def main():
    if not any(arg == '--paths' or arg.startswith('--paths=') for arg in sys.argv[1:]):
        sys.argv.extend(['--paths', str(AUTHOR_ROOT / 'configs' / 'paths.yaml')])

    spec = spec_from_file_location('_shared_haiku_run_exp', SHARED_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load shared evaluator: {SHARED_RUNNER}')
    shared = module_from_spec(spec)
    spec.loader.exec_module(shared)
    # The shared evaluator and metrics stay read-only; route generated results to this author.
    shared.CHAN = AUTHOR_ROOT
    shared.main()


if __name__ == '__main__':
    main()
