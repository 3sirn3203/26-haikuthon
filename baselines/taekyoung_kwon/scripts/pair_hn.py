#!/usr/bin/env python3
"""Run the shared HN pairing analysis with cache output kept under this author folder."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


sys.dont_write_bytecode = True
AUTHOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUTHOR_ROOT.parents[1]
SHARED_SCRIPT = REPO_ROOT / 'baselines' / 'chan_lee' / 'scripts' / 'pair_hn.py'


def main():
    spec = spec_from_file_location('_shared_haiku_pair_hn', SHARED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot load shared HN pairing helper: {SHARED_SCRIPT}')
    shared = module_from_spec(spec)
    spec.loader.exec_module(shared)
    load_shared_paths = shared.D.load_paths
    own_paths = AUTHOR_ROOT / 'configs' / 'paths.yaml'
    shared.D.load_paths = lambda path=None: load_shared_paths(path or own_paths)
    shared.main()


if __name__ == '__main__':
    main()
