#!/usr/bin/env python3
"""Run the shared SigLIP2 trainer while keeping paths and artifacts in this author folder."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


AUTHOR_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = AUTHOR_ROOT.parents[1]
SHARED_ROOT = REPO_ROOT / "baselines" / "chan_lee"
SHARED_SCRIPT = SHARED_ROOT / "scripts" / "finetune.py"
SHARED_SRC = SHARED_ROOT / "src"


def main():
    sys.path.insert(0, str(SHARED_SRC))
    spec = spec_from_file_location("_shared_haiku_finetune", SHARED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load shared fine-tuning implementation: {SHARED_SCRIPT}")
    shared = module_from_spec(spec)
    spec.loader.exec_module(shared)

    # The implementation writes checkpoints/evaluation artifacts relative to CHAN.
    shared.CHAN = AUTHOR_ROOT
    load_paths = shared.D.load_paths
    own_paths = AUTHOR_ROOT / "configs" / "paths.yaml"
    shared.D.load_paths = lambda path=None: load_paths(path or own_paths)
    shared.main()


if __name__ == "__main__":
    main()
