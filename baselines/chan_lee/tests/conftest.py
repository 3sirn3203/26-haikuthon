import sys
from pathlib import Path

sys.dont_write_bytecode = True   # importing stage1_components must not write files there
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
