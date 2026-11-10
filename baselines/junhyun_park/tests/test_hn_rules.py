import json, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetune"))
from hn_rules import make_hard_negatives


def test_examples():
    rng = random.Random(0)
    t = "In a kitchen, two bowls and one lemon sit alongside a single plate. Both bowls are positioned to the right of the lemon."
    kinds = {k: v for k, v in make_hard_negatives(t, 3, rng)}
    assert "to the left of" in kinds["direction"] or "to the right of" not in kinds["direction"]
    assert "three bowls" in kinds["count"] and "All three bowls" in kinds["count"]
    assert all(v != t for v in kinds.values())


def test_no_slot_returns_empty():
    assert make_hard_negatives("A photograph of a dog.", 3, random.Random(0)) == []
