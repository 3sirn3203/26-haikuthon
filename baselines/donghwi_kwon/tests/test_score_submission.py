"""Synthetic tests only; no competition prompts or prediction rows."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score_submission import read_ranks, score_splits, summarize


class ScoreTests(unittest.TestCase):
    def test_ranks_one_to_five_and_miss(self):
        expected_ndcg = sum(1 / math.log2(r + 1) for r in range(1, 6)) / 6
        values = summarize([1, 2, 3, 4, 5, None])
        self.assertAlmostEqual(values["recall_at_1"], 1 / 6)
        self.assertAlmostEqual(values["ndcg_at_5"], expected_ndcg)
        self.assertAlmostEqual(values["score"], (1 / 6 + expected_ndcg) / 2)
        self.assertAlmostEqual(values["recall_at_5"], 5 / 6)

    def test_split_weighted_average(self):
        full, splits = score_splits({0: 1, 1: None, 2: 2}, {2})
        for key in ("recall_at_1", "ndcg_at_5", "score"):
            self.assertAlmostEqual(full[key], (splits["public"][key] + 2 * splits["private"][key]) / 3)

    def test_csv_validation_and_zero_padding(self):
        temp_root = Path(__file__).resolve().parents[1] / "artifacts"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as tmp:
            path = Path(tmp) / "submission_synthetic.csv"
            rows = [["0000", "0 2 3 4 5"], ["0001", "2 1 3 4 5"]]
            def write():
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["prompt_id", "prediction"])
                    writer.writerows(rows)
            write()
            self.assertEqual(read_ranks(path, 2, 6), {0: 1, 1: 2})
            rows[1][1] = "1 1 2 3 4"
            write()
            with self.assertRaises(ValueError):
                read_ranks(path, 2, 6)
            rows[1] = ["0", "0 1 2 3 4"]
            write()
            with self.assertRaises(ValueError):
                read_ranks(path, 2, 6)
            rows[:] = rows[:1]
            write()
            with self.assertRaises(ValueError):
                read_ranks(path, 2, 6)


if __name__ == "__main__":
    unittest.main()
