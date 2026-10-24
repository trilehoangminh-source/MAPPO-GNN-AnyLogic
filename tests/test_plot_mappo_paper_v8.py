"""Offline structural tests for plot_mappo_paper_v8; does not require AnyLogic."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_mappo_paper_v8.py"
spec = importlib.util.spec_from_file_location("mappo_plot", SCRIPT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class TestLogParsing(unittest.TestCase):
    def test_nested_environment(self):
        item = {"episode": 42, "environment": {
            "delivered_reward": -201.5,
            "statistics": {"jobDone": 8, "targetReached": 250, "block": 70,
                           "wrongTarget": 3, "liveLock": 22}}}
        record = plot.normalize(item, 1)
        self.assertEqual(record["episode"], 42)
        self.assertEqual(record["reward"], -201.5)
        self.assertEqual(record["jobDone"], 8)
        self.assertEqual(record["wrongTarget"], 3)

    def test_evaluation_seeds(self):
        item = {"seed": 3201, "environment": {"delivered_reward": 10,
                "statistics": {"jobDone": 9}}}
        record = plot.normalize(item, 1)
        self.assertEqual(record["seed"], 3201)
        self.assertEqual(record["jobDone"], 9)

    def test_jsonl_records(self):
        with tempfile.TemporaryDirectory() as work:
            file = Path(work) / "metrics.jsonl"
            file.write_text(json.dumps({"episode": 1, "reward": 8}) + "\n", encoding="utf-8")
            self.assertEqual(len(plot.read_jsonl(file)), 1)


if __name__ == "__main__":
    unittest.main()
