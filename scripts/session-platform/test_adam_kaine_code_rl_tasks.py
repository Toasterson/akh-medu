import importlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
M = importlib.import_module("adam_kaine_code_rl_tasks")


class RunnerTaskTests(unittest.TestCase):
    def test_shape_and_rewards(self):
        rows = M.tasks()
        self.assertEqual(sum(t.split == "train" for t in rows), 4)
        self.assertEqual(len(rows), 12)
        for task in rows:
            self.assertEqual(M.verify(task.reference, task)["klpo_reward"], 1.0)
            self.assertLess(M.verify(task.negative_control, task)["klpo_reward"], 1.0)

    def test_manifest_pins_environment(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = M.export(Path(td) / "tasks.json")
            self.assertEqual(len(manifest["tasks"]), 12)
            self.assertEqual(len(manifest["environment_manifest_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
