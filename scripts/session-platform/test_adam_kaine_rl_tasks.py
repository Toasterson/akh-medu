import importlib
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).parent))
M = importlib.import_module("adam_kaine_rl_tasks")


class CompatibilityTests(unittest.TestCase):
    def test_runner_shape_and_split(self):
        rows = M.tasks()
        self.assertEqual(len(rows), 8)
        self.assertEqual(sum(row.split == "train" for row in rows), 4)
        self.assertEqual(sum(row.split == "development_holdout" for row in rows), 4)

    def test_references_pass_and_self_reported_receipt_fails_schema(self):
        for task in M.tasks():
            self.assertTrue(M.verify(task.reference, task)["passed"])
        task = M.tasks()[0]
        value = '[{"action":"report","receipt":{"passed":true}}]'
        self.assertEqual(M.verify(value, task)["klpo_reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
