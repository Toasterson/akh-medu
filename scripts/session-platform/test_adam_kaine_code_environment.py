import importlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
M = importlib.import_module("adam_kaine_code_environment")


class CodeEnvironmentTests(unittest.TestCase):
    def test_all_references_pass_and_negative_controls_fail(self):
        for task in M.tasks():
            with self.subTest(task=task.id):
                self.assertTrue(M.score(task.reference, task)["passed"])
                self.assertFalse(M.score(task.negative_control, task)["passed"])

    def test_model_cannot_supply_verifier_receipt(self):
        task = M.tasks()[0]
        source = task.reference + "\n# receipt: passed"
        result = M.score(source, task)
        self.assertTrue(result["passed"])
        self.assertNotIn("receipt: passed", str(result["episode_steps"]))

    def test_logic_failure_has_real_counterexample_and_bounded_graded_reward(self):
        task = M.tasks()[0]
        result = M.score(task.negative_control, task)
        self.assertGreaterEqual(result["klpo_reward"], 0.0)
        self.assertLess(result["klpo_reward"], 1.0)
        self.assertIsNotNone(result["evidence"]["counterexample"])

    def test_split_is_four_plus_four_and_disjoint(self):
        train = {task.id for task in M.tasks() if task.split == 'train'}
        development = {task.id for task in M.tasks() if task.split == 'development_holdout'}
        independent = {task.id for task in M.tasks() if task.split == 'independent_eval'}
        self.assertEqual((len(train), len(development), len(independent)), (4, 4, 4))
        self.assertFalse(train & development or train & independent or development & independent)
        family_sets = [{task.family for task in M.tasks() if task.split == split} for split in ('train','development_holdout','independent_eval')]
        self.assertFalse(family_sets[0] & family_sets[1] or family_sets[0] & family_sets[2] or family_sets[1] & family_sets[2])

    def test_export_freezes_sources_and_cases(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = M.export(Path(td))
            self.assertEqual(len(manifest["tasks"]), 12)
            self.assertEqual(len(manifest["training_ids"]), 4)


if __name__ == "__main__":
    unittest.main()
