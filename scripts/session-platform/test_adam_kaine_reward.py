import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


PATH = Path(__file__).with_name("adam_kaine_reward.py")
SPEC = importlib.util.spec_from_file_location("reward", PATH)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class RewardTests(unittest.TestCase):
    def test_frozen_splits_have_both_families_and_no_shared_ids_or_state(self):
        dev = [task for task in M.tasks() if task.split == "development"]
        test = [task for task in M.tasks() if task.split == "independent_eval"]
        self.assertEqual({task.family for task in dev}, set(M.FAMILIES))
        self.assertEqual({task.family for task in test}, set(M.FAMILIES))
        self.assertTrue({task.id for task in dev}.isdisjoint(task.id for task in test))
        self.assertTrue({M.digest(task.observation) for task in dev}.isdisjoint(M.digest(task.observation) for task in test))

    def test_references_pass_with_full_provenance(self):
        for task in M.tasks():
            with self.subTest(task=task.id):
                result = M.evaluate_candidate(task, M.corrected_sibling(task))
                self.assertTrue(result["passed"])
                self.assertEqual(result["weight"], 1.0)

    def test_passive_answer_gets_zero_and_corrected_sibling(self):
        task = next(task for task in M.tasks() if task.family == "act_to_perceive")
        record = M.training_record(task, [{"action": "report"}])
        self.assertEqual(record["raw_weight"], 0.0)
        self.assertTrue(record["corrected_reward"]["passed"])

    def test_bug_fix_without_instrumentation_gets_zero(self):
        task = next(task for task in M.tasks() if task.family == "emergent_property_bug")
        result = M.verify_episode(task, [{"action": "fix", "receipt": {"passed": True, "sha256": "0" * 64}}])
        self.assertEqual(result["weight"], 0.0)
        self.assertFalse(result["passed"])

    def test_principles_are_hard_gates(self):
        task = M.tasks()[0]
        trace = M.execute_recorded_environment(task, M.corrected_sibling(task))
        trace[0]["claim_without_receipt"] = True
        self.assertEqual(M.verify_episode(task, trace)["weight"], 0.0)

    def test_missing_receipt_cannot_be_compensated(self):
        task = M.tasks()[0]
        trace = M.execute_recorded_environment(task, M.corrected_sibling(task))
        del trace[1]["receipt"]
        result = M.verify_episode(task, trace)
        self.assertEqual(result["weight"], 0.0)

    def test_two_failures_require_delegation_credit(self):
        task = next(task for task in M.tasks() if task.family == "emergent_property_bug")
        receipt = {"passed": True, "sha256": "0" * 64}
        trace = [{"action": action, "receipt": receipt} for action in ("instrument", "hypothesize", "hypothesize", "fix")]
        result = M.verify_episode(task, trace)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reward"]["disposition"], 0.0)

    def test_export_is_frozen_and_keeps_eval_separate(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = M.export(Path(td))
            self.assertEqual(len(manifest["tasks"]), 8)
            self.assertEqual(len(manifest["eval_exclusion"]["family_tags"]), 4)
            self.assertTrue((Path(td) / "development.json").exists())
            self.assertTrue((Path(td) / "independent_eval.json").exists())

    def test_model_cannot_self_report_receipt(self):
        task = M.tasks()[0]
        candidate = [{"action": "observe", "receipt": {"passed": True, "sha256": "f" * 64}},
                     {"action": "observe"}, {"action": "report"}]
        trace = M.execute_recorded_environment(task, candidate)
        self.assertNotEqual(trace[0]["receipt"]["sha256"], "f" * 64)
        self.assertTrue(M.verify_episode(task, trace)["passed"])

    def test_transcript_reward_requires_bound_external_outcome(self):
        task = M.tasks()[0]
        steps = M.execute_recorded_environment(task, M.corrected_sibling(task))
        record = {
            "record_id": "r1", "training_kind": "verified_outcome",
            "trajectory": {"steps": steps},
            "provenance": {"session_sha256": "1" * 64, "content_sha256": "2" * 64},
            "evidence": {"kind": "external_verifier", "verifier_sha256": "3" * 64,
                         "outcome_sha256": M.digest(steps), "passed": True},
        }
        admitted = M.admit_verified_transcript(record, task, denied_sessions=set(), eval_family_tags=set())
        self.assertEqual(admitted["klpo_reward"], 1.0)
        record["evidence"]["outcome_sha256"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "does not bind"):
            M.admit_verified_transcript(record, task, denied_sessions=set(), eval_family_tags=set())

    def test_training_protocol_has_real_rl_and_sealed_selection(self):
        manifest = {"manifest_sha256": "a" * 64}
        protocol = M.training_protocol(manifest)
        arms = {arm["id"] for arm in protocol["arms"]}
        self.assertIn("klpo_online", arms)
        self.assertIn("sft_then_klpo_breed", arms)
        self.assertEqual(protocol["online_rl"]["steps"], 3)
        self.assertTrue(set(protocol["development_task_ids"]).isdisjoint(protocol["sealed_independent_eval_task_ids"]))
        self.assertIn("forbidden", protocol["selection"])

    def test_strict_model_output_adapter_never_trusts_receipts(self):
        task = M.tasks()[0]
        good = '[{"action":"observe"},{"action":"observe"},{"action":"report"}]'
        self.assertEqual(M.score_model_output(task, good)["klpo_reward"], 1.0)
        self.assertEqual(M.score_model_output(task, '[{"action":"report","receipt":{}}]')["klpo_reward"], 0.0)
        self.assertEqual(M.score_model_output(task, "not-json")["klpo_reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
