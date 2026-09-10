from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "run_tool_native_adaptive_loop.py"
SPEC = importlib.util.spec_from_file_location("tool_native_adaptive_loop", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
LOOP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOOP)


class AdaptiveLoopPolicyTests(unittest.TestCase):
    def test_steered_parent_verifies_full_corpus_and_retains_only_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.jsonl"
            sibling = Path(directory) / "sibling.jsonl"
            base.write_text("{}\n" * 19)
            sibling.write_text("{}\n")
            run = {
                "train_jsonl": str(base), "training_records": 20,
                "steering_sibling_jsonl": str(sibling),
                "steered_from": {"base_training_records": 19, "sibling_training_records": 1},
            }
            sources, count = LOOP.verified_base_sources(run)
            self.assertEqual(count, 19)
            self.assertEqual(sources["supplements"], [])
            sibling.write_text("{}\n{}\n")
            with self.assertRaisesRegex(RuntimeError, "training record count"):
                LOOP.verified_base_sources(run)
            run["training_records"] = 21
            with self.assertRaisesRegex(RuntimeError, "lineage"):
                LOOP.verified_base_sources(run)

    def test_ordinary_parent_still_requires_exact_corpus_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.jsonl"
            base.write_text("{}\n" * 3)
            run = {"train_jsonl": str(base), "training_records": 3}
            self.assertEqual(LOOP.verified_base_sources(run)[1], 3)
            run["training_records"] = 2
            with self.assertRaisesRegex(RuntimeError, "training record count"):
                LOOP.verified_base_sources(run)

    def test_sibling_budget_never_exceeds_five_percent(self) -> None:
        self.assertEqual(LOOP.sibling_record_budget(1_900, 500), 100)
        self.assertEqual(LOOP.sibling_record_budget(9_917, 500), 500)
        budget = LOOP.sibling_record_budget(1_899, 500)
        self.assertLessEqual(
            budget * 1_000_000,
            (1_899 + budget) * LOOP.MAX_STEERING_SHARE_PPM,
        )
        with self.assertRaises(RuntimeError):
            LOOP.sibling_record_budget(18, 500)

    def test_only_an_eligible_frontier_decision_authorizes_steering(self) -> None:
        step = {
            "next_plan": {"family": "knowledge.provider_fallback"},
            "decisions": [
                {
                    "family": "knowledge.provider_fallback",
                    "tier": "sentinel",
                    "warning": "steer",
                    "statistically_eligible": True,
                },
                {
                    "family": "knowledge.provider_fallback",
                    "tier": "frontier",
                    "warning": "steer",
                    "statistically_eligible": True,
                },
            ],
        }
        self.assertTrue(LOOP.selected_frontier_steer(step))
        step["decisions"].pop()
        self.assertFalse(LOOP.selected_frontier_steer(step))

    def test_controller_plan_must_bind_the_exact_candidate(self) -> None:
        artifact = {
            "checkpoint_step": 42,
            "checkpoint_model_hash": "blake3:model-42",
        }
        step = {
            "controller": {
                "families": {"tool_contract.agent_list": {}},
            },
            "decisions": [],
            "next_plan": {
                "tier": "frontier",
                "family": "tool_contract.agent_list",
                "checkpoint": {
                    "step": 42,
                    "artifact_hash": "blake3:model-42",
                },
                "sample_count": 24,
            },
        }
        self.assertEqual(
            LOOP.validate_controller_step(step, artifact)["family"],
            "tool_contract.agent_list",
        )
        step["next_plan"]["checkpoint"]["step"] = 43
        with self.assertRaises(RuntimeError):
            LOOP.validate_controller_step(step, artifact)

    def test_jsonl_count_ignores_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("{}\n\n{}\n", encoding="utf-8")
            self.assertEqual(LOOP.count_jsonl(path), 2)

    def test_window_ledger_is_contiguous_and_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "windows.jsonl"
            LOOP.append_window_record(ledger, {"window": 1, "action": "continue"})
            self.assertEqual(len(LOOP.read_jsonl(ledger)), 1)
            with self.assertRaises(RuntimeError):
                LOOP.append_window_record(ledger, {"window": 1, "action": "continue"})
            with self.assertRaises(RuntimeError):
                LOOP.append_window_record(ledger, {"window": 3, "action": "continue"})

    def test_retry_directory_preserves_an_incomplete_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            windows = Path(directory)
            first = LOOP.next_window_directory(windows, 2)
            first.joinpath("partial.json").write_text("{}\n", encoding="utf-8")
            second = LOOP.next_window_directory(windows, 2)
            self.assertNotEqual(first, second)
            self.assertTrue(first.joinpath("partial.json").exists())

    def test_sentinel_inventory_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            sentinels = output / "sentinels"
            sentinels.mkdir()
            knowledge = []
            for family in LOOP.KNOWLEDGE_FAMILIES:
                path = sentinels / f"{LOOP.safe_stem(family)}.json"
                path.write_text(json.dumps({"family": family}), encoding="utf-8")
                knowledge.append(path)
            tools = sentinels / "all-tools.json"
            tools.write_text("{}", encoding="utf-8")
            inventory = LOOP.sentinel_inventory(output, knowledge, tools)
            restored, restored_tools = LOOP.load_sentinel_inventory(output, inventory)
            self.assertEqual(restored, [path.resolve() for path in knowledge])
            self.assertEqual(restored_tools, tools.resolve())
            knowledge[0].write_text("tampered", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                LOOP.load_sentinel_inventory(output, inventory)

    def test_completed_window_reconstructs_exact_candidate_and_controller(self) -> None:
        def bundle(path: Path, step: int) -> dict[str, object]:
            checkpoint = path / f"checkpoints/step_{step:08}"
            checkpoint.mkdir(parents=True)
            for name in ("meta.json", "model.mpk", "optim.mpk"):
                checkpoint.joinpath(name).write_text("{}", encoding="utf-8")
            artifact: dict[str, object] = {
                "checkpoint_step": step,
                "checkpoint_model_hash": f"blake3:model-{step}",
                "checkpoint_optimizer_hash": f"blake3:optim-{step}",
                "checkpoint_dir": str(checkpoint.relative_to(path)),
                "certification": None,
            }
            path.joinpath("candidate-model.json").write_text(
                json.dumps(artifact), encoding="utf-8"
            )
            return artifact

        def controller(path: Path, artifact: dict[str, object]) -> None:
            point = {
                "step": artifact["checkpoint_step"],
                "artifact_hash": artifact["checkpoint_model_hash"],
            }
            path.write_text(
                json.dumps(
                    {
                        "controller": {
                            "families": {"tool_contract.agent_list": {}}
                        },
                        "decisions": [],
                        "next_plan": {
                            "tier": "frontier",
                            "family": "tool_contract.agent_list",
                            "checkpoint": point,
                            "sample_count": 24,
                        },
                    }
                ),
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            initial_bundle = output / "initial"
            next_bundle = output / "next"
            initial_bundle.mkdir()
            next_bundle.mkdir()
            initial = bundle(initial_bundle, 10)
            advanced = bundle(next_bundle, 20)
            initial_step = output / "initial-step.json"
            rebound = output / "rebound.json"
            controller(initial_step, initial)
            controller(rebound, advanced)
            LOOP.append_window_record(
                output / "windows.jsonl",
                {
                    "window": 1,
                    "action": "continue",
                    "bundle_after": str(next_bundle),
                    "checkpoint_after": {
                        "step": 20,
                        "artifact_hash": "blake3:model-20",
                    },
                    "controller_rebound": str(rebound),
                },
            )
            restored = LOOP.restore_completed_windows(
                output, initial_bundle, initial_step, 8
            )
            self.assertEqual(restored[0], next_bundle.resolve())
            self.assertEqual(restored[1], advanced)
            self.assertEqual(restored[4], 2)


if __name__ == "__main__":
    unittest.main()
