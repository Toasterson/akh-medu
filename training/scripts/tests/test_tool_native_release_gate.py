from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1]
RELEASE_GATE = SCRIPTS / "run_tool_native_release_gate.py"


FAKE_BINARY = r'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]

name = Path(sys.argv[0]).name
bundle = Path(value("--bundle"))
candidate = json.loads((bundle / "candidate-model.json").read_text())
checkpoint = {
    "step": candidate["checkpoint_step"],
    "artifact_hash": candidate["checkpoint_model_hash"],
}
if name == "knowledge-eval":
    exam = json.loads(Path(value("--exam")).read_text())
    evidence = {
        "observation": {
            "tier": "sentinel",
            "family": exam["plan"]["family"],
            "checkpoint": checkpoint,
            "sample_count": exam["plan"]["sample_count"],
        }
    }
    Path(value("--evidence-out")).write_text(json.dumps(evidence))
    Path(value("--predictions-out")).write_text("[]")
elif name == "tool-eval":
    exam = json.loads(Path(value("--exam")).read_text())
    observations = [{
        "tier": "sentinel",
        "family": "tool_contract.fake",
        "checkpoint": checkpoint,
        "sample_count": exam["samples_per_tool"],
    }]
    Path(value("--evidence-out")).write_text(json.dumps({"observations": observations}))
    Path(value("--observations-out")).write_text(json.dumps(observations[0]) + "\n")
    Path(value("--predictions-out")).write_text("[]")
elif name == "certify":
    active = dict(candidate)
    active["certification"] = {"test": True}
    active["checkpoint_optimizer_hash"] = None
    (bundle / "active-model.json").write_text(json.dumps(active))
else:
    raise SystemExit(f"unknown fake binary {name}")
'''


class ToolNativeReleaseGateTests(unittest.TestCase):
    def make_binary(self, directory: Path, name: str) -> Path:
        path = directory / name
        path.write_text(FAKE_BINARY, encoding="utf-8")
        path.chmod(0o755)
        return path

    def run_gate(
        self, adaptive: Path, output: Path, binaries: dict[str, Path]
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(RELEASE_GATE),
                "--adaptive-loop",
                str(adaptive),
                "--out",
                str(output),
                "--knowledge-eval-binary",
                str(binaries["knowledge"]),
                "--tool-eval-binary",
                str(binaries["tool"]),
                "--certify-binary",
                str(binaries["certify"]),
                "--poll-seconds",
                "5",
                "--maximum-wait-hours",
                "0.01",
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def check_complete_loop(self, terminal_stage: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adaptive = root / "adaptive"
            adaptive.mkdir()
            bundle = root / "bundle"
            checkpoint_dir = bundle / "checkpoints/step_00000042"
            checkpoint_dir.mkdir(parents=True)
            for name in ("meta.json", "model.mpk", "optim.mpk"):
                checkpoint_dir.joinpath(name).write_text("{}", encoding="utf-8")
            candidate = {
                "checkpoint_step": 42,
                "checkpoint_model_hash": "blake3:model-42",
                "checkpoint_optimizer_hash": "blake3:optim-42",
                "checkpoint_dir": "checkpoints/step_00000042",
                "certification": None,
            }
            bundle.joinpath("candidate-model.json").write_text(
                json.dumps(candidate), encoding="utf-8"
            )
            adaptive.joinpath("status.json").write_text(
                json.dumps(
                    {
                        "stage": terminal_stage,
                        "bundle": str(bundle),
                        "checkpoint": {
                            "step": 42,
                            "artifact_hash": "blake3:model-42",
                        },
                    }
                ),
                encoding="utf-8",
            )
            sentinels = adaptive / "sentinels"
            sentinels.mkdir()
            knowledge_entries = []
            for family in (
                "knowledge.acquire_textbook",
                "knowledge.recall_after_restart",
                "knowledge.revise_from_new_edition",
                "knowledge.provider_fallback",
                "knowledge.grounded_abstention",
                "knowledge.textbook_inventory",
                "knowledge.note_navigation",
                "knowledge.length_generalization",
            ):
                path = sentinels / f"{family.replace('.', '-')}.json"
                path.write_text(
                    json.dumps({"plan": {"family": family, "sample_count": 100}}),
                    encoding="utf-8",
                )
                knowledge_entries.append(
                    {
                        "path": str(path.relative_to(adaptive)),
                        "sha256": self.sha256(path),
                    }
                )
            tools = sentinels / "all-tools.json"
            tools.write_text(json.dumps({"samples_per_tool": 24}), encoding="utf-8")
            adaptive.joinpath("loop-config.json").write_text(
                json.dumps(
                    {
                        "tool_sentinel_samples": 24,
                        "sentinels": {
                            "knowledge": knowledge_entries,
                            "tools": {
                                "path": str(tools.relative_to(adaptive)),
                                "sha256": self.sha256(tools),
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            binaries = {
                "knowledge": self.make_binary(root, "knowledge-eval"),
                "tool": self.make_binary(root, "tool-eval"),
                "certify": self.make_binary(root, "certify"),
            }
            output = root / "release"
            completed = self.run_gate(adaptive, output, binaries)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            status = json.loads(output.joinpath("status.json").read_text())
            self.assertEqual(status["stage"], "active_text_model_ready")
            active = json.loads(bundle.joinpath("active-model.json").read_text())
            self.assertEqual(active["checkpoint_model_hash"], "blake3:model-42")
            self.assertIsInstance(active["certification"], dict)

    def test_complete_loop_runs_separate_gate_and_writes_exact_active(self) -> None:
        self.check_complete_loop("window_budget_exhausted")

    def test_early_training_stop_still_runs_independent_release_certification(self) -> None:
        self.check_complete_loop("ready_for_release")

    def test_controller_stop_never_invokes_certifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adaptive = root / "adaptive"
            adaptive.mkdir()
            adaptive.joinpath("status.json").write_text(
                json.dumps({"stage": "stopped_by_controller", "window": 2}),
                encoding="utf-8",
            )
            binaries = {
                "knowledge": self.make_binary(root, "knowledge-eval"),
                "tool": self.make_binary(root, "tool-eval"),
                "certify": self.make_binary(root, "certify"),
            }
            output = root / "release"
            completed = self.run_gate(adaptive, output, binaries)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            status = json.loads(output.joinpath("status.json").read_text())
            self.assertEqual(status["stage"], "blocked_by_adaptive_loop")
            self.assertFalse(status["active_model_written"])

    @staticmethod
    def sha256(path: Path) -> str:
        import hashlib

        return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


if __name__ == "__main__":
    unittest.main()
