from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


SCRIPTS = Path(__file__).parents[1]
TRAINING = SCRIPTS / "run_native_ocr_training.py"
OFFLINE = SCRIPTS / "run_native_ocr_offline_gate.py"
sys.path.insert(0, str(SCRIPTS))
import run_native_ocr_cross_host_gate as cross_host  # noqa: E402


FAKE_TRAINER = r'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]

artifacts = Path(value("--artifacts-dir"))
artifacts.mkdir()
checkpoint = artifacts / "checkpoints/candidate_update_00000003"
checkpoint.mkdir(parents=True)
(checkpoint / "model.mpk").write_text("model")
(checkpoint / "optim.mpk").write_text("optimizer")
(checkpoint / "meta.json").write_text(json.dumps({
    "update": 3,
    "complete": True,
    "model_hash": "blake3:model-3",
    "optimizer_hash": "blake3:optimizer-3",
}))
candidate = {
    "checkpoint_dir": "checkpoints/candidate_update_00000003",
    "checkpoint_step": 3,
    "checkpoint_model_hash": "blake3:model-3",
    "checkpoint_optimizer_hash": "blake3:optimizer-3",
    "certification": None,
}
(artifacts / "multimodal-candidate.json").write_text(json.dumps(candidate))
(artifacts / "completion.json").write_text(json.dumps({
    "candidate_only": True,
    "active_model_written": False,
}))
'''


FAKE_OFFLINE = r'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def value(flag):
    return sys.argv[sys.argv.index(flag) + 1]

name = Path(sys.argv[0]).name
bundle = Path(value("--candidate-bundle"))
candidate = json.loads((bundle / "multimodal-candidate.json").read_text())
if name == "eval-fail":
    raise SystemExit(9)
if name.startswith("eval"):
    output = Path(value("--output-dir"))
    common = {
        "checkpoint_step": candidate["checkpoint_step"],
        "checkpoint_model_hash": candidate["checkpoint_model_hash"],
    }
    (output / "multimodal-audit-evidence.json").write_text(json.dumps(common))
    (output / "multimodal-text-retention-evidence.json").write_text(json.dumps(common))
    replay = dict(common)
    replay.update({
        "network_denied_verified": True,
        "visible_interfaces": ["lo"],
        "kernel_route_entries": 0,
        "replay_cases": 30,
        "first_outputs_hash": "blake3:replay",
        "second_outputs_hash": "blake3:replay",
    })
    (output / "multimodal-network-denied-replay.json").write_text(json.dumps(replay))
    for item in (
        "audit-predictions.jsonl",
        "retention-before-outputs.jsonl",
        "retention-after-outputs.jsonl",
        "network-replay-first.jsonl",
        "network-replay-second.jsonl",
    ):
        (output / item).write_text("{}\n")
    (output / "completion.json").write_text(json.dumps(common))
elif name == "certify":
    active = dict(candidate)
    active["certification"] = {"network_denied": True}
    active["checkpoint_optimizer_hash"] = None
    (bundle / "multimodal-active.json").write_text(json.dumps(active))
    (bundle / "certifier-invoked").write_text("yes")
else:
    raise SystemExit(f"unexpected fake binary {name}")
'''


class NativeOcrPipelineTests(unittest.TestCase):
    @staticmethod
    def binary(directory: Path, name: str, source: str) -> Path:
        path = directory / name
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
        return path

    @staticmethod
    def file(directory: Path, name: str, content: str = "input\n") -> Path:
        path = directory / name
        path.write_text(content, encoding="utf-8")
        return path

    def training_command(
        self,
        root: Path,
        release: Path,
        trainer: Path,
        output: Path,
        artifacts: Path,
    ) -> list[str]:
        vision_train = self.file(root, "vision-train.jsonl")
        vision_audit = self.file(root, "vision-audit.jsonl")
        validation = self.file(root, "audit-validation.json", "{}\n")
        retention_train = self.file(root, "retention-train.jsonl")
        retention_eval = self.file(root, "retention-eval.jsonl")
        return [
            sys.executable,
            str(TRAINING),
            "--text-release-gate",
            str(release),
            "--out",
            str(output),
            "--artifacts-dir",
            str(artifacts),
            "--train-binary",
            str(trainer),
            "--vision-train-manifest",
            str(vision_train),
            "--vision-audit-manifest",
            str(vision_audit),
            "--audit-validation",
            str(validation),
            "--retention-train-jsonl",
            str(retention_train),
            "--retention-eval-jsonl",
            str(retention_eval),
            "--poll-seconds",
            "5",
            "--maximum-wait-hours",
            "0.01",
        ]

    def test_training_waits_for_exact_active_text_then_leaves_candidate_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            bundle = root / "text-bundle"
            release.mkdir()
            bundle.mkdir()
            active = {
                "checkpoint_step": 42,
                "checkpoint_model_hash": "blake3:text-42",
                "certification": {"passed": True},
            }
            active_path = bundle / "active-model.json"
            active_path.write_text(json.dumps(active))
            (release / "status.json").write_text(
                json.dumps(
                    {
                        "stage": "active_text_model_ready",
                        "bundle": str(bundle),
                        "active_model": str(active_path),
                        "checkpoint": {
                            "step": 42,
                            "artifact_hash": "blake3:text-42",
                        },
                    }
                )
            )
            trainer = self.binary(root, "train-native-ocr", FAKE_TRAINER)
            output = root / "training"
            artifacts = root / "multimodal"
            completed = subprocess.run(
                self.training_command(root, release, trainer, output, artifacts),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["stage"], "candidate_ready_for_offline_gate")
            self.assertFalse((artifacts / "multimodal-active.json").exists())

    def test_training_stops_when_text_release_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            release.mkdir()
            (release / "status.json").write_text(
                json.dumps({"stage": "blocked_by_adaptive_loop"})
            )
            trainer = self.binary(root, "train-native-ocr", FAKE_TRAINER)
            output = root / "training"
            artifacts = root / "multimodal"
            completed = subprocess.run(
                self.training_command(root, release, trainer, output, artifacts),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["stage"], "blocked_by_text_release")
            self.assertFalse(artifacts.exists())

    def offline_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        candidate_bundle = root / "candidate"
        base_bundle = root / "base"
        candidate_bundle.mkdir()
        base_bundle.mkdir()
        (candidate_bundle / "multimodal-candidate.json").write_text(
            json.dumps(
                {
                    "checkpoint_step": 9,
                    "checkpoint_model_hash": "blake3:mm-9",
                    "checkpoint_optimizer_hash": "blake3:optim-9",
                    "certification": None,
                }
            )
        )
        (base_bundle / "active-model.json").write_text("{}\n")
        audit = self.file(root, "audit.jsonl")
        retention = self.file(root, "retention.jsonl")
        return candidate_bundle, base_bundle, audit, retention

    def offline_command(
        self,
        root: Path,
        evaluator: Path,
        certifier: Path,
        output: Path,
    ) -> list[str]:
        candidate, base, audit, retention = self.offline_fixture(root)
        return [
            sys.executable,
            str(OFFLINE),
            "--candidate-bundle",
            str(candidate),
            "--base-text-bundle",
            str(base),
            "--audit-manifest",
            str(audit),
            "--retention-eval-jsonl",
            str(retention),
            "--out",
            str(output),
            "--eval-binary",
            str(evaluator),
            "--certify-binary",
            str(certifier),
        ]

    def test_offline_gate_requires_evidence_then_publishes_exact_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = self.binary(root, "eval", FAKE_OFFLINE)
            certifier = self.binary(root, "certify", FAKE_OFFLINE)
            output = root / "offline"
            completed = subprocess.run(
                self.offline_command(root, evaluator, certifier, output),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["stage"], "active_multimodal_model_ready")
            active = json.loads(Path(status["active_model"]).read_text())
            self.assertEqual(active["checkpoint_model_hash"], "blake3:mm-9")
            self.assertIsNone(active["checkpoint_optimizer_hash"])

    def test_failed_offline_evaluation_never_invokes_certifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = self.binary(root, "eval-fail", FAKE_OFFLINE)
            certifier = self.binary(root, "certify", FAKE_OFFLINE)
            output = root / "offline"
            completed = subprocess.run(
                self.offline_command(root, evaluator, certifier, output),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["stage"], "failed")
            self.assertFalse((root / "candidate/certifier-invoked").exists())
            self.assertFalse((root / "candidate/multimodal-active.json").exists())

    def test_cross_host_container_is_networkless_and_preserves_corpus_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory)
            marker = {
                "retention_files": [
                    {"path": "retention/knowledge.jsonl"},
                    {"path": "retention/dojo.jsonl"},
                    {"path": "retention/tools.jsonl"},
                ]
            }
            command = cross_host.docker_command(
                Namespace(
                    offline_image="akh/offline:pinned",
                    docker_binary=Path("/opt/homebrew/bin/docker"),
                ),
                staged,
                marker,
                True,
            )
            self.assertEqual(command[command.index("--network") + 1], "none")
            self.assertEqual(command[command.index("--cap-drop") + 1], "ALL")
            self.assertIn("--read-only", command)
            self.assertEqual(command[-1], "--resume")
            retention = [
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--retention-eval-jsonl"
            ]
            self.assertEqual(
                retention,
                [
                    "/work/retention/knowledge.jsonl",
                    "/work/retention/dojo.jsonl",
                    "/work/retention/tools.jsonl",
                ],
            )

    def test_cross_host_remote_paths_cannot_escape_the_pinned_base(self) -> None:
        self.assertEqual(
            cross_host.ensure_remote_descendant("/workspace/base", "/workspace/base/model"),
            "/workspace/base/model",
        )
        with self.assertRaises(RuntimeError):
            cross_host.ensure_remote_descendant("/workspace/base", "/workspace/other")
        with self.assertRaises(RuntimeError):
            cross_host.safe_bundle_child("../checkpoint")


if __name__ == "__main__":
    unittest.main()
