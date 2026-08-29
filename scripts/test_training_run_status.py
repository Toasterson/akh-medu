from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.training_run_status import TrainingRunReporter


class TrainingRunReporterTests(unittest.TestCase):
    def test_status_is_atomic_and_contains_disk_guard_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reporter = TrainingRunReporter("run-1", Path(directory), total_steps=20)
            reporter.publish()
            status = json.loads(reporter.status_path.read_text())
            self.assertEqual(status["run_id"], "run-1")
            self.assertEqual(status["total_steps"], 20)
            self.assertGreater(status["disk_free_bytes"], 0)
            self.assertEqual(list(Path(directory).glob("*.tmp.*")), [])

    def test_control_waits_for_checkpoint_then_only_sends_sigint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            reporter = TrainingRunReporter("run-1", run_dir)
            reporter.control_path.write_text(
                json.dumps({"run_id": "run-1", "command": "checkpoint_and_stop"})
            )
            with mock.patch("os.kill") as kill:
                reporter._honor_control_if_safe()
                kill.assert_not_called()
                (run_dir / "adapter.safetensors").write_bytes(b"checkpoint")
                reporter._honor_control_if_safe()
                kill.assert_called_once()
                self.assertEqual(kill.call_args.args[1], 2)

    def test_checkpoint_mtime_is_published_as_progress_not_just_heartbeat(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            reporter = TrainingRunReporter("run-1", run_dir)
            checkpoint = run_dir / "adapter.safetensors"
            checkpoint.write_bytes(b"first")
            reporter.publish()
            first = json.loads(reporter.status_path.read_text())
            checkpoint.write_bytes(b"second")
            reporter.publish()
            second = json.loads(reporter.status_path.read_text())
            self.assertIn("checkpoint_mtime_ns", second["metrics"])
            self.assertGreaterEqual(
                second["progress_updated_at"], first["progress_updated_at"]
            )


if __name__ == "__main__":
    unittest.main()
