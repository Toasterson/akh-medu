from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPTS))
import stop_runpod_after_pipeline as guard  # noqa: E402


class RunpodCostGuardTests(unittest.TestCase):
    def test_success_and_terminal_failure_both_trigger_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(guard.trigger_reason(root))
            (root / "status.json").write_text(
                json.dumps({"stage": "waiting_for_remote_candidate"})
            )
            self.assertIsNone(guard.trigger_reason(root))
            (root / "status.json").write_text(json.dumps({"stage": "failed"}))
            self.assertEqual(
                guard.trigger_reason(root), "pipeline_terminal:failed"
            )
            (root / "staged").mkdir()
            (root / "staged/stage-complete.json").write_text("{}\n")
            self.assertEqual(
                guard.trigger_reason(root), "candidate_staged_locally"
            )

    def test_command_is_stop_only_and_uses_enrolled_cache(self) -> None:
        config = {
            "secret_profile": "runpod",
            "pod_id": "4ypc4li9y537f2",
            "agent_tools": {"path": "/safe/agent-tools"},
            "runpodctl": {"path": "/safe/runpodctl"},
        }
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "inherited-secret"}):
            command, environment = guard.stop_command(config)
        self.assertEqual(
            command,
            [
                "/safe/agent-tools",
                "runpod",
                "--",
                "/safe/runpodctl",
                "pod",
                "stop",
                "4ypc4li9y537f2",
                "-o",
                "json",
            ],
        )
        self.assertNotIn("delete", command)
        self.assertNotIn("RUNPOD_API_KEY", environment)
        self.assertNotIn("inherited-secret", command)


if __name__ == "__main__":
    unittest.main()
