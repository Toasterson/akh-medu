from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).parents[1]
INSTALLER = SCRIPTS / "install_certified_babyllm.py"
sys.path.insert(0, str(SCRIPTS))
import install_certified_babyllm as installer  # noqa: E402


FAKE_PROBE = r'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[sys.argv.index("--bundle") + 1])
active = json.loads((bundle / "multimodal-active.json").read_text())
for relative in (active["tokenizer_file"], active["contracts_file"]):
    if not (bundle / relative).is_file():
        raise SystemExit(8)
checkpoint = bundle / active["checkpoint_dir"]
if not (checkpoint / "model.mpk").is_file() or not (checkpoint / "meta.json").is_file():
    raise SystemExit(9)
if active.get("certification") is None or active.get("checkpoint_optimizer_hash") is not None:
    raise SystemExit(10)
'''


class CertifiedBabyLlmInstallTests(unittest.TestCase):
    @staticmethod
    def executable(path: Path, source: str) -> Path:
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
        return path

    def fixture(self, root: Path, *, certified: bool = True) -> tuple[Path, Path]:
        cross_host = root / "cross-host"
        bundle = cross_host / "staged/model-small-native-ocr"
        checkpoint = bundle / "checkpoints/candidate_update_00000003"
        checkpoint.mkdir(parents=True)
        (cross_host / "launch-config.json").write_text("{}\n")
        (bundle / "tokenizer.json").write_text("tokenizer\n")
        (bundle / "contracts.json").write_text("{}\n")
        (checkpoint / "model.mpk").write_text("weights\n")
        (checkpoint / "optim.mpk").write_text("adam must not be served\n")
        (checkpoint / "meta.json").write_text("{}\n")
        digest = "a" * 64
        active = {
            "checkpoint_step": 3,
            "checkpoint_model_hash": f"blake3:{digest}",
            "checkpoint_optimizer_hash": None,
            "checkpoint_dir": "checkpoints/candidate_update_00000003",
            "tokenizer_file": "tokenizer.json",
            "contracts_file": "contracts.json",
            "certification": {"passed": True} if certified else None,
        }
        active_path = bundle / "multimodal-active.json"
        active_path.write_text(json.dumps(active))
        (cross_host / "status.json").write_text(
            json.dumps(
                {
                    "stage": "active_multimodal_model_ready",
                    "active_model_written": True,
                    "active_model": str(active_path),
                    "checkpoint_step": 3,
                    "checkpoint_model_hash": f"blake3:{digest}",
                }
            )
        )
        return cross_host, bundle

    def command(
        self, root: Path, cross_host: Path, probe: Path, output: Path
    ) -> list[str]:
        return [
            sys.executable,
            str(INSTALLER),
            "--cross-host-run",
            str(cross_host),
            "--models-dir",
            str(root / "models"),
            "--probe-binary",
            str(probe),
            "--out",
            str(output),
            "--poll-seconds",
            "5",
            "--maximum-wait-hours",
            "0.01",
        ]

    def test_installs_only_serving_files_and_advances_atomic_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cross_host, _bundle = self.fixture(root)
            probe = self.executable(root / "probe", FAKE_PROBE)
            output = root / "install-run"
            completed = subprocess.run(
                self.command(root, cross_host, probe, output),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["stage"], "installed")
            alias = root / "models/knowledge/model-small"
            self.assertTrue(alias.is_symlink())
            release = alias.resolve(strict=True)
            self.assertTrue((release / "multimodal-active.json").is_file())
            self.assertFalse(
                (release / "checkpoints/candidate_update_00000003/optim.mpk").exists()
            )
            provenance = json.loads(
                (release / "deployment-provenance.json").read_text()
            )
            self.assertFalse(provenance["optimizer_included"])
            self.assertEqual(status["model_config_value"], "knowledge/model-small")

    def test_uncertified_pointer_is_never_installed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cross_host, _bundle = self.fixture(root, certified=False)
            probe = self.executable(root / "probe", FAKE_PROBE)
            output = root / "install-run"
            completed = subprocess.run(
                self.command(root, cross_host, probe, output),
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse((root / "models/knowledge/model-small").exists())
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["stage"], "failed")

    def test_bundle_child_rejects_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            outside = root / "outside"
            bundle.mkdir()
            outside.mkdir()
            (outside / "tokenizer.json").write_text("outside")
            (bundle / "tokenizer.json").symlink_to(outside / "tokenizer.json")
            with self.assertRaises(RuntimeError):
                installer.bundle_child(bundle, "tokenizer.json")


if __name__ == "__main__":
    unittest.main()
