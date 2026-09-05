from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).parents[1]
sys.path.insert(0, str(SCRIPTS))
import hf_commit_certified_babyllm as hf_helper  # noqa: E402
import publish_certified_babyllm as publisher  # noqa: E402


FAKE_PROBE = r'''#!/usr/bin/env python3
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[sys.argv.index("--bundle") + 1])
active = json.loads((bundle / "multimodal-active.json").read_text())
checkpoint = bundle / active["checkpoint_dir"]
required = [
    bundle / active["tokenizer_file"],
    bundle / active["contracts_file"],
    checkpoint / "model.mpk",
    checkpoint / "meta.json",
]
if not all(path.is_file() for path in required):
    raise SystemExit(9)
if active.get("checkpoint_optimizer_hash") is not None:
    raise SystemExit(10)
'''


class CertifiedBabyLlmPublicationTests(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def executable(path: Path, source: str = "#!/bin/sh\nexit 0\n") -> Path:
        path.write_text(source, encoding="utf-8")
        path.chmod(0o755)
        return path

    def fixture(
        self, root: Path
    ) -> tuple[dict[str, object], dict[str, object], Path, Path, Path]:
        models = root / "models"
        alias = models / "knowledge/model-small"
        digest = "a" * 64
        release = (
            alias.parent
            / "releases"
            / f"model-small-step-00000003-{digest[:16]}"
        )
        checkpoint = release / "checkpoints/candidate_update_00000003"
        checkpoint.mkdir(parents=True)
        (release / "tokenizer.json").write_text("tokenizer\n", encoding="utf-8")
        (release / "contracts.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "model.mpk").write_bytes(b"weights")
        (checkpoint / "meta.json").write_text("{}\n", encoding="utf-8")
        certification = {
            "schema_version": 1,
            "checkpoint_step": 3,
            "checkpoint_model_hash": f"blake3:{digest}",
            "audit_manifest_hash": "blake3:audit",
            "audit_score_hash": "blake3:audit-score",
            "text_retention_score_hash": "blake3:retention",
            "audit_cases": 1050,
            "minimum_matrix_cell_cases": 10,
            "exact_rate": 0.9,
            "character_error_rate": 0.04,
            "cross_semantic_collapse_rate": 0.0,
            "metamorphic_consistency": 0.95,
            "minimum_required_slice_exact_rate": 0.75,
            "text_action_exact_before": 0.9,
            "text_action_exact_after": 0.89,
            "maximum_text_action_regression": 0.01,
            "network_denied_replay_hash": "blake3:replay",
        }
        active = {
            "schema_version": 1,
            "model_size": "small",
            "text_shape": {
                "vocab_size": 8000,
                "d_model": 512,
                "n_heads": 8,
                "n_layers": 8,
                "d_ff": 2048,
                "max_seq_len": 2048,
            },
            "vision_shape": {"canvas_width": 480, "canvas_height": 192},
            "parameter_count": 35_000_000,
            "tokenizer_file": "tokenizer.json",
            "tokenizer_hash": "blake3:tokenizer",
            "contracts_file": "contracts.json",
            "contract_hash": "blake3:contracts",
            "checkpoint_dir": "checkpoints/candidate_update_00000003",
            "checkpoint_step": 3,
            "checkpoint_model_hash": f"blake3:{digest}",
            "checkpoint_optimizer_hash": None,
            "base_text_checkpoint_hash": "blake3:base",
            "certification": certification,
        }
        self.write_json(release / "multimodal-active.json", active)
        serving = [
            Path("multimodal-active.json"),
            Path("tokenizer.json"),
            Path("contracts.json"),
            Path("checkpoints/candidate_update_00000003/model.mpk"),
            Path("checkpoints/candidate_update_00000003/meta.json"),
        ]
        provenance = {
            "schema_version": 1,
            "kind": "certified_babyllm_serving_bundle",
            "checkpoint_step": 3,
            "checkpoint_model_hash": f"blake3:{digest}",
            "optimizer_included": False,
            "payload": [
                {
                    "path": relative.as_posix(),
                    "sha256": publisher.loop.sha256_file(release / relative),
                    "bytes": (release / relative).stat().st_size,
                }
                for relative in serving
            ],
        }
        self.write_json(release / "deployment-provenance.json", provenance)
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(Path("releases") / release.name)

        probe = self.executable(root / "probe", FAKE_PROBE)
        install = root / "install"
        install.mkdir()
        install_config = {
            "models_dir": str(models),
            "model_name": "knowledge/model-small",
            "probe_binary": publisher.file_identity(probe),
        }
        self.write_json(install / "launch-config.json", install_config)
        install_status = {
            "stage": "installed",
            "active_model_installed": True,
            "model_path": str(alias),
            "model_config_value": "knowledge/model-small",
            "checkpoint_step": 3,
            "checkpoint_model_hash": f"blake3:{digest}",
        }
        self.write_json(install / "status.json", install_status)

        wrapper = self.executable(root / "op-wrapper")
        uv = self.executable(root / "uv")
        args = argparse.Namespace(
            install_run=install,
            repo_id="OpenFlowLabs/grug-rule-350m",
            release_root="releases/tool-native-small",
            credential_ref="op://Homelab/Hugging Face Access Token/credential",
            op_wrapper=wrapper,
            uv=uv,
            upload_helper=SCRIPTS / "hf_commit_certified_babyllm.py",
            poll_seconds=5,
            maximum_wait_hours=1.0,
        )
        config = publisher.launch_configuration(args)
        output = root / "publication"
        output.mkdir()
        return config, install_status, release, output, alias

    def prepare(
        self, root: Path
    ) -> tuple[dict[str, object], Path, str, dict[str, object], Path]:
        config, install_status, _release, output, alias = self.fixture(root)
        release, active, provenance = publisher.verify_installed_release(
            config, install_status, output
        )
        with mock.patch.object(
            publisher, "inspect_public_head", return_value="b" * 40
        ):
            packet, digest, plan = publisher.prepare_packet(
                config, output, release, active, provenance
            )
        return config, packet, digest, plan, output

    def test_packet_is_serving_only_and_helper_accepts_exact_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _config, packet, digest, plan, _output = self.prepare(root)
            self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")
            validated_plan, operations = hf_helper.validate_plan(packet)
            self.assertEqual(validated_plan, plan)
            remote = {item["path_in_repo"] for item in operations}
            self.assertIn("README.md", remote)
            self.assertTrue(any(path.endswith("/model.mpk") for path in remote))
            self.assertFalse(any("optim.mpk" in path for path in remote))
            self.assertFalse(any("predictions" in path for path in remote))
            self.assertFalse(
                any(path.endswith("/multimodal-candidate.json") for path in remote)
            )

    def test_extra_optimizer_file_is_refused_before_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, install_status, release, output, _alias = self.fixture(root)
            (release / "checkpoints/candidate_update_00000003/optim.mpk").write_bytes(
                b"adam"
            )
            with self.assertRaisesRegex(RuntimeError, "serving whitelist"):
                publisher.verify_installed_release(config, install_status, output)

    def test_reviews_are_bound_to_exact_models_packet_and_raw_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _config, _packet, digest, plan, output = self.prepare(root)
            reviews = output / "reviews"
            reviews.mkdir()
            for reviewer in publisher.EXPECTED_REVIEWERS:
                stem = publisher.REVIEW_FILENAMES[reviewer].removesuffix(".json")
                prompt = reviews / f"{stem}.prompt.txt"
                response = reviews / f"{stem}.response.txt"
                prompt.write_text(f"Review packet {digest}\n", encoding="utf-8")
                response.write_text(
                    f"PACKET_SHA256: {digest}\nVERDICT: PASS\n",
                    encoding="utf-8",
                )
                value = {
                    "schema_version": 1,
                    "reviewer_model": reviewer,
                    "provider": "test-provider",
                    "provider_response_id": f"response-{stem}",
                    "packet_sha256": digest,
                    "reviewed_parent_commit": plan["parent_commit"],
                    "reviewed_operation_count": len(plan["operations"]),
                    "prompt_file": prompt.name,
                    "prompt_sha256": publisher.loop.sha256_file(prompt),
                    "response_file": response.name,
                    "response_sha256": publisher.loop.sha256_file(response),
                    "verdict": "PASS",
                }
                self.write_json(
                    reviews / publisher.REVIEW_FILENAMES[reviewer], value
                )
                accepted = publisher.validate_review(
                    reviews, reviewer, digest, plan
                )
                self.assertEqual(accepted["verdict"], "PASS")

            kimi_path = reviews / publisher.REVIEW_FILENAMES["moonshotai/kimi-k3"]
            kimi = json.loads(kimi_path.read_text())
            kimi["reviewer_model"] = "substitute/model"
            self.write_json(kimi_path, kimi)
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                publisher.validate_review(
                    reviews, "moonshotai/kimi-k3", digest, plan
                )

    def test_upload_command_contains_reference_but_never_resolved_secret(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, packet, _digest, _plan, _output = self.prepare(root)
            command, environment = publisher.upload_command(config, packet)
            reference = "op://Homelab/Hugging Face Access Token/credential"
            self.assertEqual(environment["HF_TOKEN"], reference)
            self.assertNotIn(reference, command)
            self.assertEqual(command[0], config["op_wrapper"]["path"])
            self.assertIn("--offline", command)
            self.assertNotIn("hf_actual_secret", " ".join(command))


if __name__ == "__main__":
    unittest.main()
