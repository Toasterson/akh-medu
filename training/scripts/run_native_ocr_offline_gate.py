#!/usr/bin/env python3
"""Evaluate and activate a native OCR candidate under a proven network deny.

The Rust evaluator is always invoked with ``--require-network-denied`` and
refuses to load the candidate unless Linux exposes only loopback and no kernel
routes. This supervisor preserves failed/partial attempts and calls the
certifier only after complete checkpoint-bound evidence exists.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import run_tool_native_adaptive_loop as loop


EVIDENCE_FILES = (
    "multimodal-audit-evidence.json",
    "audit-predictions.jsonl",
    "multimodal-text-retention-evidence.json",
    "retention-before-outputs.jsonl",
    "retention-after-outputs.jsonl",
    "multimodal-network-denied-replay.json",
    "network-replay-first.jsonl",
    "network-replay-second.jsonl",
    "completion.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--base-text-bundle", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument(
        "--retention-eval-jsonl", type=Path, action="append", required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-binary", type=Path, required=True)
    parser.add_argument("--certify-binary", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--maximum-image-pixels", type=int, default=16_000_000)
    parser.add_argument("--network-replay-cases", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(
        args.max_new_tokens,
        args.maximum_image_pixels,
        args.network_replay_cases,
    ) <= 0:
        parser.error("evaluation token, pixel, and replay limits must be positive")
    if args.network_replay_cases < 30:
        parser.error("promotion requires at least 30 network-denied replay cases")
    return args


def file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"input is not a regular file: {resolved}")
    return {"path": str(resolved), "sha256": loop.sha256_file(resolved)}


def launch_configuration(
    args: argparse.Namespace, eval_binary: str, certify_binary: str
) -> dict[str, Any]:
    candidate_bundle = args.candidate_bundle.resolve(strict=True)
    base_bundle = args.base_text_bundle.resolve(strict=True)
    return {
        "schema_version": 1,
        "candidate_bundle": str(candidate_bundle),
        "candidate_artifact": file_identity(
            candidate_bundle / "multimodal-candidate.json"
        ),
        "base_text_bundle": str(base_bundle),
        "base_text_artifact": file_identity(base_bundle / "active-model.json"),
        "audit_manifest": file_identity(args.audit_manifest),
        "retention_eval": [file_identity(path) for path in args.retention_eval_jsonl],
        "binaries": {
            "eval": {
                "path": eval_binary,
                "sha256": loop.sha256_file(Path(eval_binary)),
            },
            "certify": {
                "path": certify_binary,
                "sha256": loop.sha256_file(Path(certify_binary)),
            },
        },
        "max_new_tokens": args.max_new_tokens,
        "maximum_image_pixels": args.maximum_image_pixels,
        "network_replay_cases": args.network_replay_cases,
        "require_network_denied": True,
        "promotion_requires_certifier_success": True,
    }


def verify_pins(config: dict[str, Any]) -> None:
    identities = [
        config["candidate_artifact"],
        config["base_text_artifact"],
        config["audit_manifest"],
        *config["retention_eval"],
        *config["binaries"].values(),
    ]
    for identity in identities:
        path = Path(identity["path"])
        if loop.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"pinned release input changed: {path}")


def run_logged(command: list[str], log_path: Path, lock_fd: int) -> None:
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            pass_fds=(lock_fd,),
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command exited {completed.returncode}; inspect {log_path}")


def next_attempt(output: Path) -> Path:
    index = 1
    while (output / f"evidence-attempt-{index:02d}").exists():
        index += 1
    attempt = output / f"evidence-attempt-{index:02d}"
    attempt.mkdir()
    return attempt


def completed_attempts(output: Path) -> list[Path]:
    attempts = [
        path
        for path in output.glob("evidence-attempt-*")
        if path.is_dir() and (path / "offline-evaluation-complete.json").is_file()
    ]
    return sorted(attempts)


def verify_evidence(attempt: Path, candidate: dict[str, Any]) -> None:
    missing = [name for name in EVIDENCE_FILES if not (attempt / name).is_file()]
    if missing:
        raise RuntimeError(f"offline evaluation omitted evidence files: {missing}")
    expected_step = candidate["checkpoint_step"]
    expected_hash = candidate["checkpoint_model_hash"]
    for name in (
        "multimodal-audit-evidence.json",
        "multimodal-text-retention-evidence.json",
        "multimodal-network-denied-replay.json",
    ):
        evidence = loop.load_json(attempt / name)
        if (
            evidence.get("checkpoint_step") != expected_step
            or evidence.get("checkpoint_model_hash") != expected_hash
        ):
            raise RuntimeError(f"{name} is not bound to the candidate checkpoint")
    replay = loop.load_json(attempt / "multimodal-network-denied-replay.json")
    if (
        replay.get("network_denied_verified") is not True
        or replay.get("visible_interfaces") != ["lo"]
        or replay.get("kernel_route_entries") != 0
        or int(replay.get("replay_cases", 0)) < 30
        or replay.get("first_outputs_hash") != replay.get("second_outputs_hash")
    ):
        raise RuntimeError("offline replay does not prove a deterministic network deny")


def eval_command(config: dict[str, Any], attempt: Path) -> list[str]:
    command = [
        config["binaries"]["eval"]["path"],
        "--candidate-bundle",
        config["candidate_bundle"],
        "--base-text-bundle",
        config["base_text_bundle"],
        "--audit-manifest",
        config["audit_manifest"]["path"],
    ]
    for identity in config["retention_eval"]:
        command.extend(["--retention-eval-jsonl", identity["path"]])
    command.extend(
        [
            "--output-dir",
            str(attempt),
            "--max-new-tokens",
            str(config["max_new_tokens"]),
            "--maximum-image-pixels",
            str(config["maximum_image_pixels"]),
            "--network-replay-cases",
            str(config["network_replay_cases"]),
            "--require-network-denied",
        ]
    )
    return command


def certify_command(config: dict[str, Any], attempt: Path) -> list[str]:
    command = [
        config["binaries"]["certify"]["path"],
        "--candidate-bundle",
        config["candidate_bundle"],
        "--audit-evidence",
        str(attempt / "multimodal-audit-evidence.json"),
        "--audit-manifest",
        config["audit_manifest"]["path"],
        "--audit-predictions",
        str(attempt / "audit-predictions.jsonl"),
        "--retention-evidence",
        str(attempt / "multimodal-text-retention-evidence.json"),
    ]
    for identity in config["retention_eval"]:
        command.extend(["--retention-eval-jsonl", identity["path"]])
    command.extend(
        [
            "--retention-before-outputs",
            str(attempt / "retention-before-outputs.jsonl"),
            "--retention-after-outputs",
            str(attempt / "retention-after-outputs.jsonl"),
            "--network-replay-evidence",
            str(attempt / "multimodal-network-denied-replay.json"),
            "--network-replay-first",
            str(attempt / "network-replay-first.jsonl"),
            "--network-replay-second",
            str(attempt / "network-replay-second.jsonl"),
        ]
    )
    return command


def verify_active(bundle: Path, candidate: dict[str, Any]) -> Path:
    active_path = bundle / "multimodal-active.json"
    active = loop.load_json(active_path)
    if (
        active.get("checkpoint_step") != candidate.get("checkpoint_step")
        or active.get("checkpoint_model_hash") != candidate.get("checkpoint_model_hash")
        or not isinstance(active.get("certification"), dict)
        or active.get("checkpoint_optimizer_hash") is not None
    ):
        raise RuntimeError("certifier did not publish the exact serving-safe candidate")
    return active_path


def main() -> int:
    args = parse_args()
    eval_binary = loop.executable(args.eval_binary)
    certify_binary = loop.executable(args.certify_binary)
    output = args.out.resolve()
    bundle = args.candidate_bundle.resolve(strict=True)
    config = launch_configuration(args, eval_binary, certify_binary)
    config_path = output / "launch-config.json"
    status_path = output / "status.json"

    if args.resume:
        if not output.is_dir() or loop.load_json(config_path) != config:
            raise RuntimeError("resume arguments or pinned inputs differ from launch config")
    else:
        if output.exists() or (bundle / "multimodal-active.json").exists():
            raise RuntimeError("new offline gate requires absent output and active pointer")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        loop.atomic_json(config_path, config)

    lock_path = bundle / ".multimodal-release.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another multimodal release process holds the lock") from error

        candidate = loop.load_json(bundle / "multimodal-candidate.json")
        if (bundle / "multimodal-active.json").exists():
            active_path = verify_active(bundle, candidate)
            loop.atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "active_multimodal_model_ready",
                    "candidate_bundle": str(bundle),
                    "active_model": str(active_path),
                    "checkpoint_step": candidate["checkpoint_step"],
                    "checkpoint_model_hash": candidate["checkpoint_model_hash"],
                },
            )
            return 0
        if candidate.get("certification") is not None:
            raise RuntimeError("candidate artifact unexpectedly contains certification")
        verify_pins(config)

        attempts = completed_attempts(output)
        if attempts:
            attempt = attempts[-1]
            verify_evidence(attempt, candidate)
        else:
            attempt = next_attempt(output)
            loop.atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "evaluating_under_network_deny",
                    "candidate_bundle": str(bundle),
                    "evidence_attempt": str(attempt),
                    "active_model_written": False,
                },
            )
            verify_pins(config)
            run_logged(
                eval_command(config, attempt),
                output / f"{attempt.name}.eval.log",
                lock_handle.fileno(),
            )
            verify_evidence(attempt, candidate)
            loop.atomic_json(
                attempt / "offline-evaluation-complete.json",
                {
                    "schema_version": 1,
                    "candidate_checkpoint_step": candidate["checkpoint_step"],
                    "candidate_checkpoint_model_hash": candidate[
                        "checkpoint_model_hash"
                    ],
                    "network_denied_verified": True,
                },
            )

        loop.atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "certifying",
                "candidate_bundle": str(bundle),
                "evidence_attempt": str(attempt),
                "active_model_written": False,
            },
        )
        verify_pins(config)
        cert_log = output / f"{attempt.name}.certify.log"
        if cert_log.exists():
            cert_log = output / f"{attempt.name}.certify-resume.log"
        run_logged(certify_command(config, attempt), cert_log, lock_handle.fileno())
        active_path = verify_active(bundle, candidate)
        loop.atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "active_multimodal_model_ready",
                "candidate_bundle": str(bundle),
                "active_model": str(active_path),
                "evidence_attempt": str(attempt),
                "checkpoint_step": candidate["checkpoint_step"],
                "checkpoint_model_hash": candidate["checkpoint_model_hash"],
            },
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        try:
            failure_args = parse_args()
            failure_out = failure_args.out.resolve()
            if failure_out.is_dir():
                loop.atomic_json(
                    failure_out / "status.json",
                    {
                        "schema_version": 1,
                        "stage": "failed",
                        "error": str(error),
                        "active_model_written": False,
                    },
                )
        except Exception as status_error:
            print(
                f"native OCR offline gate: could not persist failed status: {status_error}",
                file=sys.stderr,
            )
        print(f"native OCR offline gate: {error}", file=sys.stderr)
        raise SystemExit(1)
