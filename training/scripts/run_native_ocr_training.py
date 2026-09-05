#!/usr/bin/env python3
"""Wait for a certified text/tool model and train a native OCR candidate.

This supervisor never activates a model. It content-pins every executable and
corpus, waits for the separate text release gate, and runs the interruption-
safe Rust trainer. Promotion remains the responsibility of the network-denied
offline gate.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import run_tool_native_adaptive_loop as loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-release-gate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--train-binary", type=Path, required=True)
    parser.add_argument("--vision-train-manifest", type=Path, required=True)
    parser.add_argument("--vision-audit-manifest", type=Path, required=True)
    parser.add_argument("--audit-validation", type=Path, required=True)
    parser.add_argument(
        "--retention-train-jsonl", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--retention-eval-jsonl", type=Path, action="append", required=True
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-wait-hours", type=float, default=36.0)
    parser.add_argument("--canvas-width", type=int, default=480)
    parser.add_argument("--canvas-height", type=int, default=192)
    parser.add_argument("--vision-hidden-channels", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stem-warmup-updates", type=int, default=500)
    parser.add_argument("--joint-epochs", type=int, default=3)
    parser.add_argument("--retention-every", type=int, default=1)
    parser.add_argument("--stem-learning-rate", type=float, default=3e-4)
    parser.add_argument("--joint-learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--loss-cap", type=float, default=20.0)
    parser.add_argument("--grad-norm-cap", type=float, default=100.0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--audit-eval-semantic-items", type=int, default=15)
    parser.add_argument("--retention-eval-samples", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--maximum-image-pixels", type=int, default=16_000_000)
    parser.add_argument("--seed", type=int, default=120_400)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    positive = (
        args.poll_seconds,
        args.maximum_wait_hours,
        args.canvas_width,
        args.canvas_height,
        args.vision_hidden_channels,
        args.batch_size,
        args.joint_epochs,
        args.retention_every,
        args.stem_learning_rate,
        args.joint_learning_rate,
        args.minimum_learning_rate,
        args.loss_cap,
        args.grad_norm_cap,
        args.checkpoint_every,
        args.audit_eval_semantic_items,
        args.retention_eval_samples,
        args.max_new_tokens,
        args.maximum_image_pixels,
    )
    if args.poll_seconds < 5 or any(value <= 0 for value in positive[1:]):
        parser.error("poll must be >=5 and all budgets, dimensions, and rates positive")
    if args.stem_warmup_updates < 0:
        parser.error("stem warmup updates cannot be negative")
    return args


def file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"input is not a regular file: {resolved}")
    return {"path": str(resolved), "sha256": loop.sha256_file(resolved)}


def launch_configuration(args: argparse.Namespace, binary: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "text_release_gate": str(args.text_release_gate.resolve(strict=True)),
        "artifacts_dir": str(args.artifacts_dir.resolve()),
        "train_binary": {
            "path": binary,
            "sha256": loop.sha256_file(Path(binary)),
        },
        "vision_train_manifest": file_identity(args.vision_train_manifest),
        "vision_audit_manifest": file_identity(args.vision_audit_manifest),
        "audit_validation": file_identity(args.audit_validation),
        "retention_train": [file_identity(path) for path in args.retention_train_jsonl],
        "retention_eval": [file_identity(path) for path in args.retention_eval_jsonl],
        "poll_seconds": args.poll_seconds,
        "maximum_wait_hours": args.maximum_wait_hours,
        "canvas_width": args.canvas_width,
        "canvas_height": args.canvas_height,
        "vision_hidden_channels": args.vision_hidden_channels,
        "batch_size": args.batch_size,
        "stem_warmup_updates": args.stem_warmup_updates,
        "joint_epochs": args.joint_epochs,
        "retention_every": args.retention_every,
        "stem_learning_rate": args.stem_learning_rate,
        "joint_learning_rate": args.joint_learning_rate,
        "minimum_learning_rate": args.minimum_learning_rate,
        "loss_cap": args.loss_cap,
        "grad_norm_cap": args.grad_norm_cap,
        "checkpoint_every": args.checkpoint_every,
        "audit_eval_semantic_items": args.audit_eval_semantic_items,
        "retention_eval_samples": args.retention_eval_samples,
        "max_new_tokens": args.max_new_tokens,
        "maximum_image_pixels": args.maximum_image_pixels,
        "seed": args.seed,
        "promotion_allowed": False,
    }


def verify_pins(config: dict[str, Any]) -> None:
    identities = [
        config["train_binary"],
        config["vision_train_manifest"],
        config["vision_audit_manifest"],
        config["audit_validation"],
        *config["retention_train"],
        *config["retention_eval"],
    ]
    for identity in identities:
        path = Path(identity["path"])
        if loop.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"pinned input changed after launch: {path}")


def wait_for_text_model(
    release_gate: Path, poll_seconds: int, maximum_wait_hours: float
) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + maximum_wait_hours * 3600.0
    status_path = release_gate / "status.json"
    while time.monotonic() < deadline:
        if status_path.exists():
            status = loop.load_json(status_path)
            stage = str(status.get("stage", ""))
            if stage == "active_text_model_ready":
                return stage, status
            if stage in ("blocked_by_adaptive_loop", "failed"):
                return stage, status
        time.sleep(poll_seconds)
    raise RuntimeError("native OCR training wait exceeded the configured deadline")


def verify_text_model(status: dict[str, Any]) -> Path:
    bundle = Path(str(status["bundle"])).resolve(strict=True)
    active_path = Path(str(status["active_model"])).resolve(strict=True)
    if active_path != bundle / "active-model.json":
        raise RuntimeError("text release status points outside its active bundle")
    active = loop.load_json(active_path)
    expected = status.get("checkpoint")
    observed = {
        "step": active.get("checkpoint_step"),
        "artifact_hash": active.get("checkpoint_model_hash"),
    }
    if expected != observed or not isinstance(active.get("certification"), dict):
        raise RuntimeError("text active artifact is not the certified release checkpoint")
    return bundle


def latest_checkpoint(artifacts: Path) -> Path | None:
    checkpoint_root = artifacts / "checkpoints"
    if not checkpoint_root.is_dir():
        return None
    candidates: list[tuple[int, bool, Path]] = []
    for path in checkpoint_root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        meta_path = path / "meta.json"
        if not meta_path.is_file():
            continue
        meta = loop.load_json(meta_path)
        update = int(meta.get("update", -1))
        complete = bool(meta.get("complete", False))
        expected = (
            f"candidate_update_{update:08d}" if complete else f"update_{update:08d}"
        )
        if update < 0 or path.name != expected:
            continue
        candidates.append((update, complete, path.resolve(strict=True)))
    if not candidates:
        return None
    candidates.sort(key=lambda value: (value[0], value[1]))
    return candidates[-1][2]


def next_log(output: Path) -> Path:
    index = 1
    while True:
        path = output / f"train-attempt-{index:02d}.log"
        if not path.exists():
            return path
        index += 1


def train_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    text_bundle: Path,
    resume_checkpoint: Path | None,
) -> list[str]:
    command = [
        config["train_binary"]["path"],
        "--text-bundle",
        str(text_bundle),
        "--vision-train-manifest",
        config["vision_train_manifest"]["path"],
        "--vision-audit-manifest",
        config["vision_audit_manifest"]["path"],
        "--audit-validation",
        config["audit_validation"]["path"],
    ]
    for identity in config["retention_train"]:
        command.extend(["--retention-train-jsonl", identity["path"]])
    for identity in config["retention_eval"]:
        command.extend(["--retention-eval-jsonl", identity["path"]])
    command.extend(
        [
            "--artifacts-dir",
            config["artifacts_dir"],
            "--canvas-width",
            str(args.canvas_width),
            "--canvas-height",
            str(args.canvas_height),
            "--vision-hidden-channels",
            str(args.vision_hidden_channels),
            "--batch-size",
            str(args.batch_size),
            "--stem-warmup-updates",
            str(args.stem_warmup_updates),
            "--joint-epochs",
            str(args.joint_epochs),
            "--retention-every",
            str(args.retention_every),
            "--stem-learning-rate",
            str(args.stem_learning_rate),
            "--joint-learning-rate",
            str(args.joint_learning_rate),
            "--minimum-learning-rate",
            str(args.minimum_learning_rate),
            "--loss-cap",
            str(args.loss_cap),
            "--grad-norm-cap",
            str(args.grad_norm_cap),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--audit-eval-semantic-items",
            str(args.audit_eval_semantic_items),
            "--retention-eval-samples",
            str(args.retention_eval_samples),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--maximum-image-pixels",
            str(args.maximum_image_pixels),
            "--seed",
            str(args.seed),
        ]
    )
    if resume_checkpoint is not None:
        command.extend(["--resume", str(resume_checkpoint)])
    return command


def verify_candidate(artifacts: Path) -> dict[str, Any]:
    completion = loop.load_json(artifacts / "completion.json")
    candidate = loop.load_json(artifacts / "multimodal-candidate.json")
    if (
        not completion.get("candidate_only")
        or completion.get("active_model_written")
        or candidate.get("certification") is not None
        or (artifacts / "multimodal-active.json").exists()
    ):
        raise RuntimeError("OCR trainer did not leave an inactive candidate-only artifact")
    checkpoint_dir = (artifacts / str(candidate["checkpoint_dir"])).resolve(strict=True)
    if checkpoint_dir.parent != (artifacts / "checkpoints").resolve(strict=True):
        raise RuntimeError("multimodal candidate checkpoint escapes the artifact bundle")
    metadata = loop.load_json(checkpoint_dir / "meta.json")
    if (
        metadata.get("model_hash") != candidate.get("checkpoint_model_hash")
        or metadata.get("optimizer_hash") != candidate.get("checkpoint_optimizer_hash")
        or not metadata.get("complete")
    ):
        raise RuntimeError("multimodal candidate identity differs from its complete checkpoint")
    return candidate


def run_training(command: list[str], log_path: Path, lock_fd: int) -> None:
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
        raise RuntimeError(
            f"OCR trainer exited {completed.returncode}; inspect {log_path}"
        )


def main() -> int:
    args = parse_args()
    binary = loop.executable(args.train_binary)
    output = args.out.resolve()
    artifacts = args.artifacts_dir.resolve()
    config = launch_configuration(args, binary)
    config_path = output / "launch-config.json"
    status_path = output / "status.json"

    if args.resume:
        if not output.is_dir():
            raise RuntimeError("--resume requires the existing supervisor directory")
        if loop.load_json(config_path) != config:
            raise RuntimeError("resume arguments or pinned inputs differ from launch config")
    else:
        if output.exists() or artifacts.exists():
            raise RuntimeError("new OCR training requires absent supervisor and artifact paths")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        loop.atomic_json(config_path, config)

    with (output / ".training.lock").open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another OCR training supervisor or child holds the lock") from error

        if (artifacts / "completion.json").exists():
            candidate = verify_candidate(artifacts)
            loop.atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "candidate_ready_for_offline_gate",
                    "artifacts_dir": str(artifacts),
                    "checkpoint_step": candidate["checkpoint_step"],
                    "checkpoint_model_hash": candidate["checkpoint_model_hash"],
                    "active_model_written": False,
                },
            )
            return 0

        loop.atomic_json(
            status_path, {"schema_version": 1, "stage": "waiting_for_text_release"}
        )
        release_stage, release_status = wait_for_text_model(
            args.text_release_gate.resolve(strict=True),
            args.poll_seconds,
            args.maximum_wait_hours,
        )
        if release_stage != "active_text_model_ready":
            loop.atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "blocked_by_text_release",
                    "text_release_stage": release_stage,
                    "text_release_status": release_status,
                    "active_model_written": False,
                },
            )
            return 0
        text_bundle = verify_text_model(release_status)
        verify_pins(config)

        checkpoint = latest_checkpoint(artifacts)
        if artifacts.exists() and checkpoint is None:
            raise RuntimeError(
                "incomplete OCR artifact directory has no immutable resume checkpoint"
            )
        log_path = next_log(output)
        command = train_command(args, config, text_bundle, checkpoint)
        loop.atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "training",
                "text_bundle": str(text_bundle),
                "artifacts_dir": str(artifacts),
                "resume_checkpoint": str(checkpoint) if checkpoint else None,
                "log": str(log_path),
                "active_model_written": False,
            },
        )
        verify_pins(config)
        run_training(command, log_path, lock_handle.fileno())
        candidate = verify_candidate(artifacts)
        loop.atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "candidate_ready_for_offline_gate",
                "artifacts_dir": str(artifacts),
                "checkpoint_step": candidate["checkpoint_step"],
                "checkpoint_model_hash": candidate["checkpoint_model_hash"],
                "active_model_written": False,
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
                f"native OCR training: could not persist failed status: {status_error}",
                file=sys.stderr,
            )
        print(f"native OCR training: {error}", file=sys.stderr)
        raise SystemExit(1)
