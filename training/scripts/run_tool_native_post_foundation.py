#!/usr/bin/env python3
"""Fail-closed bridge from a completed foundation run to one SFT epoch.

This supervisor deliberately stops before adaptive checkpoint exams. It is
safe to leave beside a long GPU run: it only begins SFT after the exact final
foundation artifact is present and `train_knowledge --validate-only` accepts
the full corpus, tokenizer, contracts, split isolation, and checkpoint hashes.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXPECTED_LANGUAGES = {"de", "en", "fr"}
MINIMUM_FOUNDATION_TOKENS = 819_200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--train-binary", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-wait-hours", type=float, default=12.0)
    args = parser.parse_args()
    if args.poll_seconds < 5 or args.maximum_wait_hours <= 0:
        parser.error("--poll-seconds must be >= 5 and --maximum-wait-hours positive")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def read_last_step(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell() - 1
        while position > 0:
            handle.seek(position)
            if handle.read(1) == b"\n" and position < handle.seek(0, os.SEEK_END) - 1:
                break
            position -= 1
        handle.seek(max(position, 0))
        line = handle.readline().strip()
        if not line:
            line = handle.readline().strip()
    if not line:
        return None
    value = json.loads(line)
    return int(value["step"])


def run_logged(command: list[str], log_path: Path) -> None:
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command exited {completed.returncode}; inspect {log_path}: {command[0]}"
        )


def validate_foundation(foundation: Path) -> dict[str, Any]:
    run = load_json(foundation / "run.json")
    artifact = load_json(foundation / "foundation-model.json")
    shape = artifact.get("shape")
    if not isinstance(shape, dict):
        raise RuntimeError("foundation artifact has no model shape")
    expected_step = int(run["steps"]) - 1
    expected_tokens = int(run["steps"]) * int(run["batch_size"]) * int(
        shape["max_seq_len"]
    )
    if int(artifact["checkpoint_step"]) != expected_step:
        raise RuntimeError(
            f"foundation stopped at {artifact['checkpoint_step']}, expected {expected_step}"
        )
    if int(artifact["training_tokens_seen"]) != expected_tokens:
        raise RuntimeError(
            "foundation token count disagrees with steps x batch x context"
        )
    if expected_tokens < MINIMUM_FOUNDATION_TOKENS:
        raise RuntimeError(
            f"foundation exposed {expected_tokens} tokens, below {MINIMUM_FOUNDATION_TOKENS}"
        )
    if set(artifact.get("languages", [])) != EXPECTED_LANGUAGES:
        raise RuntimeError(
            f"foundation languages are {artifact.get('languages')}, expected de/en/fr"
        )
    checkpoint = foundation / str(artifact["checkpoint_dir"])
    for name in ("meta.json", "model.mpk", "optim.mpk"):
        candidate = checkpoint / name
        if not candidate.is_file():
            raise RuntimeError(f"foundation checkpoint is missing {candidate}")
    if read_last_step(foundation / "train.jsonl") != expected_step:
        raise RuntimeError("current foundation training segment does not end at final step")
    return artifact


def knowledge_command(
    binary: Path, base: Path, foundation: Path, artifacts: Path
) -> list[str]:
    curriculum = base / "local-curricula-20260904" / "sft"
    paths = {
        "contracts": curriculum / "contracts.json",
        "knowledge_train": curriculum / "knowledge.train.sft.jsonl",
        "dojo_train": curriculum / "dojo.train.sft.jsonl",
        "tools_train": curriculum / "tools.train.sft.jsonl",
        "knowledge_eval": curriculum / "knowledge.eval.sft.jsonl",
        "dojo_eval": curriculum / "dojo.eval.sft.jsonl",
        "tools_eval": curriculum / "tools.eval.sft.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"knowledge curriculum files are missing: {missing}")
    return [
        str(binary),
        "--contracts",
        str(paths["contracts"]),
        "--train-jsonl",
        str(paths["knowledge_train"]),
        "--supplemental-jsonl",
        str(paths["dojo_train"]),
        "--supplemental-jsonl",
        str(paths["tools_train"]),
        "--eval-jsonl",
        str(paths["knowledge_eval"]),
        "--supplemental-eval-jsonl",
        str(paths["dojo_eval"]),
        "--supplemental-eval-jsonl",
        str(paths["tools_eval"]),
        "--foundation-bundle",
        str(foundation),
        "--artifacts-dir",
        str(artifacts),
        "--model",
        "small",
        "--epochs",
        "1",
        "--batch-size",
        "8",
        "--learning-rate",
        "0.0001",
        "--warmup-steps",
        "250",
        "--bpe-vocab-size",
        "8000",
        "--eval-every",
        "1000",
        "--eval-samples",
        "8",
        "--eval-max-new-tokens",
        "128",
        "--ckpt-every",
        "1000",
    ]


def main() -> int:
    args = parse_args()
    base = args.base.resolve(strict=True)
    binary = args.train_binary.resolve(strict=True)
    foundation = base / "foundation-model-small-v1"
    knowledge = base / "knowledge-model-small-v1"
    status_path = base / "post-foundation-supervisor.json"
    lock_path = base / ".post-foundation-supervisor.lock"
    if knowledge.exists():
        raise RuntimeError(f"refusing existing knowledge artifact directory {knowledge}")

    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another post-foundation supervisor holds the lock") from error

        pid_path = foundation / "foundation-train.pid"
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        started = time.monotonic()
        deadline = started + args.maximum_wait_hours * 3600.0
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "waiting_for_foundation",
                "foundation_pid": pid,
                "supervisor_pid": os.getpid(),
            },
        )
        next_status = 0.0
        while process_alive(pid):
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError("foundation wait exceeded the configured deadline")
            if now >= next_status:
                atomic_json(
                    status_path,
                    {
                        "schema_version": 1,
                        "stage": "waiting_for_foundation",
                        "foundation_pid": pid,
                        "foundation_step": read_last_step(foundation / "train.jsonl"),
                        "supervisor_pid": os.getpid(),
                    },
                )
                next_status = now + 300.0
            time.sleep(args.poll_seconds)

        foundation_artifact = validate_foundation(foundation)
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "validating_knowledge_sft",
                "foundation_checkpoint_step": foundation_artifact["checkpoint_step"],
                "foundation_model_hash": foundation_artifact["checkpoint_model_hash"],
                "supervisor_pid": os.getpid(),
            },
        )
        command = knowledge_command(binary, base, foundation, knowledge)
        run_logged(command + ["--validate-only"], base / "knowledge-sft-validate.log")

        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "training_first_sft_epoch",
                "foundation_checkpoint_step": foundation_artifact["checkpoint_step"],
                "foundation_model_hash": foundation_artifact["checkpoint_model_hash"],
                "supervisor_pid": os.getpid(),
            },
        )
        run_logged(command, knowledge / "knowledge-train.epoch-1.log")

        candidate = load_json(knowledge / "candidate-model.json")
        run = load_json(knowledge / "run.json")
        expected_step = int(run["training_records"] + run["batch_size"] - 1) // int(
            run["batch_size"]
        ) - 1
        if int(candidate["checkpoint_step"]) != expected_step:
            raise RuntimeError(
                f"first SFT epoch stopped at {candidate['checkpoint_step']}, expected {expected_step}"
            )
        if candidate.get("certification") is not None or (
            knowledge / "active-model.json"
        ).exists():
            raise RuntimeError("the first-epoch checkpoint was unexpectedly activated")
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "ready_for_adaptive_probe",
                "foundation_checkpoint_step": foundation_artifact["checkpoint_step"],
                "candidate_checkpoint_step": candidate["checkpoint_step"],
                "candidate_model_hash": candidate["checkpoint_model_hash"],
                "supervisor_pid": os.getpid(),
            },
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        # Preserve a machine-readable terminal state for downstream watchers.
        # This best-effort write must never hide the original supervisor error.
        try:
            failure_args = parse_args()
            atomic_json(
                failure_args.base / "post-foundation-supervisor.json",
                {
                    "schema_version": 1,
                    "stage": "failed",
                    "error": str(error),
                    "supervisor_pid": os.getpid(),
                },
            )
        except Exception as status_error:
            print(
                f"post-foundation supervisor: could not persist failed status: {status_error}",
                file=sys.stderr,
            )
        print(f"post-foundation supervisor: {error}", file=sys.stderr)
        raise SystemExit(1)
