#!/usr/bin/env python3
"""Evaluate the final adaptive candidate and activate it only on full gates.

This is intentionally separate from the adaptive training loop. It reuses the
loop's immutable knowledge/all-tool sentinels, executes them once more against
the final candidate, and invokes the Rust certifier. A controller stop, partial
loop, failed exam, or failed threshold leaves the candidate inactive.

``--length-buckets`` is accepted for provenance parity with the adaptive loop
it gates (same comma-separated, strictly increasing positive integer format,
e.g. ``2048,4096,6144``, validated via
``run_tool_native_adaptive_loop.validate_length_buckets``) and is recorded in
``launch-config.json``. This gate re-evaluates the loop's already-materialized
sentinel files rather than generating new episodes, so the value is not
forwarded to any ``knowledge_workflow`` subprocess here -- there isn't one.
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
    parser.add_argument("--adaptive-loop", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--knowledge-eval-binary", type=Path, required=True)
    parser.add_argument("--tool-eval-binary", type=Path, required=True)
    parser.add_argument("--certify-binary", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-wait-hours", type=float, default=24.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--length-buckets",
        type=str,
        default=None,
        help=(
            "comma-separated, strictly increasing positive target token-count buckets "
            "(e.g. 2048,4096,6144), accepted and recorded in launch-config.json for "
            "provenance parity with the adaptive loop's --length-buckets; this gate "
            "re-evaluates already-materialized sentinels so the value is not forwarded "
            "to a knowledge_workflow subprocess. Default: unset."
        ),
    )
    args = parser.parse_args()
    if (
        args.poll_seconds < 5
        or args.maximum_wait_hours <= 0
        or args.max_new_tokens <= 0
    ):
        parser.error("poll must be >=5 and wait/token budgets must be positive")
    if args.length_buckets is not None:
        try:
            loop.validate_length_buckets(args.length_buckets)
        except ValueError as error:
            parser.error(str(error))
    return args


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def wait_for_final_candidate(
    adaptive: Path, poll_seconds: int, maximum_wait_hours: float
) -> tuple[str, dict[str, Any]]:
    status_path = adaptive / "status.json"
    deadline = time.monotonic() + maximum_wait_hours * 3600.0
    while time.monotonic() < deadline:
        if status_path.exists():
            status = loop.load_json(status_path)
            stage = str(status.get("stage", ""))
            if stage in ("window_budget_exhausted", "ready_for_release"):
                return stage, status
            if stage in ("stopped_by_controller", "failed"):
                return stage, status
        time.sleep(poll_seconds)
    raise RuntimeError("release gate wait exceeded the configured deadline")


def verify_final_candidate(status: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    bundle = Path(str(status["bundle"])).resolve(strict=True)
    artifact = loop.load_candidate(bundle)
    if status.get("checkpoint") != loop.checkpoint(artifact):
        raise RuntimeError("adaptive terminal status names a different candidate checkpoint")
    if (bundle / "active-model.json").exists():
        raise RuntimeError("final candidate bundle is already active")
    return bundle, artifact


def main() -> int:
    args = parse_args()
    adaptive = args.adaptive_loop.resolve(strict=True)
    output = args.out.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing release-gate directory {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    status_path = output / "status.json"
    lock_path = adaptive / ".release-gate.lock"
    binaries = {
        "knowledge_eval": loop.executable(args.knowledge_eval_binary),
        "tool_eval": loop.executable(args.tool_eval_binary),
        "certify": loop.executable(args.certify_binary),
    }
    atomic_json(
        output / "launch-config.json",
        {
            "schema_version": 1,
            "adaptive_loop": str(adaptive),
            "binaries": {
                name: {"path": path, "sha256": loop.sha256_file(Path(path))}
                for name, path in sorted(binaries.items())
            },
            "poll_seconds": args.poll_seconds,
            "maximum_wait_hours": args.maximum_wait_hours,
            "max_new_tokens": args.max_new_tokens,
            "length_buckets": args.length_buckets,
            "promotion_requires_certifier_success": True,
        },
    )

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another text release gate holds the lock") from error

        atomic_json(status_path, {"schema_version": 1, "stage": "waiting_for_adaptive_loop"})
        terminal_stage, terminal = wait_for_final_candidate(
            adaptive, args.poll_seconds, args.maximum_wait_hours
        )
        if terminal_stage not in ("window_budget_exhausted", "ready_for_release"):
            atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "blocked_by_adaptive_loop",
                    "adaptive_stage": terminal_stage,
                    "adaptive_status": terminal,
                    "active_model_written": False,
                },
            )
            return 0

        bundle, artifact = verify_final_candidate(terminal)
        loop_config = loop.load_json(adaptive / "loop-config.json")
        knowledge_sentinels, tool_sentinel = loop.load_sentinel_inventory(
            adaptive, loop_config.get("sentinels", {})
        )
        evidence_dir = output / "evidence"
        evidence_dir.mkdir()
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "evaluating_release_sentinels",
                "bundle": str(bundle),
                "checkpoint": loop.checkpoint(artifact),
                "active_model_written": False,
            },
        )

        knowledge_evidence: list[Path] = []
        for sentinel in knowledge_sentinels:
            prefix = evidence_dir / sentinel.stem
            observation = loop.evaluate_knowledge(
                binaries["knowledge_eval"],
                bundle,
                sentinel,
                prefix,
                args.max_new_tokens,
            )
            if (
                observation.get("tier") != "sentinel"
                or observation.get("checkpoint") != loop.checkpoint(artifact)
            ):
                raise RuntimeError("release knowledge evidence is not candidate-bound sentinel data")
            knowledge_evidence.append(prefix.with_suffix(".evidence.json"))

        tool_prefix = evidence_dir / "all-tools"
        tool_observations = loop.evaluate_tools(
            binaries["tool_eval"],
            bundle,
            tool_sentinel,
            tool_prefix,
            args.max_new_tokens,
        )
        expected_tools = int(loop_config["tool_sentinel_samples"])
        if not tool_observations or any(
            observation.get("tier") != "sentinel"
            or observation.get("checkpoint") != loop.checkpoint(artifact)
            or int(observation.get("sample_count", 0)) != expected_tools
            for observation in tool_observations
        ):
            raise RuntimeError("release tool evidence is incomplete or not candidate-bound")
        tool_evidence = tool_prefix.with_suffix(".evidence.json")

        command = [binaries["certify"], "--bundle", str(bundle)]
        for path in knowledge_evidence:
            command.extend(["--evidence", str(path)])
        command.extend(["--tool-evidence", str(tool_evidence)])
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "certifying",
                "bundle": str(bundle),
                "checkpoint": loop.checkpoint(artifact),
                "knowledge_evidence": [str(path) for path in knowledge_evidence],
                "tool_evidence": str(tool_evidence),
                "active_model_written": False,
            },
        )
        loop.run_logged(command, output / "certify.log")
        active_path = bundle / "active-model.json"
        active = loop.load_json(active_path)
        if (
            active.get("checkpoint_step") != artifact["checkpoint_step"]
            or active.get("checkpoint_model_hash") != artifact["checkpoint_model_hash"]
            or not isinstance(active.get("certification"), dict)
        ):
            raise RuntimeError("certifier returned success without the exact active artifact")
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "active_text_model_ready",
                "bundle": str(bundle),
                "checkpoint": loop.checkpoint(artifact),
                "active_model": str(active_path),
                "active_model_written": True,
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
                atomic_json(
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
                f"text release gate: could not persist failed status: {status_error}",
                file=sys.stderr,
            )
        print(f"text release gate: {error}", file=sys.stderr)
        raise SystemExit(1)
