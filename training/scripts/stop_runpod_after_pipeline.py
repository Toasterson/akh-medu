#!/usr/bin/env python3
"""Stop (never delete) a Runpod pod after a bounded BabyLLM GPU pipeline.

The guard holds no credential while waiting.  At successful staging, terminal
pipeline failure, or the configured deadline, it verifies its pinned tools and
uses the machine-local 1Password wrapper to inject ``RUNPOD_API_KEY`` only into
the short-lived ``runpodctl pod stop`` process.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import run_tool_native_adaptive_loop as loop


POD_ID = re.compile(r"^[a-z0-9]{6,32}$")
TERMINAL_STAGES = {
    "active_multimodal_model_ready",
    "blocked_by_remote_training",
    "blocked_by_text_release",
    "failed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-host-run", type=Path, required=True)
    parser.add_argument("--pod-id", required=True)
    parser.add_argument("--credential-ref", required=True)
    parser.add_argument("--op-wrapper", type=Path, required=True)
    parser.add_argument("--runpodctl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-wait-hours", type=float, default=36.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not POD_ID.fullmatch(args.pod_id):
        parser.error("pod id contains unsupported characters")
    if not args.credential_ref.startswith("op://"):
        parser.error("credential-ref must be a 1Password secret reference")
    if args.poll_seconds < 5 or args.maximum_wait_hours <= 0:
        parser.error("poll must be >=5 seconds and wait budget positive")
    return args


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"pinned executable is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": loop.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def launch_configuration(args: argparse.Namespace) -> dict[str, Any]:
    cross_host = args.cross_host_run.resolve(strict=True)
    return {
        "schema_version": 1,
        "cross_host_run": str(cross_host),
        "cross_host_launch_config": file_identity(cross_host / "launch-config.json"),
        "pod_id": args.pod_id,
        "credential_ref": args.credential_ref,
        "op_wrapper": file_identity(args.op_wrapper),
        "runpodctl": file_identity(args.runpodctl),
        "poll_seconds": args.poll_seconds,
        "maximum_wait_hours": args.maximum_wait_hours,
        "operation": "stop_never_delete",
    }


def verify_pins(config: dict[str, Any]) -> None:
    for key in ("cross_host_launch_config", "op_wrapper", "runpodctl"):
        identity = config[key]
        if loop.sha256_file(Path(identity["path"])) != identity["sha256"]:
            raise RuntimeError(f"pinned cost-guard input changed: {identity['path']}")


def trigger_reason(cross_host: Path) -> str | None:
    if (cross_host / "staged/stage-complete.json").is_file():
        return "candidate_staged_locally"
    status_path = cross_host / "status.json"
    if status_path.is_file():
        stage = str(loop.load_json(status_path).get("stage", ""))
        if stage in TERMINAL_STAGES:
            return f"pipeline_terminal:{stage}"
    return None


def stop_command(config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    environment["RUNPOD_API_KEY"] = config["credential_ref"]
    command = [
        config["op_wrapper"]["path"],
        "run",
        "--",
        config["runpodctl"]["path"],
        "pod",
        "stop",
        config["pod_id"],
        "-o",
        "json",
    ]
    return command, environment


def write_status(path: Path, stage: str, **values: Any) -> None:
    loop.atomic_json(
        path,
        {
            "schema_version": 1,
            "stage": stage,
            "pod_deleted": False,
            **values,
        },
    )


def main() -> int:
    args = parse_args()
    output = args.out.resolve()
    config = launch_configuration(args)
    config_path = output / "launch-config.json"
    status_path = output / "status.json"
    if args.resume:
        if not output.is_dir() or loop.load_json(config_path) != config:
            raise RuntimeError("resume arguments or pinned inputs differ from launch config")
    else:
        if output.exists():
            raise RuntimeError("new cost guard requires an absent output directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        loop.atomic_json(config_path, config)

    with (output / ".guard.lock").open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Runpod cost guard holds the lock") from error

        deadline = time.monotonic() + float(config["maximum_wait_hours"]) * 3600.0
        cross_host = Path(config["cross_host_run"])
        reason = trigger_reason(cross_host)
        while reason is None and time.monotonic() < deadline:
            write_status(
                status_path,
                "waiting_for_pipeline_terminal",
                pod_id=config["pod_id"],
                operation="stop",
            )
            time.sleep(float(config["poll_seconds"]))
            reason = trigger_reason(cross_host)
        if reason is None:
            reason = "maximum_wait_exceeded"

        verify_pins(config)
        write_status(
            status_path,
            "stopping_pod",
            pod_id=config["pod_id"],
            reason=reason,
            operation="stop",
        )
        command, environment = stop_command(config)
        with (output / "runpod-stop.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"runpodctl pod stop exited {completed.returncode}; inspect runpod-stop.log"
            )
        write_status(
            status_path,
            "pod_stopped",
            pod_id=config["pod_id"],
            reason=reason,
            operation="stop",
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
                write_status(
                    failure_out / "status.json",
                    "failed",
                    error=str(error),
                    operation="stop",
                )
        except Exception as status_error:
            print(
                f"Runpod cost guard: could not persist failure: {status_error}",
                file=sys.stderr,
            )
        print(f"Runpod cost guard: {error}", file=sys.stderr)
        raise SystemExit(1)
