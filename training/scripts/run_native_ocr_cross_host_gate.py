#!/usr/bin/env python3
"""Stage a remote OCR candidate and run the local network-denied release gate.

The remote GPU only trains. This watcher waits for its candidate-only terminal
status, transfers the minimum content-addressed text and multimodal bundles,
then launches the pinned CPU release image with Docker networking disabled.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import run_tool_native_adaptive_loop as loop


SAFE_REMOTE = re.compile(r"^[A-Za-z0-9_./-]+$")
SAFE_TARGET = re.compile(r"^[A-Za-z0-9_.@-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssh-target", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--known-hosts", type=Path, required=True)
    parser.add_argument("--remote-base", required=True)
    parser.add_argument("--remote-training-status", required=True)
    parser.add_argument("--remote-release-status", required=True)
    parser.add_argument("--remote-audit-manifest", required=True)
    parser.add_argument(
        "--remote-retention-eval", action="append", required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline-image", required=True)
    parser.add_argument("--ssh-binary", type=Path, default=Path("/usr/bin/ssh"))
    parser.add_argument("--rsync-binary", type=Path, default=Path("/usr/bin/rsync"))
    parser.add_argument(
        "--docker-binary", type=Path, default=Path("/opt/homebrew/bin/docker")
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-wait-hours", type=float, default=36.0)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="generation budget forwarded to the offline gate inside the image (default: the image script default, 256; ADR 104 D6 lanes use 768)",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.ssh_port < 1 or args.ssh_port > 65535:
        parser.error("SSH port must be in 1..65535")
    if args.poll_seconds < 5 or args.maximum_wait_hours <= 0:
        parser.error("poll must be >=5 seconds and wait budget positive")
    if not SAFE_TARGET.fullmatch(args.ssh_target):
        parser.error("SSH target contains unsupported characters")
    for value in (
        args.remote_base,
        args.remote_training_status,
        args.remote_release_status,
        args.remote_audit_manifest,
        *args.remote_retention_eval,
    ):
        if not value.startswith("/") or not SAFE_REMOTE.fullmatch(value):
            parser.error(f"remote path is not a safe absolute path: {value}")
    return args


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    ).stdout


def ssh_base(args: argparse.Namespace) -> list[str]:
    return [
        str(args.ssh_binary.resolve(strict=True)),
        "-p",
        str(args.ssh_port),
        "-o",
        f"UserKnownHostsFile={args.known_hosts.resolve(strict=True)}",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        args.ssh_target,
    ]


def remote_json(args: argparse.Namespace, path: str) -> dict[str, Any]:
    value = json.loads(command_output(ssh_base(args) + [f"cat -- {shlex.quote(path)}"]))
    if not isinstance(value, dict):
        raise RuntimeError(f"remote {path} is not a JSON object")
    return value


def docker_image_id(binary: Path, image: str) -> str:
    value = command_output(
        [str(binary.resolve(strict=True)), "image", "inspect", image, "--format", "{{.Id}}"]
    )
    image_id = value.strip()
    if not image_id.startswith("sha256:"):
        raise RuntimeError(f"Docker returned an invalid image identity: {image_id}")
    return image_id


def file_identity(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"executable is not a regular file: {resolved}")
    return {"path": str(resolved), "sha256": loop.sha256_file(resolved)}


def launch_configuration(args: argparse.Namespace) -> dict[str, Any]:
    known_hosts = args.known_hosts.resolve(strict=True)
    binaries = {
        "ssh": file_identity(args.ssh_binary),
        "rsync": file_identity(args.rsync_binary),
        "docker": file_identity(args.docker_binary),
    }
    return {
        "schema_version": 1,
        "ssh_target": args.ssh_target,
        "ssh_port": args.ssh_port,
        "known_hosts": {
            "path": str(known_hosts),
            "sha256": loop.sha256_file(known_hosts),
        },
        "remote_base": args.remote_base,
        "remote_training_status": args.remote_training_status,
        "remote_release_status": args.remote_release_status,
        "remote_audit_manifest": args.remote_audit_manifest,
        "remote_retention_eval": list(args.remote_retention_eval),
        "offline_image": args.offline_image,
        "offline_image_id": docker_image_id(args.docker_binary, args.offline_image),
        "binaries": binaries,
        "poll_seconds": args.poll_seconds,
        "maximum_wait_hours": args.maximum_wait_hours,
        "max_new_tokens": args.max_new_tokens,
        "docker_network": "none",
    }


def verify_launch_pins(config: dict[str, Any]) -> None:
    known_hosts = Path(config["known_hosts"]["path"])
    if loop.sha256_file(known_hosts) != config["known_hosts"]["sha256"]:
        raise RuntimeError("SSH known-hosts file changed after launch")
    for identity in config["binaries"].values():
        if loop.sha256_file(Path(identity["path"])) != identity["sha256"]:
            raise RuntimeError(f"cross-host executable changed: {identity['path']}")
    if (
        docker_image_id(Path(config["binaries"]["docker"]["path"]), config["offline_image"])
        != config["offline_image_id"]
    ):
        raise RuntimeError("offline image tag no longer names the pinned image")


def wait_for_candidate(
    args: argparse.Namespace, status_path: Path
) -> dict[str, Any] | None:
    deadline = time.monotonic() + args.maximum_wait_hours * 3600.0
    last_error = ""
    while time.monotonic() < deadline:
        try:
            status = remote_json(args, args.remote_training_status)
            stage = str(status.get("stage", ""))
            if stage == "candidate_ready_for_offline_gate":
                return status
            if stage in ("blocked_by_text_release", "failed"):
                loop.atomic_json(
                    status_path,
                    {
                        "schema_version": 1,
                        "stage": "blocked_by_remote_training",
                        "remote_status": status,
                        "active_model_written": False,
                    },
                )
                return None
            last_error = ""
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            last_error = str(error)
        loop.atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "waiting_for_remote_candidate",
                "last_connection_error": last_error or None,
                "active_model_written": False,
            },
        )
        time.sleep(args.poll_seconds)
    raise RuntimeError("cross-host candidate wait exceeded the configured deadline")


def ensure_remote_descendant(base: str, child: str) -> str:
    base_path = PurePosixPath(base)
    child_path = PurePosixPath(child)
    if not child_path.is_absolute() or child_path == base_path:
        raise RuntimeError(f"remote artifact path is invalid: {child}")
    try:
        child_path.relative_to(base_path)
    except ValueError as error:
        raise RuntimeError(f"remote artifact escapes the configured base: {child}") from error
    if not SAFE_REMOTE.fullmatch(str(child_path)):
        raise RuntimeError(f"remote artifact path is unsafe: {child}")
    return str(child_path)


def safe_bundle_child(value: Any) -> str:
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise RuntimeError(f"bundle artifact path is unsafe: {value}")
    rendered = str(path)
    if rendered == "." or not SAFE_REMOTE.fullmatch(rendered):
        raise RuntimeError(f"bundle artifact path is unsafe: {value}")
    return rendered


def rsync_transport(args: argparse.Namespace) -> str:
    return shlex.join(ssh_base(args)[:-1])


def rsync(
    args: argparse.Namespace,
    remote_path: str,
    destination: Path,
    transfer_log: Path,
    directory: bool = False,
) -> None:
    source = f"{args.ssh_target}:{remote_path}{'/' if directory else ''}"
    if directory:
        destination.mkdir(parents=True, exist_ok=True)
        target = f"{destination}/"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = str(destination)
    with transfer_log.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                str(args.rsync_binary.resolve(strict=True)),
                "-rt",
                "--partial",
                "-e",
                rsync_transport(args),
                source,
                target,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"rsync exited {completed.returncode}; inspect {transfer_log}"
        )


def stage_model_bundle(
    args: argparse.Namespace,
    remote_bundle: str,
    local_bundle: Path,
    manifest_name: str,
    transfer_log: Path,
) -> dict[str, Any]:
    local_bundle.mkdir(parents=True, exist_ok=True)
    manifest_path = local_bundle / manifest_name
    rsync(args, f"{remote_bundle}/{manifest_name}", manifest_path, transfer_log)
    artifact = loop.load_json(manifest_path)
    children = {
        safe_bundle_child(artifact["tokenizer_file"]),
        safe_bundle_child(artifact["contracts_file"]),
    }
    for child in sorted(children):
        rsync(args, f"{remote_bundle}/{child}", local_bundle / child, transfer_log)
    checkpoint = safe_bundle_child(artifact["checkpoint_dir"])
    rsync(
        args,
        f"{remote_bundle}/{checkpoint}",
        local_bundle / checkpoint,
        transfer_log,
        directory=True,
    )
    return artifact


def stage_inputs(
    args: argparse.Namespace,
    output: Path,
    candidate_status: dict[str, Any],
    release_status: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    staged = output / "staged"
    staged.mkdir(exist_ok=True)
    transfer_log = output / "transfer.log"
    remote_candidate = ensure_remote_descendant(
        args.remote_base, str(candidate_status["artifacts_dir"])
    )
    remote_text = ensure_remote_descendant(args.remote_base, str(release_status["bundle"]))
    candidate = stage_model_bundle(
        args,
        remote_candidate,
        staged / "model-small-native-ocr",
        "multimodal-candidate.json",
        transfer_log,
    )
    active = stage_model_bundle(
        args,
        remote_text,
        staged / "model-small-text",
        "active-model.json",
        transfer_log,
    )
    if (
        candidate.get("checkpoint_step") != candidate_status.get("checkpoint_step")
        or candidate.get("checkpoint_model_hash")
        != candidate_status.get("checkpoint_model_hash")
        or active.get("checkpoint_step") != release_status.get("checkpoint", {}).get("step")
        or active.get("checkpoint_model_hash")
        != release_status.get("checkpoint", {}).get("artifact_hash")
    ):
        raise RuntimeError("staged model manifests differ from terminal remote status")

    audit_remote = ensure_remote_descendant(args.remote_base, args.remote_audit_manifest)
    audit_parent = str(PurePosixPath(audit_remote).parent)
    rsync(args, audit_parent, staged / "vision-audit", transfer_log, directory=True)
    retention_dir = staged / "retention"
    seen: set[str] = set()
    retention_names: list[str] = []
    for remote_path in args.remote_retention_eval:
        remote_path = ensure_remote_descendant(args.remote_base, remote_path)
        name = PurePosixPath(remote_path).name
        if name in seen:
            raise RuntimeError(f"retention filenames collide while staging: {name}")
        seen.add(name)
        retention_names.append(name)
        rsync(args, remote_path, retention_dir / name, transfer_log)

    marker = {
        "schema_version": 1,
        "candidate_checkpoint_step": candidate["checkpoint_step"],
        "candidate_checkpoint_model_hash": candidate["checkpoint_model_hash"],
        "text_checkpoint_step": active["checkpoint_step"],
        "text_checkpoint_model_hash": active["checkpoint_model_hash"],
        "candidate_artifact_sha256": loop.sha256_file(
            staged / "model-small-native-ocr/multimodal-candidate.json"
        ),
        "text_artifact_sha256": loop.sha256_file(
            staged / "model-small-text/active-model.json"
        ),
        "audit_manifest_sha256": loop.sha256_file(staged / "vision-audit/manifest.jsonl"),
        "retention_files": [
            {
                "path": f"retention/{name}",
                "sha256": loop.sha256_file(retention_dir / name),
            }
            for name in retention_names
        ],
    }
    loop.atomic_json(staged / "stage-complete.json", marker)
    return staged, marker


def verify_staged(staged: Path) -> dict[str, Any]:
    marker = loop.load_json(staged / "stage-complete.json")
    identities = [
        (
            staged / "model-small-native-ocr/multimodal-candidate.json",
            marker["candidate_artifact_sha256"],
        ),
        (staged / "model-small-text/active-model.json", marker["text_artifact_sha256"]),
        (staged / "vision-audit/manifest.jsonl", marker["audit_manifest_sha256"]),
        *[
            (staged / identity["path"], identity["sha256"])
            for identity in marker["retention_files"]
        ],
    ]
    for path, expected in identities:
        if loop.sha256_file(path) != expected:
            raise RuntimeError(f"staged input changed after transfer: {path}")
    return marker


def next_log(output: Path) -> Path:
    index = 1
    while (output / f"offline-attempt-{index:02d}.log").exists():
        index += 1
    return output / f"offline-attempt-{index:02d}.log"


def docker_command(
    args: argparse.Namespace, staged: Path, marker: dict[str, Any], resume: bool
) -> list[str]:
    command = [
        str(args.docker_binary.resolve(strict=True)),
        "run",
        "--rm",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--pids-limit",
        "256",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{staged.resolve(strict=True)}:/work",
        args.offline_image,
        "--candidate-bundle",
        "/work/model-small-native-ocr",
        "--base-text-bundle",
        "/work/model-small-text",
        "--audit-manifest",
        "/work/vision-audit/manifest.jsonl",
    ]
    for identity in marker["retention_files"]:
        name = Path(identity["path"]).name
        command.extend(["--retention-eval-jsonl", f"/work/retention/{name}"])
    command.extend(["--out", "/work/offline-gate-v1"])
    if args.max_new_tokens is not None:
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    if resume:
        command.append("--resume")
    return command


def verify_active(staged: Path, marker: dict[str, Any]) -> Path:
    path = staged / "model-small-native-ocr/multimodal-active.json"
    active = loop.load_json(path)
    if (
        active.get("checkpoint_step") != marker["candidate_checkpoint_step"]
        or active.get("checkpoint_model_hash")
        != marker["candidate_checkpoint_model_hash"]
        or not isinstance(active.get("certification"), dict)
        or active.get("checkpoint_optimizer_hash") is not None
    ):
        raise RuntimeError("local offline gate did not publish the exact serving artifact")
    return path


def main() -> int:
    args = parse_args()
    output = args.out.resolve()
    config = launch_configuration(args)
    config_path = output / "launch-config.json"
    status_path = output / "status.json"
    if args.resume:
        if not output.is_dir() or loop.load_json(config_path) != config:
            raise RuntimeError("resume arguments or pinned host/image differ from launch config")
    else:
        if output.exists():
            raise RuntimeError("new cross-host gate requires an absent output directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        loop.atomic_json(config_path, config)

    with (output / ".cross-host.lock").open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another cross-host gate holds the local lock") from error
        verify_launch_pins(config)
        staged = output / "staged"
        if (staged / "stage-complete.json").exists():
            marker = verify_staged(staged)
        else:
            loop.atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "waiting_for_remote_candidate",
                    "active_model_written": False,
                },
            )
            candidate_status = wait_for_candidate(args, status_path)
            if candidate_status is None:
                return 0
            release_status = remote_json(args, args.remote_release_status)
            if release_status.get("stage") != "active_text_model_ready":
                raise RuntimeError("remote OCR candidate exists without its certified text base")
            loop.atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "staging_remote_candidate",
                    "remote_candidate": candidate_status,
                    "active_model_written": False,
                },
            )
            staged, marker = stage_inputs(
                args, output, candidate_status, release_status
            )

        if (staged / "model-small-native-ocr/multimodal-active.json").exists():
            active_path = verify_active(staged, marker)
        else:
            verify_launch_pins(config)
            offline = staged / "offline-gate-v1"
            resume_offline = offline.exists()
            log_path = next_log(output)
            loop.atomic_json(
                status_path,
                {
                    "schema_version": 1,
                    "stage": "running_local_network_denied_gate",
                    "offline_image_id": config["offline_image_id"],
                    "log": str(log_path),
                    "resume": resume_offline,
                    "active_model_written": False,
                },
            )
            with log_path.open("x", encoding="utf-8") as log:
                completed = subprocess.run(
                    docker_command(args, staged, marker, resume_offline),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    pass_fds=(lock_handle.fileno(),),
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"offline container exited {completed.returncode}; inspect {log_path}"
                )
            active_path = verify_active(staged, marker)

        loop.atomic_json(
            status_path,
            {
                "schema_version": 1,
                "stage": "active_multimodal_model_ready",
                "active_model": str(active_path),
                "checkpoint_step": marker["candidate_checkpoint_step"],
                "checkpoint_model_hash": marker["candidate_checkpoint_model_hash"],
                "network_denied_gate": str(staged / "offline-gate-v1"),
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
                f"native OCR cross-host gate: could not persist failed status: {status_error}",
                file=sys.stderr,
            )
        print(f"native OCR cross-host gate: {error}", file=sys.stderr)
        raise SystemExit(1)
