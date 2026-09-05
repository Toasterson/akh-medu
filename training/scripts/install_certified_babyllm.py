#!/usr/bin/env python3
"""Atomically install an exam-certified BabyLLM for Akh-Medu serving.

The cross-host OCR gate is the only accepted source.  This watcher waits for
its terminal active status, verifies the source and installed copies through
the same ``GrugChatBackend`` loader used by ``akhomed``, copies only serving
artifacts (never Adam state), and atomically advances a stable model symlink.
It does not edit any workspace's ``akh.toml``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePath
from typing import Any

import run_tool_native_adaptive_loop as loop


SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cross-host-run", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="knowledge/model-small")
    parser.add_argument("--probe-binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-wait-hours", type=float, default=48.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 5 or args.maximum_wait_hours <= 0:
        parser.error("poll must be >=5 seconds and wait budget positive")
    safe_relative(args.model_name)
    return args


def safe_relative(value: str) -> Path:
    path = PurePath(value)
    if path.is_absolute() or not path.parts:
        raise ValueError(f"path must be a non-empty relative path: {value!r}")
    if any(part in ("", ".", "..") or not SAFE_COMPONENT.fullmatch(part) for part in path.parts):
        raise ValueError(f"path contains an unsafe component: {value!r}")
    return Path(*path.parts)


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or path.is_symlink():
        raise RuntimeError(f"input is not a regular non-symlink file: {path}")
    return {
        "path": str(resolved),
        "sha256": loop.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def launch_configuration(args: argparse.Namespace) -> dict[str, Any]:
    cross_host = args.cross_host_run.resolve(strict=True)
    cross_host_config = cross_host / "launch-config.json"
    return {
        "schema_version": 1,
        "cross_host_run": str(cross_host),
        "cross_host_launch_config": file_identity(cross_host_config),
        "models_dir": str(args.models_dir.expanduser().resolve()),
        "model_name": str(safe_relative(args.model_name)),
        "probe_binary": file_identity(args.probe_binary),
        "poll_seconds": args.poll_seconds,
        "maximum_wait_hours": args.maximum_wait_hours,
        "source_policy": "exact_cross_host_active_only",
        "install_policy": "immutable_release_plus_atomic_symlink",
    }


def verify_launch_pins(config: dict[str, Any]) -> None:
    for key in ("cross_host_launch_config", "probe_binary"):
        identity = config[key]
        path = Path(identity["path"])
        if loop.sha256_file(path) != identity["sha256"]:
            raise RuntimeError(f"pinned install input changed: {path}")


def write_status(path: Path, stage: str, **values: Any) -> None:
    loop.atomic_json(
        path,
        {
            "schema_version": 1,
            "stage": stage,
            "active_model_installed": stage == "installed",
            **values,
        },
    )


def wait_for_active(
    config: dict[str, Any], status_path: Path
) -> tuple[Path, dict[str, Any]] | None:
    cross_host = Path(config["cross_host_run"])
    expected_bundle = (cross_host / "staged/model-small-native-ocr").resolve()
    expected_active = expected_bundle / "multimodal-active.json"
    deadline = time.monotonic() + float(config["maximum_wait_hours"]) * 3600.0
    while time.monotonic() < deadline:
        remote_status_path = cross_host / "status.json"
        if remote_status_path.is_file():
            remote = loop.load_json(remote_status_path)
            stage = str(remote.get("stage", ""))
            if stage == "active_multimodal_model_ready":
                declared = Path(str(remote.get("active_model", ""))).resolve(strict=True)
                if (
                    remote.get("active_model_written") is not True
                    or declared != expected_active.resolve(strict=True)
                ):
                    raise RuntimeError(
                        "cross-host terminal status does not name the exact staged active model"
                    )
                return expected_bundle, remote
            if stage in (
                "blocked_by_remote_training",
                "blocked_by_text_release",
                "failed",
            ):
                write_status(
                    status_path,
                    "blocked_by_cross_host_gate",
                    cross_host_status=remote,
                )
                return None
        write_status(
            status_path,
            "waiting_for_certified_multimodal_model",
            cross_host_status=str(remote_status_path),
        )
        time.sleep(float(config["poll_seconds"]))
    raise RuntimeError("certified-model install wait exceeded the configured deadline")


def bundle_child(bundle: Path, value: Any, *, directory: bool = False) -> Path:
    relative = safe_relative(str(value))
    unresolved = bundle / relative
    current = bundle
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"serving artifact path contains a symlink: {current}")
    resolved = unresolved.resolve(strict=True)
    base = bundle.resolve(strict=True)
    if resolved == base or base not in resolved.parents:
        raise RuntimeError(f"serving artifact escapes its bundle: {value}")
    if directory and not resolved.is_dir():
        raise RuntimeError(f"serving artifact is not a directory: {resolved}")
    if not directory and not resolved.is_file():
        raise RuntimeError(f"serving artifact is not a file: {resolved}")
    return resolved


def verify_active_manifest(bundle: Path) -> dict[str, Any]:
    active_path = bundle_child(bundle, "multimodal-active.json")
    active = loop.load_json(active_path)
    if (
        not isinstance(active.get("checkpoint_step"), int)
        or active["checkpoint_step"] < 0
        or not str(active.get("checkpoint_model_hash", "")).startswith("blake3:")
        or active.get("checkpoint_optimizer_hash") is not None
        or not isinstance(active.get("certification"), dict)
    ):
        raise RuntimeError("multimodal active manifest is not serving-safe and certified")
    bundle_child(bundle, active["tokenizer_file"])
    bundle_child(bundle, active["contracts_file"])
    checkpoint = bundle_child(bundle, active["checkpoint_dir"], directory=True)
    bundle_child(checkpoint, "model.mpk")
    bundle_child(checkpoint, "meta.json")
    return active


def next_log(output: Path, stem: str) -> Path:
    index = 1
    while (output / f"{stem}-{index:02d}.log").exists():
        index += 1
    return output / f"{stem}-{index:02d}.log"


def run_probe(config: dict[str, Any], bundle: Path, log_path: Path) -> None:
    verify_launch_pins(config)
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            [
                config["probe_binary"]["path"],
                "--bundle",
                str(bundle),
                "--max-new-tokens",
                "1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Akh-Medu model probe failed; inspect {log_path}")


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"refusing non-regular serving artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def serving_sources(bundle: Path, active: dict[str, Any]) -> list[tuple[Path, Path]]:
    checkpoint_rel = safe_relative(str(active["checkpoint_dir"]))
    values = [
        (bundle_child(bundle, "multimodal-active.json"), Path("multimodal-active.json")),
        (
            bundle_child(bundle, active["tokenizer_file"]),
            safe_relative(str(active["tokenizer_file"])),
        ),
        (
            bundle_child(bundle, active["contracts_file"]),
            safe_relative(str(active["contracts_file"])),
        ),
        (bundle_child(bundle, active["checkpoint_dir"], directory=True) / "model.mpk", checkpoint_rel / "model.mpk"),
        (bundle_child(bundle, active["checkpoint_dir"], directory=True) / "meta.json", checkpoint_rel / "meta.json"),
    ]
    promotion = bundle / "multimodal-promotion.json"
    if promotion.is_file() and not promotion.is_symlink():
        values.append((promotion.resolve(strict=True), Path("multimodal-promotion.json")))
    return values


def release_id(active: dict[str, Any]) -> str:
    digest = str(active["checkpoint_model_hash"]).split(":", 1)[1]
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("checkpoint hash is not a canonical blake3 digest")
    return f"model-small-step-{active['checkpoint_step']:08d}-{digest[:16]}"


def chmod_immutable_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def install_release(
    config: dict[str, Any], source: Path, output: Path, active: dict[str, Any]
) -> Path:
    models_dir = Path(config["models_dir"])
    alias = models_dir / safe_relative(config["model_name"])
    releases = alias.parent / "releases"
    releases.mkdir(parents=True, exist_ok=True)
    destination = releases / release_id(active)
    source_active_hash = loop.sha256_file(source / "multimodal-active.json")

    if not destination.exists():
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=releases)
        )
        try:
            payload: list[dict[str, Any]] = []
            for source_path, relative in serving_sources(source, active):
                target = temporary / relative
                copy_file(source_path, target)
                payload.append(
                    {
                        "path": str(relative),
                        "sha256": loop.sha256_file(target),
                        "bytes": target.stat().st_size,
                    }
                )
            loop.atomic_json(
                temporary / "deployment-provenance.json",
                {
                    "schema_version": 1,
                    "kind": "certified_babyllm_serving_bundle",
                    "checkpoint_step": active["checkpoint_step"],
                    "checkpoint_model_hash": active["checkpoint_model_hash"],
                    "source_active_sha256": source_active_hash,
                    "cross_host_launch_config_sha256": config[
                        "cross_host_launch_config"
                    ]["sha256"],
                    "optimizer_included": False,
                    "payload": sorted(payload, key=lambda item: item["path"]),
                },
            )
            run_probe(config, temporary, next_log(output, "source-copy-probe"))
            chmod_immutable_tree(temporary)
            os.rename(temporary, destination)
        except Exception:
            if temporary.exists():
                for path in temporary.rglob("*"):
                    try:
                        path.chmod(0o700 if path.is_dir() else 0o600)
                    except OSError:
                        pass
                shutil.rmtree(temporary, ignore_errors=True)
            raise
    else:
        provenance = loop.load_json(destination / "deployment-provenance.json")
        if (
            provenance.get("source_active_sha256") != source_active_hash
            or provenance.get("checkpoint_model_hash")
            != active["checkpoint_model_hash"]
        ):
            raise RuntimeError("immutable release path already contains different bytes")

    run_probe(config, destination, next_log(output, "installed-release-probe"))
    alias.parent.mkdir(parents=True, exist_ok=True)
    if alias.exists() and not alias.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink model path: {alias}")
    relative_target = os.path.relpath(destination, alias.parent)
    temporary_link = alias.parent / f".{alias.name}.tmp-{os.getpid()}"
    if os.path.lexists(temporary_link):
        raise RuntimeError(f"temporary model link already exists: {temporary_link}")
    os.symlink(relative_target, temporary_link)
    os.replace(temporary_link, alias)
    run_probe(config, alias, next_log(output, "active-alias-probe"))
    return alias


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
            raise RuntimeError("new install supervisor requires an absent output directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        loop.atomic_json(config_path, config)

    with (output / ".install.lock").open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another certified-model installer holds the lock") from error
        verify_launch_pins(config)
        ready = wait_for_active(config, status_path)
        if ready is None:
            return 0
        source, cross_host_status = ready
        active = verify_active_manifest(source)
        if (
            active["checkpoint_step"] != cross_host_status.get("checkpoint_step")
            or active["checkpoint_model_hash"]
            != cross_host_status.get("checkpoint_model_hash")
        ):
            raise RuntimeError("active manifest differs from cross-host terminal status")
        run_probe(config, source, next_log(output, "source-active-probe"))
        write_status(
            status_path,
            "installing_verified_release",
            checkpoint_step=active["checkpoint_step"],
            checkpoint_model_hash=active["checkpoint_model_hash"],
        )
        alias = install_release(config, source, output, active)
        write_status(
            status_path,
            "installed",
            model_path=str(alias),
            model_config_value=config["model_name"],
            checkpoint_step=active["checkpoint_step"],
            checkpoint_model_hash=active["checkpoint_model_hash"],
            multimodal=True,
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
                write_status(failure_out / "status.json", "failed", error=str(error))
        except Exception as status_error:
            print(
                f"certified BabyLLM installer: could not persist failure: {status_error}",
                file=sys.stderr,
            )
        print(f"certified BabyLLM installer: {error}", file=sys.stderr)
        raise SystemExit(1)
