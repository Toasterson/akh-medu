#!/usr/bin/env python3
"""Commit one reviewed BabyLLM publication plan to a public Hugging Face model.

This helper runs only inside a short-lived ``uv`` environment after 1Password
has resolved ``HF_TOKEN``.  It validates every local and remote path, pins the
parent commit, uploads exactly the declared operations, and downloads every
resulting file anonymously at the returned commit before reporting success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")
FORBIDDEN_NAMES = {
    "optim.mpk",
    "multimodal-candidate.json",
    "audit-predictions.jsonl",
    "retention-before-outputs.jsonl",
    "retention-after-outputs.jsonl",
    "network-replay-first.jsonl",
    "network-replay-second.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--repo-id", required=True)
    upload = subparsers.add_parser("upload")
    upload.add_argument("--packet-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def safe_relative(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or not relative.parts or any(
        part in ("", ".", "..") or not SAFE_COMPONENT.fullmatch(part)
        for part in relative.parts
    ):
        raise RuntimeError(f"unsafe relative path: {path!r}")
    return relative


def packet_file(packet: Path, value: str) -> Path:
    relative = safe_relative(value)
    current = packet
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise RuntimeError(f"publication input contains a symlink: {current}")
    resolved = current.resolve(strict=True)
    root = packet.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise RuntimeError(f"publication input escapes packet: {value}")
    return resolved


def validate_plan(packet: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    packet = packet.resolve(strict=True)
    plan = load_json(packet / "publication-plan.json")
    if (
        plan.get("schema_version") != 1
        or plan.get("kind") != "certified_babyllm_hf_publication"
        or plan.get("repo_type") != "model"
        or plan.get("revision") != "main"
        or not REPO_ID.fullmatch(str(plan.get("repo_id", "")))
        or not GIT_SHA.fullmatch(str(plan.get("parent_commit", "")))
    ):
        raise RuntimeError("publication plan header is invalid")
    prefix = safe_relative(str(plan.get("release_prefix", "")))
    if prefix.parts[:2] != ("releases", "tool-native-small"):
        raise RuntimeError("release prefix is outside releases/tool-native-small")
    operations = plan.get("operations")
    if not isinstance(operations, list) or not operations:
        raise RuntimeError("publication plan has no operations")
    for field in ("commit_message", "commit_description"):
        value = plan.get(field)
        if (
            not isinstance(value, str)
            or not value.strip()
            or len(value) > 500
            or any(character in value for character in ("\x00", "\r"))
        ):
            raise RuntimeError(f"publication {field} is invalid")

    validated: list[dict[str, Any]] = []
    local_paths: set[str] = set()
    remote_paths: set[str] = set()
    for operation in operations:
        if not isinstance(operation, dict):
            raise RuntimeError("publication operation is not an object")
        local_value = str(operation.get("local_path", ""))
        remote_value = str(operation.get("path_in_repo", ""))
        local = packet_file(packet, local_value)
        remote = safe_relative(remote_value)
        if remote_value != "README.md":
            try:
                remote.relative_to(prefix)
            except ValueError as error:
                raise RuntimeError(
                    f"remote path is outside the reviewed release prefix: {remote}"
                ) from error
        if remote.name.lower() in FORBIDDEN_NAMES:
            raise RuntimeError(f"forbidden training/evidence artifact: {remote}")
        if local_value in local_paths or remote_value in remote_paths:
            raise RuntimeError("publication plan contains duplicate paths")
        local_paths.add(local_value)
        remote_paths.add(remote_value)
        if (
            operation.get("sha256") != sha256_file(local)
            or operation.get("bytes") != local.stat().st_size
        ):
            raise RuntimeError(f"publication operation identity differs: {local_value}")
        validated.append({**operation, "path": local})

    payload_files: set[str] = set()
    for path in (packet / "payload").rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"publication payload contains a symlink: {path}")
        if path.is_file():
            payload_files.add(str(path.relative_to(packet)))
    if payload_files != local_paths:
        missing = sorted(payload_files - local_paths)
        absent = sorted(local_paths - payload_files)
        raise RuntimeError(
            f"publication plan/payload mismatch: unplanned={missing} absent={absent}"
        )
    required_names = {
        "README.md",
        "multimodal-active.json",
        "deployment-provenance.json",
        "model.mpk",
        "meta.json",
        "release-metadata.json",
        "SHA256SUMS",
    }
    names = {PurePosixPath(path).name for path in remote_paths}
    missing_names = sorted(required_names - names)
    if missing_names:
        raise RuntimeError(f"publication plan lacks serving files: {missing_names}")
    if sum(path == "README.md" for path in remote_paths) != 1:
        raise RuntimeError("publication plan must update exactly one root README")
    return plan, validated


def import_hub() -> tuple[Any, Any, Any]:
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    except ImportError as error:  # pragma: no cover - uv controls this boundary
        raise RuntimeError("huggingface_hub is unavailable") from error
    return HfApi, CommitOperationAdd, hf_hub_download


def public_head(repo_id: str) -> dict[str, Any]:
    if not REPO_ID.fullmatch(repo_id):
        raise RuntimeError("invalid Hugging Face repository id")
    HfApi, _CommitOperationAdd, _download = import_hub()
    info = HfApi(token=False).model_info(repo_id, token=False)
    if bool(info.private) or not GIT_SHA.fullmatch(str(info.sha)):
        raise RuntimeError("target model repository is not public or has no Git head")
    return {"repo_id": repo_id, "private": False, "sha": str(info.sha)}


def verify_remote(
    plan: dict[str, Any], operations: list[dict[str, Any]], revision: str
) -> None:
    HfApi, _CommitOperationAdd, hf_hub_download = import_hub()
    info = HfApi(token=False).model_info(
        plan["repo_id"], revision=revision, token=False
    )
    if bool(info.private):
        raise RuntimeError("published repository is not anonymously readable")
    with tempfile.TemporaryDirectory(prefix="babyllm-hf-verify-") as directory:
        for operation in operations:
            downloaded = hf_hub_download(
                repo_id=plan["repo_id"],
                filename=operation["path_in_repo"],
                repo_type="model",
                revision=revision,
                token=False,
                local_dir=directory,
                force_download=True,
            )
            path = Path(downloaded)
            if (
                sha256_file(path) != operation["sha256"]
                or path.stat().st_size != operation["bytes"]
            ):
                raise RuntimeError(
                    f"anonymous Hugging Face bytes differ: {operation['path_in_repo']}"
                )


def upload(packet: Path) -> dict[str, Any]:
    plan, operations = validate_plan(packet)
    token = os.environ.get("HF_TOKEN", "")
    if not token or token.startswith("op://"):
        raise RuntimeError("HF_TOKEN was not resolved by the credential boundary")
    HfApi, CommitOperationAdd, _download = import_hub()
    api = HfApi(token=token)
    current = api.model_info(plan["repo_id"], token=token)
    if bool(current.private):
        raise RuntimeError("publication target unexpectedly became private")
    current_sha = str(current.sha)
    if current_sha != plan["parent_commit"]:
        # Reconcile a lost client response only when the complete reviewed byte
        # set is already present at the current public head. Any other drift
        # requires a newly prepared and newly reviewed packet.
        verify_remote(plan, operations, current_sha)
        return {
            "schema_version": 1,
            "stage": "already_published",
            "repo_id": plan["repo_id"],
            "commit": current_sha,
            "anonymous_verification": True,
            "operation_count": len(operations),
        }

    commit = api.create_commit(
        repo_id=plan["repo_id"],
        repo_type="model",
        revision="main",
        parent_commit=plan["parent_commit"],
        commit_message=plan["commit_message"],
        commit_description=plan["commit_description"],
        operations=[
            CommitOperationAdd(operation["path_in_repo"], operation["path"])
            for operation in operations
        ],
        token=token,
    )
    oid = str(commit.oid)
    if not GIT_SHA.fullmatch(oid):
        raise RuntimeError("Hugging Face returned an invalid commit identity")
    verify_remote(plan, operations, oid)
    return {
        "schema_version": 1,
        "stage": "published",
        "repo_id": plan["repo_id"],
        "commit": oid,
        "anonymous_verification": True,
        "operation_count": len(operations),
    }


def main() -> int:
    args = parse_args()
    if args.command == "inspect":
        result = public_head(args.repo_id)
    else:
        result = upload(args.packet_dir)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"certified BabyLLM Hugging Face helper: {error}", file=sys.stderr)
        raise SystemExit(1)
