#!/usr/bin/env python3
"""Prepare, review-gate, and publish a certified BabyLLM serving bundle.

The watcher accepts only the terminal output of ``install_certified_babyllm``.
It rebuilds the serving whitelist into an immutable review packet, pins the
public Hugging Face parent commit, and waits for byte-bound PASS responses from
the two exact requested reviewer model IDs.  Only then does it expose the
1Password reference to a short-lived upload helper; no credential is held
while training, packaging, or reviewing.
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

import install_certified_babyllm as installer
import hf_commit_certified_babyllm as hf_helper
import run_tool_native_adaptive_loop as loop


HF_PACKAGE = "huggingface_hub==1.30.0"
EXPECTED_REVIEWERS = ("z-ai/glm-5.3", "moonshotai/kimi-k3")
REVIEW_FILENAMES = {
    "z-ai/glm-5.3": "glm-5.3.json",
    "moonshotai/kimi-k3": "kimi-k3.json",
}
REPO_ID = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-run", type=Path, required=True)
    parser.add_argument("--repo-id", default="OpenFlowLabs/grug-rule-350m")
    parser.add_argument(
        "--release-root", default="releases/tool-native-small"
    )
    parser.add_argument("--credential-ref", required=True)
    parser.add_argument("--op-wrapper", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--upload-helper", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--maximum-wait-hours", type=float, default=168.0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not REPO_ID.fullmatch(args.repo_id):
        parser.error("invalid Hugging Face repo id")
    relative = installer.safe_relative(args.release_root)
    if relative.parts != ("releases", "tool-native-small"):
        parser.error("release-root must be exactly releases/tool-native-small")
    if not args.credential_ref.startswith("op://"):
        parser.error("credential-ref must be a 1Password secret reference")
    if args.poll_seconds < 5 or args.maximum_wait_hours <= 0:
        parser.error("poll must be >=5 seconds and wait budget positive")
    return args


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or path.is_symlink():
        raise RuntimeError(f"pinned input is not a regular file: {path}")
    return {
        "path": str(resolved),
        "sha256": loop.sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def launch_configuration(args: argparse.Namespace) -> dict[str, Any]:
    install_run = args.install_run.resolve(strict=True)
    install_config = loop.load_json(install_run / "launch-config.json")
    probe = Path(install_config["probe_binary"]["path"])
    probe_identity = file_identity(probe)
    if probe_identity != install_config["probe_binary"]:
        raise RuntimeError("installer probe identity changed before publication launch")
    return {
        "schema_version": 1,
        "install_run": str(install_run),
        "install_launch_config": file_identity(install_run / "launch-config.json"),
        "repo_id": args.repo_id,
        "repo_type": "model",
        "release_root": str(installer.safe_relative(args.release_root)),
        "credential_ref": args.credential_ref,
        "op_wrapper": file_identity(args.op_wrapper),
        "uv": file_identity(args.uv),
        "upload_helper": file_identity(args.upload_helper),
        "publication_supervisor": file_identity(Path(__file__)),
        "probe_binary": probe_identity,
        "huggingface_package": HF_PACKAGE,
        "poll_seconds": args.poll_seconds,
        "maximum_wait_hours": args.maximum_wait_hours,
        "expected_reviewers": list(EXPECTED_REVIEWERS),
        "publication_policy": "exact_reviewed_bytes_parent_pinned_public_repo",
    }


def verify_pins(config: dict[str, Any]) -> None:
    for key in (
        "install_launch_config",
        "op_wrapper",
        "uv",
        "upload_helper",
        "publication_supervisor",
        "probe_binary",
    ):
        identity = config[key]
        if loop.sha256_file(Path(identity["path"])) != identity["sha256"]:
            raise RuntimeError(f"pinned publication input changed: {identity['path']}")


def write_status(path: Path, stage: str, **values: Any) -> None:
    loop.atomic_json(
        path,
        {
            "schema_version": 1,
            "stage": stage,
            "published": stage == "published",
            **values,
        },
    )


def wait_for_install(
    config: dict[str, Any], status_path: Path, deadline: float
) -> dict[str, Any] | None:
    install_status_path = Path(config["install_run"]) / "status.json"
    while time.monotonic() < deadline:
        if install_status_path.is_file():
            status = loop.load_json(install_status_path)
            stage = str(status.get("stage", ""))
            if stage == "installed":
                if status.get("active_model_installed") is not True:
                    raise RuntimeError("installer terminal state lacks active-model proof")
                return status
            if stage in ("failed", "blocked_by_cross_host_gate"):
                write_status(
                    status_path,
                    "blocked_by_installer",
                    install_status=status,
                )
                return None
        write_status(
            status_path,
            "waiting_for_certified_install",
            install_status=str(install_status_path),
        )
        time.sleep(float(config["poll_seconds"]))
    raise RuntimeError("certified publication wait exceeded before installation")


def relative_files(root: Path) -> set[Path]:
    values: set[Path] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"serving release contains a symlink: {path}")
        if path.is_file():
            values.add(path.relative_to(root))
    return values


def verify_installed_release(
    config: dict[str, Any], install_status: dict[str, Any], output: Path
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    install_config = loop.load_json(Path(config["install_launch_config"]["path"]))
    alias = Path(install_config["models_dir"]) / installer.safe_relative(
        install_config["model_name"]
    )
    if (
        Path(str(install_status.get("model_path", ""))) != alias
        or install_status.get("model_config_value") != install_config["model_name"]
        or not alias.is_symlink()
    ):
        raise RuntimeError("installer status does not name its exact configured alias")
    release = alias.resolve(strict=True)
    release_parent = (alias.parent / "releases").resolve(strict=True)
    if release.parent != release_parent or not release.is_dir():
        raise RuntimeError("installed model alias escapes the immutable release directory")

    active = installer.verify_active_manifest(release)
    if (
        active["checkpoint_step"] != install_status.get("checkpoint_step")
        or active["checkpoint_model_hash"]
        != install_status.get("checkpoint_model_hash")
        or release.name != installer.release_id(active)
    ):
        raise RuntimeError("installed release identity differs from installer status")
    provenance = loop.load_json(release / "deployment-provenance.json")
    if (
        provenance.get("kind") != "certified_babyllm_serving_bundle"
        or provenance.get("optimizer_included") is not False
        or provenance.get("checkpoint_step") != active["checkpoint_step"]
        or provenance.get("checkpoint_model_hash")
        != active["checkpoint_model_hash"]
    ):
        raise RuntimeError("installed deployment provenance is invalid")

    expected = {
        destination
        for _source, destination in installer.serving_sources(release, active)
    }
    expected.add(Path("deployment-provenance.json"))
    observed = relative_files(release)
    if observed != expected:
        raise RuntimeError(
            f"installed release is not the serving whitelist: extra={sorted(map(str, observed - expected))} missing={sorted(map(str, expected - observed))}"
        )
    payload = provenance.get("payload")
    if not isinstance(payload, list):
        raise RuntimeError("deployment provenance has no payload identities")
    declared = {
        Path(str(item.get("path", ""))): (item.get("sha256"), item.get("bytes"))
        for item in payload
        if isinstance(item, dict)
    }
    if set(declared) != expected - {Path("deployment-provenance.json")}:
        raise RuntimeError("deployment provenance does not cover the serving whitelist")
    for relative, (digest, size) in declared.items():
        path = release / relative
        if digest != loop.sha256_file(path) or size != path.stat().st_size:
            raise RuntimeError(f"installed serving file differs from provenance: {relative}")

    log_path = output / "publication-source-probe.log"
    if not log_path.exists():
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                [
                    config["probe_binary"]["path"],
                    "--bundle",
                    str(release),
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
            raise RuntimeError(f"publication source probe failed; inspect {log_path}")
    return release, active, provenance


def helper_command(config: dict[str, Any], *arguments: str) -> list[str]:
    return [
        config["uv"]["path"],
        "run",
        "--offline",
        "--with",
        config["huggingface_package"],
        "python",
        config["upload_helper"]["path"],
        *arguments,
    ]


def inspect_public_head(config: dict[str, Any]) -> str:
    verify_pins(config)
    completed = subprocess.run(
        helper_command(config, "inspect", "--repo-id", config["repo_id"]),
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"could not inspect public Hugging Face parent: {completed.stderr.strip()}"
        )
    value = json.loads(completed.stdout)
    if (
        not isinstance(value, dict)
        or value.get("repo_id") != config["repo_id"]
        or value.get("private") is not False
        or not GIT_SHA.fullmatch(str(value.get("sha", "")))
    ):
        raise RuntimeError("Hugging Face inspection returned an invalid public head")
    return str(value["sha"])


def copy_regular(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"refusing non-regular publication input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def model_card(active: dict[str, Any], release_prefix: str) -> str:
    certification = active["certification"]
    shape = active["text_shape"]
    return f"""---
license: mit
language:
  - en
  - de
  - fr
library_name: burn
pipeline_tag: image-text-to-text
tags:
  - rust
  - tool-use
  - optical-character-recognition
  - local-first
---

# Grug Rule: certified tool-native BabyLLM

The current certified Akh-Medu serving release is
[`{release_prefix}/`]({release_prefix}/). It is a compact
{active['parameter_count']:,}-parameter shared text-and-vision decoder, not a
350M chat model. The repository name is retained for continuity with the older
350M deduction research artifact, which remains under its versioned paths.

## Certified boundary

- checkpoint step: `{active['checkpoint_step']}`
- checkpoint model hash: `{active['checkpoint_model_hash']}`
- runtime tool-contract hash: `{active['contract_hash']}`
- native OCR audit: {certification['audit_cases']} cases, exact
  {certification['exact_rate']:.4f}, CER
  {certification['character_error_rate']:.4f}
- weakest required language/layout/corruption slice exact:
  {certification['minimum_required_slice_exact_rate']:.4f}
- metamorphic consistency:
  {certification['metamorphic_consistency']:.4f}
- cross-semantic collapse:
  {certification['cross_semantic_collapse_rate']:.4f}
- text/tool action exact before/after vision training:
  {certification['text_action_exact_before']:.4f} /
  {certification['text_action_exact_after']:.4f}
- deterministic inference was replayed with Linux networking denied.

The audit transcripts, prediction streams, retention prompts/outputs and Adam
state are deliberately not published. Their content hashes remain bound into
the active certification.

## Akh-Medu use

Download the complete versioned directory without flattening it, install it as
an Akh-Medu model bundle, build the daemon with `server,grug-backend`, then set:

```toml
backend = "grug"
model = "knowledge/model-small"
```

The verified runtime supports a {shape['max_seq_len']}-token context and the
published tool contract. Tool additions or schema changes require a new
contract-delta adaptation and fresh sentinels; the engine remains authoritative
for permission checks and actual tool execution.

## Limits

Certification covers Latin-script English, German and French across line,
paragraph, receipt, terminal and broken-API layouts and the declared synthetic
corruptions. It does not claim Arabic/CJK, unrestricted handwriting, arbitrary
photography, or knowledge absent from Akh-Medu's managed sources. OCR text is
treated as untrusted observed data, never as system instructions.

## Licence

Published weights and serving metadata are MIT licensed. Akh-Medu training and
runtime source is GPL-3.0-only. Dataset-specific provenance and licences remain
listed in the model artifact.
"""


def write_text_new(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def prepare_packet(
    config: dict[str, Any],
    output: Path,
    release: Path,
    active: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    packet = output / "review-packet"
    if packet.exists():
        digest, plan = verify_packet(packet)
        if plan["repo_id"] != config["repo_id"]:
            raise RuntimeError("existing review packet targets another repository")
        write_review_request(output, packet, digest)
        return packet, digest, plan

    parent = inspect_public_head(config)
    release_prefix = f"{config['release_root']}/{release.name}"
    temporary = Path(tempfile.mkdtemp(prefix=".review-packet-", dir=output))
    try:
        payload = temporary / "payload"
        local_release = payload / "release"
        for relative in sorted(relative_files(release), key=str):
            copy_regular(release / relative, local_release / relative)

        metadata = {
            "schema_version": 1,
            "kind": "certified_tool_native_babyllm_release",
            "model_config_value": "knowledge/model-small",
            "parameter_count": active["parameter_count"],
            "text_shape": active["text_shape"],
            "vision_shape": active["vision_shape"],
            "checkpoint_step": active["checkpoint_step"],
            "checkpoint_model_hash": active["checkpoint_model_hash"],
            "tokenizer_hash": active["tokenizer_hash"],
            "contract_hash": active["contract_hash"],
            "base_text_checkpoint_hash": active["base_text_checkpoint_hash"],
            "certification": active["certification"],
            "optimizer_included": False,
            "serving_payload": provenance["payload"],
        }
        loop.atomic_json(local_release / "release-metadata.json", metadata)

        checksummed = sorted(relative_files(local_release), key=str)
        checksum_lines = [
            f"{loop.sha256_file(local_release / relative).split(':', 1)[1]}  {relative.as_posix()}"
            for relative in checksummed
        ]
        write_text_new(local_release / "SHA256SUMS", "\n".join(checksum_lines) + "\n")
        write_text_new(
            payload / "root-README.md", model_card(active, release_prefix)
        )

        operations: list[dict[str, Any]] = []
        for relative in sorted(relative_files(payload), key=str):
            path = payload / relative
            local_path = (Path("payload") / relative).as_posix()
            if relative == Path("root-README.md"):
                remote = "README.md"
            else:
                remote = f"{release_prefix}/{relative.relative_to('release').as_posix()}"
            operations.append(
                {
                    "local_path": local_path,
                    "path_in_repo": remote,
                    "sha256": loop.sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        plan = {
            "schema_version": 1,
            "kind": "certified_babyllm_hf_publication",
            "repo_id": config["repo_id"],
            "repo_type": "model",
            "revision": "main",
            "parent_commit": parent,
            "release_prefix": release_prefix,
            "checkpoint_step": active["checkpoint_step"],
            "checkpoint_model_hash": active["checkpoint_model_hash"],
            "commit_message": f"Publish certified tool-native BabyLLM step {active['checkpoint_step']}",
            "commit_description": (
                "Exact GLM 5.3 and Kimi K3 reviewed serving-only release; "
                "optimizer, audit answers, predictions and retention streams omitted."
            ),
            "operations": operations,
        }
        loop.atomic_json(temporary / "publication-plan.json", plan)
        loop.atomic_json(
            temporary / "review-context.json",
            {
                "schema_version": 1,
                "required_reviewers": list(EXPECTED_REVIEWERS),
                "review_scope": "publication plan, model card, serving bundle, certification and orchestration identities",
                "install_launch_config_sha256": config["install_launch_config"]["sha256"],
                "installer_probe_sha256": config["probe_binary"]["sha256"],
                "publication_supervisor_sha256": config["publication_supervisor"]["sha256"],
                "upload_helper_sha256": config["upload_helper"]["sha256"],
                "forbidden_uploads": [
                    "optimizer state",
                    "candidate pointer",
                    "audit transcripts or predictions",
                    "retention prompts or outputs",
                    "credentials",
                ],
            },
        )
        manifest_files = sorted(relative_files(temporary), key=str)
        loop.atomic_json(
            temporary / "packet-manifest.json",
            {
                "schema_version": 1,
                "kind": "exact_reviewer_packet",
                "files": [
                    {
                        "path": relative.as_posix(),
                        "sha256": loop.sha256_file(temporary / relative),
                        "bytes": (temporary / relative).stat().st_size,
                    }
                    for relative in manifest_files
                ],
            },
        )
        digest = loop.sha256_file(temporary / "packet-manifest.json")
        os.rename(temporary, packet)
        installer.chmod_immutable_tree(packet)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    write_review_request(output, packet, digest)
    return packet, digest, plan


def write_review_request(output: Path, packet: Path, digest: str) -> None:
    loop.atomic_json(
        output / "review-request.json",
        {
            "schema_version": 1,
            "packet": str(packet),
            "packet_sha256": digest,
            "required_reviewers": list(EXPECTED_REVIEWERS),
            "required_response_lines": [
                f"PACKET_SHA256: {digest}",
                "VERDICT: PASS or VERDICT: BLOCK",
            ],
            "review_files": {
                reviewer: f"reviews/{REVIEW_FILENAMES[reviewer]}"
                for reviewer in EXPECTED_REVIEWERS
            },
            "review_evidence_contract": {
                "schema_version": 1,
                "required_fields": [
                    "reviewer_model",
                    "provider",
                    "provider_response_id",
                    "packet_sha256",
                    "reviewed_parent_commit",
                    "reviewed_operation_count",
                    "prompt_file",
                    "prompt_sha256",
                    "response_file",
                    "response_sha256",
                    "verdict",
                ],
            },
        },
    )


def verify_packet(packet: Path) -> tuple[str, dict[str, Any]]:
    if packet.is_symlink() or not packet.is_dir():
        raise RuntimeError("review packet is not a regular directory")
    manifest_path = packet / "packet-manifest.json"
    manifest = loop.load_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("kind") != "exact_reviewer_packet":
        raise RuntimeError("review packet manifest is invalid")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("review packet manifest has no file list")
    declared: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("review packet identity is invalid")
        relative = installer.safe_relative(str(entry.get("path", "")))
        path = installer.bundle_child(packet, relative)
        if relative in declared:
            raise RuntimeError("review packet repeats a file identity")
        declared.add(relative)
        if (
            entry.get("sha256") != loop.sha256_file(path)
            or entry.get("bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"review packet file changed: {relative}")
    observed = relative_files(packet) - {Path("packet-manifest.json")}
    if observed != declared:
        raise RuntimeError("review packet contains unmanifested or missing files")
    digest = loop.sha256_file(manifest_path)
    if not SHA256.fullmatch(digest):
        raise RuntimeError("review packet has an invalid SHA-256")
    plan, _operations = hf_helper.validate_plan(packet)
    return digest, plan


def review_response_file(reviews: Path, value: Any) -> Path:
    relative = installer.safe_relative(str(value))
    if len(relative.parts) != 1:
        raise RuntimeError("review response must be a direct child of reviews")
    path = reviews / relative
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("review response is not a regular file")
    return path.resolve(strict=True)


def validate_review(
    reviews: Path,
    reviewer: str,
    packet_digest: str,
    plan: dict[str, Any],
) -> dict[str, Any]:
    path = reviews / REVIEW_FILENAMES[reviewer]
    value = loop.load_json(path)
    prompt = review_response_file(reviews, value.get("prompt_file"))
    response = review_response_file(reviews, value.get("response_file"))
    prompt_text = prompt.read_text(encoding="utf-8")
    response_text = response.read_text(encoding="utf-8")
    if (
        value.get("schema_version") != 1
        or value.get("reviewer_model") != reviewer
        or value.get("packet_sha256") != packet_digest
        or value.get("reviewed_parent_commit") != plan["parent_commit"]
        or value.get("reviewed_operation_count") != len(plan["operations"])
        or not isinstance(value.get("provider"), str)
        or not value["provider"].strip()
        or not isinstance(value.get("provider_response_id"), str)
        or not value["provider_response_id"].strip()
        or value.get("prompt_sha256") != loop.sha256_file(prompt)
        or value.get("response_sha256") != loop.sha256_file(response)
        or packet_digest not in prompt_text
        or f"PACKET_SHA256: {packet_digest}" not in response_text
    ):
        raise RuntimeError(f"review evidence is not bound to this packet: {reviewer}")
    verdict = value.get("verdict")
    if verdict not in ("PASS", "BLOCK") or f"VERDICT: {verdict}" not in response_text:
        raise RuntimeError(f"review verdict is malformed: {reviewer}")
    return value


def wait_for_reviews(
    config: dict[str, Any],
    output: Path,
    packet_digest: str,
    plan: dict[str, Any],
    status_path: Path,
    deadline: float,
) -> list[dict[str, Any]] | None:
    reviews = output / "reviews"
    reviews.mkdir(exist_ok=True)
    while time.monotonic() < deadline:
        present = [
            reviewer
            for reviewer in EXPECTED_REVIEWERS
            if (reviews / REVIEW_FILENAMES[reviewer]).is_file()
        ]
        if len(present) == len(EXPECTED_REVIEWERS):
            evidence = [
                validate_review(reviews, reviewer, packet_digest, plan)
                for reviewer in EXPECTED_REVIEWERS
            ]
            blocked = [
                item["reviewer_model"]
                for item in evidence
                if item["verdict"] == "BLOCK"
            ]
            if blocked:
                write_status(
                    status_path,
                    "blocked_by_exact_review",
                    packet_sha256=packet_digest,
                    blocking_reviewers=blocked,
                )
                return None
            return evidence
        write_status(
            status_path,
            "waiting_for_exact_reviews",
            packet_sha256=packet_digest,
            received_reviewers=present,
            required_reviewers=list(EXPECTED_REVIEWERS),
        )
        time.sleep(float(config["poll_seconds"]))
    raise RuntimeError("publication wait exceeded before both exact reviews arrived")


def upload_command(
    config: dict[str, Any], packet: Path
) -> tuple[list[str], dict[str, str]]:
    environment = os.environ.copy()
    environment["HF_TOKEN"] = config["credential_ref"]
    command = [
        config["op_wrapper"]["path"],
        "run",
        "--",
        *helper_command(config, "upload", "--packet-dir", str(packet)),
    ]
    return command, environment


def publish(
    config: dict[str, Any], output: Path, packet: Path, plan: dict[str, Any]
) -> dict[str, Any]:
    verify_pins(config)
    current = inspect_public_head(config)
    if current != plan["parent_commit"]:
        raise RuntimeError(
            "public repository head changed after review; prepare and review a new packet"
        )
    command, environment = upload_command(config, packet)
    log_path = output / "huggingface-upload.log"
    if log_path.exists():
        raise RuntimeError(
            "an upload was already attempted; reconcile its public result before retrying"
        )
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=7200,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Hugging Face upload failed; inspect {log_path}")
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise RuntimeError("Hugging Face upload produced no result")
    result = json.loads(lines[-1])
    if (
        not isinstance(result, dict)
        or result.get("stage") not in ("published", "already_published")
        or result.get("repo_id") != config["repo_id"]
        or result.get("anonymous_verification") is not True
        or not GIT_SHA.fullmatch(str(result.get("commit", "")))
    ):
        raise RuntimeError("Hugging Face upload result is invalid")
    loop.atomic_json(output / "publication-result.json", result)
    return result


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
            raise RuntimeError("new publication watcher requires an absent output directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
        loop.atomic_json(config_path, config)

    with (output / ".publication.lock").open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another certified publication watcher holds the lock") from error
        verify_pins(config)
        if (output / "publication-result.json").is_file():
            result = loop.load_json(output / "publication-result.json")
            if result.get("anonymous_verification") is not True:
                raise RuntimeError("existing publication result is incomplete")
            write_status(status_path, "published", publication=result)
            return 0

        deadline = time.monotonic() + float(config["maximum_wait_hours"]) * 3600.0
        install_status = wait_for_install(config, status_path, deadline)
        if install_status is None:
            return 0
        release, active, provenance = verify_installed_release(
            config, install_status, output
        )
        packet, packet_digest, plan = prepare_packet(
            config, output, release, active, provenance
        )
        if args.prepare_only:
            write_status(
                status_path,
                "packet_ready_for_exact_reviews",
                packet=str(packet),
                packet_sha256=packet_digest,
                required_reviewers=list(EXPECTED_REVIEWERS),
            )
            return 0
        reviews = wait_for_reviews(
            config, output, packet_digest, plan, status_path, deadline
        )
        if reviews is None:
            return 0
        write_status(
            status_path,
            "publishing_reviewed_bytes",
            packet_sha256=packet_digest,
            reviewers=[item["reviewer_model"] for item in reviews],
        )
        result = publish(config, output, packet, plan)
        write_status(
            status_path,
            "published",
            packet_sha256=packet_digest,
            publication=result,
            reviewers=[item["reviewer_model"] for item in reviews],
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
                f"certified BabyLLM publication: could not persist failure: {status_error}",
                file=sys.stderr,
            )
        print(f"certified BabyLLM publication: {error}", file=sys.stderr)
        raise SystemExit(1)
