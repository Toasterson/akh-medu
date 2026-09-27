#!/usr/bin/env python3
"""Create a privacy-safe, leakage-resistant Lilith transcript dataset.

Input is a JSON array of complete ``get_session`` MCP responses. The exporter
must page each session to completion before using this tool. Raw database rows
are deliberately unsupported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

SCHEMA = "akh/lilith-transcript-admission@1"
HEX64 = re.compile(r"[0-9a-f]{64}")
SECRET_PATTERNS = (
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"(?i)\b(?:bearer|api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+"), "[REDACTED_CREDENTIAL]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"/home/[^/\s]+/"), "/home/[USER]/"),
    (re.compile(r"/Users/[^/\s]+/"), "/Users/[USER]/"),
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def redact(text: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _session_hash(source: str, session_id: str) -> str:
    return sha({"source": source, "session_id": session_id})


def admit(
    sessions: list[dict[str, Any]], *, deny_sessions: set[str],
    deny_families: set[str], eval_fingerprints: set[str],
    evidence: dict[str, dict[str, Any]], salt: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    seen: set[str] = set()

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    for session in sessions:
        required = ("source", "session_id", "events")
        if any(key not in session for key in required) or session.get("truncated") is True or session.get("continue_from"):
            reject("incomplete_or_invalid_session")
            continue
        source, session_id = str(session["source"]), str(session["session_id"])
        session_hash = _session_hash(source, session_id)
        if session_hash in deny_sessions:
            reject("eval_session")
            continue
        events = session["events"]
        if not isinstance(events, list):
            reject("incomplete_or_invalid_session")
            continue
        pending: dict[str, Any] | None = None
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get("render"), str):
                continue
            kind = event.get("kind")
            if kind == "prompt":
                pending = event
                continue
            if kind != "reply" or pending is None:
                continue
            prompt, reply = redact(pending["render"]), redact(event["render"])
            pending_ref = {key: pending.get(key) for key in ("stream_id", "seq", "subseq")}
            reply_ref = {key: event.get(key) for key in ("stream_id", "seq", "subseq")}
            content_hash = sha({"input": normalized(prompt), "output": normalized(reply)})
            if content_hash in seen:
                reject("duplicate_content")
                pending = None
                continue
            if content_hash in eval_fingerprints:
                reject("eval_content_collision")
                pending = None
                continue
            seen.add(content_hash)
            record_id = sha({"session": session_hash, "event_refs": [pending_ref, reply_ref]})
            ext = evidence.get(record_id)
            training_kind = "supervised_imitation"
            trajectory: dict[str, Any] = {"steps": []}
            admitted_evidence: dict[str, Any] = {"kind": "none"}
            family_tags = sorted(set(session.get("family_tags", [])))
            if set(family_tags) & deny_families:
                reject("eval_family")
                pending = None
                continue
            if ext is not None:
                steps = ext.get("trajectory", {}).get("steps")
                hashes_ok = _valid_hash(ext.get("verifier_sha256")) and _valid_hash(ext.get("outcome_sha256"))
                if ext.get("kind") != "external_verifier" or ext.get("passed") is not True or not isinstance(steps, list) or not steps or not hashes_ok or ext["outcome_sha256"] != sha(steps):
                    reject("invalid_external_evidence")
                    pending = None
                    continue
                training_kind = "verified_outcome"
                trajectory = {"steps": steps}
                admitted_evidence = {key: ext[key] for key in ("kind", "verifier_sha256", "outcome_sha256", "passed")}
            split_bucket = int(sha({"salt": salt, "session": session_hash})[:8], 16) % 10
            rows.append({
                "schema": SCHEMA, "record_id": record_id,
                "split": "validation" if split_bucket == 0 else "train",
                "training_kind": training_kind, "input": prompt, "output": reply,
                "trajectory": trajectory,
                "provenance": {"source": source, "session_sha256": session_hash,
                               "event_refs": [pending_ref, reply_ref], "content_sha256": content_hash},
                "family_tags": family_tags, "evidence": admitted_evidence,
            })
            pending = None
    rows.sort(key=lambda row: row["record_id"])
    manifest = {
        "schema": SCHEMA, "records": len(rows),
        "training_kinds": {kind: sum(r["training_kind"] == kind for r in rows) for kind in ("supervised_imitation", "verified_outcome")},
        "splits": {split: sum(r["split"] == split for r in rows) for split in ("train", "validation")},
        "rejected": dict(sorted(rejected.items())),
        "dataset_sha256": sha([sha(row) for row in rows]),
        "deny_sessions_sha256": sha(sorted(deny_sessions)),
        "deny_families_sha256": sha(sorted(deny_families)),
        "eval_fingerprints_sha256": sha(sorted(eval_fingerprints)),
        "split_policy": "sha256(salt,source+session_id), session-bound, 90/10",
        "limitations": ["MCP-rendered redaction is trusted and augmented locally", "supervised_imitation rows carry no reward"],
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite an admission artifact")
    policy = json.loads(args.policy.read_text())
    evidence_rows = json.loads(args.evidence.read_text()) if args.evidence else []
    evidence = {row["record_id"]: row for row in evidence_rows}
    rows, manifest = admit(json.loads(args.sessions.read_text()),
        deny_sessions=set(policy.get("deny_session_sha256", [])),
        deny_families=set(policy.get("deny_family_tags", [])),
        eval_fingerprints=set(policy.get("eval_content_sha256", [])), evidence=evidence, salt=args.salt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.chmod(args.output, 0o600)
    os.chmod(args.manifest, 0o600)


if __name__ == "__main__":
    main()
