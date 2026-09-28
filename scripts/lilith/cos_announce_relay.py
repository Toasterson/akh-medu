#!/usr/bin/env python3
"""Lilith Chief-of-Staff announcement relay (ADR 116 Decision 1 revision 4,
plan slice C2 "PR B"; see `src/agent/chief_of_staff/ledger.rs`).

The guard (`src/agent/chief_of_staff/guard.rs`) writes every operator-class
(T3) action Lilith takes as a `lilith/cos-announcement@1` line in her JSONL
Chief-of-Staff log (`src/agent/chief_of_staff/ledger.rs`, schema
`skills/chief-of-staff/schemas/cos-announcement.schema.json`). The log also
holds `lilith/cos-spend-reserved@1` / `lilith/cos-spend-settled@1` lines,
which this relay reads past but never posts.

This script tails that log by byte offset (crash-safe: the offset and the
processed-line digest set live in a state file beside the log, written
tmp-then-`os.replace`) and posts each announcement, once, to the operator's
Chief-of-Staff thread on the Agent Board as a readable `board/note@v1`
message. It is meant to run once per invocation under a launchd schedule
(the pattern `scripts/lilith/cos_recorder.py` already uses) — no built-in
polling loop.

Ordering and blocking: announcements are relayed in log order. A line that
cannot be parsed as JSON, has no recognized `contract`, or fails the
announcement's minimal shape check is a **poison line**: it is appended to a
quarantine file with a reason and the byte offset skips past it — it never
blocks the rest of the stream. A **board post failure** (network, or the
board rejecting the write) is different: it is retried with bounded
exponential backoff within this run, and if it still fails the offset is
left exactly before that line, so the next scheduled run retries it in
order — a real relay failure blocks only until it is fixed, it never silently
drops an announcement.

Exactly-once: each processed line's sha256 digest is both the board write's
`idempotency_key` (so a retry after a mid-post crash cannot double-post on
the board's side) and is recorded in the state file's bounded digest ring
(so a re-run that somehow revisits the same byte range skips the repost
locally too).

Secrets: the board bearer token is read from a `secrets/board-<identity>.json`
file (`{"bearer_token": "..."}`, the same shape `daemon_credentials.py` and
`observer.py` already use for the Roman akhs) and is never written to stdout,
stderr, the state file or the quarantine file — every string that might carry
it is passed through `redact()` first.

Usage:
    cos_announce_relay.py run --log PATH --thread-id UUID [options]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

SCHEMA = "lilith/cos-announce-relay-state@1"
ANNOUNCEMENT_CONTRACT = "lilith/cos-announcement@1"
SKIPPED_CONTRACTS = frozenset({"lilith/cos-spend-reserved@1", "lilith/cos-spend-settled@1"})
BOARD_NOTE_MESSAGE_TYPE = "board/note@v1"
DEFAULT_BOARD_URL = "http://127.0.0.1:18089"
DEFAULT_IDENTITY = "lilith"
DIGEST_RING_SIZE = 4096
MAX_TEXT_ARGUMENTS_CHARS = 500
MAX_SUMMARY_CHARS = 220
STATE_SUFFIX = ".relay-state.json"
QUARANTINE_SUFFIX = ".relay-quarantine.jsonl"

REQUIRED_ANNOUNCEMENT_FIELDS = (
    "akh", "announced_at", "tool", "arguments", "action", "receipts",
    "reason", "rollback", "rule_checks", "outcome",
)
LINK_RECEIPT_KINDS = frozenset({"pull-request", "work-item", "commit", "board-message", "document"})


# --- Secret redaction --------------------------------------------------------

_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)
_AUTH_HEADER_RE = re.compile(r"(['\"]?Authorization['\"]?\s*[:=]\s*)['\"]?[^'\",}\s]+", re.IGNORECASE)


def redact(text: str) -> str:
    """Strip bearer tokens / Authorization header values from a string."""
    text = _BEARER_RE.sub("Bearer [redacted]", text)
    text = _AUTH_HEADER_RE.sub(r"\1[redacted]", text)
    return text


# --- Atomic writes ------------------------------------------------------------

def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, body)


def atomic_append_bytes(path: Path, data: bytes) -> None:
    """Append `data` durably. One `write` to an `O_APPEND` file, matching the
    guard's own log so a concurrent quarantine append can never interleave."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


# --- Relay state --------------------------------------------------------------

@dataclasses.dataclass
class RelayState:
    log_path: str
    byte_offset: int = 0
    posted_digests: list[str] = dataclasses.field(default_factory=list)
    posted: int = 0
    skipped: int = 0
    quarantined: int = 0

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "log_path": self.log_path,
            "byte_offset": self.byte_offset,
            "posted_digests": self.posted_digests,
            "stats": {"posted": self.posted, "skipped": self.skipped, "quarantined": self.quarantined},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def remember(self, digest: str) -> None:
        if digest in self.posted_digests:
            return
        self.posted_digests.append(digest)
        if len(self.posted_digests) > DIGEST_RING_SIZE:
            del self.posted_digests[: len(self.posted_digests) - DIGEST_RING_SIZE]

    def save(self, path: Path) -> None:
        atomic_write_json(path, self.to_dict())


def load_state(path: Path, log_path: Path) -> RelayState:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return RelayState(log_path=str(log_path))
    stats = raw.get("stats") or {}
    return RelayState(
        log_path=raw.get("log_path", str(log_path)),
        byte_offset=int(raw.get("byte_offset", 0)),
        posted_digests=list(raw.get("posted_digests", [])),
        posted=int(stats.get("posted", 0)),
        skipped=int(stats.get("skipped", 0)),
        quarantined=int(stats.get("quarantined", 0)),
    )


# --- Log tailing ---------------------------------------------------------------

def read_new_lines(log_path: Path, offset: int) -> list[bytes]:
    """Every complete (`\\n`-terminated) line strictly after `offset`.

    A trailing partial line (the writer is mid-append) is left unread: the
    next run picks it up once it is complete. Missing file -> no new lines
    (nothing has been written yet).
    """
    try:
        with log_path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except FileNotFoundError:
        return []
    last_newline = data.rfind(b"\n")
    if last_newline == -1:
        return []
    complete = data[: last_newline + 1]
    return complete.split(b"\n")[:-1]


# --- Announcement validation + formatting --------------------------------------

class PoisonLine(ValueError):
    """A line that looks like it should be an announcement but is not one."""


def parse_record(raw_line: bytes) -> dict:
    """Parse and minimally validate one log line as a `cos-announcement@1`.

    Raises `PoisonLine` (quarantine) for a broken or unrecognized record.
    Returns `{}` for a recognized-but-uninteresting record (spend
    reservation/settlement) that the caller should skip without posting.
    """
    text = raw_line.decode("utf-8", errors="strict")
    try:
        record = json.loads(text)
    except (UnicodeDecodeError, ValueError) as error:
        raise PoisonLine(f"invalid_json: {redact(str(error))}") from None
    if not isinstance(record, dict):
        raise PoisonLine("not_an_object")
    contract = record.get("contract")
    if contract in SKIPPED_CONTRACTS:
        return {}
    if contract != ANNOUNCEMENT_CONTRACT:
        raise PoisonLine(f"unknown_contract: {contract!r}")
    missing = [f for f in REQUIRED_ANNOUNCEMENT_FIELDS if f not in record]
    if missing:
        raise PoisonLine(f"missing_fields: {sorted(missing)}")
    if not isinstance(record.get("receipts"), list) or not record["receipts"]:
        raise PoisonLine("receipts_empty_or_not_a_list")
    if not isinstance(record.get("reason"), str) or not record["reason"].strip():
        raise PoisonLine("reason_empty")
    action = record.get("action")
    if not isinstance(action, dict) or "kind" not in action:
        raise PoisonLine("action_missing_kind")
    rollback = record.get("rollback")
    if not isinstance(rollback, dict) or "kind" not in rollback:
        raise PoisonLine("rollback_missing_kind")
    outcome = record.get("outcome")
    if not isinstance(outcome, dict) or "status" not in outcome or "detail" not in outcome:
        raise PoisonLine("outcome_malformed")
    return record


def _iso(unix_seconds: Any) -> str:
    try:
        return datetime.fromtimestamp(int(unix_seconds), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return str(unix_seconds)


def _format_action(action: dict) -> str:
    kind = action.get("kind", "?")
    if kind == "deploy":
        return f"deploy {action.get('artifact_sha')} -> {action.get('target')}"
    if kind == "restart":
        return f"restart {action.get('service')} at {action.get('artifact_sha')}"
    if kind == "merge":
        return f"merge {action.get('change')}"
    if kind == "tracker":
        return f"tracker {action.get('operation')} {action.get('target')}"
    if kind == "spend":
        amount = action.get("amount") or {}
        minor = amount.get("minor_units")
        currency = amount.get("currency", "")
        display = f"{minor / 100:.2f} {currency}" if isinstance(minor, (int, float)) else f"{minor} minor units {currency}"
        return f"spend {display} -- {action.get('purpose')}"
    if kind in ("delete", "overwrite"):
        parts = [f"{kind} {action.get('target')}", f"({action.get('reversibility')}, digest {action.get('observed_digest')})"]
        if action.get("copy"):
            parts.append(f"copy at {action['copy']}")
        return " ".join(parts)
    if kind == "outward":
        return f"outward -> {action.get('destination')}"
    if kind == "activate-adapter":
        return f"activate-adapter {action.get('adapter')}"
    return f"{kind} (unrecognized action shape): {json.dumps(action, sort_keys=True)[:200]}"


def _format_rollback(rollback: dict) -> str:
    kind = rollback.get("kind", "?")
    if kind == "revert":
        return f"revert -- {rollback.get('steps')}"
    if kind == "restore-from-copy":
        return f"restore from copy -- {rollback.get('location')}"
    if kind == "irreversible":
        return f"IRREVERSIBLE -- mitigation: {rollback.get('mitigation')}"
    return f"{kind} (unrecognized rollback shape)"


def format_announcement(record: dict) -> tuple[str, str]:
    """Render one announcement record as (text, summary) for `board/note@v1`."""
    action = record.get("action") or {}
    rollback = record.get("rollback") or {}
    outcome = record.get("outcome") or {}
    receipts = record.get("receipts") or []
    rule_checks = record.get("rule_checks") or []
    action_kind = action.get("kind", "?")
    tool = record.get("tool", "?")
    akh = record.get("akh", "?")

    lines = [
        f"Lilith (akh={akh}) took an operator-class action.",
        "",
        f"Action: {_format_action(action)}",
        f"Tool: {tool}",
        f"Reason: {record.get('reason', '')}",
        f"Outcome: {outcome.get('status', '?')} -- {outcome.get('detail', '')}",
        f"Rollback: {_format_rollback(rollback)}",
        "",
        "Receipts:",
    ]
    for r in receipts:
        lines.append(f"  - [{r.get('kind', '?')}] {r.get('reference', '?')} (observed {_iso(r.get('observed_at'))})")
    links = [r for r in receipts if r.get("kind") in LINK_RECEIPT_KINDS]
    if links:
        lines.append("")
        lines.append("Links:")
        for r in links:
            lines.append(f"  - {r.get('reference')}")
    if rule_checks:
        lines.append("")
        lines.append("Rule checks:")
        for c in rule_checks:
            lines.append(f"  - {c.get('rule', '?')}: {c.get('evidence', '')}")
    args_json = json.dumps(record.get("arguments") or {}, sort_keys=True, ensure_ascii=False)
    lines.append("")
    if len(args_json) <= MAX_TEXT_ARGUMENTS_CHARS:
        lines.append(f"Arguments: {args_json}")
    else:
        lines.append(f"Arguments: (omitted, {len(args_json)} bytes -- see the Chief-of-Staff log for the full record)")
    lines.append("")
    lines.append(f"Announced at: {_iso(record.get('announced_at'))}")
    text = "\n".join(lines)

    summary = f"Lilith T3 {action_kind}: {tool} -- {outcome.get('status', '?')}"
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[: MAX_SUMMARY_CHARS - 1] + "…"
    return text, summary


# --- Board posting --------------------------------------------------------------

class Transport(Protocol):
    def post(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        ...


class UrllibTransport:
    """Default transport: stdlib `urllib`, no extra dependency."""

    def post(self, url: str, headers: dict[str, str], body: bytes) -> tuple[int, bytes]:
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()


class BoardPostError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(redact(message))
        self.status = status


def _retryable(status: int | None) -> bool:
    return status is None or status == 429 or status >= 500


@dataclasses.dataclass
class RetryPolicy:
    max_attempts: int = 6
    initial_delay: float = 0.5
    max_delay: float = 30.0

    def delay_for_attempt(self, attempt: int) -> float:
        return min(self.initial_delay * (2 ** min(attempt, 10)), self.max_delay)


def post_board_note(
    *,
    transport: Transport,
    board_url: str,
    thread_id: str,
    token: str,
    text: str,
    summary: str,
    idempotency_key: str,
    retry: RetryPolicy,
    sleep=time.sleep,
) -> dict:
    """POST one `board/note@v1` message, retrying transient failures.

    Never logs `token`: any exception raised carries only a redacted message,
    and the token is never embedded in a variable that outlives this call.
    """
    url = f"{board_url.rstrip('/')}/v1/threads/{thread_id}/messages"
    envelope = {
        "message_type": BOARD_NOTE_MESSAGE_TYPE,
        "payload": {"text": text},
        "summary": summary,
        "retention": {"kind": "durable"},
        "idempotency_key": idempotency_key,
    }
    body = json.dumps(envelope).encode("utf-8")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json"}
    last_error: BoardPostError | None = None
    for attempt in range(retry.max_attempts):
        try:
            status, response_body = transport.post(url, headers, body)
        except (OSError, urllib.error.URLError) as error:
            last_error = BoardPostError(None, f"board transport error: {error}")
        else:
            if 200 <= status < 300:
                try:
                    return json.loads(response_body.decode("utf-8")) if response_body else {}
                except ValueError:
                    return {}
            last_error = BoardPostError(status, f"board post failed (HTTP {status}): {response_body[:500]!r}")
            if not _retryable(status):
                raise last_error
        if attempt + 1 < retry.max_attempts:
            sleep(retry.delay_for_attempt(attempt))
    assert last_error is not None
    raise last_error


# --- Secrets -------------------------------------------------------------------

def resolve_token(secrets_file: Path) -> str:
    try:
        payload = json.loads(secrets_file.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read board identity secrets file {secrets_file}: {redact(str(error))}") from None
    token = payload.get("bearer_token")
    if not token or not isinstance(token, str):
        raise RuntimeError(f"{secrets_file} has no non-empty 'bearer_token'")
    return token


# --- Quarantine ------------------------------------------------------------------

def quarantine(path: Path, *, raw_line: bytes, reason: str, offset: int) -> None:
    entry = {
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
        "offset": offset,
        "reason": redact(reason),
        "line": redact(raw_line.decode("utf-8", errors="replace")),
    }
    atomic_append_bytes(path, (json.dumps(entry, sort_keys=True) + "\n").encode("utf-8"))


# --- Orchestration ---------------------------------------------------------------

@dataclasses.dataclass
class RelayResult:
    posted: int = 0
    skipped: int = 0
    quarantined: int = 0
    stopped_early: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        out = {"posted": self.posted, "skipped": self.skipped, "quarantined": self.quarantined, "stopped_early": self.stopped_early}
        if self.error:
            out["error"] = self.error
        return out


def run_relay(
    *,
    log_path: Path,
    state_path: Path,
    quarantine_path: Path,
    board_url: str,
    thread_id: str,
    token: str,
    transport: Transport | None = None,
    retry: RetryPolicy | None = None,
    sleep=time.sleep,
) -> RelayResult:
    transport = transport or UrllibTransport()
    retry = retry or RetryPolicy()
    state = load_state(state_path, log_path)
    result = RelayResult()

    lines = read_new_lines(log_path, state.byte_offset)
    for raw_line in lines:
        advance = len(raw_line) + 1  # + the '\n' the split consumed
        if not raw_line.strip():
            state.byte_offset += advance
            continue
        digest = hashlib.sha256(raw_line).hexdigest()
        try:
            record = parse_record(raw_line)
        except PoisonLine as poison:
            quarantine(quarantine_path, raw_line=raw_line, reason=str(poison), offset=state.byte_offset)
            state.quarantined += 1
            result.quarantined += 1
            state.byte_offset += advance
            state.save(state_path)
            continue
        if not record:
            # A recognized non-announcement record (spend reserved/settled).
            state.skipped += 1
            result.skipped += 1
            state.byte_offset += advance
            state.save(state_path)
            continue
        if digest in state.posted_digests:
            # Already relayed (defense in depth alongside the byte offset).
            state.byte_offset += advance
            state.save(state_path)
            continue
        text, summary = format_announcement(record)
        idempotency_key = f"lilith-cos-announce:{digest}"
        try:
            post_board_note(
                transport=transport,
                board_url=board_url,
                thread_id=thread_id,
                token=token,
                text=text,
                summary=summary,
                idempotency_key=idempotency_key,
                retry=retry,
                sleep=sleep,
            )
        except BoardPostError as error:
            # Leave the offset before this line: the next scheduled run
            # retries it, in order. This line (and everything behind it)
            # stays unrelayed until the systemic failure clears.
            result.stopped_early = True
            result.error = str(error)
            state.save(state_path)
            return result
        state.remember(digest)
        state.posted += 1
        result.posted += 1
        state.byte_offset += advance
        state.save(state_path)

    state.save(state_path)
    return result


# --- CLI -----------------------------------------------------------------------

def default_secrets_file(root: Path, identity: str) -> Path:
    return root / "secrets" / f"board-{identity}.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relay Lilith's Chief-of-Staff announcements to the Agent Board")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="relay every new announcement once and exit")
    run_p.add_argument("--log", type=Path, default=os.environ.get("LILITH_COS_LOG"),
                        help="path to the Chief-of-Staff JSONL log (env LILITH_COS_LOG)")
    run_p.add_argument("--thread-id", default=os.environ.get("LILITH_COS_THREAD_ID"),
                        help="Agent Board thread UUID to post to (env LILITH_COS_THREAD_ID)")
    run_p.add_argument("--root", type=Path,
                        default=Path(os.environ.get("LILITH_ROOT", str(Path.home() / "Library/Application Support/roman-akhs"))),
                        help="deployment root, used to default the secrets file (env LILITH_ROOT)")
    run_p.add_argument("--identity", default=os.environ.get("LILITH_BOARD_IDENTITY", DEFAULT_IDENTITY),
                        help="board identity name; secrets file defaults to <root>/secrets/board-<identity>.json "
                             "(env LILITH_BOARD_IDENTITY)")
    run_p.add_argument("--secrets-file", type=Path, default=os.environ.get("LILITH_BOARD_SECRETS"),
                        help="overrides the identity-derived secrets file path (env LILITH_BOARD_SECRETS)")
    run_p.add_argument("--board-url", default=os.environ.get("LILITH_BOARD_URL", DEFAULT_BOARD_URL),
                        help="Agent Board base URL (env LILITH_BOARD_URL)")
    run_p.add_argument("--state", type=Path, default=None, help="default: <log>" + STATE_SUFFIX)
    run_p.add_argument("--quarantine", type=Path, default=None, help="default: <log>" + QUARANTINE_SUFFIX)
    run_p.add_argument("--max-attempts", type=int, default=6, help="board post retry attempts (default 6)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run":
        return 1

    if not args.log:
        print("cos_announce_relay: --log (or LILITH_COS_LOG) is required", file=sys.stderr)
        return 2
    if not args.thread_id:
        print("cos_announce_relay: --thread-id (or LILITH_COS_THREAD_ID) is required", file=sys.stderr)
        return 2

    log_path = args.log
    state_path = args.state or log_path.with_name(log_path.name + STATE_SUFFIX)
    quarantine_path = args.quarantine or log_path.with_name(log_path.name + QUARANTINE_SUFFIX)
    secrets_file = args.secrets_file or default_secrets_file(args.root, args.identity)

    try:
        token = resolve_token(secrets_file)
    except RuntimeError as error:
        print(f"cos_announce_relay: {redact(str(error))}", file=sys.stderr)
        return 2

    result = run_relay(
        log_path=log_path,
        state_path=state_path,
        quarantine_path=quarantine_path,
        board_url=args.board_url,
        thread_id=args.thread_id,
        token=token,
        retry=RetryPolicy(max_attempts=args.max_attempts),
    )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 1 if result.stopped_early else 0


if __name__ == "__main__":
    raise SystemExit(main())
