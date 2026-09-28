#!/usr/bin/env python3
"""Lilith C5: transcript admission with tool trajectories, and the CoS corpus export.

Schema ``akh/lilith-transcript-trajectory@2`` extends
``akh/lilith-transcript-admission@1`` (``lilith_transcript_admission.py``, left
unchanged). The unit is the **whole assistant turn**: from the prompt through
every ``tool_call`` / ``tool_result`` / interim ``assistant_text`` step up to the
last assistant text before the next prompt (``turn_final_reply``), so a reply's
status claims keep their receipts (C5a finding 4, Anima WI 01a0e270-7792,
``docs/ai/research/2026-09-27-lilith-c5a-transcript-role-fit.md``).

Pairing is the v@1 rule applied to the transcript MCP's ``get_session``
``order="spliced"`` event order: the first reply after the prompt is the one the
C5a panel rated (``rated_reply``, with its ``step_index``), so ``record_id`` is
exactly the v@1 ``record_id`` and the panel decisions join on it. Steps are
collected in *deep* spliced order: a subagent stream's tool steps are inserted
right after the ``tool_result`` that spawned it (``toolUseResult.agentId``),
recursively. The training target is the steps with ``train: true`` (her own
tool calls and assistant text); tool results and subagent work are context.

Bounds: tool args/results 8 KB each (head+tail, ``full_sha256`` of the full
redacted text kept), at most ``MAX_STEPS`` steps (rated/final reply and own
assistant text kept first, then own tool steps nearest the turn's ends,
subagent steps last; ``steps_elided``), and a per-record budget of
``RECORD_BUDGET_BYTES`` met by halving the longest tool results first, then
args (``bytes_trimmed``; ``over_budget`` if still above).

Sources:

- **Local Claude Code adapter** (default): reads
  ``~/.claude/projects/<slug>/<session_id>.jsonl`` plus
  ``<slug>/<session_id>/subagents/**/agent-<id>.jsonl`` directly. Event kinds
  follow ``crates/akh-transcripts/src/normalize/claude_code.rs``; ``seq`` is the
  0-based index of non-empty lines and ``subseq`` the content-block index, as in
  ``crates/akh-transcripts/src/adapter``.
- **Transcript MCP** (fallback for sessions not on disk): only when an MCP URL
  and a token are configured. Its renders are pre-truncated (tool_call 400 B,
  tool_result 600 B), which the steps record as ``upstream_truncated``.

Redaction version ``lilith-redact@2``: the ten credential patterns of
``crates/akh-transcripts/src/redact.rs``, the path and e-mail patterns of the
v@1 admission script, plus RFC 1918 IPv4 addresses and the ``-Users-<name>-`` /
``-home-<name>-`` path-slug form. Structural patterns only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lilith_transcript_admission import (  # noqa: E402
    SCHEMA as ADMISSION_SCHEMA,
    _session_hash,
    normalized,
    sha,
)

SCHEMA = "akh/lilith-transcript-trajectory@2"
MANIFEST_SCHEMA = "lilith/cos-corpus-manifest@1"
REDACTION_VERSION = "lilith-redact@2"
SOURCE = "claude_code"
RESULT_CAP_BYTES = 8192
ARGS_CAP_BYTES = 8192
MAX_STEPS = 128
RECORD_BUDGET_BYTES = 128 * 1024
MIN_TRIM_BYTES = 512
SESSION_SHARE_CAP = 0.25
VALIDATION_SHARE = 0.10

# ── redaction ────────────────────────────────────────────────────────────────

_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
# (name, pattern, replacement). Order matters: the whole-block private key
# first, then credentials (redact.rs), then identity/network structure.
REDACTIONS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED:private_key]"),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:aws_access_key]"),
    ("github_pat", re.compile(r"gh[posru]_[A-Za-z0-9]{30,}"), "[REDACTED:github_pat]"),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"), "[REDACTED:anthropic_key]"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{32,}"), "[REDACTED:openai_key]"),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "[REDACTED:slack_token]"),
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]*"), "[REDACTED:jwt]"),
    ("bearer", re.compile(r"Bearer\s+[A-Za-z0-9._\-]{20,}"), "[REDACTED:bearer]"),
    ("pg_url_password", re.compile(r"postgres(?:ql)?://[^:/\s]+:[^@\s]{3,}@"), "[REDACTED:pg_url_password]"),
    ("assignment", re.compile(r"""(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[=:]\s*"?'?[^\s"',}]{8,}"""), "[REDACTED:assignment]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("home_path", re.compile(r"/home/[^/\s\"'`]+"), "/home/[USER]"),
    ("users_path", re.compile(r"/Users/[^/\s\"'`]+"), "/Users/[USER]"),
    # Claude Code / tmp path slugs: `/` becomes `-`, so `-Users-toasty-ws-...`.
    ("users_slug", re.compile(r"-Users-[A-Za-z0-9._]+-"), "-Users-[USER]-"),
    ("home_slug", re.compile(r"-home-[A-Za-z0-9._]+-"), "-home-[USER]-"),
    # RFC 1918 only; not part of a longer dotted number (versions, OIDs).
    ("lan_ipv4", re.compile(
        rf"(?<![\d.])(?:10\.{_OCTET}\.{_OCTET}\.{_OCTET}|172\.(?:1[6-9]|2\d|3[01])\.{_OCTET}\.{_OCTET}"
        rf"|192\.168\.{_OCTET}\.{_OCTET})(?!\d)(?!\.\d)"),
     "[LAN_IP]"),
)
_PEM_HEADER = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
_HOME_OWNER = re.compile(r"^/(?:Users|home)/([^/\s]+)")
ACCOUNT_RULE = "account_name"


def account_names(cwds: Iterable[str]) -> frozenset[str]:
    """Home-directory owners named by a session's own ``cwd`` metadata.

    Structural: the token is whatever sits in the home-directory slot of the
    working directory the session ran in, so ``ps``/``ls -l`` owner columns and
    reverse-DNS labels that repeat it are caught without any word list.
    """
    names = set()
    for cwd in cwds:
        match = _HOME_OWNER.match(cwd)
        if match and len(match.group(1)) >= 2:
            names.add(match.group(1))
    return frozenset(names)


def redact(text: str, accounts: frozenset[str] = frozenset()) -> str:
    """Apply ``lilith-redact@2``. An unterminated PEM header withholds the text (redact.rs)."""
    for _name, pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    for name in sorted(accounts, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", "[USER]", text)
    if _PEM_HEADER.search(text):
        return "[WITHHELD: private_key]"
    return text


def bound(text: str, cap: int) -> tuple[str, bool]:
    """Head-and-tail truncate to ``cap`` UTF-8 bytes with an explicit elision marker."""
    raw = text.encode()
    if len(raw) <= cap:
        return text, False
    head = raw[: cap // 2].decode(errors="ignore")
    tail = raw[len(raw) - (cap - cap // 2):].decode(errors="ignore")
    elided = len(raw) - len(head.encode()) - len(tail.encode())
    return f"{head}\n…[{elided} bytes elided]…\n{tail}", True


# ── events ───────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class EventRef:
    stream_id: str
    seq: int
    subseq: int

    def as_json(self) -> dict[str, Any]:
        return {"stream_id": self.stream_id, "seq": self.seq, "subseq": self.subseq}


@dataclasses.dataclass(frozen=True)
class Event:
    """One visible event of kind prompt | reply | tool_call | tool_result."""

    ref: EventRef
    kind: str
    text: str | None
    tool_name: str | None = None
    tool_call_id: str | None = None
    spawned_agent: str | None = None
    upstream_truncated: bool = False


class SessionParseError(Exception):
    """A session file exists but cannot be read into events."""


@dataclasses.dataclass(frozen=True)
class SessionEvents:
    """One session's visible events, ``stream_id -> events`` in spliced top-level order."""

    session_sha256: str
    streams: dict[str, list[Event]]
    adapter: str
    accounts: frozenset[str] = frozenset()


def _tool_result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif isinstance(block, dict):
                parts.append(f"[{block.get('type', 'block')}]")
        return "\n".join(parts)
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def _line_events(stream_id: str, seq: int, doc: dict[str, Any]) -> list[Event]:
    """normalize/claude_code.rs, reduced to the four kinds admission reads."""
    record_type = doc.get("type")
    if record_type not in ("user", "assistant"):
        return []
    message = doc.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    fallback_kind = "prompt" if record_type == "user" else "reply"
    if record_type == "user" and isinstance(content, str):
        return [Event(EventRef(stream_id, seq, 0), "prompt", content)]
    if not isinstance(content, list):
        return [Event(EventRef(stream_id, seq, 0), fallback_kind, None)]  # degraded
    agent = doc.get("toolUseResult", {}).get("agentId") if isinstance(doc.get("toolUseResult"), dict) else None
    events: list[Event] = []
    for index, block in enumerate(content):
        ref = EventRef(stream_id, seq, index)
        block_type = block.get("type") if isinstance(block, dict) else None
        if record_type == "user" and block_type == "text":
            events.append(Event(ref, "prompt", block.get("text")))
        elif record_type == "user" and block_type == "tool_result":
            events.append(Event(ref, "tool_result", _tool_result_text(block.get("content")),
                                tool_call_id=block.get("tool_use_id"), spawned_agent=agent))
        elif record_type == "assistant" and block_type == "text":
            events.append(Event(ref, "reply", block.get("text")))
        elif record_type == "assistant" and block_type == "tool_use":
            args = block.get("input")
            events.append(Event(ref, "tool_call", json.dumps(args, ensure_ascii=False, sort_keys=True) if args is not None else None,
                                tool_name=block.get("name"), tool_call_id=block.get("id")))
        elif record_type == "assistant" and block_type in ("thinking", "fallback"):
            continue  # thought is hidden by default; fallback is an error event
        else:
            events.append(Event(ref, fallback_kind, None))  # degraded block
    return events


def read_stream(path: Path, stream_id: str) -> tuple[list[Event], str | None, set[str]]:
    """Events of one stream, its first timestamp and its ``cwd`` values. Undecodable lines yield no event."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SessionParseError(f"{path.name}: {type(error).__name__}") from None
    events: list[Event] = []
    first_ts: str | None = None
    cwds: set[str] = set()
    complete = raw[: raw.rfind(b"\n") + 1]
    lines = [line for line in complete.split(b"\n") if line]
    for seq, line in enumerate(lines):
        try:
            doc = json.loads(line.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        if not isinstance(doc, dict):
            continue
        if first_ts is None and isinstance(doc.get("timestamp"), str):
            first_ts = doc["timestamp"]
        if isinstance(doc.get("cwd"), str):
            cwds.add(doc["cwd"])
        events.extend(_line_events(stream_id, seq, doc))
    return events, first_ts, cwds


def discover_streams(main_path: Path) -> dict[str, Path]:
    """Main file plus every ``agent-<id>.jsonl`` under ``<session_id>/subagents``."""
    streams = {"main": main_path}
    session_dir = main_path.with_suffix("")
    subagents = session_dir / "subagents"
    if subagents.is_dir():
        for path in sorted(subagents.rglob("agent-*.jsonl")):
            streams.setdefault(path.stem, path)
    return streams


def load_local_session(main_path: Path) -> SessionEvents:
    """Streams in the MCP's spliced top-level order: main first, then by first timestamp."""
    loaded: list[tuple[str, list[Event], str | None]] = []
    cwds: set[str] = set()
    for stream_id, path in discover_streams(main_path).items():
        events, first_ts, stream_cwds = read_stream(path, stream_id)
        loaded.append((stream_id, events, first_ts))
        cwds |= stream_cwds
    loaded.sort(key=lambda item: (item[0] != "main", item[2] is not None, item[2] or "", item[0]))
    return SessionEvents(_session_hash(SOURCE, main_path.stem),
                         {stream_id: events for stream_id, events, _ in loaded},
                         "claude_code_local", account_names(cwds))


def resolve_session_files(projects_root: Path, session_hashes: Iterable[str]) -> dict[str, Path]:
    """Map ``sha256({"source":"claude_code","session_id":<stem>})`` to ``<slug>/<stem>.jsonl``."""
    wanted = set(session_hashes)
    found: dict[str, Path] = {}
    if not projects_root.is_dir():
        return found
    for path in sorted(projects_root.glob("*/*.jsonl")):
        digest = _session_hash(SOURCE, path.stem)
        if digest in wanted and digest not in found:
            found[digest] = path
    return found


# ── MCP render fallback ──────────────────────────────────────────────────────

_RENDER_HEAD = re.compile(r"^(-?\d+) (prompt|reply|tool_call|tool_result)(?: \| (.*))?$", re.S)


def events_from_mcp_session(session: dict[str, Any]) -> SessionEvents:
    """Convert a complete ``get_session`` export (spliced order) into events.

    Renders carry no ``cwd``, so account names come from home-directory paths
    that survived the server-side redaction.
    """
    streams: dict[str, list[Event]] = {}
    for item in session.get("events", []):
        if not isinstance(item, dict) or not isinstance(item.get("render"), str):
            continue
        match = _RENDER_HEAD.match(item["render"])
        if not match or item.get("kind") != match.group(2):
            continue
        fields: dict[str, str] = {}
        for segment in (match.group(3) or "").split(" | "):
            key, _, value = segment.partition(" ")
            fields.setdefault(key, value.replace(" \\| ", " | "))
        kind = match.group(2)
        ref = EventRef(str(item["stream_id"]), int(item["seq"]), int(item.get("subseq") or 0))
        text = fields.get("t") if kind in ("prompt", "reply") else fields.get("a" if kind == "tool_call" else "o")
        streams.setdefault(ref.stream_id, []).append(Event(
            ref, kind, text, tool_name=fields.get("n"), upstream_truncated=kind in ("tool_call", "tool_result")))
    paths = re.findall(r"(?:/Users|/home)/[^/\s]+", " ".join(e.text or "" for events in streams.values() for e in events))
    return SessionEvents(_session_hash(str(session["source"]), str(session["session_id"])), streams,
                         "transcript_mcp", account_names(paths))


def fetch_mcp_sessions(url: str, token: str, repo: str, wanted: set[str], limit: int = 200) -> dict[str, dict[str, Any]]:
    """Page matching sessions out of the transcript MCP. Callers gate this on a configured token."""
    from lilith_transcript_mcp_export import Client  # local import: network path only

    client = Client(url, token)
    client.initialize()
    listing = client.tool("list_sessions", {"repo": repo, "limit": limit, "budget_bytes": 256000})
    out: dict[str, dict[str, Any]] = {}
    for summary in listing.get("sessions", []):
        digest = _session_hash(str(summary["source"]), str(summary["session_id"]))
        if digest not in wanted:
            continue
        cursor, events = None, []
        while True:
            args = {"source": summary["source"], "session_id": summary["session_id"], "order": "spliced",
                    "include_thoughts": False, "include_all": False, "budget_bytes": 256000}
            if cursor:
                args["cursor"] = cursor
            page = client.tool("get_session", args)
            events.extend(page["events"])
            cursor = page.get("continue_from")
            if not cursor:
                break
        out[digest] = {"source": summary["source"], "session_id": summary["session_id"], "events": events}
    return out


# ── trajectory admission ─────────────────────────────────────────────────────


@dataclasses.dataclass
class _Draft:
    """A step under construction; ``full`` is the redacted, unbounded text (never serialized)."""

    step: dict[str, Any]
    full: str
    field: str | None  # "args" | "result_text" | None (assistant text is never trimmed)
    cap: int = 0


def _draft(event: Event, names: dict[str, str], accounts: frozenset[str], own_stream: str) -> _Draft:
    full = redact(event.text or "", accounts)
    own = event.ref.stream_id == own_stream
    step: dict[str, Any] = {"kind": "assistant_text" if event.kind == "reply" else event.kind,
                            "name": None, "args": None, "result_text": None, "text": None,
                            "event_ref": event.ref.as_json(),
                            # Her output: own-stream tool calls and assistant text. Tool results and
                            # delegated subagent work are context, never a training target.
                            "train": own and event.kind in ("tool_call", "reply"),
                            "full_sha256": sha(full.encode()), "full_bytes": len(full.encode()),
                            "truncated": False}
    if event.upstream_truncated:
        step["upstream_truncated"] = True
    if event.kind == "reply":
        step["text"] = full
        return _Draft(step, full, None)
    if event.kind == "tool_call":
        step["name"] = event.tool_name
        draft = _Draft(step, full, "args", ARGS_CAP_BYTES)
    else:
        step["name"] = names.get(event.tool_call_id or "", event.tool_name)
        draft = _Draft(step, full, "result_text", RESULT_CAP_BYTES)
    _apply_cap(draft, draft.cap)
    return draft


def _apply_cap(draft: _Draft, cap: int) -> int:
    """Re-bound ``draft`` to ``cap`` bytes from its full text; returns bytes removed."""
    before = len((draft.step[draft.field] or "").encode())
    text, truncated = bound(draft.full, cap)
    draft.cap, draft.step[draft.field], draft.step["truncated"] = cap, text, truncated
    return before - len(text.encode())


def _turn_events(window: list[Event], streams: dict[str, list[Event]], visited: set[str]) -> list[Event]:
    """Turn events with spawned subagent streams' tool steps spliced in after their spawn result."""
    out: list[Event] = []
    for event in window:
        if event.kind not in ("tool_call", "tool_result", "reply"):
            continue
        out.append(event)
        child = f"agent-{event.spawned_agent}" if event.spawned_agent else None
        if child and child in streams and child not in visited and event.kind == "tool_result":
            visited.add(child)
            out.extend(e for e in _turn_events(streams[child], streams, visited) if e.kind != "reply")
    return out


def _elide(events: list[Event], protected: set[int], own_stream: str) -> tuple[list[int], int]:
    """Indices kept under ``MAX_STEPS``: rated/final reply, own text, own tool steps nearest the turn ends, subagent steps last."""
    n = len(events)
    if n <= MAX_STEPS:
        return list(range(n)), 0

    def tier(i: int) -> int:
        event = events[i]
        if i in protected:
            return 0
        if event.ref.stream_id != own_stream:
            return 3
        return 1 if event.kind == "reply" else 2

    ranked = sorted(range(n), key=lambda i: (tier(i), min(i, n - 1 - i), i))
    return sorted(ranked[:MAX_STEPS]), n - MAX_STEPS


def _fit(record: dict[str, Any], drafts: list[_Draft]) -> int:
    """Trim the longest tool results (then args) until the record fits ``RECORD_BUDGET_BYTES``."""
    trimmed = 0

    def size() -> int:
        return len(json.dumps(record, sort_keys=True, ensure_ascii=False).encode())

    current = size()
    for field in ("result_text", "args"):
        while current > RECORD_BUDGET_BYTES:
            pool = [d for d in drafts if d.field == field and d.cap > MIN_TRIM_BYTES]
            if not pool:
                break
            largest = max(pool, key=lambda d: (len((d.step[field] or "").encode()), -drafts.index(d)))
            removed = _apply_cap(largest, max(MIN_TRIM_BYTES, len((largest.step[field] or "").encode()) // 2))
            trimmed += removed
            current -= removed
            if removed <= 0:
                current = size()
    record["trajectory"]["over_budget"] = size() > RECORD_BUDGET_BYTES
    return trimmed


def admit_trajectories(
    sessions: list[SessionEvents], *, salt: str,
    deny_sessions: frozenset[str] = frozenset(), eval_fingerprints: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Admit sessions into trajectory@2 rows: one row per whole assistant turn.

    Pairing (prompt -> first reply = ``rated_reply``), dedup and eval exclusion
    follow ``lilith_transcript_admission.admit``, so ``record_id`` is the v@1 id.
    The turn then runs on to the last assistant text before the next prompt in
    the same stream (``turn_final_reply``).
    """
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    for session in sessions:
        session_hash, streams, adapter, accounts = session.session_sha256, session.streams, session.adapter, session.accounts
        if session_hash in deny_sessions:
            rejected["eval_session"] += 1
            continue
        ordered = [event for events in streams.values() for event in events]
        names = {e.tool_call_id: e.tool_name for events in streams.values() for e in events
                 if e.kind == "tool_call" and e.tool_call_id and e.tool_name}
        pending: int | None = None
        for index, event in enumerate(ordered):
            if event.kind == "prompt":
                pending = index
                continue
            if event.kind != "reply" or pending is None:
                continue
            prompt_event = ordered[pending]
            prompt, rated = redact(prompt_event.text or "", accounts), redact(event.text or "", accounts)
            content_hash = sha({"input": normalized(prompt), "output": normalized(rated)})
            start = pending + 1
            pending = None
            if content_hash in seen:
                rejected["duplicate_content"] += 1
                continue
            if content_hash in eval_fingerprints:
                rejected["eval_content_collision"] += 1
                continue
            seen.add(content_hash)
            own = event.ref.stream_id
            end = index + 1
            while end < len(ordered) and ordered[end].kind != "prompt" and ordered[end].ref.stream_id == own:
                end += 1
            turn = _turn_events(ordered[start:end], streams, set())
            rated_at = turn.index(event)
            final_at = max(i for i, e in enumerate(turn) if e.kind == "reply" and e.ref.stream_id == own)
            keep, elided = _elide(turn, {rated_at, final_at}, own)
            drafts = [_draft(turn[i], names, accounts, own) for i in keep]
            steps = [d.step for d in drafts]
            final_event = turn[final_at]
            refs = [prompt_event.ref.as_json(), event.ref.as_json()]
            split_bucket = int(sha({"salt": salt, "session": session_hash})[:8], 16) % 10
            record = {
                "schema": SCHEMA,
                "record_id": sha({"session": session_hash, "event_refs": refs}),
                "split": "validation" if split_bucket == 0 else "train",
                "training_kind": "supervised_imitation", "input": prompt,
                "output": steps[keep.index(final_at)]["text"],
                "rated_reply": {"text": rated, "step_index": keep.index(rated_at), "event_ref": refs[1]},
                "turn_final_reply": {"text": steps[keep.index(final_at)]["text"], "step_index": keep.index(final_at),
                                     "event_ref": final_event.ref.as_json()},
                "turn_extended": final_at > rated_at,
                "trajectory": {"steps": steps, "steps_total": len(turn), "steps_elided": elided,
                               "bytes_trimmed": 0, "over_budget": False,
                               "subagent_streams": sorted({e.ref.stream_id for e in turn} - {own})},
                "provenance": {"source": SOURCE, "session_sha256": session_hash, "event_refs": refs,
                               "content_sha256": content_hash, "adapter": adapter,
                               "redaction_version": REDACTION_VERSION},
                "family_tags": [], "evidence": {"kind": "none"},
            }
            record["trajectory"]["bytes_trimmed"] = _fit(record, drafts)
            rows.append(record)
    rows.sort(key=lambda row: row["record_id"])
    return rows, dict(sorted(rejected.items()))


# ── CoS selection, cap and split ─────────────────────────────────────────────


def session_cap(counts: dict[str, int], share: float = SESSION_SHARE_CAP) -> int:
    """Largest per-session cap ``k`` with ``k <= share * sum(min(c, k))`` (0 if none)."""
    best = 0
    for k in range(1, max(counts.values(), default=0) + 1):
        if k <= share * sum(min(c, k) for c in counts.values()):
            best = k
    return best


def validation_sessions(
    counts: dict[str, int], salt: str, share: float = VALIDATION_SHARE, pinned: frozenset[str] = frozenset(),
) -> set[str]:
    """Whole sessions: ``pinned`` first, then salted-hash order, filled up to ``share`` of the records."""
    target = round(share * sum(counts.values()))
    chosen: set[str] = {s for s in pinned if s in counts}
    filled = sum(counts[s] for s in chosen)
    for session in sorted(counts, key=lambda s: sha({"salt": salt, "session": s})):
        if filled >= target:
            break
        if filled + counts[session] <= target or not chosen and counts[session] <= 2 * max(target, 1):
            chosen.add(session)
            filled += counts[session]
    return chosen


def _rank(record_id: str) -> str:
    return hashlib.sha256(record_id.encode()).hexdigest()


def select_cos_corpus(
    rows_by_id: dict[str, dict[str, Any]], panel: list[dict[str, Any]],
    record_sessions: dict[str, str], unresolved: dict[str, str], *, salt: str,
    pinned_validation: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join panel CoS decisions onto trajectory rows, cap per session, re-split, and describe it."""
    excluded: list[dict[str, str]] = []
    candidates: list[dict[str, Any]] = []
    cos = [p for p in panel if p.get("cos_grade") is True]
    for decision in sorted(cos, key=lambda p: p["record_id"]):
        record_id = decision["record_id"]
        session = record_sessions.get(record_id)
        if session is None:
            excluded.append({"record_id": record_id, "session": "?", "reason": "not_in_admission_set"})
        elif session in unresolved:
            excluded.append({"record_id": record_id, "session": session[:8], "reason": unresolved[session]})
        elif record_id not in rows_by_id:
            excluded.append({"record_id": record_id, "session": session[:8], "reason": "parse_error"})
        else:
            row = dict(rows_by_id[record_id])
            row["labels"] = {"cos_class": decision["cos_class"], "operator_correction": bool(decision.get("operator_correction"))}
            candidates.append(row)
    per_session = Counter(r["provenance"]["session_sha256"] for r in candidates)
    cap = session_cap(dict(per_session))
    kept: list[dict[str, Any]] = []
    for session in sorted(per_session):
        mine = sorted((r for r in candidates if r["provenance"]["session_sha256"] == session), key=lambda r: _rank(r["record_id"]))
        kept.extend(mine[:cap])
        excluded.extend({"record_id": r["record_id"], "session": session[:8], "reason": "session_cap"} for r in mine[cap:])
    kept_counts = Counter(r["provenance"]["session_sha256"] for r in kept)
    validation = validation_sessions(dict(kept_counts), salt, pinned=pinned_validation)
    for row in kept:
        row["split"] = "validation" if row["provenance"]["session_sha256"] in validation else "train"
    kept.sort(key=lambda r: r["record_id"])
    excluded.sort(key=lambda e: (e["reason"], e["session"], e["record_id"]))

    def step_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = [len(r["trajectory"]["steps"]) for r in rows]
        tools = [sum(s["kind"] != "assistant_text" for s in r["trajectory"]["steps"]) for r in rows]
        non_empty = [n for n in tools if n]
        return {"records": len(rows), "non_empty_tool_steps": len(non_empty),
                "turn_extended": sum(r["turn_extended"] for r in rows),
                "steps_median": statistics.median(counts) if counts else 0, "steps_max": max(counts, default=0),
                "tool_steps_median": statistics.median(tools) if tools else 0,
                "tool_steps_median_non_empty": statistics.median(non_empty) if non_empty else 0,
                "tool_steps_max": max(tools, default=0),
                "records_with_elided_steps": sum(r["trajectory"]["steps_elided"] > 0 for r in rows),
                "records_trimmed": sum(r["trajectory"]["bytes_trimmed"] > 0 for r in rows),
                "records_over_budget": sum(r["trajectory"]["over_budget"] for r in rows),
                "records_with_subagent_steps": sum(bool(r["trajectory"]["subagent_streams"]) for r in rows)}

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "record_schema": SCHEMA,
        "source_admission_schema": ADMISSION_SCHEMA,
        "redaction_version": REDACTION_VERSION,
        "redaction_patterns": [name for name, _, _ in REDACTIONS] + [ACCOUNT_RULE],
        "bounds": {"result_text_bytes": RESULT_CAP_BYTES, "args_bytes": ARGS_CAP_BYTES, "max_steps": MAX_STEPS,
                   "record_budget_bytes": RECORD_BUDGET_BYTES, "min_trim_bytes": MIN_TRIM_BYTES},
        "training_target": "steps with train=true (own-stream tool_call + assistant_text); "
                           "tool_result and subagent steps are context",
        "records": len(kept),
        "panel_cos_records": len(cos),
        "operator_corrections": sum(r["labels"]["operator_correction"] for r in kept),
        "by_class": dict(sorted(Counter(r["labels"]["cos_class"] for r in kept).items())),
        "by_session": {s[:8]: {"records": kept_counts[s], "candidates": per_session[s],
                               "split": "validation" if s in validation else "train"} for s in sorted(per_session)},
        "by_split": {split: sum(r["split"] == split for r in kept) for split in ("train", "validation")},
        "by_class_split": {f"{c}/{s}": n for (c, s), n in sorted(Counter((r["labels"]["cos_class"], r["split"]) for r in kept).items())},
        "trajectory_stats": {"candidates": step_stats(candidates), "admitted": step_stats(kept)},
        "cap_policy": {"max_session_share": SESSION_SHARE_CAP, "per_session_cap": cap,
                       "order": "sha256(record_id) ascending within session"},
        "split_policy": "session-bound; pinned sessions first, then sha256(salt,session) order fills "
                        f"validation up to {VALIDATION_SHARE:.0%} of records",
        "validation": {"sessions": sorted(s[:8] for s in validation),
                       "pinned": sorted(s[:8] for s in pinned_validation),
                       "share": round(sum(r["split"] == "validation" for r in kept) / len(kept), 4) if kept else 0.0,
                       "target_share": VALIDATION_SHARE},
        "excluded_counts": dict(sorted(Counter(e["reason"] for e in excluded).items())),
        "excluded": excluded,
    }
    return kept, manifest


# ── CLI ──────────────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


def resolve_session_prefixes(prefixes: list[str], sessions: set[str]) -> frozenset[str]:
    """Expand ``--validation-sessions`` hash prefixes; each must name exactly one session."""
    out = set()
    for prefix in prefixes:
        matches = [s for s in sessions if s.startswith(prefix.lower())]
        if len(prefix) < 8 or len(matches) != 1:
            raise SystemExit(f"--validation-sessions {prefix!r}: needs >=8 hex chars matching exactly one "
                             f"session (matched {len(matches)})")
        out.add(matches[0])
    return frozenset(out)


def cos_export(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite a C5 corpus artifact")
    panel = _read_jsonl(args.panel)
    admitted = _read_jsonl(args.admitted)
    record_sessions = {r["record_id"]: r["provenance"]["session_sha256"] for r in admitted}
    cos_ids = {p["record_id"] for p in panel if p.get("cos_grade") is True}
    wanted = {record_sessions[i] for i in cos_ids if i in record_sessions}
    local = resolve_session_files(args.projects_root, wanted)
    sessions: list[SessionEvents] = []
    unresolved: dict[str, str] = {}
    for session in sorted(wanted):
        if session not in local:
            continue
        try:
            sessions.append(load_local_session(local[session]))
        except SessionParseError:
            unresolved[session] = "parse_error"
    missing = wanted - set(local)
    token = os.environ.get(args.mcp_token_env, "").strip() if args.mcp_token_env else ""
    if missing and args.mcp_url and token:
        for session, exported in fetch_mcp_sessions(args.mcp_url, token, args.mcp_repo, missing).items():
            sessions.append(events_from_mcp_session(exported))
    covered = {s.session_sha256 for s in sessions}
    for session in missing - covered:
        unresolved[session] = "missing_session_file"
    rows, rejected = admit_trajectories(sessions, salt=args.salt)
    pinned = resolve_session_prefixes(args.validation_sessions, wanted)
    kept, manifest = select_cos_corpus({r["record_id"]: r for r in rows}, panel, record_sessions, unresolved,
                                       salt=args.salt, pinned_validation=pinned)
    body = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in kept)
    manifest["dataset_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    manifest["inputs"] = {"panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                          "admitted_sha256": hashlib.sha256(args.admitted.read_bytes()).hexdigest()}
    manifest["session_adapters"] = {s.session_sha256[:8]: s.adapter for s in sorted(sessions, key=lambda s: s.session_sha256)}
    manifest["admission_rejected"] = rejected
    manifest["mcp_fallback"] = ("used" if any(s.adapter == "transcript_mcp" for s in sessions)
                                else "not_needed" if not missing else "not_configured")
    _write_private(args.output, body)
    _write_private(args.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cos = sub.add_parser("cos-export", help="C5 CoS corpus from C5a panel decisions")
    cos.add_argument("--panel", type=Path, required=True, help="C5a final.jsonl (private)")
    cos.add_argument("--admitted", type=Path, required=True, help="admission@1 jsonl the panel rated (private)")
    cos.add_argument("--projects-root", type=Path, default=Path.home() / ".claude" / "projects")
    cos.add_argument("--output", type=Path, required=True)
    cos.add_argument("--manifest", type=Path, required=True)
    cos.add_argument("--salt", required=True)
    cos.add_argument("--validation-sessions", default=[], type=lambda v: [p for p in v.split(",") if p],
                     help="comma-separated session_sha256 prefixes pinned to validation (default: none)")
    cos.add_argument("--mcp-url", default="http://transcripts.lab.home/mcp",
                     help="transcript MCP endpoint, used only for sessions not on disk and only with a token")
    cos.add_argument("--mcp-token-env", default="AKH_TRANSCRIPTS_MCP_TOKEN",
                     help="env var holding the MCP bearer token; unset means no MCP call")
    cos.add_argument("--mcp-repo", default="akh-medu")
    args = parser.parse_args(argv)
    manifest = cos_export(args)
    summary = {key: manifest[key] for key in ("records", "panel_cos_records", "by_split", "validation", "by_session",
                                              "excluded_counts", "trajectory_stats", "mcp_fallback", "dataset_sha256")}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
