#!/usr/bin/env python3
"""Lilith C5: transcript admission with tool trajectories, and the CoS corpus export.

Schema ``akh/lilith-transcript-trajectory@1`` extends
``akh/lilith-transcript-admission@1`` (``lilith_transcript_admission.py``, left
unchanged): every prompt->reply record also carries the ordered ``tool_call`` /
``tool_result`` steps between the prompt and the reply, so a reply's status
claims keep their receipts (C5a finding 4, Anima WI 01a0e270-7792,
``docs/ai/research/2026-09-27-lilith-c5a-transcript-role-fit.md``).

Pairing is the v@1 rule applied to the transcript MCP's ``get_session``
``order="spliced"`` event order, so a trajectory record's ``record_id`` is
exactly the v@1 ``record_id`` of the same turn and the C5a panel decisions join
on it. Steps are then collected in *deep* spliced order: a subagent stream is
inserted right after the ``tool_result`` that spawned it
(``toolUseResult.agentId``), recursively.

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

SCHEMA = "akh/lilith-transcript-trajectory@1"
MANIFEST_SCHEMA = "lilith/cos-corpus-manifest@1"
REDACTION_VERSION = "lilith-redact@2"
SOURCE = "claude_code"
RESULT_CAP_BYTES = 8192
ARGS_CAP_BYTES = 8192
MAX_STEPS = 256
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


def _step(event: Event, names: dict[str, str], accounts: frozenset[str]) -> dict[str, Any]:
    full = redact(event.text or "", accounts)
    if event.kind == "tool_call":
        args, truncated = bound(full, ARGS_CAP_BYTES)
        result_text = None
        name = event.tool_name
    else:
        args = None
        result_text, truncated = bound(full, RESULT_CAP_BYTES)
        name = names.get(event.tool_call_id or "", event.tool_name)
    step: dict[str, Any] = {
        "kind": event.kind, "name": name, "args": args, "result_text": result_text,
        "event_ref": event.ref.as_json(), "full_sha256": sha(full.encode()),
        "full_bytes": len(full.encode()), "truncated": truncated,
    }
    if event.upstream_truncated:
        step["upstream_truncated"] = True
    return step


def _deep_steps(window: list[Event], streams: dict[str, list[Event]], visited: set[str]) -> list[Event]:
    """Tool events of ``window`` with spawned subagent streams spliced in after their spawn result."""
    out: list[Event] = []
    for event in window:
        if event.kind not in ("tool_call", "tool_result"):
            continue
        out.append(event)
        child = f"agent-{event.spawned_agent}" if event.spawned_agent else None
        if child and child in streams and child not in visited and event.kind == "tool_result":
            visited.add(child)
            out.extend(_deep_steps(streams[child], streams, visited))
    return out


def admit_trajectories(
    sessions: list[SessionEvents], *, salt: str,
    deny_sessions: frozenset[str] = frozenset(), eval_fingerprints: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Admit sessions into trajectory@1 rows.

    Pairing, dedup and eval exclusion follow ``lilith_transcript_admission.admit``.
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
            prompt, reply = redact(prompt_event.text or "", accounts), redact(event.text or "", accounts)
            content_hash = sha({"input": normalized(prompt), "output": normalized(reply)})
            window = ordered[pending + 1:index]
            pending = None
            if content_hash in seen:
                rejected["duplicate_content"] += 1
                continue
            if content_hash in eval_fingerprints:
                rejected["eval_content_collision"] += 1
                continue
            seen.add(content_hash)
            refs = [prompt_event.ref.as_json(), event.ref.as_json()]
            spliced = _deep_steps(window, streams, set())
            steps = [_step(step, names, accounts) for step in spliced[:MAX_STEPS]]
            split_bucket = int(sha({"salt": salt, "session": session_hash})[:8], 16) % 10
            rows.append({
                "schema": SCHEMA,
                "record_id": sha({"session": session_hash, "event_refs": refs}),
                "split": "validation" if split_bucket == 0 else "train",
                "training_kind": "supervised_imitation", "input": prompt, "output": reply,
                "trajectory": {"steps": steps, "steps_elided": max(0, len(spliced) - MAX_STEPS),
                               "subagent_streams": sorted({s.ref.stream_id for s in spliced} - {event.ref.stream_id})},
                "provenance": {"source": SOURCE, "session_sha256": session_hash, "event_refs": refs,
                               "content_sha256": content_hash, "adapter": adapter,
                               "redaction_version": REDACTION_VERSION},
                "family_tags": [], "evidence": {"kind": "none"},
            })
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


def validation_sessions(counts: dict[str, int], salt: str, share: float = VALIDATION_SHARE) -> set[str]:
    """Whole sessions, in salted-hash order, filled up to ``share`` of the records."""
    target = round(share * sum(counts.values()))
    chosen: set[str] = set()
    filled = 0
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
    validation = validation_sessions(dict(kept_counts), salt)
    for row in kept:
        row["split"] = "validation" if row["provenance"]["session_sha256"] in validation else "train"
    kept.sort(key=lambda r: r["record_id"])
    excluded.sort(key=lambda e: (e["reason"], e["session"], e["record_id"]))

    def step_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = [len(r["trajectory"]["steps"]) for r in rows]
        calls = [sum(s["kind"] == "tool_call" for s in r["trajectory"]["steps"]) for r in rows]
        return {"records": len(rows), "non_empty": sum(c > 0 for c in counts),
                "steps_median": statistics.median(counts) if counts else 0, "steps_max": max(counts, default=0),
                "tool_calls_median": statistics.median(calls) if calls else 0, "tool_calls_max": max(calls, default=0),
                "records_with_elided_steps": sum(r["trajectory"]["steps_elided"] > 0 for r in rows),
                "records_with_subagent_steps": sum(bool(r["trajectory"]["subagent_streams"]) for r in rows)}

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "record_schema": SCHEMA,
        "source_admission_schema": ADMISSION_SCHEMA,
        "redaction_version": REDACTION_VERSION,
        "redaction_patterns": [name for name, _, _ in REDACTIONS] + [ACCOUNT_RULE],
        "bounds": {"result_text_bytes": RESULT_CAP_BYTES, "args_bytes": ARGS_CAP_BYTES, "max_steps": MAX_STEPS},
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
        "split_policy": f"session-bound; sessions in sha256(salt,session) order fill validation up to {VALIDATION_SHARE:.0%} of records",
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
    kept, manifest = select_cos_corpus({r["record_id"]: r for r in rows}, panel, record_sessions, unresolved, salt=args.salt)
    body = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in kept)
    manifest["dataset_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    manifest["inputs"] = {"panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
                          "admitted_sha256": hashlib.sha256(args.admitted.read_bytes()).hexdigest()}
    manifest["session_adapters"] = {s.session_sha256[:8]: s.adapter for s in sorted(sessions, key=lambda s: s.session_sha256)}
    manifest["admission_rejected"] = rejected
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
    cos.add_argument("--mcp-url", help="transcript MCP endpoint, used only for sessions not on disk")
    cos.add_argument("--mcp-token-env", default="AKH_TRANSCRIPTS_MCP_TOKEN",
                     help="env var holding the MCP bearer token; unset means no MCP call")
    cos.add_argument("--mcp-repo", default="akh-medu")
    args = parser.parse_args(argv)
    manifest = cos_export(args)
    summary = {key: manifest[key] for key in ("records", "panel_cos_records", "by_split", "excluded_counts", "trajectory_stats", "dataset_sha256")}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
