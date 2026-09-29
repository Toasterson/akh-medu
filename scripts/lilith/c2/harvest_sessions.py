#!/usr/bin/env python3
"""Harvest Lilith's operator sessions from the running core into NFS (C2, ADR 116).

    harvest_sessions.py [--out DIR] [--since ISO] [--until ISO] [--session-gap-minutes N]
                        [--endpoint HOST:PORT] [--token-file F] [--snapshot-root DIR] [--workspace lilith]

Reads captured turns through the core's read-only ACP `_akh/transcripts/list`
(never a store file: opening a live SQLite store corrupts it). Writes under
`--out` (default `/Volumes/akh-medu-runs/lilith/c2`):

    sessions/<session-id>.json     turns, tool calls with receipts, timings, model provenance
    corrections.jsonl              one `CorrectionRecord` per `✎` message (see c2/marker.py)
    corrections-unmatched.jsonl    corrections with no recorder snapshot at or before the turn
    autonomous.jsonl               turns the akh ran on its own (dispatch), outside any session

Idempotent (the full transcript is re-read each run; unchanged files are not
rewritten), never deletes, and never shrinks a session file it already holds.
Exit codes: 0 ok, 1 failed, 2 usage, 3 NFS share not mounted, 4 core unreachable
or too old.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c2 import C2_VERSION, SESSION_SCHEMA
from c2 import acp_client
from c2.errors import EXIT_FAILED, EXIT_OK, EXIT_USAGE, C2Error, NfsUnmountedError
from c2.marker import CORRECTION_MARKER, parse_correction
from dojo.snapshot import TIMESTAMP_FORMAT, canonical_json, parse_iso, parse_snapshot_time
from dojo.errors import DojoError

DEFAULT_OUT = "/Volumes/akh-medu-runs/lilith/c2"
DEFAULT_SNAPSHOT_ROOT = "~/Library/Application Support/akh-medu/lilith/cos-snapshots"
DEFAULT_TOKEN_FILE = "~/Library/Application Support/roman-akhs/secrets/operator.key"
DEFAULT_SESSION_GAP = timedelta(minutes=45)

# Tracker identifiers in prose or receipts: a full UUID, or its first two groups
# (the short form Lilith and the operator both write). Structural, not lexical.
ID_PATTERN = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}(?:-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})?\b")


def extract_ids(*texts: Any) -> list[str]:
    """Distinct tracker-shaped ids in `texts`, longest form kept, sorted."""
    found: set[str] = set()
    for text in texts:
        if isinstance(text, str):
            found.update(ID_PATTERN.findall(text))
    return sorted(i for i in found if not any(o != i and o.startswith(i) for o in found))


# --- normalisation --------------------------------------------------------------------------------

def _tier(tool: str) -> str:
    try:
        from dojo.contracts import PermissionProfile
        return PermissionProfile.load().tier(tool).value
    except Exception:  # a missing profile must not stop a harvest; the notes say "unlisted"
        return "unlisted"


def _tool_call(entry: Mapping[str, Any]) -> dict[str, Any]:
    tool = str(entry.get("tool", ""))
    return {
        "tool": tool,
        "ok": bool(entry.get("ok")),
        "wrote": bool(entry.get("wrote")),
        "tier": _tier(tool),
        "args": entry.get("args"),
        "output": entry.get("output"),
        "duration_ms": entry.get("duration_ms"),
        "receipt_captured": entry.get("output") is not None,
    }


def normalise_turn(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = row.get("payload") or {}
    when = parse_iso(payload.get("ts")) or parse_iso(row.get("applied_at"))
    calls = [_tool_call(t) for t in payload.get("tools") or [] if isinstance(t, Mapping)]
    answer = payload.get("agent_answer") or ""
    return {
        "seq": row.get("seq"),
        "ts": when.isoformat() if when else None,
        "applied_at": row.get("applied_at"),
        "model": payload.get("model"),
        "user_prompt": payload.get("user_prompt") or "",
        "agent_answer": answer,
        "dispatch": bool(payload.get("dispatch")),
        "tool_calls": calls,
        "timings": {
            "tool_ms": sum(c["duration_ms"] for c in calls if isinstance(c["duration_ms"], int)),
            "answer_chars": len(answer),
        },
    }


def _when(turn: Mapping[str, Any]) -> datetime:
    return parse_iso(turn["ts"]) or datetime.fromtimestamp(0, timezone.utc)


# --- sessions ---------------------------------------------------------------------------------------

def sessionize(turns: Sequence[dict[str, Any]], gap: timedelta = DEFAULT_SESSION_GAP) -> tuple[list[list[dict]], list[dict]]:
    """Split normalised turns (seq order) into operator sessions and autonomous turns.

    A session is a run of operator turns with no idle gap longer than `gap`. Turns
    the akh ran on its own (dispatch) never join a session and never break one.
    """
    sessions: list[list[dict]] = []
    autonomous: list[dict] = []
    for turn in turns:
        if turn["dispatch"]:
            autonomous.append(turn)
            continue
        if sessions and _when(turn) - _when(sessions[-1][-1]) <= gap:
            sessions[-1].append(turn)
        else:
            sessions.append([turn])
    return sessions, autonomous


def session_id(first: Mapping[str, Any]) -> str:
    stamp = _when(first).strftime(TIMESTAMP_FORMAT)
    return f"{stamp}-s{first['seq']}"


def build_session(turns: Sequence[dict], workspace: str, endpoint: str) -> dict[str, Any]:
    sid = session_id(turns[0])
    rendered: list[dict] = []
    previous: dict | None = None
    for index, turn in enumerate(turns):
        correction = parse_correction(turn["user_prompt"])
        gap_ms = int((_when(turn) - _when(previous)).total_seconds() * 1000) if previous else None
        rendered.append({
            "index": index,
            **turn,
            "timings": {**turn["timings"], "since_previous_turn_ms": gap_ms},
            "correction": ({"expected_tool": correction.expected_tool, "text": correction.text,
                            "corrects_turn": index - 1 if index > 0 else None} if correction else None),
        })
        previous = turn
    models = sorted({t["model"] for t in turns if t["model"]})
    return {
        "schema": SESSION_SCHEMA,
        "session_id": sid,
        "workspace": workspace,
        "started_at": turns[0]["ts"],
        "ended_at": turns[-1]["ts"],
        "turn_count": len(turns),
        "provenance": {
            "harvester": C2_VERSION,
            "source": {"endpoint": endpoint, "method": acp_client.LIST_METHOD,
                       "first_seq": turns[0]["seq"], "last_seq": turns[-1]["seq"]},
            "models": models,
            "receipts_captured": all(c["receipt_captured"] for t in turns for c in t["tool_calls"]),
            "timing_note": "tok/s is not captured per turn by the akh; the supervisor records it from the serving pod",
        },
        "turns": rendered,
    }


# --- corrections ------------------------------------------------------------------------------------

class SnapshotIndex:
    """Recorder snapshot ids (UTC timestamp directory names) under one root."""

    def __init__(self, root: Path | None):
        self.times: list[tuple[datetime, str]] = []
        if root is not None and root.is_dir():
            for child in root.iterdir():
                if child.is_dir() and (child / "manifest.json").is_file():
                    try:
                        self.times.append((parse_snapshot_time(child.name), child.name))
                    except DojoError:
                        continue
        self.times.sort()

    def at_or_before(self, when: datetime) -> str | None:
        best = None
        for stamp, name in self.times:
            if stamp <= when:
                best = name
            else:
                break
        return best


def tool_sequence_signature(calls: Iterable[Mapping[str, Any]]) -> str:
    """Structural identity of a turn: the ordered tool names it called (sha256 of canonical JSON)."""
    body = {"tool_sequence": [c["tool"] for c in calls]}
    return hashlib.sha256(canonical_json(body)).hexdigest()


def correction_records(session: Mapping[str, Any], snapshots: SnapshotIndex) -> tuple[list[dict], list[dict]]:
    """(matched, unmatched) `CorrectionRecord` dicts for one session's `✎` messages."""
    matched: list[dict] = []
    unmatched: list[dict] = []
    turns = session["turns"]
    for turn in turns:
        marker = turn["correction"]
        if marker is None or marker["corrects_turn"] is None:
            continue
        corrected = turns[marker["corrects_turn"]]
        receipts = [c for c in corrected["tool_calls"]]
        snapshot_id = snapshots.at_or_before(_when(corrected))
        record = {
            "episode_ref": f"{session['session_id']}#{corrected['index']}",
            "snapshot_id": snapshot_id,
            "corrected_signature": tool_sequence_signature(receipts),
            "expected_tool": marker["expected_tool"],
            "subject_ids": extract_ids(*(x for c in receipts for x in (c["args"], c["output"]))),
            "session_id": session["session_id"],
            "corrected_turn_seq": corrected["seq"],
            "correction_turn_seq": turn["seq"],
        }
        (matched if snapshot_id else unmatched).append(record)
    return matched, unmatched


# --- output (idempotent, never deletes) ----------------------------------------------------------------

def is_mount(path: Path) -> bool:
    return os.path.ismount(path)


def ensure_output_root(out: Path, ismount: Callable[[Path], bool] = is_mount) -> None:
    """A share path under /Volumes/<share> must be a live mount; otherwise fail before writing anything."""
    parts = out.parts
    if len(parts) >= 3 and parts[1] == "Volumes" and not ismount(Path(*parts[:3])):
        raise NfsUnmountedError(f"{Path(*parts[:3])} is not mounted, so {out} would land on the local disk")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise C2Error(f"cannot create {out}: {error.strerror}", help="check the mount is writable by this user") from None


def write_if_changed(path: Path, text: str) -> bool:
    """Atomic write; False when the file already holds exactly `text`."""
    if path.exists() and path.read_text() == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return True


def _dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def merge_jsonl(path: Path, records: Iterable[Mapping[str, Any]], key: str = "episode_ref") -> tuple[int, bool]:
    """Union by `key` with what the file already holds (a re-harvest replaces a record it
    regenerates, keeps every other one). Returns (total, changed)."""
    existing: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row[key]] = row
    for record in records:
        existing[record[key]] = dict(record)
    text = "".join(json.dumps(existing[k], sort_keys=True, ensure_ascii=False) + "\n" for k in sorted(existing))
    return len(existing), write_if_changed(path, text) if existing else False


def write_session(directory: Path, session: Mapping[str, Any]) -> str:
    """'written' | 'unchanged' | 'kept' (an existing file holds more turns than this harvest)."""
    path = directory / f"{session['session_id']}.json"
    if path.exists():
        held = json.loads(path.read_text())
        if held.get("turn_count", 0) > session["turn_count"]:
            return "kept"
    return "written" if write_if_changed(path, _dump(session)) else "unchanged"


# --- the run --------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class HarvestConfig:
    out: Path
    endpoint: str
    token: str
    workspace: str = "lilith"
    since: datetime | None = None
    until: datetime | None = None
    gap: timedelta = DEFAULT_SESSION_GAP
    snapshot_root: Path | None = None


def harvest(config: HarvestConfig, fetch: Callable[..., Iterable[dict]] = acp_client.list_turns,
            ismount: Callable[[Path], bool] = is_mount) -> dict[str, Any]:
    ensure_output_root(config.out, ismount)  # before any network or write
    rows = list(fetch(config.endpoint, config.token, config.workspace))
    turns = [normalise_turn(r) for r in rows]
    turns = [t for t in turns
             if (config.since is None or _when(t) >= config.since) and (config.until is None or _when(t) <= config.until)]
    sessions, autonomous = sessionize(turns, config.gap)
    snapshots = SnapshotIndex(config.snapshot_root)
    statuses: dict[str, str] = {}
    matched: list[dict] = []
    unmatched: list[dict] = []
    for group in sessions:
        session = build_session(group, config.workspace, config.endpoint)
        statuses[session["session_id"]] = write_session(config.out / "sessions", session)
        m, u = correction_records(session, snapshots)
        matched.extend(m)
        unmatched.extend(u)
    total_corrections, _ = merge_jsonl(config.out / "corrections.jsonl", matched)
    total_unmatched, _ = merge_jsonl(config.out / "corrections-unmatched.jsonl", unmatched)
    if autonomous:
        merge_jsonl(config.out / "autonomous.jsonl", autonomous, key="seq")
    return {
        "out": str(config.out),
        "turns": len(turns),
        "sessions": len(sessions),
        "session_files": statuses,
        "autonomous_turns": len(autonomous),
        "corrections": total_corrections,
        "corrections_unmatched": total_unmatched,
        "snapshots_indexed": len(snapshots.times),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harvest Lilith's operator sessions into NFS (C2, ADR 116)")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("LILITH_C2_OUT", DEFAULT_OUT)))
    parser.add_argument("--endpoint", default=os.environ.get("LILITH_ACP_ENDPOINT", acp_client.DEFAULT_ENDPOINT))
    parser.add_argument("--token-file", type=Path, default=Path(DEFAULT_TOKEN_FILE).expanduser())
    parser.add_argument("--workspace", default="lilith")
    parser.add_argument("--since", default=None, help="ISO time; earlier turns (dev chats before C2) are ignored")
    parser.add_argument("--until", default=None)
    parser.add_argument("--session-gap-minutes", type=float, default=DEFAULT_SESSION_GAP.total_seconds() / 60)
    parser.add_argument("--snapshot-root", type=Path,
                        default=Path(os.environ.get("LILITH_SNAPSHOT_ROOT", DEFAULT_SNAPSHOT_ROOT)).expanduser())
    args = parser.parse_args(argv)
    since, until = parse_iso(args.since), parse_iso(args.until)
    if (args.since and since is None) or (args.until and until is None):
        print("[lilith::c2::usage] --since/--until must be ISO 8601, e.g. 2026-09-30T08:00:00Z", file=sys.stderr)
        return EXIT_USAGE
    try:
        # The mount check comes first so an unmounted share never costs a token read or a connection.
        ensure_output_root(args.out)
        config = HarvestConfig(args.out, args.endpoint, acp_client.read_token(args.token_file), args.workspace, since,
                               until, timedelta(minutes=args.session_gap_minutes), args.snapshot_root)
        report = harvest(config)
    except (C2Error, DojoError) as error:
        print(error.render(), file=sys.stderr)
        return getattr(error, "exit_code", EXIT_FAILED)
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
