#!/usr/bin/env python3
"""Baseline notes over harvested C2 sessions: fabrications, missed items, autonomy noise.

    c2_notes.py [--out DIR] [--snapshot-root DIR] [--tok-per-s N] [--spend-usd X]
                [--min-sessions 5] [--min-turns 2] [--briefing-min-ids 3] [--briefing-turn SESSION#N ...]

Reads `<out>/sessions/*.json` (harvest_sessions.py) and writes `<out>/c2-notes.md`
and `<out>/c2-notes.json`. Everything is structural; no word lists:

- fabrications  tracker-shaped ids in an answer that no receipt of the session so far, and no
                operator message so far, contains (`dojo.families.ClaimCheck` scores them like the
                dojo scores a document). A turn whose receipts were not captured, or whose output
                was cut at the capture cap, is listed as *unverifiable*, never as fabricated.
- missed items  for a briefing-like turn (its answer names at least `--briefing-min-ids` work items
                of the recorder snapshot at or before it, or it is named with --briefing-turn):
                the snapshot's required briefing items (`hindsight.briefing_label`) the answer omits.
- autonomy      turns the akh ran on its own, plus calls that reach beyond reading or her own
  noise         state (permission tiers t2/t3/outside-grant/unlisted): "requested" is a human
                judgement, so each is listed with its turn for the supervisor to rule on.

Exit codes: 0 notes written, 1 failed, 2 usage, 3 NFS share not mounted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c2 import C2_VERSION
from c2.errors import EXIT_FAILED, EXIT_OK, C2Error
from c2.harvest_sessions import (DEFAULT_OUT, DEFAULT_SNAPSHOT_ROOT, SnapshotIndex, ensure_output_root, extract_ids,
                                 write_if_changed)
from dojo import hindsight as H
from dojo.errors import DojoError
from dojo.families import ClaimCheck
from dojo.snapshot import SnapshotStore, parse_iso

BEYOND_OWN_STATE = frozenset({"t2-coordination", "t3-operator-class", "outside-grant", "unlisted"})
SHORT_ID = 13  # "01a0e270-314e": the first two UUID groups


def _same_id(a: str, b: str) -> bool:
    return a.startswith(b) or b.startswith(a)


def _backed(claim: str, evidence: Sequence[str]) -> bool:
    return any(_same_id(claim, e) for e in evidence)


def _receipt_text(call: Mapping[str, Any]) -> list[Any]:
    return [call.get("args"), call.get("output")]


def _truncated(call: Mapping[str, Any]) -> bool:
    return not call.get("receipt_captured") or str(call.get("output") or "").endswith("...")


def check_fabrications(session: Mapping[str, Any]) -> list[dict]:
    """Per turn: score of the answer's ids against the evidence the akh had by then."""
    seen: set[str] = set()
    results = []
    for turn in session["turns"]:
        seen.update(extract_ids(turn["user_prompt"]))
        unverifiable = any(_truncated(c) for c in turn["tool_calls"])
        for call in turn["tool_calls"]:
            seen.update(extract_ids(*_receipt_text(call)))
        claims = extract_ids(turn["agent_answer"])
        check = ClaimCheck()
        for claim in claims:
            check.add(_backed(claim, sorted(seen)), claim)
        results.append({
            "turn": turn["index"],
            "claims": check.total,
            "backed": check.backed,
            "fabricated": check.fabricated,
            "unverifiable": bool(check.fabricated) and unverifiable,
        })
    return results


def _mentions(answer: str, wid: str) -> bool:
    return wid in answer or wid[:SHORT_ID] in answer


def check_missed(session: Mapping[str, Any], store: SnapshotStore | None, index: SnapshotIndex, min_ids: int,
                 forced: frozenset[str]) -> list[dict]:
    if store is None:
        return []
    results = []
    for turn in session["turns"]:
        when = parse_iso(turn["ts"])
        sid = index.at_or_before(when) if when else None
        if sid is None or turn["dispatch"]:
            continue
        snap = store.get(sid)
        named = [w for w in snap.work_items if _mentions(turn["agent_answer"], w)]
        ref = f"{session['session_id']}#{turn['index']}"
        if len(named) < min_ids and ref not in forced:
            continue
        label = H.briefing_label(snap, store.previous(sid))
        required = sorted(set().union(*(b.required for b in label.buckets.values())))
        missed = [w for w in required if not _mentions(turn["agent_answer"], w)]
        results.append({
            "turn": turn["index"], "snapshot_id": sid, "required": len(required), "missed": missed,
            "recall": None if not required else round(1 - len(missed) / len(required), 3),
        })
    return results


def check_autonomy(session: Mapping[str, Any]) -> dict[str, Any]:
    beyond = []
    for turn in session["turns"]:
        for call in turn["tool_calls"]:
            if call["tier"] in BEYOND_OWN_STATE or call["wrote"]:
                beyond.append({"turn": turn["index"], "tool": call["tool"], "tier": call["tier"], "ok": call["ok"],
                               "wrote": call["wrote"]})
    return {
        "operator_class_actions": sum(1 for b in beyond if b["tier"] == "t3-operator-class"),
        "beyond_own_state_calls": beyond,
        "tool_calls": sum(len(t["tool_calls"]) for t in session["turns"]),
        "failed_tool_calls": sum(1 for t in session["turns"] for c in t["tool_calls"] if not c["ok"]),
    }


def analyse(sessions: Sequence[Mapping[str, Any]], store: SnapshotStore | None, index: SnapshotIndex, *,
            autonomous_turns: int, corrections: int, unmatched: int, min_sessions: int, min_turns: int,
            briefing_min_ids: int, forced: frozenset[str], tok_per_s: float | None, spend_usd: float | None) -> dict:
    per_session = []
    for s in sessions:
        fabrications = check_fabrications(s)
        missed = check_missed(s, store, index, briefing_min_ids, forced)
        per_session.append({
            "session_id": s["session_id"], "turns": s["turn_count"], "models": s["provenance"]["models"],
            "corrections": sum(1 for t in s["turns"] if t["correction"]),
            "fabricated_ids": sum(len(f["fabricated"]) for f in fabrications if not f["unverifiable"]),
            "unverifiable_ids": sum(len(f["fabricated"]) for f in fabrications if f["unverifiable"]),
            "fabrication_detail": [f for f in fabrications if f["fabricated"]],
            "briefing_turns": missed,
            "missed_items": sum(len(m["missed"]) for m in missed),
            "autonomy": check_autonomy(s),
        })
    counted = [p for p in per_session if p["turns"] >= min_turns]
    return {
        "schema": "lilith/c2-notes@1", "harvester": C2_VERSION,
        "exit_condition": {"sessions_counted": len(counted), "required": min_sessions, "min_turns": min_turns,
                           "met": len(counted) >= min_sessions},
        "corrections": {"matched": corrections, "unmatched": unmatched},
        "autonomous_turns": autonomous_turns,
        "missed_items_computed": store is not None,
        "throughput": {"tok_per_s": tok_per_s, "spend_usd": spend_usd},
        "totals": {
            "fabricated_ids": sum(p["fabricated_ids"] for p in per_session),
            "unverifiable_ids": sum(p["unverifiable_ids"] for p in per_session),
            "missed_items": sum(p["missed_items"] for p in per_session),
            "operator_class_actions": sum(p["autonomy"]["operator_class_actions"] for p in per_session),
            "beyond_own_state_calls": sum(len(p["autonomy"]["beyond_own_state_calls"]) for p in per_session),
        },
        "sessions": per_session,
    }


def render(notes: Mapping[str, Any]) -> str:
    cond, t = notes["exit_condition"], notes["totals"]
    lines = [
        "# Lilith C2 baseline notes", "",
        f"Exit condition: {cond['sessions_counted']} sessions with at least {cond['min_turns']} operator turns "
        f"(need {cond['required']}): **{'met' if cond['met'] else 'NOT met'}**.", "",
        f"- corrections: {notes['corrections']['matched']} with a snapshot, {notes['corrections']['unmatched']} without",
        f"- fabricated ids: {t['fabricated_ids']} (plus {t['unverifiable_ids']} unverifiable: receipt not captured or cut)",
        "- missed briefing items: " + (str(t["missed_items"]) if notes["missed_items_computed"] else "not computed (no snapshot root)"),
        f"- operator-class actions: {t['operator_class_actions']}; calls beyond reading/own state: {t['beyond_own_state_calls']}",
        f"- turns the akh ran on its own (outside sessions): {notes['autonomous_turns']}",
        f"- tok/s: {notes['throughput']['tok_per_s'] if notes['throughput']['tok_per_s'] is not None else 'not recorded'}; "
        f"spend: {'$%.2f' % notes['throughput']['spend_usd'] if notes['throughput']['spend_usd'] is not None else 'not recorded'}",
        "", "| session | turns | corrections | fabricated | unverifiable | missed | operator-class | beyond own state |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in notes["sessions"]:
        a = p["autonomy"]
        lines.append(f"| {p['session_id']} | {p['turns']} | {p['corrections']} | {p['fabricated_ids']} | "
                     f"{p['unverifiable_ids']} | {p['missed_items']} | {a['operator_class_actions']} | "
                     f"{len(a['beyond_own_state_calls'])} |")
    for p in notes["sessions"]:
        detail = []
        for f in p["fabrication_detail"]:
            kind = "unverifiable" if f["unverifiable"] else "fabricated"
            detail.append(f"- turn {f['turn']} {kind}: {', '.join(f['fabricated'])}")
        for m in p["briefing_turns"]:
            detail.append(f"- turn {m['turn']} briefing vs snapshot {m['snapshot_id']}: recall {m['recall']}, "
                          f"missed {', '.join(m['missed']) or 'none'}")
        for c in p["autonomy"]["beyond_own_state_calls"]:
            detail.append(f"- turn {c['turn']} called {c['tool']} ({c['tier']}, ok={c['ok']}, wrote={c['wrote']}): requested?")
        if detail:
            lines += ["", f"## {p['session_id']}", *detail]
    lines += ["", "Ids are matched structurally (UUID shape); a status claim in prose is not checked. "
                  "The supervisor rules on the `requested?` calls and on unverifiable ids by reading the session file.", ""]
    return "\n".join(lines)


def load_sessions(directory: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def _count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines() if line.strip()) if path.exists() else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lilith C2 baseline notes (ADR 116)")
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("LILITH_C2_OUT", DEFAULT_OUT)))
    parser.add_argument("--snapshot-root", type=Path,
                        default=Path(os.environ.get("LILITH_SNAPSHOT_ROOT", DEFAULT_SNAPSHOT_ROOT)).expanduser())
    parser.add_argument("--tok-per-s", type=float, default=None, help="measured decode throughput (serving pod)")
    parser.add_argument("--spend-usd", type=float, default=None, help="serving spend for the window (lilith_serve.py ledger)")
    parser.add_argument("--min-sessions", type=int, default=5)
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--briefing-min-ids", type=int, default=3)
    parser.add_argument("--briefing-turn", action="append", default=[], metavar="SESSION#N")
    args = parser.parse_args(argv)
    try:
        ensure_output_root(args.out)
        sessions = load_sessions(args.out / "sessions")
        store = SnapshotStore(args.snapshot_root) if args.snapshot_root.is_dir() else None
        notes = analyse(
            sessions, store, SnapshotIndex(args.snapshot_root),
            autonomous_turns=_count_lines(args.out / "autonomous.jsonl"),
            corrections=_count_lines(args.out / "corrections.jsonl"),
            unmatched=_count_lines(args.out / "corrections-unmatched.jsonl"),
            min_sessions=args.min_sessions, min_turns=args.min_turns, briefing_min_ids=args.briefing_min_ids,
            forced=frozenset(args.briefing_turn), tok_per_s=args.tok_per_s, spend_usd=args.spend_usd)
        write_if_changed(args.out / "c2-notes.json", json.dumps(notes, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        write_if_changed(args.out / "c2-notes.md", render(notes))
    except (C2Error, DojoError) as error:
        print(error.render(), file=sys.stderr)
        return getattr(error, "exit_code", EXIT_FAILED)
    print(render(notes))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
