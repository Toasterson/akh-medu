#!/usr/bin/env python3
"""Compare akheventd's shadow Anima routing with laya_controller's decisions (Phase 53g PR 6).

Read-only on both sides: SQLite databases are opened with `mode=ro` URIs and
nothing is written anywhere.

Two comparisons:

- **shadow** (default): the would-be events akheventd's `roman-anima` source
  recorded in shadow mode (from its store, `state/events/akheventd.sqlite3`,
  or a JSON dump of `GET /v1/anima/roman-anima/shadow`) against the decisions
  in `state/laya-decisions.sqlite3` over the same window. Per entity it
  compares the set of roles each side routed to. Laya decided on its own
  queue cadence and the source emits on projection changes, so an entity can
  appear on only one side inside a window; the agreement that matters is on
  entities both sides routed.
- **replay** (`--replay`): runs the rendered route filters over every
  snapshot Laya recorded and compares with the role Laya's `host_action`
  chose for that same snapshot. This is the structural parity check; it
  needs no shadow period.

Exit status: 0 when every compared entity agrees, 1 on a mismatch, 2 on a
usage or input error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import full_akh as fa  # noqa: E402
import route_filter  # noqa: E402

DEFAULT_ROOT = Path.home() / "Library/Application Support/roman-akhs"
SHADOW_KEY_PREFIX = "anima_shadow/"


def connect_ro(path: Path) -> sqlite3.Connection:
    """Open `path` read-only (never creates, never writes)."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def event_from_laya_snapshot(snapshot: dict) -> tuple[str, dict]:
    """The (event type, event data) akheventd would emit for a Laya snapshot.

    Mirrors the Anima source's projection: the configured work-item fields
    plus `live_sessions` (the count of live sessions). A snapshot whose
    session read failed carries no count, which the filters treat as no live
    owner, exactly as Laya treated its empty `live_sessions` list.
    """
    if snapshot.get("kind") == "plan":
        detail = snapshot.get("detail") or {}
        return fa.PLAN_EVENT, {"kind": "plan", "entity": "plan:" + str(detail.get("id")), "current": {}}
    work_item = (snapshot.get("detail") or {}).get("work_item") or {}
    current = {k: work_item[k] for k in fa.ANIMA_FIELDS if k in work_item}
    if "sessions" in snapshot:
        current["live_sessions"] = len(snapshot.get("live_sessions") or [])
    return fa.WORK_ITEM_EVENT, {"kind": "work_item", "entity": "work_item:" + str(work_item.get("id")),
                                "current": current}


def routed_roles(routes: list[dict], source: str, event_type: str, data: dict) -> set[str]:
    """Akhs whose route delivers this event (source, type and `when`)."""
    roles = set()
    for route in routes:
        if route.get("source") != source or route.get("event_type") != event_type:
            continue
        if "when" in route and not route_filter.matches(route["when"], data):
            continue
        akh = fa.route_akh(route)
        if akh:
            roles.add(akh)
    return roles


def anima_routes(events_config: Path | None) -> list[dict]:
    """The PR 6 routes from `events_config` when it has them, else as rendered."""
    if events_config and events_config.is_file():
        routes = [r for r in json.loads(events_config.read_text()).get("routes", [])
                  if r.get("event_type") in (fa.WORK_ITEM_EVENT, fa.PLAN_EVENT)]
        if routes:
            return routes
    return fa.anima_routes()


def laya_decisions(db: sqlite3.Connection, since: float) -> list[tuple[str, str, str | None, str]]:
    """(entity, snapshot json, role, action) rows created at or after `since`."""
    return db.execute("select entity, snapshot, role, action from decisions where created_at >= ? "
                      "order by created_at", (since,)).fetchall()


def shadow_records_from_db(db: sqlite3.Connection, source: str, since_ms: int) -> list[dict]:
    rows = db.execute("select value from records_v1 where key >= ? and key < ? order by key",
                      (SHADOW_KEY_PREFIX, SHADOW_KEY_PREFIX[:-1] + "0")).fetchall()
    records = []
    for (value,) in rows:
        record = json.loads(value)
        if record.get("source") == source and record.get("recorded_at_ms", 0) >= since_ms:
            records.append(record)
    return records


def shadow_records_from_json(path: Path, source: str, since_ms: int) -> list[dict]:
    return [r for r in json.loads(path.read_text())
            if r.get("source") == source and r.get("recorded_at_ms", 0) >= since_ms]


def replay(decisions, routes: list[dict]) -> dict:
    """Filters vs host_action on each recorded snapshot."""
    agree, mismatches, by_role = 0, [], {}
    for entity, raw, role, action in decisions:
        event_type, data = event_from_laya_snapshot(json.loads(raw))
        filtered = routed_roles(routes, fa.ANIMA_SOURCE, event_type, data)
        expected = {role} if role else set()
        key = role or "-"
        by_role.setdefault(key, [0, 0])
        by_role[key][1] += 1
        if filtered == expected:
            agree += 1
            by_role[key][0] += 1
        else:
            mismatches.append({"entity": entity, "action": action, "laya": sorted(expected),
                               "filters": sorted(filtered), "current": data["current"]})
    return {"mode": "replay", "decisions": len(decisions), "agree": agree,
            "by_role": {k: {"agree": v[0], "total": v[1]} for k, v in sorted(by_role.items())},
            "mismatches": mismatches}


def compare(decisions, records: list[dict]) -> dict:
    """Per-entity role sets: Laya decisions vs shadow routes in one window."""
    laya: dict[str, set[str]] = {}
    for entity, _raw, role, _action in decisions:
        laya.setdefault(entity, set())
        if role:
            laya[entity].add(role)
    shadow: dict[str, set[str]] = {}
    for record in records:
        entity = (record.get("event") or {}).get("subject") or ""
        shadow.setdefault(entity, set()).update(r["akh"] for r in record.get("routes", []) if r.get("akh"))
    routed_laya = {e for e, roles in laya.items() if roles}
    routed_shadow = {e for e, roles in shadow.items() if roles}
    both = sorted(routed_laya & routed_shadow)
    same = [e for e in both if laya[e] == shadow[e]]
    mismatches = [{"entity": e, "laya": sorted(laya[e]), "shadow": sorted(shadow[e])}
                  for e in both if laya[e] != shadow[e]]
    return {"mode": "shadow", "laya_entities": len(laya), "shadow_entities": len(shadow),
            "routed_by_both": len(both), "same_roles": len(same),
            "laya_only": sorted(routed_laya - routed_shadow),
            "shadow_only": sorted(routed_shadow - routed_laya),
            "mismatches": mismatches}


def print_report(report: dict, limit: int) -> None:
    if report["mode"] == "replay":
        total = report["decisions"]
        print(f"replay: {report['agree']}/{total} recorded Laya decisions route to the same role "
              f"through the rendered filters")
        for role, counts in report["by_role"].items():
            print(f"  {role:8} {counts['agree']}/{counts['total']}")
    else:
        print(f"shadow: {report['laya_entities']} Laya entities, {report['shadow_entities']} shadow entities")
        print(f"  routed by both: {report['routed_by_both']}, same roles: {report['same_roles']}")
        print(f"  routed by Laya only: {len(report['laya_only'])}; by the shadow only: {len(report['shadow_only'])}")
        for label in ("laya_only", "shadow_only"):
            for entity in report[label][:limit]:
                print(f"    {label}: {entity}")
    for mismatch in report["mismatches"][:limit]:
        print("  MISMATCH " + json.dumps(mismatch, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="roman-akhs deployment root")
    parser.add_argument("--laya-db", type=Path, help="default: <root>/state/laya-decisions.sqlite3")
    parser.add_argument("--events-db", type=Path, help="default: <root>/state/events/akheventd.sqlite3")
    parser.add_argument("--shadow-json", type=Path, help="a saved GET /v1/anima/<source>/shadow body")
    parser.add_argument("--events-config", type=Path, help="default: <root>/config/events.json")
    parser.add_argument("--source", default=fa.ANIMA_SOURCE)
    parser.add_argument("--hours", type=float, default=48.0, help="comparison window (shadow mode)")
    parser.add_argument("--since", type=float, help="window start, epoch seconds (overrides --hours)")
    parser.add_argument("--replay", action="store_true", help="filters vs host_action on recorded snapshots")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument("--limit", type=int, default=20, help="entities listed per category")
    args = parser.parse_args(argv)
    root = args.root
    since = args.since if args.since is not None else time.time() - args.hours * 3600
    laya_db = connect_ro(args.laya_db or root / "state/laya-decisions.sqlite3")
    try:
        routes = anima_routes(args.events_config or root / "config/events.json")
        if args.replay:
            report = replay(laya_decisions(laya_db, 0.0), routes)
        else:
            decisions = laya_decisions(laya_db, since)
            if args.shadow_json:
                records = shadow_records_from_json(args.shadow_json, args.source, int(since * 1000))
            else:
                events_db = connect_ro(args.events_db or root / "state/events/akheventd.sqlite3")
                try:
                    records = shadow_records_from_db(events_db, args.source, int(since * 1000))
                finally:
                    events_db.close()
            report = compare(decisions, records)
    finally:
        laya_db.close()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_report(report, args.limit)
    return 0 if not report["mismatches"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, sqlite3.Error) as error:
        print("compare_anima_shadow: " + str(error), file=sys.stderr)
        sys.exit(2)
