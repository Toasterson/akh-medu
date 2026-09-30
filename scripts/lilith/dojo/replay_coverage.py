#!/usr/bin/env python3
"""What each recorded snapshot can serve through the replay server (plan slice C4, ADR 116 Decision 3).

For every snapshot, every Anima read tool is probed with the real replay resolver
(`replay_match.resolve_recorded_read` / `explain_miss`, exactly what `ReplayServer` runs) and
classified:

- recorded        the recorder captured the call itself (the argument shapes are listed);
- derived         the call is answered from recorded objects (`replay_anima`: re-filtered and
                  re-paged work-item listings, the project set, `get_work_item_by_id`, `get_plan`);
- not-replayable  a free-form query tool: every arm gets the same structured error;
- absent          the snapshot holds nothing for the tool (`tool-not-recorded`).

    python3 -m dojo.replay_coverage --root <snapshots> [--snapshot ID] [--json]

Without `--snapshot` one block per snapshot plus a summary over all of them is printed; with it, the
detailed probe table of that snapshot. Read-only over the root.
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo import replay_anima
from dojo.errors import DojoError
from dojo.replay_match import explain_miss, resolve_recorded_read
from dojo.replay_server import ANIMA_READ_TOOLS, NOT_REPLAYABLE_ANIMA_TOOLS
from dojo.snapshot import Snapshot, SnapshotStore

STATUS_RECORDED = "recorded"
STATUS_DERIVED = "derived"
STATUS_BOTH = "recorded+derived"
STATUS_NOT_REPLAYABLE = "not-replayable"
STATUS_ABSENT = "absent"
SERVED = frozenset({STATUS_RECORDED, STATUS_DERIVED, STATUS_BOTH})


@dataclasses.dataclass(frozen=True)
class Probe:
    tool: str
    arguments: Mapping[str, Any]
    served: bool
    detail: str  # the match rules when served, the miss reason otherwise

    def to_dict(self) -> dict:
        return {"tool": self.tool, "arguments": dict(self.arguments), "served": self.served, "detail": self.detail}


@dataclasses.dataclass(frozen=True)
class ToolCoverage:
    tool: str
    status: str
    recorded_calls: int
    arg_shapes: tuple[str, ...]  # recorded argument-key sets, `key+key (n)`

    def to_dict(self) -> dict:
        return dataclasses.asdict(self) | {"arg_shapes": list(self.arg_shapes)}


@dataclasses.dataclass(frozen=True)
class SnapshotCoverage:
    snapshot_id: str
    projects: int
    work_items: int
    collectors: Mapping[str, str]  # collector -> status
    tools: tuple[ToolCoverage, ...]
    probes: tuple[Probe, ...]

    @property
    def served(self) -> tuple[str, ...]:
        return tuple(t.tool for t in self.tools if t.status in SERVED)

    @property
    def absent(self) -> tuple[str, ...]:
        return tuple(t.tool for t in self.tools if t.status == STATUS_ABSENT)

    def to_dict(self) -> dict:
        return {"snapshot": self.snapshot_id, "projects": self.projects, "work_items": self.work_items,
                "collectors": dict(self.collectors), "tools": [t.to_dict() for t in self.tools],
                "probes": [p.to_dict() for p in self.probes]}


def _schema(snapshot: Snapshot, tool: str) -> Mapping[str, Any] | None:
    contract = snapshot.tool_contract()
    if contract is None:
        return None
    return next((t.get("inputSchema") for t in contract[0] if t.get("name") == tool), None)


def probe(snapshot: Snapshot, tool: str, arguments: Mapping[str, Any]) -> Probe:
    schema = _schema(snapshot, tool)
    found = resolve_recorded_read(snapshot, tool, schema, arguments)
    if found is not None:
        return Probe(tool, arguments, True, ", ".join(found.match.rules) or "exact")
    return Probe(tool, arguments, False, explain_miss(snapshot, tool, schema, arguments).reason)


def _shapes(snapshot: Snapshot, tool: str) -> tuple[int, tuple[str, ...]]:
    counts: collections.Counter[str] = collections.Counter()
    calls = 0
    for response in snapshot.responses:
        if response.call == tool:
            calls += 1
            counts["+".join(sorted(k for k, v in response.args.items() if v is not None)) or "(none)"] += 1
    return calls, tuple(f"{shape} ({n})" for shape, n in sorted(counts.items()))


def _canonical_probes(snapshot: Snapshot) -> list[Probe]:
    """Representative variants of the calls a model makes, run through the real resolver."""
    probes = [probe(snapshot, "list_projects", {}),
              probe(snapshot, "list_work_items", {}),
              probe(snapshot, "list_work_items", {"page_size": 100}),
              probe(snapshot, "list_work_items", {"current_phase": "parked"}),
              probe(snapshot, "list_work_items", {"project_id": "00000000-0000-0000-0000-000000000000"})]
    listings = replay_anima.project_listings(snapshot)
    for project_id, listing in listings.items():
        probes.append(probe(snapshot, "list_work_items", {"project_id": project_id, "page_size": 100}))
        if listing.items:
            first = listing.items[0]
            probes.append(probe(snapshot, "get_work_item_by_id", {"id": first["id"]}))
            probes.append(probe(snapshot, "get_work_item_detail", {"id": first["id"]}))
            if first.get("plan_id"):
                probes.append(probe(snapshot, "list_work_items", {"project_id": project_id, "plan_id": first["plan_id"]}))
    plans = replay_anima.recorded_plans(snapshot)
    if plans:
        probes.append(probe(snapshot, "get_plan", {"id": next(iter(plans))}))
    for project_id in sorted(replay_anima.project_ids(snapshot))[:1]:  # one per snapshot: they are exact recordings
        probes.append(probe(snapshot, "list_plans", {"project_id": project_id}))
    return probes


def coverage_of(snapshot: Snapshot) -> SnapshotCoverage:
    probes = _canonical_probes(snapshot)
    derived_tools = {p.tool for p in probes if p.served and "derive:" in p.detail}
    tools = []
    for tool in sorted(ANIMA_READ_TOOLS | NOT_REPLAYABLE_ANIMA_TOOLS):
        calls, shapes = _shapes(snapshot, tool)
        if tool in NOT_REPLAYABLE_ANIMA_TOOLS:
            status = STATUS_NOT_REPLAYABLE
        elif calls and tool in derived_tools:
            status = STATUS_BOTH
        elif calls:
            status = STATUS_RECORDED
        elif tool in derived_tools:
            status = STATUS_DERIVED
        else:
            status = STATUS_ABSENT
        tools.append(ToolCoverage(tool, status, calls, shapes))
    listings = replay_anima.project_listings(snapshot)
    collectors = {name: str(c.get("status")) for name, c in sorted(snapshot.manifest.get("collectors", {}).items())}
    return SnapshotCoverage(snapshot.id, len(replay_anima.project_ids(snapshot)),
                            sum(len(listing.items) for listing in listings.values()), collectors, tuple(tools),
                            tuple(probes))


def render_detail(cov: SnapshotCoverage) -> str:
    lines = [f"## snapshot {cov.snapshot_id}: {cov.projects} projects, {cov.work_items} listed work items", "",
             "collectors: " + ", ".join(f"{n}={s}" for n, s in cov.collectors.items() if not n.startswith("anima:")) +
             f", anima:<project> x{sum(1 for n in cov.collectors if n.startswith('anima:'))}", "",
             "| tool | status | recorded calls | recorded argument shapes |", "|---|---|---|---|"]
    for t in cov.tools:
        lines.append(f"| {t.tool} | {t.status} | {t.recorded_calls} | {'; '.join(t.arg_shapes) or '-'} |")
    lines += ["", "probes (the real resolver):", ""]
    for p in cov.probes:
        lines.append(f"- {'SERVED' if p.served else 'MISS  '} {p.tool} {json.dumps(dict(p.arguments), sort_keys=True)}  -> {p.detail}")
    return "\n".join(lines) + "\n"


def render_summary(covs: list[SnapshotCoverage]) -> str:
    lines = []
    for cov in covs:
        lines.append(f"{cov.snapshot_id}  projects={cov.projects} items={cov.work_items}  "
                     f"served[{len(cov.served)}]={','.join(cov.served) or '-'}  absent[{len(cov.absent)}]={','.join(cov.absent) or '-'}")
    lines += ["", f"summary over {len(covs)} snapshot(s):", "", "| tool | status (snapshots) |", "|---|---|"]
    tools = sorted({t.tool for c in covs for t in c.tools})
    for tool in tools:
        tally = collections.Counter(next(t.status for t in c.tools if t.tool == tool) for c in covs)
        lines.append(f"| {tool} | " + ", ".join(f"{s} {n}" for s, n in sorted(tally.items())) + " |")
    probes = collections.Counter((p.tool, p.served, p.detail.split(",")[0]) for c in covs for p in c.probes)
    lines += ["", "probe outcomes (tool, served, first rule or miss reason): count", ""]
    for (tool, served, detail), n in sorted(probes.items()):
        lines.append(f"- {tool} {'SERVED' if served else 'MISS'} {detail}: {n}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="snapshot root (holds <ts>/manifest.json and objects/)")
    parser.add_argument("--snapshot", default=None, help="one snapshot id: print its detailed probe table")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        store = SnapshotStore(args.root)
        snapshots = [store.get(args.snapshot)] if args.snapshot else store.snapshots()
        covs = [coverage_of(s) for s in snapshots]
    except DojoError as error:
        print(error.render(), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps([c.to_dict() for c in covs], indent=2, sort_keys=True))
    elif args.snapshot:
        print(render_detail(covs[0]))
    else:
        print(render_summary(covs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
