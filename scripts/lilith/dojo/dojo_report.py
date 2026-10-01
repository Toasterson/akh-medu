#!/usr/bin/env python3
"""How many CoS dojo episodes the snapshots recorded so far can support.

    dojo_report.py [--root R] [--repo PATH] [--illumos-pack FILE] [--corrections FILE] [--json]

`--root` defaults to `LILITH_SNAPSHOT_ROOT`, then the recorder's default.
`--repo` is a local akh-medu clone for the landed-claim ancestry checks; without
it those episodes are excluded (undecidable), not guessed.
`--corrections` is the C2 `corrections.jsonl` (`c2/harvest_sessions.py`); it feeds the correction family.

The recorder only started on 2026-09-28. The report states the window it has
and what that window can and cannot label; it never extrapolates volume.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dojo import DOJO_VERSION
from dojo import memory_epistemics as ME
from dojo.episode import Family
from dojo.errors import DojoError
from dojo.families import (
    DEFAULT_CORRECTION_ANNOTATIONS,
    BuildContext,
    BuildResult,
    build_all,
    load_corrections,
    load_illumos_pack,
)
from dojo.hindsight import hours_between, phase_changes_in_window
from dojo.replay_server import GitAncestryOracle
from dojo.snapshot import SnapshotStore
from dojo.splits import Split, SplitConfig, assign_splits

RECORDER_DEFAULT_ROOT = "/Volumes/akh-medu-runs/lilith/cos-snapshots"
RECORDER_START = "2026-09-28"

LABEL_KEY = {
    Family.PARK_TRIAGE: "cause",
    Family.LANDED_CLAIM: "verdict",
    Family.COMMITMENT: "expected_state",
    Family.OPERATOR_ACTION: "warranted",
    Family.CAPTAIN_SELECTION: "voice",
    Family.ILLUMOS_GUIDANCE: "outcome",
    Family.CHASE: "recipient_class",
    Family.SCOPE: "scope",
    Family.READ_FAILURE: "cause",
    Family.MEMORY_EPISTEMICS: "variant",
}

NEEDS = {
    Family.CORRECTION: "needs operator corrections from C2 sessions; none are recorded yet",
    Family.ILLUMOS_GUIDANCE: "needs an illumos pack (--illumos-pack) and settled later outcomes",
    Family.LANDED_CLAIM: "needs --repo for the ancestry oracle",
    Family.MEMORY_EPISTEMICS: "needs the harvested C2 sessions (--memory-sessions) and memory-annotations.jsonl",
}


def build_report(root: Path, repo: Path | None, pack: Path | None, corrections: Path | None = None,
                 annotations: Path | None = DEFAULT_CORRECTION_ANNOTATIONS, memory_sessions: Path | None = None) -> dict:
    store = SnapshotStore(root)
    recorded = load_corrections(corrections, annotations) if corrections else ()
    ctx = BuildContext(
        oracle=GitAncestryOracle(repo) if repo else None,
        illumos_pack=load_illumos_pack(pack) if pack else (),
        corrections=recorded,
        memory=ME.memory_world(memory_sessions),
    )
    snaps = store.snapshots()
    report: dict = {"dojo": DOJO_VERSION, "root": str(root), "snapshots": len(snaps)}
    if not snaps:
        report["families"] = {}
        return report
    report["first"] = snaps[0].taken_at.isoformat()
    report["last"] = snaps[-1].taken_at.isoformat()
    report["span_hours"] = round(hours_between(snaps[0].taken_at, snaps[-1].taken_at), 2)
    report["phase_transitions"] = len(phase_changes_in_window(store))
    results: dict[Family, BuildResult] = build_all(store, ctx)
    families = {}
    all_specs = []
    for family, res in results.items():
        all_specs.extend(res.specs)
        key = LABEL_KEY.get(family)
        labels = collections.Counter(str(s.label.get(key)) for s in res.specs) if key else {}
        note = NEEDS.get(family, "")
        if family is Family.LANDED_CLAIM and repo:
            note = ""
        if family is Family.MEMORY_EPISTEMICS and ctx.memory is not None and ctx.memory.sessions is not None and ctx.memory.notes:
            note = ""
        if family is Family.CORRECTION and corrections:
            orphaned = sum(1 for c in recorded if c.snapshot_id not in store.ids)
            note = f"{len(recorded)} recorded corrections" + (f", {orphaned} name a snapshot this root does not hold" if orphaned else "")
        families[family.value] = {
            "episodes": len(res.specs),
            "excluded": len(res.excluded),
            "excluded_reasons": dict(collections.Counter(e.reason.split(":")[0] for e in res.excluded)),
            "labels": dict(sorted(labels.items())),
            "note": note,
        }
    report["families"] = families
    manifest = assign_splits(all_specs, SplitConfig())
    report["splits"] = {s.value: len(manifest.members(s)) for s in Split} | {"dropped_for_leakage": len(manifest.dropped)}
    return report


def render(report: dict) -> str:
    lines = [f"Lilith CoS dojo report ({report['dojo']})", f"snapshot root: {report['root']}"]
    if not report["snapshots"]:
        lines.append("no snapshots yet: nothing can be built.")
        return "\n".join(lines)
    lines += [
        f"snapshots: {report['snapshots']} ({report['first']} .. {report['last']}, span {report['span_hours']} h)",
        f"The recorder started on {RECORDER_START}. This is all the evidence there is: "
        f"{report['snapshots']} snapshots over {report['span_hours']} hours, "
        f"{report['phase_transitions']} WI phase transitions between them.",
        "Families that label from later snapshots (park resolution, stale confirmation, commitments, "
        "illumos and steering outcomes) stay thin until the window covers days of transitions.",
        "",
        f"{'family':<20} {'episodes':>8} {'excluded':>8}  labels / note",
    ]
    for name, fam in report["families"].items():
        detail = ", ".join(f"{k}={v}" for k, v in fam["labels"].items())
        extra = "; ".join(x for x in (detail, fam["note"],
                                      ", ".join(f"{k}: {v}" for k, v in fam["excluded_reasons"].items())) if x)
        lines.append(f"{name:<20} {fam['episodes']:>8} {fam['excluded']:>8}  {extra}")
    s = report["splits"]
    lines += ["", f"splits (7-day windows from {RECORDER_START}): train {s['train']} / dev {s['dev']} / "
                  f"sealed {s['sealed']} / dropped for leakage {s['dropped_for_leakage']}"]
    if s["train"] == 0 and s["dev"] == 0:
        lines.append("Only one window exists, so everything is sealed; train/dev start filling after the first week.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lilith CoS dojo: episodes supported by the snapshots so far")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None, help="local akh-medu clone for ancestry checks")
    parser.add_argument("--illumos-pack", type=Path, default=None)
    parser.add_argument("--memory-sessions", type=Path, default=None,
                        help="harvested C2 sessions (c2/sessions): the memory-epistemics seeds")
    parser.add_argument("--corrections", type=Path, default=None,
                        help="C2 corrections.jsonl from c2/harvest_sessions.py")
    parser.add_argument("--correction-annotations", type=Path, default=DEFAULT_CORRECTION_ANNOTATIONS,
                        help="what each correction asked for (expected_tool / expected_arguments), default: "
                             "correction-annotations.jsonl next to the dojo")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    root = args.root or Path(os.environ.get("LILITH_SNAPSHOT_ROOT", RECORDER_DEFAULT_ROOT))
    try:
        report = build_report(root, args.repo, args.illumos_pack, args.corrections, args.correction_annotations,
                              args.memory_sessions)
    except DojoError as error:
        print(error.render(), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
