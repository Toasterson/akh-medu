#!/usr/bin/env python3
"""RESULT.md skeleton for a Lilith run, in the 8k-lane / P6.4 template (workflow §4 step 9).

    python3 scripts/lilith/exam/result_doc.py --run run.json [-o training/results/lilith-<run>-RESULT.md]

Input: one `lilith/run-result@1` JSON object. Every section is always rendered, in the
template order (verdict first); a section without data renders as an explicit
"not recorded yet" line, so a skeleton shows what is still missing instead of hiding it.

Fields (all optional except `schema`, `run_id`, `title`):
  verdict        {"status", "summary"}
  sources        ["<path>", ...]
  config         [{"setting", "value", "source"}]
  timeline       [{"time", "event", "source"}]
  matrices       {"frozen-parent" | "teacher" | "candidate": FamilyMatrix.to_dict()}
  comparisons    {"<family>": {"vs": "frozen-parent", **PairedDifference.to_dict()}}
  gates          [{"gate", "family", "metric", "floor", "parent", "teacher", "candidate", "verdict", "source"}]
                 (extra rows appended after the matrix-derived rows: panel, belts, E)
  best_candidates [{"checkpoint", "summary", "source"}]
  cases          [{"case", "class", "analysis", "source"}]; class in model | decoder/template |
                 tool/environment | harness | data (workflow §4 step 7)
  incidents      [{"incident", "detail", "fix"}]
  cost           {"rows": [{"pod", "day", "usd", "hours", "source"}], "total_usd", "credit_usd", "cap_usd"}
  retained       ["<artifact>", ...]
  hypotheses     ["<hypothesis>", ...]
  exit_condition {"text", "items": [{"item", "met"}]}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exam.errors import ResultDocError  # noqa: E402

RUN_RESULT_SCHEMA = "lilith/run-result@1"
ARMS = ("frozen-parent", "teacher", "candidate")
FAILURE_CLASSES = ("model", "decoder/template", "tool/environment", "harness", "data")
DEFAULT_EXIT = "gates evaluated + RESULT doc + pods stopped + peer removed"


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return out


def _missing(hint: str) -> list[str]:
    return [f"_Not recorded yet: {hint}._"]


def _metric_cell(row: Mapping[str, Any] | None, metric: str) -> str | None:
    if row is None:
        return None
    if metric == "catastrophic":
        return f"{row['catastrophic']}/{row['n']}"
    for test in row.get("tests", []):
        if test["metric"] == metric:
            interval = test.get("interval")
            ci = f" ({interval['lower']:.2f}–{interval['upper']:.2f})" if interval else " (point)"
            return f"{test['mean']:.3f}{ci} n={test['n']}"
    return None


def gate_rows(matrices: Mapping[str, Mapping[str, Any]], comparisons: Mapping[str, Mapping[str, Any]]) -> list[list[Any]]:
    """One row per (family, gated metric) plus the catastrophic row, parent/teacher/candidate."""
    for arm in matrices:
        if arm not in ARMS:
            raise ResultDocError(f"unknown arm {arm!r}; expected one of {ARMS}")
    candidate = matrices.get("candidate", {}).get("families", {})
    families = sorted(set().union(*(m.get("families", {}) for m in matrices.values()))) if matrices else []
    rows: list[list[Any]] = []
    for family in families:
        per_arm = {arm: matrices.get(arm, {}).get("families", {}).get(family) for arm in ARMS}
        ref = next((r for r in (per_arm["candidate"], per_arm["frozen-parent"], per_arm["teacher"]) if r), None)
        if ref is None:
            continue
        metrics = [(t["metric"], t["floor"]) for t in ref.get("tests", [])]
        metrics.append(("catastrophic", f"≤ {ref['catastrophic_ceiling']}"))
        cand = candidate.get(family)
        verdict = cand["verdict"] if cand else "missing"
        if cand and cand.get("blockers"):
            verdict += ": " + "; ".join(cand["blockers"])
        elif cand and cand.get("missing"):
            verdict += ": " + "; ".join(cand["missing"])
        comparison = comparisons.get(family)
        if comparison:
            verdict += (f" · vs {comparison.get('vs', 'frozen-parent')}: {comparison['verdict']} "
                        f"(Δ {comparison['delta']:+.3f}, {comparison['lower']:+.3f}…{comparison['upper']:+.3f})")
        for metric, floor in metrics:
            rows.append([ref["gate"], family, metric, floor if metric == "catastrophic" else f"≥ {floor}",
                         _metric_cell(per_arm["frozen-parent"], metric), _metric_cell(per_arm["teacher"], metric),
                         _metric_cell(per_arm["candidate"], metric), verdict,
                         "family matrices (" + ", ".join(a for a in ARMS if per_arm[a]) + ")"])
    return rows


def render(run: Mapping[str, Any]) -> str:
    if run.get("schema") != RUN_RESULT_SCHEMA:
        raise ResultDocError(f"schema {run.get('schema')!r} is not {RUN_RESULT_SCHEMA}")
    for key in ("run_id", "title"):
        if not run.get(key):
            raise ResultDocError(f"run JSON has no {key!r}")
    out: list[str] = [f"# {run['title']} — RESULT", ""]
    refs = [f"Anima WI `{run['anima_wi']}`" if run.get("anima_wi") else None,
            f"plan `{run['plan']}`" if run.get("plan") else None, f"run `{run['run_id']}`"]
    out += ["; ".join(r for r in refs if r) + ".", ""]
    if run.get("sources"):
        out += ["Sources:"] + [f"- `{s}`" for s in run["sources"]] + [""]

    out += ["## Verdict", ""]
    verdict = run.get("verdict") or {}
    if verdict.get("status"):
        out += [f"**{verdict['status']}.** {verdict.get('summary', '')}".rstrip()]
    else:
        out += _missing("verdict status and one-paragraph summary (write this first)")
    out.append("")

    out += ["## Configuration", ""]
    config = run.get("config") or []
    out += _table(["Setting", "Value", "Source"], [[c.get("setting"), c.get("value"), c.get("source")] for c in config]) \
        if config else _missing("model/parent, adapter, data manifests, exam manifest hash, numerics, code main SHA, pod")
    out.append("")

    out += ["## Timeline", ""]
    timeline = run.get("timeline") or []
    out += _table(["Time (UTC)", "Event", "Source"], [[t.get("time"), t.get("event"), t.get("source")] for t in timeline]) \
        if timeline else _missing("windows, controller decisions, selection freeze, sealed eval, pod stop")
    out.append("")

    out += ["## Per-family gates", ""]
    rows = gate_rows(run.get("matrices") or {}, run.get("comparisons") or {})
    rows += [[g.get("gate"), g.get("family"), g.get("metric"), g.get("floor"), g.get("parent"), g.get("teacher"),
              g.get("candidate"), g.get("verdict"), g.get("source")] for g in run.get("gates") or []]
    out += _table(["Gate", "Family", "Metric", "Floor", "Frozen parent", "Teacher", "Candidate", "Verdict", "Source"],
                  rows) if rows else _missing("sealed per-family matrices for parent, teacher and candidate (gates A–E)")
    out += ["", "Intervals are Wilson 95%; floor tests are Benjamini–Hochberg-corrected across families. "
                "No aggregate score is reported: every family stands on its own row.", ""]

    out += ["## Best candidates", ""]
    best = run.get("best_candidates") or []
    out += _table(["Checkpoint", "Summary", "Source"], [[b.get("checkpoint"), b.get("summary"), b.get("source")]
                                                        for b in best]) \
        if best else _missing("the dev-selected complete-episode checkpoint(s), frozen before sealed scoring")
    out.append("")

    out += ["## Case analysis", ""]
    cases = run.get("cases") or []
    bad = [c.get("class") for c in cases if c.get("class") not in FAILURE_CLASSES]
    if bad:
        raise ResultDocError(f"case classes {bad} are not in {FAILURE_CLASSES}")
    out += _table(["Case", "Class", "Analysis", "Source"],
                  [[c.get("case"), c.get("class"), c.get("analysis"), c.get("source")] for c in cases]) \
        if cases else _missing("every failure classified with receipts as " + " / ".join(FAILURE_CLASSES))
    out.append("")

    out += ["## Incidents", ""]
    incidents = run.get("incidents") or []
    out += _table(["Incident", "Detail", "Fix / resolution"],
                  [[i.get("incident"), i.get("detail"), i.get("fix")] for i in incidents]) \
        if incidents else _missing("incidents, or 'none' with the watchdog log as source")
    out.append("")

    out += ["## Cost", ""]
    cost = run.get("cost") or {}
    if cost.get("rows"):
        out += _table(["Pod", "Day", "Cost (USD)", "Time billed (h)", "Source"],
                      [[r.get("pod"), r.get("day"), r.get("usd"), r.get("hours"), r.get("source")] for r in cost["rows"]])
        extra = [f"Total: ${cost['total_usd']}" if cost.get("total_usd") is not None else None,
                 f"credit at last check: ${cost['credit_usd']}" if cost.get("credit_usd") is not None else None,
                 f"run cap: ${cost['cap_usd']}" if cost.get("cap_usd") is not None else None]
        if any(extra):
            out += ["", "; ".join(e for e in extra if e) + "."]
    else:
        out += _missing("spend per pod-hour from the Runpod billing readout, against the run WI's dollar cap")
    out.append("")

    out += ["## Retained artifacts", ""]
    retained = run.get("retained") or []
    out += [f"- {r}" for r in retained] if retained else _missing("pods/volume state and the checkpoints the candidate references")
    out.append("")

    out += ["## Open hypotheses", ""]
    hypotheses = run.get("hypotheses") or []
    out += [f"{n}. {h}" for n, h in enumerate(hypotheses, 1)] if hypotheses else _missing("what the next run should test, one factor per control arm")
    out.append("")

    out += ["## Exit condition", ""]
    exit_condition = run.get("exit_condition") or {}
    out += [f"Exit condition: {exit_condition.get('text') or DEFAULT_EXIT}.", ""]
    items = exit_condition.get("items") or []
    out += [f"- [{'x' if i.get('met') else ' '}] {i.get('item')}" for i in items] if items else \
        [f"- [ ] {i}" for i in ("gates evaluated", "RESULT doc", "pods stopped", "WireGuard peer removed")]
    out.append("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True, help="lilith/run-result@1 JSON")
    parser.add_argument("-o", "--output", type=Path, help="write here instead of stdout (refuses to overwrite)")
    args = parser.parse_args(argv)
    try:
        text = render(json.loads(args.run.read_text()))
        if args.output:
            if args.output.exists():
                raise ResultDocError(f"{args.output} exists", help="RESULT docs are evidence; write a new file or "
                                                                   "edit the existing one by hand")
            args.output.write_text(text)
        else:
            sys.stdout.write(text)
    except ResultDocError as err:
        print(err.render(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
