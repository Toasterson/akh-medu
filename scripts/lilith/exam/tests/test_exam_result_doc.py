"""RESULT.md skeleton: template order, explicit gaps, gate rows from family matrices."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from exam.controller import FamilyGate, ItemOutcome, family_matrix, paired_difference
from exam.errors import ResultDocError
from exam.manifest import MetricFloor
from exam.result_doc import render

SCRIPT = Path(__file__).resolve().parents[1] / "result_doc.py"
SECTIONS = ["## Verdict", "## Configuration", "## Timeline", "## Per-family gates", "## Best candidates",
            "## Case analysis", "## Incidents", "## Cost", "## Retained artifacts", "## Open hypotheses",
            "## Exit condition"]


def skeleton(**extra):
    return {"schema": "lilith/run-result@1", "run_id": "lilith-train-r1", "title": "Lilith train r1",
            "anima_wi": "01a0e2db-ed79"} | extra


def test_skeleton_has_every_section_in_template_order():
    text = render(skeleton())
    positions = [text.index(s) for s in SECTIONS]
    assert positions == sorted(positions)
    assert text.count("_Not recorded yet:") == 10  # all but the exit condition, which has a default
    assert "- [ ] pods stopped" in text and "- [ ] WireGuard peer removed" in text


def test_gate_rows_from_matrices_and_comparisons():
    gate = {"park-triage": FamilyGate("park-triage", "B", None, 0.0, (MetricFloor("B", "accuracy", 0.80),))}

    def arm(name, passed):
        items = [ItemOutcome(f"{name}-{i}", "park-triage", i < passed, metrics={"accuracy": float(i < passed)})
                 for i in range(120)]
        return family_matrix(items, gate, arm=name).to_dict()

    run = skeleton(
        verdict={"status": "NOT certified", "summary": "dry run"},
        matrices={"frozen-parent": arm("frozen-parent", 80), "teacher": arm("teacher", 115),
                  "candidate": arm("candidate", 110)},
        comparisons={"park-triage": {"vs": "frozen-parent", **paired_difference(
            [(i < 110, i < 80) for i in range(120)]).to_dict()}},
        gates=[{"gate": "D", "family": "white-belt", "metric": "kata", "floor": "≥ 0.80", "verdict": "not run"}],
        cases=[{"case": "p-3", "class": "decoder/template", "analysis": "template drift", "source": "x"}],
        cost={"rows": [{"pod": "abc", "day": "2026-09-28", "usd": 3.1, "hours": 1.2}], "total_usd": 3.1,
              "cap_usd": 40},
        exit_condition={"items": [{"item": "gates evaluated", "met": True}, {"item": "pods stopped", "met": False}]},
    )
    text = render(run)
    assert text.index("## Verdict") < text.index("NOT certified") < text.index("## Configuration")
    row = next(line for line in text.splitlines() if line.startswith("| B | park-triage | accuracy"))
    assert "0.667" in row and "0.958" in row and "0.917" in row and "vs frozen-parent: better" in row
    assert "| D | white-belt | kata" in text
    assert "- [x] gates evaluated" in text and "- [ ] pods stopped" in text
    assert "Total: $3.1; run cap: $40." in text


def test_errors_are_diagnostics():
    with pytest.raises(ResultDocError):
        render({"schema": "other"})
    with pytest.raises(ResultDocError):
        render(skeleton(cases=[{"case": "x", "class": "vibes"}]))
    with pytest.raises(ResultDocError):
        render(skeleton(matrices={"student": {}}))


def test_cli_writes_and_refuses_to_overwrite(tmp_path):
    run = tmp_path / "run.json"
    run.write_text(json.dumps(skeleton()))
    out = tmp_path / "RESULT.md"
    ok = subprocess.run([sys.executable, str(SCRIPT), "--run", str(run), "-o", str(out)], capture_output=True, text=True)
    assert ok.returncode == 0 and out.read_text().startswith("# Lilith train r1 — RESULT")
    again = subprocess.run([sys.executable, str(SCRIPT), "--run", str(run), "-o", str(out)], capture_output=True,
                           text=True)
    assert again.returncode == 1 and "lilith::exam::result_doc" in again.stderr
