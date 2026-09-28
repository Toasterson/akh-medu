"""Judging panel: rotation, blinding, work orders, ingestion, adjudication, κ."""
import json

import pytest

from exam.errors import PanelError
from exam.panel import PAIRS, Panel, PanelItem, cohen_kappa
from exam_helpers import rubric

DIMS = ("effect", "guidance", "tone")


def items(batches=6, per_batch=4):
    out = []
    for b in range(1, batches + 1):
        for i in range(per_batch):
            arm = "candidate" if i % 2 == 0 else "frozen-parent"
            out.append(PanelItem(f"b{b}-{i}", f"b{b}", arm, {
                "turns": [{"id": "t1", "text": "chase message", "voice_label": "weaver"}],
                "arm": arm, "meta": {"checkpoint": 1200, "situation": "blocked on WI-1"}}))
    return out


def result(panel, a, judge, verdict="pass", scores=None, turns=("t1",), rubric_sha=None):
    return {"schema": "lilith/panel-result@1", "blind_id": a.blind_id, "judge": judge,
            "rubric_sha256": rubric_sha or panel.rubric.rubric_sha256, "verdict": verdict,
            "scores": scores or {d: 3 for d in DIMS}, "cited_turns": list(turns), "note": ""}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_rotation_matches_the_c5a_first_use():
    panel = Panel(rubric(), salt="s")
    pairs = {a.batch: a.judges for a in panel.assign(items())}
    assert pairs == {"b1": PAIRS[0], "b2": PAIRS[0], "b3": PAIRS[1], "b4": PAIRS[1], "b5": PAIRS[2], "b6": PAIRS[2]}
    assert PAIRS == (("fable", "opus"), ("opus", "astra"), ("astra", "fable"))


def test_work_orders_are_blinded_and_rubric_is_frozen(tmp_path):
    panel = Panel(rubric(), salt="s")
    index = panel.work_orders(items(), tmp_path)
    assert len(index["orders"]) == 12  # 6 batches x 2 judges
    first = index["orders"][0]
    lines = [json.loads(line) for line in (tmp_path / first["file"]).read_text().splitlines()]
    text = json.dumps(lines)
    for leaked in ("candidate", "frozen-parent", "voice_label", "weaver", "checkpoint"):
        assert leaked not in text
    assert lines[0]["content"]["turns"][0]["text"] == "chase message"
    assert all(line["rubric_sha256"] == panel.rubric.rubric_sha256 for line in lines)
    assert json.loads((tmp_path / "rubric.json").read_text())["id"] == "cos-social-effect"
    unblind = json.loads((tmp_path / "supervisor" / "unblind.json").read_text())
    assert {v["arm"] for v in unblind.values()} == {"candidate", "frozen-parent"}
    assert "codex exec" in next(o for o in index["orders"] if o["judge"] == "astra")["dispatch"]


def test_ingest_validation(tmp_path):
    panel = Panel(rubric(), salt="s")
    a = panel.assign(items(batches=1, per_batch=1))[0]  # fable + opus
    bad = [
        result(panel, a, "astra") | {"blind_id": "b-unknown"},
        result(panel, a, "fable", rubric_sha="0" * 64),
        result(panel, a, "fable", scores={"effect": 4, "guidance": 1, "tone": 1}),
        result(panel, a, "fable", scores={"effect": 1}),
        result(panel, a, "fable", turns=()),
        result(panel, a, "fable", verdict="maybe"),
    ]
    for row in bad:
        with pytest.raises(PanelError):
            panel.ingest([write(tmp_path / "r.jsonl", [row])])
    with pytest.raises(PanelError, match="second rating"):
        panel.ingest([write(tmp_path / "r.jsonl", [result(panel, a, "fable")] * 2)])
    # the adjudicator (astra) may rate; a judge outside the item is refused only when unassigned
    assert len(panel.ingest([write(tmp_path / "r.jsonl", [result(panel, a, "astra")])])) == 1


def test_adjudication_on_verdict_disagreement_or_gap_of_two(tmp_path):
    panel = Panel(rubric(), salt="s")
    its = items(batches=1, per_batch=3)
    agree, verdict_split, gap = panel.assign(its)
    ratings_rows = [
        result(panel, agree, "fable"), result(panel, agree, "opus", scores={"effect": 2, "guidance": 3, "tone": 3}),
        result(panel, verdict_split, "fable"), result(panel, verdict_split, "opus", verdict="fail"),
        result(panel, gap, "fable"), result(panel, gap, "opus", scores={"effect": 1, "guidance": 3, "tone": 3}),
    ]
    ratings = panel.ingest([write(tmp_path / "primary.jsonl", ratings_rows)])
    index = panel.adjudication_orders(its, ratings, tmp_path)
    assert sum(o["items"] for o in index["orders"]) == 2
    assert all(o["judge"] == "astra" for o in index["orders"])
    report = panel.report(ratings)
    assert len(report["pending"]) == 2 and any("adjudication" in b for b in report["blockers"])
    third = [result(panel, verdict_split, "astra", verdict="fail"),
             result(panel, gap, "astra", scores={"effect": 2, "guidance": 3, "tone": 3})]
    ratings += panel.ingest([write(tmp_path / "third.jsonl", third)])
    report = panel.report(ratings)
    finals = {f["blind_id"]: f for f in report["finals"]}
    assert not report["pending"]
    assert finals[verdict_split.blind_id]["verdict"] == "fail" and finals[verdict_split.blind_id]["adjudicated"]
    assert finals[gap.blind_id]["scores"]["effect"] == 2.0  # median of 3, 1, 2
    assert finals[agree.blind_id]["scores"]["effect"] == 2.5 and not finals[agree.blind_id]["adjudicated"]


def test_kappa_per_dimension_and_exclusion(tmp_path):
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-12
    assert cohen_kappa([], []) is None
    panel = Panel(rubric(), salt="s")
    assigned = panel.assign(items(batches=2, per_batch=10))
    rows = []
    for n, a in enumerate(assigned):
        first, second = a.judges
        effect = n % 4  # both judges agree on effect: κ = 1
        tone_a, tone_b = n % 2 * 3, (n + 1) % 2 * 3  # always disagree on tone: κ < 0.6
        verdict = "pass" if n % 3 else "fail"
        rows.append(result(panel, a, first, verdict, {"effect": effect, "guidance": 2, "tone": tone_a}))
        rows.append(result(panel, a, second, verdict, {"effect": effect, "guidance": 2, "tone": tone_b}))
    ratings = panel.ingest([write(tmp_path / "r.jsonl", rows)])
    third = [result(panel, a, a.adjudicator, "pass", {"effect": n % 4, "guidance": 2, "tone": 3})
             for n, a in enumerate(assigned)]
    ratings += panel.ingest([write(tmp_path / "t.jsonl", third)])
    report = panel.report(ratings, arm_thresholds={"verdict": 0.9, "effect": 1.0, "tone": 3.0})
    assert report["kappa"]["effect"] == 1.0 and report["kappa"]["tone"] < 0.6
    assert "effect" in report["gated_dimensions"] and "tone" in report["excluded_dimensions"]
    assert "guidance" in report["degenerate_kappa"]
    assert not any("tone" in b for b in report["blockers"])  # excluded dimensions never gate
    assert any("verdict pass rate" in b for b in report["blockers"])
    json.dumps(report)
