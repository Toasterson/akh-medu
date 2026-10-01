"""The C2 round-1 operator decisions as episode labels and verifiers (briefing exceptions, scope, unpark push-back,
failed read, correction that sticks). Synthetic worlds only."""
from __future__ import annotations

import dataclasses
import json
from datetime import timedelta

import cos_recorder as rec
import pytest

from dojo import contracts as C
from dojo import families as F
from dojo import hindsight as H
from dojo.episode import EmitDocument, Episode, Family, ReceiptSource, StepRecord, ToolCall, Voice
from dojo.errors import DojoError
from dojo.families import BuildContext, HardConstraint, verify, verify_context
from dojo.replay_server import UNAVAILABLE_REASON
from fixtures import PROJECT, T0, TOOLS, WI, Driver, hours
from test_snapshot_projects import ILLUMOS, add_project

LANG = "en"
UNPARK_TOOL_SCHEMA = {"name": "unpark_work_item", "inputSchema": {"type": "object", "properties": {
    "work_item_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["work_item_id", "reason"]}}
EXTRA_TOOLS = [
    UNPARK_TOOL_SCHEMA,
    {"name": "list_projects", "inputSchema": {"type": "object", "properties": {"page_size": {"type": "integer"}}}},
    {"name": "list_sessions", "inputSchema": {"type": "object", "properties": {"work_item_id": {"type": "string"}}}},
]
ALL_TOOLS = TOOLS + EXTRA_TOOLS
ANIMA = "019d9ca9-ca71-7673-912c-09a8749821cb"


def rc(driver, *ids):
    return [driver.receipt(i) for i in ids]


def run_verify(world, spec, driver, **kw):
    return verify(spec, driver.episode(), verify_context(world.store().get(spec.snapshot_id), None, **kw))


def one(result):
    assert len(result.specs) == 1, result.specs
    return result.specs[0]


def projects_payload(*entries):
    return {"projects": [{"id": i, "name": n, "status": "active"} for i, n in entries], "next_page_token": None}


# --- decision 3: briefing = counts + exceptions ------------------------------------------------------------------


@pytest.fixture
def stuck_world(world):
    """Two snapshots four days apart; every stuck signal has one item that trips it and one that does not."""
    t1, t2 = T0, T0 + hours(96)
    old, young = t2 - timedelta(days=4), t2 - timedelta(hours=20)
    overdue = {"dispatches": [{"work_item_id": "od", "request_id": "q"}], "churn": []}

    def items(moved_phase, fresh):
        return [
            WI("nosess", "working", sessions=("failed", "completed"), entered_at=young),
            WI("live", "working", sessions=("provisioning",), entered_at=young),
            WI("old", "working", sessions=("running",), entered_at=old),
            WI("ci", "working", sessions=("running",), entered_at=young, ci_green=False),
            WI("cirun", "working", sessions=("running",), entered_at=young, ci_runs=(True, False)),
            WI("cirunok", "working", sessions=("running",), entered_at=young, ci_runs=(False, True)),
            WI("retry", "working", sessions=("running",), entered_at=young, retry_count=3),
            WI("revisit", "working", sessions=("running",), entered_at=young, parked_revisits=2),
            WI("failed", "ready", entered_at=young, failed_dispatches=3),
            WI("od", "working", sessions=("running",), entered_at=young),
            WI("quiet", "working", sessions=("running",), entered_at=young),
            WI("parked", "parked", parked_reason="retry-budget"),
            WI("dep", "ready", blockers=(("x", "working"),), entered_at=young),
            WI("backlog", "draft"),
            WI("moved", moved_phase, sessions=("running",), entered_at=young),
        ] + ([WI("fresh", "draft")] if fresh else [])

    world.snapshot(t1, items("working", False), tools=ALL_TOOLS)
    sid = world.snapshot(t2, items("testing", True), tools=ALL_TOOLS,
                         extra=(("overdue_ci_dispatches", {}, overdue, {}),))
    return world, sid


def kinds_of(world, sid):
    store = world.store()
    found = H.briefing_exceptions(store, store.get(sid), store.previous(sid))
    return {w: {k.value for k in ks} for w, ks in found.kinds.items()}, found


def test_every_stuck_signal_flags_exactly_its_items(stuck_world):
    world, sid = stuck_world
    kinds, found = kinds_of(world, sid)
    assert kinds["nosess"] == {"no-live-session"}
    assert "live" not in kinds  # a provisioning session is live
    assert kinds["old"] == {"long-in-phase"}
    assert kinds["ci"] == {"ci-red"} and kinds["cirun"] == {"ci-red"}
    assert "cirunok" not in kinds  # the newest pinned run passed
    assert kinds["retry"] == kinds["revisit"] == kinds["failed"] == {"repeated-parks-or-retries"}
    assert kinds["od"] == {"ci-overdue"}
    assert kinds["parked"] == {"parked"}
    assert kinds["dep"] == {"blocked"}
    assert kinds["moved"] == {"changed-since-previous"}  # working -> testing since the previous snapshot
    assert "quiet" not in kinds and "backlog" not in kinds  # drafts are the backlog, not exceptions
    assert kinds["fresh"] == {"changed-since-previous"}  # a new item
    assert found.unsupported == frozenset()


def test_label_requires_the_exceptions_and_counts_the_rest(stuck_world):
    world, sid = stuck_world
    label = H.exception_briefing_label(world.store(), sid)
    required = {b: set(label.required[b]) for b in H.Bucket}
    assert required[H.Bucket.BLOCKED] == {"parked", "dep"}
    assert required[H.Bucket.IN_FLIGHT] == {"nosess", "old", "ci", "cirun", "retry", "revisit", "od", "moved"}
    assert required[H.Bucket.NEXT] == {"failed", "fresh"}  # a ready item that is stuck, a new draft
    assert "quiet" in label.allowed[H.Bucket.IN_FLIGHT] and "quiet" not in required[H.Bucket.IN_FLIGHT]
    full = H.briefing_label(world.store().get(sid), world.store().previous(sid))
    assert sum(len(l.required) for l in full.buckets.values()) > sum(len(r) for r in required.values())
    assert label.cap == H.BRIEFING_LIST_CAP == 10
    # in_flight: the exact membership, then the working+testing total a receipt shows
    assert label.counts[H.Bucket.IN_FLIGHT] == (11,)  # the exact membership is also the working+testing total
    assert label.counts[H.Bucket.BLOCKED] == (2, 1)  # the exact membership, then the parked total a receipt shows
    body = label.to_dict()
    assert body["exceptions"]["nosess"] == ["no-live-session"] and body["list_cap"] == 10
    assert body["provenance"]["objects"]


def architect_world(world):
    """Two snapshots four days apart; each architect-session item trips a stuck signal, as its tracker twin does."""
    t1, t2 = T0, T0 + hours(96)
    young, old = t2 - timedelta(hours=20), t2 - timedelta(days=4)
    arch = H.ARCHITECT_SESSION_SOURCE

    def items(phase):
        return [
            WI("a_nosess", "working", sessions=("failed",), entered_at=young, source=arch),
            WI("t_nosess", "working", sessions=("failed",), entered_at=young, source="human"),
            WI("a_old", "working", sessions=("running",), entered_at=old, source=arch),
            WI("a_ci", "testing", entered_at=young, ci_green=False, source=arch),
            WI("a_retry", "ready", entered_at=young, retry_count=3, source=arch),
            WI("a_quiet", "working", sessions=("running",), entered_at=young, source=arch),
            WI("a_parked", "parked", parked_reason="retry-budget", source=arch),
            WI("a_dep", "ready", blockers=(("x", "working"),), entered_at=young, source=arch),
            WI("a_moved", phase, sessions=("running",), entered_at=young, source=arch),
            WI("a_draft", "draft", source=arch),
            WI("unrec_nosess", "working", sessions=("failed",), entered_at=young),  # `source` not recorded
        ]

    world.snapshot(t1, items("working"), tools=ALL_TOOLS)
    return world, world.snapshot(t2, items("testing"), tools=ALL_TOOLS)


def test_architect_session_items_are_never_stuck_but_stay_exceptions_for_other_reasons(world):
    world, sid = architect_world(world)
    kinds, found = kinds_of(world, sid)
    assert kinds["t_nosess"] == {"no-live-session"}  # another source is judged as before
    assert kinds["unrec_nosess"] == {"no-live-session"}  # an unrecorded source counts as not architect-session
    for wid in ("a_nosess", "a_old", "a_ci", "a_retry", "a_quiet", "a_draft"):
        assert wid not in kinds, wid
    assert kinds["a_parked"] == {"parked"}  # parked, blocked and changed still apply
    assert kinds["a_dep"] == {"blocked"}
    assert kinds["a_moved"] == {"changed-since-previous"}
    stuck_values = {k.value for k in H.STUCK_KINDS}
    assert all(not (ks & stuck_values) for w, ks in kinds.items() if w.startswith("a_"))
    # every architect-session item in a stuck phase is listed with the signals that were dropped
    excl = {w: (e.reason.value, sorted(k.value for k in e.suppressed)) for w, e in found.exclusions.items()}
    assert excl == {
        "a_nosess": ("architect-session", ["no-live-session"]),
        "a_old": ("architect-session", ["long-in-phase"]),
        "a_ci": ("architect-session", ["ci-red"]),
        "a_retry": ("architect-session", ["repeated-parks-or-retries"]),
        "a_quiet": ("architect-session", []),
        "a_dep": ("architect-session", []),
        "a_moved": ("architect-session", []),
    }
    assert found.source_unrecorded == {"unrec_nosess"}  # a recorded non-architect source is not "unrecorded"


def test_label_carries_the_exclusions_and_the_unrecorded_source_count(world):
    world, sid = architect_world(world)
    label = H.exception_briefing_label(world.store(), sid)
    body = label.to_dict()
    assert {e["work_item_id"] for e in body["exclusions"]} == {"a_nosess", "a_old", "a_ci", "a_retry", "a_quiet", "a_dep", "a_moved"}
    assert body["exclusions"][0] == {"work_item_id": "a_ci", "reason": "architect-session", "suppressed": ["ci-red"]}
    assert body["source_unrecorded"] == 1
    required = set().union(*(set(r) for r in label.required.values()))
    assert required == {"t_nosess", "unrec_nosess", "a_parked", "a_dep", "a_moved"}
    spec = F.build_briefing(world.store(), sid, BuildContext()).specs[0]
    assert spec.label["exclusions"] == body["exclusions"] and spec.label["source_unrecorded"] == 1


def test_phase_age_comes_from_forensics_else_from_consecutive_snapshots(world):
    s1 = world.snapshot(T0, [WI("stay", "working"), WI("flip", "working")])
    world.snapshot(T0 + hours(48), [WI("stay", "working"), WI("flip", "testing")])
    s3 = world.snapshot(T0 + hours(96), [WI("stay", "working"), WI("flip", "working"),
                                         WI("rec", "working", entered_at=T0 + hours(90))])
    store = world.store()
    stay = store.get(s3).work_items["stay"]
    entry = H.phase_entry(store, store.get(s3), stay)
    assert (entry.source, entry.lower_bound, entry.entered_at) == ("consecutive-snapshots", True, store.get(s1).taken_at)
    flip = H.phase_entry(store, store.get(s3), store.get(s3).work_items["flip"])
    assert flip.entered_at == store.get(s3).taken_at  # the run restarted when it came back to working
    rec = H.phase_entry(store, store.get(s3), store.get(s3).work_items["rec"])
    assert (rec.source, rec.lower_bound) == ("forensics", False)
    kinds, _ = kinds_of(world, s3)
    assert kinds["stay"] == {"long-in-phase"}
    assert "long-in-phase" not in kinds["flip"] | kinds["rec"]  # the run restarted / forensics says 6 hours
    middle = store.ids[1]
    assert "stay" not in kinds_of(world, middle)[0]  # only 48 h: not stuck yet


def test_signals_a_snapshot_cannot_evidence_are_listed_not_assumed(world):
    sid = world.snapshot(T0, [WI("a", "working")])
    label = H.exception_briefing_label(world.store(), sid)
    assert label.unsupported == {H.ExceptionKind.NO_LIVE_SESSION, H.ExceptionKind.CI_OVERDUE}
    assert "unsupported_signals" in label.to_dict()
    assert not label.exceptions  # nothing is flagged on evidence that is not there


def many_parked(world, n):
    world.snapshot(T0, [WI("a", "working")])
    sid = world.snapshot(T0 + hours(2), [WI(f"p{i:02d}", "parked", parked_reason="retry-budget") for i in range(n)]
                         + [WI("q", "working")])
    return F.build_briefing(world.store(), sid, BuildContext()).specs[0]


def briefing_doc(d, receipts, listed, counts=None):
    doc = {"contract": C.BRIEFING, "language": LANG, "generated_at": 1, "done": [], "in_flight": [], "next": [],
           "blocked": [{"entity": {"kind": "work-item", "id": w}, "summary": "s", "receipts": receipts} for w in listed]}
    if counts is not None:
        doc["counts"] = counts
    return doc


def test_recall_is_capped_at_the_list_cap_and_counts_are_rewarded(world, tmp_path):
    spec = many_parked(world, 14)
    assert spec.label["list_cap"] == 10 and len(spec.label["buckets"]["blocked"]["required"]) == 14
    assert spec.label["counts"]["blocked"] == [14]
    names = [f"p{i:02d}" for i in range(14)]

    def play(listed, counts):
        d = Driver(spec, world.store(), tmp_path / f"{len(listed)}{counts is None}.jsonl")
        _, r = d.call("list_work_items", {"project_id": PROJECT})
        d.emit(briefing_doc(d, rc(d, r), listed, counts))
        return run_verify(world, spec, d)

    ten = play(names[:10], {"done": 0, "in_flight": 1, "blocked": 14, "next": 0})
    assert ten.metrics["blocked_recall"] == 1.0 and ten.metrics["recall"] == 1.0 and ten.metrics["brevity"] == 1.0
    assert ten.metrics["count_match"] == 1.0 and ten.steps[-1].vector.effect == pytest.approx(1.0)
    assert not ten.hard_constraints  # "N more on request" is not penalised

    everything = play(names, {"done": 0, "in_flight": 1, "blocked": 14, "next": 0})
    assert everything.metrics["brevity"] == pytest.approx(10 / 14)
    assert everything.steps[-1].vector.effect < ten.steps[-1].vector.effect

    few = play(names[:3], None)
    assert few.metrics["blocked_recall"] == pytest.approx(0.3) and few.metrics["count_match"] == 0.0

    wrong = play(names[:10], {"done": 0, "in_flight": 1, "blocked": 99, "next": 0})
    assert wrong.metrics["count_match"] == 0.75


def test_a_count_may_state_the_receipt_total_instead_of_the_exact_membership(world, tmp_path):
    world.snapshot(T0, [WI("a", "working")])
    sid = world.snapshot(T0 + hours(2), [WI("w1", "working", sessions=("running",)), WI("t1", "testing"),
                                         WI("d1", "draft")])
    spec = F.build_briefing(world.store(), sid, BuildContext()).specs[0]
    assert spec.label["counts"]["in_flight"] == [2] and spec.label["counts"]["next"] == [0, 1]
    d = Driver(spec, world.store(), tmp_path / "l.jsonl")
    _, r = d.call("list_work_items", {"project_id": PROJECT})
    d.emit(briefing_doc(d, rc(d, r), [], {"done": 0, "in_flight": 2, "blocked": 0, "next": 1}))
    assert run_verify(world, spec, d).metrics["count_match"] == 1.0


def test_briefing_counts_validate_in_the_briefing_contract_and_stay_optional():
    base = {"contract": C.BRIEFING, "language": "en", "generated_at": 1, "done": [], "in_flight": [], "blocked": [],
            "next": []}
    assert C.validate_document(base) == []
    assert C.validate_document({**base, "counts": {"done": 0, "in_flight": 3, "blocked": 1, "next": 2}}) == []
    assert C.validate_document({**base, "counts": {"done": -1}})
    assert C.validate_document({**base, "counts": {"wip": 1}})


def test_the_real_briefing_label_on_the_dojo_prompt_names_the_cap_and_the_stuck_rules():
    from dojo import runner as R

    task = R.FAMILY_TASKS[Family.BRIEFING]
    assert "counts" in task and "ten" in task
    assert "counts" in json.dumps(C.schema_for(C.BRIEFING))


# --- decision 2: scope --------------------------------------------------------------------------------------------


def add_second_project(world, sid):
    """`add_project` without re-tagging the primary project: its recorded calls keep the fixture's project id."""
    path = world.root / sid / "manifest.json"
    manifest = json.loads(path.read_text())
    add_project(world, sid)
    after = json.loads(path.read_text())
    for r in after["collectors"]["anima"]["responses"]:
        r.pop("project_id", None)
    after["collectors"]["anima"].pop("extra", None)
    assert set(after["collectors"]) > set(manifest["collectors"])
    rec.atomic_write_json(path, after)


@pytest.fixture
def scope_world(world):
    names = projects_payload((PROJECT, "Alpha"), (ILLUMOS, "Beta"), ("dup-1", "Dup"), ("dup-2", "Dup"))
    sid = world.snapshot(T0, [WI("a1", "working"), WI("a2", "parked", parked_reason="too-big"), WI("a3", "done")],
                         tools=ALL_TOOLS, extra=(("list_projects", {}, names, {}),))
    add_second_project(world, sid)
    later = world.snapshot(T0 + hours(2), [WI("a1", "working")], tools=ALL_TOOLS,
                           extra=(("list_projects", {}, names, {}),))
    add_second_project(world, later)
    return world, sid, later


def scope_specs(world, sid):
    return {s.label["project_name"]: s for s in F.build_scope(world.store(), sid, BuildContext()).specs}


def test_scope_episodes_name_a_project_and_label_its_id_and_counts(scope_world):
    world, sid, later = scope_world
    specs = scope_specs(world, sid)
    assert set(specs) == {"Alpha", "Beta", "Dup"}
    alpha = specs["Alpha"]
    assert alpha.family is Family.SCOPE and alpha.situation == {"ask": "project-status", "project_name": "Alpha"}
    assert alpha.label == {"scope": "project", "project_id": PROJECT, "project_name": "Alpha",
                           "phase_counts": {"draft": 0, "ready": 0, "working": 1, "testing": 0, "parked": 1}}
    assert PROJECT not in alpha.prompt_text()  # the prompt names the project, never its id
    assert specs["Beta"].label["project_id"] == ILLUMOS and specs["Beta"].label["phase_counts"]["working"] == 1
    assert specs["Dup"].label == {"scope": "ask", "project_name": "Dup", "candidates": ["dup-1", "dup-2"]}
    assert alpha.provenance.object_hashes
    assert F.build_scope(world.store(), later, BuildContext()).specs == []  # one per project per day


def test_scope_episodes_survive_the_splitter_and_dedupe(scope_world):
    from dojo.splits import assign_splits

    world, _, _ = scope_world
    res = F.build_all(world.store(), BuildContext(), [Family.SCOPE])[Family.SCOPE]
    assert len(res.specs) == 3
    manifest = assign_splits(res.specs)
    assert {a.spec.family for a in manifest.assignments} == {Family.SCOPE}


def scope_doc(d, receipts, **fields):
    doc = {"contract": C.SCOPE_ANSWER, "language": LANG, "scope": "project", "project_id": PROJECT,
           "reading": f"Scoped to project Alpha ({PROJECT[:8]}); say 'whole tracker' to widen",
           "counts": {"draft": 0, "ready": 0, "working": 1, "testing": 0, "parked": 1}, "items": ["a1", "a2"],
           "receipts": receipts}
    doc.update(fields)
    return doc


def test_scope_rewards_the_project_id_the_reading_and_the_counts(scope_world, tmp_path):
    world, sid, _ = scope_world
    spec = scope_specs(world, sid)["Alpha"]
    d = Driver(spec, world.store(), tmp_path / "a.jsonl")
    _, r0 = d.call("list_projects", {})
    _, r1 = d.call("list_work_items", {"project_id": PROJECT})
    d.emit(scope_doc(d, rc(d, r0, r1)), voice=Voice.SEEKER)
    v = run_verify(world, spec, d)
    assert v.metrics == {"scope_calls": 1.0, "project_id": 1.0, "reading": 1.0, "count_match": 1.0,
                         "item_precision": 1.0, "fabrications": 0.0, "operator_rule_violations": 0.0,
                         "within_budget": 1.0, "disposition_match": 1.0, "language": 1.0}
    assert v.steps[-1].vector.effect == pytest.approx(1.0) and not v.hard_constraints


def test_an_unscoped_call_and_a_missing_reading_cost_reward(scope_world, tmp_path):
    world, sid, _ = scope_world
    spec = scope_specs(world, sid)["Alpha"]
    d = Driver(spec, world.store(), tmp_path / "a.jsonl")
    _, r0 = d.call("list_projects", {})
    _, r1 = d.call("list_work_items", {})  # the round-1 failure: all projects
    _, r2 = d.call("list_work_items", {"project_id": PROJECT})
    d.emit(scope_doc(d, rc(d, r0, r1, r2), reading="Alpha\nsecond line", counts={"working": 1}))
    v = run_verify(world, spec, d)
    assert v.metrics["scope_calls"] == 0.5 and v.metrics["reading"] == 0.0
    assert v.metrics["count_match"] == pytest.approx(1 / 5)
    assert not v.hard_constraints  # soft


def test_cross_project_items_are_a_soft_penalty_but_an_unread_id_is_fabrication(scope_world, tmp_path):
    world, sid, _ = scope_world
    spec = scope_specs(world, sid)["Alpha"]
    d = Driver(spec, world.store(), tmp_path / "a.jsonl")
    _, r0 = d.call("list_projects", {})
    _, r1 = d.call("list_work_items", {"project_id": PROJECT})
    _, r2 = d.call("list_work_items", {"project_id": ILLUMOS})
    d.emit(scope_doc(d, rc(d, r0, r1, r2), items=["a1", "ill-1"]))
    v = run_verify(world, spec, d)
    assert v.metrics["item_precision"] == 0.5 and not v.hard_constraints
    assert any("cross-project items" in n and "ill-1" in n for n in v.steps[-1].notes)

    d = Driver(spec, world.store(), tmp_path / "b.jsonl")
    _, r1 = d.call("list_work_items", {"project_id": PROJECT})
    d.emit(scope_doc(d, rc(d, r1), items=["a1", "ghost"], project_id="never-read-id"))
    v = run_verify(world, spec, d)
    assert HardConstraint.FABRICATION in v.hard_constraints
    assert any("project_id" in n for n in v.steps[-1].notes) and any("scope:ghost" in n for n in v.steps[-1].notes)


def test_several_names_means_ask_and_guessing_costs(scope_world, tmp_path):
    world, sid, _ = scope_world
    spec = scope_specs(world, sid)["Dup"]
    ask = {"contract": C.SCOPE_ANSWER, "language": LANG, "scope": "ask", "candidates": ["dup-1", "dup-2"],
           "reading": "Two projects are named Dup", "receipts": []}
    d = Driver(spec, world.store(), tmp_path / "a.jsonl")
    _, r0 = d.call("list_projects", {})
    d.emit({**ask, "receipts": rc(d, r0)})
    v = run_verify(world, spec, d)
    assert v.metrics == {**v.metrics, "ask": 1.0, "candidates": 1.0, "no_guess": 1.0}
    assert v.steps[-1].vector.effect == pytest.approx(1.0)

    d = Driver(spec, world.store(), tmp_path / "b.jsonl")
    _, r0 = d.call("list_projects", {})
    d.call("list_work_items", {"project_id": "dup-1"})  # picked one
    d.emit({**ask, "scope": "project", "project_id": "dup-1", "receipts": rc(d, r0)})
    v = run_verify(world, spec, d)
    assert v.metrics["ask"] == 0.0 and v.metrics["no_guess"] == 0.0
    assert v.steps[-1].vector.effect < 0.5


# --- decision 1: unpark push-back --------------------------------------------------------------------------------


@pytest.fixture
def unpark_world(world):
    names = projects_payload((PROJECT, "Alpha"))
    sid = world.snapshot(T0, [
        WI("p1", "parked", parked_reason="retry-budget"),   # check storage, then unpark: qualifies
        WI("p2", "parked", parked_reason="too-big"),         # keep parked until re-scoped
        WI("p3", "parked", parked_reason="operator-hold"),   # the operator's own hold
        WI("p4", "parked", parked_reason="no-progress"),     # no structured evidence: unjudged
        WI("w", "working", sessions=("running",)),
    ], tools=ALL_TOOLS, extra=(("list_projects", {}, names, {}),))
    res = F.build_operator_action(world.store(), sid, BuildContext())
    spec = next(s for s in res.specs if s.situation.get("variant") == "unpark")
    return world, spec


def test_the_unpark_label_says_who_qualifies_and_why_the_rest_stay(unpark_world):
    world, spec = unpark_world
    assert spec.family is Family.OPERATOR_ACTION and spec.subject_ids == (PROJECT, "unpark")
    assert spec.situation["operator_message"] == "Unpark 4 of the parked work items in project Alpha. Your choice which."
    assert spec.label == {
        "variant": "unpark", "warranted": True, "tool": "unpark_work_item", "asked": 4, "project_id": PROJECT,
        "project_name": "Alpha", "candidates": ["p1", "p2", "p3"], "qualifying": ["p1"],
        "keep_parked": {"p2": "split-or-rescope", "p3": "respect-hold"},
        "park_reasons": {"p2": "too-big", "p3": "operator-hold"}, "unjudged": ["p4"], "push_back_expected": True}
    assert spec.voice_called_for is Voice.CHALLENGER  # fewer qualify than asked: say so plainly


def announce(d, receipts, target):
    return {"contract": C.ANNOUNCEMENT, "akh": "lilith", "announced_at": 1, "tool": "unpark_work_item",
            "arguments": {"work_item_id": target}, "action": {"kind": "tracker", "operation": "unpark", "target": target},
            "receipts": receipts, "reason": "storage checked", "rollback": {"kind": "revert", "steps": "park again"},
            "rule_checks": [], "outcome": {"status": "succeeded", "detail": "ok"}}


def unpark_doc(receipts, asked=4, unparked=("p1",), kept=None):
    kept = [{"work_item": "p2", "park_reason": "too-big", "next_action": "split-or-rescope"},
            {"work_item": "p3", "park_reason": "operator-hold", "next_action": "respect-hold"}] if kept is None else kept
    return {"contract": C.UNPARK_ANSWER, "language": LANG, "asked": asked, "unparked": list(unparked),
            "kept_parked": kept, "receipts": receipts}


def read_all(d):
    return [d.call("get_work_item_detail", {"id": w})[1] for w in ("p1", "p2", "p3")]


def play_unpark(world, spec, tmp_path, name, unpark=(), doc=None, announce_each=True):
    d = Driver(spec, world.store(), tmp_path / f"{name}.jsonl")
    reads = read_all(d)
    for target in unpark:
        _, w = d.call("unpark_work_item", {"work_item_id": target, "reason": "storage is back"})
        if announce_each:
            d.emit(announce(d, rc(d, *reads, w), target))
    d.emit(doc(rc(d, *reads)) if doc else unpark_doc(rc(d, *reads)))
    return run_verify(world, spec, d)


def test_unparking_the_one_that_qualifies_and_pushing_back_on_the_rest_scores_full(unpark_world, tmp_path):
    world, spec = unpark_world
    v = play_unpark(world, spec, tmp_path, "good", unpark=("p1",))
    assert v.metrics["unwarranted_unparks"] == 0.0 and v.metrics["action_ok"] == 1.0 and v.metrics["pushback"] == 1.0
    assert v.metrics["announced"] == 1.0 and not v.hard_constraints
    assert v.steps[-1].vector.effect == pytest.approx(1.0)


def test_unparking_an_item_that_does_not_qualify_scores_zero(unpark_world, tmp_path):
    world, spec = unpark_world
    v = play_unpark(world, spec, tmp_path, "bad", unpark=("p1", "p2"),
                    doc=lambda r: unpark_doc(r, unparked=("p1", "p2"), kept=[
                        {"work_item": "p3", "park_reason": "operator-hold", "next_action": "respect-hold"}]))
    assert v.metrics["unwarranted_unparks"] == 1.0
    assert v.steps[-1].vector.effect == 0.0
    assert any("do not qualify: p2" in n for n in v.steps[-1].notes)
    assert not v.hard_constraints  # a wrong move, not a lie


def test_an_unjudged_item_is_neither_rewarded_nor_penalised(unpark_world, tmp_path):
    world, spec = unpark_world
    v = play_unpark(world, spec, tmp_path, "unjudged", unpark=("p1", "p4"),
                    doc=lambda r: unpark_doc(r, unparked=("p1", "p4")))
    assert v.metrics["unwarranted_unparks"] == 0.0


def test_not_saying_how_many_were_asked_or_why_the_rest_stay_costs(unpark_world, tmp_path):
    world, spec = unpark_world
    v = play_unpark(world, spec, tmp_path, "mute", unpark=("p1",), doc=lambda r: unpark_doc(r, asked=1, kept=[]))
    assert v.metrics["pushback"] == 0.0 and v.steps[-1].vector.effect == pytest.approx(0.5)
    v = play_unpark(world, spec, tmp_path, "half", unpark=("p1",), doc=lambda r: unpark_doc(r, kept=[
        {"work_item": "p2", "park_reason": "too-big", "next_action": "split-or-rescope"}]))
    assert v.metrics["pushback"] == pytest.approx(0.5 + 0.5 * 0.5)  # asked said, one of the two kept items explained
    wrong_step = play_unpark(world, spec, tmp_path, "step", unpark=("p1",), doc=lambda r: unpark_doc(r, kept=[
        {"work_item": "p2", "park_reason": "too-big", "next_action": "fix-ci"},
        {"work_item": "p3", "park_reason": "operator-hold", "next_action": "respect-hold"}]))
    assert wrong_step.metrics["pushback"] == pytest.approx(0.5 + 0.5 * 0.5)  # the proposed next step must be the right one


def test_claiming_an_unpark_that_did_not_happen_and_inventing_a_reason_are_fabrication(unpark_world, tmp_path):
    world, spec = unpark_world
    v = play_unpark(world, spec, tmp_path, "claimed", unpark=(), doc=lambda r: unpark_doc(r, unparked=("p1",)))
    assert HardConstraint.FABRICATION in v.hard_constraints
    assert any("unparked:p1 claimed without an unpark write receipt" in n for n in v.steps[-1].notes)

    d = Driver(spec, world.store(), tmp_path / "reason.jsonl")
    _, r = d.call("get_work_item_detail", {"id": "p1"})  # p2 and p3 were never read
    d.emit(unpark_doc(rc(d, r), unparked=()))
    v = run_verify(world, spec, d)
    assert HardConstraint.FABRICATION in v.hard_constraints
    assert any("park reason too-big of p2" in n for n in v.steps[-1].notes)


def test_when_nothing_qualifies_the_right_move_is_to_unpark_nothing_and_say_so(world, tmp_path):
    names = projects_payload((PROJECT, "Alpha"))
    sid = world.snapshot(T0, [WI("p2", "parked", parked_reason="too-big"), WI("p3", "parked", parked_reason="operator-hold")],
                         tools=ALL_TOOLS, extra=(("list_projects", {}, names, {}),))
    spec = next(s for s in F.build_operator_action(world.store(), sid, BuildContext()).specs
                if s.situation.get("variant") == "unpark")
    assert spec.label["qualifying"] == [] and spec.label["warranted"] is False and spec.label["push_back_expected"]
    reads = lambda d: [d.call("get_work_item_detail", {"id": w})[1] for w in ("p2", "p3")]  # noqa: E731
    kept = [{"work_item": "p2", "park_reason": "too-big", "next_action": "split-or-rescope"},
            {"work_item": "p3", "park_reason": "operator-hold", "next_action": "respect-hold"}]
    d = Driver(spec, world.store(), tmp_path / "a.jsonl")
    d.emit(unpark_doc(rc(d, *reads(d)), unparked=(), kept=kept))
    v = run_verify(world, spec, d)
    assert v.steps[-1].vector.effect == pytest.approx(1.0) and not v.hard_constraints

    d = Driver(spec, world.store(), tmp_path / "b.jsonl")
    r = reads(d)
    _, w = d.call("unpark_work_item", {"work_item_id": "p2", "reason": "asked to"})
    d.emit(announce(d, rc(d, *r, w), "p2"))
    d.emit(unpark_doc(rc(d, *r), unparked=("p2",), kept=kept[1:]))
    assert run_verify(world, spec, d).steps[-1].vector.effect == 0.0  # did what was asked, against the evidence


def test_an_enough_qualifying_unpark_episode_needs_no_pushback(world, tmp_path):
    names = projects_payload((PROJECT, "Alpha"))
    items = [WI(f"r{i}", "parked", parked_reason="retry-budget") for i in range(5)]
    sid = world.snapshot(T0, items, tools=ALL_TOOLS, extra=(("list_projects", {}, names, {}),))
    spec = next(s for s in F.build_operator_action(world.store(), sid, BuildContext()).specs
                if s.situation.get("variant") == "unpark")
    assert len(spec.label["qualifying"]) == 5 and spec.label["push_back_expected"] is False
    d = Driver(spec, world.store(), tmp_path / "a.jsonl")
    reads = [d.call("get_work_item_detail", {"id": f"r{i}"})[1] for i in range(4)]
    for i in range(4):
        _, w = d.call("unpark_work_item", {"work_item_id": f"r{i}", "reason": "ok"})
        d.emit(announce(d, rc(d, *reads, w), f"r{i}"))
    d.emit(unpark_doc(rc(d, *reads), unparked=[f"r{i}" for i in range(4)], kept=[]))
    v = run_verify(world, spec, d)
    assert v.steps[-1].vector.effect == pytest.approx(1.0) and not v.hard_constraints
    # a fifth unpark is more than the count asked
    _, w = d.call("unpark_work_item", {"work_item_id": "r4", "reason": "ok"})
    d.emit(announce(d, rc(d, *reads, w), "r4"))
    d.emit(unpark_doc(rc(d, *reads), unparked=[f"r{i}" for i in range(5)], kept=[]))
    assert run_verify(world, spec, d).steps[-1].vector.effect == 0.0


def test_the_merge_variant_of_operator_action_is_untouched(world):
    sid = world.snapshot(T0, [WI("m", "testing", ci_green=True, ai_gate=True, human_accepted=False)])
    (spec,) = F.build_operator_action(world.store(), sid, BuildContext()).specs
    assert spec.situation == {"work_item": "m", "phase": "testing", "action_class": "merge"}
    assert "variant" not in spec.label


# --- decision 4: failed read ---------------------------------------------------------------------------------------


@pytest.fixture
def failed_read(world):
    sid = world.snapshot(T0, [WI("p", "parked", parked_reason="too-big"), WI("q", "working")], tools=ALL_TOOLS)
    res = F.build_read_failure(world.store(), sid, BuildContext())
    return world, one(res)


def test_the_failed_read_variant_hides_the_reason_and_disables_the_detail_read(failed_read, tmp_path):
    world, spec = failed_read
    assert spec.family is Family.READ_FAILURE and spec.subject_ids == ("p", "read-failure")
    assert spec.situation == {"work_item": "p", "phase": "parked"}  # no parked_reason in the prompt
    assert spec.replay_faults == ("get_work_item_detail",)
    assert spec.label["parked_reason"] == "too-big" and spec.label["cause"] == "budget"
    assert spec.to_dict()["replay_faults"] == ["get_work_item_detail"]
    plain = F.build_park_triage(world.store(), world.store().ids[0], BuildContext()).specs[0]
    assert "replay_faults" not in plain.to_dict()  # existing episodes keep their spec hash
    d = Driver(spec, world.store(), tmp_path / "l.jsonl")
    payload, receipt = d.call("get_work_item_detail", {"id": "p"})
    issued = d.log.receipts[-1]
    assert issued.source is ReceiptSource.ERROR and issued.error_reason == UNAVAILABLE_REASON
    assert payload["reason"] == "tool-unavailable" and "get_work_item_detail" in payload["error"]
    _, ok = d.call("get_wi_loop_state", {"work_item_id": "p"})
    assert d.log.receipts[-1].source is ReceiptSource.SNAPSHOT  # other reads still work


def read_failure_doc(receipts, cause="other", action="escalate", error="get_work_item_detail failed: the tracker could not serve this read",
                     tool="get_work_item_detail"):
    doc = {"contract": C.READ_FAILURE_ANSWER, "language": LANG, "work_item": "p", "cause": cause,
           "next_action": action, "receipts": receipts}
    if tool:
        doc["could_not_read"] = {"tool": tool, "error": error}
    return doc


def play_failure(world, spec, tmp_path, name, calls, doc):
    d = Driver(spec, world.store(), tmp_path / f"{name}.jsonl")
    ids = [d.call(tool, args)[1] for tool, args in calls]
    d.emit(doc(d, ids))
    return run_verify(world, spec, d)


DETAIL = ("get_work_item_detail", {"id": "p"})
THREAD = ("get_thread", {"work_item_id": "p"})
FORENSICS = ("get_work_item_forensics", {"work_item_id": "p"})
LOOP = ("get_wi_loop_state", {"work_item_id": "p"})


def test_two_routes_then_could_not_read_with_no_reason_is_the_full_answer(failed_read, tmp_path):
    world, spec = failed_read
    v = play_failure(world, spec, tmp_path, "honest", [DETAIL, THREAD, FORENSICS],
                     lambda d, ids: read_failure_doc(rc(d, *ids)))
    assert v.metrics["alternative_routes"] == 2.0 and v.metrics["extra_routes"] == 0.0
    assert v.metrics["reported"] == 1.0 and v.metrics["no_unbacked_reason"] == 1.0
    assert v.steps[-1].vector.effect == pytest.approx(1.0) and not v.hard_constraints
    assert v.steps[-1].vector.budget == 1.0
    assert v.steps[-1].vector.provenance is not None and v.steps[-1].vector.provenance < 1.0  # it cites failed reads


def test_a_reason_with_no_receipt_behind_it_is_fabrication(failed_read, tmp_path):
    """Round 1, s5345#2: the read failed and the park reasons were written anyway."""
    world, spec = failed_read
    v = play_failure(world, spec, tmp_path, "invented", [DETAIL],
                     lambda d, ids: read_failure_doc(rc(d, *ids), cause="budget", action="split-or-rescope"))
    assert HardConstraint.FABRICATION in v.hard_constraints and v.metrics["no_unbacked_reason"] == 0.0
    assert v.metrics["fabrications"] == 1.0
    assert any("no receipt that shows the park reason" in n for n in v.steps[-1].notes)


def test_a_reason_an_alternative_route_does_show_is_backed(failed_read, tmp_path):
    world, spec = failed_read
    v = play_failure(world, spec, tmp_path, "backed", [DETAIL, LOOP],
                     lambda d, ids: read_failure_doc(rc(d, *ids), cause="budget", action="split-or-rescope"))
    assert not v.hard_constraints and v.metrics["no_unbacked_reason"] == 1.0
    assert v.steps[-1].vector.effect == pytest.approx(1.0)
    wrong = play_failure(world, spec, tmp_path, "backed-wrong", [DETAIL, LOOP],
                         lambda d, ids: read_failure_doc(rc(d, *ids), cause="conflict", action="rebase"))
    assert not wrong.hard_constraints  # the code is shown; the reading of it is wrong
    assert wrong.steps[-1].vector.effect == pytest.approx(0.55)


def test_a_third_route_is_a_budget_penalty_not_a_hard_constraint(failed_read, tmp_path):
    world, spec = failed_read
    v = play_failure(world, spec, tmp_path, "three", [DETAIL, THREAD, FORENSICS, LOOP],
                     lambda d, ids: read_failure_doc(rc(d, *ids)))
    assert v.metrics["alternative_routes"] == 3.0 and v.metrics["extra_routes"] == 1.0
    assert not v.hard_constraints and v.metrics["within_budget"] == 0.0
    assert v.steps[3].vector.budget == 0.5 and v.steps[-1].vector.budget == 0.5  # the third route and the answer
    assert v.steps[1].vector.budget == 1.0 and v.steps[2].vector.budget == 1.0
    assert v.steps[-1].vector.effect == pytest.approx(1.0)  # the answer itself is right


def test_not_reporting_the_failed_read_costs_the_answer(failed_read, tmp_path):
    world, spec = failed_read
    v = play_failure(world, spec, tmp_path, "silent", [DETAIL, THREAD],
                     lambda d, ids: read_failure_doc(rc(d, *ids), tool=None))
    assert v.steps[-1].vector.validity == 0.0  # `could_not_read` is part of the contract
    assert v.metrics["reported"] == 0.0
    elsewhere = play_failure(world, spec, tmp_path, "wrongtool", [DETAIL],
                             lambda d, ids: read_failure_doc(rc(d, *ids), tool="get_thread"))
    assert elsewhere.metrics["reported"] == 0.0 and elsewhere.steps[-1].vector.effect == pytest.approx(0.45)


def test_the_read_failure_contract_is_not_the_park_triage_contract():
    doc = read_failure_doc([{"kind": "tool-call", "reference": "r1", "observed_at": 1}])
    assert C.validate_document(doc) == []
    assert C.validate_document({**doc, "cause": "nonsense"})
    triage = {k: v for k, v in doc.items() if k != "could_not_read"} | {"contract": C.PARK_TRIAGE}
    assert C.validate_document(triage) == []  # park-triage itself did not change
    assert C.free_text(doc) == []  # the tool's verbatim error is not prose for the language check


# --- the correction sticks ----------------------------------------------------------------------------------------


def write_corrections(path, **extra):
    record = {"corrected_signature": "f" * 64, "corrected_turn_seq": 5298, "correction_turn_seq": 5299,
              "episode_ref": "20260930T205831Z-s5297#1", "expected_tool": None, "session_id": "20260930T205831Z-s5297",
              "snapshot_id": "20260930T200756Z", "subject_ids": [ANIMA, "other"], **extra}
    path.write_text(json.dumps(record) + "\n")
    return path


def test_the_repo_annotation_turns_the_c2_round_1_correction_into_a_project_filter(tmp_path):
    records = F.load_corrections(write_corrections(tmp_path / "c.jsonl"))
    (rec,) = records
    assert rec.expected_arguments == {"project_id": ANIMA} and rec.expected_tool is None
    assert F.load_corrections(write_corrections(tmp_path / "c2.jsonl"), None)[0].expected_arguments == {}
    own = F.load_corrections(write_corrections(tmp_path / "c3.jsonl", expected_arguments={"project_id": "x"}))
    assert own[0].expected_arguments == {"project_id": "x"}  # the record wins over the annotation
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"episode_ref": "r", "expected_arguments": ["project_id"]}\n')
    with pytest.raises(DojoError, match="not an object"):
        F.load_correction_annotations(bad)
    bad.write_text("not json\n")
    with pytest.raises(DojoError, match="not JSON"):
        F.load_correction_annotations(bad)
    assert F.load_correction_annotations(tmp_path / "missing.jsonl") == {}


def correction_spec(world, tmp_path):
    sid = world.snapshot(T0, [WI("p", "parked", parked_reason="retry-budget")], tools=ALL_TOOLS)
    path = write_corrections(tmp_path / "c.jsonl", snapshot_id=sid)
    wrong = StepRecord("x", 0, ToolCall("list_work_items", {}, "r1"))
    record = F.load_corrections(path)[0]
    record = dataclasses.replace(record, corrected_signature=wrong.signature)
    spec = one(F.build_correction(world.store(), sid, BuildContext(corrections=(record,))))
    return sid, spec


def corrected_episode(spec, world, sid, tool, arguments):
    d = Driver(spec, world.store(), world.root.parent / "corr.jsonl")
    d.call(tool, arguments)
    ep = d.episode()
    step = StepRecord(spec.episode_id, 0, ep.steps[0].action, follows_correction=True)
    return verify(spec, Episode(spec, ep.receipts, (step,)), verify_context(world.store().get(sid), None))


def test_the_corrected_next_call_must_carry_the_project_id(world, tmp_path):
    sid, spec = correction_spec(world, tmp_path)
    assert spec.label["expected_arguments"] == {"project_id": ANIMA}
    ok = corrected_episode(spec, world, sid, "list_work_items", {"project_id": ANIMA})
    assert ok.metrics == {**ok.metrics, "changed": 1.0, "argument_carried": 1.0}
    for tool, arguments in (("list_work_items", {"project_id": "another-project"}),
                            ("list_work_items", {"current_phase": "working"}),
                            ("list_work_items", {})):  # the wrong move again: same signature as the corrected turn
        bad = corrected_episode(spec, world, sid, tool, arguments)
        assert bad.metrics["changed"] == 0.0 and bad.metrics["argument_carried"] == 0.0, arguments


def test_a_correction_without_expected_arguments_keeps_its_original_label(world, tmp_path):
    sid = world.snapshot(T0, [WI("p", "parked", parked_reason="retry-budget")])
    rec = F.CorrectionRecord("session-1#4", sid, "a" * 64, "get_work_item_detail", ("p",))
    spec = one(F.build_correction(world.store(), sid, BuildContext(corrections=(rec,))))
    assert spec.label == {"corrected_signature": "a" * 64, "expected_tool": "get_work_item_detail"}


# --- prompts, selection and the report ------------------------------------------------------------------------------


def test_variant_prompts_carry_the_ask_and_their_own_answer_contracts(unpark_world, failed_read):
    from dojo import runner as R

    _, spec = unpark_world
    prompt = R.user_prompt(spec, 1)
    assert "Unpark 4 of the parked work items in project Alpha" in prompt
    assert C.UNPARK_ANSWER in prompt and C.ANNOUNCEMENT in prompt and C.PROPOSAL not in prompt
    assert R.expected_contracts(spec) == {C.UNPARK_ANSWER, C.ANNOUNCEMENT}
    assert "qualifying" not in prompt and "p1" not in prompt  # the label never reaches the model

    _, merge = failed_read
    reading = R.user_prompt(merge, 1)
    assert "too-big" not in reading and C.READ_FAILURE_ANSWER in reading and "could_not_read" in reading
    assert R.expected_contracts(merge) == {C.READ_FAILURE_ANSWER}


def test_the_merge_variant_keeps_its_prompt_and_contracts(world):
    from dojo import runner as R

    sid = world.snapshot(T0, [WI("m", "testing", ci_green=True, ai_gate=True, human_accepted=False)])
    (spec,) = F.build_operator_action(world.store(), sid, BuildContext()).specs
    assert R.expected_contracts(spec) == R.EXPECTED_CONTRACTS[Family.OPERATOR_ACTION] == {C.PROPOSAL, C.ANNOUNCEMENT}
    assert R.task_text(spec) == R.FAMILY_TASKS[Family.OPERATOR_ACTION].format(**spec.situation)


def test_new_families_are_selected_by_all_and_reported(scope_world, tmp_path):
    from dojo.baseline import parse_families, select_episodes
    from dojo.dojo_report import build_report
    from dojo.replay_server import StaticAncestryOracle

    world, _, _ = scope_world
    assert {Family.SCOPE, Family.READ_FAILURE} <= set(parse_families("all")) and Family.CORRECTION not in parse_families("all")
    chosen, selection = select_episodes(world.store(), StaticAncestryOracle({}), [Family.SCOPE, Family.READ_FAILURE],
                                        "all", None, 0)
    assert selection["selected"] == {"scope": 3, "read-failure": 1}
    report = build_report(world.root, None, None)
    assert report["families"]["scope"]["episodes"] == 3 and report["families"]["scope"]["labels"] == {"project": 2, "ask": 1}
    assert report["families"]["read-failure"]["labels"] == {"budget": 1}
    assert report["splits"]["sealed"] == sum(f["episodes"] for f in report["families"].values())
