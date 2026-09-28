import subprocess

import pytest

from dojo import hindsight as H
from dojo.contracts import LandedVerdict, NextAction, ParkCause
from dojo.replay_server import GitAncestryOracle, StaticAncestryOracle
from fixtures import MAIN_1, T0, WI, hours


def test_briefing_buckets_from_phase_transitions(world):
    s1 = world.snapshot(T0, [WI("a", "working"), WI("b", "working"), WI("c", "ready"), WI("d", "draft")])
    s2 = world.snapshot(T0 + hours(2), [
        WI("a", "done"),
        WI("b", "working", blockers=(("x", "ready"),)),
        WI("c", "ready"),
        WI("d", "draft"),
        WI("e", "parked", parked_reason="retry-budget"),
        WI("f", "testing"),
        WI("g", "draft", pending_deps=("x",)),
    ])
    store = world.store()
    label = H.briefing_label(store.get(s2), store.get(s1))
    b = label.buckets
    assert b[H.Bucket.DONE].required == {"a"}
    assert b[H.Bucket.BLOCKED].required == {"b", "e"}
    assert b[H.Bucket.BLOCKED].allowed == {"b", "e", "g"}
    assert b[H.Bucket.IN_FLIGHT].required == {"f"}
    assert b[H.Bucket.NEXT].required == {"c"} and b[H.Bucket.NEXT].allowed == {"c", "d"}
    assert set(label.provenance.snapshot_ids) == {s1, s2}
    assert label.provenance.object_hashes


def test_stale_labels_need_later_confirmation(world):
    old = T0 - hours(120)
    s1 = world.snapshot(T0, [WI("old", "working", updated_at=old), WI("new", "working", updated_at=T0 - hours(1)),
                             WI("moves", "working", updated_at=old)])
    store = world.store()
    labels, excluded = H.stale_labels(store, s1)
    assert {e.subject for e in excluded} == {"old", "moves"}
    assert [(l.work_item_id, l.verdict) for l in labels] == [("new", H.Staleness.FRESH)]
    world.snapshot(T0 + hours(30), [WI("old", "working", updated_at=old), WI("new", "working"),
                                    WI("moves", "testing", updated_at=T0)])
    labels, excluded = H.stale_labels(world.store(), s1)
    verdicts = {l.work_item_id: l.verdict for l in labels}
    assert verdicts == {"old": H.Staleness.STALE, "new": H.Staleness.FRESH, "moves": H.Staleness.FRESH}
    assert not excluded


@pytest.mark.parametrize("wi,cause,action", [
    (WI("p", "parked", parked_reason="retry-budget"), ParkCause.RETRY_S3, NextAction.CHECK_STORAGE_THEN_UNPARK),
    (WI("p", "parked", parked_reason="rebase-conflict"), ParkCause.CONFLICT, NextAction.REBASE),
    (WI("p", "parked", parked_reason="operator-hold"), ParkCause.OTHER, NextAction.RESPECT_HOLD),
    (WI("p", "parked", parked_reason="too-big"), ParkCause.BUDGET, NextAction.SPLIT_OR_RESCOPE),
    (WI("p", "parked", parked_reason="ci-infra-failing", ci_green=False), ParkCause.CI_RED, NextAction.FIX_CI),
    (WI("p", "parked", parked_reason="no-progress", ci_green=False), ParkCause.CI_RED, NextAction.FIX_CI),
    (WI("p", "parked", parked_reason="no-progress", blockers=(("q", "working"),)), ParkCause.DEPENDENCY,
     NextAction.CHASE_BLOCKER),
    (WI("p", "parked", parked_reason="gate-needs-human"), ParkCause.REVIEW_GATE, NextAction.ADDRESS_REVIEW),
])
def test_park_cause_classes(world, wi, cause, action):
    sid = world.snapshot(T0, [wi])
    (label,), excluded = H.park_labels(world.store(), sid)
    assert (label.cause, label.next_action) == (cause, action)
    assert not excluded


@pytest.mark.parametrize("wi", [
    WI("p", "parked", parked_reason="ci-infra-failing", ci_green=True),  # code contradicted by the loop state
    WI("p", "parked", parked_reason="no-progress"),  # symptom without structured evidence
    WI("p", "parked", parked_reason="no-progress", ci_green=False, blockers=(("q", "working"),)),  # conflicting
    WI("p", "parked", parked_reason="brand-new-code"),
])
def test_ambiguous_parks_are_excluded_and_listed(world, wi):
    sid = world.snapshot(T0, [wi])
    labels, excluded = H.park_labels(world.store(), sid)
    assert labels == [] and [e.subject for e in excluded] == ["p"]
    assert excluded[0].provenance.snapshot_ids == (sid,)


def test_park_resolution_comes_from_a_later_snapshot(world):
    s1 = world.snapshot(T0, [WI("p", "parked", parked_reason="retry-budget")])
    s2 = world.snapshot(T0 + hours(2), [WI("p", "parked", parked_reason="retry-budget")])
    s3 = world.snapshot(T0 + hours(4), [WI("p", "working")])
    (label,), _ = H.park_labels(world.store(), s1)
    assert label.resolution == H.ParkResolution("working", s3)
    assert set(label.provenance.snapshot_ids) == {s1, s3}
    assert s2 not in label.provenance.snapshot_ids


def test_landed_verdicts_use_the_ancestry_oracle(world):
    head_in, head_out = "a" * 40, "b" * 40
    sid = world.snapshot(T0, [WI("t1", "testing", prs=(("c1", head_in),)), WI("t2", "testing", prs=(("c2", head_out),)),
                              WI("t3", "testing", prs=(("c3", "c" * 40),)), WI("w", "working", prs=(("c4", head_in),))],
                         main_sha=MAIN_1)
    oracle = StaticAncestryOracle({(head_in, MAIN_1): True, (head_out, MAIN_1): False})
    labels, excluded = H.landed_labels(world.store(), sid, oracle)
    assert {(l.work_item_id, l.verdict) for l in labels} == {("t1", LandedVerdict.LANDED), ("t2", LandedVerdict.NOT_LANDED)}
    assert [e.subject for e in excluded] == ["t3"]
    assert all(l.main_sha == MAIN_1 for l in labels)


def test_git_ancestry_oracle_runs_merge_base(tmp_path):
    repo = tmp_path / "repo"
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True).stdout.strip()
    repo.mkdir()
    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "one")
    first = run("rev-parse", "HEAD")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "two")
    second = run("rev-parse", "HEAD")
    oracle = GitAncestryOracle(repo)
    assert oracle.is_ancestor(first, second) is True
    assert oracle.is_ancestor(second, first) is False
    assert oracle.is_ancestor("e" * 40, second) is None


def test_commitment_labels(world):
    s1 = world.snapshot(T0, [WI("c", "working"), WI("b", "working"), WI("r", "working")])
    s2 = world.snapshot(T0 + hours(20), [WI("c", "done"), WI("b", "parked", parked_reason="retry-budget"), WI("r", "working")])
    store = world.store()
    far = int((T0 + hours(24 * 7)).timestamp())
    near = int((T0 + hours(24)).timestamp())
    closed = H.commitment_label(store, "c", s1, s2, far)
    assert closed.expected_state is H.CommitmentState.CLOSED and closed.closed_snapshot == s2
    assert H.commitment_label(store, "b", s1, s2, far).expected_state is H.CommitmentState.BLOCKED
    assert H.commitment_label(store, "r", s1, s2, far).expected_state is H.CommitmentState.OPEN
    at_risk = H.commitment_label(store, "r", s1, s2, near)
    assert at_risk.expected_state is H.CommitmentState.AT_RISK and at_risk.must_surface
    assert isinstance(H.commitment_label(store, "ghost", s1, s2, far), H.Excluded)


def test_later_outcomes(world):
    s1 = world.snapshot(T0, [WI("x", "ready"), WI("y", "ready", revisions=1), WI("z", "ready"), WI("o", "ready")])
    world.snapshot(T0 + hours(2), [WI("x", "cancelled", status="cancelled"), WI("y", "working", revisions=2),
                                   WI("z", "done"), WI("o", "ready")])
    store = world.store()
    assert H.later_outcome(store, "x", s1).outcome is H.Outcome.CANCELLED
    assert H.later_outcome(store, "y", s1).outcome is H.Outcome.REWORKED
    assert H.later_outcome(store, "z", s1).outcome is H.Outcome.CLEAN
    assert H.later_outcome(store, "o", s1).outcome is H.Outcome.OPEN
