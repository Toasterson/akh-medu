"""Hindsight labels: ground truth for snapshot t_k from t_{k-1}, t_k and later t_{k+n}.

Every label carries `Provenance` (snapshot ids + object hashes). Labels that
the evidence does not settle are not guessed: they are returned in an
`excluded` list with the reason, so the report can show them.

Park causes come from Anima's typed `parked_reason` wire codes
(`anima-core ParkedReason::as_str` / `ParkReason::as_str`) plus the structured
loop state; these are machine enum values, not natural language.
"""
from __future__ import annotations

import dataclasses
import enum
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from dojo.contracts import LandedVerdict, NextAction, ParkCause
from dojo.replay_server import AncestryOracle
from dojo.snapshot import (
    IN_FLIGHT_PHASES,
    PHASE_CANCELLED,
    PHASE_DONE,
    PHASE_DRAFT,
    PHASE_PARKED,
    PHASE_READY,
    Provenance,
    Snapshot,
    SnapshotStore,
    WorkItemState,
)


@dataclasses.dataclass(frozen=True)
class Excluded:
    subject: str
    reason: str
    provenance: Provenance

    def to_dict(self) -> dict:
        return {"subject": self.subject, "reason": self.reason, "provenance": self.provenance.to_dict()}


# --- phase transitions --------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PhaseTransition:
    work_item_id: str
    from_phase: str
    to_phase: str
    from_snapshot: str
    to_snapshot: str
    provenance: Provenance


def transitions(before: Snapshot, after: Snapshot) -> list[PhaseTransition]:
    out = []
    old = before.work_items
    for wid, new in sorted(after.work_items.items()):
        prev = old.get(wid)
        if prev is not None and prev.phase != new.phase:
            out.append(PhaseTransition(wid, prev.phase, new.phase, before.id, after.id,
                                       prev.provenance.merge(new.provenance)))
    return out


# --- briefing buckets -----------------------------------------------------------------


class Bucket(enum.StrEnum):
    DONE = "done"
    IN_FLIGHT = "in_flight"
    BLOCKED = "blocked"
    NEXT = "next"


@dataclasses.dataclass(frozen=True)
class BucketLabel:
    required: frozenset[str]  # recall is measured against these
    allowed: frozenset[str]  # precision is measured against these (superset of required)


@dataclasses.dataclass(frozen=True)
class BriefingLabel:
    snapshot_id: str
    previous_snapshot_id: str | None
    buckets: Mapping[Bucket, BucketLabel]
    provenance: Provenance

    def to_dict(self) -> dict:
        return {
            "snapshot": self.snapshot_id,
            "previous": self.previous_snapshot_id,
            "buckets": {b.value: {"required": sorted(l.required), "allowed": sorted(l.allowed)}
                        for b, l in self.buckets.items()},
            "provenance": self.provenance.to_dict(),
        }


def _is_blocked(wi: WorkItemState, snap: Snapshot) -> bool:
    if wi.phase == PHASE_PARKED:
        return True
    if wi.phase in IN_FLIGHT_PHASES or wi.phase == PHASE_READY:
        loop = snap.loop_state(wi.id)
        if loop is not None and loop.unsatisfied_blockers:
            return True
        return bool(wi.pending_dependencies)
    return False


def briefing_label(snap: Snapshot, previous: Snapshot | None) -> BriefingLabel:
    """Done since the previous snapshot / In flight / Blocked / Next at t_k.

    - done: moved to `done` between t_{k-1} and t_k (required); nothing else allowed.
    - blocked: parked, or ready/working/testing with an unsatisfied dependency (required);
      drafts with a pending dependency are also allowed.
    - in_flight: working/testing and not blocked.
    - next: ready and not blocked (required); unblocked drafts are allowed.
    """
    items = snap.work_items
    done_req: set[str] = set()
    prov = Provenance()
    if previous is not None:
        for t in transitions(previous, snap):
            if t.to_phase == PHASE_DONE:
                done_req.add(t.work_item_id)
                prov = prov.merge(t.provenance)
    blocked_req, blocked_ok, flight, next_req, next_ok = set(), set(), set(), set(), set()
    for wid, wi in items.items():
        if wi.is_terminal:
            continue
        blocked = _is_blocked(wi, snap)
        if blocked:
            blocked_req.add(wid)
        elif wi.phase in IN_FLIGHT_PHASES:
            flight.add(wid)
        elif wi.phase == PHASE_READY:
            next_req.add(wid)
        elif wi.phase == PHASE_DRAFT:
            (blocked_ok if wi.pending_dependencies else next_ok).add(wid)
        else:
            continue
        prov = prov.merge(wi.provenance)
    buckets = {
        Bucket.DONE: BucketLabel(frozenset(done_req), frozenset(done_req)),
        Bucket.IN_FLIGHT: BucketLabel(frozenset(flight), frozenset(flight)),
        Bucket.BLOCKED: BucketLabel(frozenset(blocked_req), frozenset(blocked_req | blocked_ok)),
        Bucket.NEXT: BucketLabel(frozenset(next_req), frozenset(next_req | next_ok)),
    }
    return BriefingLabel(snap.id, previous.id if previous else None, buckets, prov)


# --- stale items ----------------------------------------------------------------------------


class Staleness(enum.StrEnum):
    STALE = "stale"
    FRESH = "fresh"


@dataclasses.dataclass(frozen=True)
class StaleLabel:
    work_item_id: str
    verdict: Staleness
    age_hours: float
    provenance: Provenance


def stale_labels(store: SnapshotStore, snapshot_id: str, *, stale_after: timedelta = timedelta(hours=48),
                 confirm_for: timedelta = timedelta(hours=24)) -> tuple[list[StaleLabel], list[Excluded]]:
    """A live ready/working/testing WI is STALE at t_k when its last update is older than
    `stale_after` *and* later snapshots covering at least `confirm_for` show no phase change.
    It is FRESH when it was updated recently or moved within the horizon. Old items without
    enough later coverage are excluded (not confirmed either way)."""
    snap = store.get(snapshot_id)
    later = store.later(snapshot_id)
    covered_until = later[-1].taken_at if later else snap.taken_at
    labels: list[StaleLabel] = []
    excluded: list[Excluded] = []
    for wid, wi in sorted(snap.work_items.items()):
        if wi.phase not in IN_FLIGHT_PHASES | {PHASE_READY} or wi.updated_at is None:
            continue
        age = snap.taken_at - wi.updated_at
        prov = wi.provenance
        moved = None
        for other in later:
            nxt = other.work_items.get(wid)
            if nxt is not None and nxt.phase != wi.phase:
                moved = other
                prov = prov.merge(nxt.provenance)
                break
        hours = age.total_seconds() / 3600
        if age < stale_after or moved is not None:
            labels.append(StaleLabel(wid, Staleness.FRESH, hours, prov))
        elif covered_until - snap.taken_at >= confirm_for:
            labels.append(StaleLabel(wid, Staleness.STALE, hours, prov.merge(*(
                o.work_items[wid].provenance for o in later if wid in o.work_items))))
        else:
            excluded.append(Excluded(wid, "old but later snapshots do not cover the confirmation window", prov))
    return labels, excluded


# --- park causes ------------------------------------------------------------------------------

# Anima `parked_reason` wire codes -> cause class. Codes absent here fall back to
# structured evidence; conflicting evidence makes the case ambiguous.
PARK_CODE_CLASS: Mapping[str, ParkCause] = {
    "retry-budget": ParkCause.RETRY_S3,
    "gate-needs-human": ParkCause.REVIEW_GATE,
    "ci-config-unresolvable": ParkCause.CI_RED,
    "ci-infra-failing": ParkCause.CI_RED,
    "ci-toolchain-broken": ParkCause.CI_RED,
    "rebase-conflict": ParkCause.CONFLICT,
    "too-big": ParkCause.BUDGET,
    "revisit-budget-exhausted": ParkCause.BUDGET,
    "operator-hold": ParkCause.OTHER,
    "human-parked": ParkCause.OTHER,
    "human-override": ParkCause.OTHER,
    "agent-refusal": ParkCause.OTHER,
    "no-captured-diff": ParkCause.OTHER,
}
HOLD_CODES = frozenset({"operator-hold", "human-parked", "human-override"})
# Codes that name a symptom, not a cause: the structured evidence decides.
EVIDENCE_CODES = frozenset({"no-progress"})

CAUSE_NEXT_ACTION: Mapping[ParkCause, NextAction] = {
    ParkCause.RETRY_S3: NextAction.CHECK_STORAGE_THEN_UNPARK,
    ParkCause.REVIEW_GATE: NextAction.ADDRESS_REVIEW,
    ParkCause.CI_RED: NextAction.FIX_CI,
    ParkCause.DEPENDENCY: NextAction.CHASE_BLOCKER,
    ParkCause.CONFLICT: NextAction.REBASE,
    ParkCause.BUDGET: NextAction.SPLIT_OR_RESCOPE,
    ParkCause.OTHER: NextAction.ESCALATE,
}


@dataclasses.dataclass(frozen=True)
class ParkResolution:
    to_phase: str
    snapshot_id: str


@dataclasses.dataclass(frozen=True)
class ParkLabel:
    work_item_id: str
    parked_reason: str | None
    cause: ParkCause
    next_action: NextAction
    resolution: ParkResolution | None  # None: still parked at the end of the window
    provenance: Provenance

    def to_dict(self) -> dict:
        return {
            "work_item": self.work_item_id,
            "parked_reason": self.parked_reason,
            "cause": self.cause.value,
            "next_action": self.next_action.value,
            "resolution": dataclasses.asdict(self.resolution) if self.resolution else None,
            "provenance": self.provenance.to_dict(),
        }


def _evidence_classes(snap: Snapshot, wi: WorkItemState) -> tuple[set[ParkCause], Provenance]:
    classes: set[ParkCause] = set()
    prov = wi.provenance
    loop = snap.loop_state(wi.id)
    if loop is not None:
        prov = prov.merge(Provenance.of(loop.ref))
        if loop.unsatisfied_blockers:
            classes.add(ParkCause.DEPENDENCY)
        if loop.ci_green_for_head is False:
            classes.add(ParkCause.CI_RED)
        if loop.review_required and loop.review_approved is False:
            classes.add(ParkCause.REVIEW_GATE)
    elif wi.pending_dependencies:
        classes.add(ParkCause.DEPENDENCY)
    return classes, prov


def classify_park(snap: Snapshot, wi: WorkItemState) -> tuple[ParkCause | None, NextAction | None, Provenance, str | None]:
    """(cause, next action, provenance, ambiguity reason)."""
    code = wi.parked_reason
    if code is None:
        loop = snap.loop_state(wi.id)
        code = loop.parked_reason if loop else None
    evidence, prov = _evidence_classes(snap, wi)
    if code in PARK_CODE_CLASS:
        cause = PARK_CODE_CLASS[code]
        conflicting = evidence - {cause}
        # A typed code wins unless the loop state positively contradicts a failure-class
        # code (e.g. `ci-*` while CI is green for the head is not settled evidence).
        if cause is ParkCause.CI_RED:
            loop = snap.loop_state(wi.id)
            if loop is not None and loop.ci_green_for_head is True:
                return None, None, prov, f"code {code} but CI green for the current head"
        if cause is not ParkCause.OTHER and ParkCause.DEPENDENCY in conflicting and cause is not ParkCause.DEPENDENCY:
            return None, None, prov, f"code {code} and unsatisfied dependencies both present"
        action = NextAction.RESPECT_HOLD if code in HOLD_CODES else CAUSE_NEXT_ACTION[cause]
        return cause, action, prov, None
    if code in EVIDENCE_CODES or code is None:
        if len(evidence) == 1:
            (cause,) = evidence
            return cause, CAUSE_NEXT_ACTION[cause], prov, None
        why = "no structured evidence" if not evidence else f"conflicting evidence {sorted(c.value for c in evidence)}"
        return None, None, prov, f"code {code}: {why}"
    return None, None, prov, f"unknown parked_reason code {code!r}"


def park_labels(store: SnapshotStore, snapshot_id: str) -> tuple[list[ParkLabel], list[Excluded]]:
    snap = store.get(snapshot_id)
    later = store.later(snapshot_id)
    labels: list[ParkLabel] = []
    excluded: list[Excluded] = []
    for wid, wi in sorted(snap.work_items.items()):
        if wi.phase != PHASE_PARKED:
            continue
        cause, action, prov, ambiguity = classify_park(snap, wi)
        if cause is None or action is None:
            excluded.append(Excluded(wid, ambiguity or "ambiguous", prov))
            continue
        resolution = None
        for other in later:
            nxt = other.work_items.get(wid)
            if nxt is not None and nxt.phase != PHASE_PARKED:
                resolution = ParkResolution(nxt.phase, other.id)
                prov = prov.merge(nxt.provenance)
                break
        labels.append(ParkLabel(wid, wi.parked_reason, cause, action, resolution, prov))
    return labels, excluded


# --- landed claims ----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LandedLabel:
    work_item_id: str
    commit: str
    main_sha: str
    verdict: LandedVerdict
    phase: str
    provenance: Provenance

    def to_dict(self) -> dict:
        return {"work_item": self.work_item_id, "commit": self.commit, "main_sha": self.main_sha,
                "verdict": self.verdict.value, "phase": self.phase, "provenance": self.provenance.to_dict()}


CLAIM_PHASES = frozenset({"testing", PHASE_DONE})


def landed_labels(store: SnapshotStore, snapshot_id: str, oracle: AncestryOracle | None) -> tuple[list[LandedLabel], list[Excluded]]:
    """Every WI whose tracker state implies "this landed" (testing or done) with a recorded PR
    head: verdict = `merge-base --is-ancestor head main@t_k` in a local clone. Heads the clone
    cannot resolve are excluded, never guessed."""
    snap = store.get(snapshot_id)
    main = snap.main_sha()
    labels: list[LandedLabel] = []
    excluded: list[Excluded] = []
    if main is None:
        return labels, [Excluded(snapshot_id, "no refs/heads/main recorded", Provenance())]
    main_sha, main_ref = main
    for wid, wi in sorted(snap.work_items.items()):
        if wi.phase not in CLAIM_PHASES:
            continue
        source = store.latest_detail_at_or_before(wid, snapshot_id)
        heads = source.change_heads(wid) if source else []
        if not heads:
            continue
        head = max(heads, key=lambda h: (h.lifecycle_state == "merged", h.change_id))
        prov = wi.provenance.merge(Provenance.of(head.ref, main_ref))
        verdict = oracle.is_ancestor(head.head_sha, main_sha) if oracle else None
        if verdict is None:
            excluded.append(Excluded(wid, f"ancestry of {head.head_sha[:12]} undecidable locally", prov))
            continue
        labels.append(LandedLabel(wid, head.head_sha, main_sha,
                                  LandedVerdict.LANDED if verdict else LandedVerdict.NOT_LANDED, wi.phase, prov))
    return labels, excluded


# --- commitments ------------------------------------------------------------------------------


class CommitmentState(enum.StrEnum):
    OPEN = "open"
    AT_RISK = "at-risk"
    BLOCKED = "blocked"
    CLOSED = "closed"
    DROPPED = "dropped"


@dataclasses.dataclass(frozen=True)
class CommitmentLabel:
    work_item_id: str
    created_snapshot: str
    evaluated_snapshot: str
    due_at: int
    expected_state: CommitmentState
    must_surface: bool  # due (or blocked) and not closed: she has to raise it
    closed_snapshot: str | None
    provenance: Provenance

    def to_dict(self) -> dict:
        return dataclasses.asdict(self) | {"expected_state": self.expected_state.value,
                                           "provenance": self.provenance.to_dict()}


def commitment_label(store: SnapshotStore, wid: str, created: str, evaluated: str, due_at: int,
                     at_risk_within: timedelta = timedelta(hours=12)) -> CommitmentLabel | Excluded:
    """"Make sure <wid> is done by <due_at>", created at `created`, evaluated at `evaluated`.
    Closing evidence = the WI reached `done` in a snapshot <= evaluated."""
    ids = store.ids
    window = [store.get(i) for i in ids if created <= i <= evaluated]
    prov = Provenance()
    closed = None
    last: WorkItemState | None = None
    for snap in window:
        wi = snap.work_items.get(wid)
        if wi is None:
            continue
        last = wi
        prov = prov.merge(wi.provenance)
        if wi.phase == PHASE_DONE and closed is None:
            closed = snap.id
    if last is None:
        return Excluded(wid, "work item absent from the commitment window", prov)
    eval_time = store.get(evaluated).unix_time
    if closed is not None:
        state = CommitmentState.CLOSED
    elif last.phase == PHASE_CANCELLED or last.status == PHASE_CANCELLED:
        state = CommitmentState.DROPPED
    elif last.phase == PHASE_PARKED or last.pending_dependencies:
        state = CommitmentState.BLOCKED
    elif eval_time >= due_at - int(at_risk_within.total_seconds()):
        state = CommitmentState.AT_RISK
    else:
        state = CommitmentState.OPEN
    must_surface = state in (CommitmentState.BLOCKED, CommitmentState.AT_RISK, CommitmentState.DROPPED)
    return CommitmentLabel(wid, created, evaluated, due_at, state, must_surface, closed, prov)


# --- later outcomes (illumos guidance, PM/PO steering) -----------------------------------------------


class Outcome(enum.StrEnum):
    CLEAN = "clean"  # reached done without a new change revision after t_k
    REWORKED = "reworked"  # new change revisions after t_k
    FAILED = "failed"  # parked on a failure-class code after t_k
    CANCELLED = "cancelled"
    OPEN = "open"  # nothing settled inside the window


@dataclasses.dataclass(frozen=True)
class OutcomeLabel:
    work_item_id: str
    outcome: Outcome
    settled_snapshot: str | None
    provenance: Provenance

    @property
    def materialised_problem(self) -> bool:
        return self.outcome in (Outcome.REWORKED, Outcome.FAILED, Outcome.CANCELLED)


FAILURE_CAUSES = frozenset({ParkCause.CI_RED, ParkCause.CONFLICT, ParkCause.BUDGET, ParkCause.RETRY_S3})


def later_outcome(store: SnapshotStore, wid: str, snapshot_id: str) -> OutcomeLabel:
    snap = store.get(snapshot_id)
    start = snap.work_items.get(wid)
    prov = start.provenance if start else Provenance()
    base_revisions = snap.change_revision_count(wid)
    for other in store.later(snapshot_id):
        wi = other.work_items.get(wid)
        if wi is None:
            continue
        prov = prov.merge(wi.provenance)
        if wi.phase == PHASE_CANCELLED or wi.status == PHASE_CANCELLED:
            return OutcomeLabel(wid, Outcome.CANCELLED, other.id, prov)
        if wi.phase == PHASE_PARKED and PARK_CODE_CLASS.get(wi.parked_reason or "") in FAILURE_CAUSES:
            return OutcomeLabel(wid, Outcome.FAILED, other.id, prov)
        revisions = other.change_revision_count(wid)
        if base_revisions is not None and revisions is not None and revisions > base_revisions:
            return OutcomeLabel(wid, Outcome.REWORKED, other.id, prov)
        if wi.phase == PHASE_DONE:
            return OutcomeLabel(wid, Outcome.CLEAN, other.id, prov)
    return OutcomeLabel(wid, Outcome.OPEN, None, prov)


def phase_changes_in_window(store: SnapshotStore) -> list[PhaseTransition]:
    snaps = store.snapshots()
    out: list[PhaseTransition] = []
    for a, b in zip(snaps, snaps[1:]):
        out.extend(transitions(a, b))
    return out


def iter_pairs(store: SnapshotStore) -> Iterable[tuple[Snapshot | None, Snapshot]]:
    prev = None
    for snap in store.snapshots():
        yield prev, snap
        prev = snap


def hours_between(a: datetime, b: datetime) -> float:
    return abs((b - a).total_seconds()) / 3600
