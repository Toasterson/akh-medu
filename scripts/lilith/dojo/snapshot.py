"""Read-only loader for `lilith/cos-snapshot@1` roots (the C3 recorder's output).

Layout (see `scripts/lilith/README-recorder.md`):

    <root>/<YYYYMMDDTHHMMSSZ>/manifest.json
    <root>/objects/<sha256>.json      canonical JSON, content-addressed, shared

Nothing here writes to the root. Every object read is re-hashed and compared
with its file name, so a label or a replayed answer can always be traced back
to exact recorded bytes (`ObjectRef`).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Any, Iterator, Mapping

from dojo.errors import ObjectIntegrityError, SnapshotFormatError, UnknownSnapshotError

SCHEMA = "lilith/cos-snapshot@1"
PRIMARY_ANIMA_COLLECTOR = "anima"
EXTRA_PROJECT_PREFIX = "anima:"
DEFAULT_PROJECT_ID = "019d9844-ce74-7993-a9f0-c953b8f04a4e"  # Akh-Medu
TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
MAIN_REF = "refs/heads/main"

# Anima phases (wire strings from `list_work_items` / `get_work_item_detail`).
PHASE_DONE = "done"
PHASE_CANCELLED = "cancelled"
PHASE_PARKED = "parked"
PHASE_DRAFT = "draft"
PHASE_READY = "ready"
PHASE_WORKING = "working"
PHASE_TESTING = "testing"
IN_FLIGHT_PHASES = frozenset({PHASE_WORKING, PHASE_TESTING})
TERMINAL_PHASES = frozenset({PHASE_DONE, PHASE_CANCELLED})


def canonical_json(payload: Any) -> bytes:
    """Byte-identical to `cos_recorder.canonical_json` (tested)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_args(args: Mapping[str, Any]) -> str:
    """Argument identity used for replay lookup: canonical JSON, `None` values dropped."""
    return canonical_json({k: v for k, v in args.items() if v is not None}).decode("utf-8")


def parse_snapshot_time(snapshot_id: str) -> datetime:
    try:
        return datetime.strptime(snapshot_id, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        raise SnapshotFormatError(f"snapshot id {snapshot_id!r} is not a {TIMESTAMP_FORMAT} UTC timestamp") from None


def parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# --- provenance ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True, order=True)
class ObjectRef:
    """One recorded response: which snapshot, which call, which bytes."""

    snapshot_id: str
    call: str
    args: str  # canonical_args(...)
    sha256: str

    def to_dict(self) -> dict:
        return {"snapshot": self.snapshot_id, "call": self.call, "args": json.loads(self.args), "sha256": self.sha256}


@dataclasses.dataclass(frozen=True)
class Provenance:
    """Every label carries the snapshot ids and object hashes it was derived from."""

    refs: tuple[ObjectRef, ...] = ()

    @staticmethod
    def of(*refs: ObjectRef | None) -> "Provenance":
        return Provenance(tuple(sorted({r for r in refs if r is not None})))

    def merge(self, *others: "Provenance") -> "Provenance":
        merged = set(self.refs)
        for other in others:
            merged.update(other.refs)
        return Provenance(tuple(sorted(merged)))

    @property
    def snapshot_ids(self) -> tuple[str, ...]:
        return tuple(sorted({r.snapshot_id for r in self.refs}))

    @property
    def object_hashes(self) -> tuple[str, ...]:
        return tuple(sorted({r.sha256 for r in self.refs}))

    def to_dict(self) -> dict:
        return {"snapshots": list(self.snapshot_ids), "objects": list(self.object_hashes)}


# --- typed views ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RecordedResponse:
    collector: str
    call: str
    args: Mapping[str, Any]
    sha256: str
    # Additive since multi-project capture; absent in older snapshots (all Akh-Medu).
    project_id: str | None = None
    # Recorder proof that a token-less listing was not truncated (see `cos_recorder.ResponseRecord`).
    complete: bool | None = None


@dataclasses.dataclass(frozen=True)
class WorkItemState:
    """What one snapshot says about one WI. Detail-derived fields are `None` when the
    recorder did not fetch detail (it only does so for live items)."""

    id: str
    code: str
    title: str
    plan_id: str | None
    phase: str
    status: str
    priority: str | None
    list_ref: ObjectRef
    detail_ref: ObjectRef | None = None
    parked_reason: str | None = None
    executor: str | None = None
    updated_at: datetime | None = None
    depends_on: tuple[str, ...] = ()
    pending_dependencies: tuple[str, ...] = ()
    # `get_work_item_detail` `work_item.source` (what created or drives the item, e.g. `architect-session`).
    # `None` = not recorded: no detail read, or the tracker holds no value. The listing does not carry it.
    source: str | None = None

    @property
    def is_live(self) -> bool:
        return self.phase not in TERMINAL_PHASES and self.status != PHASE_CANCELLED

    @property
    def is_terminal(self) -> bool:
        return not self.is_live

    @property
    def provenance(self) -> Provenance:
        return Provenance.of(self.list_ref, self.detail_ref)


@dataclasses.dataclass(frozen=True)
class LoopState:
    ci_green_for_head: bool | None
    review_required: bool | None
    review_approved: bool | None
    ai_gate_cleared: bool | None
    human_review_accepted: bool | None
    unsatisfied_blockers: tuple[tuple[str, str], ...]  # (work_item_id, phase)
    parked_reason: str | None
    ref: ObjectRef


@dataclasses.dataclass(frozen=True)
class ChangeHead:
    """A change (PR) head the tracker recorded for a WI."""

    work_item_id: str
    change_id: str
    head_sha: str
    lifecycle_state: str | None
    integration_decision: str | None
    ref: ObjectRef


def _walk_pages(payload: Any) -> list:
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return value
    return []


class Snapshot:
    """One `<root>/<ts>/manifest.json` plus access to the shared object store."""

    def __init__(self, root: Path, snapshot_id: str):
        self.root = root
        self.id = snapshot_id
        self.taken_at = parse_snapshot_time(snapshot_id)
        path = root / snapshot_id / "manifest.json"
        try:
            raw = path.read_bytes()
            self.manifest: dict = json.loads(raw)
        except (OSError, ValueError) as error:
            raise SnapshotFormatError(f"cannot read {path}: {error}") from None
        if self.manifest.get("schema") != SCHEMA:
            raise SnapshotFormatError(f"{path} has schema {self.manifest.get('schema')!r}, expected {SCHEMA!r}")
        self.manifest_sha256 = sha256_hex(raw)
        self._object_cache: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"Snapshot({self.id})"

    @property
    def unix_time(self) -> int:
        return int(self.taken_at.timestamp())

    # -- raw access ------------------------------------------------------------

    def collector(self, name: str) -> dict:
        return self.manifest.get("collectors", {}).get(name, {})

    @cached_property
    def responses(self) -> tuple[RecordedResponse, ...]:
        out = []
        for cname, cresult in sorted(self.manifest.get("collectors", {}).items()):
            for r in cresult.get("responses", []):
                out.append(RecordedResponse(cname, r["call"], r.get("args") or {}, r["sha256"], r.get("project_id"),
                                              r.get("complete")))
        return tuple(out)

    @cached_property
    def _index(self) -> dict[tuple[str, str], RecordedResponse]:
        index: dict[tuple[str, str], RecordedResponse] = {}
        for r in self.responses:
            index.setdefault((r.call, canonical_args(r.args)), r)
        return index

    def load_object(self, sha256: str) -> Any:
        if sha256 in self._object_cache:
            return self._object_cache[sha256]
        path = self.root / "objects" / f"{sha256}.json"
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ObjectIntegrityError(f"object {sha256} referenced by {self.id} is missing: {error}") from None
        if sha256_hex(data) != sha256:
            raise ObjectIntegrityError(f"object {path} does not hash to its name")
        payload = json.loads(data)
        self._object_cache[sha256] = payload
        return payload

    def ref(self, response: RecordedResponse) -> ObjectRef:
        return ObjectRef(self.id, response.call, canonical_args(response.args), response.sha256)

    def find(self, call: str, args: Mapping[str, Any]) -> RecordedResponse | None:
        return self._index.get((call, canonical_args(args)))

    def by_call(self, call: str, collector: str | None = None) -> Iterator[RecordedResponse]:
        """Responses for `call`. Without `collector`, only the primary (Akh-Medu) Anima
        collector and the non-Anima ones: extra-project collectors are named
        `anima:<project_id>` and are read explicitly via `collector=` so their work items
        never leak into the Akh-Medu families."""
        if collector is not None:
            return (r for r in self.responses if r.call == call and r.collector == collector)
        return (r for r in self.responses if r.call == call and not r.collector.startswith(EXTRA_PROJECT_PREFIX))

    @cached_property
    def project_collectors(self) -> dict[str, str]:
        """project id -> Anima collector name, for every project this snapshot captured."""
        out: dict[str, str] = {}
        for cname, cresult in sorted(self.manifest.get("collectors", {}).items()):
            if cname == PRIMARY_ANIMA_COLLECTOR:
                pid = (cresult.get("extra") or {}).get("project_id") or DEFAULT_PROJECT_ID
            elif cname.startswith(EXTRA_PROJECT_PREFIX):
                pid = (cresult.get("extra") or {}).get("project_id") or cname[len(EXTRA_PROJECT_PREFIX):]
            else:
                continue
            out[pid] = cname
        return out

    def answer(self, call: str, args: Mapping[str, Any]) -> tuple[Any, ObjectRef] | None:
        response = self.find(call, args)
        if response is None:
            return None
        return self.load_object(response.sha256), self.ref(response)

    def tool_contract(self, collector: str = "anima") -> tuple[list[dict], str] | None:
        sha = self.collector(collector).get("tool_contract_sha256")
        if not sha:
            return None
        payload = self.load_object(sha)
        return list(payload.get("tools", [])), sha

    # -- typed views --------------------------------------------------------------

    def detail(self, wid: str) -> tuple[dict, ObjectRef] | None:
        return self.answer("get_work_item_detail", {"id": wid})

    def thread(self, wid: str) -> tuple[Any, ObjectRef] | None:
        return self.answer("get_thread", {"work_item_id": wid})

    def scheduling(self, wid: str) -> tuple[dict, ObjectRef] | None:
        return self.answer("get_scheduling_state", {"work_item_id": wid})

    def sessions(self, wid: str) -> tuple[dict, ObjectRef] | None:
        """`list_sessions` for one work item (recorded for live items of the primary project only)."""
        return self.answer("list_sessions", {"work_item_id": wid})

    def forensics(self, wid: str) -> tuple[dict, ObjectRef] | None:
        """`get_work_item_forensics`: the phase-transition and dispatch-attempt event log."""
        return self.answer("get_work_item_forensics", {"work_item_id": wid})

    def overdue_ci(self) -> tuple[dict, ObjectRef] | None:
        """The global `overdue_ci_dispatches` read, when the snapshot recorded it."""
        return self.answer("overdue_ci_dispatches", {})

    def loop_state(self, wid: str) -> LoopState | None:
        found = self.answer("get_wi_loop_state", {"work_item_id": wid})
        if found is None:
            return None
        payload, ref = found
        gates = payload.get("pipeline_gates") or {}
        ci = gates.get("ci") or {}
        review = gates.get("review") or {}
        exit_bar = payload.get("exit_bar") or {}
        blockers = tuple(
            (str(b.get("work_item_id")), str(b.get("current_phase")))
            for b in payload.get("unsatisfied_blockers") or []
            if isinstance(b, dict) and b.get("work_item_id")
        )
        return LoopState(
            ci_green_for_head=ci.get("green_for_current_head") if ci.get("current_head") else None,
            review_required=review.get("required"),
            review_approved=review.get("approved_for_current_head"),
            ai_gate_cleared=exit_bar.get("ai_gate_cleared"),
            human_review_accepted=exit_bar.get("human_review_accepted"),
            unsatisfied_blockers=blockers,
            parked_reason=payload.get("parked_reason"),
            ref=ref,
        )

    @cached_property
    def work_items(self) -> dict[str, WorkItemState]:
        return self.work_items_of()

    def work_items_of(self, collector: str | None = None) -> dict[str, WorkItemState]:
        """Work items of one Anima collector (`None` = the primary Akh-Medu one)."""
        items: dict[str, WorkItemState] = {}
        for response in self.by_call("list_work_items", collector):
            ref = self.ref(response)
            for wi in _walk_pages(self.load_object(response.sha256)):
                if not isinstance(wi, dict) or not wi.get("id"):
                    continue
                items[wi["id"]] = self._state(wi, ref)
        return items

    def _state(self, listed: dict, list_ref: ObjectRef) -> WorkItemState:
        wid = listed["id"]
        found = self.detail(wid)
        base = dict(
            id=wid,
            code=str(listed.get("code") or ""),
            title=str(listed.get("title") or ""),
            plan_id=listed.get("plan_id"),
            phase=str(listed.get("current_phase") or ""),
            status=str(listed.get("status") or ""),
            priority=listed.get("priority"),
            list_ref=list_ref,
        )
        if found is None:
            return WorkItemState(**base)
        detail, detail_ref = found
        wi = detail.get("work_item") or {}
        pending = tuple(
            str(e.get("depends_on_id"))
            for e in wi.get("dependency_edges") or []
            if isinstance(e, dict) and e.get("state") == "pending" and e.get("depends_on_id")
        )
        return WorkItemState(
            **base,
            detail_ref=detail_ref,
            parked_reason=wi.get("parked_reason"),
            executor=wi.get("executor") or None,
            updated_at=parse_iso(wi.get("updated_at")),
            depends_on=tuple(str(x) for x in wi.get("depends_on_ids") or []),
            pending_dependencies=pending,
            source=wi.get("source") or None,
        )

    def change_heads(self, wid: str) -> list[ChangeHead]:
        found = self.detail(wid)
        if found is None:
            return []
        detail, ref = found
        heads = []
        for pr in detail.get("prs") or []:
            if isinstance(pr, dict) and pr.get("head_sha"):
                heads.append(
                    ChangeHead(
                        work_item_id=wid,
                        change_id=str(pr.get("change_id") or ""),
                        head_sha=str(pr["head_sha"]),
                        lifecycle_state=pr.get("lifecycle_state"),
                        integration_decision=pr.get("integration_decision"),
                        ref=ref,
                    )
                )
        return heads

    def change_revision_count(self, wid: str) -> int | None:
        found = self.detail(wid)
        if found is None:
            return None
        return len(found[0].get("change_history") or [])

    @cached_property
    def git_refs(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for response in self.by_call("git_ls_remote"):
            payload = self.load_object(response.sha256)
            for entry in payload.get("refs", []):
                out[str(entry.get("ref"))] = str(entry.get("sha"))
        return out

    def main_sha(self) -> tuple[str, ObjectRef] | None:
        for response in self.by_call("git_ls_remote"):
            payload = self.load_object(response.sha256)
            for entry in payload.get("refs", []):
                if entry.get("ref") == MAIN_REF:
                    return str(entry["sha"]), self.ref(response)
        return None

    def plans(self, collector: str | None = None) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for response in self.by_call("list_plans", collector):
            for plan in _walk_pages(self.load_object(response.sha256)):
                if isinstance(plan, dict) and plan.get("id"):
                    out[plan["id"]] = plan
        return out


class SnapshotStore:
    """All snapshots under one root, ordered by capture time."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise SnapshotFormatError(f"snapshot root {self.root} does not exist")
        ids = []
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and (child / "manifest.json").is_file():
                try:
                    parse_snapshot_time(child.name)
                except SnapshotFormatError:
                    continue
                ids.append(child.name)
        self._ids = tuple(ids)
        self._cache: dict[str, Snapshot] = {}

    @property
    def ids(self) -> tuple[str, ...]:
        return self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def get(self, snapshot_id: str) -> Snapshot:
        if snapshot_id not in self._ids:
            raise UnknownSnapshotError(f"no snapshot {snapshot_id!r} under {self.root}")
        if snapshot_id not in self._cache:
            self._cache[snapshot_id] = Snapshot(self.root, snapshot_id)
        return self._cache[snapshot_id]

    def snapshots(self) -> list[Snapshot]:
        return [self.get(i) for i in self._ids]

    def previous(self, snapshot_id: str) -> Snapshot | None:
        index = self._ids.index(snapshot_id)
        return self.get(self._ids[index - 1]) if index > 0 else None

    def later(self, snapshot_id: str) -> list[Snapshot]:
        index = self._ids.index(snapshot_id)
        return [self.get(i) for i in self._ids[index + 1 :]]

    def latest_detail_at_or_before(self, wid: str, snapshot_id: str) -> Snapshot | None:
        """Newest snapshot <= `snapshot_id` that recorded detail for `wid`."""
        index = self._ids.index(snapshot_id)
        for sid in reversed(self._ids[: index + 1]):
            snap = self.get(sid)
            if snap.detail(wid) is not None:
                return snap
        return None
