"""Derived Anima reads for the replay server (plan slice C4, ADR 116 Decision 3).

The recorder does not capture every *call* the tracker can be asked; it captures the *data*: per
project one paginated `list_work_items` walk (50 per page, cursor-linked), one `list_plans`, and
a detail/loop-state/thread record per live work item (layout: `scripts/lilith/README-recorder.md`,
collectors `anima` and `anima:<project_id>`). A model that asks for the same data with another
spelling (`list_work_items {}` , `{"page_size": 100, "project_id": ...}`, a `current_phase` or
`plan_id` filter, `get_work_item_by_id`, `get_plan`, `list_projects`) used to get an error although
the recorded objects answer it exactly. This module answers those calls from the recorded objects
and only from them:

- `list_work_items`  filter + paginate over the recorded, cursor-linked pages of the project's
                     collector (all recorded projects when `project_id` is omitted). A project
                     whose collector was not recorded is `project-not-recorded`; a recorded walk
                     that does not end in an empty cursor is `listing-incomplete` (a filter over
                     it could silently miss items).
- `list_projects`    the recorded project set (ids, with the counts the recorder noted).
- `get_work_item_by_id`  the `work_item` object of the recorded detail, else the listed entry.
- `get_plan`         the recorded `list_plans` entry.
- `get_work_item_detail` for a listed item the recorder fetched no detail for (it fetches detail
                     for live items only) is `detail-not-recorded`, with the listed summary, and
                     for an id in no listing `work-item-not-recorded`.

Every derived answer names its derivation in the receipt's `match_rules` (`derive:...`), so an
audit can tell recorded bytes from bytes assembled out of recorded objects. Tools the snapshot
holds nothing for (`lane_occupancy`, `stranded_work_items`, `seshat_status`, ...) stay
`tool-not-recorded`.
"""
from __future__ import annotations

import base64
import binascii
import dataclasses
import enum
import weakref
from typing import Any, Mapping

from dojo.snapshot import PRIMARY_ANIMA_COLLECTOR, Snapshot, SnapshotStore

DEFAULT_PAGE_SIZE = 50
RECORDED_HISTORY_PAGE = 50  # the tool's default change/message page, which is what the recorder asked for
MIN_PAGE_SIZE = 1
MAX_PAGE_SIZE = 100
LIST_WORK_ITEM_ARGS = frozenset({"project_id", "current_phase", "plan_id", "page_size", "page_token"})
# Arguments a recorded walk may carry and still be a plain, unfiltered listing of one project.
_WALK_ARGS = frozenset({"project_id", "page_token", "page_size"})
_DOJO_TOKEN_PREFIX = "dojo-offset:"
_NEXT_PAGE_KEYS = frozenset({"nextpagetoken", "nextcursor", "nexttoken"})


class MissReason(str, enum.Enum):
    """Why a read the snapshot has data for could still not be answered. Machine-readable, stable."""

    TOOL_NOT_RECORDED = "tool-not-recorded"
    ARGUMENTS_NOT_RECORDED = "arguments-not-recorded"
    PROJECT_NOT_RECORDED = "project-not-recorded"
    LISTING_INCOMPLETE = "listing-incomplete"
    PAGE_TOKEN_UNKNOWN = "page-token-unknown"
    WORK_ITEM_NOT_RECORDED = "work-item-not-recorded"
    DETAIL_NOT_RECORDED = "detail-not-recorded"
    PLAN_NOT_RECORDED = "plan-not-recorded"


@dataclasses.dataclass(frozen=True)
class Derived:
    """An answer assembled from recorded objects."""

    payload: Any
    rules: tuple[str, ...]
    matched: Mapping[str, Any]  # the recorded call(s) it was derived from, as arguments


@dataclasses.dataclass(frozen=True)
class DeriveMiss:
    reason: MissReason
    detail: str = ""
    context: Mapping[str, Any] = dataclasses.field(default_factory=dict)  # extra recorded facts for the error payload


DeriveResult = Derived | DeriveMiss | None  # None: this module has no derivation for the tool


# --- recorded listings -------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ProjectListing:
    """One project's recorded `list_work_items` walk, reassembled in recorded order."""

    project_id: str
    collector: str
    items: tuple[dict, ...]
    complete: bool  # the last recorded page carried an empty next-page cursor
    offsets: Mapping[str, int]  # recorded next-page cursor -> the offset of the page it opens


def _first_list(payload: Any) -> list:
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return value
    return []


def _next_token(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(key, str) and key.replace("_", "").replace("-", "").lower() in _NEXT_PAGE_KEYS:
                return value if isinstance(value, str) and value else None
    return None


def _collector_projects(snapshot: Snapshot) -> dict[str, str]:
    """collector name -> project id for every Anima collector. The id the recorded calls were made
    with wins over the manifest's `extra.project_id` (older recordings carry only the former)."""
    out: dict[str, str] = {}
    for pid, cname in snapshot.project_collectors.items():
        used = next((r.project_id or r.args.get("project_id") for r in snapshot.responses
                     if r.collector == cname and (r.project_id or r.args.get("project_id"))), None)
        out[cname] = used if isinstance(used, str) else pid
    return out


def _collector_order(item: tuple[str, str]) -> tuple[bool, str]:
    return item[0] != PRIMARY_ANIMA_COLLECTOR, item[0]


def _walk(snapshot: Snapshot, collector: str, project_id: str) -> ProjectListing | None:
    responses = [r for r in snapshot.responses
                 if r.call == "list_work_items" and r.collector == collector and set(r.args) <= _WALK_ARGS]
    if not responses:
        return None
    by_token = {str(r.args["page_token"]): r for r in responses if r.args.get("page_token")}
    page = next((r for r in responses if not r.args.get("page_token")), None)
    items: list[dict] = []
    offsets: dict[str, int] = {}
    seen: set[str] = set()
    visited: set[str] = set()
    complete = False
    while page is not None and page.sha256 not in visited:
        visited.add(page.sha256)
        payload = snapshot.load_object(page.sha256)
        for wi in _first_list(payload):
            if isinstance(wi, dict) and isinstance(wi.get("id"), str) and wi["id"] not in seen:
                seen.add(wi["id"])
                items.append(wi)
        token = _next_token(payload)
        if token is None:
            complete = True
            break
        offsets[token] = len(items)
        page = by_token.get(token)
    return ProjectListing(project_id, collector, tuple(items), complete, offsets)


_LISTINGS: "weakref.WeakKeyDictionary[Snapshot, dict[str, ProjectListing]]" = weakref.WeakKeyDictionary()


def project_listings(snapshot: Snapshot) -> dict[str, ProjectListing]:
    """project id -> recorded listing, primary project first. Empty when no walk was recorded."""
    cached = _LISTINGS.get(snapshot)
    if cached is None:
        cached = {}
        for cname, pid in sorted(_collector_projects(snapshot).items(), key=_collector_order):
            listing = _walk(snapshot, cname, pid)
            if listing is not None:
                cached[pid] = listing
        _LISTINGS[snapshot] = cached
    return cached


def recorded_plans(snapshot: Snapshot) -> dict[str, dict]:
    """plan id -> recorded plan entry, over every recorded Anima project."""
    out: dict[str, dict] = {}
    for cname in _collector_projects(snapshot):
        for plan_id, plan in snapshot.plans(cname).items():
            out.setdefault(plan_id, plan)
    return out


def project_ids(snapshot: Snapshot) -> frozenset[str]:
    """Every project this snapshot recorded an Anima collector for."""
    return frozenset(_collector_projects(snapshot).values())


def _listed_item(snapshot: Snapshot, wid: str) -> tuple[dict, ProjectListing] | None:
    for listing in project_listings(snapshot).values():
        for wi in listing.items:
            if wi.get("id") == wid:
                return wi, listing
    return None


# --- tokens ---------------------------------------------------------------------------------


def _encode_token(offset: int) -> str:
    return base64.urlsafe_b64encode(f"{_DOJO_TOKEN_PREFIX}{offset}".encode()).decode()


def _decode_token(token: str, listings: list[ProjectListing]) -> int | None:
    """An offset from a dojo cursor, or from a cursor the recorder saw (single project only)."""
    if len(listings) == 1 and token in listings[0].offsets:
        return listings[0].offsets[token]
    try:
        text = base64.urlsafe_b64decode(token.encode()).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if text.startswith(_DOJO_TOKEN_PREFIX) and text[len(_DOJO_TOKEN_PREFIX):].isdigit():
        return int(text[len(_DOJO_TOKEN_PREFIX):])
    return None


# --- derivations ----------------------------------------------------------------------------


def derive_list_work_items(snapshot: Snapshot, arguments: Mapping[str, Any]) -> DeriveResult:
    """`arguments` are already schema-normalised (defaults and empty tokens dropped)."""
    unknown = sorted(set(arguments) - LIST_WORK_ITEM_ARGS)
    if unknown:
        return DeriveMiss(MissReason.ARGUMENTS_NOT_RECORDED, f"list_work_items has no argument(s) {unknown}")
    listings = project_listings(snapshot)
    if not listings:
        return None
    rules = ["derive:list_work_items"]
    project_id = arguments.get("project_id")
    if project_id is not None:
        if not isinstance(project_id, str) or project_id not in listings:
            return DeriveMiss(MissReason.PROJECT_NOT_RECORDED,
                              f"snapshot {snapshot.id} recorded no work-item collector for project {project_id!r}",
                              {"recorded_projects": sorted(listings)})
        scope = [listings[project_id]]
    else:
        scope = list(listings.values())
        rules.append("scope:all-recorded-projects")
    incomplete = [p.project_id for p in scope if not p.complete]
    if incomplete:
        return DeriveMiss(MissReason.LISTING_INCOMPLETE,
                          "the recorded walk of the project does not end in an empty cursor, so a filtered or "
                          "re-paged answer could miss items", {"incomplete_projects": incomplete})
    items = [wi for p in scope for wi in p.items]
    for key in ("current_phase", "plan_id"):
        if key in arguments:
            value = arguments[key]
            items = [wi for wi in items if wi.get(key) == value]
            rules.append(f"filter:{key}")
    size = arguments.get("page_size", DEFAULT_PAGE_SIZE)
    if isinstance(size, bool) or not isinstance(size, (int, float)) or int(size) != size:
        return DeriveMiss(MissReason.ARGUMENTS_NOT_RECORDED, f"page_size {size!r} is not an integer")
    size = int(size)
    clamped = max(MIN_PAGE_SIZE, min(MAX_PAGE_SIZE, size))
    if clamped != size:
        rules.append(f"clamp:page_size={clamped}")
    if "page_size" in arguments:
        rules.append(f"page-size:{clamped}")
    offset = 0
    token = arguments.get("page_token")
    if token is not None:
        offset = _decode_token(str(token), scope) if isinstance(token, str) else None
        if offset is None or offset > len(items):
            return DeriveMiss(MissReason.PAGE_TOKEN_UNKNOWN,
                              "the cursor was neither issued by this replay nor one the recorder saw for this project")
        rules.append(f"page:offset={offset}")
    page = items[offset:offset + clamped]
    more = offset + clamped < len(items)
    payload = {"work_items": page, "total": len(items), "next_page_token": _encode_token(offset + clamped) if more else None}
    matched = {"project_id": scope[0].project_id} if len(scope) == 1 else {}
    return Derived(payload, tuple(rules), matched)


PROJECT_METADATA_FIELDS = ("name", "description", "git_remote", "status")
_NAMES: dict[str, tuple[str, dict[str, dict]] | None] = {}


def project_metadata(snapshot: Snapshot) -> tuple[str, dict[str, dict]] | None:
    """(snapshot id, project id -> recorded project entry) from the earliest snapshot under the same root
    that recorded `list_projects` itself. Project metadata (name, git remote) is stable across snapshots;
    a snapshot made before the recorder captured the call still has the project ids and lacks the names."""
    # Below the same snapshot cutoff as the store that opened `snapshot`: a later snapshot must not supply names.
    key = f"{snapshot.root}|{snapshot.visible_until}"
    if key not in _NAMES:
        found = None
        try:
            store = SnapshotStore(snapshot.root, snapshot.visible_until)
            for candidate in store.snapshots():
                entries: dict[str, dict] = {}
                for response in candidate.by_call("list_projects"):
                    for entry in _first_list(candidate.load_object(response.sha256)):
                        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                            entries.setdefault(entry["id"], entry)
                if entries:
                    found = (candidate.id, entries)
                    break
        except OSError:
            found = None
        _NAMES[key] = found
    return _NAMES[key]


def derive_list_projects(snapshot: Snapshot, arguments: Mapping[str, Any]) -> DeriveResult:
    collectors = _collector_projects(snapshot)
    if not collectors or any(True for _ in snapshot.by_call("list_projects")):
        return None  # no project set to derive from, or the recorder captured the call itself: its bytes rule
    unknown = sorted(set(arguments) - {"page_size"})
    if unknown:
        return DeriveMiss(MissReason.ARGUMENTS_NOT_RECORDED, f"list_projects has no argument(s) {unknown}")
    size = arguments.get("page_size", DEFAULT_PAGE_SIZE)
    if isinstance(size, bool) or not isinstance(size, (int, float)) or int(size) != size or size < 1:
        return DeriveMiss(MissReason.ARGUMENTS_NOT_RECORDED, f"page_size {size!r} is not a positive integer")
    projects = []
    names = project_metadata(snapshot)
    named = False
    for cname, pid in sorted(collectors.items(), key=_collector_order):
        extra = snapshot.collector(cname).get("extra") or {}
        entry: dict[str, Any] = {"id": pid}
        recorded = names[1].get(pid) if names else None  # only ids this snapshot's collectors hold: no later project leaks in
        if recorded:
            entry.update({k: recorded[k] for k in PROJECT_METADATA_FIELDS if k in recorded})
            named = True
        for key in ("work_item_count", "plan_count"):
            if isinstance(extra.get(key), int):
                entry[key] = extra[key]
        projects.append(entry)
    payload = {"projects": projects[: int(size)], "total": len(projects), "next_page_token": None}
    source = f"derive:list_projects<-collectors+names@{names[0]}" if names and named else "derive:list_projects<-collectors"
    return Derived(payload, (source,), {})


def derive_get_work_item_by_id(snapshot: Snapshot, arguments: Mapping[str, Any]) -> DeriveResult:
    wid = arguments.get("id")
    if not isinstance(wid, str) or set(arguments) != {"id"}:
        return None
    found = snapshot.detail(wid)
    if found is not None and isinstance(found[0].get("work_item"), dict):
        return Derived({"work_item": found[0]["work_item"]}, ("derive:get_work_item_by_id<-get_work_item_detail",),
                       {"id": wid})
    listed = _listed_item(snapshot, wid)
    if listed is not None:
        return Derived({"work_item": listed[0]}, ("derive:get_work_item_by_id<-list_work_items",), {})
    return DeriveMiss(MissReason.WORK_ITEM_NOT_RECORDED, f"no recorded detail or listing holds work item {wid!r}")


def derive_get_work_item_detail(snapshot: Snapshot, arguments: Mapping[str, Any]) -> DeriveResult:
    """What the exact-match path could not answer: a detail asked with history page sizes the recorded
    (default-sized) detail provably already covers, or the reason no detail exists."""
    wid = arguments.get("id")
    if not isinstance(wid, str):
        return None
    found = snapshot.detail(wid)
    if found is not None:
        detail, _ = found
        sizes = {k: v for k, v in arguments.items() if k != "id"}
        if set(sizes) - {"message_page_size", "change_page_size"}:
            return None
        messages, changes = detail.get("thread_messages"), detail.get("change_history")
        covered = (
            isinstance(messages, list) and isinstance(changes, list)
            and detail.get("thread_messages_elided") is False  # the recorder's own "nothing was cut" flag
            and len(changes) < RECORDED_HISTORY_PAGE  # the recorder asked for the default page and did not fill it
            and _covers(sizes.get("message_page_size"), len(messages))
            and _covers(sizes.get("change_page_size"), len(changes))
        )
        return Derived(detail, ("derive:get_work_item_detail:history-page-sizes",), {"id": wid}) if covered else None
    listed = _listed_item(snapshot, wid)
    if listed is None:
        return DeriveMiss(MissReason.WORK_ITEM_NOT_RECORDED, f"no recorded detail or listing holds work item {wid!r}")
    return DeriveMiss(MissReason.DETAIL_NOT_RECORDED,
                      "the recorder fetches detail for live work items only; the listing entry is attached",
                      {"listed": listed[0]})


def _covers(size: Any, recorded: int) -> bool:
    return size is None or (isinstance(size, int) and not isinstance(size, bool) and size >= recorded)


def derive_get_plan(snapshot: Snapshot, arguments: Mapping[str, Any]) -> DeriveResult:
    plan_id = arguments.get("id")
    if not isinstance(plan_id, str) or set(arguments) != {"id"}:
        return None
    if not any(True for c in _collector_projects(snapshot) for _ in snapshot.by_call("list_plans", c)):
        return None
    plan = recorded_plans(snapshot).get(plan_id)
    if plan is None:
        return DeriveMiss(MissReason.PLAN_NOT_RECORDED, f"no recorded list_plans holds plan {plan_id!r}")
    return Derived({"plan": plan}, ("derive:get_plan<-list_plans",), {})


DERIVATIONS = {
    "list_work_items": derive_list_work_items,
    "list_projects": derive_list_projects,
    "get_work_item_by_id": derive_get_work_item_by_id,
    "get_work_item_detail": derive_get_work_item_detail,
    "get_plan": derive_get_plan,
}


def derive(snapshot: Snapshot, name: str, arguments: Mapping[str, Any]) -> DeriveResult:
    function = DERIVATIONS.get(name)
    return None if function is None else function(snapshot, arguments)
