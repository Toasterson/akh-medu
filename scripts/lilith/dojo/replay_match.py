"""Schema-driven argument equivalence for the replay server (plan slice C4, ADR 116 Decision 3).

The recorder captured each read with one argument set. A model that asks for the same
information with a different spelling used to get "not recorded" and, downstream, a
`fabrication` flag for citing that receipt. This module answers such a call from the
recorded bytes **only when the equivalence is provable from the tool's own input schema and
the recorded response**; anything else stays an error. Nothing is invented and the answer is
always exactly one recorded object.

Rules (each one is named in the receipt's `match_rules`, so a normalised match is auditable
next to the requested and the recorded arguments):

- `coerce:<param>:<type>`   the schema types the parameter (`integer`, `number`, `boolean`,
  `string`) and the value is the same value in another JSON type (`"50"` for `50`).
- `default:<param>`         the value equals the schema's own `default` keyword, so it is the
  omitted argument.
- `empty-token:<param>`     a page-token-role parameter set to the empty string; the first page.
- `page-size-satisfied:<param>=<n>`
                            a page-size-role parameter (below) whose value the recorded,
                            page-size-less response already satisfies:
                            * the recorded response is complete (its next-page token key is
                              present and empty) and holds at most `n` items, or
                            * the recorded response has a next-page token and exactly `n`
                              items: a page is full iff a next page exists, so the server's
                              page size *is* `n`, or
                            * the recorded response is a token-less single listing that the
                              recorder marked `complete` (it got the same bytes for a much larger
                              page) and holds at most `n` items (`list_projects`).
                            A smaller `n` (would truncate) or a larger `n` on a partial
                            listing (would return more than was recorded) is not equivalent
                            and stays an error.
- `ref-alias:<requested>-><recorded>`  (git only) a ref spelled the way `git rev-parse` DWIMs
                            it (`main`, `heads/main`, `origin/main`) resolved against the refs
                            the recorder saw on the remote.

Role detection uses API parameter identifiers, never natural language: a page-size parameter is
an `integer` property named `page_size` / `per_page` / `limit` / `top_k` / `max_results` / `k`
(optionally prefixed, e.g. `change_page_size`); a page-token parameter is a `string` property
named `page_token` / `cursor` (optionally prefixed). The response's own next-page key is
matched on the normalised identifier (`next_page_token`, `nextPageToken`, `next_cursor`).
Unrecorded tools and unrecorded ids are never normalised away.

Beyond argument spelling, `replay_anima` answers reads the recorder captured as *data* rather
than as a call: `list_work_items` with any `current_phase` / `plan_id` / `project_id` / `page_size`
/ `page_token` (filter + paginate over the recorded pages), `list_projects`,
`get_work_item_by_id` and `get_plan`. Those receipts carry `derive:...` rules.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Any, Mapping

from dojo import replay_anima
from dojo.replay_anima import Derived, DeriveMiss, MissReason
from dojo.snapshot import MAIN_REF, ObjectRef, Snapshot, canonical_args, canonical_json, sha256_hex

PAGE_SIZE_PARAM = re.compile(r"^(?:[a-z][a-z0-9]*_)*(?:page_size|per_page|limit|top_k|max_results|k)$")
PAGE_TOKEN_PARAM = re.compile(r"^(?:[a-z][a-z0-9]*_)*(?:page_token|cursor)$")
NEXT_PAGE_KEYS = frozenset({"nextpagetoken", "nextcursor", "nexttoken"})
# git's default remote name: `origin/<branch>` is the local spelling of the remote's
# `refs/heads/<branch>`, which is what the recorder's ls-remote captured.
GIT_DEFAULT_REMOTE = "origin"


@dataclasses.dataclass(frozen=True)
class ArgumentMatch:
    requested: Mapping[str, Any]
    matched: Mapping[str, Any]  # the recorded call's arguments
    rules: tuple[str, ...] = ()

    @property
    def exact(self) -> bool:
        return not self.rules


@dataclasses.dataclass(frozen=True)
class RecordedRead:
    payload: Any
    ref: ObjectRef
    match: ArgumentMatch


def _properties(input_schema: Mapping[str, Any] | None) -> Mapping[str, Mapping[str, Any]]:
    props = (input_schema or {}).get("properties")
    return props if isinstance(props, Mapping) else {}


def _is_page_size(name: str, prop: Mapping[str, Any]) -> bool:
    return prop.get("type") in ("integer", "number") and bool(PAGE_SIZE_PARAM.match(name))


def _is_page_token(name: str, prop: Mapping[str, Any]) -> bool:
    return prop.get("type") == "string" and bool(PAGE_TOKEN_PARAM.match(name))


def _coerce(value: Any, prop: Mapping[str, Any]) -> tuple[Any, str | None]:
    """The same value in the JSON type the schema declares, or the value unchanged."""
    declared = prop.get("type")
    if declared == "integer":
        if isinstance(value, float) and not isinstance(value, bool) and value.is_integer():
            return int(value), "integer"
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value), "integer"
    elif declared == "number":
        if isinstance(value, str):
            try:
                number = float(value)
            except ValueError:
                return value, None
            return (int(number) if number.is_integer() else number), "number"
    elif declared == "boolean":
        if value == "true" or value == "false":  # JSON literals, not prose
            return value == "true", "boolean"
    elif declared == "string":
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value), "string"
    return value, None


def normalize_arguments(input_schema: Mapping[str, Any] | None, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce declared types, drop schema defaults and empty page tokens. Unknown keys are kept."""
    props = _properties(input_schema)
    out: dict[str, Any] = {}
    rules: list[str] = []
    for key in sorted(arguments):
        value = arguments[key]
        if value is None:
            continue
        prop = props.get(key)
        if prop is None:
            out[key] = value
            continue
        coerced, kind = _coerce(value, prop)
        if kind is not None:  # `_coerce` reports a kind only when the JSON type changed
            rules.append(f"coerce:{key}:{kind}")
        value = coerced
        if "default" in prop and value == prop["default"] and type(value) is type(prop["default"]):
            rules.append(f"default:{key}")
            continue
        if _is_page_token(key, prop) and value == "":
            rules.append(f"empty-token:{key}")
            continue
        out[key] = value
    return out, rules


def _next_page_key(payload: Mapping[str, Any]) -> str | None:
    for key in payload:
        if isinstance(key, str) and key.replace("_", "").replace("-", "").lower() in NEXT_PAGE_KEYS:
            return key
    return None


def page_size_satisfied(payload: Any, size: int, complete: bool = False) -> bool:
    """Would the live tool, asked for `size` items, have returned exactly this recorded response?

    `complete` is the recorder's own proof that a token-less listing was not truncated (it asked for
    a page larger than the answer and got the same bytes); without it a listing that has no
    next-page key is not provably complete."""
    if not isinstance(payload, Mapping) or isinstance(size, bool) or size < 1:
        return False
    lists = [v for v in payload.values() if isinstance(v, list)]
    token_key = _next_page_key(payload)
    if complete and token_key is None and len(lists) == 1:
        return len(lists[0]) <= size
    if len(lists) != 1 or token_key is None:
        return False  # not one cursor-paginated listing: completeness is unknowable
    count = len(lists[0])
    if not payload[token_key]:
        return count <= size  # complete listing: any adequate size returns all of it
    return count == size  # a page is full iff another page exists, so size == page size


def resolve_recorded_read(snapshot: Snapshot, name: str, input_schema: Mapping[str, Any] | None,
                          arguments: Mapping[str, Any]) -> RecordedRead | None:
    """The recorded object that answers `name(arguments)`, exactly or provably equivalently."""
    requested = dict(arguments)
    response = snapshot.find(name, requested)
    if response is not None:
        return _read(snapshot, response, ArgumentMatch(requested, dict(response.args)))
    normalised, rules = normalize_arguments(input_schema, requested)
    response = snapshot.find(name, normalised)
    if response is not None:
        return _read(snapshot, response, ArgumentMatch(requested, dict(response.args), tuple(rules)))
    props = _properties(input_schema)
    sized = [k for k, v in normalised.items()
             if k in props and _is_page_size(k, props[k]) and isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(sized) == 1:
        key = sized[0]
        rest = {k: v for k, v in normalised.items() if k != key}
        response = snapshot.find(name, rest)
        if response is not None and page_size_satisfied(snapshot.load_object(response.sha256), int(normalised[key]),
                                                        bool(response.complete)):
            return _read(snapshot, response, ArgumentMatch(
                requested, dict(response.args), (*rules, f"page-size-satisfied:{key}={int(normalised[key])}")))
    derived = replay_anima.derive(snapshot, name, normalised)
    if isinstance(derived, Derived):
        ref = ObjectRef(snapshot.id, name, canonical_args(requested), sha256_hex(canonical_json(derived.payload)))
        return RecordedRead(derived.payload, ref, ArgumentMatch(requested, dict(derived.matched), (*rules, *derived.rules)))
    return None


def _read(snapshot: Snapshot, response, match: ArgumentMatch) -> RecordedRead:
    return RecordedRead(snapshot.load_object(response.sha256), snapshot.ref(response), match)


def recorded_shape(snapshot: Snapshot, name: str) -> str:
    """Why a read missed: was the tool ever recorded in this snapshot?"""
    return "arguments-not-recorded" if any(True for _ in snapshot.by_call(name)) else "tool-not-recorded"


@dataclasses.dataclass(frozen=True)
class MissExplanation:
    """Why a read could not be answered: a stable `reason`, what the replay knows about it."""

    reason: str
    detail: str = ""
    context: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def explain_miss(snapshot: Snapshot, name: str, input_schema: Mapping[str, Any] | None,
                 arguments: Mapping[str, Any]) -> MissExplanation:
    """The most specific reason `resolve_recorded_read` returned `None` for this call."""
    normalised, _ = normalize_arguments(input_schema, arguments)
    derived = replay_anima.derive(snapshot, name, normalised)
    if isinstance(derived, DeriveMiss):
        return MissExplanation(derived.reason.value, derived.detail, derived.context)
    shape = recorded_shape(snapshot, name)
    project = arguments.get("project_id")
    if (shape == MissReason.ARGUMENTS_NOT_RECORDED.value and isinstance(project, str)
            and project not in replay_anima.project_ids(snapshot)):
        return MissExplanation(MissReason.PROJECT_NOT_RECORDED.value,
                               f"snapshot {snapshot.id} recorded no collector for project {project!r}",
                               {"recorded_projects": sorted(replay_anima.project_ids(snapshot))})
    return MissExplanation(shape)


def resolve_ref(refs: Mapping[str, str], requested: str | None) -> tuple[str | None, tuple[str, ...]]:
    """A recorded ref name for `requested` under git's own DWIM order, and the rule applied.

    `None` for the request means the documented default (`refs/heads/main`), not a rule.
    """
    if not requested:
        return (MAIN_REF if MAIN_REF in refs else None), ()
    if requested in refs:
        return requested, ()
    spellings = [requested]
    for prefix in (f"refs/remotes/{GIT_DEFAULT_REMOTE}/", f"{GIT_DEFAULT_REMOTE}/"):
        if requested.startswith(prefix):
            spellings.append(requested[len(prefix):])
            break
    for spelling in spellings:  # the literal spelling wins over the remote-stripped one
        for candidate in (f"refs/{spelling}", f"refs/tags/{spelling}", f"refs/heads/{spelling}", spelling):
            if candidate in refs:
                return candidate, (f"ref-alias:{requested}->{candidate}",)
    return None, ()


__all__ = [
    "ArgumentMatch",
    "MissExplanation",
    "RecordedRead",
    "explain_miss",
    "normalize_arguments",
    "page_size_satisfied",
    "recorded_shape",
    "resolve_recorded_read",
    "resolve_ref",
]
