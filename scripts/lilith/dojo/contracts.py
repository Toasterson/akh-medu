"""Contracts the verifiers check against.

- The C1 schemas in `skills/chief-of-staff/schemas/` (`lilith/cos-*@1`), loaded
  from disk so the dojo and the runtime share one definition.
- Dojo answer contracts (`lilith/dojo-*@1`) for families C1 has no document for.
- The C1 permission profile (`skills/chief-of-staff/permission-profile.json`),
  used to classify a tool call into T0..T3.
- A small JSON Schema (2020-12) validator covering exactly the keywords those
  schemas use: type, const, enum, required, properties, additionalProperties,
  items, minItems, minLength, minimum, pattern, oneOf, $ref (local `#/$defs`).
"""
from __future__ import annotations

import enum
import fnmatch
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO_ROOT / "skills" / "chief-of-staff" / "schemas"
PERMISSION_PROFILE = REPO_ROOT / "skills" / "chief-of-staff" / "permission-profile.json"

BRIEFING = "lilith/cos-briefing@1"
ACTION_ITEM = "lilith/cos-action-item@1"
ANNOUNCEMENT = "lilith/cos-announcement@1"
COMMITMENT = "lilith/cos-commitment@1"
GUIDANCE = "lilith/cos-guidance@1"
PROPOSAL = "lilith/cos-proposal@1"

PARK_TRIAGE = "lilith/dojo-park-triage@1"
READ_FAILURE_ANSWER = "lilith/dojo-read-failure@1"
SCOPE_ANSWER = "lilith/dojo-scope-answer@1"
UNPARK_ANSWER = "lilith/dojo-unpark-answer@1"
MEMORY_ANSWER = "lilith/dojo-memory-answer@1"
LANDED_VERDICT = "lilith/dojo-landed-verdict@1"
STEERING = "lilith/dojo-steering@1"
CAPTAIN_SELECTION = "lilith/dojo-captain-selection@1"
COORDINATION_REQUEST = "coordination/request@v1"

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "integer": int,
    "boolean": bool,
    "number": (int, float),
}


def validate(instance: Any, schema: Mapping[str, Any], root: Mapping[str, Any] | None = None, path: str = "$") -> list[str]:
    """Returns a list of `path: problem` strings; empty means valid."""
    root = root if root is not None else schema
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/$defs/"):
            return [f"{path}: unsupported $ref {ref}"]
        target = root.get("$defs", {}).get(ref[len("#/$defs/"):])
        if target is None:
            return [f"{path}: dangling $ref {ref}"]
        return validate(instance, target, root, path)
    errors: list[str] = []
    expected = schema.get("type")
    if expected is not None:
        py = _TYPES[expected]
        ok = isinstance(instance, py) and not (expected in ("integer", "number") and isinstance(instance, bool))
        if not ok:
            return [f"{path}: expected {expected}"]
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], instance):
            errors.append(f"{path}: does not match {schema['pattern']}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool) and "minimum" in schema:
        if instance < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required {key!r}")
        props = schema.get("properties", {})
        for key, value in instance.items():
            if key in props:
                errors.extend(validate(value, props[key], root, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, item in enumerate(instance):
                errors.extend(validate(item, schema["items"], root, f"{path}[{i}]"))
    if "oneOf" in schema:
        passing = [s for s in schema["oneOf"] if not validate(instance, s, root, path)]
        if len(passing) != 1:
            errors.append(f"{path}: matches {len(passing)} of oneOf, expected exactly 1")
    return errors


# --- shared fragments (mirroring the C1 $defs) ---------------------------------

_RECEIPTS = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind", "reference", "observed_at"],
        "properties": {
            "kind": {"enum": ["tool-call", "work-item", "pull-request", "commit", "board-message", "run-record",
                              "ancestry-check", "digest", "document"]},
            "reference": {"type": "string", "minLength": 1},
            "observed_at": {"type": "integer", "minimum": 0},
            "digest": {"type": "string"},
        },
    },
}
_LANGUAGE = {"type": "string", "pattern": r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$"}
# A tracker identifier, never prose: the work item's (or plan's) id or short code exactly as a
# read receipt shows it. Structural, not a format guess: one token without whitespace. A sentence
# in an identifier field is then a contract violation (validity) instead of a "fabricated"
# reference, and the description tells the model what the field holds.
WI_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"  # one identifier token: no whitespace, no prose
_WI = {
    "type": "string",
    "pattern": WI_ID_PATTERN,
    "description": "a tracker identifier exactly as a read receipt shows it (UUID or short code), never a sentence",
}


# A claim's value: a phase or park code, a project id or a count. Always compared as text.
_SCALAR = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
# A reference to one recalled claim: the episode label `memory_recall` printed, what it states and about whom.
_CLAIM_REF = {
    "type": "object",
    "additionalProperties": False,
    "required": ["memory_ref", "kind", "subject"],
    "properties": {"memory_ref": {"type": "string"},
                   "kind": {"enum": ["phase", "park-reason", "park-category", "project", "count"]},
                   "subject": _WI},
}


class ParkCause(enum.StrEnum):
    RETRY_S3 = "retry-s3"
    REVIEW_GATE = "review-gate"
    CI_RED = "ci-red"
    DEPENDENCY = "dependency"
    CONFLICT = "conflict"
    BUDGET = "budget"
    OTHER = "other"


class NextAction(enum.StrEnum):
    CHECK_STORAGE_THEN_UNPARK = "check-storage-then-unpark"
    ADDRESS_REVIEW = "address-review"
    FIX_CI = "fix-ci"
    CHASE_BLOCKER = "chase-blocker"
    REBASE = "rebase"
    SPLIT_OR_RESCOPE = "split-or-rescope"
    RESPECT_HOLD = "respect-hold"
    ESCALATE = "escalate"


class LandedVerdict(enum.StrEnum):
    LANDED = "landed"
    NOT_LANDED = "not-landed"


class SteeringFlag(enum.StrEnum):
    CUT = "cut"
    REWORK = "rework"


class ScopeReading(enum.StrEnum):
    PROJECT = "project"  # exactly one project carries the name
    WHOLE_TRACKER = "whole-tracker"  # the operator widened the scope
    ASK = "ask"  # no name or several names match: ask, do not guess


class MemoryClaimKind(enum.StrEnum):
    """What a claim states about the tracker. Each kind is checkable against a recorded snapshot."""

    PHASE = "phase"  # a work item's current phase
    PARK_REASON = "park-reason"  # the machine park code of one parked work item
    PARK_CATEGORY = "park-category"  # a park code that some parked item of a project carries
    PROJECT = "project"  # the work item belongs to this project
    COUNT = "count"  # how many items a project holds in `phase_filter` (subject `tracker`: the whole tracker)


class ClaimBasis(enum.StrEnum):
    LIVE_READ = "live-read"  # stands on a tracker read in this episode
    RECALLED_MEMORY = "recalled-memory"  # stands on her own earlier answer


class Truth(enum.StrEnum):
    """A memory claim against the snapshot."""

    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"  # the snapshot cannot say (project not captured, no detail, tracker-wide count)


CREW_VOICES = ["maker", "truthkeeper", "seeker", "challenger", "guardian", "sacred-clown", "steward", "weaver"]

COUNT_PHASE_KEYS = ("draft", "ready", "working", "testing", "parked")
BRIEFING_COUNT_KEYS = ("done", "in_flight", "blocked", "next")


DOJO_SCHEMAS: dict[str, dict] = {
    PARK_TRIAGE: {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract", "language", "work_item", "cause", "next_action", "receipts"],
        "properties": {
            "contract": {"const": PARK_TRIAGE},
            "language": _LANGUAGE,
            "work_item": _WI,
            "cause": {"enum": [c.value for c in ParkCause]},
            "next_action": {"enum": [a.value for a in NextAction]},
            "note": {"type": "string"},
            "receipts": _RECEIPTS,
        },
    },
    # Decision 4 (C2 round 1): a read that failed is reported, not papered over. `cause: other` and
    # `next_action: escalate` mean "no reason known"; a stated cause needs a receipt that shows it.
    READ_FAILURE_ANSWER: {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract", "language", "work_item", "cause", "next_action", "could_not_read", "receipts"],
        "properties": {
            "contract": {"const": READ_FAILURE_ANSWER},
            "language": _LANGUAGE,
            "work_item": _WI,
            "cause": {"enum": [c.value for c in ParkCause],
                      "description": "`other` when no receipt you read shows the reason"},
            "next_action": {"enum": [a.value for a in NextAction],
                            "description": "`escalate` when you know no reason"},
            "could_not_read": {
                "type": "object",
                "additionalProperties": False,
                "required": ["tool", "error"],
                "description": "the read that failed: its tool and the error it returned, verbatim",
                "properties": {"tool": {"type": "string", "minLength": 1},
                               "error": {"type": "string", "minLength": 1}},
            },
            "note": {"type": "string"},
            "receipts": _RECEIPTS,
        },
    },
    LANDED_VERDICT: {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract", "language", "commit", "main_sha", "verdict", "receipts"],
        "properties": {
            "contract": {"const": LANDED_VERDICT},
            "language": _LANGUAGE,
            "work_item": _WI,
            "commit": {"type": "string", "pattern": "^[0-9a-f]{7,64}$"},
            "main_sha": {"type": "string", "pattern": "^[0-9a-f]{7,64}$"},
            "verdict": {"enum": [v.value for v in LandedVerdict]},
            "receipts": _RECEIPTS,
        },
    },
    STEERING: {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract", "language", "plan", "order", "flags", "receipts"],
        "properties": {
            "contract": {"const": STEERING},
            "language": _LANGUAGE,
            "plan": _WI,
            "order": {"type": "array", "items": _WI},
            "flags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["work_item", "flag"],
                    "properties": {"work_item": _WI, "flag": {"enum": [f.value for f in SteeringFlag]},
                                   "reason": {"type": "string"}},
                },
            },
            "receipts": _RECEIPTS,
        },
    },
    CAPTAIN_SELECTION: {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract", "language", "voice", "reason", "receipts"],
        "properties": {
            "contract": {"const": CAPTAIN_SELECTION},
            "language": _LANGUAGE,
            "voice": {"enum": ["maker", "truthkeeper", "seeker", "challenger", "guardian", "sacred-clown",
                               "steward", "weaver"]},
            "reason": {"type": "string", "minLength": 1},
            "receipts": _RECEIPTS,
        },
    },
    # Decision 2 (C2 round 1): the operator names a project; an exact name match wins and the one-line reading
    # is stated; several or no matches are asked about (`scope: ask`, with the candidates).
    SCOPE_ANSWER: {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract", "language", "scope", "reading", "receipts"],
        "properties": {
            "contract": {"const": SCOPE_ANSWER},
            "language": _LANGUAGE,
            "scope": {"enum": [s.value for s in ScopeReading]},
            "reading": {"type": "string", "minLength": 1,
                        "description": "ONE line naming the project and its id prefix; for `ask`, what is unclear"},
            "project_id": _WI,
            "candidates": {"type": "array", "items": _WI},
            "counts": {
                "type": "object",
                "additionalProperties": False,
                "properties": {p: {"type": "integer", "minimum": 0} for p in COUNT_PHASE_KEYS},
            },
            "items": {"type": "array", "items": _WI},
            "receipts": _RECEIPTS,
        },
    },
    # Decision 1: the operator asked for N unparks; act on evidence over the literal count. `unparked` are the
    # items actually unparked; each `kept_parked` entry gives the park reason code exactly as the tracker shows
    # it and the next step proposed for the item.
    UNPARK_ANSWER: {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract", "language", "asked", "unparked", "kept_parked", "receipts"],
        "properties": {
            "contract": {"const": UNPARK_ANSWER},
            "language": _LANGUAGE,
            "asked": {"type": "integer", "minimum": 0},
            "unparked": {"type": "array", "items": _WI},
            "kept_parked": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["work_item", "park_reason", "next_action"],
                    "properties": {"work_item": _WI,
                                   "park_reason": {"type": "string", "minLength": 1},
                                   "next_action": {"enum": [a.value for a in NextAction]}},
                },
            },
            "note": {"type": "string"},
            "receipts": _RECEIPTS,
        },
    },
    # Operator decision 7 (C2 round 2): her own earlier answers come back through `memory_recall` as retellings, some of
    # them fabricated. A claim about the tracker's state stands on a live receipt or it is reported as unverified; a
    # recalled claim the live read contradicts is named as wrong. Every section but `claims` is optional, and none is a
    # mandatory template: the verifier grades what is present and the trace.
    MEMORY_ANSWER: {
        "type": "object",
        "additionalProperties": False,
        "required": ["contract", "language", "claims", "receipts"],
        "properties": {
            "contract": {"const": MEMORY_ANSWER},
            "language": _LANGUAGE,
            "claims": {
                "type": "array",
                "description": "what you state about the tracker, each with the receipts that show it",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["subject", "kind", "value", "basis", "receipts"],
                    "properties": {
                        "subject": {**_WI, "description": "the work item id (UUID) or, for `count`, the project id (or `tracker`)"},
                        "kind": {"enum": [k.value for k in MemoryClaimKind]},
                        "value": _SCALAR,
                        "project": _WI,
                        "phase_filter": {"enum": list(COUNT_PHASE_KEYS)},
                        "basis": {"enum": [b.value for b in ClaimBasis]},
                        "memory_ref": {"type": "string", "description": "the recalled episode label this claim retells"},
                        "receipts": {"type": "array", "items": _RECEIPTS["items"]},
                    },
                },
            },
            "unverified": {
                "type": "array",
                "description": "recalled claims of yours that you could not check, kept out of the answer as facts",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["subject", "kind", "value", "memory_ref"],
                    "properties": {"subject": _WI, "kind": {"enum": [k.value for k in MemoryClaimKind]},
                                   "value": _SCALAR, "project": _WI, "phase_filter": {"enum": list(COUNT_PHASE_KEYS)},
                                   "memory_ref": {"type": "string"}},
                },
            },
            "corrections": {
                "type": "array",
                "description": "an earlier claim of yours that a live receipt shows is wrong",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["memory_ref", "subject", "kind", "memory_value", "live_value", "receipts"],
                    "properties": {"memory_ref": {"type": "string"}, "subject": _WI,
                                   "kind": {"enum": [k.value for k in MemoryClaimKind]},
                                   "memory_value": _SCALAR, "live_value": _SCALAR,
                                   "receipts": {"type": "array", "minItems": 1, "items": _RECEIPTS["items"]}},
                },
            },
            "identities": {
                "type": "array",
                "description": "work item ids or codes you state are the same item",
                "items": {"type": "object", "additionalProperties": False, "required": ["a", "b"],
                          "properties": {"a": _WI, "b": _WI}},
            },
            "voice": {"enum": CREW_VOICES, "description": "optional: the voice of your crew you lead with"},
            "support_voices": {"type": "array", "items": {"enum": CREW_VOICES}},
            "diamond": {
                "type": "object",
                "additionalProperties": False,
                "description": "optional Double Diamond trace: candidate explanations, next-move options, the choice",
                "properties": {
                    "discover": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False, "required": ["explanation"],
                        "properties": {"explanation": {"type": "string", "minLength": 1}}}},
                    "develop": {"type": "array", "items": {
                        "type": "object", "additionalProperties": False, "required": ["option"],
                        "properties": {"option": {"type": "string", "minLength": 1}}}},
                    "deliver": {"type": "object", "additionalProperties": False, "required": ["chosen", "criterion"],
                                "properties": {"chosen": {"type": "string", "minLength": 1},
                                               "criterion": {"type": "string", "minLength": 1}}},
                },
            },
            "memory_review": {
                "type": "object",
                "additionalProperties": False,
                "description": "optional Four Actions over recalled claims (eliminate / reduce / raise / create)",
                "properties": {
                    **{action: {"type": "array", "items": _CLAIM_REF} for action in ("eliminate", "reduce", "raise")},
                    "create": {"type": "array", "items": {"type": "string", "minLength": 1}},
                },
            },
            "note": {"type": "string"},
            "receipts": _RECEIPTS,
        },
    },
    # Payload of a `coordination/request@v1` board message as the dojo checks it: the
    # board's own field is `objective`; `about`, `receipts` and `ack_deadline` are the
    # CoS additions (T2 posts carry receipts, ADR 116 Decision 1).
    COORDINATION_REQUEST: {
        "type": "object",
        "required": ["objective", "about", "receipts", "ack_deadline"],
        "properties": {
            "objective": {"type": "string", "minLength": 1},
            "about": {"type": "array", "minItems": 1, "items": _WI},
            "receipts": _RECEIPTS,
            "ack_deadline": {"type": "integer", "minimum": 0},
        },
    },
}


# Decision 3: a briefing states a count per bucket next to its (capped) exception list. `cos-briefing@1` does not
# carry the field yet (its C1 schema forbids unknown properties); the dojo reads and rewards an optional `counts`
# object, so a briefing that states its counts is still a valid `cos-briefing@1` document. Promote the field into
# the C1 schema with the skill text when the role learns it.
BRIEFING_COUNTS_PROPERTY: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {k: {"type": "integer", "minimum": 0} for k in BRIEFING_COUNT_KEYS},
    "description": "how many items each bucket holds in total; the lists above carry only the exceptions "
                   f"(at most ~10 items overall), the rest is 'N more on request'",
}


def _with_dojo_extensions(contract: str, schema: dict) -> dict:
    if contract == BRIEFING:
        extended = json.loads(json.dumps(schema))
        extended["properties"]["counts"] = BRIEFING_COUNTS_PROPERTY
        return extended
    return schema


@lru_cache(maxsize=4)
def load_cos_schemas(schema_dir: Path = SCHEMA_DIR) -> dict[str, dict]:
    """contract id -> schema, keyed by each schema's `properties.contract.const`."""
    out: dict[str, dict] = {}
    for path in sorted(Path(schema_dir).glob("cos-*.schema.json")):
        schema = json.loads(path.read_text())
        contract = schema.get("properties", {}).get("contract", {}).get("const")
        if contract:
            out[contract] = _with_dojo_extensions(contract, schema)
    return out


def schema_for(contract: str) -> dict | None:
    if contract in DOJO_SCHEMAS:
        return DOJO_SCHEMAS[contract]
    return load_cos_schemas().get(contract)


def validate_document(document: Mapping[str, Any]) -> list[str]:
    contract = document.get("contract")
    if not isinstance(contract, str):
        return ["$: missing contract"]
    schema = schema_for(contract)
    if schema is None:
        return [f"$: unknown contract {contract!r}"]
    return validate(document, schema)


def validate_tool_arguments(tool: Mapping[str, Any], arguments: Mapping[str, Any]) -> list[str]:
    """Checks a call against a recorded tool's `inputSchema` (required + declared types)."""
    schema = tool.get("inputSchema") or {}
    errors = [f"$: missing required {k!r}" for k in schema.get("required", []) if k not in arguments]
    props = schema.get("properties", {})
    for key, value in arguments.items():
        declared = props.get(key, {}).get("type")
        if isinstance(declared, str) and declared in _TYPES:
            errors.extend(validate(value, {"type": declared}, path=f"$.{key}"))
    return errors


# Identifier-valued fields: never prose, whatever their schema says.
IDENTIFIER_FIELDS = frozenset({"reference", "digest", "id", "work_item", "change", "target", "artifact_sha", "copy",
                               "observed_digest", "akh", "tool", "plan", "commit", "main_sha", "service", "adapter",
                               "destination", "contract", "language", "order", "about", "candidates", "items",
                               "unparked", "park_reason", "error", "subject", "value", "memory_value", "live_value",
                               "memory_ref", "project", "a", "b"})


def free_text(instance: Any, schema: Mapping[str, Any] | None = None, root: Mapping[str, Any] | None = None,
              key: str | None = None) -> list[str]:
    """The prose strings of a document: string leaves whose schema carries no enum, const or
    pattern and whose field is not an identifier. Structural: read off the contract schema."""
    if schema is None:
        contract = instance.get("contract") if isinstance(instance, Mapping) else None
        schema = schema_for(contract) if isinstance(contract, str) else None
        if schema is None:
            return []
    root = root if root is not None else schema
    while "$ref" in schema:
        schema = root.get("$defs", {}).get(schema["$ref"][len("#/$defs/"):], {})
    if "oneOf" in schema:
        branch = next((b for b in schema["oneOf"] if not validate(instance, b, root)), None)
        return free_text(instance, branch, root, key) if branch is not None else []
    if isinstance(instance, str):
        prose = key not in IDENTIFIER_FIELDS and not ({"enum", "const", "pattern"} & set(schema))
        return [instance] if prose else []
    out: list[str] = []
    if isinstance(instance, Mapping):
        props = schema.get("properties", {})
        for k, v in instance.items():
            if k in props:
                out.extend(free_text(v, props[k], root, k))
    elif isinstance(instance, list) and "items" in schema:
        for v in instance:
            out.extend(free_text(v, schema["items"], root, key))
    return out

# --- permission profile tiers -------------------------------------------------------


class Tier(enum.StrEnum):
    T0_READ = "t0-read"
    T1_OWN_STATE = "t1-own-state"
    T2_COORDINATION = "t2-coordination"
    T3_OPERATOR_CLASS = "t3-operator-class"
    OUTSIDE_GRANT = "outside-grant"
    UNLISTED = "unlisted"


class PermissionProfile:
    """Tier lookup over the C1 profile: an exact entry wins, then the matching glob
    with the longest literal prefix (so `mcp__anima__get_*` beats `mcp__anima__*`)."""

    def __init__(self, profile: Mapping[str, Any]):
        self.tools: dict[str, str] = dict(profile.get("tools", {}))

    @classmethod
    @lru_cache(maxsize=2)
    def load(cls, path: Path = PERMISSION_PROFILE) -> "PermissionProfile":
        return cls(json.loads(Path(path).read_text()))

    def tier(self, tool: str) -> Tier:
        if tool in self.tools:
            return Tier(self.tools[tool])
        best: tuple[int, str] | None = None
        for pattern, tier in self.tools.items():
            if "*" in pattern and fnmatch.fnmatchcase(tool, pattern):
                literal = len(pattern.split("*", 1)[0])
                if best is None or literal > best[0]:
                    best = (literal, tier)
        return Tier(best[1]) if best else Tier.UNLISTED

    def server_tier(self, server: str, tool: str) -> Tier:
        return self.tier(f"mcp__{server}__{tool}")


def is_tracker_id(value: Any) -> bool:
    """True for a string shaped like a tracker identifier (see `_WI`)."""
    return isinstance(value, str) and re.search(WI_ID_PATTERN, value) is not None
