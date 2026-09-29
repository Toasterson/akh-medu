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


@lru_cache(maxsize=4)
def load_cos_schemas(schema_dir: Path = SCHEMA_DIR) -> dict[str, dict]:
    """contract id -> schema, keyed by each schema's `properties.contract.const`."""
    out: dict[str, dict] = {}
    for path in sorted(Path(schema_dir).glob("cos-*.schema.json")):
        schema = json.loads(path.read_text())
        contract = schema.get("properties", {}).get("contract", {}).get("const")
        if contract:
            out[contract] = schema
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
                               "destination", "contract", "language", "order", "about"})


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
