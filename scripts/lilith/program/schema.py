"""Record, token and manifest schemas of the guided Lilith program corpus.

`lilith/program-record@1`  untokenized JSONL, one training situation per line.
`lilith/program-tokens@1`  the tokenized archive another agent's trainer consumes.
`lilith/program-manifest@1` the frozen stage manifest.

Nothing here matches natural language. Every classification is a structural field
(a source name, a stage, an id) and the only name mapping is the operator-approved
voice table (dojo manual section 2a).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

RECORD_SCHEMA = "lilith/program-record@1"
TOKENS_SCHEMA = "lilith/program-tokens@1"
MANIFEST_SCHEMA = "lilith/program-manifest@1"
RUN_ID = "lilith-program-r1-20261003"
SNAPSHOT_UNTIL = "20261001T235959Z"
SNAPSHOT_EXCLUDED_FROM = "20261002T000000Z"
CONTEXT_LIMIT = 8192

STAGES = ("white", "yellow", "green", "brown", "black")
SOURCES = (
    "historical_receipt", "cos_transcript", "teacher_dojo", "c2_session", "c2_sibling", "dojo_sibling",
    "kata_drill", "sparring", "variation", "guidance", "lineage", "rehearsal", "onpolicy",
)
KINDS = ("sft", "preference")
SPLITS = ("train", "dev", "excluded")
ROLES = ("system", "user", "assistant", "tool")
VOICES = ("maker", "truthkeeper", "seeker", "challenger", "guardian", "sacred-clown", "steward", "weaver")

# Dojo manual 2a, operator decision 2026-09-27: the source corpus names -> the English crew names.
VOICE_MAP: Mapping[str, str] = {
    "creator": "maker", "sage": "truthkeeper", "explorer": "seeker", "rebel": "challenger",
    "caregiver": "guardian", "mirror": "sacred-clown", "witness": "steward", "teacher": "weaver",
}

RECORD_FIELDS = (
    "record_id", "schema", "source", "source_ref", "stage", "families", "voice", "kata", "language", "kind",
    "weight", "messages", "chosen_messages", "rejected_messages", "tools", "component_id", "split",
    "provenance", "verification",
)
TOKEN_FIELDS = ("record_id", "stage", "kind", "weight", "source", "families", "voice", "language", "component_id", "split")


class ProgramError(Exception):
    """A diagnosable export failure: what failed and what the caller can do about it."""

    def __init__(self, message: str, help: str = ""):
        super().__init__(message)
        self.help = help

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base}\n  help: {self.help}" if self.help else base


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(value: Any) -> str:
    data = value if isinstance(value, bytes) else (value if isinstance(value, str) else canonical(value)).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_voice(name: str | None) -> str | None:
    """Source voice names map through the approved table; canonical names pass; anything else is an error."""
    if name is None:
        return None
    lowered = name.strip().lower()
    if lowered in VOICE_MAP:
        return VOICE_MAP[lowered]
    if lowered in VOICES:
        return lowered
    raise ProgramError(f"unknown voice {name!r}", help="extend VOICE_MAP only with an operator-approved name")


# -- messages ---------------------------------------------------------------------------------


def system(text: str) -> dict:
    return {"role": "system", "content": text}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str, *, train: bool = True) -> dict:
    msg: dict = {"role": "assistant", "content": text}
    if not train:
        msg["train"] = False
    return msg


def assistant_calls(calls: Sequence[tuple[str, str, Any]], text: str = "", *, train: bool = True) -> dict:
    """An assistant action with tool calls: (call_id, tool name, arguments)."""
    msg: dict = {"role": "assistant", "content": text, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": dict(args) if isinstance(args, Mapping) else args}}
        for cid, name, args in calls]}
    if not train:
        msg["train"] = False
    return msg


def tool_message(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def trained(message: Mapping[str, Any]) -> bool:
    """Only assistant messages carry loss; `train: false` marks context-only assistant steps."""
    return message.get("role") == "assistant" and message.get("train", True) is not False


def tool_response_hashes(messages: Iterable[Mapping[str, Any]]) -> list[str]:
    return [sha256_hex(m["content"]) for m in messages if m.get("role") == "tool" and isinstance(m.get("content"), str)]


# -- records ----------------------------------------------------------------------------------


def identity_payload(rec: Mapping[str, Any]) -> dict:
    keys = ("schema", "source", "stage", "kind", "families", "voice", "kata", "language",
            "messages", "chosen_messages", "rejected_messages", "tools")
    return {k: rec.get(k) for k in keys}


def make_record(*, source: str, source_ref: Mapping[str, Any] | str, stage: str, families: Sequence[str], language: str,
                kind: str = "sft", weight: float = 1.0, messages: Sequence[Mapping[str, Any]] | None = None,
                chosen_messages: Sequence[Mapping[str, Any]] | None = None,
                rejected_messages: Sequence[Mapping[str, Any]] | None = None, tools: Sequence[Mapping[str, Any]] | None = None,
                voice: str | None = None, kata: str | None = None, provenance: Mapping[str, Any] | None = None,
                verification: Mapping[str, Any] | None = None, component_hint: str | None = None) -> dict:
    rec: dict = {
        "schema": RECORD_SCHEMA, "source": source, "source_ref": source_ref if isinstance(source_ref, str) else dict(source_ref),
        "stage": stage, "families": list(families), "voice": canonical_voice(voice), "kata": kata, "language": language,
        "kind": kind, "weight": float(weight),
        "messages": [dict(m) for m in messages] if messages is not None else None,
        "chosen_messages": [dict(m) for m in chosen_messages] if chosen_messages is not None else None,
        "rejected_messages": [dict(m) for m in rejected_messages] if rejected_messages is not None else None,
        "tools": [dict(t) for t in tools] if tools else [],
        "component_id": component_hint or "", "split": "train",
        "provenance": dict(provenance or {}), "verification": dict(verification or {}),
    }
    rec["record_id"] = sha256_hex(identity_payload(rec))
    return rec


def _check_messages(messages: Any, where: str, problems: list[str]) -> None:
    if not isinstance(messages, list) or not messages:
        problems.append(f"{where}: needs a non-empty message list")
        return
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in ROLES:
            problems.append(f"{where}[{i}]: role must be one of {ROLES}")
            continue
        if m["role"] == "assistant" and not (m.get("content") or m.get("tool_calls")):
            problems.append(f"{where}[{i}]: assistant message is empty")
        if m["role"] == "tool" and not m.get("tool_call_id"):
            problems.append(f"{where}[{i}]: tool message needs tool_call_id")
        if m["role"] != "assistant" and "tool_calls" in m:
            problems.append(f"{where}[{i}]: only assistant messages carry tool_calls")


def shared_prefix_length(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> int:
    n = 0
    for x, y in zip(a, b):
        if canonical(x) != canonical(y):
            break
        n += 1
    return n


def _prompt_length(messages: Sequence[Mapping[str, Any]]) -> int:
    k = len(messages)
    while k > 0 and messages[k - 1].get("role") in ("assistant", "tool"):
        k -= 1
    return k


def validate_record(rec: Mapping[str, Any]) -> list[str]:
    """Structural problems of a program record; empty means valid."""
    problems: list[str] = []
    if rec.get("schema") != RECORD_SCHEMA:
        problems.append(f"schema must be {RECORD_SCHEMA}")
    for key in RECORD_FIELDS:
        if key not in rec:
            problems.append(f"missing field {key}")
    if problems:
        return problems
    if rec["source"] not in SOURCES:
        problems.append(f"source {rec['source']!r} not in {SOURCES}")
    if rec["stage"] not in STAGES:
        problems.append(f"stage {rec['stage']!r} not in {STAGES}")
    if rec["kind"] not in KINDS:
        problems.append(f"kind {rec['kind']!r} not in {KINDS}")
    if rec["split"] not in SPLITS:
        problems.append(f"split {rec['split']!r} not in {SPLITS}")
    if rec["voice"] is not None and rec["voice"] not in VOICES:
        problems.append(f"voice {rec['voice']!r} is not canonical")
    if not isinstance(rec["families"], list) or not all(isinstance(f, str) for f in rec["families"]):
        problems.append("families must be a list of strings")
    if not isinstance(rec["language"], str) or not rec["language"]:
        problems.append("language must be a non-empty string")
    w = rec["weight"]
    if not isinstance(w, (int, float)) or isinstance(w, bool) or not 0.0 <= w <= 1.0:
        problems.append("weight must be within [0, 1]")
    if rec["kind"] == "sft":
        _check_messages(rec["messages"], "messages", problems)
        if rec["chosen_messages"] or rec["rejected_messages"]:
            problems.append("sft records carry messages only")
        if isinstance(w, (int, float)) and w <= 0:
            problems.append("an sft record needs a positive weight (weight 0 appears only as a rejected side)")
        if isinstance(rec["messages"], list) and not any(trained(m) for m in rec["messages"]):
            problems.append("sft record has no supervised assistant action")
    else:
        _check_messages(rec["chosen_messages"], "chosen_messages", problems)
        _check_messages(rec["rejected_messages"], "rejected_messages", problems)
        if rec["messages"]:
            problems.append("preference records carry chosen_messages and rejected_messages only")
        c, r = rec["chosen_messages"], rec["rejected_messages"]
        if isinstance(c, list) and isinstance(r, list) and c and r:
            kc, kr = _prompt_length(c), _prompt_length(r)
            if kc < 1 or kc != kr or shared_prefix_length(c, r) < kc:
                problems.append("chosen and rejected must share the same prompt, ending before the assistant action")
            if canonical(c) == canonical(r):
                problems.append("chosen and rejected are identical")
    if rec["record_id"] != sha256_hex(identity_payload(rec)):
        problems.append("record_id does not match the canonical content")
    return problems


def validate_tokens(tok: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    for key in TOKEN_FIELDS:
        if key not in tok:
            problems.append(f"missing field {key}")
    if problems:
        return problems
    if tok["kind"] == "sft":
        problems += _check_side(tok, "")
    elif tok["kind"] == "preference":
        for side in ("chosen", "rejected"):
            if not isinstance(tok.get(side), dict):
                problems.append(f"missing {side}")
            else:
                problems += _check_side(tok[side], side + ".")
        if not problems:
            a, b = tok["chosen"]["input_ids"], tok["rejected"]["input_ids"]
            n = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                n += 1
            first = next((i for i, m in enumerate(tok["chosen"]["target_mask"]) if m), len(a))
            if n < first:
                problems.append("chosen and rejected do not share their prompt tokens")
    else:
        problems.append("kind must be sft or preference")
    if tok["split"] not in ("train", "dev"):
        problems.append("a tokens archive holds train and dev records only")
    return problems


def _check_side(side: Mapping[str, Any], prefix: str) -> list[str]:
    ids, mask = side.get("input_ids"), side.get("target_mask")
    if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask) or not ids:
        return [f"{prefix}input_ids and target_mask must be equal-length non-empty lists"]
    if not any(mask):
        return [f"{prefix}target_mask supervises nothing"]
    if any(m not in (0, 1) for m in mask):
        return [f"{prefix}target_mask must hold 0 or 1"]
    return []
