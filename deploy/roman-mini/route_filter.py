#!/usr/bin/env python3
"""Python evaluator of akheventd's `EventRoute.when` filters (Phase 53g PR 6).

Mirrors `crates/akh-session-services/src/route_filter.rs` exactly, so the
deployment can test the filters it renders (`full_akh.py`) against recorded
Laya decisions without running akheventd. Both evaluators read the shared
cases in `fixtures/route_filter_cases.json` in their tests.

Semantics (see the Rust module for the full table): pointers are RFC 6901
JSON pointers into the event `data`; a pointer that does not resolve makes
`eq`, `in`, `prefix` and `exists` false; `exists` is false for `null`; JSON
equality is exact (`0` differs from `0.0` and from `false`).
"""
from __future__ import annotations

_MISSING = object()


class FilterError(ValueError):
    """A filter that akheventd would refuse (same checks as `validate`)."""


def resolve(data, pointer: str):
    """The value `pointer` addresses in `data`, or `_MISSING`."""
    if pointer == "":
        return data
    if not pointer.startswith("/"):
        return _MISSING
    current = data
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                return _MISSING
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                return _MISSING
            index = int(token)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def json_equal(a, b) -> bool:
    """serde_json `Value` equality: bools, ints, floats and strings never mix."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, float) and isinstance(b, float):
        return a == b
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        return False
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def matches(filter_: dict, data) -> bool:
    """Whether `data` satisfies `filter_` (one-key dict: the operator)."""
    ((op, arg),) = filter_.items()
    if op == "eq":
        found = resolve(data, arg["pointer"])
        return found is not _MISSING and json_equal(found, arg["value"])
    if op == "in":
        found = resolve(data, arg["pointer"])
        return found is not _MISSING and any(json_equal(found, v) for v in arg["values"])
    if op == "prefix":
        found = resolve(data, arg["pointer"])
        return isinstance(found, str) and found.startswith(arg["value"])
    if op == "exists":
        found = resolve(data, arg["pointer"])
        return found is not _MISSING and found is not None
    if op == "not":
        return not matches(arg, data)
    if op == "all":
        return all(matches(f, data) for f in arg)
    if op == "any":
        return any(matches(f, data) for f in arg)
    raise FilterError(f"unknown operator {op!r}")


def _pointer_ok(pointer) -> bool:
    if not isinstance(pointer, str) or len(pointer.encode()) > 512:
        return False
    if pointer and not pointer.startswith("/"):
        return False
    return all(pointer[i + 1:i + 2] in ("0", "1") for i, c in enumerate(pointer) if c == "~")


def validate(filter_: dict, at: str = "when", depth: int = 1, nodes: list | None = None) -> None:
    """Raise FilterError where akheventd's `RouteFilter::validate` would."""
    nodes = nodes if nodes is not None else [0]
    nodes[0] += 1
    if nodes[0] > 256:
        raise FilterError("the filter has more than 256 nodes")
    if depth > 16:
        raise FilterError(f"{at}: nesting exceeds 16 levels")
    if not isinstance(filter_, dict) or len(filter_) != 1:
        raise FilterError(f"{at}: a filter is an object with exactly one operator")
    ((op, arg),) = filter_.items()
    here = f"{at}.{op}"
    if op in ("eq", "in", "prefix", "exists"):
        expected = {"eq": {"pointer", "value"}, "in": {"pointer", "values"},
                    "prefix": {"pointer", "value"}, "exists": {"pointer"}}[op]
        if not isinstance(arg, dict) or set(arg) != expected:
            raise FilterError(f"{here}: operands must be exactly {sorted(expected)}")
        if not _pointer_ok(arg["pointer"]):
            raise FilterError(f"{here}: {arg['pointer']!r} is not a JSON pointer")
        if op == "in" and (not isinstance(arg["values"], list) or not arg["values"] or len(arg["values"]) > 256):
            raise FilterError(f"{here}: `in` needs 1..256 values")
        if op == "prefix" and (not isinstance(arg["value"], str) or not arg["value"]
                               or len(arg["value"].encode()) > 512):
            raise FilterError(f"{here}: the prefix is empty or too long")
    elif op == "not":
        validate(arg, here, depth + 1, nodes)
    elif op in ("all", "any"):
        if not isinstance(arg, list) or not arg:
            raise FilterError(f"{here}: `{op}` has no members")
        if len(arg) > 256:
            raise FilterError(f"{here}: more than 256 members")
        for index, member in enumerate(arg):
            validate(member, f"{here}[{index}]", depth + 1, nodes)
    else:
        raise FilterError(f"{at}: unknown operator {op!r}")
