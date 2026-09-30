"""Cumulative context budgeting for the in-process dojo runner (ADR 116, Anima WI 01a0e270-475e).

The Broker caps one tool result (`RunnerConfig.max_result_chars`) but nothing bounded the growing
conversation: a chase episode on the frozen parent overflowed a 65K-token server window through the
sum of several individually acceptable results. This module mirrors the production seam
(`src/agent/context/tool_budget.rs`, PR #458): before a tool result joins the conversation, estimate
what the next request already holds (system prompt + tool schemas + the conversation so far), and
cap the result to what is left of the window minus the output reserve, with the same conservative
constants: `len/4` (ceil) to count what is committed, 3 characters per token to size a cut of
JSON, an 85% safety margin, 512 characters reserved for the marker, a 3/4 head with the rest as tail.

A cut result carries a structured marker (data, not prose) under the `_akh_tool_result_truncated`
key, with the tool's own narrowing parameters named by API identifier. A broker-rendered result
(`{"isError", "_meta", "result"}`) is cut inside its envelope: `_meta` (receipt id and digest) always
survives, so a cut result can still be cited even when the window is spent, and the message stays one
JSON object. Any other text is cut head/tail with the marker appended. The receipt and its digest
always cover the full recorded response.

The ACP teacher arm does not use this: its akh bounds results itself with #458 against its own configured
`context_window`.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any, Mapping, Sequence

from dojo.model_adapters import ChatMessage, ToolSpec

DEFAULT_CONTEXT_WINDOW = 49_152  # tokens; the serve pod runs 65,536 with headroom for counter drift
DEFAULT_RESERVED_OUTPUT = 4_096  # tokens; matches the runner's default completion cap per call
JSON_CHARS_PER_TOKEN = 3
SAFETY_MARGIN_NUM, SAFETY_MARGIN_DEN = 85, 100
RESERVED_MARKER_CHARS = 512
HEAD_FRACTION_NUM, HEAD_FRACTION_DEN = 3, 4
MIN_USEFUL_TAIL_CHARS = 256
MARKER_KEY = "_akh_tool_result_truncated"
# API-identifier substrings that mark a tool parameter as narrowing or paging (same set as
# `NARROWING_PARAM_MARKERS` in tool_budget.rs); matched only against the tool's own schema names.
NARROWING_PARAM_MARKERS = ("page_token", "cursor", "limit", "offset", "top_k", "max_results", "page_size",
                           "query", "filter")


def estimate_tokens(text: str) -> int:
    """`HeuristicTokenCounter`: ceil(len_bytes / 4)."""
    return -(-len(text.encode("utf-8")) // 4)


def estimate_running_tokens(messages: Sequence[ChatMessage], specs: Sequence[ToolSpec]) -> int:
    """Everything already committed to the next request: system + user + assistant + tool messages,
    the tool calls the assistant made, and the offered tool schemas."""
    total = 0
    if specs:
        total += estimate_tokens(json.dumps([[s.name, s.description, s.schema()] for s in specs], ensure_ascii=False))
    for message in messages:
        total += estimate_tokens(message.content)
        if message.tool_calls:
            total += estimate_tokens(json.dumps(
                [{"id": c.id, "name": c.name, "arguments": dict(c.arguments or {})} for c in message.tool_calls],
                ensure_ascii=False))
    return total


def narrowing_params(spec: ToolSpec | None) -> list[str]:
    if spec is None:
        return []
    properties = spec.schema().get("properties") or {}
    return sorted(name for name in properties if any(m in name.lower() for m in NARROWING_PARAM_MARKERS))


@dataclasses.dataclass(frozen=True)
class Truncation:
    tool: str
    original_bytes: int
    kept_bytes: int
    original_tokens_est: int
    kept_tokens_est: int
    narrowing_params: tuple[str, ...]

    def marker(self) -> dict:
        return {MARKER_KEY: {"tool": self.tool, "original_bytes": self.original_bytes,
                             "kept_bytes": self.kept_bytes, "original_tokens_est": self.original_tokens_est,
                             "kept_tokens_est": self.kept_tokens_est,
                             "narrowing_params": list(self.narrowing_params), "reason": "context_window_budget"}}

    def to_dict(self) -> dict:
        return self.marker()[MARKER_KEY] | {"tool": self.tool}


@dataclasses.dataclass(frozen=True)
class BoundedResult:
    content: str
    truncation: Truncation | None = None


def _head_tail(text: str, budget: int) -> str:
    """At most `budget` UTF-8 bytes of `text`: a head, and a tail when enough is left. Never splits a character."""
    raw = text.encode("utf-8")
    if len(raw) <= budget:
        return text
    if budget <= 0:
        return ""
    head = budget * HEAD_FRACTION_NUM // HEAD_FRACTION_DEN
    tail = budget - head
    head_text = raw[:head].decode("utf-8", errors="ignore")
    if tail < MIN_USEFUL_TAIL_CHARS:
        return head_text
    tail_text = raw[len(raw) - tail:].decode("utf-8", errors="ignore")
    return f"{head_text}\n\n{tail_text}"


def bound_tool_result(tool: str, spec: ToolSpec | None, result: str, used_tokens: int,
                      window: int, reserved_output: int) -> BoundedResult:
    """Cap `result` to what is left of `window` after `used_tokens` and the output reserve."""
    available = max(0, window - reserved_output)
    safe_remaining = max(0, available - used_tokens) * SAFETY_MARGIN_NUM // SAFETY_MARGIN_DEN
    original = estimate_tokens(result)
    if original <= safe_remaining:
        return BoundedResult(result)
    char_budget = safe_remaining * JSON_CHARS_PER_TOKEN
    envelope = _envelope(result)
    if envelope is not None:
        kept_payload = _head_tail(json.dumps(envelope["result"], ensure_ascii=False),
                                  max(0, char_budget - RESERVED_MARKER_CHARS
                                      - len(json.dumps(envelope["_meta"], ensure_ascii=False))))
        make = lambda truncation: json.dumps(  # noqa: E731
            {"isError": envelope["isError"], "_meta": envelope["_meta"], "result_truncated": kept_payload,
             **truncation.marker()}, ensure_ascii=False)
    else:
        kept_payload = _head_tail(result, max(0, char_budget - RESERVED_MARKER_CHARS))
        make = lambda truncation: f"{kept_payload}\n\n{json.dumps(truncation.marker(), ensure_ascii=False)}"  # noqa: E731
    truncation = Truncation(tool, len(result.encode("utf-8")), len(kept_payload.encode("utf-8")), original,
                            estimate_tokens(kept_payload), tuple(narrowing_params(spec)))
    return BoundedResult(make(truncation), truncation)


def _envelope(text: str) -> Mapping[str, Any] | None:
    """The broker's rendering of a tool result, when `text` is one."""
    try:
        decoded = json.loads(text)
    except ValueError:
        return None
    if isinstance(decoded, dict) and isinstance(decoded.get("_meta"), dict) and "result" in decoded and "isError" in decoded:
        return decoded
    return None
