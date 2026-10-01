"""Replay of her own episodic memory: the live `memory_recall` and `memory_write` tools over seeded memories.

`memory-epistemics` episodes seed her real C2 episodic memories (her earlier answers, verbatim, including the
fabricated ones). The `akh` replay server offers the live akh's two memory tools under their live names and
argument shapes (`src/agent/tools/memory_recall.rs`, `memory_write.rs`), so the model sees what she sees: a
recall prints each episode as `- [<label>] <stored summary>` with its facts and tags, newest first, ranked by how
many query terms the episode's text contains, with a recency fallback.

A recall is a retelling. Its receipt carries `ReceiptSource.MEMORY`: the dojo counts it as issued (a citation of
it is not a fabrication) but never as evidence of the tracker's state (`Receipt.is_read` is false). A
`memory_write` goes to the dojo tenancy ledger like every other write and the written episode becomes
recallable inside the same episode.

The server is offered only to episodes whose spec seeds memories, so every other family keeps its exact tool list.
In the broker the tools are `mcp__akh__memory_recall` and `mcp__akh__memory_write`: the dojo's convention for any
server (the teacher akh is sealed to session tools only, so its memory tools could not be offered natively).
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from dojo.episode import RecalledMemory, ReceiptSource, ToolReceipt
from dojo.replay_server import (
    DojoLedger,
    ReceiptLog,
    _McpServer,
)
from dojo.snapshot import Snapshot, canonical_json, sha256_hex

MEMORY_SERVER = "akh"
MEMORY_RECALL = "memory_recall"
MEMORY_WRITE = "memory_write"
DEFAULT_TOP_K = 5
MIN_TERM_CHARS = 3  # the live tool drops shorter tokens: they match noise
MISSING_ARGUMENT_REASON = "missing-argument"

# The live descriptions and parameter names, verbatim. All live parameters are strings (`ToolInput`); `top_k` and the
# array parameters are left untyped here so a model that sends a number or an array is not refused for its JSON type.
MEMORY_TOOLS: list[dict] = [
    {
        "name": MEMORY_RECALL,
        "description": (
            "Recall episodic memories matching query terms (fuzzy -- terms need not be exact symbol names; "
            "'cluster' matches `cluster-topology`). Returns each episode's stored FACTS and tags inline, so you can "
            "answer directly without further kg_query drill-down. Recency fallback if nothing matches."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_symbols": {"type": "string", "description": (
                    "Comma-separated symbol names or IDs to query. Generic terms like 'recent' trigger a temporal "
                    "scan of all episodes.")},
                "top_k": {"description": "Maximum number of episodes to return (default: 5)."},
            },
            "required": ["query_symbols"],
        },
    },
    {
        "name": MEMORY_WRITE,
        "description": (
            "Record one episodic memory: a summary plus distilled facts. Use this (not a raw-text ingester) -- read "
            "the source yourself, then emit the structured episode."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "One-sentence summary of what happened / was learned."},
                "facts": {"description": "Array of {subject, predicate, object, confidence?} triples."},
                "tags": {"description": "Array of short topical tags."},
                "source": {"type": "string", "description": (
                    "Where this memory came from (e.g. a file path). Used for provenance and idempotency.")},
                "origin": {"type": "string", "description": '"openclaw" to attribute as a self-import, else "agent" (default).'},
                "timestamp": {"description": "Optional unix timestamp for deterministic episode identity."},
            },
            "required": ["summary"],
        },
    },
]


def render_recall(memories: Sequence[RecalledMemory], query: str, top_k: int) -> str:
    """The text the live `memory_recall` prints for these stored episodes (see the module docstring)."""
    terms = [t.strip().lower() for t in query.split(",")]
    terms = [t for t in terms if len(t) >= MIN_TERM_CHARS]
    newest_first = sorted(memories, key=lambda m: -m.order)

    def blob(m: RecalledMemory) -> str:
        return f"{m.label} {m.text} {' '.join(m.facts)} {' '.join(m.tags)}".lower()

    matched = [m for m in newest_first if any(t in blob(m) for t in terms)] if terms else []
    mode = "matched" if matched else ("recent (no symbol match)" if terms else "recent")
    pool = matched or newest_first
    if not pool:
        return "No episodic memories found."
    ranked = sorted(pool, key=lambda m: -sum(1 for t in terms if t in blob(m)))  # stable: recency order within a score
    shown = ranked[:top_k]
    lines = []
    for m in shown:
        facts = "; ".join(m.facts) if m.facts else "(none recorded)"
        lines.append(f"- [{m.label}] {m.text}\n    facts: {facts}\n    tags: {', '.join(m.tags)}")
    return f"Recalled {len(shown)} episode(s) [{mode}]:\n" + "\n".join(lines)


def _top_k(raw: Any) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_TOP_K
    return value if value > 0 else DEFAULT_TOP_K


def _slug(source: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-") or "agent"


class MemoryReplayServer(_McpServer):
    """`memory_recall` / `memory_write` over the spec's seeded memories. Deterministic: ids and digests only."""

    def __init__(self, snapshot: Snapshot, memories: Sequence[RecalledMemory], ledger: DojoLedger,
                 receipts: ReceiptLog | None = None):
        self.name = MEMORY_SERVER
        self.snapshot = snapshot
        self.ledger = ledger
        self.receipts = receipts or ReceiptLog()
        self.memories: list[RecalledMemory] = list(memories)

    def tools(self) -> list[dict]:
        return MEMORY_TOOLS

    def contract_sha256(self) -> str:
        return sha256_hex(canonical_json({"tools": MEMORY_TOOLS}))

    def call(self, name: str, arguments: Mapping[str, Any]) -> tuple[ToolReceipt, Any, bool]:
        if name == MEMORY_RECALL:
            query = arguments.get("query_symbols")
            if not isinstance(query, str) or not query.strip():
                return self._missing(name, arguments, "query_symbols")
            text = render_recall(self.memories, query, _top_k(arguments.get("top_k")))
            return self._issue(name, arguments, text, ReceiptSource.MEMORY), text, False
        summary = arguments.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            return self._missing(name, arguments, "summary")
        entry = self.ledger.append(self.snapshot.id, self.name, name, arguments)
        facts = arguments.get("facts") if isinstance(arguments.get("facts"), list) else []
        tags = arguments.get("tags") if isinstance(arguments.get("tags"), list) else []
        source = str(arguments.get("source") or "agent")
        label = f"episode:{_slug(source)}:{sha256_hex(canonical_json(dict(arguments)))[:16]}"
        order = max((m.order for m in self.memories), default=0) + 1
        fact_text = tuple(
            "; ".join(str(f.get(k)) for k in ("subject", "predicate", "object") if f.get(k) is not None)
            for f in facts if isinstance(f, dict))
        self.memories.append(RecalledMemory(label, summary, tuple(f"tag:{t}" for t in tags if isinstance(t, str)),
                                            fact_text, order))
        text = f'Recorded episode "{label}" ({len(facts)} fact(s), {len(tags)} tag(s)).'
        return self._issue(name, arguments, text, ReceiptSource.LEDGER, digest_of=entry.entry_sha256), text, False

    def _missing(self, name: str, arguments: Mapping[str, Any], argument: str) -> tuple[ToolReceipt, Any, bool]:
        payload = {"error": f"{name} needs {argument}", "reason": MISSING_ARGUMENT_REASON, "tool": name,
                   "arguments": dict(arguments)}
        receipt = self._issue(name, arguments, payload, ReceiptSource.ERROR, reason=MISSING_ARGUMENT_REASON)
        return receipt, payload, True

    def _issue(self, name: str, arguments: Mapping[str, Any], payload: Any, source: ReceiptSource,
               reason: str | None = None, digest_of: str | None = None) -> ToolReceipt:
        body = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, ensure_ascii=False)
        digest = digest_of or sha256_hex(body.encode("utf-8"))
        return self.receipts.issue(server=self.name, tool=name, arguments=arguments, response=payload,
                                   response_sha256=digest, source=source, snapshot=self.snapshot, error_reason=reason)


def memory_tool_entries() -> dict[str, dict]:
    return {t["name"]: t for t in MEMORY_TOOLS}
