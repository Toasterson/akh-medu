---
type: grilling
status: closed
resolved: 2026-09-19
---

## Question

What does one row represent — what is the harvested unit?

## Resolution

**The source record, byte-exact, plus a derived normalized projection.** The
system-of-record row is the raw JSONL line (Claude Code, Codex) or the raw SQLite
row (OpenCode), keyed `(source, session_id, seq)`. A second layer derives typed
cross-tool events from it.

Two properties this buys, both load-bearing:

1. **The normalizer can be wrong.** It will be — see
   [source format stability](source-format-stability.md). Re-running derivation
   over retained verbatim rows costs nothing; re-harvesting from source files
   that have rotated or been deleted is impossible.
2. **"Line by line" stays literal** for the two JSONL sources, and OpenCode gets a
   synthetic `seq` so it fits the same addressing despite having no lines at all.

Rejected: normalized-only (a normalizer bug or a newly-interesting field becomes
unrecoverable); verbatim-only (every consumer re-learns three incompatible
schemas, and cross-tool questions get hard); semantic-turn-as-unit (cheapest to
embed, but sub-turn detail like an individual `tool_result` stops being
addressable — which is the stated requirement).

## Confirmed by research (2026-09-19)

[Source format stability](source-format-stability.md) found that Claude Code's
own documentation declares the JSONL format internal and liable to
*"break on any release"*, and that Codex and OpenCode publish no guarantee while
both have open issues showing real breakage in the wild. Verbatim capture is
therefore the load-bearing half of this decision, not the cautious half.
