---
type: research
status: closed
resolved: 2026-09-19
branch: research/seshat-ingest-contract
asset: assets/seshat-ingest-contract.md (on the branch, commit c61fcf0d)
---

## Question

What does Seshat actually accept on ingest and return on search, and what does
that imply for pushing a redacted derived projection of transcripts into it?

Worked from `crates/seshat/` source only — the live service answers 401 from this
box, so nothing here is live-validated.

## Resolution

Full findings with file:line citations for every claim:
[assets/seshat-ingest-contract.md](../assets/seshat-ingest-contract.md) (also on the research branch)

**Seshat already anticipated this use case.** `format: "agent_session_transcript"`
exists today (`crates/seshat/src/transcript.rs`) and does role-based extraction —
assistant text and tool results only, dropping ids, timestamps and tool-call
arguments. That is prior art the projection should start from rather than invent.

**The workable parts.** One ingest path, the MCP tool `seshat_ingest`
(`mcp/mod.rs:511-566` → `ingest::ingest_document`, `ingest.rs:92-170`); no REST
ingest route. Chunking is paragraph-aware, non-overlapping, 1500-char target,
embedded by `all-MiniLM-L6-v2` into 384-dim pgvector. Auth is a single shared
bearer secret (`auth.rs`), fail-closed, no registration needed. Crucially,
document-level `tags`/`metadata` JSONB **round-trips verbatim on every search
hit** — so our `(source, session_id, seq)` addressing can ride in `metadata` and
come back on a semantic hit. That is the handoff hook
[the projection ticket](seshat-projection.md) needed; there is no native
line-addressing field, and `chunk_index` counts extracted segments, not source
lines.

**Two gaps that change the design:**

1. **No upsert.** `seshat_ingest` is a raw `INSERT`. `pa_documents.id` is
   server-generated and `slug` auto-suffixes to *avoid* collisions. The only
   delete-then-reinsert upsert in the codebase (`scripts.rs:180-275`) is private
   to the ELIZA-script path and unreachable from the MCP tool. A re-runnable
   harvester therefore either **duplicates** documents or, with a stable slug,
   hits a raw unique-constraint error surfaced as a generic MCP internal error.
   Re-derivation was an explicit goal of [the harvest unit](harvest-unit.md), so
   this is a direct collision, not a detail.
2. **No deletion or expunge.** No MCP tool and no REST route removes anything.
   FK cascades make deletion clean *if you have direct database access*, and
   there is precedent in a one-off purge migration — but there is **no
   self-service retraction path**. A secret that reaches Seshat cannot be
   recalled through Seshat's own interface.

**Other constraints worth carrying forward:** Seshat's transcript redaction is
role-based only, with no content or secret scanning — so redaction must happen
**before** the call, exactly as [the secrets posture](secrets-posture.md)
specifies. A whole-document JSONL parse failure aborts the entire ingest, which
interacts badly with [format drift](source-format-stability.md). The embed worker
is async, so hits are invisible for ~30s unless `seshat_embed_now` is called.
Body-size and rate limits were not found in Seshat's source and are marked an
unverified gap rather than an absence.

## Consequences

**Outcome (2026-09-19):** these findings triggered the reopening condition on
[the read surface](read-surface.md), which was revised to **collapse to a single
store**. Seshat is no longer a destination for transcripts — but the research is
not wasted: the chunker, the `all-MiniLM-L6-v2` embedder and the
`agent_session_transcript` extraction format documented here are exactly what the
harvester now reuses **as a library**. The two consequences below are superseded.

- [Seshat projection](seshat-projection.md) inherits both gaps as required
  design work, and gains a concrete answer for the pointer-back question.
- [Read surface](read-surface.md) is **flagged for reconsideration**: it retained
  Seshat on the assumption that its cost was sync and dedup work. "A leaked
  secret cannot be retracted" is a different class of cost, and that ticket said
  explicitly that this finding would reopen it. Not reopened unilaterally — the
  decision is the user's.
