# ADR 064 — Context Runtime: Manage the Window, Don't Just Fill It

> Date: 2026-06-18
> Status: PROPOSED — deferred; seed for a future phase plan (see "Sequencing")
> Phase: TBD (post-41, post-42a/42d)
> Depends on: Phase 41 (thinking engine / agentic chat loop — ADR 052, ADR 062),
>             Phase 42 (Heka → pgvector/Postgres: 42a crate, 42b LongMemEval/LoCoMo,
>             42d hybrid pgvec+VSA+KG retrieval, 42e Heka MCP), redb→Postgres
>             migration (Anima WI `019e5982-fa80-7731-b937-9acaa5fe4dbe`)
> Related: ADR 024 (ChatProcessor unification), ADR 025 (VSA dialogue system),
>          ADR 037 (LLM-VSA deep integration), ADR 047 (Seshat's Archive)
> Research: [`docs/ai/research/2026-06-18-lean-ctx-context-management-lessons.md`](../research/2026-06-18-lean-ctx-context-management-lessons.md)

## Context

akh-medu assembles LLM/OODA context today with strong *knowledge* selection but
naive *conversation* and *memory* handling. The research note details the sweep;
the load-bearing findings:

- Conversation turns live in a raw FIFO-10 ring buffer
  (`src/agent/conversation.rs:86`) and are **never injected into the LLM prompt** —
  only psyche + KG context are.
- Token budgets are three independent hardcoded constants — 1 800 tok psyche
  (`src/compartment/prompt.rs:13`), 8 000 char KG block (`src/nlu/kg_context.rs:19`),
  10-fact creative cap (`src/chat.rs:1263`) — with no shared allocator and a
  `len/4` token estimator (`src/compartment/prompt.rs:314`). They can collectively
  overflow the real window with nothing to arbitrate.
- The KG context block is relevance-scored well (keyword + VSA + bounded triples,
  `src/nlu/kg_context.rs:91`) but built **once at parse time** and never
  delta-refreshed across OODA cycles.
- Heka episodic memory (`crates/heka/src/drawer.rs:30`) is a write-mostly drawer:
  no consolidation, decay, wakeup, or embedding recall — the proximate cause of
  recurring "agent_recall returns nothing" incidents (Anima WI `019e587a-afb9`).

We took [lean-ctx](https://leanctx.com/docs) — a purpose-built "context runtime
for AI agents" — as a design mirror. Its organizing principle is that **context
is a managed resource rendered at graduated fidelity, not a binary
include/exclude decision**, expressed as: graduated read modes (P1), reference +
dedup (P2), delta-only updates (P3), task-relative assembly (P4), a managed +
measured budget (P5), a layered memory with consolidation/decay/wakeup (P6),
progressive capability disclosure (P7), and graph-driven relevance (P8). akh
already implements the *hardest* of these (P4, P8 via VSA similarity and
spreading activation) — but only on the knowledge path.

Crucially, this overlaps two efforts already in flight, and the ADR's value is
in **drawing the line between them**:

- **Phase 42** is already building the long-term *retrieval* layer (P6 storage):
  pgvector-backed Heka, hybrid pgvec+VSA+KG recall, a `remember`/`recall` MCP
  surface, benchmarked by LongMemEval/LoCoMo, on the redb→Postgres migration and
  the Seshat (Phase 37) pgvector substrate.
- **Phase 41** is building the agentic chat *loop* (ADR 062) — but not a managed
  context budget or a graduated-fidelity transcript.
- **No current phase owns the working-context *assembly* layer** that decides
  what, of everything retrievable, enters the window — at what fidelity, within
  what budget. That gap is this ADR's subject.

## Decision

Adopt lean-ctx's "manage the window" posture for akh's own runtime, **scoped to
the working-context assembly layer** and explicitly *not* re-implementing Phase
42's retrieval. Concretely, when this becomes a phase:

1. **Graduated fidelity (P1)** for conversation/episodes/facts: full →
   summary → reference-handle → dropped, chosen by relevance × recency ×
   information — replacing FIFO-10 eviction and `truncate(10)`.
2. **One managed context budget (P5)**: a single allocator dividing the window
   across {psyche, KG block, conversation history, recalled episodes} by
   priority, co-designed with Phase 41's chat loop; replace the `len/4` estimator
   with the real tokenizer.
3. **Delta-refreshed KG context (P3)**: extend `KgContextCache` to emit per-cycle
   deltas instead of a stale once-built block.
4. **Discourse selection via spreading activation (P8)**: reuse the OODA Orient
   graph walk (`src/agent/ooda.rs:60`) to pick which prior turns/entities enter
   the prompt.

And as a **constraint on Phase 42, not new work here**: ensure Heka ships the P6
*behaviours* — `consolidate`, `decay`, `wakeup` — not merely CRUD, with
LongMemEval/LoCoMo (42b) as the regression guard. Progressive capability
disclosure (P7) and context handoff are noted as forward-looking, deferred to the
Phase 28/29–31/36–37 line.

### Explicitly out of scope
- Re-implementing pgvector storage or hybrid retrieval (Phase 42 owns this).
- Adopting lean-ctx the MCP server into akh's runtime — it is a *reference*, not a
  dependency. akh has equivalent primitives (VSA, spreading activation, KG).
- Any change before Phase 41 and Phase 42a/42d land.

## Consequences

**Positive**
- Closes the "forgets vs. drowns" failure mode at the assembly layer, complementing
  Phase 42's storage-layer fix.
- Draws a clean seam: Phase 42 = *what is retrievable*; this = *what enters the
  window, at what fidelity, within budget*. Prevents the two efforts from
  colliding or duplicating.
- Reuses akh's existing strengths (VSA, spreading activation) rather than adding
  machinery.

**Negative / risks**
- A second consumer of the budget allocator (chat + OODA) raises coupling with
  Phase 41; must be co-designed, not bolted on.
- Graduated-fidelity summarization needs an NL summarizer — cost/latency on the
  CPU-only candle path is unmeasured (see research §6).
- Premature if conversation history turns out not to improve NL quality — must
  measure first.

**Neutral**
- The eventual phase number slots after 42; no number is claimed now to avoid
  collision with the parallel ADR/phase numbering (ADRs already reach 063 on
  sibling worktrees).

## Sequencing (how this becomes a phase plan)

This ADR is deliberately a **seed**, not an actionable plan. Promote it to a
`docs/ai/plans/<date>-phaseNN-context-runtime.md` plan when **both** hold:

1. Phase 41 (agentic chat loop, ADR 062) has landed — there is a loop to assemble into.
2. Phase 42a + 42d have landed — the budget allocator can pull from real pgvector
   hybrid recall, not the write-mostly drawer.

The plan then covers research-note §4 lessons **1, 2, 4, 5** (the working-context
layer); lesson 3 rides Phase 42; lesson 6 stays with the Phase 28/29–31 line.
Validate against the LongMemEval/LoCoMo harness (42b) plus a new "does history
help NL quality?" measurement (research §6) before building.
