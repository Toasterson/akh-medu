# Phase 44 — Working-Context Assembly Runtime

- **Status:** **44a + 44f landed; measure-first gate CLEARED for 44b/44c/44e**
  (2026-06-20, `src/agent/context/`). Landed: the `ContextManager` seam
  (byte-identical `PassThroughManager`) + `TokenCounter` + `BudgetAllocator`/
  telemetry, and provenance `Retention` + `SafetyEnforced`. Measure-first results
  ([`docs/ai/2026-06-20-phase44-measure-first.md`](../2026-06-20-phase44-measure-first.md)):
  **Q1** history demonstrably helps (live coreference A/B) → build 44c/44e; **Q2**
  the **tool catalog dominates the window** (~6540 of gemma4's 8192 tokens; cloud
  unaffected) → build 44b (budget/trim tools first; route the catalog *through* the
  manager — it's currently added in the backend, downstream of the seam); **Q3**
  recall baseline = KG symbol-overlap (Phase 42 absent), precision measurement
  deferred to the 44g build. Next build order: **44b → 44c → 44e** (the core
  `CompactingManager`), then 44d/44g/44h–44j. Two corrections baked into 44a:
  Phase 42 "Heka"/pgvector is absent (44e/g consume the existing KG recall), and
  no tokenizer reaches the agent path (hence the `TokenCounter` abstraction).
- **Date:** 2026-06-18 (updated 2026-06-20)
- **ADR:** [`docs/ai/decisions/065-context-assembly-runtime.md`](../decisions/065-context-assembly-runtime.md)
- **Promotes:** [ADR 064 — Context Runtime](../decisions/064-context-runtime-lessons.md)
  (lessons 1/2/4/5) + its [research](../research/2026-06-18-lean-ctx-context-management-lessons.md)
- **Builds on:** ADR 062 (agentic loop — *implemented*; see
  [ADR 066](../decisions/066-llm-agent-loop-reconciliation.md)) · develop Phase 42a/42d
  (Heka→pgvector retrieval) · provenance ledger + TMS (Phase 9) · episodic memory (Phase 8)
- **Phase number provisional** — parallel worktrees; confirm against develop before landing.
- **Anima WI:** _(file on ingest)_

## TL;DR

Develop already has the two layers below this one: the **agentic loop** that lets
the LLM act (ADR 062, built) and the **retrieval layer** that stores/recalls
memory (Phase 42 Heka→pgvector, in flight). What no phase owns — and what
ADR 064 deferred as a seed — is the **assembly layer**: deciding *what, of
everything retrievable, enters the window, at what fidelity, within what budget*.
Phase 44 builds it: graduated fidelity (replace FIFO-10 + `truncate(10)`), one
budget allocator (replace three hardcoded caps + the `len/4` estimator),
delta-refreshed KG context, spreading-activation discourse selection — **plus**
the additions from this session's audit: provenance-derived `Retention` +
`SafetyEnforced`, episodic session-of-the-past seeding, and VSA-ranked recall.
It is **measure-first**: confirm conversation history even helps NL quality
before building the quality-driven parts.

## Scope (from ADR 065)

The assembly layer is built behind a swappable **`ContextManager` trait**
(ADR 065 D9) that the loop (ADR 062) calls to build each prompt:

- **In:** the trait + loop integration (44a), then the default
  **`CompactingManager`** = ADR 064 lessons 1/2/4/5 + additions D6–D8
  (44b–44g), then the **`PassiveManager`** and **`VirtualContextManager`**
  strategies (44h–44i) and an **A/B harness** (44j) — all behind `SafetyEnforced`.
- **Specced but deferred (two axes):** **`KvCachePolicy`** token-level KV
  retention (ADR 065 D10, recorded as 44k) — Candle-only, gated on Candle being
  the chat backend (ADR 037); and **`RetrievalPolicy`** — *how* context is pulled
  in (ADR 065 D11, recorded as 44l) — owned in practice by Phase 42 Heka, built
  as a swap-axis only if retrieval proves the bottleneck.
- **Out:** retrieval/storage (Phase 42 Heka — consumed, not built); the loop
  itself (ADR 062); progressive disclosure + handoff (Phase 28/29–31/36–37).

## Sequencing — measure first

Do **not** build the quality-driven sub-phases until these are answered
(ADR 064 §6 / research §6):

1. **Does conversation history improve akh's NL quality?** The chat path omits it
   today — confirm the win on a fixed prompt set before 44b/44e.
2. **Real token window** of the deployed Gemma/candle config, and what the three
   current caps (`prompt.rs:13` + `kg_context.rs:19` + `chat.rs:1263`) sum to in
   practice — sizes the allocator.
3. **VSA vs pgvector for episodic recall** — answered by Phase 42b's
   LongMemEval/LoCoMo harness; the allocator consumes the winner.

44a (budget instrumentation) and 44f (safety pinning — correctness, not quality)
may proceed before the measurements; the rest gate on them.

## Phased breakdown

> 44a is the foundation; **44b–44g build the default `CompactingManager`**;
> 44h–44i add the other two strategies; 44j is the A/B harness; 44k is the
> deferred KV axis.

### 44a — `ContextManager` trait + loop integration + budget instrumentation [foundation]

**New** `src/agent/context/` (`mod.rs`, `manager.rs`, `types.rs`); **touches**
[`src/agent/agentic_chat.rs`](../../src/agent/agentic_chat.rs),
[`src/agent/model_manager.rs`](../../src/agent/model_manager.rs),
[`src/compartment/prompt.rs`](../../src/compartment/prompt.rs),
[`src/nlu/kg_context.rs`](../../src/nlu/kg_context.rs).

- Define `ContextManager` (ADR 065 D9): `assemble(episode, budget) ->
  AssembledContext`, `commit(outcome)`; plus `EpisodeState`, `TokenBudget`,
  `AssembledContext`, `StrategyId`.
- **Integration point:** `run_agentic_turn` / `AgentModelManager::chat_with_tools`
  call `assemble`/`commit` instead of building the prompt ad hoc. Land a
  **pass-through manager first** (current behavior, byte-identical prompt) so the
  seam exists with zero change.
- Replace `estimate_tokens = len/4` (`prompt.rs:314`) with the real tokenizer;
  instrument the three caps (psyche 1 800 tok `prompt.rs:13`, KG 8 000 char
  `kg_context.rs:19`, 10-fact `chat.rs:1263`) to answer Sequencing Q2.

**Acceptance:** the loop produces a byte-identical prompt via the pass-through
manager; per-turn telemetry reports real token usage per context section;
tokenizer within a few % of the model's count.

### 44b — One managed budget allocator (ADR 064 P5 / lesson 2)

**New** `src/agent/context/budget.rs`; co-designed with the loop's prompt build
(`AgentModelManager::chat_with_tools`, ADR 062).

- A single allocator divides the window across `{psyche, KG block, conversation
  history, recalled episodes}` by priority, replacing the isolated caps.
- The loop calls it when assembling the prompt; OODA's context build shares it.

**Acceptance:** total assembled prompt stays within the measured window (Q2);
sections degrade by priority under pressure, not by independent truncation;
golden test on a budget-exceeding input.

### 44c — Graduated fidelity (ADR 064 P1 / lesson 1)

**Touches** [`src/agent/conversation.rs`](../../src/agent/conversation.rs),
[`src/chat.rs`](../../src/chat.rs); **new** `src/agent/context/fidelity.rs`.

- Render a turn/fact/episode at **full → one-line summary → reference-handle →
  dropped**, chosen by relevance × recency × information; replaces FIFO-10
  eviction (`conversation.rs:86`) and `facts.truncate(10)` (`chat.rs:1263`).
- Summarizer per ADR 065 OQ1 (extractive first; model-based if measured worth it).

**Acceptance:** a long transcript stays under budget with goals/decisions
preserved at full fidelity and filler demoted to summary/handle; no hard cliff.

### 44d — Delta-refreshed KG context (ADR 064 P3 / lesson 4)

**Touches** [`src/nlu/kg_context.rs`](../../src/nlu/kg_context.rs) (`KgContextCache`).

- Emit per-cycle deltas (newly spreading-activation-surfaced triples) instead of
  the once-built block (`kg_context.rs:91`) that goes stale across cycles.

**Acceptance:** across N OODA cycles the KG block reflects newly activated
entities without rebuilding from scratch; cache hit-rate maintained.

### 44e — Spreading-activation discourse selection + session seeding (ADR 064 P8 + ADR 065 D7)

**Touches** [`src/agent/ooda.rs`](../../src/agent/ooda.rs) (Orient walk, `:60`),
episodic memory; **new** `src/agent/context/seed.rs`.

- Select which prior turns/entities enter the prompt by graph proximity (reuse
  the Orient spreading-activation walk), not last-N replay.
- Seed the window from `restore_session_summary` (wakeup) + `recall_episodes`
  ranked against current goal/entities. Wakeup/recall *behaviours* come from
  Phase 42 Heka; this wires their output into the prompt.

**Acceptance:** a turn in session 2 surfaces a relevant fact from session 1;
discourse selection beats last-N on a fixed relevance set.

### 44f — Provenance-derived `Retention` + `SafetyEnforced` (ADR 065 D6) — can land early

**New** `src/agent/context/retention.rs`.

- `Retention { Volatile, Recoverable, PinnedSafety }` + a
  `ProvenanceRetentionClassifier` reading the ledger kind-index
  ([`src/provenance.rs`](../../src/provenance.rs)) + danger metadata
  ([`src/agent/tool_manifest.rs`](../../src/agent/tool_manifest.rs)).
- `SafetyEnforced` decorator: assert every `PinnedSafety` span survives assembly
  → hard error otherwise. Complements ADR-060's *execution* gate (this guards
  *eviction*). Counter for caught violations.

**Acceptance:** an assembler that would drop an `AgentDecision`/`ShadowVeto`/
danger-tool span is rejected; classification unit-tested per rule; runs
independently of the quality sub-phases.

### 44g — VSA-ranked recall + TMS hygiene (ADR 065 D8)

**Touches** episodic recall, [`src/tms.rs`](../../src/tms.rs).

- Rank recalled episodes by VSA similarity blended with recency/confidence;
  consume whichever recall Phase 42b proves best (VSA vs pgvector).
- Use TMS `effective_confidence`/`retract` so superseded facts stay out of the
  window (supersede-don't-delete).

**Acceptance:** recall ranking improves precision vs. the symbol-overlap baseline
on a fixed query set; superseded facts are absent from the assembled window.

> 44a–44g above are the foundation + the default `CompactingManager`. 44h–44i add
> the other two `ContextManager` strategies; 44j compares them; 44k is deferred.

### 44h — `PassiveManager` strategy (ADR 065 D9) [background]

**New** `src/agent/context/passive.rs`; reuses the consolidation pipeline
(`DerivationKind::AgentConsolidation`) and **rides Phase 42 Heka** (consolidate
behaviour).

- Main loop runs a plain transcript and never self-edits; a background pass
  extracts salient entities/relations and writes them to memory via consolidation.
- Dedup via TMS support sets + `memory_sync`-style idempotency anchors.

**Acceptance:** transcript facts appear in memory with `AgentConsolidation`
provenance without spending main-loop tokens; re-running the extractor is
idempotent.

### 44i — `VirtualContextManager` strategy (ADR 065 D9) [model self-edit]

**New** `src/agent/context/virtual.rs` + self-edit tools under
[`src/agent/tools/`](../../src/agent/tools/), called through the **ADR 062 tool
loop**.

- Tiers: core memory (pinned system block), recall (recent history), archival
  (episodic memory); self-edit tools `core_memory_append`/`replace`,
  `archival_insert` (≙ `persist_episode`), `archival_search` (≙ `recall_episodes`).
- Wrapped in `SafetyEnforced`; **`PassiveManager` runs in parallel as the safety
  net** so a missed self-edit never permanently drops a fact. Note-taking is a
  degenerate single-append-tool config, not a separate manager.

**Acceptance:** self-edits mutate core/archival memory and survive across turns;
with the passive net on, lost-salient-fact rate ≈ 0; `SafetyEnforced` blocks any
self-edit that would drop a `PinnedSafety` span.

### 44j — A/B experiment harness (ADR 065 D9)

**New** `src/bench_phase44/` (mirror `bench_phase41g` layout).

Run `SafetyEnforced<{Compacting, Passive, Virtual}>` behind one trait object,
same loop, same traces. Measure: task quality, token cost per task (Virtual's
self-edit overhead), retrieval precision/recall, context-rot incidents,
`safety_pin_violation_attempts` caught (must reach the backend as 0),
lost-salient-fact rate for Virtual with vs. without the passive net.

**Acceptance:** a comparison table across the three managers on a fixed trace
set, written as a dated report under `docs/ai/`.

### 44k — `KvCachePolicy` — deferred (ADR 065 D10), recorded not built

Token-level KV retention, **Candle-only**, gated on Candle being the chat backend
(ADR 037). `StreamingSink { sinks, window }` and `HeavyHitter { budget, recency }`;
KV quantization / paged attention are backend *config*, not policy. Today the loop
is Ollama → llama.cpp owns the KV cache (context-shift ≈ StreamingLLM for free),
so an akh-controlled policy is unreachable. **Build when Candle lands in the loop**
(ADR 037); cross with the three managers on the harness (44j) then.

### 44l — `RetrievalPolicy` — deferred fourth axis (ADR 065 D11), recorded not built

*How* context is pulled in, independent of curation and KV retention.
`JustInTime` (VSA-ranked `kg_query` — the current default, = 44g), `PreloadedRag`
(= Phase 42 pgvector hybrid recall), `TemporalKg` (Graphiti-style bitemporal
validity windows — the future home of the bitemporal intervals deferred in favour
of TMS). Retrieval *proper* is **Phase 42 Heka**; this assembler consumes whatever
recall wins (42b). Build the swap-axis **only if measurement shows retrieval, not
curation, is the bottleneck** (RFD §3.7).

## Dependency order

44a (foundation) → 44b → (44c, 44d, 44e) → 44g — these build the
`CompactingManager`. 44f (safety pinning, correctness) is independent and can land
first. Then 44h (Passive) and 44i (Virtual, needs 44f's `SafetyEnforced` + 44h as
net) → 44j (harness). 44b–44e/44g/44h–44j gate on the §Sequencing measurements;
44a + 44f do not. 44k (KV) is deferred to the Candle-in-loop work (ADR 037);
44l (retrieval swap-axis) is deferred to Phase 42 + a measured retrieval
bottleneck. Both are recorded, not built this phase.

## Risks & mitigations

- **History doesn't help NL quality** → measure first (Sequencing Q1) before
  44b/44c/44e.
- **Summarizer cost on CPU/candle** → extractive first; model-based only if
  measured worth it (ADR 065 OQ1).
- **Budget allocator couples with the loop** → co-design with ADR 062's prompt
  build, not bolt-on (ADR 064 risk).
- **Per-span classifier latency** → ledger kind-index + per-episode cache.

## Validation

LongMemEval/LoCoMo (Phase 42b) as the recall regression guard, plus a new "does
history help NL quality?" measurement (Sequencing Q1) and a context-rot check
over long episodes.

## Doc-maintenance follow-ups (CLAUDE.md flow)

- Ingest ADR 065 + ADR 066 + this plan into Seshat (dated slugs).
- File an Anima work item for Phase 44 under Akh-Medu; link paths; promote when
  the measure-first gate clears.
- On first code landing: add `src/agent/context/` to `docs/ai/architecture.md`
  (bump date) and the CLAUDE.md roadmap; confirm the final phase/ADR numbers
  against develop.
