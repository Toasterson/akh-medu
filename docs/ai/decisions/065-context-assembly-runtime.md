# ADR 065 — Working-Context Assembly Runtime (Phase 44)

> Date: 2026-06-18
> Status: PROPOSED
> Phase: 44 (number provisional — parallel worktrees; ADR 064 deliberately
>          claimed none. Confirm against develop before landing.)
> Supersedes: synthesizes (and replaces) this session's earlier context-management
>             drafts; the verbatim RFD/notes research is retained under
>             `docs/ai/research/2026-06-18-{rfd,context-management}-*.md`
> Depends on:
>   - **develop Phase 41 — the agentic loop is built** (ADR 062, *Implemented*):
>     `Agent::run_agentic_turn` / `OllamaClient::chat` with tools — the loop this
>     layer assembles into.
>   - **develop Phase 42a/42d — Heka → pgvector retrieval** (in flight): the
>     storage/recall layer this layer pulls from. Out of scope here.
>   - provenance ledger + TMS (Phase 9), episodic memory (Phase 8).
> Related: ADR 064 (Context Runtime — the seed this promotes; lessons 1/2/4/5),
>          ADR 062 (LLM agentic chat loop — reconciliation in ADR 066),
>          ADR 060 (tool sandbox + permission — the *execution* gate this
>          complements), ADR 047 (Seshat). Research:
>          [`../research/2026-06-18-lean-ctx-context-management-lessons.md`](../research/2026-06-18-lean-ctx-context-management-lessons.md)
>          (develop), and this session's
>          [`../research/2026-06-18-rfd-context-management.md`](../research/2026-06-18-rfd-context-management.md).
> Plan: [`../plans/2026-06-18-phase44-context-assembly-runtime.md`](../plans/2026-06-18-phase44-context-assembly-runtime.md)

## Context

[ADR 064](064-context-runtime-lessons.md) established the posture — *context is a
managed resource rendered at graduated fidelity, not a binary include/exclude
decision* — and, crucially, drew the seam between three efforts:

- **develop Phase 41** builds the agentic chat *loop* (ADR 062, now
  **Implemented**: `Agent::run_agentic_turn`, `OllamaClient::chat` with tools,
  `ollama_tool_catalog`, 8-round cap, ADR-060-gated tool execution, symbolic
  fallback).
- **develop Phase 42** builds the long-term *retrieval* layer (Heka → pgvector,
  hybrid pgvec+VSA+KG recall, LongMemEval/LoCoMo).
- **No phase owns the working-context *assembly* layer** — what, of everything
  retrievable, enters the window, at what fidelity, within what budget. ADR 064
  deferred itself as a *seed*, to be promoted once Phase 41 and Phase 42a/42d
  land.

Phase 41's loop has landed. Per ADR 064's own Sequencing, the seed is ready to
become a phase. **This ADR promotes it** — covering ADR 064 lessons 1, 2, 4, 5 —
and **enriches it with design ADR 064 does not contain**, drawn from a parallel
code audit of three subsystems the lean-ctx sweep did not cover: the
**provenance ledger** (104 `DerivationKind` variants, kind-indexed), the **TMS**
(`retract`/`effective_confidence`), and **episodic memory** (`consolidate`,
`recall_episodes`, `restore_session_summary`). The additions are
**provenance-derived retention**, a **`SafetyEnforced` assembly decorator**,
**episodic session-of-the-past seeding**, and **VSA-ranked recall**.

The recurring failure ADR 064/its research names — the agent either **forgets**
(`agent_recall` returns nothing, Anima WI `019e587a-afb9`) or **drowns** (raw
replay, no budget) — is closed at the *assembly* layer here, complementing Phase
42's storage-layer fix.

## Decision

Build the working-context assembly runtime as a layer the agentic loop (ADR 062)
calls to build its prompt, pulling from Phase 42's retrieval. Eleven decisions:
**D2–D5 promote ADR 064**; **D6–D8 are the additions**; **D9 fixes the runtime's
*structure* (the `ContextManager` trait + three swappable strategies — the
mechanisms above are how one of them is implemented); D10 records the deferred,
Candle-only KV-management axis; D11 records the deferred fourth axis,
`RetrievalPolicy`.**

### D1 — Scope: assembly only

This layer decides *what enters the window, at what fidelity, within budget*. It
does **not** own retrieval/storage (Phase 42 Heka) — it *consumes* whatever
recall wins (VSA vs pgvector, decided by LongMemEval/LoCoMo in 42b). It does
**not** own the loop (ADR 062) — it is *invoked by* the loop's prompt assembly.

### D2 — Graduated fidelity (ADR 064 P1, lesson 1)

A conversation turn / fact / episode is renderable **full → one-line summary →
reference-handle → dropped**, chosen by *relevance × recency × information* —
replacing the raw FIFO-10 eviction in `ConversationState`
([`src/agent/conversation.rs:86`](../../src/agent/conversation.rs), develop) and
the `facts.truncate(10)` creative cap
([`src/chat.rs:1263`](../../src/chat.rs), develop). This is the single
highest-leverage lesson.

### D3 — One managed budget allocator (ADR 064 P5, lesson 2)

A single allocator divides the real window across `{psyche prompt, KG block,
conversation history, recalled episodes}` by priority, replacing the three
isolated hardcoded caps — 1 800 tok psyche
([`src/compartment/prompt.rs:13`](../../src/compartment/prompt.rs)), 8 000 char
KG ([`src/nlu/kg_context.rs:19`](../../src/nlu/kg_context.rs)), 10-fact creative
cap (`chat.rs:1263`) — and upgrading the `len/4` estimator
([`src/compartment/prompt.rs:314`](../../src/compartment/prompt.rs)) to the real
tokenizer. **Co-designed with the loop (ADR 062)**: the allocator feeds the
prompt that `chat_with_tools` assembles, since that is where the window is built.

### D4 — Delta-refreshed KG context (ADR 064 P3, lesson 4)

Extend `KgContextCache`
([`src/nlu/kg_context.rs:40`](../../src/nlu/kg_context.rs)) to emit per-cycle
*deltas* (newly spreading-activation-surfaced triples) instead of a block built
once at parse time (`kg_context.rs:91`) and gone stale across cycles.

### D5 — Discourse selection via spreading activation (ADR 064 P8, lesson 5)

Point the OODA Orient graph walk
([`src/agent/ooda.rs:60`](../../src/agent/ooda.rs)) at *discourse*: select which
prior turns/entities enter the prompt by graph proximity to the current input's
entities, not "replay the last N". akh already owns this primitive for
knowledge; this applies it to conversation.

### D6 — Provenance-derived `Retention` + `SafetyEnforced` (addition)

Not in ADR 064; load-bearing because ADR 062 lets an LLM *act*, so the assembler
must never silently evict a safety-relevant span from the window.

```rust
pub enum Retention { Volatile, Recoverable, PinnedSafety }

/// Classifies a context span using the provenance ledger + danger metadata —
/// trusted symbolic signals, not a tag the neural policy controls.
pub trait RetentionClassifier { fn classify(&self, span: &ContextSpan, engine: &Engine) -> Retention; }
```

- `PinnedSafety` ⇐ provenance `kind` ∈ {`AgentDecision`, `ShadowVeto`,
  `PsycheEvolution`} or a tool-exec record with `danger_level ≥ Dangerous`, via
  the ledger's **kind-index**.
- `Recoverable` ⇐ has a resolvable `provenance_id` (re-fetchable) and not pinned.
- `Volatile` ⇐ everything else.

A `SafetyEnforced` decorator asserts every `PinnedSafety` span survives
assembly (hard error otherwise). This **complements ADR 060**: ADR 060 gates tool
*execution*; this gates what the assembler *drops*. Different layer, same
untrusted-policy boundary.

### D7 — Episodic session-of-the-past seeding (addition)

The assembler seeds the window from `restore_session_summary` (wakeup) +
`recall_episodes` ranked against the current goal/entities, plus the
`ConversationState` ring buffer — turning the stateless turn into one that
remembers prior sessions. The wakeup/recall *behaviours* ride Phase 42 Heka
(ADR 064 lesson 3); the *seeding into the prompt* is this layer.

### D8 — VSA-ranked recall + TMS hygiene (addition; consumes Phase 42)

Close the recall-ranking gap: rank recalled episodes by VSA similarity
(`Engine::search_similar`) blended with recency/confidence — but **consume
whichever recall Phase 42b's LongMemEval/LoCoMo proves best** (VSA vs pgvector),
per research §6. Use TMS `effective_confidence` / `retract` so superseded facts
are kept *out* of the window without schema surgery (supersede-don't-delete).

### D9 — Structure: a swappable `ContextManager` trait, 3 strategies, `SafetyEnforced`

The runtime is not a single hardcoded assembler. It is a trait the loop calls,
with interchangeable strategies for *who curates* — so they can be A/B-tested on
identical traces (the carry-over from the original RFD's three-axis design).

```rust
#[async_trait]
pub trait ContextManager: Send + Sync {
    fn strategy(&self) -> StrategyId;
    /// Assemble the prompt for the next turn within the budget the backend enforces.
    async fn assemble(&mut self, episode: &EpisodeState, budget: TokenBudget) -> Result<AssembledContext>;
    /// Persist whatever the policy considers salient after a turn (off the critical path where allowed).
    async fn commit(&mut self, outcome: &TurnOutcome) -> Result<()>;
}
```

The loop (ADR 062 `run_agentic_turn` / `AgentModelManager::chat_with_tools`)
calls `assemble`/`commit` instead of building the prompt ad hoc. Three strategies:

- **`CompactingManager` [system / deterministic]** — the default; *implements
  D2–D8* (graduated fidelity, budget allocator, delta KG, discourse selection,
  provenance retention, session seeding, VSA recall). Most of Phase 44's build is
  this manager.
- **`PassiveManager` [background]** — main loop runs a plain transcript and never
  self-edits; a background pass extracts salient facts into memory via the
  existing consolidation pipeline (`DerivationKind::AgentConsolidation`). *Rides
  Heka / Phase 42* (consolidate behaviour); cheapest in tokens, most reliable.
- **`VirtualContextManager` [model]** — the LLM curates its own window via
  self-edit tools (core/recall/archival on episodic memory) called through the
  ADR 062 tool loop; `PassiveManager` runs in parallel as the safety net so a
  missed self-edit never permanently drops a fact. Note-taking is a degenerate
  Virtual config (single append tool), not a separate manager.

Every manager is constructed inside **`SafetyEnforced<_>`** (D6) before the loop
sees it, and the three are A/B-compared behind one trait object (the experiment).

### D10 — `KvCachePolicy` (token-level KV retention): deferred, Candle-only axis

The third orthogonal axis from the original RFD, recorded so it is not lost.
Operates *below* the message layer — inside the forward pass, on the live KV
cache — so it is **not** a `ContextManager`:

- `StreamingSink { sinks, window }` — attention sinks + sliding window
  (StreamingLLM); stabilises long generation.
- `HeavyHitter { budget, recency }` — H2O/SnapKV-style attention-score eviction;
  gate carefully (can break long chains-of-thought).
- KV **quantization** (TurboQuant/FP8) and paged attention are backend *config*,
  not policy (precision/layout, not *what* is retained).

**Build deferred, gated on Candle being the chat backend (ADR 037).** Today the
loop is Ollama → llama.cpp owns the KV cache (its context-shift ≈ StreamingLLM
for free), so an akh-controlled `KvCachePolicy` is not reachable until Candle is
the inference path. Specced now; built when Candle lands in the loop.

### D11 — `RetrievalPolicy` (the fourth axis): deferred, recorded

The RFD's fourth orthogonal axis — *how context is pulled in*, independent of
curation (D9) and KV retention (D10) — so retrieval strategies can be A/B'd
without touching the managers. Carried over so it is not lost again.

```rust
/// DEFERRED. How context is *pulled in*, independent of ContextManager / KvCachePolicy.
pub trait RetrievalPolicy: Send + Sync {
    async fn fetch(&self, query: &RetrievalQuery, kg: &KnowledgeGraph) -> Result<Vec<Fact>>;
}
```

Candidates, reconciled to the develop world:

- **`JustInTime`** — on-demand `kg_query`, VSA-ranked. The current default and
  what D8/44g implements.
- **`PreloadedRag`** — embedding-based pre-inference retrieval = develop **Phase
  42**'s pgvector hybrid recall.
- **`TemporalKg`** — Graphiti-style bitemporal subgraph search with validity
  windows ("what was true at time T", auditable). This is the **future home of
  true bitemporal validity intervals** we deliberately deferred in favour of TMS
  + provenance timestamps (the supersede-don't-delete decision from the loop/
  context drafts).

**Reconciliation:** retrieval *proper* is **Phase 42 Heka's** domain (ADR 064's
seam); this assembler *consumes* whatever recall wins (42b LongMemEval/LoCoMo).
`RetrievalPolicy` is recorded as the seam/interface to that retrieval, **built
only if measurement shows retrieval — not curation — is the bottleneck** (RFD
§3.7 motivation). Not built in Phase 44.

### Explicitly out of scope
- Heka storage / pgvector / hybrid retrieval — **Phase 42** (this only consumes
  it; its sole ask of 42 is that *consolidate/decay/wakeup* are first-class, per
  ADR 064 lesson 3). The swappable `RetrievalPolicy` seam over it is recorded
  (D11), not built.
- The agentic loop itself — **ADR 062** (see ADR 066).
- Building (not specifying) `KvCachePolicy` — **deferred to the Candle-in-loop
  work** (ADR 037); recorded here as D10 so the axis survives.
- Progressive capability disclosure (P7) + akh↔akh context handoff (lesson 6) —
  the Phase 28/29–31/36–37 line.

## Alternatives considered

- **Fold into ADR 064 with no new ADR** — rejected: D6–D8 (provenance retention,
  `SafetyEnforced`, session seeding) are decisions ADR 064 does not make and that
  matter precisely because the loop can act; they deserve a record.
- **Own retrieval here too** — rejected; duplicates/collides with Phase 42 Heka.
  ADR 064's clean seam is the whole point.
- **Build before measuring** — rejected; gated on the §6 measurements below.
- **Re-introduce our removed Phase 42/43 numbering** — rejected; the loop is
  ADR 062 by content, and the assembly layer is one phase (44), not two.

## Consequences

**Positive.** Closes "forgets vs. drowns" at the assembly layer; one budget
allocator instead of three drifting caps; the loop gains session memory and a
safety-pinned window; reuses akh's existing strengths (VSA, spreading
activation, provenance, TMS) rather than adding machinery.

**Costs / risks.** A second budget consumer (chat + OODA) couples tightly with
the loop — must be co-designed (ADR 064 risk). Graduated-fidelity summarization
needs a CPU-path NL summarizer of unmeasured cost (research §6). A
`RetentionClassifier` lookup per span adds ledger reads (mitigated by the
kind-index + per-episode caching). Premature if conversation history turns out
not to improve NL quality — hence the gate.

## Sequencing — measure first (ADR 064 §6 / research §6)

Before building, answer: (1) **does conversation history improve akh's NL
quality?** (the chat path omits it today — confirm the win); (2) the **real
token window** of the deployed Gemma/candle config and what the three current
caps sum to; (3) **VSA vs pgvector for episodic recall** (LongMemEval/LoCoMo,
42b) — the allocator consumes the winner. Build D2–D5 + D7 after (1)–(3); D6
(safety pinning) can land earlier as it is correctness, not quality.

## Open questions

1. Summarizer for graduated fidelity — reuse the candle/Gemma path, a small
   distilled summarizer, or extractive (no model)? Cost/latency unmeasured.
2. Does `SafetyEnforced` at assembly subsume or duplicate any ADR-060 guarantee?
   (Believe complementary — exec vs. eviction — but verify.)
3. Procedural vs. episodic split (ADR 064 / research §6 OQ) — does the KG already
   serve procedural, or is a distinct store worth it?
4. Blend weights for VSA-similarity × recency × confidence in recall ranking.
