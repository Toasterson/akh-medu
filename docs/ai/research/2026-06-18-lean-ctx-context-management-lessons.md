# lean-ctx Context-Management Lessons — akh-medu

- **Date:** 2026-06-18
- **Status:** Research draft (no implementation)
- **Scope:** akh-medu's *own* runtime context management — the conversation, prompt-assembly, and memory paths that feed the akh's LLM (Gemma/candle) and OODA loop. **Not** about using lean-ctx as a dev tool.
- **Subject system:** [lean-ctx](https://leanctx.com/docs) — "Context Runtime for AI Agents" (MCP server: 69 tools, 10 read modes, tree-sitter AST, cross-session memory). Used here as a *design reference*, mined for transferable ideas.
- **Parallel work it must dovetail with:** Phase 41 (thinking engine / agentic chat loop — ADR 052, ADR 062), Phase 42 (Heka memory → pgvector/Postgres: 42a/42b/42d/42e), redb→Postgres migration (Anima WI `019e5982-fa80-7731-b937-9acaa5fe4dbe`), Seshat (Phase 37, already pgvector/SeaORM).
- **Companion decision:** [ADR 064 — Context Runtime Lessons](../decisions/064-context-runtime-lessons.md)
- **Author:** Claude (research agent)

---

## 1. Goal & scope

akh-medu is a cognitive agent that assembles context for two consumers: (a) its NL model (Gemma via candle/ELIZA) on the chat path, and (b) its OODA loop on the autonomy path. The recurring operational pain — visible in the Anima backlog ("Zero episodic memory: `agent_recall` returns nothing after 2+ weeks of cycles", WI `019e587a-afb9`; repeated goal/episode explosions) — is that the agent either **forgets** (memory never consolidated or recalled) or **drowns** (raw replay, hairball injection, no budget). lean-ctx is a tool built *specifically* to manage an LLM agent's context window, so it is a useful mirror: which of its design choices does akh already have, which is it missing, and where do the gaps line up with Phase 41/42 work already in flight?

This document is the research backing for a future phase ticket. It does **not** propose implementation now; it records the analysis so the eventual phase plan can be grounded once Phase 41/42 and the Postgres migration land. The decision posture lives in ADR 064.

## 2. What lean-ctx actually is (distilled)

Strip away the 69 tools and lean-ctx rests on one idea: **context is a managed resource rendered at graduated fidelity, not a binary include/exclude decision.** Everything else follows.

| Principle | lean-ctx mechanism |
|---|---|
| **P1 — Graduated fidelity** | One artifact, 10 render modes: `full \| map \| signatures \| diff \| task \| reference \| aggressive \| entropy \| lines:N-M \| auto`. Editing → `full`; orienting → `signatures`/`map`; re-checking → `diff`; large/noisy → `aggressive`/`entropy` (Shannon filtering). |
| **P2 — Reference + dedup** | Re-reads cost ~13 tokens: content already in context is referenced by handle (`F1`, `F2…`), not re-emitted. Pay for context once. |
| **P3 — Delta-only updates** | `diff` mode + `ctx_delta`: after a change, emit only what changed (`+ / - / ~`), not the whole artifact. The CEP output discipline ("DELTA ONLY", "ACT FIRST", "ONE LINE", never echo/narrate) treats the agent's *own* output as context it must not pollute. |
| **P4 — Task-relative assembly** | `ctx_overview(task)`, `task` read mode, relevance scoring decide *what is in* against a stated goal — not "everything recent". |
| **P5 — Managed budget** | `ctx_session budget` (limits) + `slo` (observability) + `verify` (output-verification stats): the window is a budget allocated across competing demands and measured. |
| **P6 — Layered persistent memory** | `ctx_knowledge`: `remember / recall / consolidate / relate / feedback(up,down) / decay / wakeup / embeddings`, with confidence scores and typed relations. `ctx_session`: distinct **episodic** and **procedural** stores, `wakeup` restores ~400 tokens of prior findings at session start. |
| **P7 — Progressive disclosure** | 10 tools always visible, 59 lazy-loaded on demand (the same deferred-tool mechanism this very harness uses). Don't pay context for capability you aren't using yet. |
| **P8 — Graph-driven relevance** | `ctx_impact` / `ctx_callgraph`: walk the code graph to bound what is relevant ("blast radius"), rather than dumping neighbours. |

## 3. akh-medu's current context handling

A read-only sweep of the runtime context path. File:line references are repo-root-relative.

### 3.1 Conversation / chat loop
- `src/agent/conversation.rs:86-101` — `ConversationState`: a `VecDeque` FIFO ring buffer (line 90) bounded at `max_turns = 10` (line 92), tracking active referents and active topic. Turns stored **raw**; eviction is pure age-FIFO.
- `src/chat.rs:62-80,127,172-178` — unified `ChatProcessor`: `process_input()` → NLU parse → `record_turn(Speaker::Operator, …)` → dispatch. No turn replay or summarization in the loop.
- `src/agent/chat.rs:113-138` — REPL `Conversation`: bincode-serialized turns persisted to the durable store; no compression.
- **Critical finding:** prior conversation turns are **not injected into the LLM prompt at all** (only psyche + KG context are). The chat path has no discourse continuity on open-ended turns.

### 3.2 Prompt assembly & token budget
- `src/compartment/prompt.rs:31-114` — `PsychePromptSerializer`: psyche → system prompt, hard-capped at `DEFAULT_MAX_TOKENS = 1_800` (line 13); shadow patterns sorted by severity and truncated to fit (lines 158-192) — a *local* graduated-fidelity behaviour (good).
- `src/compartment/prompt.rs:313-316` — `estimate_tokens(text) = text.len().div_ceil(4)`: a fixed 4-chars/token heuristic.
- `src/nlu/kg_context.rs:19,28` — KG context block: `MAX_CONTEXT_CHARS = 8_000` (~2k tokens), `TOP_K_SIMILAR = 5`, `TRIPLES_PER_SYMBOL = 3`, `MAX_TRIPLES = 15`.
- `src/chat.rs:1231-1265` — `collect_kg_context_for_input()` for the creative LLM fallback: caps at `facts.truncate(10)` (line 1263).
- **Critical finding:** budgets are independent hardcoded constants in three modules (1 800 tok psyche + 8 000 char KG + 10 facts) with no shared allocator. They can collectively exceed the real window; nothing arbitrates when they compete.

### 3.3 KG / entity-aware context selection (the strong part)
- `src/nlu/kg_context.rs:91-155,176-272` — `build_kg_context()`: keyword extraction (≤8 content words) → symbol resolution → VSA similarity expansion (`TOP_K_SIMILAR`, 0.05 noise floor) → bounded triple collection with dedup → char-capped block. This is genuine **task-relative, relevance-scored assembly** (lean-ctx P4/P8).
- `src/nlu/kg_context.rs:40-77` — `KgContextCache`: session-scoped, FIFO eviction at `MAX_CACHE_ENTRIES = 64`, keyed by sorted lowercased keywords — a real caching layer (lean-ctx P2-adjacent).
- **Finding:** built **once at parse time** and never refreshed across OODA cycles — the cache exists but there is no *delta* refresh (missing lean-ctx P3).

### 3.4 Memory
- `crates/heka/src/drawer.rs:30-146` — Heka episodic store: `DrawerRecord` (subject/claim/source/confidence/timestamp/topic_ids) in a redb K/V table; insert + retrieve only. **No consolidation, no summarization, no decay, no wakeup, no embedding recall** (missing most of lean-ctx P6).
- `src/agent/ooda.rs:25-97` — OODA cycle: `Observation.recalled_episodes` (line 53) + `Orientation.relevant_knowledge` (line 60) assembled per cycle via spreading activation — akh *does* do graph-driven relevance here (lean-ctx P8), but only for the autonomy path, and the episodic side is starved by §3.4's drawer being write-mostly.

### 3.5 Scorecard

| Area | lean-ctx principle | akh state |
|---|---|---|
| Conversation turns | P1 graduated fidelity | **Naive** — raw FIFO-10, not even injected into prompt |
| KG context selection | P4/P8 relevance | **Strong** — keyword + VSA + bounded triples |
| KG context refresh | P3 delta | **Missing** — built once, goes stale across cycles |
| Prompt token budget | P5 managed budget | **Fragmented** — 3 isolated hardcoded caps; `len/4` estimator |
| Episodic memory | P6 consolidate/decay/wakeup/recall | **Basic** — write-mostly drawer; recall starved (see WI `019e587a-afb9`) |
| Episodic + procedural split | P6 distinct stores | **Partial** — episodic exists; procedural implicit |
| OODA relevance | P8 graph-driven | **Strong** — spreading activation in Orient |
| Capability/tool exposure | P7 progressive disclosure | **Not yet** — relevant as Phases 29–31 land |
| Context effectiveness | P5 verify | **Missing** — no measurement |

The pattern is clear: **akh already owns lean-ctx's hardest primitives (VSA similarity, spreading activation, KG-backed dialogue state, confidence) but applies them only to *knowledge*. The conversation and memory paths get none of that discipline.**

## 4. Lessons (prioritized)

Highest leverage first. Each is grounded to the code above and to the parallel work it should ride on.

1. **Replace binary truncation with graduated fidelity (P1).** The single biggest lesson. A turn / fact / episode should be renderable at full → one-line summary → reference-handle → dropped, chosen by relevance × recency × information — not `truncate(10)` / FIFO-10. Target: `ConversationState` (`src/agent/conversation.rs:86`) eviction, and the creative-fallback cap (`src/chat.rs:1263`).

2. **One managed context budget, not three constants (P5).** Introduce a single allocator that divides the window across {psyche prompt, KG block, conversation history, recalled episodes} by priority, replacing the isolated caps in `src/compartment/prompt.rs:13` and `src/nlu/kg_context.rs:19`. Upgrade the `len/4` estimator (`src/compartment/prompt.rs:314`) to the actual tokenizer once the candle/Gemma path is settled. **This must be co-designed with Phase 41's agentic chat loop (ADR 062)** — that is where the prompt is assembled.

3. **Upgrade Heka from a drawer to a knowledge layer (P6).** `consolidate` (merge/dedupe — akh already has the `DerivationKind::AgentConsolidation` hook), `wakeup` (surface relevant episodes in ~400 tokens at session start), embedding-based `recall` (reuse VSA, or pgvector once 42a lands), and decay. **This is the same surface Phase 42 is already building** (42a Heka+pgvector, 42d hybrid pgvec+VSA+KG retrieval, 42e Heka MCP `remember`/`recall` tools) — the lesson is to make sure 42 ships *consolidation + wakeup*, not just storage, so the §3.4 recall starvation (WI `019e587a-afb9`) is actually closed and stays closed (LongMemEval/LoCoMo in 42b is the regression guard).

4. **Make KG context delta-refreshed, not static (P3).** Extend `KgContextCache` (`src/nlu/kg_context.rs:40`) to emit per-cycle *deltas* (newly spreading-activation-surfaced triples) instead of a once-built block that goes stale across OODA cycles.

5. **Reuse spreading activation for conversation selection (P8).** akh already walks the KG by relevance in OODA Orient (`src/agent/ooda.rs:60`). Point the same mechanism at discourse: select which prior turns/entities to inject by graph proximity to the current input's entities, instead of replaying the last N.

6. **(Forward-looking) Progressive capability disclosure + context handoff (P7).** As Phases 29–31 (capability/skill exposure) and 36–37 (delegation, Seshat sharing) grow the tool/skill surface, surface a relevance-ranked *core* to the akh's LLM and lazy-load the rest (ties into the Phase 28 semantic router); package task context for akh↔akh handoff rather than re-deriving it (lean-ctx `ctx_pack`/`ctx_handoff` analog).

## 5. Relationship to parallel work — what this does and does NOT duplicate

This is the crux for sequencing. The lessons split cleanly into two layers:

- **The long-term retrieval layer (storage + embeddings + hybrid recall)** is **Phase 42's job and is already being built** — pgvector-backed Heka (42a), hybrid pgvec+VSA+KG retrieval (42d), the Heka MCP `remember`/`recall` surface (42e), benchmarked by LongMemEval/LoCoMo (42b), all sitting on the redb→Postgres migration (WI `019e5982`) and reusing the Seshat (Phase 37) pgvector/SeaORM substrate. **This research does not propose re-doing any of that.** Its only ask of Phase 42 is that the *behaviours* in lesson 3 (consolidate, decay, wakeup) are first-class, not just CRUD.

- **The working-context assembly layer (fidelity, budget, delta, discourse selection)** is **not owned by any current phase.** Phase 41 builds the agentic chat *loop* (ADR 062) but not a managed context budget or graduated-fidelity transcript. That is the genuinely new surface this would become — and it sits *above* Phase 42's retrieval (it decides what, of everything retrievable, actually enters the window, at what fidelity, within budget).

**Sequencing:** the eventual phase plan should land **after** Phase 41 (so it has a chat loop to assemble into) and **after** Phase 42a/42d (so the budget allocator can pull from real pgvector hybrid recall rather than the write-mostly drawer). Until then this stays research + ADR.

## 6. Open questions / what to measure before committing

- **Does conversation history even help akh's NL quality?** Measure before building lesson 1/2 — the chat path currently omits it entirely; confirm the win.
- **What is the real token window** of the deployed Gemma/candle config, and how much do the three current caps actually sum to in practice? (Needed to size the allocator.)
- **Is VSA similarity good enough for episodic recall**, or is pgvector (42a) strictly better? 42b's LongMemEval/LoCoMo harness is the place to answer this — the budget allocator should consume whichever wins.
- **Procedural memory:** lean-ctx separates episodic from procedural. akh's procedural learning (chunking, `src/agent/`) is implicit — is a distinct store worth it, or does the KG already serve?

## 7. Pointer

Decision posture and the deferral rationale: [ADR 064 — Context Runtime Lessons](../decisions/064-context-runtime-lessons.md). When Phase 41/42 land, ADR 064 becomes the seed for a `docs/ai/plans/` phase plan covering §4 lessons 1, 2, 4, 5 (the working-context layer).
