# Research / Pondering: Mapping Claude Code's prompt→plan→tool workflow onto akh-medu

> Date: 2026-04-28
> Status: Research synthesis (the user said "just pondering"). Treat this as a design exploration, not a commitment. The output is a recommendation + concrete sub-phase additions to existing Phase 26/28/33/36/41 plans, not a new top-level phase.
> Anima ticket: `019dd0c5-9f4b-74e3-ad5c-3bc0a22e0ef9` (Phase 41 — The Thinking Engine)
> Related ADRs (proposed by this plan): 054 (hidden-state fidelity), 055 (API trace re-encoding), 056 (unified trace envelope)
> Related plans: [Phase 41 plan](2026-04-27-phase41-thinking-engine.md), [Phase 26 plan](2026-03-26-phase26-llm-vsa-deep-integration.md), [Phase 33 plan](2026-03-26-phase33-grounded-verbalization.md), [Phase 36 plan](2026-03-28-phase36-delegation-external-scheduling.md)

## Context

The user just published, on 2026-04-27:
- [Phase 41 plan](2026-04-27-phase41-thinking-engine.md) — the unified `ThinkingLoop` over their existing reasoning primitives.
- [ADR 052](../decisions/052-unified-thinking-loop.md) — locks in the architecture, frames the LLM as *peripheral, not core*.
- [Research synthesis](../research/2026-04-27-thinking-in-llms-vs-vsa.md) — the "frontier LLMs implement an approximate VSA; akh-medu has the exact one" thesis.

The pondering question: how does the bridge from an LLM into the Hypervector VSA actually work, and how could a Claude-Code-style workflow (a simple prompt → a plan → tool calls → a result) be made to work on this substrate? This is asked at the moment Phase 41 is freshly designed but unbuilt, while Phase 26d (the bridge) is **~80 % implemented but un-wired**, and Phase 36 (delegation to Claude Code) is designed but not coded.

The goal of this document: produce a deep, concrete answer that surfaces the integration gap, names the existing pieces that close it, and proposes a minimal sub-phase sequence that would carry the architecture from "research artefact" to "working demo".

## 1. The bridge as it exists today

The bridge is *physically present* in tree, just not yet wired into the runtime path that NLU output flows through.

**Forward pass (text → HyperVec → KG concept).**
1. Tier-3 NLU calls [`CandleBackend::extract_hidden_state`](../../src/nlu/candle_backend.rs:275) on a Qwen2.5-1.5B forward pass. (Note: the current implementation returns the last-token *logits* as a stand-in. This is a known infidelity that gets a dedicated phase before Pattern A ships at production quality — see §7 commitment 1: candle-transformers hook patch vs. switching to an LLM with native hidden-state extraction.)
2. The f32 vector enters [`NeuralVsaBridge::encode`](../../src/vsa/neural_bridge.rs:191). A learned `[10000 × 1536]` linear projection plus sign-binarization at threshold 0 produces a 10K-bit bipolar `HyperVec`. The encoder is trainable via `BridgeTrainer` (line 253) using sigmoid-approximated Hamming loss against grounded vectors from `ItemMemory`.
3. The HyperVec is then fed to [`ItemMemory::search`](../../src/vsa/item_memory.rs:138), an HNSW ANN search that returns `Vec<(SymbolId, similarity)>`. This is the moment the noisy continuous representation lands back in the symbolic graph.
4. `encoding_confidence` (the average margin from the binarization threshold, line 211) acts as a quality gate: if the encoded vector lands far from any known concept, the call is treated as a potential hallucination signal rather than a fact.

**Reverse pass (answer HyperVec → text).**
1. The conclusion HyperVec hits [`NeuralVsaBridge::decode`](../../src/vsa/neural_bridge.rs:177), which uses an optional decoder linear projection (initialised via `init_decoder` at line 162) to project 10K bits back to a 1536-D approximate hidden state.
2. That hidden state is the *steering signal* for a Phase 33 verbalizer (T5/mT5 + NeuroLogic constrained decoding) — not yet built; design lives in [Phase 33 plan](2026-03-26-phase33-grounded-verbalization.md) and [ADR 043](../decisions/043-grounded-verbalization.md).

**Compression for transport.** [`PolarQuant`](../../src/vsa/polar_quant.rs) provides FWHT + polar transform + 3-bit Lloyd-Max compression — 10000 bits down to ~3750 bits with controlled distortion. Relevant to multi-Akh sharing and WAL transport, not to the inner loop.

**The wiring gap (Phase 26e).** `NluParseResult.neural_concepts` is a declared field that no current code path populates. Wiring it requires roughly 150 LOC of plumbing: a single `ground_phrase(s: &str) -> Vec<(SymbolId, f32)>` that chains `CandleBackend::extract_hidden_state → NeuralVsaBridge::encode → ItemMemory::search`, exposed both to NLU and to the Phase 41 ThinkingLoop. This is the single highest-leverage piece of remaining bridge work.

## 2. Claude Code's workflow, mechanistically

Strip the marketing: Claude Code's prompt → plan → tool-calls → result loop is one autoregressive token stream that emits, alternately, *thinking text* and *tool-use blocks*. The model's reasoning is RL-trained on outcome-verified rollouts; its "plan" is whatever fragment of the trace the harness chooses to surface to the user (Plan mode is a UI affordance, not a separate model). The tool-call mechanism is the harness pausing the stream, executing the named tool, and feeding the result back as a tool_result block.

The mechanistic primitives, mapped:

| Claude Code | Akh-medu equivalent | State |
|---|---|---|
| Autoregressive token sampling | Phase 41 `ThinkingLoop` over symbolic primitives | Designed |
| Hidden state as working memory | `ThoughtState.focus_vec: HyperVec` | Designed |
| In-trace verification (PRM-style) | Phase 17 Dempster-Shafer + Phase 9l contradiction + Phase 18 source reliability | Implemented; wiring designed in 41c |
| RL on outcomes | Phase 11g VOC + outcome-driven goal scoring | Implemented for goals; Phase 41e extends to per-step |
| Tool-use blocks | Existing `ToolRegistry` + 26 implemented tools, plus Phase 36 delegation | Implemented; Phase 36 designed |
| Plan mode (visible plan) | Phase 29 capability exposure (`reasoning_trace`, `explain_decision`) | Designed |
| Surface response generation | Phase 33b T5 + NeuroLogic NLG | Designed |

The key structural difference is that Claude Code keeps everything in one stream — thinking, planning, tool-calling, and replying are sampled tokens. Akh-medu separates them: thinking is symbolic primitives operating on a HyperVec; planning is HTN + MCTS over an action graph; tool-calling is the OODA Act phase; replying is downstream NLG. The integration work is making those four surfaces feel like one stream from outside, while keeping their distinct, faithful, auditable nature inside.

## 3. Three architectures for the unified surface

### Option A — `ThinkingLoop` subsumes OODA

Make `ThoughtStep` (per [Phase 41 plan, line 91](2026-04-27-phase41-thinking-engine.md)) include `ToolCall { tool, args, result, observation_vec }` as a primitive variant. The Loop deliberates and one of its primitive moves is executing a tool via the existing `ToolRegistry`. OODA collapses into a degenerate single-primitive Loop config.

**Pros:** Closest to the Claude Code analogy. Single trace, single budget, single surface. Phase 41g's benchmark harness automatically covers agency.
**Cons:** Discards 2,035 lines of working OODA code ([`src/agent/ooda.rs`](../../src/agent/ooda.rs)) plus the WM/episodic/JITIR observation logic and the multi-feature tool-scoring (base + recency + novelty + episodic + pressure + archetype + goal-value-factor). High regression risk.
**Effort:** Heavy. ~2–3 sub-phases beyond Phase 41 (re-implement Observe/Orient/Decide as primitives, port HTN, retire `run_ooda_cycle`).

### Option B — OODA wraps `ThinkingLoop` *(recommended)*

Keep `run_ooda_cycle` as the outer agent loop. In the Decide phase, when the existing tool scorer is uncertain (top-k tools tied within ε, or `DecisionImpasse` triggered), call `ThinkingLoop::pick_tool(goal, candidates, budget)` to pick the next tool. The Loop's trace is the agent's "thinking transcript" surfaced via Phase 29 `reasoning_trace`. The Loop's `Conclusion::answer_symbol` resolves to one of the candidate `ToolId`s.

**Pros:** Zero regression on existing OODA. Phase 41 ships standalone for inference *and* is available as a deliberation oracle. Plan-mode UX falls out naturally — when the Loop runs, surface its trace; when fast-path Decide is enough, no overhead. The "Pattern" infrastructure (next section) plugs in cleanly because the ThinkingLoop is already the natural call site.
**Cons:** Two reasoning surfaces, two trace formats. Risk that the agent rarely actually invokes the Loop because the heuristic scorer is good enough and the Loop's value-add stays invisible. Doesn't fully match Claude's architecture — Claude has no fast-path decision.
**Effort:** Light. ~150–300 LOC adapter plus a hook in [`decide()`](../../src/agent/ooda.rs:300).

### Option C — Coexist with router handoff

A goal classifier routes: knowledge questions → `ThinkingLoop`; agency tasks → OODA. Both write into the same `thinking_traces` table (Phase 41d schema) under a shared `trace_id`. The router is a Phase 28 LinUCB extension.

**Pros:** Preserves both implementations; lets each surface optimise for its native task shape; bandit learns when each wins.
**Cons:** Two architectures the user must understand. Trace unification is non-trivial. Bad classification produces bad UX. Worst match for the Claude analogy.
**Effort:** Medium. Phase 28 LinUCB extension (~200 LOC), shared trace envelope schema, goal-shape classifier (~300 LOC).

### Recommendation

Pick **Option B for the next 6–8 weeks**, keep Option A as a planned end-state. OODA works (2,035 LOC, 26 tools, HTN, MCTS, provenance). Phase 41 is already designed as a standalone. The wrap is a 200–300 LOC adapter, not a redesign. After a release cycle of empirical data on (a) when Loop tool selection beats heuristic, (b) what fraction of OODA cycles benefit from deliberation, (c) where the trace formats actually need to converge — *then* a Phase 42 ADR can collapse OODA into a degenerate Loop config with confidence.

## 4. Four LLM-call patterns inside the Loop

ADR 052 commits to "LLM as peripheral". Concretely, the Loop never calls an LLM by default; it emits LLM-call primitives only when the symbolic core hits a wall an LLM is uniquely suited to widen. Every pattern returns a HyperVec into `ThoughtState`; the loop never reads tokens.

### Pattern A — Concept grounding (encode-only, ~800 ms, $0)
**Trigger.** A goal symbol unbound in `ItemMemory`, or a `spreading_activation_step` whose frontier shows zero novelty for K rings.
**Action.** `ground_phrase(s) → Vec<(SymbolId, f32)>` — the Phase 26e plumbing. Top-k grounded SymbolIds enter `state.focus_vec`.
**Pattern role.** Enrichment, not generation. Reads where Qwen *points* in the residual stream, not what Qwen *would have said*.

### Pattern B — Hypothesis generator via API-trace re-encoding (~5–30 s, cents)
**Trigger.** Argumentation produced no clear winner (verifier `score < 0.5`, `BeliefIntervalNarrowed` flat); MCTS oscillating; degenerate hypothesis set; or an explicit "borrow a frontier model's thinking" step.
**Action.** Hit a thinking-trace-emitting API — primarily **OpenRouter** as the multiplexer to Anthropic / OpenAI / DeepSeek / Gemini extended-thinking endpoints. The response payload includes the model's *thinking blocks* (extended thinking text, reasoning summaries, chain-of-thought spans) as well as the final answer. Each thinking-block chunk is re-encoded through the bridge into a `HyperVec` and lands in the trace as a synthesised `ThoughtStep` whose kind reflects the chunk's structure (tentative, contradiction-noticed, hypothesis-emitted, conclusion). The final answer also encodes; both the per-chunk vectors and the answer vector enter `state.hypotheses` in superposition. `superposition_merge` bundles them; `verifier_score` (Dempster combination) prunes weak ones.
**Why API and not local Claude subprocess.** Local Claude doesn't expose intermediate thinking blocks the way the OpenRouter pass-through does. The trace-emitting API is the principled hidden-state stand-in for Pattern B; the local subprocess remains the right backend for Pattern C (concrete side effects), where the trace isn't the point.
**Pattern role.** Two-fold:
  (a) **Information import** — pull frontier-model reasoning into akh-medu's symbolic state for downstream use.
  (b) **Benchmark target** — Phase 41g now includes a comparator: re-encoded frontier-model thinking traces vs. akh-medu symbolic Loop traces on the same prompts. Either the bridge faithfully captures their reasoning (validating the bridge), or it doesn't (validating that symbolic reasoning is doing something LLM traces can't), and either outcome is informative.
**Pattern role (continued).** The COCONUT analogue, in spirit: Tree-of-Thoughts pays O(N) downstream because each branch persists as text; akh-medu pays O(N) once for the API call and the trace re-encoding, then bundles into superposition. The difference is we're not generating N parallel rollouts — we're consuming someone else's already-traced reasoning and folding it into our state.

### Pattern C — Tool executor (delegate to Claude Code, 10–30 s + maybe API $)
**Trigger.** Goal with concrete external side effect: `intent ∈ {AddCode, EditCode, RunCommand, ProduceArtifact}`, or step where the loop has converged on *what to do* but the *doing* is outside the symbolic substrate.
**Action.** Emit `ThoughtStep::Delegate { task, target }` dispatched via `DelegationManager::dispatch` (Phase 36). Subprocess receives `.mcp.json` pointing back to akhomed; can call `triples_of`, `explain_decision`, `query` over MCP and ground its decisions in the KG. The `akh-result { triples, summary, files_modified }` block parses into `DelegationOutput`.
**Return path.** Each output triple encodes via the bridge into `state.focus_vec`; `files_modified` and the summary become side-effect provenance. `verifier_score` checks for contradictions with prior beliefs.
**Pattern role.** This is where Claude Code's planning/coding capability flows in — but as a controlled side-effect step, not a reasoner. `CostGuard` enforces `per_task_max_cents`.

### Pattern D — Verbalizer (surface generation, ~50–200 ms)
**Trigger.** Loop terminates with `Converged` or `VerifierSufficient`; user asked a natural-language question.
**Action.** `Conclusion.answer` HyperVec → Phase 33 T5 NLG with NeuroLogic constraints derived from `Conclusion.trace` (verbalizer is *forbidden* to assert facts not in the trace). Fidelity check round-trips the verbalization through the bridge and measures cosine to `Conclusion.answer`; regenerate if it diverges past τ.
**Surface format.** Two outputs in parallel:
  (a) A short prose response for the chat surface ("Added struct Foo at src/foo.rs…").
  (b) A **markdown plan document** rendered from the trace: ordered ThoughtStep summaries, the chosen tools and their results, the verifier scores, the final confidence interval. This is the Plan-mode equivalent surface — a human-readable rendering of the auditable trace, not a separate planning artefact. Generated by a deterministic template-and-T5 hybrid; the template ensures structure, T5 fills prose where free-form is acceptable.
**Pattern role.** Verbalization is downstream of reasoning, never inside the loop. The trace is the truth; the words are decoration. The markdown surface is the user-facing equivalent of `claude --print --plan`.

### Routing across patterns (Phase 28)

For each LLM-call primitive (A/B/C), a Phase 28 LinUCB bandit chooses the backend:
- No-LLM (cheapest; symbolic widen — extra spreading rings, e-graph rewrites)
- Local Candle / Qwen2.5-1.5B (Pattern A natively; Pattern B fallback when a hidden-state-yielding backend is needed cheaply)
- Local Claude subprocess (B/C; free, 10–30 s)
- Peer akh-medu (B/C; free, 1–5 s, may have orthogonal bias)
- Remote Claude API (B/C; cents, fastest for hard problems)

**Bandit context vector:** `(goal_kind, focus_vec_entropy, hypothesis_count, current_belief_width, voc_remaining, wall_remaining, recent_failure_rate_per_target)`.
**Reward:** `Δ Bel(top hypothesis) − cost_in_voc_units`.

### Cost-aware halting (Phase 41e VOC)

Price every backend in the same VOC currency as symbolic primitives. Halt rule:
1. If cheapest *symbolic* next step has VOC > 0 → take it.
2. Else if any LLM backend has `expected_progress − cost > 0` and total spend < `ThinkingBudget.voc_credits` → route to bandit.
3. Else → halt with `DiminishingReturns`.

The Apple-illusion-of-thinking fix made economic: doesn't give up early, doesn't burn cents on hopeless polishing. The user's "depth dial" becomes a literal `voc_credits` slider.

## 5. Worked example — the demo

User prompts: **"add a Rust struct Foo with proper error handling and tests"**

```
NLU Tier 1–3 → AbsTree { intent: AddCode, language: Rust, subject: "Foo",
                          constraints: [error_handling, tests] }
goal_add_struct_foo := mint_goal(...)
budget := { steps: 200, wall_ms: 120_000, voc_credits: $0.50 }

ThinkingLoop::deliberate(goal_add_struct_foo, budget):
  step 1  spreading_activation       → 31 frontier concepts: rust_struct_pattern,
                                       miette_idiom, project_uses_thiserror, ...
  step 2  argumentation              → topic="thiserror vs miette",
                                       verdict: project_uses_thiserror, Bel=0.78
  step 3  Pattern A (Qwen)           → ground_phrase("Rust struct Foo with error
                                       handling and tests") → 4 SymbolIds added
                                       to focus_vec (rstest_pattern, etc.)
  step 4  Pattern B (local Claude,   → 3 plan outlines (file path, derive macros,
            N=3, "draft 3 outlines")    test framework). bridge.encode each →
                                       state.hypotheses with Bel intervals.
  step 5  superposition_merge        → mean_similarity = 0.81
  step 6  verifier_score             → 3/3 agree on src/foo.rs, 2/3 on
                                       derive(Debug,Clone), 1 outlier on proptest
  step 7  argumentation              → topic="proptest vs rstest", verdict by
                                       Phase 18 source reliability: project
                                       conventions dominate → rstest
  step 8  router → Pattern C, Sonnet  → Delegate to local Claude subprocess.
                                       Subprocess MCP-calls back: triples_of(Foo),
                                       reads project conventions, writes file,
                                       runs cargo check. Returns akh-result
                                       { triples: [...], files_modified: [...],
                                         summary: "Added Foo with thiserror..." }
  step 9  superposition_merge        → fold subprocess triples into focus_vec
  step 10 verifier_score             → 0.94, sufficient=true
  step 11 HALT                       → VerifierSufficient
  step 12 Pattern D (T5 verbalize)   → "Added struct Foo at src/foo.rs with
                                       thiserror diagnostics, two rstest cases,
                                       derive(Debug, Clone). Cargo check passes."
Conclusion { answer: <hypervec>, trace: [12 ThoughtSteps],
             confidence: [0.91, 0.97], steps_used: 11 }
```

LLM was called four times (A, B, C, D). It never *drove* the loop. The argumentation steps and the verifier remained symbolic; the LLM was a peripheral the budget controller priced into existence at four specific moments.

## 6. The minimal demo plumbing — what closes the gap

Add three new commitments to the existing roadmap (each gets its own ADR — see §7):

- **Phase 26f — Hidden-state fidelity** (new sub-phase): patch candle-transformers vs. switch to a hidden-state-native LLM. Prerequisite for production-quality Pattern A; demo can ship on logits-as-stand-in.
- **Phase 41h — Agent integration adapter** (new sub-phase under Phase 41, between 41f and 41g).
- **Phase 41i — Trace envelope** (new sub-phase): unifying schema for `AgentDecision` + `ThoughtStep` under a shared `trace_id`. Single ADR.

| Piece | Status | Effort | File path |
|---|---|---|---|
| **Phase 26f**: candle-transformers hook patch OR migration to a model with native hidden-state extraction (Mistral, Pythia, GPT-NeoX as candidates) — prerequisite for production Pattern A | not started | ~600–1,500 LOC depending on path | upstream PR to candle-transformers, OR new `src/llm/<model>_backend.rs` |
| Phase 26e: `ground_phrase()` plumbing for Pattern A | bridge exists, plumbing missing | ~150 LOC | new `src/vsa/grounding.rs`, extend `src/nlu/candle_backend.rs` |
| Phase 41a–c: Loop skeleton + scheduler + verifier | does not exist | ~2,100 LOC (already in plan) | new `src/thinking/{mod,state,loop_runner,scheduler,verifier,budget}.rs` |
| Phase 41h.1: `ThinkingLoop::pick_tool(goal, candidates, budget)` adapter | does not exist | ~200 LOC | new `src/thinking/agent_adapter.rs` |
| Phase 41h.2: `decide()` impasse hook in [`src/agent/ooda.rs:300`](../../src/agent/ooda.rs:300) calling `pick_tool` | does not exist | ~50 LOC | modify `src/agent/ooda.rs`; add `Decision::Deliberated { trace_id, tool, confidence }` |
| Phase 41h.3: MCP tool `get_deliberation_trace(trace_id) → ThoughtStep[]` + markdown plan rendering | does not exist | ~150 LOC | extend Phase 29 capability exposure; new `src/nlg/plan_markdown.rs` |
| **Phase 41i**: `trace_id` envelope joining `AgentDecision` + `ThoughtStep` (own ADR) | does not exist | ~150 LOC + schema migration | extend `src/provenance.rs`; new migration |
| **Pattern B**: OpenRouter API client + thinking-trace re-encoder | bridge exists, client missing | ~500 LOC | new `src/thinking/openrouter.rs`, `src/thinking/trace_reencode.rs` |
| Phase 41e: VOC budget controller pricing LLM calls | partially exists for goals | ~400 LOC (already in plan) | new `src/thinking/voc_budget.rs` |
| Phase 28: LinUCB router | does not exist | ~600 LOC | new `src/thinking/router.rs` + sub-modules |
| Pattern C: Phase 36a LocalClaude delegation | designed, not coded | ~1,500 LOC | new `src/agent/delegation/local_claude.rs` + `mod.rs` + `types.rs` |
| Pattern D: Phase 33b stub T5 verbalizer + markdown plan template | does not exist | ~500 LOC for stub, ~800 LOC full | new `src/nlg/t5_constrained.rs`, `src/thinking/verbalize.rs`, `src/nlg/plan_markdown.rs` |
| Phase 41d: persistence to `akhomed.thinking_traces` | does not exist; depends on Phase 38 | ~600 LOC (already in plan) | new `src/thinking/persistence.rs` + migration |
| **Phase 41g (extended)**: benchmark including Pattern B comparator (frontier-model trace re-encoding vs. akh-medu symbolic Loop on same prompts) | does not exist | ~400 LOC on top of existing 41f harness | extend `crates/akh-bench-thinking/` |

**The minimum viable demo** (logit-stand-in Pattern A acceptable, full hidden-state fidelity deferred to a follow-up release): Phase 26e (~150 LOC), Phase 41a–c (~2,100 LOC), Phase 41h (~400 LOC), Phase 41i envelope (~150 LOC), Phase 36a LocalClaude (~1,500 LOC), Phase 33b stub + markdown template (~500 LOC). Roughly **4,800 net new LOC**.

**Production-quality follow-up** adds Phase 26f (hidden-state fidelity, ~600–1,500 LOC depending on path), Pattern B (~500 LOC), Phase 41e VOC budget (~400 LOC), Phase 28 router (~600 LOC), Phase 41g benchmark extension (~400 LOC), Phase 41d persistence (~600 LOC). Another **~3,500 LOC**.

Suggested ordering:
1. Phase 41a–c (loop skeleton — establishes the surface)
2. Phase 41i trace envelope ADR + implementation (~150 LOC — needed before any cross-surface stitching)
3. Pattern A logit-proxy glue (Phase 26e, ~150 LOC — cheapest, highest leverage; production-quality A waits on Phase 26f)
4. Phase 41h adapter (~400 LOC — gets the Loop wired into OODA)
5. Phase 36a LocalClaude (~1,500 LOC — Pattern C's tool executor)
6. Phase 33b stub verbalizer + markdown plan template (~500 LOC — closes the visible loop)
7. **Demo lands here. ~4,800 LOC, all four patterns reach end-to-end at acceptable quality.**
8. Phase 26f hidden-state fidelity (own ADR; ~600–1,500 LOC) — investigate candle-transformers patch vs. model migration; the right answer drives the LOC count.
9. Pattern B OpenRouter trace re-encoder (~500 LOC) — once 26f gives us a faithful encode/decode for hypothesis vectors.
10. Phase 41e VOC budget (~400 LOC — economic halting)
11. Phase 28 router (~600 LOC — when N backends exist)
12. Phase 41g benchmark extension with Pattern B comparator (~400 LOC — empirical bridge-fidelity test)
13. Phase 41d persistence (~600 LOC — when Phase 38 ships)

## 7. Resolved decisions (commitments)

These were open questions in an earlier draft; the user resolved each. They become commitments and drive the §6 implementation table.

1. **Hidden-state fidelity gets a dedicated phase (Phase 26f) before production Pattern A.** The current logits-as-stand-in is acceptable for the §5 demo but not for the long term. The phase evaluates two paths:
   - (a) Patch candle-transformers to expose hidden-state hooks for Qwen2 (and submit upstream).
   - (b) Migrate the bridge's source LLM to a model with native, well-documented hidden-state extraction (candidates: Mistral via candle, Pythia, GPT-NeoX, or another candle-supported architecture with a clean inner-state API).
   Decision criterion: whichever path delivers a faithful 1536-D (or comparable) hidden-state stream with the lowest maintenance burden and the cleanest licence story. Capture in **[ADR 054 — Hidden-state fidelity for the LLM↔VSA bridge](../decisions/054-hidden-state-fidelity.md)**.

2. **Pattern B targets thinking-trace-emitting APIs (primarily OpenRouter), not local subprocesses.** Local Claude doesn't expose intermediate thinking blocks; OpenRouter's pass-through to Anthropic / OpenAI / DeepSeek / Gemini extended-thinking endpoints does. Pattern B's job is to consume those traces, re-encode each thinking block through the bridge as a synthesised `ThoughtStep`, and use the result both for *information import* (folding frontier reasoning into akh-medu's state) and as a *benchmark target* (Phase 41g comparator: re-encoded frontier traces vs. symbolic Loop traces on identical prompts). Capture in **[ADR 055 — API trace re-encoding (Pattern B)](../decisions/055-api-trace-reencoding.md)** and reflected in the Pattern B description in §4 above.

3. **A trace-envelope ADR lands before cross-surface stitching.** Phase 41i (new sub-phase) defines a single `trace_id` envelope schema joining `DerivationKind::AgentDecision` records and `DerivationKind::ThoughtStep` records, with the `Phase 41h.2` `Decision::Deliberated { trace_id, ... }` variant as the join key. Capture in **[ADR 056 — Unified trace envelope](../decisions/056-unified-trace-envelope.md)**.

4. **The plan surfaces in human-readable markdown.** Pattern D produces *two* outputs in parallel: short prose for the chat surface, and a markdown plan document rendered from the trace (ordered ThoughtSteps, chosen tools, results, verifier scores, confidence interval). This is akh-medu's Plan-mode equivalent. Implementation is template-driven for structure with T5 fill-in for free-form prose; the template lives at `src/nlg/plan_markdown.rs` and is part of the Pattern D stub effort. Reflected in the Pattern D description in §4 above.

5. **Phase 41g benchmark explicitly extends to cover prompt → plan → execute.** Beyond the inference benchmarks already in the Phase 41 plan (multi-hop KG, analogical retrieval, Raven's matrices, bootstrap-domain Q&A), Phase 41g now also benchmarks:
   - The full §5 worked-example flow (prompt → NLU → ThinkingLoop → Pattern C dispatch → tool calls → verbalised result), measured for end-to-end success rate, wall time, and VOC consumed.
   - The Pattern B comparator (commitment 2): for the same set of prompts, run the symbolic Loop, also pull a frontier-model thinking trace via OpenRouter and re-encode it, and measure where they agree, where they disagree, and where each beats the other. Either outcome is informative.

### New ADRs to write alongside this plan

- **[ADR 054 — Hidden-state fidelity for the LLM↔VSA bridge](../decisions/054-hidden-state-fidelity.md)** (commitment 1; Phase 26f)
- **[ADR 055 — API trace re-encoding (Pattern B)](../decisions/055-api-trace-reencoding.md)** (commitment 2)
- **[ADR 056 — Unified trace envelope](../decisions/056-unified-trace-envelope.md)** (commitment 3; Phase 41i)

(ADR 053 stays reserved for "Thinking Loop Findings" per the existing Phase 41g plan.)

## 8. How to verify the architecture is real (the demo's success criteria)

Run the worked-example prompt against a developer-laptop akh-medu daemon. Verify:

1. **Tier-3 NLU populates `neural_concepts`** — Pattern A bridge wired in. Inspectable via the NLU debug dump.
2. **`ThinkingLoop::deliberate` runs to convergence on the goal** — at least 8 distinct `ThoughtStep` kinds in the trace. Trace persisted (in-memory if Phase 41d hasn't shipped, Postgres if it has) and queryable via `get_deliberation_trace`.
3. **Pattern C dispatches a Claude Code subprocess that calls back into the akh-medu MCP server** — verify by checking the subprocess's MCP call log shows at least one `triples_of` or `explain_decision` invocation that returns real KG data.
4. **The subprocess writes `src/foo.rs` and `cargo check` exits 0** — actual code, not a mock.
5. **Pattern D verbalisation passes the round-trip fidelity check** — re-encoding the response text through the bridge produces a HyperVec within τ cosine of `Conclusion.answer`.
6. **The full trace is auditable**: `get_deliberation_trace(trace_id)` returns 10+ ThoughtSteps including the AgentDecision impasse marker, the LLM-call primitives with their cost stamps, the verifier scores, and the halt reason.
7. **`CostGuard` daily spend under the configured cap** — one prompt should not exceed `per_task_max_cents`.

If all seven hold on the demo prompt, the architecture is real: a 10K-bit HyperVec carries faithful trace through deliberation, the bridge to a frontier LLM is the peripheral, the agent's planning surface is integrated with the Loop, and the user gets an auditable plan-and-execute trace they can replay.

## 9. Critical files

The five files most load-bearing for this synthesis to materialise:

- [src/vsa/neural_bridge.rs](../../src/vsa/neural_bridge.rs) — exists; encode/decode + `BridgeTrainer`. Patterns A, B, C all flow through it.
- [src/nlu/candle_backend.rs](../../src/nlu/candle_backend.rs) — exists; Qwen hidden-state extraction (currently logits proxy) for Pattern A and Pattern B's text fallback.
- [src/agent/ooda.rs](../../src/agent/ooda.rs) — exists; `decide()` at line 300 is the impasse hook for `ThinkingLoop::pick_tool`.
- [src/agent/resource.rs](../../src/agent/resource.rs) — exists; `compute_voc` is what Phase 41e extends to price LLM calls in the same currency as symbolic steps.
- `src/thinking/loop_runner.rs` — does **not** yet exist; Phase 41a's home for the scheduler that emits `ThoughtStep::LlmCall { pattern, target }` variants and is the integration point all four patterns plug into.

Plus the documents this builds on:
- [Phase 41 plan](2026-04-27-phase41-thinking-engine.md) — add `41h` (agent integration adapter) and `41i` (trace envelope) between 41f and 41g per §6
- [ADR 052](../decisions/052-unified-thinking-loop.md) — frames the LLM-as-peripheral commitment this elaborates
- [Phase 26 plan](2026-03-26-phase26-llm-vsa-deep-integration.md) and [ADR 037](../decisions/037-llm-vsa-deep-integration.md) — the bridge whose 26e wiring this depends on; **add Phase 26f** (hidden-state fidelity) per §7 commitment 1
- [Phase 28 plan](2026-03-26-phase28-semantic-router.md) and [ADR 038](../decisions/038-semantic-router.md) — the router that the four-pattern scheme assumes
- [Phase 33 plan](2026-03-26-phase33-grounded-verbalization.md) and [ADR 043](../decisions/043-grounded-verbalization.md) — Pattern D's home; the **markdown plan template** is a new addition per §7 commitment 4
- [Phase 36 plan](2026-03-28-phase36-delegation-external-scheduling.md) and [ADR 046](../decisions/046-delegation-external-scheduling.md) — Pattern C's home

New ADRs proposed by this plan (per §7):
- **[ADR 054 — Hidden-state fidelity for the LLM↔VSA bridge](../decisions/054-hidden-state-fidelity.md)** (Phase 26f)
- **[ADR 055 — API trace re-encoding (Pattern B)](../decisions/055-api-trace-reencoding.md)**
- **[ADR 056 — Unified trace envelope](../decisions/056-unified-trace-envelope.md)** (Phase 41i)
