# ADR 055 — API trace re-encoding (Pattern B)

> Date: 2026-04-28
> Status: Proposed
> Phase: 41 (one of the four LLM-call patterns inside the Thinking Loop)
> Depends on: ADR 052 (Unified Thinking Loop), ADR 054 (Hidden-state fidelity — for production-quality vectors), Phase 26d (NeuralVsaBridge), Phase 28 (Semantic Router — uses Pattern B as a routable backend)
> Related: [Phase 41 plan](../plans/2026-04-27-phase41-thinking-engine.md), [synthesis plan](../plans/2026-04-28-prompt-to-plan-tool-workflow.md)
> Anima ticket: TBD (file under Phase 41 thread `019dd0c5-9f4b-74e3-ad5c-3bc0a22e0ef9` until promoted)

## Context

ADR 052 establishes that frontier LLMs implement an approximate, gradient-discovered VSA, while akh-medu has the exact, audited version. The synthesis plan (§4) defines four LLM-call patterns the Phase 41 `ThinkingLoop` can emit. Pattern B is the one that *imports* frontier-model reasoning into akh-medu's symbolic state.

Two design questions to resolve about Pattern B:

1. **Which backend** does Pattern B target? Local Claude Code subprocess (Phase 36 LocalClaude)? Local Candle/Qwen inference? A remote API?
2. **What does Pattern B consume** from that backend? Just final answers? Full hidden states? Something else?

The answers are linked. Local Claude Code subprocess returns text + structured triples but does **not** expose intermediate "thinking" — Anthropic's CLI hides the extended-thinking blocks from the print-mode output. Local Candle/Qwen exposes hidden states (or will once ADR 054 resolves) but its reasoning quality is far below frontier; the value of fanning out N=3 small-model rollouts is real but limited. Remote frontier APIs — Anthropic, OpenAI, DeepSeek, Gemini — *do* expose extended-thinking trace blocks alongside the final answer, but each has a different API shape.

**OpenRouter** sits at the obvious intersection: it's a multiplexer over all the major frontier providers, normalises their thinking-block formats into a single response schema, handles rate limits and fallbacks, and is the ergonomic way to consume *traced* frontier reasoning from a single client.

This shifts what Pattern B *is*. It is not the analogue of "ask Claude Code to think and tell me what it concluded" (that's Pattern C, where the doing is the point). It is the analogue of "consume someone else's already-traced thinking and fold it into my own state" — an *information import* operation on a thinking trace, not an outsourced computation.

The mechanism: the OpenRouter response includes thinking-block chunks alongside the final answer. Each chunk is run through the bridge encoder ([`NeuralVsaBridge::encode`](../../src/vsa/neural_bridge.rs:191)) producing a HyperVec; that HyperVec lands in the akh-medu trace as a synthesised `ThoughtStep` whose kind reflects the chunk's structural role (tentative-hypothesis, contradiction-noticed, hypothesis-emitted, conclusion). The final answer is similarly encoded. Both per-chunk vectors and the answer vector enter `state.hypotheses` in superposition; `superposition_merge` bundles, `verifier_score` (Dempster combination over `state.hypotheses`) prunes the weak ones.

## Decision

Pattern B is **API-trace re-encoding via OpenRouter** (and equivalent thinking-trace-emitting APIs). Scope:

1. **HTTP client + thinking-trace parser** in `src/thinking/openrouter.rs` (~250 LOC):
   - Auth via env-stored API key (`AKH_OPENROUTER_API_KEY`); explicit error-mode if absent.
   - Request schema: prompt + model + thinking-mode flag + `voc_credits` cost ceiling.
   - Response schema: parses OpenAI-compatible delta stream, extracts `thinking_blocks` (Anthropic-style), `reasoning` (DeepSeek-style), and similar fields per provider; normalises to akh-medu's internal `ApiThinkingTrace { chunks: Vec<TraceChunk>, final_answer: String }`.
   - Cost accounting: every response carries a `usage` block (tokens in/out) which is converted into `voc_credits` using the same model-cost map Phase 36 `CostGuard` uses.

2. **Trace re-encoder** in `src/thinking/trace_reencode.rs` (~250 LOC):
   - Input: `ApiThinkingTrace`. Output: `Vec<ThoughtStep>` plus a `Vec<(HyperVec, BeliefInterval)>` for `state.hypotheses`.
   - For each chunk: encode the chunk text via [`CandleBackend::extract_hidden_state`](../../src/nlu/candle_backend.rs:275) → [`NeuralVsaBridge::encode`](../../src/vsa/neural_bridge.rs:191) → HyperVec. Map the chunk's structural shape (tentative-hypothesis, contradiction-noticed, conclusion, etc.) to a `ThoughtStep` variant.
   - For the final answer: encode similarly; emit a `ThoughtStep::ApiTraceConclusion` plus a hypothesis with `BeliefInterval` derived from the API's reported confidence (where exposed) or a flat default.
   - All emitted `ThoughtStep`s carry a `provenance: ProvenanceTag::ApiTrace { provider, model, chunk_index }` so downstream queries can tell symbolic-derived steps from API-imported ones.

3. **Loop integration:** `ThinkingLoop` scheduler (Phase 41b) gains a new primitive `ApiTraceReencode { provider, model, prompt }`. The scheduler emits this primitive when:
   - Argumentation produced no clear winner (verifier `score < 0.5`, `BeliefIntervalNarrowed` flat over K steps), OR
   - The user's goal explicitly requested a "frontier comparison" (`Goal::tag = "frontier_comparison"`), OR
   - The Phase 28 router (when it ships) elects this backend over symbolic-only continuation.

4. **Provenance commitment:** API-trace re-encoded steps are clearly distinguishable in the trace. They are *not* native akh-medu reasoning. They appear as `ThoughtStep`s for trace continuity but with a non-default `provenance` tag and never count toward the `verifier_score`'s "convergence" criterion alone — convergence requires symbolic verification of the imported claim.

## Consequences

### Positive

- Frontier reasoning becomes a *consumable* in akh-medu, not a competitor. The symbolic core remains the reasoner; the API trace is an *input* to verification, never a substitute for it.
- Phase 41g benchmark gets a comparator: same prompt, two paths (symbolic Loop vs. re-encoded frontier trace), measured against the same ground truth. Either akh-medu's symbolic reasoning beats the imported trace, validating the architecture, or it doesn't, validating that frontier traces capture something the symbolic primitives don't yet — both informative.
- COCONUT-analogue cost profile: pay O(N) once for the API call and trace re-encoding, then bundle into superposition. Tree-of-Thoughts pays O(N) downstream because each branch persists as text — Pattern B doesn't.
- Multi-provider via OpenRouter; not locked to a single vendor.

### Negative

- Cost: cents per call, scales with thinking-budget the API exposes. `CostGuard` (Phase 36) extends to cover Pattern B spend; the `ThinkingBudget.voc_credits` includes API-cost as a halt criterion (Phase 41e).
- API instability: thinking-block formats vary across providers and change without notice. The trace re-encoder needs a per-provider parser tier; will require maintenance.
- Privacy and data egress: prompts shipped to a remote provider may contain user content. Pattern B is opt-in, gated by configuration (`[delegation.api]` in `akh.toml`) and the daily-budget cost guard.
- ADR 054 (hidden-state fidelity) is a quality prerequisite. Pattern B can ship at lower quality with logits-as-proxy, but its fairness as a benchmark comparator depends on 26f.

### Neutral

- Pattern B does not replace Pattern C (local Claude subprocess). They cover different problems: Pattern B imports a thinking trace, Pattern C dispatches a tool-using subprocess for concrete side effects.
- A `ThoughtStep::ApiTraceReencode` variant is a new addition to the Phase 41 enum; ADR 056 (unified trace envelope) covers schema impact.

## Rejected alternatives

### "Use only local Claude subprocess for Pattern B"

Local Claude doesn't expose intermediate thinking blocks. Re-encoding the print-mode output is degenerate: it's encoding a final answer, not a trace. Lower-fidelity, defeats the COCONUT-analogue purpose.

### "Use only local Candle for Pattern B"

Quality gap is too large. A Qwen2.5-1.5B chain-of-thought trace versus a frontier model's extended-thinking trace is not an apples-to-apples benchmark; the comparator value of Pattern B vanishes.

### "Skip the bridge and just use the API's final answer text"

Then it's not Pattern B, it's Pattern D (verbalize someone else's answer to the user). Pattern B's purpose is to bring the *intermediate reasoning* into akh-medu's state for the verifier and for the comparator benchmark, not just the answer.

### "Ship Pattern B before ADR 054 resolves"

Possible — bridge encodes work today (logits proxy). But Phase 41g's benchmark fairness suffers, and the encoder weights need retraining once 26f settles. Doable as a demo path; honest accounting says the comparator metric is not trustworthy until 26f lands.

## References

- [ADR 052 — Unified Thinking Loop](052-unified-thinking-loop.md)
- [ADR 054 — Hidden-state fidelity](054-hidden-state-fidelity.md)
- [ADR 056 — Unified trace envelope](056-unified-trace-envelope.md)
- [Phase 41 plan](../plans/2026-04-27-phase41-thinking-engine.md) — Pattern B integrates as a new scheduler primitive between 41b and 41c
- [Synthesis plan](../plans/2026-04-28-prompt-to-plan-tool-workflow.md) §4 (Pattern B), §7 commitment 2
- [`src/vsa/neural_bridge.rs`](../../src/vsa/neural_bridge.rs)
- [`src/nlu/candle_backend.rs`](../../src/nlu/candle_backend.rs)
- OpenRouter API: https://openrouter.ai/docs (thinking-block format documentation)
- COCONUT (Hao et al., Meta 2024): https://arxiv.org/abs/2412.06769
