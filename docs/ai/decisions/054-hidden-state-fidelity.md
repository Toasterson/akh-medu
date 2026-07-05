# ADR 054 — Hidden-state fidelity for the LLM↔VSA bridge (Phase 26f)

> Date: 2026-04-28
> Status: Proposed
> Phase: 26f (new sub-phase)
> Depends on: Phase 26b (Candle backend, shipped), Phase 26d (NeuralVsaBridge, shipped)
> Related: [Phase 26 plan](../plans/2026-03-26-phase26-llm-vsa-deep-integration.md), [ADR 037](037-llm-vsa-deep-integration.md), [Phase 41 plan](../plans/2026-04-27-phase41-thinking-engine.md), [ADR 052](052-unified-thinking-loop.md), [synthesis plan](../plans/2026-04-28-prompt-to-plan-tool-workflow.md)
> Anima ticket: TBD (file under Phase 41 thread `019dd0c5-9f4b-74e3-ad5c-3bc0a22e0ef9` until promoted)

## Context

The LLM↔VSA bridge in [`src/vsa/neural_bridge.rs`](../../src/vsa/neural_bridge.rs) (Phase 26d, ~450 LOC) expects a faithful intermediate hidden-state vector from the source LLM as input to its `[10000 × 1536]` learned linear projection. Today, [`CandleBackend::extract_hidden_state`](../../src/nlu/candle_backend.rs:275) returns the **last-token logits** as a stand-in: a `vocab_size`-D distribution over output tokens, not the residual-stream activation that the bridge's design assumes.

This is acceptable for the bring-up demo described in the [prompt → plan → tool workflow synthesis](../plans/2026-04-28-prompt-to-plan-tool-workflow.md) §5 — the bridge can be trained against logits-as-proxy and Pattern A (concept grounding) will produce *some* useful clustering. But it is not faithful to the underlying thesis (Phase 41 / ADR 052) that *attention is approximate VSA binding* — that thesis lives in the residual stream, not in the output logit distribution. As long as the bridge consumes logits, Pattern A's encoder is a softmax-shaped proxy of the true latent, and Pattern B (re-encoding frontier-model thinking traces, [ADR 055](055-api-trace-reencoding.md)) cannot fairly compare against akh-medu's symbolic Loop — different layers of the LLM are being measured.

The constraint is upstream: candle-transformers, the Rust inference library akh-medu uses for in-process LLM serving, does not expose a stable hook API for capturing intermediate residual-stream activations during inference. There is no `forward_with_hooks`, no `OutputHiddenStates` flag the way HuggingFace's `transformers` exposes. The two known paths to a faithful hidden state are:

1. **Patch candle-transformers** to add a hook API for Qwen2 (and ideally a generic trait so other architectures inherit). This means writing a small change to upstream and either upstreaming it or vendoring the patch. The maintainer responsiveness, API design, and licence story all need evaluation.
2. **Migrate the bridge's source LLM** to a candle-supported model that already exposes intermediate states cleanly, or one whose architecture is small enough that akh-medu can vendor a forked implementation. Candidate models worth investigating: Mistral 7B (well-supported in candle, similar quality to Qwen2.5-1.5B), Pythia (small, designed for interpretability research, MIT licensed), GPT-NeoX, or a Llama 3 small variant.

A wrong choice is expensive to undo: the bridge encoder is *trained against* whichever vector source is chosen. Migrating the source LLM after the encoder has been trained against another model means retraining the encoder from scratch.

## Decision

Add **Phase 26f — Hidden-state fidelity** as a sub-phase under Phase 26 (LLM-VSA Deep Integration). Scope:

1. **Investigation pass (~1 week):**
   - Audit candle-transformers to estimate the size and risk of a hook-API patch for Qwen2 specifically. Read the Qwen2 model implementation in candle-transformers, identify the hookable points (post-attention residual, post-MLP residual, layer-output), draft a minimal hook trait. Estimate maintenance burden of vendoring the patch indefinitely.
   - Survey candidate hidden-state-native models in candle: Mistral, Pythia, GPT-NeoX, Llama 3. Compare on (a) inference quality at akh-medu scale, (b) memory/latency budget, (c) licence, (d) availability of pre-quantised weights, (e) ease of extracting layer-N residual stream.
   - Score paths against the criteria: faithful 1536-D (or comparable) hidden-state stream, lowest maintenance burden, cleanest licence story.

2. **Decision and implementation (~2–4 weeks depending on path):**
   - **Path A (candle-transformers hook patch):** vendor the patch in akh-medu's `vendor/candle-transformers/`, file an upstream PR, modify [`CandleBackend::extract_hidden_state`](../../src/nlu/candle_backend.rs:275) to use the hook. Expected ~600 LOC including the patch.
   - **Path B (model migration):** add `src/llm/<model>_backend.rs` implementing the same `LlmBackend` trait the existing `CandleBackend` exposes; deprecate Qwen-as-bridge-source (Qwen continues to serve NLU Tier 3); retrain `NeuralVsaBridge` against the new hidden-state distribution. Expected ~1,500 LOC plus retraining time.

3. **Cross-cutting: bridge encoder retraining.** Whichever path, the encoder weights in [`NeuralVsaBridge`](../../src/vsa/neural_bridge.rs) need to be re-fit against the new vector source. The `BridgeTrainer` (line 253) infrastructure already exists; the work is rerunning the SGD against the same `ItemMemory` ground-truth.

4. **Evaluation gate:** before declaring 26f complete, demonstrate that re-encoding a held-out set of NLU phrases produces tighter `ItemMemory::search` similarity scores than the current logits-as-proxy baseline. The metric is mean nearest-neighbour cosine similarity on a fixed eval set. Improvement of ≥ 0.05 versus baseline is the acceptance bar; lower than that, reopen the investigation.

## Consequences

### Positive

- Pattern A (concept grounding) becomes faithful — the bridge encodes what it claims to encode.
- Pattern B (API trace re-encoding) can fairly benchmark frontier-model thinking traces against akh-medu's symbolic Loop, because both paths now end in the same kind of hidden-state vector.
- The "frontier LLMs implement an approximate VSA, akh-medu has the exact one" thesis (ADR 052) is testable rather than aspirational — the bridge's encode/decode error is now a well-defined quantity over a well-defined input.
- Future Phase 27 (Live KG→LLM training) inherits a clean encode/decode round-trip for free.

### Negative

- Implementation cost: ~600–1,500 LOC depending on path, plus retraining time.
- Path A creates a vendor-patch maintenance liability if upstream rejects or stalls. Path B creates a migration that touches every test fixture trained against Qwen residuals.
- Either path delays Pattern B's full quality bar; the bring-up demo (synthesis §5) ships without Pattern B's frontier-trace comparator.

### Neutral

- Qwen2.5 likely remains the NLU Tier 3 backend regardless of path — NLU's GBNF-constrained generation needs are independent of the bridge's hidden-state needs. Path B specifically would add a *second* in-process LLM to the daemon, with the memory implications that follow.
- Whichever path is chosen drives the LOC count for follow-up work in Pattern B (because the encoder retraining fixes one variable in that comparator).

## Rejected alternatives

### "Stay with logits-as-proxy and accept the infidelity"

Defensible for the demo, indefensible for the production thesis. Phase 41g's benchmark depends on Pattern B, and Pattern B depends on a faithful encode. Punting 26f means punting the empirical validation of the architecture's frontier claim.

### "Train a separate logit→hidden projector to reconstruct the residual"

Tempting but circular: a projector trained from logits to hidden states is by definition a partial inverse of the LLM head, learned with no ground truth except "what makes downstream tasks work better". It buys nothing the encoder retraining doesn't already buy and adds a moving part.

### "Implement a Python-process inference server using HuggingFace transformers"

Rejected for the same reason ADR 037 rejected it for the bridge originally: Python in-process is not how akh-medu deploys, and IPC overhead per call breaks the per-step VOC budgeting in Phase 41e.

## References

- [Phase 26 plan](../plans/2026-03-26-phase26-llm-vsa-deep-integration.md)
- [ADR 037 — LLM-VSA Deep Integration](037-llm-vsa-deep-integration.md)
- [ADR 052 — Unified Thinking Loop](052-unified-thinking-loop.md)
- [ADR 055 — API trace re-encoding (Pattern B)](055-api-trace-reencoding.md)
- [Synthesis plan: prompt → plan → tool workflow](../plans/2026-04-28-prompt-to-plan-tool-workflow.md) — §7 commitment 1
- [`src/vsa/neural_bridge.rs`](../../src/vsa/neural_bridge.rs)
- [`src/nlu/candle_backend.rs:275`](../../src/nlu/candle_backend.rs:275)
- candle-transformers project: https://github.com/huggingface/candle (audit hookability of Qwen2 implementation)
