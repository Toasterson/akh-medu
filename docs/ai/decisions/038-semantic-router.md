# ADR 038 — `n akh` Semantic Router

> Date: 2026-03-26
> Status: Proposed
> Phase: 28
> Depends on: Phase 26b (Candle backend), Phase 14m (NLU complete)
> Enhances: Phase 27 (live training feedback loop)

## Context

The current NLU pipeline uses a fixed tier cascade: rule parser (Tier 1) → ONNX
NER (Tier 2) → Qwen LLM (Tier 3) → VSA ranker (Tier 4). Each tier is tried in
order; the first successful parse wins. This is effective but rigid — there is no
cost awareness, no learning from success/failure patterns, and no ability to skip
tiers or route directly to the best handler for a given query type.

Research on dynamic LLM routing (MixLLM, GreenServ, vLLM Semantic Router) shows
that intelligent routing can reduce costs by up to 75% while maintaining quality,
by matching task complexity to the cheapest capable model.

## Decision

### Implement `n akh` as a VSA-based semantic router

`n akh` sits in front of all inference layers and routes each incoming request to
the optimal handler based on learned task-model affinity.

**Routable backends** (all existing + Phase 26):

| Backend | Cost | Latency | Capability | When to Use |
|---|---|---|---|---|
| Rule parser (Tier 1) | Free | <1ms | Structured commands, simple facts | Commands, greetings, known patterns |
| ONNX NER (Tier 2) | Free | ~5ms | Entity extraction | Named entity queries |
| Candle LLM (Tier 3) | Free | ~200ms | Complex NLU, dialogue | Ambiguous natural language |
| VSA direct (no LLM) | Free | <1ms | Similarity search, KG queries | Symbolic queries, known concepts |
| Cloud API (optional) | $$$ | ~1-3s | Frontier reasoning | Complex multi-step, code generation |
| Burn-trained local (Phase 27) | Free | ~200ms | Domain-specialized | Domain tasks the local model was fine-tuned for |

### Routing via Contextual Multi-Armed Bandit (LinUCB)

Each query is classified into a feature vector using lightweight VSA encoding.
LinUCB maintains per-backend confidence intervals and selects the backend with
the highest upper confidence bound for the query's feature context.

When Phase 27 training improves the local model, `n akh` receives a signal and
increases exploration for that backend. As it succeeds on tasks it previously
failed, the bandit shifts traffic automatically.

### VSA-native feature extraction (no extra embedding model)

Unlike MixLLM (which uses a separate ModernBERT encoder), `n akh` uses the
existing VSA infrastructure for feature extraction:
- Encode query text via `encode_label()` → HyperVec
- Compute similarity against task archetype vectors (stored in ItemMemory)
- Extract features: text length, entity count, question type, domain match score
- These features form the context vector for LinUCB

This avoids adding another model and leverages the VSA stack we already have.

## Rejected Alternatives

### Fixed cascade (current approach)
Works but wastes resources: simple commands still flow through ONNX + LLM before
the rule parser result is accepted. No learning.

### Separate embedding model for routing (ModernBERT)
Adds ~300 MB model + inference overhead just for routing. Our VSA feature
extraction achieves the same with zero additional memory.

### Static routing table
Cannot adapt as the local model improves via Phase 27 training.

## Consequences

- New module: `src/nlu/router.rs`
- New idle task: bandit weight persistence (save/load LinUCB state)
- NLU pipeline changes from sequential cascade to router-first dispatch
- Existing tier cascade becomes fallback when bandit confidence is low
- Phase 27 training completion signals feed into router exploration parameter
- MCP tool: `route_query` for external callers to use the router directly

## References

- MixLLM: https://arxiv.org/abs/2502.18482
- GreenServ: https://arxiv.org/html/2601.17551
- vLLM Semantic Router: https://vllm.ai/blog/semantic-router
- LinUCB algorithm: Li et al., "A Contextual-Bandit Approach to Personalized
  News Article Recommendation" (WWW 2010)
