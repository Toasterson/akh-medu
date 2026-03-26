# Phase 28 — `n akh` Semantic Router

> Date: 2026-03-26
> Status: Planned
> Phase: 28
> Depends on: Phase 26b (Candle backend available as routing target)
> Enhances: Phase 27 (training feedback loop), Phase 14m (NLU pipeline)
> ADR: [038-semantic-router](../decisions/038-semantic-router.md)

## Motivation

The current NLU pipeline is a fixed cascade: Tier 1 → 2 → 3 → 4. Every query
flows through the same sequence regardless of complexity. A greeting wastes
cycles in ONNX NER and the LLM. A complex philosophical question gets the same
treatment as "hello."

`n akh` ("the intelligence of akh") is a semantic router that learns which
backend handles which query type best, and routes directly.

## Architecture

```
Incoming query
    │
    ▼
┌──────────────────────────────────┐
│ n akh Router                     │
│                                  │
│ 1. VSA feature extraction        │
│    (encode_label → archetype     │
│     similarity + text features)  │
│                                  │
│ 2. Semantic cache check          │
│    (recent identical queries)    │
│                                  │
│ 3. LinUCB bandit selection       │
│    (context → backend scores)    │
│                                  │
│ 4. Confidence gate               │
│    (if low → fall back to        │
│     sequential cascade)          │
└──────┬───────────────────────────┘
       │
       ├──▶ Rule parser         (free, <1ms)
       ├──▶ ONNX NER            (free, ~5ms)
       ├──▶ Candle LLM          (free, ~200ms)
       ├──▶ Candle LLM + probe  (free, ~300ms, with hidden state extraction)
       ├──▶ VSA direct           (free, <1ms, skip LLM entirely)
       ├──▶ Cloud API            ($$$, ~1-3s, frontier capability)
       └──▶ Burn-trained local   (free, ~200ms, domain-specialized, Phase 27)
```

## Sub-phases

### 28a — Router core + VSA feature extraction (~500 lines)

**New module**: `src/nlu/router.rs`

```rust
pub struct NakhRouter {
    bandits: Vec<BackendBandit>,     // One LinUCB arm per backend
    archetypes: Vec<QueryArchetype>, // VSA prototype vectors per task type
    cache: LruCache<u64, CachedResult>, // Semantic result cache
    config: RouterConfig,
}

pub struct QueryArchetype {
    pub name: String,           // "greeting", "fact_query", "complex_reasoning", etc.
    pub prototype: HyperVec,    // VSA centroid for this query type
    pub preferred_backend: Option<BackendId>, // Hint (overridable by bandit)
}

pub struct RouterConfig {
    pub exploration_rate: f32,    // LinUCB alpha (default 0.25)
    pub cache_size: usize,       // Semantic cache capacity (default 256)
    pub min_confidence: f32,     // Below this → fall back to cascade (default 0.4)
    pub enable_cloud: bool,      // Allow routing to cloud APIs (default false)
}
```

**Feature extraction** (from query text, no extra model):
- `text_length`: normalized character count
- `entity_density`: ratio of resolved tokens to total tokens (from lexer)
- `question_type`: extracted from grammar parser (fact/command/goal/query)
- `domain_similarity`: max cosine similarity to loaded compartment prototypes
- `archetype_scores`: similarity to each QueryArchetype's VSA prototype
- `has_negation`, `has_temporal`, `has_modal`: grammar feature flags

### 28b — LinUCB bandit implementation (~400 lines)

**New module**: `src/nlu/bandit.rs`

LinUCB maintains per-backend:
- `A`: d×d matrix (context correlation)
- `b`: d×1 vector (reward correlation)
- `theta`: A⁻¹b (estimated reward weights)

Selection: `score = theta·x + alpha * sqrt(x·A⁻¹·x)` (UCB)

**Reward signal**: After routing, observe outcome:
- Parse succeeded → reward = 1.0
- Parse failed, fallback succeeded → reward = 0.0 (penalize)
- Latency bonus: reward += max(0, 1.0 - latency/budget)
- Cost penalty: reward -= cost_factor * api_cost

**Persistence**: Save/load bandit state via redb (same pattern as parse ranker).
Wire into daemon idle tasks for periodic persistence.

### 28c — Semantic cache (~200 lines)

Hash query text → check LRU cache for recent identical or near-identical results.
"Near-identical" uses VSA similarity: if `similarity(encode(query), cached_key) > 0.95`,
return cached result.

Invalidation: cache entries expire after configurable TTL (default 5 minutes).
KG mutation events (new triples) invalidate relevant cache entries.

### 28d — Pipeline integration (~400 lines)

Replace the sequential cascade in `NluPipeline::parse()` with router-first dispatch:

```rust
pub fn parse(&self, input: &str, ctx: &NluContext) -> NluResult<NluParseResult> {
    // 1. Router selects backend
    let route = self.router.route(input, ctx)?;

    // 2. Execute selected backend
    let result = match route.backend {
        Backend::RuleParser => self.tier1_parse(input, ctx),
        Backend::OnnxNer => self.tier2_parse(input, ctx),
        Backend::CandleLlm => self.tier3_parse(input, ctx),
        Backend::VsaDirect => self.vsa_direct_parse(input, ctx),
        Backend::CloudApi => self.cloud_parse(input, ctx),
        Backend::BurnLocal => self.burn_local_parse(input, ctx),
    };

    // 3. If failed, fall back to sequential cascade
    let result = result.or_else(|_| self.cascade_parse(input, ctx))?;

    // 4. Report reward to bandit
    self.router.report_reward(route.id, &result);

    Ok(result)
}
```

### 28e — Phase 27 integration hook (~100 lines)

When Phase 27 training completes and hot-swaps model weights:
1. Burn signals completion via channel
2. `n akh` receives signal, bumps exploration rate for `BurnLocal` backend
3. Bandit gradually discovers the model's improved capabilities
4. Traffic shifts from cloud API → local as the model proves itself

This creates the self-optimizing feedback loop: train → deploy → explore → exploit.

## Query Archetype Bootstrap

Initial archetypes (seeded during workspace initialization):

| Archetype | Example Queries | Expected Best Backend |
|---|---|---|
| `greeting` | "hello", "good morning" | Rule parser |
| `fact_assertion` | "dogs are mammals" | Rule parser |
| `entity_query` | "who is Alan Turing?" | ONNX NER + VSA direct |
| `relationship_query` | "how are X and Y related?" | VSA direct |
| `complex_reasoning` | "if A causes B and B prevents C..." | Candle LLM |
| `code_request` | "write a function that..." | Cloud API or Burn-trained |
| `meta_command` | "show my goals", "list triples" | Rule parser |
| `temporal_query` | "what happened after X?" | Rule parser (temporal patterns) |

Archetypes are VSA-encoded from their example queries via `encode_label()` +
bundling. New archetypes can be discovered via clustering in the parse ranker's
exemplar memory (Tier 4).

## Estimated Effort

| Sub-phase | Lines | Dependencies |
|---|---|---|
| 28a — Router core | ~500 | Phase 26b (Candle as target) |
| 28b — LinUCB bandit | ~400 | None |
| 28c — Semantic cache | ~200 | None |
| 28d — Pipeline integration | ~400 | 28a, 28b |
| 28e — Phase 27 hook | ~100 | Phase 27 |
| **Total** | **~1,600** | |

## New Dependencies

None — LinUCB is ~100 lines of linear algebra, implementable with existing
`nalgebra` or even raw f32 arrays for the small dimensionality involved.
