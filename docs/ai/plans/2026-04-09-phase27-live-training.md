# Phase 27 — Live KG→LLM Training

> Date: 2026-04-09
> Status: Planned
> Phase: 27
> Depends on: Phase 26 (Candle backend, neural bridge)
> Enhances: Phase 33 (Grounded Verbalization), Phase 32 (Knowledge Extraction)
> ADR: [047-live-training](../decisions/047-live-training.md)

## Motivation

Phase 26 established the **read path** (LLM → VSA) via the neural bridge. Phase
27 adds the **write path** (VSA → LLM) and the **training loop** that allows the
local LLM to learn from the knowledge graph.

The key insight: akh-medu accumulates verified knowledge in its KG through
normal operation (user assertions, rule inference, consolidation). This knowledge
can be verbalized into training pairs and used to fine-tune the local LLM,
creating a self-improving loop where the model gets better at understanding
the user's domain over time.

### Cold-start reality

On a fresh workspace, there's nothing to train on. The training data collection
pipeline runs from day one but produces no training output until thresholds are
met (~500 NLU pairs, ~200 NLG pairs). All training is deferred until sufficient
data accumulates. This is by design — the model adapts organically.

## Architecture

```
  KG (verified triples)
       │
       ├──▶ Grammar linearize() → natural text
       │         │
       │         ▼
       │    Training pair: (text, AbsTree JSON)
       │         │
       │         ├──▶ NLU pairs:  (input_text, abstree_json)
       │         ├──▶ NLG pairs:  (triples, output_text)
       │         └──▶ Elicitation pairs: (prompt, extracted_triples)
       │
       ▼
  TrainingDataStore (redb table)
       │
       ├── Threshold check: enough pairs?
       │     │
       │     No → wait, keep collecting
       │     Yes ↓
       │
       ▼
  Burn Training Loop (spawn_blocking)
       │
       ├── LoRA adapters for Qwen2.5
       ├── Gradient checkpointing (M2 memory budget)
       ├── AdamW optimizer, cosine LR schedule
       │
       ▼
  Export LoRA weights (safetensors)
       │
       ├──▶ Candle loads base + LoRA
       └──▶ Neural bridge verification (probe before/after)
```

## Sub-phases

### 27f — Training Data Collection Pipeline (~500 lines)

**Priority**: Highest (prerequisite for ALL fine-tuning)
**Depends on**: Nothing (uses existing engine APIs)

**Rationale**: Must be implemented first because it starts collecting data
immediately. All other sub-phases consume data from this pipeline.

**New module**: `src/training/data_collector.rs`

```rust
pub struct TrainingDataCollector {
    store: TrainingDataStore,
    config: CollectorConfig,
}

pub struct CollectorConfig {
    /// Minimum NLU pairs before training can start (default: 500).
    pub nlu_threshold: usize,
    /// Minimum NLG pairs before training can start (default: 200).
    pub nlg_threshold: usize,
    /// Maximum pairs to store per category (default: 50_000).
    pub max_pairs: usize,
}

pub enum TrainingPair {
    /// NLU: natural text → AbsTree JSON.
    Nlu { input: String, abstree_json: String, source_tier: u8 },
    /// NLG: triples → natural text.
    Nlg { triples: Vec<LabelTriple>, output_text: String, grammar: String },
    /// Elicitation: prompt → extracted triples (positive or negative).
    Elicitation { prompt: String, triples: Vec<LabelTriple>, accepted: bool },
}

pub struct TrainingDataStore {
    db: redb::Database,
}
```

**Data sources**:
1. **NLU pairs**: Every successful parse from any NLU tier is logged.
   Tier 1 (rule parser) provides ~70% of examples. Tier 3 (LLM) complex cases.
2. **NLG pairs**: Every grammar linearization is logged as `(triples, text)`.
3. **Elicitation pairs**: Phase 32 LLM-generated triples that pass validation
   become positive examples. Rejected ones are negative examples.

**Storage**: redb table `training_pairs` with category key prefix.

**Integration points**:
- Hook into `NluPipeline::parse()` success path
- Hook into `Engine::linearize()` output
- Hook into future Phase 32 knowledge extraction

### 27a — Triple Verbalization Pipeline (~400 lines)

**Priority**: High
**Depends on**: 27f (stores pairs), existing grammar linearization

**New module**: `src/training/verbalizer.rs`

Converts KG triples into diverse natural language training pairs:

```rust
pub struct TripleVerbalizer {
    grammar_registry: Arc<GrammarRegistry>,
}

pub struct VerbalizationResult {
    pub pairs: Vec<(String, String)>,  // (text, abstree_json)
    pub triples_verbalized: usize,
}
```

**Verbalization strategies**:
1. **Direct linearization**: Use grammar framework's 4 archetypes
   (narrative, formal, terse, custom)
2. **Template variation**: "X is a Y" → "X is a type of Y" / "X belongs to Y"
3. **Reverse construction**: From AbsTree, generate multiple surface forms
4. **Negative sampling**: Corrupt triples (swap subject/object) for contrastive pairs

### 27b — Burn Training Loop + LoRA (~800 lines)

**Priority**: Medium
**Depends on**: 27f (data), 27a (verbalization)

**New module**: `src/training/burn_trainer.rs`

```rust
pub struct BurnTrainerConfig {
    pub lr: f64,                    // Learning rate (default: 1e-4)
    pub batch_size: usize,          // Batch size (default: 4)
    pub max_epochs: usize,          // Maximum epochs (default: 3)
    pub lora_rank: usize,           // LoRA rank (default: 8)
    pub lora_alpha: f32,            // LoRA alpha (default: 16.0)
    pub gradient_checkpointing: bool, // Memory optimization (default: true)
    pub early_stopping_patience: usize, // (default: 2)
}
```

**Architecture**:
- LoRA adapter layers for attention projections (Q, K, V, O)
- AdamW optimizer with cosine LR schedule
- Gradient checkpointing for M2 memory budget
- `tokio::task::spawn_blocking` to isolate from async I/O
- Export LoRA weights as safetensors

**Memory budget** (Qwen2.5-1.5B on M2 16GB):
- Base model (GGUF Q4_K_M): ~1 GB
- KV cache (PolarQuant): ~200 MB
- LoRA adapters (rank 8): ~12 MB
- Gradient state: ~50 MB (with checkpointing)
- Overhead: ~200 MB
- Total: ~1.5 GB (fits comfortably)

### 27c — VSA→Neural Decoder (~400 lines)

**Priority**: Low (Phase 33 will benefit more)
**Depends on**: 27b (training infrastructure)

**Extends**: `src/vsa/neural_bridge.rs`

Reverse the Phase 26d bridge: HyperVec → LLM hidden state.

```rust
impl NeuralVsaBridge {
    /// Decode a HyperVec back to an approximate LLM hidden state.
    pub fn decode(&self, hypervec: &HyperVec) -> Vec<f32> {
        self.decoder.as_ref().unwrap().forward(&hypervec_to_float(hypervec))
    }
}
```

Trained alongside LoRA adapters using paired (hidden_state, HyperVec) data.

### 27d — Verification & Conflict Detection (~300 lines)

**Priority**: Medium (safety gate)
**Depends on**: 27b (training), Phase 26d (neural bridge)

**New module**: `src/training/verifier.rs`

After each training batch:
1. Re-probe the model via the neural→VSA bridge
2. Verify alignment hasn't degraded (cosine similarity to known concepts)
3. E-graph contradiction check before accepting training data
4. Rollback LoRA weights if verification fails

```rust
pub struct TrainingVerifier {
    bridge: Arc<NeuralVsaBridge>,
    baseline_similarities: HashMap<SymbolId, f32>,
    degradation_threshold: f32,  // default: 0.1
}

pub enum VerificationResult {
    Passed,
    Degraded { worst_concept: SymbolId, delta: f32 },
    Contradicted { rule: String },
}
```

### 27e — Background Training Scheduler (~300 lines)

**Priority**: Medium
**Depends on**: 27b (training), daemon infrastructure

**Extends**: `src/agent/daemon.rs`

Wire training into the daemon's idle task system:

```rust
pub training_interval: Duration,  // default: 2 hours

// In tokio::select!:
_ = training_tick.tick() => {
    self.run_training_cycle();
}
```

**Scheduling rules**:
- Only train during idle periods (no active goals, no pending delegations)
- Rate-limit to avoid starving inference
- Skip if below data thresholds
- Integrate with sleep/consolidation cycle

## Implementation Order

1. **27f** — Data collection (start accumulating immediately)
2. **27a** — Verbalization (generates more training pairs from KG)
3. **27b** — Burn training loop (the core training infrastructure)
4. **27d** — Verification (safety gate before accepting weights)
5. **27e** — Background scheduler (wires into daemon)
6. **27c** — VSA→Neural decoder (optional, enhances Phase 33)

## Estimated Effort

| Sub-phase | Lines | Complexity | Dependencies |
|-----------|-------|------------|--------------|
| 27f — Data collection | ~500 | Low | None |
| 27a — Verbalization | ~400 | Medium | 27f |
| 27b — Burn training | ~800 | High | 27f, 27a |
| 27c — VSA decoder | ~400 | Medium | 27b |
| 27d — Verification | ~300 | Medium | 27b, Phase 26d |
| 27e — Background scheduler | ~300 | Low | 27b |
| **Total** | **~2,700** | | |

## New Dependencies

```toml
# Feature: burn-training
burn = { version = "0.21", optional = true, features = ["train", "ndarray"] }
```

NdArray backend for CPU training (no GPU required). The `train` feature pulls
in `burn-train` with LearnerBuilder, metrics, checkpointing.

## Feature Flags

```toml
[features]
burn-training = ["burn", "candle-backend"]  # Requires Candle for weight export
training-data = []                          # Just data collection (no Burn dep)
```

`training-data` has no extra dependencies — it only uses redb (already a dep)
and the grammar framework. This allows data collection to start immediately
even without Burn installed.

## Relationship to Other Phases

| Phase | Relationship |
|-------|-------------|
| **Phase 26** | Provides Candle backend + neural bridge (prerequisite) |
| **Phase 32** | Knowledge extraction feeds elicitation training pairs |
| **Phase 33** | Grounded verbalization benefits from fine-tuned NLG |
| **Phase 35** | Offline T5 training produces base models; Phase 27 refines in-situ |
