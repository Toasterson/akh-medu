# Phase 26 — LLM-VSA Deep Integration

> Date: 2026-03-26
> Status: Planned
> Phase: 26
> Depends on: Phase 14m (NLU complete), Phase 23 (Affective System complete)
> Enhances: Phase 24 (Sensory Grounding), future Phase 27 (Live Training)
> ADR: [037-llm-vsa-deep-integration](../decisions/037-llm-vsa-deep-integration.md)

## Motivation

The current NLU pipeline treats the local LLM (Qwen2.5-1.5B) as an opaque
text-to-AST translator — we feed it text and get structured JSON back. The LLM's
internal representations (hidden states, attention patterns) are discarded.

Research on Hyperdimensional Probing (2025) and TurboQuant/PolarQuant (ICLR 2026)
suggests that:

1. LLM hidden states can be **projected into VSA space** via a shallow encoder,
   enabling the symbolic reasoning core to "see" what the LLM is thinking
2. KV cache compression via polar coordinate quantization can **6x reduce memory**,
   enabling larger models (7B-8B) on commodity hardware
3. Symbolic solutions can be **decoded back** into LLM hidden states, enabling
   the reasoning core to steer the LLM's generation

This creates a bidirectional neural↔symbolic bridge that our architecture is
uniquely positioned to exploit — we already have the VSA operations, grounding
infrastructure, and provenance tracking to make this interpretable.

## Architecture Overview

```
                         ┌─────────────────────────────┐
                         │      Candle Runtime          │
                         │  (GGUF model, Metal on M2)   │
                         │                              │
  Human text ──────────▶ │  Tokenize → Embed → Layers   │
                         │       │          │           │
                         │       │    ┌─────┘           │
                         │       │    │ Hidden states    │
                         │       ▼    ▼                 │
                         │  KV Cache (PolarQuant 3-bit) │
                         │       │                      │
                         └───────┼──────────────────────┘
                                 │
                    ┌────────────┼────────────────┐
                    │            │                │
                    ▼            ▼                ▼
              Token output  Hidden states   Attention weights
              (text/JSON)   (f32 tensors)   (optional)
                    │            │
                    ▼            ▼
            Existing NLU    Neural→VSA Encoder
            (AbsTree)       (linear + binarize)
                    │            │
                    ▼            ▼
              Symbolic      HyperVec (10K-bit)
              reasoning     │
                    │        ▼
                    │    ItemMemory similarity search
                    │    → concept identification
                    │    → provenance: DerivationKind::NeuralProbe
                    │
                    └────────────────────┐
                                         ▼
                                   KG Integration
                                   (grounding, inference)
```

## Sub-phases

### 26a — ARM NEON SIMD Kernels (~400 lines)

**Priority**: High (prerequisite for M2 performance)
**Depends on**: Nothing

Add NEON intrinsics to `src/simd/` alongside existing AVX2:

| Operation | NEON Intrinsic | Expected Speedup |
|---|---|---|
| XOR bind | `veorq_u8` (16-byte) | ~4x over scalar |
| Bundle accumulate | `vaddq_s8` / `vpaddlq_s8` | ~4x over scalar |
| Hamming popcount | `vcntq_u8` + horizontal add | ~8x over scalar |
| Cosine similarity | `vmulq_s16` + `vpadalq_s16` | ~4x over scalar |

**Files**:
- New: `src/simd/neon.rs`
- Modify: `src/simd/mod.rs` (add NEON detection to `select_kernel()`)

**Testing**: Ensure bit-exact results with generic kernel on same inputs.

### 26b — Candle LLM Backend (~1,200 lines)

**Priority**: High (prerequisite for hidden state access)
**Depends on**: Nothing (parallel with 26a)

Replace `llama-cpp-2` with Candle for GGUF inference.

#### Verified Candle Architecture (from source)

Candle's Qwen2 has a two-tier model structure that is the key enabler:

```rust
// candle-transformers/src/models/qwen2.rs

/// Base model — returns hidden states (post-norm, pre-lm_head)
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,  // each is a separate forward() call
    norm: RmsNorm,
    // ...
}

impl Model {
    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize,
                   attn_mask: Option<&Tensor>) -> Result<Tensor>
    // Returns: hidden states [batch, seq_len, hidden_dim]
}

/// Causal LM wrapper — applies lm_head on top, returns logits
pub struct ModelForCausalLM {
    pub model: Model,  // <-- pub! direct access to base model
    lm_head: Linear,
}
```

The forward pass is a plain Rust loop — no opaque graph:
```rust
let mut xs = self.embed_tokens.forward(input_ids)?;
for layer in self.layers.iter_mut() {
    xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
    // <-- insert xs.clone() here for per-layer capture
}
xs.apply(&self.norm)
```

**Quantized path** (`quantized_qwen2.rs`): Same loop structure with `QMatMul`
weights. Hidden states between layers are **full-precision f32 tensors** — only
weight storage is quantized. The neural bridge receives clean signals.

**KV cache**: Explicit Rust structs (`Cache` / `RotatingCache`), not opaque C
state. We own it — direct manipulation for PolarQuant (Phase 26c).

#### New module: `src/nlu/candle_backend.rs`

```rust
pub struct CandleBackend {
    model: ModelForCausalLM,    // Candle Qwen2 (GGUF quantized)
    tokenizer: Tokenizer,       // HuggingFace tokenizer (already a dep)
    device: Device,             // Device::Metal or Device::Cpu
    config: CandleConfig,
}

pub struct CandleConfig {
    pub max_tokens: u32,        // Generation limit (default: 512)
    pub temperature: f32,       // Sampling temperature (default: 0.0 = greedy)
    pub extract_hidden: bool,   // Capture hidden states (default: false)
    pub hidden_layer: Option<usize>, // Which layer (default: last)
    pub kv_quantize: bool,      // Enable PolarQuant on KV cache
}

pub struct InferenceResult {
    pub text: String,
    pub tokens_generated: u32,
    pub hidden_states: Option<Vec<Tensor>>,  // Per-layer if extract_hidden
    pub final_hidden: Option<Tensor>,        // Last-layer hidden state
    pub abs_tree: Option<AbsTree>,           // Parsed structured output
}
```

#### Hidden state extraction modes

1. **Final only** (default when `extract_hidden = true`):
   Call `model.model.forward()` directly (bypass lm_head).
   Cost: zero overhead — this is the normal forward pass minus one matmul.

2. **Per-layer** (when `hidden_layer = None`):
   Wrap the layer loop to clone `xs` at each step.
   Cost: ~28 tensor clones for Qwen2.5-1.5B (28 layers × [1, seq_len, 1536]).
   Only enable for probing/debugging.

3. **Specific layer** (when `hidden_layer = Some(n)`):
   Clone only at layer N, skip the rest.
   Cost: one tensor clone.

#### Constrained decoding

Candle has no GBNF grammar support. Two strategies:

1. **LogitsProcessor** (preferred): Implement a `JsonLogitsProcessor` that masks
   invalid tokens based on a JSON state machine. Candle's sampling pipeline
   supports custom processors. More robust than GBNF (handles tokenizer edge cases).

2. **Brace-depth tracking** (fallback): Our existing approach from
   `llm_translator.rs` — track `{`/`}` depth during generation, stop when balanced.
   Already works reliably. Use as fallback if LogitsProcessor is too complex initially.

#### Model support (verified in Candle source)

| Model | Candle module | GGUF support | Hidden dim | License |
|---|---|---|---|---|
| Qwen2.5-1.5B | `quantized_qwen2.rs` | Yes (Q4_K_M) | 1,536 | Apache 2.0 |
| Qwen2.5-7B | `quantized_qwen2.rs` | Yes | 3,584 | Apache 2.0 |
| Qwen3 | `quantized_qwen3.rs` | Yes | varies | Apache 2.0 |
| Llama 3.x | `quantized_llama.rs` | Yes | 4,096 | Meta license |
| Mistral-7B | `quantized_mistral.rs` | Yes | 4,096 | Apache 2.0 |
| Gemma-2 | `quantized_gemma2.rs` | Yes | 2,304 | Gemma license |

#### Metal backend

```rust
let device = Device::metal_if_available(0)?;  // graceful CPU fallback
// BF16 supported on Metal (device.supports_bf16() == true)
```

Feature-gated: `--features metal` in Candle. We gate behind our own
`candle-backend` feature which pulls in `candle-core/metal`.

#### Migration path

1. Implement CandleBackend behind `candle-backend` feature flag
2. Implement `LlmTranslator` trait (same interface as llama-cpp-2 backend)
3. Benchmark against llama-cpp-2 on M2:
   - Expected: ~70-90% of llama.cpp throughput (30-55 vs 40-80 tok/s for 1.5B)
   - Acceptable for boundary translator workload
4. If competitive: make default, deprecate llama-cpp-2
5. If slower: keep both — Candle for hidden state extraction, llama-cpp-2 for
   pure NLU speed when probing not needed

### 26c — PolarQuant KV Cache Compression (~600 lines)

**Priority**: Medium (enables larger models)
**Depends on**: 26b (Candle backend)

Implement PolarQuant (stage 1 of TurboQuant) for KV cache compression:

**New module**: `src/vsa/polar_quant.rs`

**Algorithm**:
1. **Random rotation**: Apply Fast Walsh-Hadamard Transform (FWHT) to KV vectors
   - O(d log d) complexity vs O(d²) for dense rotation matrix
   - Deterministic given seed (reproducible across sessions)
   - Preserves Euclidean norms (orthogonal transform)

2. **Polar transform** (for RoPE dimension pairs):
   ```
   (x, y) → (r, θ) where r = √(x²+y²), θ = atan2(y, x)
   ```
   - Smooths outliers that appear in single dimensions
   - Exploits RoPE's pairwise rotation structure

3. **Lloyd-Max scalar quantization** (3-bit):
   - Pre-computed optimal quantization levels for Beta distribution
   - 8 reconstruction levels per coordinate
   - No per-block calibration constants (data-oblivious)

**Memory budget** (Qwen2.5-7B, 4096 context):
- Uncompressed KV cache: ~4 GB (fp16, 32 layers × 32 heads × 128 dim × 4096 tokens × 2)
- PolarQuant 3-bit: ~750 MB (~5.3x reduction)
- Total model + KV: ~5 GB (fits in M2 16 GB with room for engine)

**Integration with Candle**:
- KV cache in Candle is an explicit Rust struct (`Cache` / `RotatingCache`),
  not opaque C state — we have full control over read/write paths
- Implement PolarQuant as a wrapper around Candle's `Cache` that quantizes on
  write and dequantizes on read
- Alternative: implement as `CustomOp` (Candle trait for custom operations)
- SIMD-accelerated: AVX2 + NEON for FWHT and quantization

**SIMD for FWHT**:
- Butterfly operations map naturally to SIMD (parallel adds/subs)
- 10,000-dim FWHT: ~13 passes (log2(16384) for next power of 2)
- Each pass: ~8K additions/subtractions → ~500 SIMD ops (AVX2 16-wide)

### 26d — Neural→VSA Bridge (Hyperdimensional Encoder) (~800 lines)

**Priority**: Medium (the core innovation)
**Depends on**: 26b (Candle backend for hidden state access)

Map LLM hidden states to the 10K-bit HyperVec space:

**New module**: `src/vsa/neural_bridge.rs`

**Architecture**:
```rust
pub struct NeuralVsaBridge {
    encoder: LinearProjection,    // hidden_dim → 10_000 (f32)
    threshold: f32,               // Binarization threshold (default 0.0)
    decoder: Option<LinearProjection>,  // 10_000 → hidden_dim (for Phase 27)
    training_buffer: Vec<(Tensor, HyperVec)>,  // Alignment pairs
}

pub struct LinearProjection {
    weight: Tensor,    // [out_dim, in_dim]
    bias: Tensor,      // [out_dim]
}
```

**Encoding pipeline**:
1. Extract hidden state from LLM layer L (configurable, default: last layer)
   - For Qwen2.5-1.5B: hidden_dim = 1,536 (f32 tensor, even from quantized GGUF)
   - For Qwen2.5-7B: hidden_dim = 3,584
   - Access: `model.base_model.forward()` returns `Tensor [batch, seq_len, hidden_dim]`
   - Take last token position: `hidden.i((.., seq_len - 1, ..))?` → `[1, hidden_dim]`
2. Project: `z = W·h + b` where W is [10000, hidden_dim]
   - Candle `candle_nn::Linear` handles this natively
   - ~15.4M f32 params for 1.5B model (1536 × 10000 + 10000)
   - ~35.8M f32 params for 7B model (3584 × 10000 + 10000)
   - Store as safetensors (~60 MB / ~140 MB)
3. Binarize: `v[i] = 1 if z[i] > threshold else 0`
   - Differentiable approximation during training: `sigmoid(z[i] * temperature)`
   - Hard threshold during inference (no gradient needed)
4. Result: 10K-bit HyperVec compatible with all existing VSA operations

**Training the encoder** (self-supervised, CPU-feasible):
- **Alignment objective**: Symbols with known VSA representations (from grounding)
  serve as training targets
- For each known (text, SymbolId) pair:
  1. Run text through LLM, extract hidden state `h`
  2. Encode via bridge: `v_pred = encode(h)`
  3. Look up ground-truth: `v_true = item_memory.get(symbol_id)`
  4. Loss: `hamming_distance(v_pred, v_true)` (differentiable approximation via sigmoid)
- Train via SGD on CPU (Candle), ~100K parameters, converges in minutes
- Store trained weights in `data/models/neural_bridge.safetensors`

**What this enables**:
- **Concept identification**: Run arbitrary text through LLM → hidden state → HyperVec
  → nearest-neighbor search in ItemMemory → "the LLM is thinking about X"
- **Interpretability**: Which KG concepts activate when processing a sentence?
- **Quality gate**: If encoded vector is far from all known concepts, the LLM
  is likely hallucinating or confused
- **Provenance**: New `DerivationKind::NeuralProbe { layer, similarity, source_text }`

**Integration with existing grounding**:
- Neural-encoded vectors can be blended with graph-grounded vectors
- `encode_with_neighborhood()` already supports blending — extend to accept
  neural vectors as an additional input
- Iterative grounding rounds propagate neural signal through the graph

### 26e — Inference Pipeline Integration (~500 lines)

**Priority**: Medium
**Depends on**: 26b, 26d

Wire the neural bridge into the existing NLU and inference pipelines:

**Enhanced NLU flow**:
```
Input text
    │
    ├──▶ Tier 1-4 (existing NLU) → AbsTree
    │
    └──▶ Candle backend (with hidden states)
              │
              ├──▶ Token output → validate against Tier 1-4 parse
              │
              └──▶ Hidden states → Neural bridge → HyperVec
                        │
                        ├──▶ Concept identification (ItemMemory search)
                        ├──▶ Confidence signal (distance to nearest concept)
                        └──▶ Grounding enrichment (blend with graph vectors)
```

**Enhanced inference flow**:
- During spreading activation, neural-probed concepts can serve as additional
  seed activations (weighted by encoder confidence)
- Superposition reasoning can incorporate neural hypotheses alongside symbolic ones
- The OODA Orient phase can use neural concept vectors to score tool relevance

**New NLU tier** (optional, between Tier 3 and Tier 4):
- "Tier 3.5: Neural concept extraction"
- If the LLM produced text but the parse is ambiguous, use the hidden state
  to identify which KG concepts were most active during generation
- This helps disambiguate between multiple valid parses

### 26f — Benchmarking and Evaluation (~300 lines)

**Priority**: Low (but needed before declaring phase complete)
**Depends on**: All above

**Benchmarks**:
1. **Candle vs llama-cpp-2**: tokens/sec, memory, first-token latency
   - On M2 (Metal) and x86_64 (CPU)
   - With and without PolarQuant
2. **NEON vs generic**: VSA operation throughput on M2
   - XOR bind, bundle, Hamming similarity, permute
3. **Neural bridge quality**: How well do encoded vectors match grounded vectors?
   - Measure: mean reciprocal rank of correct symbol in nearest-neighbor search
   - Baseline: random projection (no training)
4. **PolarQuant quality**: Perplexity degradation on standard benchmarks
   - Compare: fp16 KV vs 3-bit PolarQuant KV
   - Target: <0.5% perplexity increase

**Evaluation dataset**:
- Use existing bootstrap domains (the engine's own awakening conversations)
- For each concept discussed: does the neural bridge correctly identify the
  corresponding KG symbol?
- Threshold: >80% top-5 recall for known concepts

## Estimated Effort

| Sub-phase | Lines | Complexity | Dependencies |
|---|---|---|---|
| 26a — NEON SIMD | ~400 | Low | None |
| 26b — Candle backend | ~1,200 | High | None |
| 26c — PolarQuant | ~600 | Medium | 26b |
| 26d — Neural bridge | ~800 | High | 26b |
| 26e — Pipeline integration | ~500 | Medium | 26b, 26d |
| 26f — Benchmarks | ~300 | Low | All |
| **Total** | **~3,800** | | |

## New Dependencies

```toml
# Feature: candle-backend
candle-core = { version = "0.8", optional = true, features = ["metal"] }
candle-nn = { version = "0.8", optional = true }
candle-transformers = { version = "0.8", optional = true }
safetensors = { version = "0.5", optional = true }

# Note: tokenizers already a dependency (used by nlu-ml)
# Note: metal feature is macOS-only; use cfg to conditionally enable
```

Metal feature should be conditional on target OS:
```toml
[target.'cfg(target_os = "macos")'.dependencies]
candle-core = { version = "0.8", optional = true, features = ["metal"] }

[target.'cfg(not(target_os = "macos"))'.dependencies]
candle-core = { version = "0.8", optional = true }
```

## Feature Flags

```toml
[features]
candle-backend = ["candle-core", "candle-nn", "candle-transformers", "safetensors"]
polar-quant = ["candle-backend"]    # PolarQuant requires Candle tensors
nlu-candle = ["candle-backend"]     # Use Candle for NLU Tier 3 (replaces nlu-llm)
neural-bridge = ["candle-backend"]  # Neural→VSA encoder (Phase 26d)
```

## Future: Phase 27 — Live KG→LLM Training (Burn framework)

Phase 26 establishes the **read path** (LLM → VSA) using Candle. Phase 27 adds
the **write path** (VSA → LLM) and the **training loop** using **Burn**.

### Why Burn for training, not Candle

Candle is inference-first: it has basic training support (`candle-nn` with VarMap,
AdamW, autograd) which suffices for the small neural→VSA encoder in Phase 26d.
But LoRA fine-tuning of a 1.5B-7B model requires real training infrastructure:

| Capability | Candle | Burn |
|---|---|---|
| LoRA adapter layers | No built-in support | Composable module system |
| Gradient checkpointing | No | Yes (memory-efficient backprop) |
| Mixed precision training | Basic | Full (bf16/fp16 with loss scaling) |
| Learning rate schedulers | Manual | Built-in (cosine, linear warmup, etc.) |
| JIT kernel fusion | No | Yes (auto-fuses custom ops for CPU/GPU) |
| Checkpoint save/resume | Manual safetensors | Built-in training state serialization |

### Dual-framework architecture

```
Phase 26 (Candle only):                Phase 27 (Burn + Candle):

  GGUF model ──Candle──▶ inference       KG facts ──verbalize──▶ training pairs
  hidden states ──▶ neural bridge              │
  neural bridge training (small, SGD)          ▼
                                         Burn training loop
                                         (LoRA adapters, gradient checkpointing)
                                               │
                                               ▼
                                         Export LoRA weights (safetensors)
                                               │
                                               ▼
                                         Candle loads base model + LoRA adapters
                                         → inference with fine-tuned model
```

**Key**: Both frameworks use safetensors format. Burn trains and exports weights,
Candle loads and runs them. Clean separation — no framework mixing at runtime.

### Phase 27 sub-phases (preliminary)

1. **27a — Triple verbalization pipeline**: Discovered KG facts → natural language
   training pairs. Use the grammar framework's linearization (4 archetypes) to
   produce text from AbsTree representations of triples.

2. **27b — Burn LoRA training loop**: Implement LoRA adapter layers for Qwen2.5
   in Burn. Background daemon task — runs during idle cycles, does not block
   inference. Gradient checkpointing to fit training in M2's memory budget.

3. **27c — VSA→Neural decoder**: Reverse the bridge (HyperVec → LLM hidden state).
   Train via Burn alongside the LoRA adapters. Enables the symbolic reasoning core
   to "inject" solutions back into the LLM's generation.

4. **27d — Verification & conflict detection**: After each training batch, re-probe
   the model via the neural→VSA bridge (Phase 26d) to verify alignment hasn't
   degraded. E-graph contradiction check before accepting training data. Rollback
   LoRA weights if verification fails.

5. **27e — Background training scheduler**: Wire into the daemon's idle task system.
   Train during sleep/consolidation cycles. Rate-limit to avoid starving inference.

6. **27f — Training data collection pipeline**: Automatic logging of training
   pairs from the running system. This is the prerequisite for ALL fine-tuning.
   - **NLU pairs**: Every successful parse from any tier is logged as
     `(input_text, AbsTree_json)`. Tier 1 (rule parser) provides ~70% of
     examples. Tier 3 (Qwen) provides complex cases. Stored in redb.
   - **NLG pairs**: Every verbalization is logged as `(triples, output_text)`.
     Grammar-generated text serves as faithful baseline. User corrections
     (if any) become gold-standard examples.
   - **Elicitation pairs**: Phase 32 LLM-generated triples that pass validation
     become positive training examples. Rejected triples become negative examples.
   - **Threshold**: T5 NLU becomes viable at ~500-1000 NLU pairs. T5 NLG domain
     specialization benefits from ~200+ domain-specific NLG pairs.
   - **Cold-start note**: On a fresh workspace, the collection pipeline runs
     from day one but produces no training output until thresholds are met.
     All training is deferred until sufficient data accumulates. This is by
     design — the model adapts to each user's patterns organically.

### Concrete Burn integration patterns

These patterns were identified from the Gastown scheduling research and Burn
documentation. They define how the training loop integrates with the daemon.

**1. `AsyncProcessorTraining` for non-blocking training:**
```rust
// Burn's built-in async training processor
// Runs training steps on a background thread, reports metrics via callback
use burn_train::AsyncProcessorTraining;

let processor = AsyncProcessorTraining::new(training_config);
// Metrics flow back via channel — wire to TUI/dashboard
```

**2. `spawn_blocking` to isolate training from async I/O:**
```rust
// Training is CPU/GPU-bound — MUST NOT run on tokio async executor
// Use spawn_blocking to dedicate a thread pool
let handle = tokio::task::spawn_blocking(move || {
    let learner = LearnerBuilder::new(&artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .build(model, optim, lr_scheduler);

    learner.fit(dataloader_train, dataloader_valid);
});
// Daemon continues serving inference while training runs
```

**3. Burn `Learner` API with metric callbacks:**
- `LearnerBuilder` configures training with metrics, checkpointing, schedulers
- Custom `MetricEntry` can push loss/accuracy to crossbeam channel → TUI dashboard
- `CompactRecorder` saves checkpoints as safetensors-compatible format
- `EarlyStoppingStrategy` halts training if loss plateaus

**4. Memory-mapped training datasets (zero-copy from KG history):**
```rust
// We already use memmap2 for warm storage tier — reuse for training
// Burn's MappedDataset trait maps file offsets to training samples
// Single initial pass builds byte-offset index
// Subsequent epochs: O(1) random access, no RAM bloat
```

**5. On-the-fly synthetic data from VSA state:**
```rust
// Custom Burn Batcher that generates contrastive pairs from KG
// Positive: verified triples verbalized via grammar framework
// Negative: corrupted triples (swap subject/object, random predicate)
// Parallelized via rayon — data prep scales with CPU cores
struct VsaContrastiveBatcher {
    engine: Arc<Engine>,
    grammar: Arc<ConcreteGrammar>,
}
impl Batcher<KgSample, TrainingBatch> for VsaContrastiveBatcher {
    fn batch(&self, items: Vec<KgSample>) -> TrainingBatch {
        // Query KG for verified triples
        // Verbalize via grammar linearization
        // Generate negative samples by corruption
        // Tokenize and pad to batch
    }
}
```

**6. Hot-swap model weights after training:**
```rust
// After Burn training completes:
// 1. Export LoRA adapter weights as safetensors
// 2. Signal Candle backend to reload
// 3. Candle merges base GGUF + LoRA adapters
// 4. n akh router receives signal: increase exploration for updated model
// No daemon restart required
```

### Why deferred

- Requires the neural→VSA bridge (Phase 26d) trained and validated first
- Burn's model zoo doesn't include Qwen2.5 yet — may need to implement or adapt
- Training data pipeline (verbalization + quality gate) needs design
- Memory budget on M2 requires PolarQuant (Phase 26c) to be working first
- The bootstrapping loop is more valuable with more KG content (from continued use)

**Phase 26 unblocks Phase 27 specifically because:**
- Candle gives us full control over model weights (Rust structs, not C opaque pointers)
- KV cache is an explicit Rust struct — PolarQuant frees RAM for training tensors
- The encoder bridge provides a verification mechanism (probe before/after training)
- PolarQuant frees enough RAM to hold LoRA adapter tensors alongside the base model

Full plan: [Phase 27 plan](2026-03-26-phase27-live-training.md) (to be created when Phase 26 nears completion)

## Relationship to Phase 24 (Sensory Grounding)

Phase 24 plans a "VSA Grounding Bridge" for mapping perceptual features (images,
audio) to HyperVecs. The neural bridge from Phase 26d uses the **same architecture**
— a linear projection + binarization. The Phase 24 bridge can reuse the
`NeuralVsaBridge` struct with different weight matrices:

- Phase 26d: LLM hidden dim (e.g., 1536 for Qwen2.5-1.5B) → 10K-bit
- Phase 24a: CLIP/SigLIP feature dim (e.g., 512 for MobileCLIP) → 10K-bit
- Phase 24b: Audio feature dim (e.g., 128 mel bands) → 10K-bit

This shared infrastructure means Phase 24 becomes simpler once Phase 26d is done.

## Relationship to Known Limitations

### Binary VSA analogy degeneracy
The neural bridge provides an alternative path: instead of relying on
bind/unbind arithmetic for analogy (which is degenerate in 10K-bit Hamming space),
we can:
1. Probe the LLM with "A is to B as C is to ?" prompts
2. Extract hidden states for A, B, C, and the predicted D
3. Encode all four into HyperVec space via the neural bridge
4. The resulting vectors carry the LLM's learned analogy structure,
   which is richer than pure VSA arithmetic

This doesn't fix the VSA analogy math, but provides a workaround via the
neural path when analogical reasoning is needed.
