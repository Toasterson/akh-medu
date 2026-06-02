# ADR 037 — LLM-VSA Deep Integration Architecture

> Date: 2026-03-26
> Status: Proposed
> Phase: 26 (LLM-VSA Deep Integration)
> Depends on: Phase 14m (NLU complete), Phase 24 (Sensory Grounding)
> Supersedes: None (extends ADR 022 NLU Architecture)

## Context

A deep-research session with Gemini produced two documents analyzing how TurboQuant
(Google Research, ICLR 2026) and Rust-native deep learning frameworks (Burn, Candle)
could benefit the akh-medu architecture. The analysis described a future where LLM
hidden states are bidirectionally mapped to the VSA space, enabling the symbolic
reasoning core to directly manipulate neural representations.

This ADR evaluates what is realistic, what is hallucinated, and what architectural
direction to pursue.

## Gemini Claims vs. Reality

### Hallucinated or Incorrect

1. **Burn/Candle already in use** — False. We use `llama-cpp-2` (C++ bindings to
   llama.cpp) for Qwen2.5-1.5B GGUF inference and `ort` for ONNX NER.
2. **`vsa-optim-rs` crate** — Does not exist. Gradient compression via VSA is
   research-stage with no Rust implementation.
3. **`turbo-quant` crate at lib.rs** — Unverified. Even if a thin wrapper exists,
   no production Rust implementation of full TurboQuant (PolarQuant + QJL) is known.
4. **Neural VSA encoder already maps LLM states** — False. Our symbol encoding is
   deterministic (seeded RNG from SymbolId), not neural.
5. **Agent swarms on workstation** — We have a single synchronous OODA agent.

### Already Implemented (Gemini Didn't Know)

- Full VSA stack (10K-bit binary, AVX2 SIMD, HNSW ANN, 5 core ops)
- Spreading activation with superposition reasoning (hypothesis forking/merging)
- E-graph reasoning (AkhLang, rewrite rules)
- Background learning (6 idle task types, sleep/dream consolidation)
- Event calculus, causal world model, counterfactuals (Pearl Level 3)
- MCTS planning with TD value function
- Dempster-Shafer evidence theory + source reliability
- 60 derivation kinds in provenance ledger
- Code2vec-style encoding without neural networks
- Iterative graph-driven semantic grounding

### Genuinely Valuable New Capabilities

1. **Candle migration** — Pure Rust model loading, hidden state access, WASM, Metal
2. **Neural→VSA bridge** — Map LLM hidden states to HyperVec for interpretability
3. **KV cache compression** — Enable larger local models (7B-8B) on M2
4. **Live KG→LLM training** — Close the bootstrap loop with incremental fine-tuning
5. **ARM NEON SIMD** — 2-4x speedup on M2 (currently generic scalar on ARM)

## Decision

### 1. Migrate from llama-cpp-2 to Candle for T5 model stack

**Note (revised 2026-03-27)**: The original plan was to migrate Qwen inference
from llama-cpp-2 to Candle. The architecture has since evolved further: Qwen
is fully replaced by three purpose-built T5 models (NLU, NLG, Knowledge) via
Candle's `quantized_t5` module. See ADR 043 (verbalization), ADR 045 (training
infrastructure). The Candle migration remains the prerequisite — the model
running on it changed from Qwen to T5.

**Why migrate from llama-cpp-2**: llama-cpp-2 wraps C++ (llama.cpp) which:
- Cannot expose hidden states — only `llama_get_embeddings()` for final embeddings;
  per-layer access requires undocumented C-level eval callbacks not exposed by Rust bindings
- Requires C++ toolchain (CMake) to build
- Has no WASM path for browser deployment
- Crashes via GGML assertions that Rust cannot catch (`catch_unwind` doesn't help with C aborts)

**Candle hidden state access — verified from source code:**

Candle's Qwen2 implementation (`candle-transformers/src/models/qwen2.rs`) has a
two-tier architecture that is the key enabler:

- `Model::forward()` returns **hidden states** (post-RMSNorm, pre-lm_head) as a `Tensor`
- `ModelForCausalLM::forward()` wraps `Model`, applies `lm_head`, returns **logits**
- The `base_model` field on `ModelForCausalLM` is **`pub`** — direct access

The forward pass is a plain Rust loop over layers:
```rust
let mut xs = self.embed_tokens.forward(input_ids)?;
for layer in self.layers.iter_mut() {
    xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
}
xs.apply(&self.norm)
```

No opaque graph, no compiled kernel. Each layer is a separate method call.
Per-layer hidden state capture requires only cloning `xs` inside the loop.

**Quantized path also works**: `quantized_qwen2.rs` uses the same loop-over-layers
pattern with `QMatMul` for weights. Critically, **hidden states between layers are
full-precision f32 tensors** even when weights are Q4_K_M quantized — only weight
storage is compressed, activations flow normally. The neural→VSA bridge receives
clean signals.

**No hook mechanism exists** (unlike PyTorch's `register_forward_hook()`), but none
is needed — the source is plain Rust with explicit layer iteration. We own the code.

**Additional Candle capabilities:**
- Pure Rust, no C++ dependency
- First-class Metal backend: `Device::Metal` is a peer to `Device::Cuda`, not an afterthought
- `Device::metal_if_available(ordinal)` for graceful CPU fallback
- BF16 supported on Metal
- WASM compilation for browser-based inference
- GGUF quantized model loading (Q2-Q8, reads same files as llama.cpp)
- Active HuggingFace maintenance

**Risk**: Candle's GGUF inference is ~70-90% of llama.cpp throughput on Metal,
due to llama.cpp's years of hand-tuned Metal compute shaders for quantized matmul.
For a 1.5B boundary translator this is acceptable (~30-55 tok/s vs ~40-80 tok/s).
Mitigation: keep llama-cpp-2 as fallback feature flag; use Candle when hidden
states are needed, llama-cpp-2 for raw speed if needed.

**Constrained decoding gap**: Candle has no built-in GBNF grammar support. Must
implement a `LogitsProcessor` for AbsTree JSON structure enforcement, or use our
existing brace-depth tracking approach (already working as fallback in current code).

### 2. Implement Neural→VSA Bridge (small encoder via Candle)

**Why**: The "Hyperdimensional Probe" concept (mapping LLM residual stream → HyperVec)
is architecturally sound and aligns with our existing grounding infrastructure.

**Architecture**:
```
LLM forward pass (Candle)
    │
    ├── Token output (text) ──▶ existing NLU pipeline
    │
    └── Hidden states (f32 tensors) ──▶ Neural VSA Encoder
                                            │
                                            ▼
                                       HyperVec (10K-bit)
                                            │
                                            ▼
                                       ItemMemory (similarity search)
                                            │
                                            ▼
                                       Grounded symbolic concepts
```

The encoder is a small linear projection (LLM hidden dim → 10K) + binarization
threshold. ~15M parameters, trainable on CPU via Candle's `candle-nn` (which
provides autograd, `VarMap`, and AdamW — sufficient for a model this small).

### 3. Implement PolarQuant for KV cache compression (not full TurboQuant)

**Why**: Full TurboQuant (PolarQuant + QJL two-stage) is complex and the QJL
residual stage adds implementation burden for marginal gain over PolarQuant alone.
PolarQuant alone achieves ~4x compression on KV cache with near-zero quality loss.

**Implementation**: ~500 lines of Rust:
- Random orthogonal rotation via Fast Walsh-Hadamard Transform (O(d log d))
- Polar coordinate transform for RoPE dimension pairs
- 3-bit Lloyd-Max scalar quantization per coordinate
- SIMD-accelerated (AVX2 + NEON)

This enables running Qwen2.5-7B (or similar) on M2 with 16 GB RAM.

### 4. Dual-framework strategy: Candle for inference, Burn for training (Phase 27)

The Rust ML ecosystem has two complementary frameworks with distinct strengths:

| | **Candle** (HuggingFace) | **Burn** (burn.dev) |
|---|---|---|
| **Focus** | Inference, model loading, deployment | Training, optimization, portability |
| **Strengths** | GGUF/safetensors loading, pre-built model zoo (Qwen, Llama, Mistral), Metal inference, WASM | JIT kernel fusion, automatic differentiation, backend-agnostic, gradient checkpointing |
| **Training** | Basic (`candle-nn`: VarMap, AdamW, autograd) | Full (custom training loops, mixed precision, distributed) |
| **Sweet spot** | Running pre-trained models + training small auxiliary models | Training/fine-tuning models from scratch or with LoRA adapters |

**Decision**: Use both, with clear separation of concerns:

- **Phase 26 (Candle only)**: LLM inference, hidden state extraction, and training
  the small neural→VSA encoder (~15M params). Candle's basic training support is
  sufficient for a linear projection — no JIT compiler or gradient checkpointing
  needed.

- **Phase 27 (Burn for training)**: Live KG→LLM fine-tuning via LoRA adapters.
  This requires real training infrastructure — gradient accumulation, learning rate
  scheduling, mixed precision, checkpoint management. Burn is purpose-built for this.
  The trained LoRA weights would be exported as safetensors and loaded by Candle
  for inference.

This avoids premature complexity: Phase 26 ships with one dependency (Candle),
Phase 27 adds Burn only when the training loop is actually needed.

**Why defer Phase 27**: Training requires LoRA infrastructure, a training data
pipeline (triple verbalization + quality gate), and an evaluation harness —
substantial work. The bootstrap loop is more valuable once we have:
- The neural→VSA bridge (Phase 26d, to verify alignment before/after training)
- A larger model (7B, enabled by PolarQuant in Phase 26c)
- More KG content (from continued use)

The fact discovery pipeline already exists (spreading activation + background
learning). What's missing is the verbalization→training→evaluation cycle.

### 5. Add ARM NEON SIMD as part of this phase

**Why**: The M2 is the primary target hardware. Currently ARM falls back to generic
scalar. NEON intrinsics for XOR bind, bundle accumulate, Hamming popcount, and
cosine similarity would give 2-4x speedup on all VSA operations.

## Rejected Alternatives

### Burn for Phase 26
Burn is the right tool for serious training workloads (Phase 27: LoRA fine-tuning),
but adding it in Phase 26 would be premature. The only training in Phase 26 is
the small neural→VSA encoder (~15M params), which Candle's basic `candle-nn`
training support handles fine. Burn arrives in Phase 27 when we actually need
JIT-compiled training loops, gradient checkpointing, and mixed precision.

### Full TurboQuant (two-stage)
Over-engineered for our needs. PolarQuant stage alone gives ~4x compression.
The QJL residual stage adds unbiased inner product estimation which matters for
attention-heavy workloads, but our primary bottleneck is memory, not attention
accuracy.

### VSA gradient compression (vsa-optim-rs)
The crate doesn't exist. The concept (using bind/bundle/unbind to compress gradients)
is interesting but research-stage. If we need gradient compression, standard
techniques (gradient accumulation, mixed precision) are proven and available in
Candle.

### Agent swarms
Single OODA agent serves the current use case. Multi-agent coordination adds
complexity (context isolation, message passing, consensus) without clear benefit
until we have a deployment scenario requiring it.

## Consequences

- New feature flag: `candle-backend` (parallel to existing `nlu-llm`)
- Transition period: both `llama-cpp-2` and `candle` backends available
- New module: `src/nlu/candle_backend.rs` for Candle-based LLM inference
- New module: `src/vsa/neural_bridge.rs` for hidden state → HyperVec mapping
- New module: `src/vsa/polar_quant.rs` for KV cache compression
- New SIMD kernels: `src/simd/neon.rs` for ARM acceleration
- Phase 24 (Sensory Grounding) benefits from the same Candle infrastructure
- **Dual-framework trajectory**: Candle (Phase 26, inference + small training) →
  Burn added in Phase 27 (LoRA fine-tuning, live KG→LLM training loop)
- Burn's trained LoRA weights export as safetensors → Candle loads them for inference
- Both frameworks share the same tensor format (safetensors), enabling clean handoff

## Verified Technical Details

### Candle API surface (verified from source, 2026-03-26)

**Hidden state access patterns**:

1. **Final hidden states** (post all layers + norm, pre lm_head):
   ```rust
   // Use Model directly instead of ModelForCausalLM
   let hidden = model.base_model.forward(&input_ids, seqlen_offset, mask)?;
   // hidden: Tensor [batch, seq_len, hidden_dim]
   ```

2. **Per-layer hidden states** (requires wrapping the forward loop):
   ```rust
   let mut xs = self.embed_tokens.forward(input_ids)?;
   let mut layer_outputs = Vec::with_capacity(self.layers.len());
   for layer in self.layers.iter_mut() {
       xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
       layer_outputs.push(xs.clone());  // capture per-layer
   }
   ```

3. **Quantized models**: Same loop structure in `quantized_qwen2.rs`.
   Weights are `QMatMul` (quantized), but intermediate `xs` is f32 `Tensor`.

**KV cache**: Explicit Rust structs (`Cache` / `RotatingCache`), not opaque C state.
Direct manipulation possible for PolarQuant integration.

**Metal backend**: `Device::new_metal(0)` or `Device::metal_if_available(0)`.
Feature-gated via `--features metal`. BF16 supported.

**GGUF loading**: `candle::quantized::gguf_file` module. `ModelWeights::from_gguf()`
constructor. Supports Q2-Q8 quantization levels.

### llama-cpp-2 limitations (verified, 2026-03-26)

- `llama_get_embeddings()` returns only final-layer embeddings
- Per-layer hidden states require `llama_set_eval_callback()` (C API),
  which is not exposed in the `llama-cpp-2` Rust crate
- KV cache is managed internally by C library — no Rust-side access
- GGML assertion failures call `abort()`, bypassing Rust's panic/catch_unwind

## References

- TurboQuant paper: https://openreview.net/forum?id=tO3ASKZlok
- PolarQuant (NeurIPS 2025): https://neurips.cc/virtual/2025/poster/118745
- Hyperdimensional Probe: https://arxiv.org/html/2509.25045v1
- Candle framework: https://github.com/huggingface/candle
- Candle Qwen2 source: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/qwen2.rs
- Candle quantized Qwen2: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/quantized_qwen2.rs
- llama.cpp hidden states discussion: https://github.com/ggml-org/llama.cpp/discussions/11274
- ADR 022: NLU Architecture (current LLM integration)
