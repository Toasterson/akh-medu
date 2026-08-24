# Plan: Candle Qwen3.8 inference on RDNA3

- **Date:** 2026-08-24
- **Status:** In progress — slice 0 complete, slice 1 next
- **Scope:** inference only; target-model training remains out of scope
- **Production oracle:** Ptah's exact custom Qwen3.8-27B Q4_K_M + vision projector
- **Related:** [ADR 037](../decisions/037-llm-vsa-deep-integration.md),
  [ADR 065](../decisions/065-context-assembly-runtime.md),
  [ADR 073](../decisions/073-full-candle-hip-device-backend.md),
  [ADR 074](../decisions/074-candle-static-shape-inference-and-training.md),
  [ADR 098](../decisions/098-candle-qwen35-rdna3-inference-stack.md)
- **Kernel reference:** [tinygrad PR #17512](https://github.com/tinygrad/tinygrad/pull/17512),
  merged 2026-08-24 under the MIT license

## Outcome

Run Ptah's exact custom Qwen3.8 model in-process through Candle on the RX 7900 XTX,
with one owned inference session that supports:

- target-only RDNA3 decode fast enough to make speculative decoding optional;
- selected hidden-state taps for Virtual Context and VSA retrieval;
- the exact vision projector and structured image turns;
- a 100K-token ceiling with TurboQuant-compressed full-attention KV;
- explicit recurrent-state and KV checkpoints for safe session isolation;
- an optional DFlash2 layer only if target-only measurement justifies it.

The existing llama.cpp runtime remains the production oracle until the cutover
gates pass. This plan does not stop or reconfigure the running services.

## Non-negotiable invariants

1. **Exact model identity.** Never substitute another Qwen model or distill. Tests and
   canaries use the locally installed custom Qwen3.8 target and matching projector.
2. **Fail closed on architecture.** `qwen35`/Qwen3.8 must never fall through to the
   Qwen2 loader. Unsupported partial states return a diagnostic before weight loading.
3. **One inference owner per session.** KV, gated-delta recurrent state, DFlash state,
   hidden taps, and vision embeddings cannot leak across ACP sessions.
4. **Hidden-state fidelity.** BF16 CPU/GPU cosine must remain at least 0.999 at every
   optimized stage. VSA signals are derived from the same forward, not an extra pass.
5. **Small physical context.** Virtual Context normally materializes 8K–32K tokens;
   100K is a ceiling, not a target. Semantic spans are rematerialized as tokens rather
   than spliced into position-dependent KV state.
6. **Measure before composing.** Target-only Candle is completed and benchmarked before
   deciding whether DFlash2 earns its model, state, rollback, and maintenance cost.

## Deep module and seam

The external seam is an inference session, not individual GPU kernels:

```text
ContextPlan
    -> InferenceSession.prefill(..., HiddenTapPlan)
    -> ContextSignals
    -> virtual retrieval
    -> InferenceSession.extend_and_decode(...)
    -> ModelTurn + SessionUsage
```

`Qwen35Model` hides the reference and RDNA3 implementations. Its internal adapters are:

- **reference adapter:** ordinary Candle tensor operations, used as the correctness oracle;
- **RDNA3 adapter:** private HIP kernels chosen on gfx1100 after parity gates pass.

Callers do not select quantized GEMV, flash attention, or gated-delta kernels directly.
The context runtime receives compact `ContextSignals`, never Candle tensors or raw cache
storage.

## Sequence

### Slice 0 — Reserve the model family and fail closed

- Recognize GGUF architecture tags `qwen35`, `qwen3.5`, and `qwen3_5` as one explicit
  `Qwen35` family.
- Give Qwen35 its native ChatML stop/prompt behavior.
- Until the loader lands, return a precise, actionable unsupported-architecture
  diagnostic rather than interpreting Qwen3.8 tensors as Qwen2.
- Gate: focused architecture-classification and stop-token tests pass with and without
  the Candle feature.

### Slice 1 — Reference Qwen3.8 text forward

- Add a vendored quantized Qwen35 loader for the exact GGUF metadata/tensor schema.
- Implement embedding, RMSNorm, MLP, full attention, multimodal RoPE text positions,
  and the 3:1 gated-delta/full-attention layer schedule.
- Represent growing attention KV and fixed-size recurrent/conv state explicitly.
- Expose `prefill`, `decode_step`, `checkpoint`, and `rewind` only inside the model module.
- Gates: tensor-schema fixture; CPU logits against llama.cpp on fixed prompts; recurrent
  state after chunked prefill equals token-at-a-time prefill.

### Slice 2 — RDNA3 target kernels

Port the algorithms from tinygrad PR #17512 into the existing Candle HIP kernel library:

1. Q8 activation quantization plus packed Q4_K `dp4a` decode GEMV.
2. WMMA Q4_K prefill/short-batch matrix multiplication.
3. Fused gated-delta prefill scan with in-place recurrent state.
4. RDNA3 WMMA flash attention; retain the current split-K kernel when it wins.

The existing f32/Q4_K kernels remain the reference adapter until each replacement passes
operation parity and full-forward hidden/logit parity. Benchmark the exact custom Q4_K_M,
not tinygrad's headline IQ4_XS model.

- Gates: operation parity; hidden cosine >= 0.999; coherent exact-model output; measured
  decode and 32-token/long-prefill throughput; no regression accepted without an explained
  quality or memory win.

### Slice 3 — Owned inference session and VSA hidden taps

- Replace prompt-string-at-position-zero generation with an appendable per-ACP session.
- Capture target layers 5, 19, 33, 47, and 61 during the existing forward.
- Pool selected query/span states on GPU and transfer only a 5120-vector or finished
  10K-bit HyperVec to the host.
- Add a structured/low-rank context projector; do not allocate the current dense
  `5120 x 10000` bridge (~205 MB) per session.
- Wire the two feedback loops:
  - query signal -> VSA retrieval -> materialized memory before decode;
  - completed span signal -> archival index during context commit.
- Make `ContextManager::commit` reachable through the live turn path.
- Gates: two-session isolation, no duplicate target forward for signals, retrieval A/B,
  real non-zero virtual-context statistics.

### Slice 4 — Exact vision path

- Accept structured text/image content rather than flattening chat content to strings.
- Load the installed qwen3vl merger/projector, implement image preprocessing and 3D
  multimodal positions, and inject the resulting 5120-dimensional image tokens.
- Emit pooled pre-language and contextualized visual signals for VSA indexing.
- Gates: existing smoke images, multi-image ordering, text-only byte stability, visual
  memory retrieval, and exact-projector fingerprint enforcement.

### Slice 5 — TurboQuant at the physical-cache layer

- Add fixed compressed pages for the 16 growing full-attention layers; recurrent state
  remains separately typed and unquantized initially.
- Implement PolarQuant plus QJL residual storage and compressed-domain attention. Never
  decompress the complete cache for one attention operation.
- Keep BF16 and q4 cache adapters as A/B references.
- Gates at 100K: target KV memory, LongBench/RULER/needle/code parity, attention latency,
  graph-compatible fixed addresses, and rollback correctness.

### Slice 6 — Target-only canary

- Run exact-model text, tools, vision, 44K retrieval, and 100K pressure fixtures.
- Compare task success, total turn latency, prefill, decode, VRAM, physical tokens, and VSA
  recall against the production llama.cpp+DFlash2 oracle.
- Target-only Candle may replace production only after stability and quality gates pass;
  raw decode alone is not a sufficient cutover decision.

### Slice 7 — Optional DFlash2 composition

Reopen DFlash2 only when the target-only canary shows a material latency gap worth the
additional complexity. If reopened:

- reuse the existing 5/19/33/47/61 taps;
- use the WMMA short-batch path for target verification;
- checkpoint and rewind both full-attention KV and gated-delta recurrent state;
- require acceptance and end-to-end throughput to beat target-only Candle after draft
  overhead and VRAM are included.

## Baselines and reporting

Record every benchmark with model fingerprint, quant type, context length, prompt class,
ROCm version, GPU clocks, and whether vision/TurboQuant/DFlash are enabled. The first
apples-to-apples baseline is the exact custom Q4_K_M with no speculative decoding; the
tinygrad 46 tok/s IQ4_XS result is a feasibility reference, not Ptah's acceptance number.

## Current progress

- [x] Architecture and kernel audit completed.
- [x] Sequence and gates recorded locally.
- [x] Slice 0: explicit Qwen35 family recognition and fail-closed load.
- [ ] Slice 1: reference text forward.
- [ ] Slice 2: RDNA3 target kernels.
- [ ] Slice 3: inference session + VSA context signals.
- [ ] Slice 4: vision.
- [ ] Slice 5: TurboQuant.
- [ ] Slice 6: target-only canary.
- [ ] Slice 7: optional DFlash2.
