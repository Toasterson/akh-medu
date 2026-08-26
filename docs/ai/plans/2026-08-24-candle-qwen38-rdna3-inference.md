# Plan: Candle Qwen3.8 inference on RDNA3

- **Date:** 2026-08-24
- **Status:** In progress — slices 0–1 and Slice 2a complete; Slice 2b next
- **Scope:** inference only; target-model training remains out of scope
- **Production oracle:** Ptah's exact custom Qwen3.8-27B Q4_K_M + vision projector
- **Related:** [ADR 037](../decisions/037-llm-vsa-deep-integration.md),
  [ADR 065](../decisions/065-context-assembly-runtime.md),
  [ADR 073](../decisions/073-full-candle-hip-device-backend.md),
  [ADR 074](../decisions/074-candle-static-shape-inference-and-training.md),
  [ADR 098](../decisions/098-candle-qwen35-rdna3-inference-stack.md)
- **Kernel reference:** [tinygrad PR #17512](https://github.com/tinygrad/tinygrad/pull/17512),
  merged 2026-08-24 under the MIT license
- **Concurrency reference:** [bounded batched decode and MTP research](../research/2026-08-24-batched-mtp-concurrency-reference.md)
- **Anima:** existing “Lane: activate ptah on hephaistos…” work item; ID unavailable
  because no Anima project tool is exposed in this session
- **Seshat:** ingested 2026-08-24 after Slice 1 completion under the
  `akh-medu-plan-candle-qwen38-rdna3-slice1` slug; the Slice 2a update is pending
  because no Anima/Seshat integration is available in this session

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

1. [x] Q8 activation quantization plus packed Q4_K `dp4a` decode GEMV.
2. [ ] WMMA Q4_K prefill/short-batch matrix multiplication.
3. [ ] Fused gated-delta prefill scan with in-place recurrent state.
4. [ ] RDNA3 WMMA flash attention; retain the current split-K kernel when it wins.

The existing f32/Q4_K kernels remain the reference adapter until each replacement passes
operation parity and full-forward hidden/logit parity. Benchmark the exact custom Q4_K_M,
not tinygrad's headline IQ4_XS model.

- Gates: operation parity; hidden cosine >= 0.999; coherent exact-model output; measured
  decode and 32-token/long-prefill throughput; no regression accepted without an explained
  quality or memory win.

Slice 2a evidence (hephaistos, RX 7900 XTX, ROCm 6.3.4, exact installed custom
Qwen3.8-27B Q4_K_M, 2026-08-26): the packed kernel matches an independent CPU
Q8-group oracle at `3.51e-7` relative error. Twelve counterbalanced warm f32/Q8 runs
from the same post-`Hello` checkpoint compare three forced continuation steps, not
only the first token: minimum hidden-state cosine is `0.999960874`, minimum full-logit
cosine is `0.999991194`, and every greedy token (`[0, 369, 271]`) is unchanged.
Median three-step time improves from `0.171 s` to `0.147 s` (`1.162x`). Q8/Q4_K is
therefore the default decode
adapter; `AKH_HIP_Q8_GEMV=0` restores the f32 reference, and selected-kernel errors
fail closed rather than silently falling back.

Ptah reviewed the adapter in a fresh ACP turn and found four concrete hygiene risks:
an optional sidecar ABI symbol must not disable the f32 escape hatch; scratch lifetime
depends on the backend's single ordered stream; process-global route selectors require
serialized RAII restoration in tests; and the path performs two ordered launches, not
one. All four findings are reflected in the implementation and tests. The same review
discipline is stored in Ptah's KG-backed core memory and was recalled verbatim in a new
session. The label-index migration restoring Ptah's long-symbol snapshots has also been
applied and observed through subsequent persisted snapshot ticks.

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

### Follow-up 3b — Bounded, session-safe batched decode

Once each ACP session owns an `InferenceSession`, add admission-controlled C1/C2/C4/C8
decode rounds around the shared resident target model. Compact ready rows only at round
boundaries. Each row retains independent KV pages, gated-delta/conv state, sampling RNG,
checkpoint state, image positions, and VSA hidden taps. Start target-only; MTP-3 remains a
later optional composition, not a prerequisite for batching.

- Gates: greedy token/logit parity and two-session cancellation/rewind isolation at every
  concurrency; saturation and mixed-agent sweeps report aggregate and per-stream throughput,
  TTFT, occupancy, live KV/recurrent bytes, VRAM high-water mark, and HBM bytes per committed
  token; one batched setting must reach 1.5x C1 aggregate; the selected mixed-workload setting
  must improve makespan by 25% while keeping p95 TTFT below 2x C1; 100K total live physical
  tokens must queue or reject cleanly rather than corrupt another session.

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
- [x] Slice 1: reference text forward (exact schema; GGUF tiled-head contract; hermetic
  non-zero chunk/state replay; exact-model multi-token CPU prefill and HIP stepped-decode
  greedy + log-probability parity with llama.cpp).
- [ ] Slice 2: RDNA3 target kernels (2a Q8/Q4_K decode complete; 2b WMMA prefill next).
- [ ] Slice 3: inference session + VSA context signals.
- [ ] Follow-up 3b: bounded C1/C2/C4/C8 session-safe decode.
- [ ] Slice 4: vision.
- [ ] Slice 5: TurboQuant.
- [ ] Slice 6: target-only canary.
- [ ] Slice 7: optional DFlash2.
