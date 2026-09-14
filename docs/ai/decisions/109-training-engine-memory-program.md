# ADR 109 — Training-engine memory program: after recompute attention, the remaining memory is FFN activations, the logits tensor and Adam state; each has a Burn-level fix

> Date: 2026-09-14
> Status: Accepted
> Phase: 51 P7 (plan: `docs/ai/plans/2026-09-14-phase51-p7-kaine-architecture-ladder-and-training-engine.md`)
> Anima: project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan `019f90cb-2e66-7f92-ba88-a4060be97c6e`; WIs P7.1 `01a0a1a2-aa69`, P7.2 `01a0a1a2-cf26`, P7.3 `01a0a1a2-ef2c`, P7.4 `01a0a1a3-157a`, P7.8 `01a0a1a3-d7c3`
> Related: ADR 106 (Burn on Metal lane), ADR 107 (recompute-backward attention; supersedes nothing there), ADR 108 (architecture ladder)
> Evidence: Burn 0.21 source audit (registry + `vendor/burn-autodiff`) and 14 papers read in full text on 2026-09-14

## Question

ADR 107 removed the quadratic attention term (8192×2 in 7.0 GB on the M2). The operator asked for a better engine that brings
training memory down further, so the ADAM ladder (33M → 350M → ≈1.1B) and Kaine can train on the Mac where possible and on one pod
GPU otherwise. Where does the remaining memory go, and what can Burn 0.21 do about it?

## Where the memory is now

ADR 107's measured residual at 8192 context: `small_8k` 2.7 GB non-attention activations + 0.4 GB weights/Adam; `350m_8k` ≈18 GB +
4.2 GB. Korthikanti et al. (2205.05198) give the per-layer activation formula `34·s·b·h + 5·a·s²·b` bytes in 16-bit; with the
quadratic part gone, the linear term decomposes as **MLP 19sbh** (GELU input 8, down-proj input 8, up-proj input 2, dropout 1),
attention linears 11sbh, LayerNorms 4sbh. We train in f32 (double those bytes) and have no dropout.

Burn 0.21 audit (file:line in the P7 plan):

- Checkpointing is **per primitive op**, not per block. `BalancedCheckpointing` recomputes only ops tagged `memory_bound`
  (elementwise, GELU, exp, tanh); `float_matmul`, reductions, gather/mask and every `ops/module.rs` op (linear, embedding, attention)
  are `compute_bound` and are always retained. So `grad_checkpointing` never frees a Linear input, and the FFN's d_ff-sized
  activations dominate. Block-level recompute requires a custom `Backward` op, exactly as `RecomputeAttention` did.
- The LM head: `masked_cross_entropy` reshapes to `[b·s, vocab]` and materialises `log_softmax`; ≈0.5 GB f32 per 8192 sequence at
  vocab 8000, linear in vocab and batch.
- Adam state is `Tensor<B, D>` in the backend dtype (8 bytes/param in f32). burn-optim ships Adam, AdamW, Adagrad, RMSProp, SGD,
  LBFGS, Adan and **Muon** (`muon.rs`, Keller-Jordan and Moonshot variants); no Adafactor, no 8-bit. `SimpleOptimizer` is a public
  two-method trait.
- `metal = ["wgpu"]` does **not** enable Burn's `fusion` feature; grug-train builds today have zero operator fusion. Even with fusion,
  only generic elementwise/matmul/reduce fusers exist; the CubeCL flash kernel is still bypassed under `Autodiff`.
- Mixed precision: wgpu/Metal exposes f16 but no bf16. The default `layer_norm` composition reduces mean/var in the backend float
  type with no cast, while Burn's own `RmsNorm` upcasts to f32 first. Our model uses `LayerNorm`. The P6.1b(c) gate failure
  (f16 + f32 master weights: loss 6.30 vs 3.76, 2× slower) was shown *not* to be loss-scale-related; unguarded f16 normalisation and
  softmax statistics are the identified mechanism, and the slowdown points at unfused casts.
- Alternatives: `candle` 0.8 has Metal bf16 but no checkpointing and no fused attention backward; `mlx-rs` is a community binding
  with closure-traced autodiff and no documented checkpointing or bf16 story. Neither gives anything for free.

## What the literature offers per bucket

| Bucket | Lever | Source | Number |
|---|---|---|---|
| FFN activations | block recompute / blockwise FFN | Korthikanti 2205.05198; BPT 2305.19370; Mini-Sequence 2407.15892 | selective recompute: 5× activation memory at < 10 % overhead; MsT 12–24× longer context on Llama-3-8B, no throughput loss |
| Logits | Cut Cross-Entropy | Apple 2411.09009 | Gemma-2 2B loss layer 24 GB → 1 MB, head 28 GB → 1 GB |
| Optimizer state | Muon (momentum only) | 2502.16982 | 4 B/param vs 8; ≈52 % of AdamW FLOPs at matched loss (needs weight decay + update scaling) |
| Optimizer state | Adam-mini / Adafactor / 8-bit Adam | 2406.16793 / 1804.04235 / 2110.02861 | 50 % / ≈0 second-moment / 2 B/param, parity from 39M–1.5B |
| Optimizer + grads | LOMO / AdaLomo fused backward | 2306.09782 / 2310.10195 | zero optimizer state, O(1) gradient memory; AdaLomo matches AdamW |
| Precision | f16 done right | Micikevicius 1710.03740; 2010.06192; Collage 2405.03637 | needs master weights **and** loss scaling **and** fp32 reductions; Collage fixes mantissa, not range |
| Activations (generic) | GACT | 2206.11357 | up to 8.1× at 4-bit, runtime cost |
| ≥ 1B multi-GPU | ZeRO 1/2/3, FSDP, ZeRO-Offload | 1910.02054, 2304.11277, 2101.06840 | 16 B/param baseline; stage 2 ≈ 8× fewer state bytes per GPU |

FP8 (2310.18313) needs Hopper tensor cores and is out of scope for wgpu/Metal. No primary MLX training-memory paper exists.

## Decision

1. **Order of engine work, by GB saved per effort, all independent and stackable:**
   - P7.1 enable `fusion`; f32-statistics LayerNorm; fp32-accumulated softmax and cross-entropy; re-run the f16 gate with dynamic
     loss scaling. No vendored patch. The f16 lane is admitted only if it matches f32 loss within 0.05 *and* is faster.
   - P7.2 `RecomputeFeedForward` custom `Backward` (row-chunked up-proj → GELU → down-proj, save only the block input). Uses the ADR
     107 vendored visibility patch. Target: ≥ 50 % of the 18 GB at `350m_8k` 8192.
   - P7.3 Cut Cross-Entropy custom `Backward` for the LM head; priority rises with vocabulary size.
   - P7.4 optimizer selector: Muon on 2-D weights + AdamW on embeddings/norms/head (Moonlight recipe); Adafactor via `SimpleOptimizer`
     only if Muon misses parity. Gate: loss parity on the 2048×1 smoke and on one remediation window scored on the unchanged sentinels.
2. **No engine swap.** Burn stays the only training driver (ADR 106). mlx-rs remains P6.6's throughput-only fallback; candle is
   inference-only for BabyLLM. Every memory lever above is a downstream op or a feature flag, plus the existing two-line vendored patch.
3. **1.1B fits one 80 GB A100 without sharding** by the ZeRO arithmetic (≈17.6 GB of parameter/optimizer state, ≈24 GB of residual
   activations at 8192×1 with P7.2/P7.3 landed); P7.8 measures it before any preset is trusted, and records what Burn offers for
   2–4-GPU data parallelism, since Burn has no ZeRO/FSDP and a Kaine-sized run may need it.
4. **Activation quantisation (GACT) and LOMO are not adopted now**; they are the next rung if P7.2–P7.4 leave a measured gap on the
   Mac for the 350M rung. TurboQuant remains inference-only (ADR 107).

## Consequences

- `crates/grug-train` gains: a `fusion`-enabled backend alias, `LayerNormF32Stats`, `RecomputeFeedForward`, `CutCrossEntropy`, an
  optimizer selector. Each ships with a CPU gradient-equality test against the naive path, as `RecomputeAttention` did.
- `cargo update` of burn-autodiff must keep re-applying the vendored patch until Burn exposes `graph`/`tensor` upstream; P7.2 and
  P7.3 add two more users of it.
- The P6 lane is unaffected: P6.4/P6.5 windows keep running on f32 + recompute attention; P7 levers enter a lane only after their gate.
- The 350M rung's Mac feasibility is re-evaluated after P7.2: analytic residual at 8192×1 ≈ (15·s·h·4 B)·24 ≈ 12 GB activations +
  4.2 GB state — borderline on 16 GB; the honest expectation is that 350M trains on the pod and 33M–125M rungs on the Mac.
