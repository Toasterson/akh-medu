# 4-bit weights and 4-bit K/V in training: read, measured against our own stack, and not buildable here

Follow-up to [ADR 107](../decisions/107-training-memory-turboquant-vs-recompute-attention.md), which
rejected TurboQuant for training and landed recompute-backward attention. ADR 107 answered "can we
compress *saved activations*". This document answers the two questions it did not: **can we train with
4-bit weights**, and **can we 4-bit-encode a KV cache during training**.

Every paper below was read in full text, not from its abstract — several of these abstracts claim
speedups the body of the paper then discloses were never measured. Burn and ROCm claims come from
grepping the installed source, not from documentation; where the documentation disagrees with the
source, the source is cited and the disagreement is noted.

The answer is no on both counts, for four independent reasons, any one of which is sufficient. The
most interesting of them is that **4-bit weight training would cost us memory rather than save it.**

## 1. "4-bit KV cache during training" has three possible meanings; two are category errors here

There is no autoregressive KV cache in a teacher-forced training step, so the phrase has to mean one
of three things. For our lane:

| Candidate meaning | Applies to us? | Bytes at stake |
|---|---|---|
| (a) K/V tensors retained for the backward pass | **Yes** — `RecomputeAttentionState` holds them | 256 MiB (`small_8k` 8192×1), 512 MiB (8192×2) |
| (b) Rollout KV cache in engine-in-the-loop episode generation | **No — category error** | zero |
| (c) Sequence/context-parallel cached activations | **No** — single device, no sequence parallelism | zero |

**(b) is a category error in this lane specifically.** P6.3's `akh_episode_generator`
(`src/bin/akh_episode_generator.rs`) generates episodes from akh-medu's own `Engine` — rule engine,
event calculus, causal manager — with symbolic completion targets. It never runs a transformer decode,
so it has no KV cache to quantize. The engine-in-the-loop design (ADR 106) is precisely what removes
the model from the generation path.

The only genuine autoregressive KV cache anywhere in the training lane is *eval* generation:
`generate_knowledge_outputs` → `generate_from_prompt_ids_cached`
(`crates/grug-train/src/knowledge_train.rs:1146`), which runs batch 1 with no autodiff graph, between
training steps. ADR 107 already sized that at ≈268 MB per sequence. It is not on the memory critical
path and shares no peak with the training step.

**(a) is real but small.** Reading the landed custom backward op: `RecomputeAttentionState<B>`
(`crates/grug-train/src/transformer.rs:295-304`) retains `q`, `k`, `v`, `o` — each
`[batch, n_heads, seq_len, d_head]` — plus `lse` at `[batch, n_heads, seq_len]`, the optional causal
mask, and two scalars. The softmax weight block `p` is recomputed inside the backward loop
(`transformer.rs:406`) rather than stored; that recomputation is the entire ADR 107 win.

`akh_tool_small_8k` (`transformer.rs:1047`) is `d_model 512, n_heads 8, n_layers 8, d_ff 2048,
ctx 8192`, so `d_head = 64`. Per layer at batch 1, each of Q/K/V/O is
`1 × 8 × 8192 × 64 × 4 B = 16 MiB`:

| tensor group | `small_8k` 8192×1 | 8192×2 | `350m_8k` 8192×1 (extrapolated) |
|---|---:|---:|---:|
| Q+K+V+O retained, all layers | 512 MiB | 1024 MiB | 3.0 GiB |
| **K+V only** (the 4-bit target) | **256 MiB** | **512 MiB** | **1.5 GiB** |
| `lse`, all layers | 2 MiB | 4 MiB | 12 MiB |
| transient `p` block (last chunk) | 256 MiB | 512 MiB | 768 MiB |
| weights + Adam | 0.4 GB | 0.4 GB | 4.2 GB |
| **measured total** | **4.67 GB** | **7.0 GB** | not measured |

So a *perfect* 8× K/V compression saves **224 MiB at 8192×1 and 448 MiB at 8192×2** — 6.3% of the
measured 7.0 GB. For `350m_8k` it saves ≈1.3 GiB against an analytic floor of ≈25 GB (4.2 weights+Adam
+ 18 non-attention activations + 3.0 Q/K/V/O, using ADR 107's own pre-fix figures for the first two).
**It does not get the 350M rung under 16 GB, and it is not the reason the 33M rung costs what it costs.**

That last point deserves emphasis: the retained Q/K/V/O, weights, Adam and one transient chunk sum to
≈1.15 GB analytically, against 4.67 GB measured. **≈2.5–3.5 GB of our own 8192×1 footprint is currently
unattributed** — presumably FFN/LayerNorm/embedding activations under ordinary autodiff, weight gradient
buffers, and CubeCL allocator overhead, but the repo has no per-tensor profiler dump to prove it. The
uncertainty in our own memory model is roughly **seven times larger than the entire prize** from 4-bit K/V.

## 2. 4-bit weight training would *increase* our memory, not reduce it

This is the finding that settles the weights question regardless of everything else.

Every low-precision *pretraining* method in the literature keeps a high-precision master copy:

| method | 4-bit operands | master weights | optimizer moments |
|---|---|---|---|
| MSFT FP4 (2501.17116) | W+A, fwd and bwd GEMMs | high precision | FP8/FP16 |
| QuEST (2502.05003) | W+A (W4A4) | "standard precision" | standard |
| Quartet (2505.14669) | fwd W+A and both bwd GEMMs, MXFP4 | — | **FP32** |
| MXFP4 / Tseng (2502.20586) | **backward GEMMs only** | **FP32** master + BF16 params | FP32 |
| NVFP4 (2509.25149) | all 3 GEMMs, minus excluded blocks | **FP32** | **FP32** |

The 4-bit tensors are a *transient GEMM operand*, not the stored parameter. So for `small_8k` the
0.4 GB of weights-plus-Adam does not shrink — it stays, and a ≈17 MB quantized copy is **added** to it.
The win these papers are buying is FP4 tensor-core throughput and interconnect bandwidth at the 1B–12B
scale, not optimizer footprint.

And we have no FP4 tensor cores on either box (§4). On our hardware the throughput win is zero and the
memory delta is slightly negative. **4-bit weight training is strictly worse than what we run today at
the `small_8k` rung**, and the rung where it would begin to matter is not 350M either — it is wherever
weights+Adam dominate *and* the GEMMs run on hardware that can execute 4-bit natively, which describes
neither of our machines at any model size.

Two further points specific to our shape:

- **NVFP4's own recipe excludes more layers than our model has.** 2509.25149 keeps the **first two and
  last eight transformer blocks** in BF16/MXFP8 — a 10-block exclusion set. `small_8k` has **8 layers
  total**, so under NVIDIA's published recipe the entire model stays high precision. `350m_8k` (24
  layers) would have 10 of 24 blocks — 42% — excluded. The paper also states directly that *"smaller
  models trained on shorter token horizons may not require all of these techniques"*, which is a
  statement about which mitigations are needed, not a promise that small models are easier.
- **QLoRA is not 4-bit training and has no from-scratch analogue.** 2305.14314 backpropagates through a
  *frozen* 4-bit base into BF16 LoRA adapters; the paper is explicit that it dequantizes to BF16 before
  every matmul and that gradients are needed only for the adapters, never for the 4-bit weights. NF4's
  optimality argument is derived from the measured near-normality of *already-pretrained* weights
  (Shapiro-Wilk on LLaMA-7B), so it presupposes a base to freeze. Double quantization compresses the
  scale constants (≈0.37 bit/param); paged optimizers are CPU↔GPU unified-memory paging and have nothing
  to do with numerics. None of this transfers to training a model from random initialization.

## 3. The scaling laws point away from 4 bits at 33M, and they measured our size directly

ADR 107 rested on ActNN's Theorem 1 (2104.14129): the gradient stays unbiased iff the quantizer is
unbiased. Mapping the 4-bit schemes onto that test:

| method | estimator | claim |
|---|---|---|
| MXFP4 / Tseng (2502.20586) | stochastic rounding + random Hadamard | **Provably unbiased** — Lemma 3.1; Theorem 3.2 bounds SR variance from O(block) to O(log block) |
| NVFP4 (2509.25149) | SR on gradients only | Unbiasedness claimed for the gradient path |
| QuEST (2502.05003) | Hadamard + Gaussian fit + masked "trust" STE | Explicitly **not** unbiased; framed as *minimizing* bias |
| Quartet (2505.14669) | biased RTN + Hadamard | Explicitly **biased by choice** — rejected unbiased SR as *"incompatible with coarse group-wise scaling of the MXFP4 format, so we cannot use it in practice"* |
| MSFT FP4 (2501.17116) | DGE + outlier clamp/compensate | No bias claim made anywhere in the text |

So the single provably-unbiased scheme (2502.20586) quantizes **only the two backward GEMMs**, leaves the
forward in BF16, was never tested below **345M params**, and its authors state plainly that they had no
FP4 hardware — the "1.3×/1.7×" speedups in its abstract are extrapolated from INT4/INT8 proxy kernels on
A100. Quartet's authors, working in the same format, found unbiasedness unusable in practice.

**Scaling Laws for Precision (2411.04330) is the decisive input, because it measured 30M and 60M params
directly** — bracketing our 33.4M, so this is interpolation rather than extrapolation for our rung
(465 pretraining runs, N ∈ {30, 60, 110, 220}M × D ∈ {1.5…26}B tokens). Its findings run against us in
both regimes it separates:

- Post-training quantization: *"for a fixed dataset size larger models incur a smaller degradation"* —
  small models are **less** tolerant — and *"the degradation introduced by post-training quantization
  increases as models are trained on more data, eventually making additional pretraining data actively
  harmful"*, fitted as `δ_PTQ = C_T·(D^γD / N^γN)·e^(−P_post/γpost)`, a power law in tokens-per-parameter.
- Training-time precision: `N_eff(P_w,P_a,P_kv) = N·(1−e^{−P_w/γw})(1−e^{−P_a/γa})(1−e^{−P_kv/γkv})`,
  and the guidance that *"as precision of training decreases at fixed compute, we should increase
  parameters and decrease data"*. Low precision wants a **bigger** model, not a smaller one.
- It puts optimal pretraining precision at **7–8 bits**, independent of compute budget, and states:
  *"the race to low-precision training may have to stop before going below 4-bits, since this would force
  model sizes to become disproportionately (more than 4x) larger."*

A 33.4M model trained on a real corpus is by construction deep in the overtrained D/N regime where that
paper's fitted degradation is worst. The one genuinely encouraging datapoint in the whole field at our
scale is **QuEST's 30M W4A4 run** (3B tokens, C4 loss 3.272) — but that is INT4 on real 4090 tensor
cores, via a biased trust-STE, and it requires quantization-aware training, which §4 shows we cannot
express.

On K/V specifically: **no paper trains the softmax attention-probability matrix in 4 bits during
backprop.** ADR 107's search is confirmed. The closest artifact is LLM-QAT (2305.17888), which does train
through a 4-bit KV cache — but those are K/V *activations*, not the attention matrix, quantized with
per-token MinMax RTN folded into the gradient (STE-style, no bias claim anywhere in the text), and it is
data-free distillation *finetuning* against a full-precision teacher, not pretraining. GACT (2206.11357)
declined to quantize the attention matrix at all, answering attention memory with Rabe & Staats'
exact-recompute algorithm instead — the same architectural choice ADR 107 made.

## 4. Neither our framework nor our hardware can execute any of it

**Burn 0.21 cannot express quantization-aware training at all.** `burn-autodiff-0.21.0/src/ops/qtensor.rs:13-35`:

```rust
impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn q_from_data(...) -> QuantizedTensor<Self> { todo!() }
    fn quantize(...) -> QuantizedTensor<Self> { todo!() } // required for QAT
    fn quantize_dynamic(...) -> QuantizedTensor<Self> { todo!() }
    fn dequantize(...) -> FloatTensor<Self> { todo!() }
}
```

These are literal `todo!()` — calling any of them through an `Autodiff`-wrapped backend panics at
runtime. There is no `Backward` impl and no straight-through estimator; the `// required for QAT`
comment is the crate authors' own acknowledgment. Burn's book agrees: *"Burn does not currently support
QAT. Only post-training quantization (PTQ) is implemented at this time."* Our vendored
`burn-autodiff` patch cannot help — it only widens two module visibilities so `RecomputeAttention` can
exist, and the `todo!()`s are in the op impls, not behind a visibility wall.

The type system does define 4-bit values — `QuantValue::{Q4F, Q4S, E2M1}` in
`cubecl-common-0.10.0/src/quant/scheme.rs:124-144`, with `QuantStore::PackedU32` packing 8×4-bit into a
`u32` — and the CubeCL backends do have a genuinely native quantized matmul
(`burn-cubecl-0.21.0/src/ops/qtensor.rs:281-313` → `cubek::matmul::launch_ref`, contradicting one Burn
doc page that claims a blanket dequantize-then-compute; the source wins). But **the 4-bit float variants
are not implemented for data conversion** — `burn-std-0.21.0/src/tensor/quantization.rs:222-224` has
`QuantValue::E4M3 | E5M2 | E2M1 => unimplemented!("Not yet supported")`. Only the integer variants round
trip, and only through the inference path. Burn 0.22-pre.3 exists on crates.io; its one quantization PR
(`feat(quant)!: support two-level quantization scales`) adds QLoRA-style nested *scales* for PTQ and does
not close the autodiff gap.

Burn 0.21 also still has no mixed precision: `AdamState<B, D>` (`burn-optim-0.21.0/src/optim/adam.rs:54`)
stores moments as plain `Tensor<B, D>` in the backend dtype. That is why our own
`crates/grug-train/src/mixed_precision.rs` had to hand-roll f32 masters against an f16 graph — **and per
ADR 107 that landed as opt-in `metal_f16` and failed its gate** (loss 6.30 vs 3.76 at 2048×1, 2× slower).
We have already built the exact structural mechanism every FP4 paper relies on — low-precision compute
graph, high-precision master weights, loss scaling — and measured it breaking this model at *sixteen*
bits. That is the most direct evidence available about what 4 bits would do here, and it is ours.

**Neither GPU has 4-bit arithmetic:**

| box | finding | source |
|---|---|---|
| Apple M2, Metal | MSL `simdgroup_matrix` is `half`/`float` only; no sub-8-bit ALU type | MSL spec / GPU-family docs |
| wgpu/naga 29.0.4 | `ScalarKind` = `Sint, Uint, Float, Bool, AbstractInt, AbstractFloat` — **no sub-byte kind exists in the IR** | `naga-29.0.4/src/ir/mod.rs:578-593` |
| RX 7900 XTX, gfx1100, ROCm 6.3.4-76 | hipblaslt Tensile kernels for gfx1100 ship **`SS`/`HH`/`BB`/`I8I8`/`DD` only — no int4, no fp8** (fp8 appears first at gfx1200/RDNA4; int4 nowhere) | `/opt/rocm-6.3.4/lib/hipblaslt/library/` |
| RX 7900 XTX | rocWMMA **not installed**; only CubeCL's rocWMMA *code generator* is present | `find / -iname "rocwmma*"` |

`hip/library_types.h:49-52` does define `HIP_R_4I`/`HIP_R_4U`, but a defined enum constant is not a
kernel. On both boxes a 4-bit path would be pack/unpack-for-bandwidth with the arithmetic still done in
fp16/fp32 — which is why Burn packs into `u32` and unpacks with shift/mask in-kernel
(`burn-std-0.21.0/src/tensor/quantization.rs:283-311`). That is a software workaround for a hardware gap,
and it costs ALU time to save bytes.

## Recommendation: no-go, same shape as ADR 107

**4-bit weights and 4-bit K/V are inference levers misapplied to training, exactly as TurboQuant was.**
Nothing here should be built for Phase 51 P6. The four independent reasons, in order of how hard they are
to argue with:

1. **Not expressible.** Burn 0.21's autodiff quantization ops are `todo!()`; 0.22-pre does not fix it.
   Everything else is moot until that changes upstream.
2. **Memory-negative for weights.** Every method keeps FP32 masters and moments, so our 0.4 GB does not
   shrink — a 17 MB quantized copy is added. The win is FP4 tensor-core throughput, which we do not have.
3. **Too small to matter for K/V.** The maximum prize is 448 MiB at 8192×2, about 6% of the measured
   7.0 GB, and roughly one-seventh of the part of our own footprint we currently cannot account for.
4. **The evidence at our scale runs the wrong way.** 2411.04330 measured 30M/60M directly and puts
   optimal precision at 7–8 bits, with small models *less* tolerant and low precision demanding *more*
   parameters. Our own `metal_f16` gate failure at 16 bits is a local confirmation.

**The experiment actually worth running instead** is not about precision at all: **attribute the
unaccounted ≈2.5–3.5 GB at `small_8k` 8192×1.** Add a per-tensor allocation dump to the Metal lane and
break the 4.67 GB into retained-activation / gradient-buffer / allocator-overhead. The number to beat is
ADR 107's analytic 1.15 GB versus the measured 4.67 GB. Until that gap is explained, every memory
optimisation proposed for this lane — including this one — is being sized against a model that is off by
4×. If the dump shows the remainder is FFN activations, Blockwise Parallel Transformer (2305.19370) is
the cited follow-up and is worth a great deal more than 4 bits.

**Reopening triggers.** Revisit if any of these becomes true:

- Burn implements the autodiff quantization ops with a real `Backward` (watch `burn-autodiff`'s
  `ops/qtensor.rs` across releases), **and** a target box gains native 4-bit GEMM kernels — RDNA4/gfx1200
  already ships fp8 in this same ROCm install, so a hardware refresh is the plausible path.
- A BabyLLM rung reaches a size where weights+Adam, not activations, dominate the footprint **on hardware
  with FP4 tensor cores** — this is a GPU-box decision (Blackwell/MI350-class), never an M2 one.
- Someone publishes native-FP4 *pretraining* validated below ~100M params, with a measured (not
  extrapolated) wall-clock number. QuEST's 30M INT4 point is the only thing close today, and it needs QAT.
- **Inference is a separate question and stays open**: `QuantValue::Q8F` round-trips today, CubeCL has a
  native quantized matmul, and gfx1100 *does* ship int8 Tensile kernels. If we want to shrink a grug
  checkpoint for serving, **int8 PTQ is buildable now and 4-bit is not** — and 2411.04330's 7–8 bit
  optimum happens to land exactly there.

## What we do NOT know

- **Whether the ≈2.5–3.5 GB unattributed portion of our 8192×1 footprint is activations, gradient
  buffers, or allocator slack.** No per-tensor profiler dump exists in the repo. This is the single
  biggest hole and it is ours to close, not the literature's.
- **What FP4 training does below ~100M params in a native-FP4 format.** Every FP4-specific paper starts
  at 345M–400M. The only sub-100M evidence is QuEST's 30M (INT4, different format family) and
  2411.04330's fitted law. Nobody has run NVFP4 or MXFP4 at 33M.
- **How these methods behave on hardware without FP4 tensor cores.** None of the five papers discusses
  Metal or ROCm; whether they degrade to memory-savings-only or to an outright emulation slowdown on
  M2/RDNA3 is not stated anywhere and had to be inferred from the kernel-level facts in §4.
- **Whether the techniques compose** — NVFP4's two-level block-16 scaling plus RHT plus QuEST's trust
  estimator, all at once, at 33M — is untested by anyone; each paper validates its own recipe at its own
  chosen scale.
- **The `350m_8k` numbers in §1 are extrapolated**, from the per-tensor formula plus ADR 107's pre-fix
  weights/activation figures. No `350m_8k` run at 8192 exists under `training/results/`. Given that ADR
  107's own analytic projection came in 43% under the measured result for `small_8k`, the ≈25 GB floor
  should be read as a lower bound.
- **Apple's Metal findings are documentation-derived**, not grepped from an installed SDK on this Linux
  box, and are therefore lower-confidence than the Burn and ROCm findings — though naga having no
  sub-8-bit `ScalarKind` independently forecloses the wgpu path regardless of what Metal itself exposes.
