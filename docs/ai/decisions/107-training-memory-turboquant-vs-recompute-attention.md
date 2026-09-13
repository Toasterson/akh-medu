# ADR 107 — Training memory on the 16 GB Mac: TurboQuant is an inference lever; the local 8192 fit comes from recompute-backward attention + causal K/V slicing + mixed precision

> Date: 2026-09-13
> Status: Accepted
> Phase: 51 P6 (local training lane, ADR 106)
> Anima: project `019d9844-ce74-7993-a9f0-c953b8f04a4e`; WI P6.1 (`01a09ac7-8638`) measured the problem, WI P6.1b (`01a09b49-ff82-7d40-9696-6bdf44b1eb1f`) implements the fix
> Related: ADR 037 (PolarQuant for inference KV cache, Phase 26c), ADR 104 (RoPE + chunked attention), ADR 106 (Metal lane)
> Evidence: `training/results/p6-1-metal-smoke-20260913-RESULT.md`

## Question

P6.1 measured that `small_8k` (33.4M params) cannot train at 8192 context on the M2 (16 GB unified): 4096×1
needs 11.9 GB even with gradient checkpointing, 8192 swaps past 21 GB. The operator asked whether
**TurboQuant** (Google Research, arXiv 2504.19874) should be used for *model and context compression during
training* so the run fits before we consider the GPU box.

## What the memory actually is

Weights + Adam state for `small_8k` are **0.4 GB**. Everything else is activations retained for backward, and
the retained **attention score tensors** dominate. Our `chunked_causal_attention` (ADR 104) scores each
1024-query chunk against the **full** key length, keeps pre-mask scores, masked scores and softmax weights
for backward, so per layer that is ≈ 3 × H × L² × 4 bytes:

| shape | ctx | attention scores | other activations | weights+Adam | measured (no ckpt / ckpt) |
|---|---:|---:|---:|---:|---|
| small_8k | 2048 | 3.2 GB | 0.7 GB | 0.4 GB | 4.7 / 3.5 GB |
| small_8k | 4096 | 12.9 GB | 1.3 GB | 0.4 GB | 16.3 / 11.9 GB |
| small_8k | 8192 | 51.5 GB | 2.7 GB | 0.4 GB | killed / >21 GB swap |
| 350m_8k | 8192 | 309 GB | 18 GB | 4.2 GB | — |

The analytic model reproduces the measurements within 15 %. Attention scores are 82–95 % of the footprint;
the quadratic term is the whole problem. Chunking (ADR 104) bounds the *forward* working set but under
`Autodiff` every chunk's scores are retained, so it never reduced backward memory.

Two further measurements taken for this ADR (same 5-step smoke, `metal,grad_checkpointing`):

| variant | 2048×1 | 4096×1 | note |
|---|---|---|---|
| `Metal<f32>` | 1.41 s, 3.5 GB | 4.20 s, 11.9 GB | loss@4 = 4.35 |
| `Metal<f16>` (whole backend) | 0.99 s, 2.2 GB | 2.61 s, 5.6 GB | loss@4 = 5.25 — converges visibly slower |

f16 halves memory and time, but Burn 0.21 has no mixed precision: with `Metal<f16>` the weights, gradients
**and Adam moments** are f16 (`burn-optim` keeps state as `Tensor<B, D>` in the backend dtype), which is why
the loss lags. f16 *activations* are fine; f16 *optimizer state* is not.

## What TurboQuant is (literature, full text read 2026-09-13)

- **TurboQuant** (2504.19874): data-oblivious vector quantizer — random rotation → per-coordinate Lloyd-Max
  scalar quantizer on the (Beta-distributed) rotated coordinates → **QJL** (2406.03482) 1-bit residual for an
  *unbiased inner-product estimator*. Within ≈2.7× of the Shannon distortion bound at every bit width; no
  calibration. Evaluated as **KV-cache quantization** on Llama-3.1-8B / Ministral-7B: quality-neutral at
  3.5 bits/channel, marginal loss at 2.5, KV cache 4.5–6× smaller, up to 8× decode throughput on H100
  (Google Research blog, 2026-03). **Neither the paper nor the blog mention training, gradients or backward
  passes** — the technique is defined for a frozen model's inference cache and for ANN indices.
- **PolarQuant** (2502.02617) is TurboQuant's stage 1 predecessor (already chosen for Phase 26c inference KV in
  ADR 037); **QJL** is the stage 2 it reuses. Same inference-only scope.
- **Activation-compressed training** is a different lineage: ActNN (2104.14129) / GACT (2206.11357) quantize
  saved-for-backward activations with *stochastic* rounding — ActNN's Theorem 1 proves the gradient stays an
  unbiased estimator iff the quantizer is unbiased, which deterministic rounding (and TurboQuant's Lloyd-Max
  stage) is not. ActNN reaches 1.25-bit average on CNNs (12× activation memory); GACT ≈4-bit adaptive on
  BERT-large within 0.1–0.3 pt, ≈2-bit at a 1.5–2.5 pt cost (8.1× memory). **GACT explicitly does not
  quantize the attention/softmax matrix**: it names the O(L²) term as the reason per-tensor quantization is
  insufficient and composes with Rabe & Staats' linear-memory self-attention (2112.05682) instead — the same
  split this ADR makes. No 2024–26 work that quantizes the score tensor during backward was found (arXiv
  rate-limited; OpenAlex checked). All read in full text on 2026-09-13.
- **Removing the quadratic term**: FlashAttention (2205.14135) stores only O and the row logsumexp and
  recomputes P in backward → memory O(L·d) per layer instead of O(L²); Blockwise Parallel Transformer
  (2305.19370) fuses attention + FFN blockwise: per layer 2·b·s·h instead of FlashAttention's 8·b·s·h FFN
  floor, i.e. the follow-up if non-attention activations ever matter; Mini-Sequence Transformer (2407.15892)
  explicitly leaves attention alone and cuts MLP/LM-head memory by 1/M — irrelevant for us (non-attention
  activations are 2.7 GB at 8192). All three read in full text on 2026-09-13; at our shape FlashAttention-style
  recomputation replaces ≈2 GiB of scores per layer with ≈512 KiB of row statistics.

Burn 0.21 ships a CubeCL flash-attention kernel (`burn::tensor::module::attention`, wgpu/Metal) — but the
`Autodiff` backend routes `attention` to `attention_fallback` (naive matmul/softmax), so **training through
it retains the same quadratic tensors**. It helps inference only.

## Decision

1. **TurboQuant is not adopted for training.** It has no gradient path; applying it to saved activations
   would mean re-deriving unbiased backward estimators through rotation + Lloyd-Max + QJL that no paper
   attempts, for a term (non-attention activations ≈ 2.7 GB at 8192, ≈ 18 GB for `350m_8k`) that gradient
   checkpointing already handles. "Model compression during training" is moot: weights + Adam are 0.4 GB.
2. **TurboQuant/PolarQuant stay an inference decision** (ADR 037, Phase 26c): the `small_8k` KV cache is only
   8 layers × 2 × 8192 × 512 × 4 B ≈ 268 MB per sequence, so the win there is for the Candle/Gemma serving
   path on this box, not for BabyLLM. Revisit when a BabyLLM rung ≥ 350m serves 8k context locally.
3. **The local 8192 fit is P6.1b, three changes in `crates/grug-train/src/transformer.rs`, in this order:**
   - **(a) Causal-prefix K/V slicing** in `chunked_causal_attention`: chunk *i* scores against keys
     `[0, (i+1)·chunk)` instead of the full length. Halves score memory, exact, ~10 lines.
   - **(b) Recompute-backward chunked attention**: a custom `Backward` op (`burn-autodiff` `Backward<B, N>`,
     as `Matmul` does) whose forward runs the chunked/online softmax through the *inner* backend (may use
     the CubeCL flash kernel), saves only `O` and per-row logsumexp, and whose backward recomputes each
     chunk's `P`, then `dV = Pᵀ·dO`, `dP = dO·Vᵀ`, `dS = P ∘ (dP − rowsum(dO ∘ O))`, `dQ = dS·K`,
     `dK = dSᵀ·Q`. Memory becomes O(L·d) per layer plus one transient chunk of scores: analytic total at
     8192 ≈ 2.7 GB (from 54 GB). Unit test: gradients equal the naive path on CPU to 1e-4 at L = 256.
   - **(c) f16 activations with f32 master weights**: keep `Metal<f16>` for the forward/backward graph and add
     an optimizer wrapper that holds f32 copies of params and Adam moments, casting per step (Burn 0.21 has
     no built-in mixed precision). Must reproduce the f32 loss curve on the 2048×1 smoke (loss@4 = 4.35 ± 0.05)
     before it is allowed in a lane. Optional if (a)+(b) already fit 8192×2; mandatory for any `350m_8k` stage.
4. **Go/no-go for the GPU box moves behind P6.1b.** With (a)+(b) the 8192 stage of remediation v4 is expected
   to fit at batch 1–2 and run at roughly the measured 1.5k tokens/s. Only a *measured* miss after P6.1b
   (throughput, not memory) triggers P6.6.

## Consequences

- ADR 104's "chunked attention" is re-characterised: it was a forward memory bound, not a training one.
- P6.6 (pod / mlx-rs fork) loses its memory motivation; its remaining trigger is throughput.
- ADR 037's PolarQuant item is unchanged; it gains a note that TurboQuant's QJL stage is worth adding when
  the serving model exceeds ~1B params.
- `train_foundation --context` (P6.1) stays: it is how the 2048/4096 stages of a run are scheduled.
