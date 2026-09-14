# ADR 108 — The ADAM → Kaine architecture ladder: attention stays the backbone; hybrids and memory layers are measured at the rung where the literature validates them

> Date: 2026-09-14
> Status: Accepted
> Phase: 51 P7 (plan: `docs/ai/plans/2026-09-14-phase51-p7-kaine-architecture-ladder-and-training-engine.md`)
> Anima: project `019d9844-ce74-7993-a9f0-c953b8f04a4e`, plan `019f90cb-2e66-7f92-ba88-a4060be97c6e`; WIs P7.5 `01a0a1a3-439f`, P7.6 `01a0a1a3-6c2b`, P7.7 `01a0a1a3-a8a2`, P7.9 `01a0a1a4-03cb`
> Related: ADR 037 (LLM–VSA integration), ADR 104 (copy-faithful decoding, RoPE + chunked attention), ADR 107 (training memory), ADR 109 (training-engine memory program)
> Evidence: 17 papers read in full text on 2026-09-14 (arXiv IDs inline)

## Naming

- **ADAM** is the current BabyLLM lineage: `small_8k` (33.4M) through the 350M preset up to a planned ≈1.1B rung.
- **Kaine** is the next lineage, larger than ADAM's top rung. Its size is an *output* of the 350M and 1.1B measurements, not fixed here.

## Question

The Phase 51 model is a plain decoder transformer (d_model 512, 8 heads, 8 layers, d_ff 2048, RoPE, 8192 context, 1024-row
chunked recompute attention, ≈3k–8k BPE vocabulary). Before designing Kaine: does the research on attention alternatives
(state-space models, linear attention / delta-rule layers, test-time memory) say we should leave the transformer, and if so
at which size?

## What the literature establishes

### Linear attention *is* a VSA store

- Schlag, Irie, Schmidhuber 2021 (2102.11174) prove the capacity limit of the linear-attention state `S = Σ vᵢ ⊗ φ(kᵢ)` by citing
  Smolensky's tensor-product-representation crosstalk theorems directly; plain linear attention fails past 64 associations at
  head dim 64. Their delta rule `W ← W + β(v − W·φ(k)) ⊗ φ(k)` first *unbinds* the key to read the current filler, then writes only
  the residual — a cleanup-corrected write in VSA terms.
- Hrrformer (2305.19534) rebuilds attention with holographic reduced representations: circular convolution as binding, superposition
  as the memory, and **softmax as an explicit cleanup step**.
- DeltaNet (2406.06484): `S_t = S_{t−1} − β_t(S_{t−1}k_t − v_t)k_tᵀ`, chunk-parallel via the WY form of Householder products; perfect
  MQAR at 512/64 pairs. Gated DeltaNet (2412.06464): `S_t = S_{t−1}(α_t(I − β_t k_t k_tᵀ)) + β_t v_t k_tᵀ`; on the S-NIAH test with
  **UUID values** Mamba-2 "drops quickly", Gated DeltaNet degrades more slowly but is not exact. RWKV-7 (2503.14456) measures its state
  at **0.547 bits per dimension** (8192-dim state, 256 KV pairs → 72.9 % recall).
- Consequence: a delta-rule layer and the engine's HyperVec memory are the same algebra with different binding operators. This is the
  concrete bridge Phase 26 (ADR 037) was missing; it does **not** by itself argue for replacing attention.

### Copying has a hard floor that no model size removes

Jelassi et al. 2024 (2402.01032), Corollary 2.8: any fixed-state recurrent model whose state holds fewer than **L·log₂(D)** bits fails to
copy a length-L string over a D-symbol alphabet more than half the time. This is a construction limit, not a training limit. Their
160M Mamba drops to zero accuracy past training length; a Hard-ALiBi transformer trained to 50 tokens copies 1000. The learned
mechanism is an ≈5-gram hash lookup. Copy-faithful decoding (ADR 104) therefore requires real KV-cache attention over the raw tokens
on at least some layers at *every* size.

### Hybrids work, and the ratio is forgiving — from 300M upward

| Source | Scale | Attention share | Finding |
|---|---|---|---|
| Mamba-2 ablation (2405.21060 §9.2.3) | 350M, 48 layers, 7B tokens | 0→8.60 ppl, 1→8.38, 2→8.32, 6→**8.26**, 15→8.34, 24→8.50, all-attention 8.68 | flat from 2 to 15 layers; space them, not first/last |
| Jamba (2403.19887) | 1.3B | 1:3 vs 1:7 | "no substantial difference" |
| Hymba (2411.13676) | 300M | 3 global layers of the stack | all-SWA loses > 20 % on recall; 3 global layers recover it at 2.7× throughput, 3.8× smaller cache |
| Hymba parallel heads | 300M | attention + SSM in one layer | recall **51.8 %** vs 40.0 % pure attention vs 19.2 % pure Mamba |
| Kimi Linear (2510.26692) | 48B/3B active | 3:1 linear:global | 25 % global at large scale |
| Zoology (2312.04927) / Based (2402.18668) | 70M–1.3B | — | 82 % of the gated-conv gap is associative recall; a 70M attention model beats a 1.4B gated-conv model on MQAR; a 64–128-token window + linear attention spans the recall/state Pareto front |

### At tiny scale the evidence says plain attention

- BabyLM 2025 controlled comparison (2510.19493): 125M Mamba vs OPT on the 100M-word track — Mamba loses on entity tracking and long
  reading; "architectural efficiency does not automatically translate into performance gains in low-data or small-scale contexts".
- No paper ablates a hybrid below 125M. MobileLLM (2402.14905) is the only controlled ≤350M architecture study: deep-and-thin beats
  wide at fixed params, SwiGLU +1.3 pts, GQA neutral, block layer sharing +0.4/+0.6 pts; embedding tying saves little at small vocab.
- NoPE beats RoPE/ALiBi on length generalisation in Kazemnejad et al. (2305.19466); RoPE's high frequencies carry position and its
  low frequencies carry content (Barbero et al. 2410.06205).

### Test-time memory is unproven below 125M and hard to parallelise

TTT (2407.04620) starts at 125M, Titans (2501.00663) at 170M, ATLAS (2505.23735) headline tables at 340M, TTT-E2E (2512.23675) proves its
scaling claim at 3B; gains widen with scale rather than appearing at it, and ATLAS/TTT-E2E state that their inner loops "cannot be
parallelized" without bespoke kernels. Miras (2504.13173) lists only vector / matrix / MLP memories — no external-store seam. The one
memory paper with gains at its smallest size is **Memory Layers at Scale** (2412.09764, 134M → 8B), a product-key top-k lookup table,
i.e. the same operation as querying an external vector index. Larimar (2403.11901) is the LLM-agnostic episodic-module pattern.

## Decision

1. **ADAM stays a transformer at every rung.** No SSM or recurrent backbone below 300M; the tiny-scale evidence is unanimous and the
   tool-calling skills (verbatim copy, entity tracking, recall) are exactly where recurrent models lose.
2. **Attention layout becomes sliding-window on most layers + 2–3 spaced global layers** (P7.5). Safe at all sizes per Hymba/Mamba-2,
   keeps the copy bound satisfied through the global KV cache, and shrinks the recomputed score chunks from L×L to L×w. It is the
   first Kaine-lineage change and is validated on `small_8k` against the unchanged release gate.
3. **The 33M rung takes the MobileLLM ablations** (P7.6): deep-and-thin at fixed params, GQA, block layer sharing, SwiGLU; NoPE as one
   ablation rung (it conflicts with the RoPE context-extension path of ADR 104 and must earn its place).
4. **The 350M rung runs the hybrid comparison** (P7.7): Gated DeltaNet as Hymba-style *parallel* heads (sequential stacking showed no
   small-scale gain) against the P7.5 SWA transformer as control, on the akh gates plus MQAR and UUID/path copy probes. Gated DeltaNet is
   the only recurrent candidate: best published UUID-needle behaviour, chunked WY form is matmuls + triangular solves (no custom kernels
   in Burn). If it does not win on the akh corpus at 350M, Kaine does not carry it.
5. **Global attention never goes away.** Whatever the 350M/1.1B results, Kaine keeps real KV-cache attention on the spaced global layers.
6. **No Titans / TTT / ATLAS.** Memory belongs in the engine (KG + HyperVec + TMS), reached through tools today. The published seam for
   moving it *inside* Kaine is a product-key memory layer (P7.9 design spike); the engine remains the store of record.
7. **Kaine's size is decided after P7.7 and P7.8**, from measured gate results, not chosen now.

## Consequences

- ADR 104's chunked attention gains a window parameter and a global-layer set; the Grug serving runtime must honour the window in its KV cache.
- ADR 037 (Phase 26) gains a concrete neural↔VSA bridge: a delta-rule layer's state is a bind/superpose store; HRR binding (Hrrformer)
  is the alternative algebra. Phase 26's "Neural→VSA bridge" item should be re-planned around that equivalence.
- Runs at 350M and above leave the Mac lane (ADR 106) and fall under ADR 105 pod governance (WI with exit condition, dollar cap,
  operator approval); the Runpod scope rule applies.
- Training-memory work needed to reach the 1.1B rung and Kaine is ADR 109.
