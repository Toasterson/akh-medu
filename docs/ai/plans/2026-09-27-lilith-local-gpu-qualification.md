# Lilith: Runpod serving limited to testing; qualifying the RTX 3090 and RX 7900 XTX for long runs

- **Date:** 2026-09-27
- **Status:** Proposed (slice C6b, after the H100 smoke C6)
- **Related:** [ADR 116](../decisions/116-lilith-chief-of-staff-guided-rl.md),
  [CoS plan](2026-09-27-lilith-chief-of-staff-rl-runpod.md),
  [exam and evaluation workflow](2026-09-27-lilith-exam-and-evaluation-workflow.md),
  [hardware plan](2026-09-19-capable-8-12b-akh-hardware-plan.md) §1 and §6, ADR 107 (recompute-backward
  attention), ADR 109 (Cut CE)

## Operator decision (2026-09-27)

Runpod serving exists for **testing** and is limited. After successful H100 tests, **longer runs move
to the local GPUs**: the RX 7900 XTX on hephaistos now, and the RTX 3090 once it is installed
(**not installed as of 2026-09-27**; still a planned purchase in the hardware plan). Serving and training capacity on them must be **quantified**, not assumed.

## 1. Serving policy

| Where | Use | Limit |
|---|---|---|
| Runpod `akh-medu-lilith-serve` (H100) | Test sessions: Stage 0 checks, candidate smoke after training, exam live tier | ≤ 2 h per session; idle stop after 20 min; ≤ 6 GPU-hours per week and ≤ 25% of the remaining credit per month. Training has priority on credit |
| Local 7900 XTX (hephaistos, Vulkan GGUF, proven at 23–26 tok/s) or 3090 | Everyday Chief-of-Staff serving and the C8 canary, once qualified (§4) | Local power only |

> **2026-09-29 ([ADR 121](../decisions/121-lilith-serving-on-hephaistos-llama-cpp.md)):** the operator moved
> everyday serving to hephaistos now, ahead of §4 qualification, for the frozen parent only: llama.cpp
> `llama-server` (HIP, gfx1100), Q8_0 GGUF, 2 slots x 65536 tokens; Runpod is for training and test
> sessions. Runbook: [lilith-serve-hephaistos.md](../../ops/lilith-serve-hephaistos.md). Serving a
> trained adapter still waits for the §4a LoRA-to-GGUF export and adapter-as-served parity check.

## 2. Memory arithmetic (Gemma 4 12B, from the pinned parent's config)

The architecture:
- 48 layers: 40 sliding-window (1,024) and 8 global.
- Hidden size 3,840; MLP size 15,360; 16 query heads.
- Sliding layers: head dim 256, 8 KV heads. Global layers: head dim **512**, 1 KV head, K = V.
- Vocabulary 262,144; embeddings tied.

Parameters:
- 10.90 B in the transformer blocks + 1.01 B in the embeddings = **11.91 B**.
- LoRA rank 16 on every attention and MLP projection: **66 M** parameters, whose fp32 master weights
  and Adam state take **1.05 GB**.

| Item | Size |
|---|---|
| BF16 base (blocks + embeddings) | **23.8 GB**: fills a 24 GB card by itself |
| 4-bit NF4 blocks + BF16 embeddings (QLoRA) | **7.8 GB** |
| Checkpointed hidden states (bf16, 48 layers) | 3.0 GB at 8K · 6.0 GB at 16K · 12.1 GB at 32K |
| MLP working set of the layer being recomputed | 0.8 / 1.5 / 3.0 GB |
| Full logits in bf16 (262k vocabulary) | 4.3 / 8.6 / **17.2 GB**: chunked cross-entropy is mandatory |
| Naive attention scores in one global layer (16 heads) | 2.1 / 8.6 / **34.4 GB**: needs chunked, recompute-backward attention |

Consequences:
- **On a 24 GB card, full-depth LoRA requires a 4-bit (QLoRA) base.**
  - At 16K context: about 7.8 + 1.05 + 6.0 + 1.5 + ~1.5 (runtime overhead) ≈ **18 GB**, which fits.
  - At 32K: about **25.5 GB**, which does not fit. It needs the checkpointed activations offloaded to
    host RAM. hephaistos has 31 GiB of RAM; the 3090 host is to be confirmed.
- **Two traps apply to every device, the H100 included.**
  - Head dim 512 in the global layers exceeds FlashAttention-2's limit of 256.
  - Full logits do not fit at 32K.

  C6 must therefore run with chunked cross-entropy (ADR 109 Cut CE / fused linear CE) and chunked,
  recompute-backward attention for the global layers (the P6.1b approach, ADR 107). The engine's
  existing rank-2 seam avoided both only because it trains a single query projection.

### 2a. Compressed KV (TurboQuant / int8): assumed on the local card

Gemma 4's KV cache is small. The 40 sliding layers are capped at 1,024 tokens (≈ 0.34 GB), and the 8
global layers carry one KV head with K = V (1 KB per token per layer).

| Context | KV per sequence, bf16 | int8 | ~3.5-bit (TurboQuant-class) |
|---|---|---|---|
| 8K | 0.40 GB | 0.20 GB | 0.09 GB |
| 32K | 0.60 GB | 0.30 GB | 0.13 GB |
| 128K | 1.41 GB | 0.70 GB | 0.31 GB |
| 262K (the architectural maximum) | 2.48 GB | 1.24 GB | 0.54 GB |

**Where it matters: parallel rollouts.** Sibling samples for preference contrast and the ADR 116
single-executor siblings all hold KV at once. At 32K, 16 rollouts take **9.7 GB in bf16, 4.8 GB in
int8, and about 2.1 GB at 3.5 bits**. On a 24 GB card that already holds the 7.8 GB 4-bit base,
**compressed KV (int8, or TurboQuant where the runtime supports it) is assumed for local rollouts
and long-context serving**.

**Where it does not help: backprop.** Full-sequence backpropagation keeps no KV cache. Its memory is
the checkpointed activations (12.1 GB at 32K), which need offload or activation compression instead.
This matches ADR 107: TurboQuant is an inference-side tool.

**Exception: the KLPO replay seam.** Token-wise replay reuses the cached KV. If that KV is compressed:
- **the sampler and the replay must use the identical quantization**, because KLPO needs the
  sampler's exact log-probabilities;
- the existing sampler/replay drift gate (≤ 1e-4) is re-run with the compressed cache.

**Measure in C6b:** tok/s and quality parity with bf16, int8 and TurboQuant-class KV. That covers
the adapter-as-served check and dev-family TOST on rollouts generated with compressed KV.

## 3. Throughput ceilings (analytic bounds, not forecasts)

A frozen-base LoRA step with activation recompute costs about 6·N FLOPs per token plus the LM head:
**about 71 GFLOP per training token**.

| Device | Peak basis | Training-token ceiling at 40% MFU | Single-stream decode bound (memory bandwidth ÷ weight bytes) |
|---|---|---|---|
| H100 SXM | 989 TFLOPS bf16 dense | ~5,500 tok/s | ~140 tok/s (BF16); far higher batched |
| RTX 3090 | 71 TFLOPS (hardware plan); 35.6 with fp32 accumulate | ~200–400 tok/s, plus the QLoRA dequantization cost | ~120 tok/s (4-bit) |
| RX 7900 XTX | ~30 TFLOPS effective (hardware plan estimate, unverified) | ~420 tok/s if kernels deliver; ROCm QLoRA support is itself unproven | ~135 tok/s (Q4 GGUF). **Measured: 23–26 tok/s (Vulkan) and 5.19 tok/s (BF16 HF/ROCm)**, so the runtime, not the card, is the limit today |

Read: local cards are about **14–28× slower per training token** than an H100 and cost electricity
instead of credit. They suit long, unattended windows. Recipe validation, sweeps and anything
time-critical stay on the H100.

These are ceilings for designing measurements. The Runpod handoff's rule stands: **do not quote them
as forecasts**. Only §4 measurements count.

## 4. Qualification protocol (C6b)

**Reference, on the H100 (C6).** A frozen **reference micro-run**:
- fixed seed, data slice and config: 200 SFT steps plus one preference step at 8K and 16K;
- record the loss curve, adapter hash, dev-gate scores, peak memory at 8K / 16K / 32K, training
  tok/s, generation tok/s and wall clock.

**Per local device.** Run the same micro-run under that device's recipe:
- 3090: CUDA, QLoRA NF4 base, bitsandbytes + PEFT, chunked CE, chunked global attention.
- 7900 XTX: ROCm 6.x. First check that 4-bit QLoRA works on gfx1100 (bitsandbytes ROCm backend or a
  torchao equivalent). If it does not, record BF16 + offload as the only fallback and measure its cost.

Measure:
- peak memory per context length, with ≥ 10% headroom required;
- training tok/s;
- generation tok/s (vLLM / llama.cpp on CUDA; the Vulkan GGUF path and HF on ROCm);
- time per window;
- the numerical controls: `lilith_training_cache_control.py` equivalents, gradient finite and
  non-zero, exact cold reload;
- stability over a ≥ 6 h soak: no GPU resets. A mixed-backend episode reset the 7900 XTX in
  September, so this is a known risk.

**Quality parity.**
- Score the locally trained micro-run adapter against the H100 adapter on the same dev families,
  paired. **TOST equivalence with a margin of 0.03 per family**, predeclared.
- Also score it **as served** on the target serving backend. A QLoRA/NF4-trained adapter applied to
  a Q4 GGUF base is a different arithmetic path, so the Vulkan adapter controls from
  `training/lilith/vulkan-adapter-20260922/` are re-run.

**A device qualifies for long runs when:**
1. it fits the chosen context with ≥ 10% headroom;
2. it passes the numerical controls;
3. it passes the quality-parity TOST;
4. the 6 h soak has no fault;
5. the projected wall clock of the run is ≤ the operator's window budget. The projection comes from the
   **measured** tok/s × the planned tokens.

A device qualifies for serving when it:
- meets the p95 first-token and report-latency budget from the exam's live tier;
- passes the adapter-as-served parity check.

**Division of work once qualified:**
- H100: recipe validation, sweeps, the sealed evaluation of release candidates.
- Local: long training windows, the C8 canary and everyday serving.

The supervisor still runs every local window under ADR 105: a run WI, an exit condition, and a RESULT
doc. Local spend is reported as kWh.

## 4a. Measured: RX 7900 XTX on hephaistos (C6b, 2026-09-28)

Source: `~/lilith-private/c6b/c6b-findings-20260928.json` and `SUMMARY-20260928.md` (private; raw
result JSONs, GPU telemetry and scripts under `results-20260928/`).

- **Environment:** ROCm venv on gfx1100; the parent's three sha256 pins verified on every load; GPU
  idle (36 MB) before each launch.

**NF4 QLoRA, training**

| Measure | 7900 XTX (NF4) | H100 (bf16) reference |
|---|---|---|
| 8K fits? | yes: 18.36 GB allocated, 21.71 GB reserved (15.7% headroom) | 34.6 GB allocated |
| 16K / 32K | **out of memory** | fit |
| Steady training speed at 8K | **411.5 tok/s** | 1,723.6 tok/s |
| Micro-run (200 SFT steps + preference step, admitted subset) | completed in 3,988 s; loss 1.889 → 0.941 at step 10, tracking the H100 run (1.918 → 0.917) | completed |
| Power | about 303 W at 75 °C | n/a |

- `expandable_segments` is unsupported on this ROCm build.
- `--offload-activations` hung the GPU: the driver reset it twice. One run was also OOM-killed on the
  31 GB host.

**Cold reload: failed the gate** (tolerance 1e-3)
- The adapter reloads byte-identically, and the argmax tokens are identical.
- The probe logits drifted by 0.678 max-abs, then 0.364 on a rerun of the same reload.
- The drift differs between reloads, so this is ROCm NF4 **forward nondeterminism**, not an adapter
  I/O fault. It is the same class as the CUDA finding (C6 round 2).
- Next: test the equivalents of the CUDA fix, forced MATH/EFFICIENT SDPA or deterministic mode.

**Generation**
- bitsandbytes' fast ROCm kernel for inputs of 8 rows or fewer (the single-token decode path) is
  **numerically wrong** on gfx1100: relative error 0.065, against 0.0017 for the dequant fallback.
  Stock generation produced garbage at 7.42 tok/s; with the dequant workaround it was coherent but
  only 2.77 tok/s.
- Serving on this card should use the proven Vulkan GGUF path (23–26 tok/s), which needs a
  full-depth-LoRA → GGUF export step that does not exist yet.

**Verdict (§4 rules)**
- **Long training: not qualified yet.** It fits 8K only, fails the cold-reload identity gate, and
  trains at about 24% of H100 speed.
- **Serving: not qualified via HF/bitsandbytes.** The Vulkan GGUF route needs the adapter export.
- **To requalify:**
  - fix ROCm determinism;
  - add the LoRA → GGUF export plus the Vulkan adapter-as-served parity check;
  - long runs capped at 8K context on this card.

The RTX 3090 is not installed yet (operator, 2026-09-27).

## 5. Open inputs

- **RTX 3090: not installed yet (operator, 2026-09-27).** C6b qualifies the 7900 XTX first and the 3090
  when it arrives. Its host's RAM decides whether 32K via activation offload is possible.
- **Whether 4-bit QLoRA works on the 7900 XTX (gfx1100).** This decides whether AMD can run
  full-depth LoRA at all, or only the rank-2 seam and serving.
