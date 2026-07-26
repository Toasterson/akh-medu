# ADR 074 — Candle static-shape inference + in-process training engine

> Date: 2026-06-22
> Status: Accepted (in progress — Phase 46)
> Builds on: [ADR 073](073-full-candle-hip-device-backend.md) (the resident
>            `Device::Hip` backend + this session's GEMV win), [ADR 070](070-selectable-chat-backing.md)
>            (Candle is a *selectable* per-akh backing, not a replacement for Ollama)
> Related: [ADR 067](067-candle-conversational-boundary.md) (hidden states for the
>          Neural→VSA bridge), Phase 26 (LLM-VSA), **Phase 27 (Live KG→LLM training)**,
>          [Phase 45 plan](../plans/2026-06-21-phase45-full-candle-hip-backend.md),
>          [Phase 46 plan](../plans/2026-06-22-phase46-candle-static-shape-inference.md)
> Hardware: AMD RX 7900 XTX (`gfx1100`, 24 GB, ~960 GB/s), ROCm 6.3.4

## Context — the measured gap, and why it is not compute

Phase 45 made the 12B Gemma run **fully resident** on `Device::Hip` and drove decode
from **0.17 → 11.5 tok/s** (gemma3-12b Q4_K), ~68× over the op-offload baseline. The
event-based GPU profiler then localised every remaining cost. To calibrate "good", we
benchmarked **the same GGUF on the same GPU** through Ollama (local llama.cpp / ROCm):

| engine | decode | VRAM | % of 131 tok/s roofline |
|---|---|---|---|
| **Ollama (llama.cpp / ROCm)** | **~49 tok/s** | 9.1 GB | 37% |
| Candle `Device::Hip` (Phase 45) | 11.5 tok/s | 19 GB | 8.8% |

Ollama is **4.3× faster on half the VRAM**, out of the box, locally (no cloud — also
FLOSS and sovereign). The decisive finding from Phase 45 is **the gap is not compute**:
the warp-per-row quantized GEMV already streams the weights at **~440 GB/s ≈ 46% of
roofline** — competitive with llama.cpp's `mul_mat_vec_q`. The gap is two structural
things llama.cpp solved years ago and Candle's functional Tensor model fights:

1. **Dispatch-bound decode.** The forward issues *hundreds* of tiny kernel launches per
   token. The event profiler showed the GPU is idle-gapped between them, waiting on the
   CPU to dispatch the next kernel (Σ of clean per-kernel windows = 123% of wall — the
   over-count *is* the idle). Fusing any *single* op (custom attention, fused softmax)
   gave **zero** gain — removing a few kernels just redistributes the idle. The lever is
   eliminating the dispatch overhead wholesale: **HIP graph capture/replay** (one
   `hipGraphLaunch`/token instead of hundreds of launches).
2. **Dynamic shapes + per-op allocation.** Candle reallocates the KV cache every token
   (`Tensor::cat`) and `hipMalloc`s ~388×/token (caching-allocator misses on the growing
   attention/KV buffers). The HIP-graph feasibility spike ([Phase 45](../plans/2026-06-21-phase45-full-candle-hip-backend.md),
   commit `a969090`) **proved** capture/replay works on this gfx1100 *but* a synchronous
   `hipMalloc` during capture invalidates it — so the live forward cannot be captured
   as-is.

## Decision

Build a **static-shape, allocation-free decode path** in the vendored Candle HIP
backend, then capture it as a HIP graph. Concretely:

1. **Static KV cache.** Pre-allocate per-layer K/V to a fixed capacity (the akh's
   context length), write the new position in place, and run attention over the valid
   prefix with a length-masked tail — no `cat`, no realloc, fixed buffer addresses.
2. **Flash-attention decode kernel.** Fuse q·kᵀ → (scale, mask) → online-softmax → ·v
   per head over the static KV, in one kernel — collapsing the ~9-kernel attention swarm
   and removing the `[b,nh,1,seq_kv]` scores materialisation.
3. **Allocation-free forward.** All per-token scratch comes from a pre-warmed pool /
   bump arena reset per token, so every token reuses the *same* device addresses (no
   `hipMalloc` during the forward).
4. **HIP graph capture/replay.** Capture the now-static, allocation-free forward once;
   replay with one host dispatch per token. Re-capture only when crossing a context
   bucket. (Bindings + proven capture/replay already committed in Phase 45.)

### Why this is the *training* architecture too (the strategic reason)

This is not only an inference optimisation. A **stable, pre-allocated, fixed-shape
forward is the exact prerequisite for an in-process backward / LoRA pass** (Phase 27,
Live KG→LLM training):

- Static buffers ⇒ a stable autograd tape / activation checkpoints to differentiate.
- Allocation-free forward ⇒ the backward reuses the same buffers; no churn.
- Owned, in-process forward ⇒ gradients and hidden states are directly reachable for
  the Neural→VSA bridge and for adapting the model **from the live knowledge graph** —
  something an external server (Ollama/llama.cpp) structurally cannot do.

So Candle is **not** chosen to beat Ollama at raw serving — Ollama remains the
**workhorse** for speed-only akhs ([ADR 070](070-selectable-chat-backing.md), the
backing is selectable per-akh). Candle is chosen to be the **one engine the project
owns** that does fast inference **and** in-process training/adaptation on the same
static-shape forward. Closing the inference gap is what makes that single engine viable
as a product, rather than a slow research path.

## Consequences

- **Major, multi-session effort.** This rewrites the decode path's buffer management
  below Candle's functional Tensor API (the KV cache and scratch become owned device
  regions). It is sequenced and parity-gated (Phase 46 plan) so each step lands green.
- **Target:** materially close the 4.3× gap — graph replay + flash-attention should
  recover most of the dispatch-bound loss; ≥30 tok/s is the near-term goal, with the
  roofline (131) the ceiling. Every step is measured with the Phase 45 event profiler;
  **revert anything that does not move unprofiled tok/s** (the Phase 45 discipline:
  6 experiments, 2 kept).
- **CPU path stays byte-identical.** All new kernels are GPU-only `cuda_fwd`/device
  routes; CPU keeps the candle-nn fast ops, parity-gated on gfx1100.
- **Risk:** static-shape attention computes over a padded KV tail (wasted compute at
  low position) unless bounded by a device-side length; flash-attention with the causal
  mask + GQA is a non-trivial kernel. Both are addressed in the Phase 46 plan.

## Alternatives considered

- **Just use Ollama everywhere.** Fastest path to fast inference, but forecloses
  in-process training/adaptation — the actual product differentiator. Kept as the
  workhorse backing, not the engine.
- **Match llama.cpp kernel-for-kernel without graphs.** The profiler shows the gap is
  dispatch, not compute; more kernel fusion alone plateaus (proven: fused softmax = no
  gain). Graphs are the structural fix.
- **Adopt llama.cpp as a library (link ggml).** Gives speed but not a differentiable,
  owned forward; defeats the training goal and the Neural→VSA hidden-state access.
