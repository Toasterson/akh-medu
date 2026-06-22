# Phase 46 — Candle static-shape inference + in-process training engine

> Date: 2026-06-22
> Status: In progress
> ADR: [074](../decisions/074-candle-static-shape-inference-and-training.md)
> Builds on: [Phase 45](2026-06-21-phase45-full-candle-hip-backend.md) (resident
>            `Device::Hip`, GEMV win → 11.5 tok/s, event profiler, HIP-graph spike)
> Goal: close the measured **4.3× gap to Ollama** (11.5 → ≥30 tok/s near-term, roofline
>       131) by making the decode forward **static-shape + allocation-free**, then
>       capturing it as a HIP graph — the same architecture the **in-process training
>       engine** (Phase 27) needs.
> Hardware: AMD RX 7900 XTX (`gfx1100`), ROCm 6.3.4. Model: `gemma3-12b.gguf` Q4_K.

## Why (one line)

The gap to Ollama is **dispatch + dynamic shapes, not compute** (GEMV already ≈ 46% of
roofline). A static-shape, allocation-free forward unlocks HIP-graph replay (kills the
per-token dispatch storm) **and** is the prerequisite for a differentiable in-process
backward/LoRA pass. See ADR 074.

## Discipline (carried from Phase 45)

- **Measure-or-revert.** Every step is gated on **unprofiled tok/s** (the ground truth)
  + bit-exact parity vs CPU. Profiler percentages are diagnostic only (whole-op windows
  over-count host-gap). Revert anything that does not move tok/s.
- **One reviewable commit per green gate.** `git -c commit.gpgsign=false`, push via
  `git push https-origin HEAD` (ssh remote key is broken on this box).
- **CPU path byte-identical** — new work is GPU-only `cuda_fwd`/device routes.

## Sequenced phases (forced by dependency: 46a → 46b → 46c → 46d)

### 46a — Static KV cache
Pre-allocate per-layer K/V device buffers to the akh's context length; write the new
position in place each token; attention reads the `[0..pos+1]` prefix. Removes the
per-token `Tensor::cat` (which copies the whole growing cache) and its malloc.
- **Where:** `quantized_gemma3.rs` / `quantized_gemma4.rs` `forward_attn` (the
  `kv_cache: Option<(Tensor, Tensor)>` + `Tensor::cat(..., 2)` path), backed by an owned
  `QHipStorage`-style device region (below Candle's functional Tensor for the in-place
  write — a small `copy2d`/scatter kernel writes `k[pos]`).
- **Gate:** parity vs the cat path (identical logits); tok/s ≥ baseline (win grows with
  context); VRAM bounded by max-ctx KV (pre-allocated, not growing).

### 46b — Flash-attention decode kernel
One HIP kernel per (batch, head): stream the static KV, accumulate q·kᵀ with online
softmax (running max + sum), apply scale + causal/length mask, and produce `·v` — no
intermediate `[b,nh,1,seq_kv]` scores tensor, no separate softmax swarm. GQA via the
KV-head mapping; f32 accumulation.
- **Gate:** bit-exact-ish vs the current `q.matmul(kᵀ)`→mask→softmax→`matmul(v)` (≤1e-3,
  online-softmax reordering); measure tok/s (collapses ~9 attention kernels → 1).

### 46c — Allocation-free forward + HIP-graph capture *(the dispatch lever)*
Make the per-token forward issue **zero `hipMalloc`**: pre-warm the caching pool so all
scratch is a cache hit (fixed shapes from 46a/46b make sizes stable), reusing the same
device addresses each token. Then capture the forward on a non-blocking stream and
replay with one `hipGraphLaunch`/token. Route the kernels + rocBLAS + memcpy onto the
capture stream (currently the null stream — uncapturable). Re-capture only on context
bucket change; feed `seq_kv` as a device-side length so the graph stays valid across
tokens.
- **Foundation:** stream/graph bindings + proven capture/replay already committed
  (Phase 45 `a969090`); the blocker (`hipMalloc` aborts capture) is removed by the
  pre-warm + fixed shapes here.
- **Gate:** capture succeeds (no malloc-abort); replay produces identical logits; tok/s
  — this is where most of the 4.3× should land.

### 46d — In-process training path *(the product goal, Phase 27 bridge)*
With a static, allocation-free forward: add a backward / LoRA-adapter pass over the
quantized base for live KG→LLM adaptation. Hidden states + gradients reachable in-proc
for the Neural→VSA bridge. (Scoped in detail in the Phase 27 plan once 46a–46c land.)

## Out of scope (this phase)
- 27B / multi-GPU; bf16/f16 KV-cache compression (PolarQuant, Phase 26); the f16/bf16
  matmul paths beyond what decode needs.
- Beating Ollama on raw serving for speed-only akhs — those stay on Ollama (ADR 070).

## Tracking
- Anima WI: Akh-Medu project `019d9844-ce74-7993-a9f0-c953b8f04a4e` (create a Phase 46
  work item; promote to agent-ready once 46a lands).
- Seshat: ingest this plan + ADR 074 so the corpus stays searchable.
