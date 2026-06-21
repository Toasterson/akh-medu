# ADR 073 — Full Candle `Device::Hip` backend (first-class device, not op-offload)

> Date: 2026-06-21
> Status: Accepted (in progress — Phase 45)
> Supersedes: the performance note in [ADR 068](068-rocm-candle-gpu-backend.md)
>             (op-offload was the *correctness* step; this is the *speed* step)
> Related: [ADR 067](067-candle-conversational-boundary.md) (Candle-only hot path,
>          hidden states for the Neural→VSA bridge), Phase 26/27 (training),
>          [Phase 45 plan](../plans/2026-06-21-phase45-full-candle-hip-backend.md)
> Hardware: AMD RX 7900 XTX (`gfx1100`, 24 GB, ~960 GB/s), ROCm 6.3.4 userspace

> **Numbering note.** The original GPU ADR was drafted as "053" and renumbered to
> **068** on this branch (053 is now `first-class-akh-config`). Stale "ADR 053"
> references in `src/nlu/gpu.rs` and the 2026-06-04 plan header are corrected
> alongside this ADR.

## Context

ADR 068 + the 2026-06-04 plan delivered a **working** GPU path: Gemma 12B/27B
generate on the gfx1100 via resident Q4_K weights, rocBLAS GEMMs, and bespoke HIP
kernels (`akh_dequant_q4`, `akh_dequant_q4k`, `akh_silu_mul`), with hidden-state
parity (cosine 0.9988). That path was deliberately **op-offload**: only the linear
algebra (MLP, attention projections, lm_head) runs on the GPU; every non-linear op
— RMSNorm, RoPE, softmax/attention, KV-cache, residual-add — stays in Candle on the
**CPU**, by design, so the post-norm hidden state stays extractable.

The cost of that design is structural and unfixable in place: the activation is
**downloaded and re-uploaded around every GEMM**. Tracing one transformer layer at
single-token decode shows **~8 PCIe crossings/layer → ~216/token** for a 27-layer
12B. The PCIe bus, not the GPU, is the ceiling. There is also **no tok/s metric**
in the codebase — only wall-clock for a 3-route example.

Candle has **no production upstream AMD backend** — `Device` is `Cpu | Cuda | Metal`.
Only a WIP exists ([huggingface/candle#3424](https://github.com/huggingface/candle/pull/3424),
[discussion #3327](https://github.com/huggingface/candle/discussions/3327)):
`rocm-rs`-based, BERT-only, no k-quant. Not dependable; worth seeding from.

## Decision

Make HIP a **first-class Candle device**. Vendor `candle-core` + `candle-nn`
(pinned 0.10.2) into the workspace, add `Device::Hip(HipDevice)` /
`Storage::Hip(HipStorage)`, and implement `BackendDevice` + `BackendStorage` for HIP
so the **entire** forward runs GPU-resident — tensors live in VRAM from the embedding
gather through the final norm, collapsing PCIe crossings from ~216/token to **~2**
(token ids in, logits/hidden out). `candle-transformers`' stock models then run on
`Device::Hip` unmodified.

Chosen over the narrower "device-resident forward only" because a real device also
(a) retires the vendored op-offload model forks, and (b) **unlocks on-GPU training**
for Phase 26/27 (Neural→VSA bridge, Burn/LoRA), which op-offload cannot.

### Acceptance metric: % of the memory-bandwidth roofline

Decode is **memory-bandwidth-bound** — each token streams every weight once across
~960 GB/s. So "fast" is measured as **% of the bandwidth roofline**, not raw tok/s.
Counterintuitively, **compression makes decode faster** (less bandwidth per token):

| 12B dtype | weights | fits 24 GB | roofline | target (≥60%) |
|---|---|---|---|---|
| f16    | ~24.4 GB | ❌ (no KV/display room) | ~39 tok/s | fidelity path only |
| **q8_0** | ~12.5 GB | ✅ ~11 GB headroom | ~77 tok/s | **≥45 tok/s** |
| q4_K_M | ~7.3 GB | ✅ ~16 GB headroom | ~131 tok/s | ≥78 tok/s |

**12B default = q8_0**: near-lossless hidden states for the VSA bridge, fits with
headroom. f16 is supported in-backend for max-fidelity / smaller models. The 24 GB
headroom is spent on **KV cache / batching / speculative decoding**, not on
uncompressed weights ("12B doesn't need compression" is true for *VRAM*, but
compression is still the faster decode choice).

### Hard gate (every GPU stage)

**Hidden-state cosine GPU vs CPU ≥ 0.999 (f16).** The Phase 26 bridge consumes the
post-norm last-position hidden state; routing through the device must not corrupt it.
On a real device this is one D2H copy of one vector — cheap. A regression is a
**hard stop**, not a retry.

## Consequences

- **Positive:** fast GPU-resident 12B; stock candle-transformers models on AMD;
  op-offload forks retire; on-GPU training becomes possible; CPU stays the
  always-available fallback (feature off / runtime absent → `Device::Cpu` + miette
  diagnostic, via the existing `nlu::gpu` probe).
- **Negative / risk:** a **maintained candle fork** (`Device`/`Storage` are closed
  enums — extension requires patching candle-core). Rebase cost on candle upgrades.
  Quantized HIP kernels and fused attention are the hard parts.
- **Mitigations:** pin candle 0.10.2; keep the HIP diff surgical and in clearly
  marked vendored files; gate behind the `hip` feature; per-op CPU-parity tests
  before any op is used; the hidden-cosine hard gate; a documented rebase playbook;
  upstream the backend + k-quant kernels to candle #3424 where viable.
- **Fallback if the fork stalls:** the op-offload path (ADR 068) remains functional
  and is retired only once the device backend passes all gates.
