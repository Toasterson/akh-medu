# ADR 068: ROCm/HIP GPU Backend for the Candle LLM (AMD RX 7900 XTX)

- **Date:** 2026-06-04
- **Status:** Accepted (in progress)
- **Supersedes the GPU note in:** [ADR 067](067-candle-conversational-boundary.md)
- **Related memory:** `reference_candle_gpu_amd.md`, `project_chat_e2e_candle.md`

## Context

The conversational boundary runs on native Candle (ADR 067) so we can extract true
hidden states for the Neural→VSA bridge (Phase 26) — a capability llama.cpp / Ollama
cannot expose. The constraint is therefore **Candle-only**: no C++ inference engine in the
hot path.

The target box is an **AMD Radeon RX 7900 XTX** (Navi 31, `gfx1100`, 24 GB GDDR6).
Candle 0.10's `Device` enum is `Cpu | Cuda | Metal` — **there is no AMD backend
upstream**, neither ROCm/HIP nor Vulkan/wgpu. So on this machine Candle runs CPU-only
(~13–33 s/answer), which does not satisfy the project goal of "exercise all route paths
including LLMs … on the graphics card."

### Why not Vulkan/wgpu (the cheaper-looking path)

A wgpu/Vulkan compute backend (the Mesa RADV driver is already installed here) avoids a
system ROCm install, but **cannot run GGML-quantized ops** — it would force **f16**
weights. The VRAM math rules this out for the stated end goal:

| Model | bf16/f16 | q8 | q4_K_M |
|---|---|---|---|
| 4B  | ~8 GB ✅  | ~4 GB ✅  | ~2.5 GB ✅ |
| 12B | **~24 GB ❌** | ~12.5 GB ✅ | ~7.3 GB ✅ |
| 27B | ~54 GB ❌ | ~28 GB ❌ | ~16.5 GB ✅ |

A **full/f16 12B does not fit** in 24 GB (weights alone ≈ 24 GB, leaving nothing for the
KV cache or the display). The end goal (12B / 27B Gemma-class) therefore **requires
quantization**, and quantization on this GPU requires either llama.cpp-Vulkan (C++, vetoed
for hidden states) or a quantized GPU path inside Candle. Vulkan-f16 caps out at ~7–9B and
is a dead end for the goal.

## Decision

Build a **ROCm/HIP backend for Candle** so quantized Gemma (12B/27B at q4_K) runs on the
gfx1100, while keeping Candle (and thus hidden-state extraction) intact.

Rationale: it is the **only** path that satisfies both constraints simultaneously —
*native Candle* (hidden states survive) **and** *quantized 12B/27B on the GPU*.

### Hardware/kernel prerequisites (already satisfied on this box)

- `amdgpu` kernel module loaded; KFD compute node `/dev/kfd` present (236,0).
- Render node `/dev/dri/renderD128`; user in the `render` group.
- gfx1100 is **officially supported** by ROCm 6.x — no `HSA_OVERRIDE_GFX_VERSION` needed.

So **no DKMS / kernel-driver install and no reboot** are required — only the ROCm
**userspace** (HIP runtime, rocBLAS/hipBLAS, rocminfo).

## Implementation plan (incremental, each step independently verifiable)

1. **ROCm userspace install** (privileged, one-time) — `rocminfo` must enumerate
   `gfx1100`; a rocBLAS sgemm must run on the device. See the runbook:
   [plan 2026-06-04-gpu-rocm-candle.md](../plans/2026-06-04-gpu-rocm-candle.md).
2. **`rocm` feature + runtime detection in akh-medu** — `nlu::gpu` probes `/dev/kfd` +
   `libamdhip64.so` + `rocminfo`, returns a `RocmStatus`, and emits a **miette** diagnostic
   telling the user exactly what to install if the GPU is present but the userspace is
   missing. `CandleBackend::select_device` consults it and logs the resolved device.
3. **Candle HIP device** (the fork) — vendor/patch Candle's CUDA backend to HIP: replace
   the `cudarc` driver calls with a HIP wrapper (`libamdhip64`), hipify the elementwise +
   GEMM kernels, and wire rocBLAS/hipBLASLt for matmul. Start with f16 to get *any* op on
   the GPU, then port the **quantized** (q4_K/q8) dequant+matmul kernels — the part that
   unlocks 12B/27B.
4. **Hidden-state parity** — `forward_with_hidden()` must return the same post-norm
   last-position tensor on GPU as on CPU (tolerance check in a test).
5. **Wire into `select_device`** — prefer ROCm when `feature=rocm` and runtime is present;
   fall back to CPU with a clear log otherwise.

Steps 1–2 are completable now; step 3 is the multi-day fork.

## Consequences

- **Positive:** quantized 12B/27B Gemma on the GPU; Candle and hidden states preserved;
  no C++ in the hot path; reversible userspace-only system change.
- **Negative / risk:** Candle has no upstream HIP backend, so this is a maintained fork
  (rebase cost on Candle upgrades). The quantized HIP kernels are the hard part. rocBLAS is
  large (~GBs). Mitigation: keep the HIP backend behind the `rocm` feature; CPU remains the
  always-available default; gate the fork behind passing hidden-state-parity tests.
- **Fallback if the fork stalls:** f16 Gemma-4B via a future Vulkan/wgpu backend (fits
  comfortably), accepting the smaller model — documented as the degraded path, not the goal.
