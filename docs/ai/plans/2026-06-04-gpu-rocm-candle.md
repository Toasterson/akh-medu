# Plan: GPU LLM Route via ROCm/HIP (AMD RX 7900 XTX)

- **Date:** 2026-06-04
- **Status:** Complete (all checklist items done — 12B+27B validated; on-device activations,
  hidden-state parity, and native Q4_K landed)
- **ADR:** [053](../decisions/068-rocm-candle-gpu-backend.md)

## Goal

Run quantized Gemma (12B / 27B at q4_K) on the AMD RX 7900 XTX (`gfx1100`) through
**native Candle**, so the chat/inference route exercises the GPU while hidden-state
extraction (Phase 26 Neural→VSA bridge) keeps working. CPU stays the always-available
fallback.

## Why ROCm (not Vulkan)

The end goal is a 12B/27B-class model. A full/f16 12B needs ~24 GB **just for weights** —
it does not fit in 24 GB once you add the KV cache and the display. The goal therefore
requires **quantization**, and a Vulkan/wgpu Candle backend can only do f16 (no GGML
quantized ops) — capping it at ~7–9B. ROCm/HIP keeps quantization *and* keeps Candle. See
ADR 068 for the full VRAM table and trade-off.

## Hardware/kernel status on this box (already ROCm-ready)

- `amdgpu` loaded; `/dev/kfd` (KFD compute node) present; `/dev/dri/renderD128` present.
- User in `render` group. gfx1100 is officially ROCm-6.x-supported (no `HSA_OVERRIDE`).
- ⇒ **userspace-only install, no DKMS, no reboot.**

## Runbook — privileged userspace install (one-time)

Already applied on this box (2026-06-04); recorded here for reproducibility / other hosts.

```bash
# 1. ROCm 6.3.4 apt repo (jammy / Ubuntu 22.04 base — Zorin 17 qualifies)
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key \
  | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.3.4 jammy main' \
  | sudo tee /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
  | sudo tee /etc/apt/preferences.d/rocm-pin-600
sudo apt-get update

# 2. Userspace only — HIP runtime + BLAS + rocminfo. NOT amdgpu-dkms (kernel is fine).
sudo apt-get install -y rocminfo rocm-hip-runtime rocblas hipblas

# 3. Make ROCm libs discoverable
echo '/opt/rocm/lib' | sudo tee /etc/ld.so.conf.d/rocm.conf
sudo ldconfig
```

### Verification (the "GPU is alive for compute" proof)

```bash
/opt/rocm/bin/rocminfo | grep -E 'Name:|gfx'          # must list gfx1100
/opt/rocm/bin/rocm-smi                                  # board present, VRAM, temp
# rocBLAS sanity (a real GEMM on the device):
/opt/rocm/bin/rocblas-bench -f gemm -r s -m 4096 -n 4096 -k 4096 2>/dev/null | tail
```

If `rocminfo` shows `gfx1100`, the compute stack works and the Candle HIP fork has a
foundation.

## Candle HIP backend (the fork — multi-day)

1. `nlu::gpu` runtime probe + miette diagnostics (DONE first — guides install, no GPU
   needed to compile). `CandleBackend::select_device` consults it.
2. Vendor Candle's `candle-core` CUDA backend; swap `cudarc` → a `libamdhip64` wrapper
   (HIP is source-compatible with CUDA's runtime API; `hipify-perl` does ~90% mechanically).
3. f16 path first: get one matmul + one elementwise kernel running on the GPU, verified
   against the CPU result.
4. Quantized path: port the q4_K/q8 dequant+matmul kernels (the part that unlocks 12B/27B).
5. Hidden-state parity test: GPU `forward_with_hidden()` ≈ CPU within tolerance.
6. `select_device` prefers ROCm under `feature=rocm`; CPU fallback with a clear log.

## Verification of the end goal

- `akh chat` answers with Gemma-12B-q4 generated **on the GPU** (tok/s ≫ CPU).
- `examples/candle_hidden.rs` extracts hidden states on the GPU matching CPU.
- VRAM (`rocm-smi`) shows the model resident on-device during generation.

## Status / follow-ups

- [x] Kernel prerequisites confirmed (`/dev/kfd`, render group, gfx1100).
- [x] ROCm 6.3.4 userspace installed; GPU compute proven (`scripts/rocm_smoke.sh` → PASS).
- [x] `nlu::gpu` probe + miette diagnostics; `select_device` consults it.
- [x] **First in-engine GPU kernel** — `nlu::hip` exposes a rocBLAS sgemm as a Candle
      `CustomOp2` (`gpu_matmul`); a Candle matmul runs on the gfx1100 from the Rust engine and
      matches the CPU result (`gpu_matmul_matches_cpu`). Libraries are dlopen'd at runtime so a
      `--features rocm` binary still starts without ROCm.
- [x] **Resident weights** — `DeviceBuffer` + `RocmBlas::{alloc_upload, sgemm_resident}` +
      `ResidentLinear` upload a weight to the GPU once, then ship only the activation per call.
- [x] **LLM runs on the GPU** — `AKH_GPU_MATMUL=1` routes the vendored Qwen2 model's SwiGLU
      MLP (gate/up/down — majority of transformer FLOPs) through `GpuMlp` on the gfx1100. Real
      translate/dialogue/verbalize generation verified: VRAM 34 MB → 4846 MB (resident f32 MLP
      weights), gpu_busy 59%; GPU (f32) and CPU (q4) output **identical**. Attention, norms,
      sampling stay in Candle (hidden states intact).
- [x] **Attention + lm_head on GPU** — `GpuAttnProj` (wq/wk/wv/wo) + `GpuLinear` lm_head;
      full Qwen linear stack on GPU (VRAM 6412 MB f32, gpu_busy 75%, deterministic outputs
      match CPU).
- [x] **Quantized resident weights (int4)** — `ResidentQuantLinear` + HIP dequant kernel
      (`src/nlu/hip_kernels/dequant.hip`, built by build.rs). `AKH_GPU_QUANT=1`. Verified
      Qwen: VRAM 1511 MB vs 6412 MB f32 (4.2× smaller), gpu_busy 95%. Exactness test
      (`quant_linear_matches_cpu_when_lossless`) passes.
- [x] **Gemma model on GPU** — same path wired into the vendored Gemma3 model. Verified
      Gemma3-4B int4: VRAM 3306 MB, gpu_busy 96%, coherent.
- [x] **Gemma 12B validated on GPU (the end-goal model)** — `unsloth/gemma-3-12b-it-Q4_K_M`
      (7.3 GB GGUF) loaded through the vendored Gemma3 model with `AKH_GPU_MATMUL=1
      AKH_GPU_QUANT=1`. **Peak VRAM 8251 MB** (fits 24 GB with ~16 GB headroom), gpu_busy 94–99%.
      All three Candle routes pass on the GPU:
        - translate: `dogs are mammals` → clean `Triple{dogs, are, mammals}` AbsTree.
        - dialogue: "Oh, hello there, it's lovely to meet you…" (coherent greeting).
        - verbalize (grounded): "I state that Paris is the capital of France." (correct).
      EXIT=0. The 8.25 GB resident matches the ~7–8 GB int4 projection.
- [x] **Gemma 27B validated on GPU** — `unsloth/gemma-3-27b-it-Q4_K_M` (16.5 GB GGUF) through the
      same path. **Peak VRAM 17666 MB** (fits 24 GB, ~6 GB headroom), gpu_busy 63–87%, EXIT=0,
      clean teardown to 34 MB. All three routes coherent: translate → clean `Triple{dogs, are,
      mammals}`; dialogue → "Oh, hello there—it's lovely to meet you…"; verbalize-grounded →
      "Paris is the capital of France." **Both end-goal models (12B + 27B) now run on the AMD
      RX 7900 XTX** — the ROCm/HIP + int4-resident path delivers exactly what it was designed for.
- [x] **Per-call GPU overhead removed (persistent rocBLAS handle)** — `RocmBlas` now creates the
      rocBLAS handle **once** at load and reuses it across every GEMM (destroyed in `Drop`),
      instead of `rocblas_create_handle`/`destroy` per call. For single-token decode the activation
      is only a few KB, so handle creation (device workspace alloc + device query) dominated; a 27B
      forward issues ~420 GEMMs/token, so this removes ~420 create/destroy cycles per token.
      **Measured A/B** (Qwen-1.5B int4 on GPU, full 3-route `candle_route`, 2 runs each):
      86.2 s → 68.9 s wall — **~20 % faster** (generation-only speedup larger; model load + VRAM
      sampling are fixed overhead in both). Numerically identical — all 3 `nlu::hip` parity tests
      still pass. `RocmBlas` is now `unsafe Send+Sync` (handle used only on the synchronous
      generation thread).
- [x] **Keep activations on-device across the layer stack** (fused SwiGLU MLP + shared qkv upload).
      - *Fused MLP*: `GpuMlp::forward` uploads the hidden state **once**, runs gate → `silu·up` →
        down entirely on the GPU (new `akh_silu_mul` HIP kernel + `sgemm_into` pure device-to-device
        GEMM + `forward_dev` device output), and downloads **once** — vs the old two uploads, three
        downloads, and a host-side `silu`. The int4 chunked dequant scatters each column-chunk
        straight into the wide output via `ldc=out_dim` (no extra copy). A single default stream
        serializes dequant→sgemm→silu_mul, so no inter-op syncs are needed. Falls back to the
        unfused path if the kernel isn't built or the device path errors.
      - *Shared qkv upload*: `GpuAttnProj::project_qkv` uploads the normed hidden state once and
        runs all three projections on the GPU (wired into both vendored Qwen2 + Gemma3). Attention
        still downloads q/k/v because RoPE / softmax / the KV cache run in Candle between qkv and
        `wo`; only the redundant uploads are removed.
      - **Measured** (Qwen-1.5B int4 GPU, full `candle_route`, 2 runs each): 86.2 s (original) →
        68.9 s (persistent handle) → 64.0 s (fused MLP) → 64.6 s (qkv share — flat within noise, as
        expected: the shared activation is only a few KB at single-token decode; the win shows at
        larger batch/seq). **~25% cumulative.** Output coherent and structurally identical at every
        step. Parity test `fused_mlp_matches_cpu` (+ existing 3) all pass.
      - *Note:* truly never leaving the GPU across the whole stack would also require RoPE / RMSNorm
        / softmax / residual-add kernels (the full device backend); those stay in Candle by design
        so hidden states remain extractable for the Neural→VSA bridge.
- [x] **Hidden-state parity on GPU** (`forward_with_hidden()` GPU ≈ CPU) — the Neural→VSA bridge
      (Phase 26) consumes the post-norm hidden state, so routing the model through the GPU must not
      corrupt it. `examples/candle_hidden_parity.rs` loads the same model three ways (CPU, GPU-f32,
      GPU-int4) and compares the extracted hidden vector. **Measured (Qwen-1.5B, dim 1536):**
      GPU-f32 vs CPU **cosine 0.999787** (relL2 2.1%) — tight, since both use the identical q4_K
      weights and differ only in matmul accumulation order; GPU-int4 vs CPU cosine 0.992659
      (relL2 12%) quantifies the extra int4 quant noise (direction well preserved — what the VSA
      bridge needs). The example exits non-zero if GPU-f32 cosine < 0.999, so it doubles as a
      CI-style gate when ROCm is present.
- [x] **Native GGUF Q4_K kernel** (use the shipped weights as-is, no dequant→requant). New HIP
      kernel `akh_dequant_q4k` mirrors candle's `BlockQ4K::to_float` bit-for-bit (f16 d/dmin,
      6-bit packed sub-scales via `get_scale_min_k4`, 144-byte/256-weight blocks). `ResidentQ4KLinear`
      uploads the raw Q4_K bytes from the `QTensor` (`QTensor::data()`/`dtype()`) and dequant-chunks
      them on the GPU straight into the rocBLAS B-operand; `GpuLinear::from_qtensor` prefers it when
      `AKH_GPU_QUANT=1` and the tensor is Q4_K-shaped, falling back to int4/f32 for other dtypes
      (e.g. Q6_K). Wired into both vendored Qwen2 + Gemma3 (MLP, attention, lm_head). **Payoff —
      hidden-state cosine GPU-int4 vs CPU (Qwen-1.5B): 0.992659 → 0.998787** (relL2 12.1% → 5.0%),
      now nearly matching the f32 path (0.999787); the dequant→requant loss is gone. Output coherent.
      New parity test `q4k_native_matches_candle` (GPU dequant+GEMM == candle's own Q4_K dequant).
