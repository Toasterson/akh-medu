# Plan: Phase 45 — Full Candle `Device::Hip` Backend (fast on-host 12B GPU akh)

- **Date:** 2026-06-21
- **Status:** In progress — **45a + 45b landed** (2026-06-21); 45c next.
- **ADR:** extends [068](../decisions/068-rocm-candle-gpu-backend.md); a new **ADR 073** records the
  general-backend decision (op-offload → first-class device).

> **Progress (2026-06-21, hephaistos).**
> - **45a ✅** — tok/s + PCIe-crossing harness (`examples/candle_bench`, `StopMode::Fixed`,
>   `nlu::hip` counters). **Measured 12B Q4_K baseline: op-offload GPU 0.17 tok/s (419 PCIe
>   crossings/token, 0.1% of roofline) — ~7× slower than CPU's 1.16 tok/s.** The number to beat.
>   (commit `010a878`)
> - **45b ✅** — vendored candle-core fork (`vendor/candle-core`, `[patch.crates-io]`) + `hip`
>   feature; HIP backend fills the **cuda module slot** (zero match-arm changes). `cargo build
>   --features hip` OK; full lib suite under `--features hip` **2628 passed / 0 failed** incl. the
>   `hip_device_constructs_via_cuda_slot` test. (commit `b84e5a6`)
> - **45c ✅** — real `HipDevice` init (dlopen `libamdhip64`: device count / `hipSetDevice`) + on-device
>   memory: `HipStorage` owns a real allocation with working H2D/D2H/D2D transfers, `zeros`+`memset`,
>   `synchronize`. Gate: `hip_storage_host_device_roundtrip` — `Tensor::new(data,&gpu)` lives in VRAM and
>   reads back exact on the gfx1100. (commit `ec06b23`)
> - **45d matmul ✅** — rocBLAS added to the `hip_backend` runtime (persistent handle);
>   `BackendStorage::matmul` is batched f32 GEMM mapping candle's `(b,m,n,k)`+layouts onto
>   `rocblas_sgemm_strided_batched` (per `cuda_backend::gemm_config`). Gate: `hip_matmul_matches_cpu`
>   (2D + batched 3D) matches CPU on the gfx1100. (commit `434515a`)
> - **45d elementwise ✅** — new AOT HIP-kernel pipeline: `vendor/candle-core/build.rs` compiles
>   `src/hip_kernels/elementwise.hip` (amdclang++ `--offload-arch=gfx1100`) → `libcandle_hip_kernels.so`,
>   dlopen'd by `runtime::Kernels`. `affine`, `unary_impl` (neg/exp/silu/gelu/sqr/sqrt/recip/relu/tanh/abs
>   on `B::NAME`), `binary_impl` (add/sub/mul/div), contiguous f32. Gate: `hip_elementwise_matches_cpu`
>   (affine/gelu/silu/add/mul) matches CPU on the gfx1100. (commit `5f85eba`) — **4/4 HIP tests green.**
> - **45d decode ops ✅** (parallel subagents, integrated + re-verified) — `reduce_op` (sum multi-dim /
>   max,min single-dim), `to_dtype` (F32↔F16/BF16/U32/I64, F16↔BF16), `index_select` (U32/I64, embedding
>   gather), `where_cond` (U8/U32 mask), `copy_strided_src` (D2D fast path + rank-generic f32 unravel →
>   powers `.contiguous()`). Gate: `op_tests` 5/5 match CPU; candle-core + akh `--features hip` green.
>   (commit `c17cbde`) — **the core decode-path `BackendStorage` op surface is now on the GPU.**
> - **45e ✅** (two parallel subagents, integrated) — **`QHipStorage` (GGUF Q4_K)**: weights resident as
>   raw 144B superblocks; HIP dequant kernel bit-for-bit matches `BlockQ4K::to_float`; `fwd` dequants →
>   rocBLAS f32 GEMM. **f16/bf16 matmul** (rocBLAS hgemm + gemm_ex). Gate: candle-core `--features hip`
>   **15/15** incl. `hip_q4k_dequant_and_matmul_parity` (dequant exact, fwd rel 3.7e-7) + f16/bf16 matmul;
>   akh `--features hip` builds. (commit `2311a5a`)
> - **45f ✅ — the 12B runs FULLY RESIDENT on `Device::Hip`.** `gemma3-12b.gguf` loads + runs end-to-end
>   on the HIP device with **coherent output** ("Okay, let's break down photosynthesis, step by step…"
>   over 48 tokens). **0.17 → 0.87 tok/s (5× the op-offload baseline)**; **22.6 GB weights resident**
>   (Q4_K + Q6_K + f32 norms), zero weight re-upload. `select_device()` returns `new_cuda(0)` under `hip`;
>   `quantized_gemma3` routes rms_norm/softmax/rope to candle-nn's on-device *slow* variants when
>   `device.is_cuda()` (CPU path byte-identical). Residual ops the real forward surfaced, **all added +
>   parity-gated on gfx1100** (candle-core `--features hip` **16/16** op_tests + quant): unary cos/sin/log;
>   **strided** binary/unary/where_cond/cmp(u32) (broadcast layouts in rms_norm/softmax/rope/mask);
>   `copy2d` via `hipMemcpy2D` (cat/KV-cache); **argmin/argmax → u32** (greedy sampling). PCIe counters
>   added to the candle-core HIP runtime. (commits `c3b9165`, `731199b`, `3315e60`)
>   - **gemma4-12b too** (`894ba10`): same device-gated routing in `quantized_gemma4`; identical dtype mix
>     (F32+Q4_K+Q6_K, no new kernels); its GELU MLP / weightless-V rmsnorm / final-logit tanh-softcap /
>     per-block output-scale all route through ops already on the GPU. Resident on gfx1100 = **0.85 tok/s,
>     23.46 GB**, coherent output. Both on-host 12B akhs (gemma3 **and** gemma4, the June-2026 release) now
>     run their full forward on `Device::Hip`.
>   - **NOT YET the ≤2-crossings/token target nor roofline.** Measured **3710 H2D/token** — but these are
>     **tiny control-plane uploads** (the per-call dims/stride arrays of the strided ops, <40 B each), NOT
>     weight streaming (weights stay resident). **Profiling verdict:** the 1.15 s/token is dominated by
>     **per-op `hipMalloc`/`hipFree` + `hipDeviceSynchronize`** (~1900 ops/token, each synchronous, each
>     with its own scratch alloc) — *not* residency, *not* dequant (~10%). The architecture is proven; the
>     speed is an execution-model problem.
> - **45g (perf, partial) ✅ — pool + fused GEMV: 0.87 → ~1.26 tok/s (7.4× over op-offload baseline).**
>   **(a) Caching device allocator** (`35fa751`): `hipFree` *synchronizes the whole device*, so freeing a
>   temp after every op serialized the ~1900 ops/token. The HIP runtime now recycles freed blocks by exact
>   size (no `hipFree`, no sync). 1.47× alone, but the resident 12B already sits at ~23.5 GB so a big pool
>   spills to GTT → default cap a conservative 512 MB (`AKH_HIP_POOL_MB`). **(b) Fused quantized GEMV**
>   (`8d7ffcd`): decode is `b·m == 1`, yet `fwd` dequantized each weight to a 4×-larger f32 temp every
>   token. New `akh_gemv_q4k/q6k_f32` kernels dequant inline + accumulate the dot product — **VRAM 22.6 →
>   19–20 GB, per-gen Δ +5.5 → +0.5 GB** (f32 temps eliminated), headroom restored, both 12B akhs ~1.25
>   tok/s. Parity `hip_gemv_decode_parity` (Q4_K+Q6_K); candle-core `--features hip` **17/17**.
>   - **Remaining wall = per-op synchronization.** The GEMV proved dequant traffic was *not* the tok/s wall
>     (same 1.25 with/without the big pool). Two serial-sync sources dominate now: (i) ~1900 explicit
>     `hipDeviceSynchronize` (one per kernel wrapper); (ii) ~3998 **synchronous** metadata H2D/token (each
>     strided op uploads its dims/strides — a stream-stalling `hipMemcpy`). Both must go for async.
> - **45g/45h (perf, cont) ✅ — by-value metadata + async + coalesced GEMV: → 2.65 (gemma4) / 2.96 (gemma3) tok/s.**
>   **(1) by-value strided metadata** (`f391979`): the strided kernels (binary/unary/reduce/reduce_arg/ucopy)
>   uploaded dims/strides to a per-call device buffer via a *synchronous* `hipMemcpy` — ~4000 stream stalls/
>   token. Now passed as by-value launch params (`AkhI64x8`, rank ≤ 8). **PCIe crossings 4052 → 11/token**
>   (decode ~2; residual is one-time prefill) — the plan's metric. **(2) async** (`866686c`): rocBLAS shares
>   the null stream, so the per-kernel `hipDeviceSynchronize` (23 wrappers + 5 Rust) was redundant — removed;
>   decode syncs ~2×/token (input H2D, logits D2H). **(3) coalesced GEMV** (`4ea8933`): the naive
>   one-thread-per-row GEMV was non-coalesced (~88× off peak) — *the* wall once sync/metadata were gone. Now
>   one BLOCK/row, threads stride consecutive positions (coalesced) + shared-mem reduction → **1.49 → 2.65
>   (1.78×)**. candle-core `--features hip` **17/17** throughout.
>   - **The profiling story (each step disproved the prior hypothesis):** removing dequant traffic (GEMV)
>     didn't move tok/s → not bandwidth; removing 4000 H2D (by-value) barely moved it → not the crossings;
>     removing 1900 syncs (async) barely moved it → not sync. The actual wall was **GEMV memory-access
>     coalescing**. Measure, don't guess.
> - **45h (profile-driven) ✅ — GEMV profiler then fused RMSNorm: → 6.75 (gemma3) / 6.23 (gemma4) tok/s.**
>   **(a) GEMV profiler** (`a8d37f1`, `AKH_HIP_PROFILE=1`): timed the fused GEMV — **only 22.6% of decode**.
>   So the quantized matmuls were *not* the wall; the other 77% was ~2880 small elementwise kernels/token,
>   dominated by `rms_norm_slow`'s ~7-kernel decomposition (~2000 launches/token). This **redirected** the
>   work away from GEMV micro-opt. **(b) Fused RMSNorm** (`d92c429`): one `akh_rms_norm_f32` kernel (block/
>   row: reduce Σx², normalize, scale) as a candle `CustomOp2` (`candle_core::cuda::rms_norm_fused`); akh's
>   `GemmaRmsNorm` routes to it on the HIP device (gemma3 + gemma4), CPU byte-identical. **2.96 → 6.75 tok/s
>   (2.3×)**, decode 377 → ~160 ms/token, VRAM ~20 GB, coherent. Parity `fused_rms_norm_matches` (rel<1e-4);
>   candle-core `--features hip` **18/18**.
> - **Cumulative Phase 45: 0.17 (op-offload) → 6.75 tok/s = ~40× over baseline, ~5.8× over CPU (1.16).**
>   Both 12B akhs, VRAM 19–20 GB, coherent. ~5% roofline.
> - **45h per-category profiler ✅** (`99b287e`, `candle_core::cuda::op_profile`): split decode (gemma3, 32 tok):
>   **GEMV 43.5% · attention-matmul (rocBLAS) 37.5% · elementwise (rope/softmax) 10.5% · strided-copy 5% ·
>   other ~3.5%.** The **surprise: the attention rocBLAS is 37.5%** — ~686 µs/call for *tiny* decode GEMMs
>   (m=1, head-batched) = pure rocBLAS small-batch overhead, not compute. And rope/softmax are only 10.5%, so
>   **fusing them is NOT worth it** (the profiler saved that work). 
> - **45h custom decode-attention kernel — BUILT, PARITY-PASSED, MEASURED ZERO GAIN, REVERTED.** Followed
>   lever (1): added a custom `m==1` f32 batched-GEMV (`akh_attn_m1_f32`, block-per-output, reduce over k
>   on candle's *logical* strides) inside `HipStorage::matmul`, bypassing rocBLAS for the two decode
>   attention GEMMs. Bit-exact vs CPU (new `matmul_decode_attention_parity` covered q·kᵀ *transposed-rhs* and
>   scores·v *contiguous-rhs*; 16/16). A/B with a runtime toggle (`AKH_HIP_ATTN_KERNEL`): **short ctx
>   custom 6.63 vs rocBLAS 6.66; long ctx (~3500 seq_kv) custom 3.28 vs rocBLAS 3.27 — within noise both
>   directions.** So replacing rocBLAS changes **nothing**. Reverted (git keeps it: a kernel that does
>   nothing in the default build doesn't earn tree space).
> - **THE LESSON, sharpened (4th time a profiler-named bottleneck was a mirage):** under `AKH_HIP_PROFILE=1`
>   the custom kernel *still* measured ~671 µs/call — identical to rocBLAS. On a single ordered (null) stream
>   the sync-bracketed per-category split is **not GEMM compute** (the attention compute is microseconds);
>   it's launch/queue/drain latency the bracketing sync attributes to whichever op it wraps. **Per-category
>   sync-profiling is unreliable for finding the real bottleneck in an async pipeline.** The 37.5% was never
>   addressable by a faster attention kernel.
> - **45h event-based GPU profiler ✅ — the reliable instrument; GEMV is the real wall** (`0b45cb5`). Built
>   `AKH_HIP_PROFILE=event`: hipEvent stream markers around each op → `elapsed(start,stop)` is the **pure GPU
>   time** of that op, no device drain, no mis-attribution. Pooled events (recycled, flush one sync/4096
>   pairs); GEMV timed at the kernel launch (clean window); + `hipMalloc` instrumentation (calls/misses/wall).
>   **Honest decode map (gemma3-12b, gfx1100):**
>   - **short ctx:** GEMV **46.5%** (218 µs/call) · attention 27% · eltwise 4.7% · copy 2.2% · malloc 0.8% · gap 19%
>   - **long ctx (~3500 seq_kv):** attention 36% · GEMV 21% · copy 7.8% · malloc 6.9% · gap 29%
> - **Findings (data, not guesses):** (1) **malloc-sync REFUTED** as the bottleneck (0.8% short ctx; this
>   ROCm `hipMalloc` is ~3 µs, not a device sync — my own theory, killed by measurement). (2) **The #1 general
>   lever is GEMV:** it streams all 7.3 GB of weights/token in 73 ms = **~99 GB/s — only ~10% of the 960 GB/s
>   roofline.** Context-independent (weight-streaming), so it helps every prompt. (3) Attention's share is a
>   *whole-op* window (includes host enqueue gap) — consistent with the custom-kernel A/B showing zero gain
>   (its real GEMM time is small). Long-ctx attention/copy/malloc growth is real KV-cache work.
> - **Next focused pass — GEMV → bandwidth roofline.** Cooperative-superblock dequant (stop recomputing
>   d/dmin/scales per thread; one superblock per warp, vectorized byte loads), higher occupancy (rows/block).
>   At ~10% of BW there's ~1.7× of decode to recover. Long-ctx: a flash-attention-style fused KV kernel
>   (reads KV once) only if long prompts matter. The model is **usable now (6.75 tok/s)** and the profiler now
>   makes every further step *measured*, not guessed.
>
> _(historical) Next: 45g/45h (the speed work). In priority order: **(1)** stop the per-op `malloc/free` +
>   metadata H2D — pass strided-op dims/strides **by value** (kills the 3710 crossings → ~2/token, the
>   plan's metric) and pool scratch buffers; **(2)** **streamed execution** — drop per-kernel
>   `hipDeviceSynchronize`, sync once per token (the default stream is ordered) — removes ~1900 serial
>   syncs/token; **(3)** **fused quantized GEMV** for decode (m=1): dequant-on-the-fly inside the matmul so
>   only the 7.3 GB Q-weights stream (no per-token f32 materialization); **(4)** fused rms_norm/rope/softmax.
>   Then **45g** hidden-state cosine ≥ 0.999. This is what turns 0.87 → roofline.
- **Target box:** AMD Radeon RX 7900 XTX (`gfx1100`, 24 GB, ~960 GB/s), ROCm 6.3.4 userspace installed.
- **Supersedes the perf ceiling of:** [plan 2026-06-04-gpu-rocm-candle.md](2026-06-04-gpu-rocm-candle.md)
  (op-offload — *complete*, but PCIe-round-trip-bound).

> **ADR numbering note.** The task referenced "ADR 053". On this branch ADR **053** is
> `first-class-akh-config`; the GPU ADR is **068** (`rocm-candle-gpu-backend`). "053" is the
> *historical* number — stale links remain in `src/nlu/gpu.rs` and the 2026-06-04 plan header and are
> corrected as part of 45j.

## Problem

The op-offload GPU path (plan 2026-06-04, ADR 068) is **complete and validated**: Gemma 12B and 27B
generate on the gfx1100 via resident Q4_K weights + rocBLAS GEMMs + bespoke HIP kernels
(`akh_dequant_q4`, `akh_dequant_q4k`, `akh_silu_mul`), persistent rocBLAS handle, fused MLP, shared
qkv upload, hidden-state parity (cosine 0.9988). **But it is not fast**, and cannot be made fast in its
current shape, because only GEMMs run on the GPU while every non-linear op (RMSNorm, RoPE,
softmax/attention, KV-cache, residual-add) stays in Candle **on the CPU by design**. The activation is
therefore downloaded and re-uploaded around every GEMM:

- Measured data flow (per transformer layer, single-token decode): **~8 PCIe crossings/layer** →
  **~216 crossings/token** for a 27-layer 12B. The bus, not the GPU, is the bottleneck.
- There is **no tok/s instrumentation anywhere** — the only numbers are wall-clock for the 3-route
  example. We do not yet have a baseline.

## Goal & decision

Build a **first-class Candle `Device::Hip` backend** so the *entire* forward pass runs on-device:
tensors live in GPU memory from the embedding gather through the final norm, with PCIe crossings
collapsed from ~216/token to **~2** (input ids in, logits/hidden out). Chosen over the narrower
"device-resident forward only" because it (a) lets candle-transformers' **stock** Gemma/Qwen models run
unmodified — retiring the vendored op-offload forks — and (b) unlocks **on-GPU training** for Phase
26/27 (Neural→VSA bridge, Burn/LoRA), which the op-offload path cannot.

### Acceptance bar — "close to hardware capability"

Decode is **memory-bandwidth-bound**: each token streams every weight once across ~960 GB/s. So the
honest "fast" metric is **% of the bandwidth roofline**, not raw tok/s.

| 12B dtype | weights | fits 24 GB | roofline | phase target (≥60% roofline) |
|---|---|---|---|---|
| f16    | ~24.4 GB | ❌ no KV/display room | ~39 tok/s | (fidelity path only) |
| **q8_0** | ~12.5 GB | ✅ ~11 GB headroom | ~77 tok/s | **≥45 tok/s** |
| q4_K_M | ~7.3 GB | ✅ ~16 GB headroom | ~131 tok/s | ≥78 tok/s |

### Measured baseline (45a — 2026-06-21, hephaistos gfx1100, `gemma3-12b` Q4_K)

`examples/candle_bench`, 128-token `Fixed`-mode decode (prefill excluded):

| path | tok/s | s/token | PCIe crossings/token | % of q4_K roofline |
|---|---|---|---|---|
| **op-offload GPU** (`AKH_GPU_MATMUL=1 AKH_GPU_QUANT=1`) | **0.17** | 5.95 | **419** | **0.1 %** |
| pure CPU (no GPU flags) | 1.16 | 0.86 | 0 | 0.9 % |
| **resident `Device::Hip`** (45f, `--features hip`, 48 tok) | **0.87** | 1.15 | 3710¹ | **0.7 %** |
| roofline (q4_K) | 131.5 | — | 0 | 100 % |

¹ The 3710 H2D/token are **control-plane** stride-array uploads from the strided ops (<40 B each), **not
weight streaming** — weights are resident (22.6 GB VRAM, never re-uploaded). 45g item (1) (by-value
strides) drops this to ~2/token. The 0.87 tok/s is bottlenecked on **per-op `malloc`/`free`/`sync`**
(~1900 synchronous ops/token), not residency or dequant — see the Progress block. The architecture is
proven correct (coherent 48-token output); 45g/45h is the execution-model speed work.

**The op-offload "GPU" path is ~7× *slower* than plain CPU for 12B** — each token does 419
device-synced PCIe round-trips (~8.7/layer × 48 layers), which costs far more than the single-token
GEMM offload saves. (The path was previously validated as *working*, never *measured*.) The
device-resident `Device::Hip` target (≥78 tok/s, q4_K) is therefore a **~460× speedup over op-offload**
and **~67× over CPU**. Output is coherent and byte-identical across both paths, so correctness holds —
this is purely a transport problem, exactly what the full backend removes. **This 0.17 tok/s is the
number Phase 45 must beat.**

**Implication of "12B doesn't need compression":** true for *VRAM* — but compression *reduces decode
bandwidth*, so it makes generation **faster**, not slower. Recommended 12B default is **q8_0**
(near-lossless hidden states for the VSA bridge, fits with headroom). f16 is supported in the backend for
max-fidelity paths and smaller models. The 24 GB headroom is spent on **KV cache / batching /
speculative decoding**, not on uncompressed weights. **Hard gate (every GPU stage): hidden-state cosine
GPU vs CPU ≥ 0.999 (f16) — protects the Phase 26 bridge.**

## Binding constraints

1. **Hidden states stay extractable.** `forward_with_hidden` returns the post-final-norm last-position
   vector. On a real device that is a tensor in GPU memory; we copy *one* vector to host (cheap), not
   per-op round-trips. Parity gate guards it.
2. **Candle-only in the hot path** (ADR 067). HIP C++ kernels compiled by `build.rs` are already accepted;
   no C++ *inference engine* (llama.cpp/Ollama) in the path.
3. **CPU stays the always-available fallback.** The `hip` feature off, or runtime absent ⇒ `Device::Cpu`,
   with a miette diagnostic (the `nlu::gpu` probe already does this).
4. **No upstream dependency risk.** Candle has no production HIP backend — only WIP
   [#3424](https://github.com/huggingface/candle/pull/3424) / [discussion #3327](https://github.com/huggingface/candle/discussions/3327)
   (`rocm-rs`, BERT-only, no k-quant). We seed *from* it where useful and contribute back, but vendor a
   pinned fork in-tree so we never block on upstream.

## Architecture

> **Implementation note (45b landed, 2026-06-21).** The original "add a `Hip` enum variant" plan would
> have required a `Hip` arm at **~130 match sites** in candle-core. Instead — since HIP is
> source-compatible with the CUDA runtime API — the HIP backend **fills candle's `cuda` *module slot***
> (`#[cfg(all(not(feature="cuda"), feature="hip"))] pub use hip_backend as cuda;` in `lib.rs` +
> `quantized/mod.rs`). Every existing `Storage::Cuda` / `Device::Cuda` / `QStorage::Cuda` dispatch then
> routes to AMD with **zero match-arm changes** — the "hipify the cuda backend" approach (ADR 068 step 3).
> `HipDevice`/`HipStorage`/`QHipStorage` are re-exported as `CudaDevice`/`CudaStorage`/`QCudaStorage`;
> `DeviceLocation::Cuda` labels the AMD GPU; the device is built with `Device::new_cuda(0)` (a
> `Device::new_hip` alias can be added for ergonomics). This is ~10× less code and the same full-device
> result. The text below reflects the original framing; the slot approach supersedes the per-variant arms.

Candle's `Device`/`Storage`/`DType` are closed enums → **vendor candle-core (pinned 0.10.2)** into the
workspace (`vendor/candle-core`, own `[workspace]` root) and `[patch.crates-io]` so `candle-nn` /
`candle-transformers` bind to the fork. The backend implements two traits:

- `BackendDevice` for `HipDevice` — alloc/free, H2D/D2H, streams, events, RNG, `synchronize`. This is the
  existing `nlu::hip` runtime layer (dlopen `libamdhip64`/`librocblas`, persistent handle, `DeviceBuffer`)
  promoted to the device role.
- `BackendStorage` for `HipStorage` (owns a `DeviceBuffer`) — the op kernels.

**Kernel surface, decode-critical first** (everything else `unimplemented!` with a clear message — conv/
pool/upsample are never hit by transformer decode):

| Op group | Source |
|---|---|
| `matmul` | rocBLAS / hipBLASLt — **have** (`RocmBlas::sgemm*`) |
| `unary_impl` (silu, gelu, exp, neg, sqr, sqrt, recip…) | AOT HIP, templated over dtype |
| `binary_impl` (add, mul, sub, div) | AOT HIP — residual/scale |
| `affine`, `to_dtype` (f16↔f32↔bf16) | AOT HIP |
| `reduce_op` (sum, max) | AOT HIP — softmax/rmsnorm reductions |
| `index_select` / `gather` | AOT HIP — embedding gather |
| `copy_strided_src` / `copy2d` / `contiguous` | AOT HIP — transpose materialization |
| `where_cond` | AOT HIP — causal mask |
| Quantized `QHipStorage` (Q4_K, Q6_K, Q8_0) | extend `akh_dequant_q4k` + rocBLAS — **partly have** |
| **fused (speed):** rms_norm, rope, flash-attention + on-device KV-cache, residual | AOT HIP (Phase 45h) |

Stock candle-transformers models run on `Device::Hip` the moment the base ops exist (slow, generic);
the fused kernels close the gap to the roofline.

## Sub-phases

Each is independently buildable and gated by a runnable check. **[CPU]** = buildable/authorable on any
box (HIP device simply never constructed); **[GPU]** = gate must run on hephaistos.

- **45a — Baseline & roofline harness [CPU build / GPU measure].** Add tok/s + PCIe-crossing-count
  instrumentation to the generate loop; ship `examples/candle_bench.rs` (or `akh bench llm`) printing
  tok/s and % of the 960 GB/s roofline. Record op-offload baseline (12B q4_K, 27B) + CPU.
  *Gate:* bench runs, prints baseline; numbers committed to this doc.

- **45b — Vendored candle fork + `hip` feature scaffold [CPU].** Vendor candle-core/candle-nn @0.10.2,
  add `Device::Hip` + `HipDevice`/`HipStorage` skeleton (ops `unimplemented!`), `[patch.crates-io]`,
  `hip` cargo feature. *Gate:* `cargo build --release --features candle-backend,hip` on a CPU box; full
  existing test suite green; `Device::new_hip(0)` errors cleanly without ROCm.

- **45c — Device/memory/stream layer [GPU].** Promote `nlu::hip` runtime into `HipDevice`: alloc/free,
  H2D/D2H, streams, events, persistent rocBLAS handle, extend `build.rs` kernel `.so`. *Gate:* alloc +
  round-trip + `synchronize` test on gfx1100.

- **45d — Core elementwise + matmul kernels (f32/f16) [GPU].** Implement the decode-critical
  `BackendStorage` ops above. *Gate:* per-op parity test vs CPU for each op, f32 **and** f16
  (candle's own test pattern).

- **45e — Quantized `QHipStorage` (Q4_K/Q6_K/Q8_0) [GPU].** Implement candle quantized storage + matmul
  for HIP from the existing dequant kernel + rocBLAS. *Gate:* dequant bit-exact vs candle CPU (extends
  `q4k_native_matches_candle`); quantized-matmul parity test.

- **45f — Stock model on `Device::Hip`, fully resident [GPU].** Load Gemma-3-12B via candle-transformers
  (or the now device-generic vendored model) on `Device::Hip`. *Gate:* coherent generation; output
  matches CPU within tolerance; instrumentation proves **≤2 PCIe crossings/token** (down from ~216).

- **45g — Hidden-state parity on the real device [GPU].** `forward_with_hidden` copies one device vector
  to host. *Gate:* `candle_hidden_parity --features hip` → cosine ≥ 0.999 (f16). **Hard stop on
  regression.**

- **45h — Fused kernels → push to the roofline [GPU].** Fused RMSNorm, RoPE, flash-attention with
  on-device KV-cache, fused residual, f16 KV cache; hipBLASLt GEMM; hipGraph capture of the static decode
  graph to kill launch overhead. *Gate:* **≥60% of roofline** for the chosen 12B dtype (q8 ≥ ~45 tok/s);
  A/B vs 45a baseline; hidden cosine still ≥ 0.999.

- **45i — dtype/precision policy [GPU].** Add f16/bf16 weight path; benchmark 12B f16 (paged KV) vs q8 vs
  q4_K on this card; pick per-model default (recommend q8 for 12B). *Gate:* recorded
  dtype × throughput × hidden-cosine table; default selected.

- **45j — Integration, retire op-offload, docs [CPU/GPU].** `select_device` prefers `Device::Hip` under
  `feature=hip` + runtime ready; gate or remove the `AKH_GPU_MATMUL` op-offload forks; write **ADR 073**;
  fix stale "053" links; update `reference_candle_gpu_amd.md`, runbook, and a **candle-rebase playbook**.
  *Gate:* `akh -w gemma chat` runs device-resident by default; full suite green; ADR + plan status updated.

**Parallelizable:** 45d and 45e are embarrassingly parallel — each op is an independent
*author kernel → compile → parity-test* unit. **Sequential spine:** 45a→45b→45c→(45d‖45e)→45f→45g→45h→45i→45j.

**Follow-ons (not in this phase):** upstream the HIP backend / k-quant kernels to candle #3424; on-GPU
LoRA for Phase 26/27 (the training unlock this scope buys).

## Risks & mitigations

- **Candle-rebase burden** (the standing cost of a fork). *Mitigate:* pin to the 0.10.2 tag; keep the HIP
  diff surgical and in clearly-marked vendored files; the rebase playbook (45j); upstream what we can.
- **Kernel correctness drift from CPU.** *Mitigate:* every op has a CPU-parity gate before it's used; the
  hidden-cosine gate is a hard stop.
- **Roofline target proves hardware-bound** (real backends hit ~50–65%, not 100%). *Mitigate:* the bar is
  *% of roofline*, set at 60%; if 45h plateaus below, escalate to re-scope the target rather than chase
  diminishing returns (the user asked for "close to hardware capability", which 60% of roofline satisfies).
- **f16 12B doesn't fit.** *Mitigate:* q8 is the default; f16 reserved for smaller/fidelity paths — decided
  in 45i, not assumed.

---

## Autonomous execution flow

A **gate-driven state machine** over 45a→45j. Each stage: `entry criteria → work → run gate → on green:
commit + advance; on red: diagnose + retry (≤3) → escalate to human`. The flow is autonomous *with a GPU
runner* — the honest constraint below.

### Two execution tiers

- **Off-GPU agent (any host):** vendoring, scaffolding, HIP C++ kernel authoring, Rust wiring, and
  **compile-only** checks (`--features hip` builds on a CPU box because the device is never constructed).
  Covers all of 45b and the *authoring* half of 45c–45e/45h + the docs of 45j.
- **GPU runner (hephaistos, RX 7900 XTX):** every **parity** and **perf** gate. Reached over SSH / a
  remote-trigger. This is the verification authority — nothing advances past a GPU gate without it.

### Gate commands (the machine-checkable contract)

```
build      cargo build --release --features candle-backend,hip            # [CPU] every stage
op-parity  cargo test  --release --features hip  hip_backend::            # [GPU] 45d/45e
device     cargo test  --release --features hip  hip_device::roundtrip    # [GPU] 45c
model      cargo run   --release --example candle_route   --features hip  # [GPU] 45f  (EXIT=0)
pcie       AKH_HIP_TRACE=1 candle_route | assert crossings/token <= 2     # [GPU] 45f
hidden     cargo run   --release --example candle_hidden_parity --features hip   # [GPU] 45g (cosine≥0.999)
perf       cargo run   --release --example candle_bench --features hip     # [GPU] 45a/45h (tok/s, %roofline)
```

### Driver

1. **Supervisor loop** walks the sequential spine. For each stage it dispatches the *authoring* work to an
   off-GPU agent, then submits the stage's gate to the GPU runner.
2. **Parallel fan-out for 45d/45e** — a `Workflow` over the op list (`matmul`, `unary`, `binary`, `affine`,
   `to_dtype`, `reduce`, `index_select`, `where_cond`, `q4k`, `q6k`, `q8`): each item is
   *author kernel → compile → GPU parity gate*, independent, retried in isolation. A red op blocks only
   itself; the stage completes when all ops are green.
3. **Green → commit + advance.** Each stage is one reviewable commit (`feat(phase45X): …`) on a
   `claude/phase45-hip-backend` branch off `develop`.
4. **Red → diagnose → retry ≤3 → escalate.** Retries feed the gate's stderr (compile error / parity diff /
   tok-s shortfall) back to the authoring agent.

### Hard stops & escalation (do **not** auto-retry past these)

- **Hidden-cosine regression** (< 0.999) at 45g/45h — breaks the Phase 26 bridge. Halt, surface to human.
- **Perf plateau** below 60% roofline after 45h's fused kernels — likely hardware-bound; escalate to
  re-scope the target, don't burn cycles.
- **Candle-rebase conflict** that changes backend trait signatures — needs a human call on fork strategy.
- **GPU host unreachable** — park the stage, notify, resume when hephaistos is back (GPU gates cannot be
  faked on CPU).

### Drivers available in this environment

- `/loop` (self-paced) as the supervisor for the sequential spine, with hephaistos as the gate runner.
- A `Workflow` for the 45d/45e op fan-out (deterministic, per-op isolation, parity gate per op).
- The project's own **DelegationManager** (Phase 36, local Claude Code subprocess) once it lands — the
  native home for this on-host autonomy.

### Definition of done

`akh -w gemma chat` answers from a **fully device-resident** 12B forward at **≥60% of the memory-bandwidth
roofline** (q8 default), **≤2 PCIe crossings/token**, **hidden-state cosine ≥ 0.999** vs CPU; stock
candle-transformers models run on `Device::Hip`; the op-offload forks are retired or gated; ADR 073 + this
plan's status + the rebase playbook are committed.
