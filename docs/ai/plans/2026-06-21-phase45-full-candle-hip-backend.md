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
> - **Next** — remaining decode-path ops on `Device::Hip`: `reduce_op` (sum/max → softmax/rmsnorm),
>   `to_dtype` (cast), `index_select` (embedding), `where_cond` (mask), `copy_strided_src` (contiguous),
>   then f16/bf16 matmul, **45e** `QHipStorage` (reuse `akh_dequant_q4k`), **45f** resident model,
>   **45g** hidden-state parity, **45h** fused kernels → roofline.
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
| roofline (q4_K) | 131.5 | — | 0 | 100 % |

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
