# Native Vulkan for the existing Candle Qwen35 runtime

Date: 2026-09-14. Scope: in-process Rust Vulkan, original Q4_K/Q6_K packed
weights, original Qwen35 graph, no llama.cpp runtime or model substitution.

## Finding

Current local defaults: bounded queue submission, shared command encoders and
compute passes, block GEMV and fused weighted RMS. Full-model numerical,
throughput, resident-idle and isolated four-daemon gates passed before promotion.
The latest matched counting run measured 11.061/13.148 decode steps/s with exact
text; settled resident idle was 0.267% / 11.75 W from a 10.667 W baseline.
This is progress, not equivalence with native HIP (~19.7) or the original runner
(30.731 steps/s). HIP settings remain unchanged.

Reference escapes are `AKH_VULKAN_BATCH_SUBMIT=0`, `AKH_VULKAN_BATCH_ENCODER=0`,
`AKH_VULKAN_BATCH_PASS=0`, `AKH_VULKAN_GEMV_BLOCKS=0` and
`AKH_VULKAN_QWEN35_FUSED_RMS=0`. Encoder-off also disables default pass batching;
explicit pass-on/encoder-off is rejected. Convolution remains opt-in: its public
and full-model numerical/idle tests passed, but whole-model speed changed only
slightly despite a large isolated-kernel improvement.

Retained compute passes were bit-identical across all 33 model positions
(162.16 s). Their combination with block GEMV passed the same full-model
cosine/token gate (152.62 s), though arithmetic values were not bit-identical.
Both immediate and deferred validation failures prevent unfinished output reads;
fresh devices recover. The latest private four-daemon run passed real tools,
approval-before-effects, durable receipts, and session/runtime restart recovery.
It was not a shared homelab cutover. The following entries record earlier stages.

Shared command encoders are now enabled by default. Their
33-position full-model comparison is bit-identical to the existing Vulkan route
(150.83 s), and preliminary core throughput is 5.540/6.154 steps/s. An independent
Vulkan-versus-HIP-f32 comparison also passed all 33 positions (141.06 s): minimum
printed hidden cosine 0.999999999, logit cosine rounded to 1.000000000, and all
greedy tokens matched. Cross-backend values are not bit-identical. This comparison
explicitly excludes HIP Q8 activation quantization and the rejected HIP recurrence
fusion. The no-compilation encoder repeat measured 5.485/6.127 steps/s with
exact text; settled resident idle passed at 0.433% / 12.15 W from an 11 W
baseline. The four-daemon Vulkan check passed authentication, approval-before-
effects, real write/read tools and receipts, pending-approval recovery after a
session restart, and conversation recovery after a runtime restart/artifact
reload. This used private loopback services, all stopped and cleaned up by the
harness; it was not homelab deployment.

A native packed-weight compute implementation is practical. The initial isolated
probe at `experiments/native-vulkan` implements actual Vulkan compute through
wgpu 29.0.4, with no external shader compiler or inference subprocess. Hardware
numerical validation passed all eight format/shape cases during an idle GPU
window coordinated with the HIP investigation. A subsequent implementation now
adds first-class, feature-gated Vulkan Device/Storage/QStorage to vendored Candle,
coexisting with HIP in one executable. Public tensor parity and exact-model
short-turn inference have passed. Full-model logit/hidden parity is now covered by
the comparisons above; overall performance acceptance remains pending. Neither
a projection kernel nor matching short replies establish production equivalence.

The exact Ptah model loaded in 34.305 s and returned `Hello!` twice for the same
19-token prompt. Resident idle was 2.9%/18.5 W. Subsequent local work added
cooperative 5120-wide sums and four-activation packed-prefill reuse, with hardware
parity for signed/squared/offset/strided reductions and Q4_K/Q6_K batch tails
1/2/3/4/5/19. Release sum timings improved from roughly 1054/828 microseconds
(one/19 rows) to 105/105 microseconds. Exact-weight 19-token projection medians
are now about 4.10 ms Q4_K ffn_up and 4.67 ms Q6_K ffn_down; these isolated
timings are not end-to-end throughput. Matched 128-token runs subsequently
produced the exact expected counting response at about 3.64 warm decode steps/s.
A 30-second settle/60-sample resident-idle observation passed at 0.783% / 12.68 W;
an earlier shorter observation missed its utilization threshold and remains
recorded in the implementation ledger.

An opt-in blockwise single-token Q4_K/Q6_K kernel
(`AKH_VULKAN_GEMV_BLOCKS=1`) passed the full-model **Vulkan reference versus
Vulkan candidate** gate at all 33 forced positions: hidden/logit cosine rounded
to 1.000000000 and all greedy IDs matched. This is not Vulkan-versus-HIP parity.
Actual-weight projection medians improved (FFN up 780.7 -> 681.9 us, FFN down
990.4 -> 680.4 us, vocabulary output 6074.5 -> 3267.1 us), but a concurrent-build
full-model run measured only 3.056/3.286 steps/s. Its resident idle passed at
0.717% / 12.4 W. No overall speedup or default enablement is justified yet.

A separate opt-in queue-submission candidate (`AKH_VULKAN_BATCH_SUBMIT=1`)
groups up to 64 command buffers, preserving ordering and forcing incomplete
tails to submit before synchronization/readback. Counterbalanced public-tensor
microbenchmarks measured 9.921/11.177 ms immediate versus 6.560/6.446 ms batched
for 256 dependent tiny operations. Ordering checks include copies, incomplete
batches, explicit fences and two callers on one device; all passed, as did the
broader public tensor and packed-weight suites. Full-model numerical/idle/speed
gates are pending; both new optimizations remain disabled by default.

Subsequent full-model batching validation passed all 33 hidden/logit/greedy
positions (151.16 s). A no-compilation matched counting comparison measured
3.542/3.769 decode steps/s immediate versus 4.487/4.258 batched, with identical
text and block GEMV disabled. Batched resident idle was 0.867% / 13.817 W.
The probe now settles before collecting its pre-load power baseline as well.

Weighted RMS normalization now has a GPU-only, out-of-place fused candidate
behind `AKH_VULKAN_QWEN35_FUSED_RMS=1`. Its independent f64 oracle checks
strides, offsets, width boundaries, 65537-row dispatch tails, immutable inputs,
and invalid inputs. All passed with maximum observed absolute error 2.8e-7.
Full-model reference/candidate parity with batching enabled also passed all
33 positions (156.87 s), with cosine rounding to 1.000000000 and matching
greedy tokens throughout. Throughput and resident-idle validation of this
combination are pending; no default enablement or overall parity is claimed.

The later native completion/idle validation and default promotion are recorded
at the top of this document and in the implementation ledger; the preceding
candidate-stage statements describe the earlier test sequence.

wgpu is a native Rust API with a Vulkan backend, and accepts WGSL internally;
its `vulkan` and `wgsl` features suffice for this experiment. These are first-party
documented capabilities, not a WebGPU browser dependency. [wgpu documentation](https://docs.rs/wgpu/29.0.4/wgpu/)

## What exists locally and upstream

The local cache contains ash 0.38, wgpu 29.0.4, Naga 29.0.4, Burn 0.21, and
CubeCL 0.10. The root already vendors Candle 0.10.2 with a HIP implementation
occupying Candle's CUDA backend slot. Inspection of local `device.rs`,
`storage.rs`, and `quantized/mod.rs` initially found CPU/CUDA/Metal enum dispatch with no
Vulkan variant; the local implementation now adds Vulkan. Current upstream Candle's Device enum has those three
variants. [Upstream Device source](https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/device.rs)

Ash exposes Vulkan bindings with more direct resource and synchronization
control; wgpu handles validation, resource ownership, and WGSL compilation.
The probe chooses wgpu to establish numerical correctness with a small
implementation. This does not establish its eventual inference performance.
[Ash documentation](https://docs.rs/ash/0.38.0%2B1.3.281/ash/)

Burn's WGPU backend is another available GPU tensor implementation, but using
it would still require a model/tensor bridge and exact packed GGUF kernels;
it does not make the existing Candle graph a Vulkan graph. This is an inference
from the distinct backend interfaces, not a claim that no Burn extension could
ever support it. [Burn WGPU documentation](https://docs.rs/burn-wgpu/0.21.0/burn_wgpu/)

## Probe and numerical contract

`src/q4k.wgsl` decodes Q4_K and Q6_K directly from an array of packed u32 words.
It retains original block bytes on the device: Q4_K uses 144 bytes per 256
weights; Q6_K uses 210, so the shader explicitly handles alternating half-word
alignment. The last buffer alone is padded to four bytes. CPU quantization and
dequantization come from the existing vendored Candle `GgmlType` implementation.
These exact layouts and scale extraction are sourced from:

- `vendor/candle-core/src/quantized/k_quants.rs`: BlockQ4K, BlockQ6K, and their
  `to_float` implementations.
- `vendor/candle-core/src/quantized/utils.rs`: `get_scale_min_k4`.

One 64-thread workgroup computes each output row, accumulates f32 partial sums,
and reduces them in workgroup memory. Activations remain f32. This matches the
mathematical dot product of the packed weights' decoded values; it does not
promise bitwise agreement with Candle CPU's Q8_K-quantized activation dot path.
The oracle is Candle's f32 dequantization followed by a f64 dot product.

The executable tests both formats at shapes 1x256, 7x512, 64x2048, and 127x4096,
with signed, nonconstant input and weight values. Its threshold is
`abs_error <= 0.002 + 0.00002 * abs(reference)` per output and it rejects NaNs.
The adapter must report Vulkan and must not report CPU; there is no software
fallback success path. It prints adapter identity and maximum absolute error.

Build and CPU-only shader validation:

```sh
cargo run --manifest-path experiments/native-vulkan/Cargo.toml --bin akh-native-vulkan-probe -j 2 -- --validate-shader
```

Hardware validation, only during an agreed idle GPU window:

```sh
cargo run --manifest-path experiments/native-vulkan/Cargo.toml --bin akh-native-vulkan-probe -j 2
```

Validation on 2026-09-14: build, CPU-only WGSL validation, and all eight hardware
cases passed. Adapter: AMD Unknown (RADV GFX1100), discrete GPU, PCI
`0000:03:00.0`, vendor 4098, device 29772, Mesa
`23.2.1-1ubuntu3.1~22.04.4`, Vulkan; reported subgroup sizes 32–64.

| Format | Shape | Maximum absolute error | Packed bytes |
|---|---:|---:|---:|
| Q4_K | 1x256 | 0.00000026 | 144 |
| Q4_K | 7x512 | 0.00000347 | 2,016 |
| Q4_K | 64x2048 | 0.00001955 | 73,728 |
| Q4_K | 127x4096 | 0.00002430 | 292,608 |
| Q6_K | 1x256 | 0.00000210 | 210 |
| Q6_K | 7x512 | 0.00000738 | 2,940 |
| Q6_K | 64x2048 | 0.00001365 | 107,520 |
| Q6_K | 127x4096 | 0.00002006 | 426,720 |

This is synthetic format validation, not proof against actual model tensors,
full inference, or a speed benchmark. No throughput number is justified yet.

## First-class Candle implementation and evidence

The `vulkan` feature adds distinct `Device::Vulkan`, `Storage::Vulkan`,
`DeviceLocation::Vulkan`, and `QStorage::Vulkan` variants; it does not occupy
the HIP/CUDA slot. `Device::new_vulkan(ordinal)` selects only hardware Vulkan
adapters. Both `hip` and `vulkan` compile into the isolated test executable.
Unsupported operations return named errors. Custom and in-place custom operations
have Vulkan hooks that default to unsupported, preserving an explicit extension
point without invoking CPU implementations.

Implemented operations use GPU buffers and compute: f32 unary/binary/affine,
comparisons and masks, reductions including argmin/argmax, broadcast and strided
copy, dense batched matmul, index selection, integer index/mask storage, and
quantized upload/dequantization/GEMV/batched projection/embedding row extraction.
Q4_K/Q6_K embedding lookup decodes only requested rows. Invalid indices set a
GPU error flag; host checking of that flag reports an error. The explicit
CPU-source `QTensor::quantize_onto` performs its documented CPU-source quantization
before upload; no Vulkan tensor operation uses CPU numerical fallback.

Public API parity command:

```sh
cargo run --manifest-path experiments/native-vulkan/Cargo.toml --bin candle -j 2
```

Initial hardware result: upload, transpose/materialization, offset affine, SiLU,
broadcast, sum, max, batched dense matmul, index_select, masks/where, zeros,
both formats' dequantization, native quantized embedding lookup, and batched
QMatMul passed. Invalid indices returned errors. Maximum projection absolute
error was `7.63e-6`; dequantization and embedding values matched exactly. A later
audit found and fixed storage-copy layout preservation and large dispatch grids.
The final hardware rerun passed strided/offset copy, argmax, a 4,194,305-element
operation spanning two dispatch dimensions, and a 65,537-row packed projection
with a padded final dispatch row. The combined HIP+Vulkan executable completed
successfully; this subsecond test duration is not an inference benchmark.

The exact installed model was inspected with the metadata-only executable,
using its original blob path ending
`3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`.
It contains 360 F32, 439 Q4_K, and 67 Q6_K tensors. The largest packed tensor,
`output.weight`, is 1,042,944,000 bytes. Device allocation checks enforce the
actual negotiated Vulkan storage-buffer binding limit, measured at
2,147,483,644 bytes on this adapter, above the largest tensor. Output projection has
more than 65,535 rows, so dispatch now spans two dimensions with explicit tail
guards and flattening helpers.

Current performance limitations are deliberate and unmeasured: dense matmul
and ordinary reductions are straightforward reference kernels; there is a submission
per operation, pipelines specialize on shape/layout, and index-bounds validation
reads back one flag. Ordinary dispatch does not wait for GPU completion, while
explicit readback and synchronize do. Full-model acceptance must precede any
runtime support or performance claim.

### Shared GGUF loader and sampling follow-up

The first exact-model attempt exposed that GGUF's `TensorInfo::read` calls the
shared `ggml_file::qtensor_from_ggml` path internally. That path now uploads
Vulkan packed bytes directly before creating a typed host slice, checking the
expected byte count and block-size alignment. A public GGUF write/read fixture
passes for F32, F16, BF16, Q4_K, and Q6_K: loaded packed bytes are identical,
dequantized values match exactly, and truncated raw inputs return errors.

Contiguous last-axis f32 argmin/argmax of width >=256 now uses a cooperative
256-lane workgroup (64 lanes on adapters with smaller workgroup limits).
Equal values choose the first index. First NaNs remain sticky, later NaNs are
ignored, and bitwise NaN checks avoid dependence on floating-point fast-math
assumptions. The generic strided/short-row path was also corrected to initialize
from the first element. Hardware parity passed widths 7, 255, 256, 257, 4097,
and 248,320, including infinities, ties, all NaNs, signed zero, offsets, and
noncontiguous layouts.

The isolated public `Tensor::argmax` plus u32 readback benchmark (25 measured
runs following five warmups, one 248,320-value row) improved from median
12,803.2 us / p95 14,031.7 us to median 1,010.6 us / p95 1,093.9 us, about
12.7x at the median. These are kernel-path measurements, not model throughput.
Reproduce with `cargo run --manifest-path experiments/native-vulkan/Cargo.toml
--bin argreduce -j 2`; `-- --validate-shader` validates WGSL without a GPU.

## Work required for the model

Local `vendor/candle-core/src/backend.rs` defines the storage/device contract;
`src/nlu/vendored/quantized_qwen35.rs` shows the actual graph. A production path
must address all of the following, with unsupported operations returning errors:

1. Device and storage representation, buffer lifetime, upload/readback,
   synchronization, layout/offset handling, dtype dispatch, and allocator limits.
2. QStorage/QTensor/QMatMul dispatch for both packed formats, including batched
   prefill as well as single-token GEMV. Actual GGUF tensor verification must
   precede model-level numerical claims.
3. Embedding/index selection, strided materialization/copy/cat, broadcast binary
   operations, reductions, comparisons/masks, affine/unary operations, casts,
   and dense/batched matmul for attention and gated delta recurrence.
4. RMS normalization, softmax, rotary trig/application, SiLU/sigmoid, and custom
   op handling. Existing HIP accommodations in Qwen35 avoid some CUDA-only
   custom operations, but must not accidentally misclassify Vulkan as CUDA.
5. Persistent attention cache and recurrent state, token-by-token and prefill
   parity, long-context tests, allocation-bound checks, and backend selection
   that reports Vulkan accurately.

The prototype uploads weights once per case and reads back once, but its row
kernel repeats scale decoding and uses a simple workgroup reduction. High
throughput will likely require tiled prefill kernels, subgroup-aware reductions,
cached pipelines/bind groups, reusable buffers, batched submissions, and minimal
host synchronization. Those are performance hypotheses requiring measurement.
Large model tensors also require explicitly negotiated device storage-buffer
limits or sharding; the probe's default limits are not suitable evidence that
every Qwen35 tensor fits.

The next acceptance milestone is full-model hidden/logit parity and matched
throughput against HIP. Experimental short-turn inference is operational, but
production equivalence, long-context compression and performance are not established.
