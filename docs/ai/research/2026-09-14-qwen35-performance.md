# Qwen35 HIP bottlenecks and matched baseline

Current fusion gate status: the opt-in `AKH_HIP_QWEN35_FUSED_DELTA=1` candidate
passes isolated scalar/HIP primitive tests but **fails full-model hidden-state
parity** at forced decode step 3 (cosine 0.999899431 versus required 0.9999).
It remains disabled by default and is not approved for deployment. Investigate
reduction/FMA ordering and Q8 activation sensitivity without weakening the gate.

The expanded 33-position, single-resident-model comparison now isolates the
interaction (113.48 s including hash/load). With Q8 disabled, fused versus
primitive minimum logit/hidden cosines both round to 1.000000000 and every greedy
ID matches. With Q8 enabled, minimum logit cosine is 0.999386782, minimum hidden
cosine 0.996354179 (worst step 29); every greedy ID still matches. Repeating each
primitive run produces bit-identical full hidden/logit vectors at all positions.
This rules out state leakage/nondeterminism in this fixture and supports
amplification of small arithmetic differences by activation quantization. It
does not establish the exact responsible floating-point operation. The unchanged
0.9999 gate still rejects the Q8/fusion combination; no runtime default changed.

This is a source/shape analysis of the exact Ptah GGUF
`3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`.
The initial analysis used source and metadata only; the later isolated argmax
measurement is recorded below. The root runtime smoke reported a
40.58-second model load, 19 prompt tokens and two output tokens with total
completion times of 1.18/1.01 seconds, and 14 W idle power. Those completion times
do not establish decode tokens/second or equivalence to another runtime.

## First experiment: the 16-row prefill boundary

The HIP quantized path uses fused batched GEMV for 2–16 activation rows. Above
16 rows it materializes the complete projection matrix as f32 and calls rocBLAS.
Evidence: `vendor/candle-core/src/quantized/hip.rs:46` (cutoff), `:791` (dispatch),
`:827` (dequantization), and `:832` (GEMM).

Reading the exact GGUF tensor directory gives 496 active block projections,
whose f32 expansions total **97,402,224,640 bytes per prefill** when that path is
selected. The largest individual expansion is 356,515,840 bytes. These are
cumulative expanded bytes, not simultaneous allocation or measured traffic.
The lm_head is excluded because the model projects only the final hidden row;
embedding, normalization, and unused MTP weights are excluded too.

Thus a 19-token prompt exercises a different kernel route from a 16-token prompt.
Measure 15, 16, 17, 19, and 32 repeated raw token IDs before attributing the smoke
latency to recurrent decode. Keep tokenizer/template overhead outside this test.
The later optimization candidate is quantized prefill GEMM/WMMA, as already
sequenced in [ADR 098](../decisions/098-candle-qwen35-rdna3-inference-stack.md)
and its [implementation plan](../plans/2026-08-24-candle-qwen38-rdna3-inference.md).

## Other costs established by source

| Cost | Evidence | Scope and implication |
| --- | --- | --- |
| Token-index roundtrip | `vendor/candle-core/src/quantized/mod.rs`, `QTensor::get_rows`; `quantized/hip.rs:477` | Indices already on GPU are downloaded, then uploaded for the embedding kernel. One avoidable D2H/H2D pair per forward, beyond input upload and greedy scalar readback. |
| Mask metadata uploads | `hip_backend.rs:3458`, strided `where_cond`; Qwen35 `masked_fill:88` | Four synchronous metadata uploads per full-attention layer; 64 per multi-token prefill. Main elementwise/reduction metadata already travels by value, so this is a specific remaining path. |
| Unfused recurrence | `quantized_qwen35.rs:21` and `:430` | Three batched rocBLAS matmuls plus decay/update primitives per recurrent layer/token. There are 48 recurrent layers: 144 small recurrence GEMMs per token, before projections. |
| Unfused convolution/gates | `quantized_qwen35.rs:332`, `:396`, `:413` | Four convolution taps become separate broadcasts/adds; L2 normalization, sigmoid, and softplus are decomposed. Many short kernels and intermediate allocations are candidates for fusion. |
| Growing KV copies | `quantized_qwen35.rs:273` | Each decode appends K/V using `Tensor::cat`, copying the previous cache. No paged/static KV seam is used here. |
| Materialized attention | `quantized_qwen35.rs:306` | Explicit scores, mask, softmax, and value matmul; no flash-attention path. This matters for both memory and longer-context latency. |
| Serial vocabulary argmax | `hip_kernels/elementwise.hip:298` | One thread per output element means a single logits row uses one active GPU thread to scan all 248,320 entries, including generic coordinate calculations. Production scalar sampling has this cost; the earlier Q8 parity benchmark instead copied logits and selected the maximum on CPU. |

These are candidate costs, not measured percentages. The recurrent state is
48 value heads × 128 × 128 f32 = 3 MiB per recurrent layer, or 144 MiB across
48 layers. A fused step/scan could keep each head's state near its computation
and avoid repeated state buffers, but must preserve tiled head ordering and
update-before-query semantics. An in-place version also needs checkpoint
copy-on-write/ownership treatment: current checkpoints share immutable tensor
storage, so mutating a live state buffer would otherwise corrupt rollback.

Measure argmax separately on an already resident vocabulary row. A specialized
parallel last-dimension reduction must preserve the first-index tie rule and
document nonfinite behavior before replacing the generic route. Its timing
explains a possible difference between historical full-vector parity benchmarks
and the new production scalar-readback path; it is not evidence of a measured
regression by itself.

## Long-context memory gates

Exact metadata has 16 full-attention layers, 24 query heads, four KV heads, and
256-wide keys/values, currently stored as f32.

| Structure | Size from exact shapes |
| --- | --- |
| K+V per cached token, all attention layers | 131,072 bytes (128 KiB) |
| K+V at 8,192 tokens | 1 GiB |
| K+V at 32,768 tokens | 4 GiB |
| K+V at 100,000 tokens | 12.207 GiB |
| K+V at 102,400 tokens (100 Ki tokens) | 12.5 GiB |
| One full-attention f32 score tensor at 8,192-token prefill | 6 GiB |

Do not run an 8K full-prefill or 100K context sweep on the current 24-GB target
as a routine smoke test. The 6-GiB score tensor is for a single layer and excludes
its mask/softmax temporaries, weights, expanded projections, allocator pool, and
other states. KV at 100K alone leaves too little room alongside the model.
Flash/chunked attention and bounded/compressed KV are prerequisites for those
acceptance runs, not optional speed tuning.

RoPE duplication is **not** a remaining cost: `from_gguf:975` constructs one
`PartialRope`, and attention layers use `rope.clone()` at line 938. Tensor clones
share storage. The two tables total 2 × 262,144 × 32 × 4 = 64 MiB per model,
not 16 copies. The 40-second load needs separate stage timing rather than a
speculative RoPE-sharing change.

## Opt-in diagnostic test

`installed_artifact_prefill_threshold_probe` is an ignored HIP-only test in
`src/nlu/vendored/quantized_qwen35.rs`. It requires the exact file fingerprint,
loads one model, and warms both routes at each shape. It then counterbalances
`AKH_GEMV_BATCH=0/1`, with Q8 decode fixed on, for three repetitions by default.
Each run prefills repeated token ID 9419, then performs eight forced decode
steps with that same token. This isolates shapes and recurrent accumulation;
it is not a conversational quality benchmark.

It reports separate synchronized prefill and decode wall times, PCIe call counts,
allocator calls/misses/miss time, end-of-stage VRAM readings by DRM card, and nine
greedy IDs. Full-logit and hidden-vector diagnostic transfers occur outside the
timed/counted stages. First-prefill and final-decode cosine must both reach 0.999,
and all greedy IDs must match across routes. End-of-stage VRAM is not a transient
high-water mark. Inputs, scalar greedy readbacks, and forced-step synchronization
are included in stage times; tokenizer and model load are excluded.

Only run after confirming exclusive GPU access:

```sh
AKH_QWEN35_TEST_MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec \
AKH_QWEN35_PROBE_REPEATS=3 \
cargo test --release --lib --no-default-features --features hip \
  installed_artifact_prefill_threshold_probe -- --ignored --nocapture --test-threads=1
```

Use unprofiled runs for timing. A separate `AKH_HIP_PROFILE=event` run can help
attribute kernels, with stage counter deltas. Existing `candle_bench` reports
decode-only wall time but reads accumulated operation/GEMV counters including
prefill; dividing these directly can misattribute prefill cost to decode.
Its matmul/elementwise event windows also include enqueue gaps, so category sums
are not an additive kernel-time partition. PCIe counters count calls, not bytes.

After this matched baseline, select one measured bottleneck: quantized prefill
GEMM, fused recurrence, or redundant transfer removal. Keep the existing f32
and ordinary-tensor reference routes for independent parity, then measure warm
counterbalanced runs before composing optimizations or claiming speedup.

## Isolated vocabulary argmax baseline

With an exclusive GPU window, the existing release HIP kernel was called through
its unchanged C ABI on one resident 248,320-f32 row. Values were generated as
`f32(((i * 48271) % 2147483647) / 2147483647)`, and an independent strict CPU
maximum selected index 44,488. Three warmups preceded 15 timed iterations.
Each timing includes kernel launch and a synchronous four-byte result readback:
median **48.956737 ms**, minimum 48.850006 ms, maximum 49.006103 ms. Every result
matched the CPU oracle. No model weights or service were loaded.

This establishes a material sampling cost, not whole-model speedup. An ignored
`hip_backend::op_tests::arg_reduce_vocabulary_probe` now covers argmin/argmax widths
1, 3, 31, 33, 255, 256, 257, 1,025, and 248,320 with earliest ties, signed zero,
infinities, first NaN, and later NaNs. It also checks the non-last-axis fallback
against Candle CPU and reports a separate end-to-end Tensor argmax microtiming.
The parallel path uses one cooperative block per contiguous last-axis row of at
least 256 elements; other shapes retain the generic kernel.

The parallel kernel passed that hardware parity probe and the existing
`argmax_last_dim` test. Repeating the identical direct-C-ABI measurement on the
parallel sidecar gave median **0.471861 ms**, minimum 0.301613 ms, maximum
0.741484 ms (15 iterations, same input and scalar-readback scope). This is about
104× faster for this isolated sampling stage; it is not a whole-model speedup.
The separate debug Tensor-API probe measured median 0.618377 ms, including
Tensor dispatch/allocation and readback. All results selected index 44,488.

Compile/run the persistent parity probe without enabling other model tests:

```sh
CARGO_TARGET_DIR="$PWD/target" cargo test \
  --manifest-path vendor/candle-core/Cargo.toml --lib --features hip \
  --no-default-features arg_reduce_vocabulary_probe \
  -- --ignored --nocapture --test-threads=1
```

The exact-model prefill/forced-decode probe still awaits its exclusive GPU run.

## Bounded prefill and state ownership foundation

The production Qwen35 `prefill`/`forward(tokens, 0)` path now uses query chunks
of at most 128 tokens. It runs the existing attention and recurrent operators
at monotonically increasing positions and applies the vocabulary projection
only to the final hidden row. `prefill_controlled(tokens, chunk_size, check)`
offers a Candle-result callback before each chunk and before that projection.
The entire non-empty prompt length and chunk size are validated against the
model context before mutation; execution/cancellation errors clear partial
state. A single rectangular causal-mask cache replaces the growing map of
all previous `(query_length, kv_length)` masks.

This bounds one 24-head f32 attention score tensor at 8,192 total tokens to
96 MiB for a 128-token query chunk, versus 6 GiB for a dense 8,192-token query.
It does **not** implement flash attention, bound total model memory, or compress
KV. KV still grows via tensor concatenation: the earlier 12.5 GiB KV estimate
at 102,400 tokens remains prohibitive. Dense scores still grow linearly with
cached context even with bounded queries, and snapshots can retain older KV
allocations. Do not treat this foundation as permission for a 100K smoke test.

Checkpoints now carry an opaque shared identity belonging to one loaded model.
Foreign or reloaded-model snapshots are rejected before restoration. Historical
same-model checkpoints intentionally remain valid across reset and cancellation,
supporting rewind and later session switching. The snapshots clone immutable
tensor handles; any future in-place recurrent/KV update must preserve snapshot
isolation. Live state is still held by `ModelWeights`; this change does not yet
move it into a runtime `InferenceSession` or reuse it across native requests.
The CPU-slice PolarQuant helpers are not connected to live attention KV; routing
through them would introduce host transfers and lossy state without established
parity. No live-compression claim is made.

CPU fixture coverage compares chunk sizes 1, 2, 3, 7 and 128 against the dense
seven-token oracle, including unequal tails, final hidden/logits, every attention
KV and recurrent/conv state, subsequent decode and checkpoint rewind. Separate
tests cover invalid input, cancellation at each boundary, snapshot isolation,
and foreign/reloaded-model rejection. Both new tests passed, followed by all
11 Qwen35 CPU-filtered tests (0.21 s); the two installed-artifact tests return
early without their opt-in environment, so this is not an exact-model oracle
run. `git diff --check` passed. Hardware chunk parity remains to be measured.

## Original reference runtime measurement (parent-coordinated)

The installed original reference binary was
`/home/toasty/.local/opt/llama.cpp-dflash2/bin/llama-server`, not the generic
Ollama build. The exact artifact loaded in 13.275 s with context 2,048,
`load-mode dio`, flash attention, Q4_0 K/V, all GPU, one slot, and no DFlash.
Generation used `cache_prompt=false`, temperature zero and seed 42. The raw
ChatML prompt had closed thinking and the user text:
“Write a numbered list of integers from 1 through 100, one line per number.
Do not summarize.” It encoded to 36 prompt tokens and produced 128 output
tokens, stopped by the length limit, containing integers 1 through 46.

First warm throughput was 30.815 tokens/s. Three further trials reported
29.557581, 31.059410 and 30.730768 tokens/s (median **30.730768**), with prompt
times 260.736, 220.448 and 227.976 ms respectively. Reference `predicted_ms`
counts 127 decode steps because the first output comes from prefill; do not
compare it to 128 divided by total elapsed time. A concurrent CPU Vulkan build
may perturb scheduling. This is the exact model, but not the former 100K-context
DFlash production maximum. The same 19-token hello prompt returned “Hello!”,
as Candle did. Post-generation idle was 99% GPU utilization and 129 W.
The reference process was stopped afterward and its boot service remains
disabled. Matched native throughput is pending.

## Opt-in fused HIP delta recurrence (not enabled by default)

`AKH_HIP_QWEN35_FUSED_DELTA=1` selects an inference-only HIP implementation
of the existing single-token delta step. This is a runtime process flag, not
a build-time setting. Without exactly `1`, the ordinary Tensor implementation
remains selected. CPU and Vulkan always retain that reference implementation.
The fused operation uses one 128-thread block per value row and supports key
widths 1 through 256 (the exact artifact uses 128). Each lane retains up to two
decayed state elements, reduces the prediction, computes the delta update,
writes fresh next-state storage, then reduces the query against **updated**
state. Existing state/checkpoint storage is never overwritten.

The public HIP interface validates contiguous f32 tensors, shapes and common
device, respects view offsets, and returns `(output, next_state)`. The model
adapter materializes small strided token views on-device before calling it.
Missing kernel symbols or launch errors fail closed; there is no CPU fallback.
The kernel uses only the default stream and rejects a configured graph stream.
It introduces no streams, graph capture, events or persistent GPU activity.
This optimization neither implements nor changes live KV compression.

The ignored `hip_backend::op_tests::gated_delta_step_parity_probe` checks an
independent scalar CPU oracle and the existing HIP primitive decomposition,
including odd key widths, distinct batch/head values, zero/unit gates,
nonzero initial state, nonzero storage offsets, 128 recurrent steps, input
immutability and invalid shapes/dtypes/devices. It also times the exact
48-head, 128-by-128 recurrence shape separately from model execution. Build
without running GPU tests:

```sh
CARGO_TARGET_DIR="$PWD/target" cargo test \
  --manifest-path vendor/candle-core/Cargo.toml --lib --features hip \
  --no-default-features --no-run --offline
```

Only after an exclusive GPU window is granted, run that binary with
`gated_delta_step_parity_probe --ignored --nocapture --test-threads=1`.
Hardware parity, isolated benefit, exact-model multi-token parity and matched
whole-decode throughput are gates before considering default enablement.
