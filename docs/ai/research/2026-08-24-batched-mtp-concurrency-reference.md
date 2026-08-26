# Batched MTP concurrency as a reference for Candle/RDNA3

- **Date:** 2026-08-24
- **Question:** Should Marshall Gould's eight-stream Qwen3.8/NInfer result become a
  follow-up reference design for Ptah's exact custom Qwen3.8-27B Q4_K_M Candle
  runtime on RDNA3?
- **Verdict:** Yes for the bounded scheduler and measurement contract; no for the
  CUDA kernels, quantization layout, or headline throughput target.
- **Related plan:** [Candle Qwen3.8 inference on RDNA3](../plans/2026-08-24-candle-qwen38-rdna3-inference.md)

## Evidence status

The [original post](https://x.com/MarshallJGould/status/2091582412522201130)
only claims roughly 600 tok/s for Qwen3.8-27B on one RTX 5090. A later
[author reply](https://x.com/MarshallJGould/status/2091884327814525315) identifies
an NVFP4 model, NInfer Turbo, MTP-3, eight simultaneous streams, and 12,288
output tokens. Those posts establish that the headline is aggregate concurrent
throughput rather than a documented single-stream result.

The detailed figures supplied in our conversation—733.0 tok/s aggregate
end-to-end, 767.9 sustained aggregate decode, 1,024.5 one-second peak, 39.14%
MTP acceptance, 462 ms mean TTFT, 19.73 GiB resident weights, and 95.5K–100.2K
live KV tokens—were attributed to the author but are not present in the public
post, current upstream documentation, or a linked raw benchmark artifact. Treat
them as **author-supplied but not independently reproduced**.

The arithmetic in that reply is internally consistent with NInfer's definitions.
NInfer defines committed tokens per speculative round as
`1 + accepted_tokens / speculative_rounds`. If all MTP-3 rounds draft three
tokens, `1 + 3 * 0.3914 = 2.1742` committed tokens per stream per round, or
17.3936 across eight full rows. High acceptance is helpful, but 95% acceptance
is not required for speculation to produce more than one committed token per
round. Whether that produces a net speedup still depends on draft, verification,
sampling, and scheduling cost. See NInfer's
[measurement definitions](https://github.com/Neroued/ninfer/blob/feaf4dd0983fdaeb2ba4c06eec6da350e644fb3a/docs/performance.md#single-request-serving-performance-method)
and NVIDIA's [speculative-decoding guidance](https://github.com/NVIDIA/TensorRT-LLM/blob/d9329fb8d3cd2d50807daafcc4d1d4e7a6473af0/docs/source/features/speculative-decoding.md).

One number remains ambiguous: 767.9 / 8 is 95.99 tok/s and 733.0 / 8 is
91.63 tok/s, neither 96.4. The reported 96.4 may be the arithmetic mean of
per-stream rates with different active lifetimes, but it needs a raw trace before
we use it.

## What upstream independently corroborates

NInfer's current first-party documentation is unusually useful here:

- Its Qwen3.8 artifact is 20.02 GiB and is not a simple uniform Q4 file. It uses
  mixed NVFP4 MLP weights and row-scaled FP8 for embeddings, attention, gated
  delta projections, the output head, and other weights.
- One engine owns one model and a startup-fixed capacity of one to eight active
  requests. Decode-ready rows are compacted at round boundaries and executed in
  one batched model traversal. KV capacity is a shared startup-sized pool, not a
  static equal partition among lanes.
- Its published saturated Qwen3.8 NVFP4 MTP-3 sweep rises from 143.8 aggregate
  tok/s at C1 to 766.6 at C8, a 5.33x rather than 8x speedup; C8 acceptance is
  46.0%. This strongly corroborates the *shape* of the supplied claim.

These facts are recorded in NInfer's commit-pinned
[README](https://github.com/Neroued/ninfer/blob/feaf4dd0983fdaeb2ba4c06eec6da350e644fb3a/README.md#concurrent-mtp3-decode)
and [runtime limits](https://github.com/Neroued/ninfer/blob/feaf4dd0983fdaeb2ba4c06eec6da350e644fb3a/README.md#current-limits).
NInfer is an independent CUDA project, not an NVIDIA product.

Most importantly, the headline is not representative of every serving workload.
NInfer's 75-request mixed corpus achieved its shortest makespan at C4. C8 fell
from 432.9 to 334.2 aggregate decode tok/s, increased makespan from 1,647.74 to
2,164.90 seconds, and averaged only a 2.36-row batch because memory pressure and
workload transitions prevented sustained occupancy. See the first-party
[mixed-corpus campaign](https://github.com/Neroued/ninfer/blob/feaf4dd0983fdaeb2ba4c06eec6da350e644fb3a/docs/performance.md#qwen38-27b-nvfp4-concurrent-mtp3-corpus-makespan).

## What transfers

| Transfer to Candle/RDNA3 | Why |
|---|---|
| One resident model with bounded C1/C2/C4/C8 decode microbatches | A batched traversal amortizes weight movement and launch overhead across ready rows. |
| Stable request IDs and row compaction at round boundaries | Finished, cancelled, or blocked sessions can leave without corrupting another session's KV, recurrent state, hidden taps, or speculative state. |
| One admission-controlled KV pool sized in tokens and bytes | Configured maximum context is not resident context; admission must use actual live pages across all sessions. Block allocation is also the core lesson of [PagedAttention](https://arxiv.org/abs/2309.06180). |
| Separate prefill and decode scheduling, with chunked prefill | Large prompts must not monopolize a round or starve active decode. NVIDIA's first-party [scheduler guide](https://github.com/NVIDIA/TensorRT-LLM/blob/d9329fb8d3cd2d50807daafcc4d1d4e7a6473af0/docs/source/legacy/performance/performance-tuning-guide/tuning-max-batch-size-and-max-num-tokens.md) makes the latency/memory trade-off explicit. |
| Two benchmark modes | A full-batch decode saturation wave measures the kernel ceiling; a mixed agent corpus measures the service users actually experience. |
| Committed-token accounting | Aggregate and per-stream committed tok/s, MTP acceptance, committed tokens/round, occupancy, TTFT, and KV pressure must be reported together. |

The statement that weights are "shared across the batch" is directionally right,
but should remain a hypothesis at the memory-traffic level. A batched kernel
applies a loaded weight tile to multiple sequence rows instead of launching a
complete independent traversal per request. It does not follow that every byte
is fetched from HBM exactly once per round: tiling, caches, occupancy, and the
Q4_K_M dequantization path decide actual traffic. Verify the benefit with ROCm
profiling and report HBM bytes per committed token.

## What does not transfer

- **The number 600, 733, 768, or 1,024 tok/s.** NInfer targets CUDA 13.1,
  Blackwell `sm_120a`, NVFP4/FP8 Tensor Cores, CUDA Graphs, and a purpose-built
  artifact. Ptah targets gfx1100, Candle/HIP, and the exact custom Q4_K_M model.
- **NInfer's kernels or artifact layout.** Its 20.02 GiB mixed NVFP4/FP8 artifact
  and Ptah's GGUF have different quantization, dequantization, packing, and
  hardware instructions.
- **C8 as the assumed optimum.** The upstream mixed-corpus result already shows
  that C4 can beat C8. Concurrency must be selected from a workload sweep.
- **The 131K configured ceiling as proof of eight long contexts.** It is an
  admission limit per request. The supplied trace says roughly 100K live KV
  tokens total, around 12.5K per stream near the end. This aligns with our plan:
  8K–32K is the normal physical working set and 100K is a pressure ceiling.
- **MTP as a prerequisite for batching.** First establish target-only C1–C8
  behavior. Add MTP counters and verification only after exact target correctness;
  speculative decoding can regress at larger batches, as NVIDIA also cautions.
- **A single saturated decode sample as a cutover gate.** It excludes queueing,
  prompt ingestion, drain, tools, vision, Virtual Context retrieval, and uneven
  generation lengths.

For Ptah, each batch row must retain its own full-attention KV pages, gated-delta
recurrent/conv state, sampling RNG, checkpoint/rewind state, image positions,
and selected VSA hidden taps. Batching may share immutable weights and execution;
it must never merge mutable session state.

## Recommended follow-up goal

Add **bounded, session-safe batched decode** after the exact single-stream Qwen3.8
reference forward and owned `InferenceSession` are correct. The RDNA3 adapter may
fuse row execution, but the public seam remains a collection of independent
sessions admitted into one round.

Acceptance gates:

1. Greedy C1 and C2/C4/C8 executions match the reference token stream; per-row
   logits/selected hidden states meet the existing cosine >= 0.999 gate, and
   cancellation, checkpoint/rewind, and row compaction pass adversarial
   two-session isolation tests.
2. A target-only C1/C2/C4/C8 saturation sweep reports aggregate and per-stream
   committed tok/s, TTFT, time per output token, actual batch occupancy, live KV
   tokens/bytes, recurrent-state bytes, VRAM high-water mark, and HBM bytes per
   committed token. One of C2/C4/C8 must deliver at least 1.5x C1 aggregate
   throughput without OOM or starvation to justify keeping the scheduler path.
3. A representative Ptah corpus mixes short/long prompts, uneven decode lengths,
   tool pauses, Virtual Context retrieval, and vision. Its selected concurrency
   must improve makespan by at least 25% over C1 while keeping p95 TTFT below 2x
   C1 and preserving per-session correctness.
4. Admission survives 100K **total live physical tokens** across active sessions
   without treating the configured 100K per-request ceiling as preallocated KV.
   Over-capacity requests queue or reject cleanly; they never partially corrupt
   another session.
5. When MTP-3 is later enabled, report target-only and MTP results separately,
   including acceptance, committed tokens/round, draft/verify time, rollback
   count, and end-to-end makespan. A one-second peak is diagnostic only, never the
   acceptance metric.

This is a worthwhile follow-up goal, but it belongs after correctness and session
ownership, not as a detour that delays the reference forward. The strongest idea
to copy is the experimental discipline: bounded concurrency, shared capacity,
and separate saturation from realistic makespan.
