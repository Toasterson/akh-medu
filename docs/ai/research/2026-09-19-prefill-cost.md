# Prefill at length: where the 271 seconds go

Date: 2026-09-19. HIP, pinned Qwen3.8 artifact, 10,417-token prompt. Evidence:
[`prefill-chunk-20260919-hip-v1`](../../../experiments/native-recall/prefill-chunk-20260919-hip-v1).

## The problem

The long-context soak measured prefill of 10,417 tokens at **271-298 s**, about
38 tokens/s, against 0.86 s for a 15-token prompt. Long context is unusable on
time well before it becomes unusable on memory, which the context gate now caps
at 10,816 tokens anyway.

Prefill runs in chunks, and the chunk size was hard-coded to 128, so that prompt
is 82 sequential forwards. Every chunk re-expands the whole accumulated K and V
through `repeat_kv` in each of the 16 attention layers, to lift the GQA kv-heads
to query-head count. That expansion is rebuilt from scratch each chunk, so its
cost scales as `(prompt / chunk) * positions` - halving the chunk count should
halve it.

## The measurement

| chunk | time to first token | prefill rate | outcome |
| --- | --- | --- | --- |
| 128 | 271.4 s | 38.4 tok/s | baseline |
| 256 | 205.3 s | 50.7 tok/s | **1.32x** |
| 512 | - | - | `hipMalloc(452984832)` failed |
| 1024 | - | - | `hipMalloc(503316480)` failed |

## The hypothesis was half right, and the half matters

Doubling the chunk predicted a 2x speedup if the expansion dominated. It gave
**1.32x**. By Amdahl, the fraction of prefill that scales with chunk count is

    1.322 = 1 / (1 - f/2)  =>  f = 0.49

So the repeated KV expansion is about **half** of prefill time, not nearly all
of it. The other half is per-token work that no chunking change will touch -
the quantized matmuls, which are weight-bandwidth bound and run once per token
regardless of how the tokens are grouped.

Had the sweep come back at 1.9x I would have concluded the expansion was the
whole story and chased only that. Had it come back at 1.05x the expansion would
have been a red herring. 1.32x says it is worth eliminating and worth about 2x,
not more.

**Eliminating the expansion entirely caps out at 1.95x - roughly 75 tok/s.**
That is the honest ceiling for this line of work: it makes a 10k prompt take
~140 s instead of ~271 s. Useful, not transformative, and it does not make long
context interactive.

## Larger chunks are bounded by memory, not by diminishing returns

512 and 1024 both fail on allocation. Peak memory during prefill scales with
`chunk * positions` - the attention scores for a chunk are computed against every
position seen so far - so the chunk size trades time against peak memory
directly. This is the same term that makes a context token cost 547.5 KiB
against 128 KiB of KV, and it is why a fixed memory reserve could not absorb it.

## What has not been changed

**The default is still 128.** The 1.32x at 256 was measured at exactly one
context length, and chunk size raises peak memory, which is the resource that
already broke the memory gate once today. Raising the default on one data point
would repeat the mistake this session has already made twice - generalising from
a single measurement that happened to be convenient.

`AKH_QWEN35_PREFILL_CHUNK` makes it selectable so the trade can be measured
across lengths before any default moves.

## The real fix, not attempted here

Avoid materialising the expansion at all: broadcast the kv-heads inside the
attention matmul rather than building an expanded copy. That removes the ~49%
term and reduces peak memory at the same time, since the expanded K and V are
themselves large transient allocations.

It is a change to the numerical path, so it needs the 0.9999 cosine gate against
the current implementation before it could be trusted - and per the standing
rule, not greedy-token agreement. Not attempted in this slice.
