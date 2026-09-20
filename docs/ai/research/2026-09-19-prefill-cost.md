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

## Retracted: the hypothesis, the 49%, and the 1.95x ceiling

The table above is **single runs**. Today's chunk-128 baselines at this prompt
came in at 271.4, 282.3, 298.5, 302.0 and 316.9 s - a **17% spread**. A 1.32x
figure read off one run against another cannot survive that, and neither can the
49% share or the 1.95x ceiling derived from it. All three are withdrawn.

This is the rule recorded in this same ledger after the Q8 GEMV retraction -
*single-run comparisons cannot support causal claims* - broken again, and
published with a precision the evidence never had.

## Repeated, interleaved, identical preambles

Three alternating pairs at ~5,200 tokens, so drift cannot load onto one arm.

| chunk | mean | sd | range |
| --- | --- | --- | --- |
| 128 | 171.7 s | 9.2 | 164.9 - 184.8 |
| 256 | 100.6 s | 3.5 | 97.1 - 105.3 |

**1.708x, clean 3/3 separation.** The single-run estimate *under*-reported it:
that run caught chunk 128 near its fast end.

## The mechanism was wrong, which is the more useful finding

The 1.32x was attributed to avoiding the repeated `repeat_kv` expansion. That
hypothesis was then tested directly by removing the expansion - see below - and
the result was **10% slower**. So the mechanism is falsified, and the Amdahl
decomposition built on it was never load-bearing.

Re-running that arithmetic on the better data would give 83% and a 5.84x
ceiling. **Neither is published**, because it is the same inference whose premise
has now been disproved by experiment. What is established is that chunk 256 is
1.71x faster. *Why* is open - plausibly fewer kernel launches and better GEMM
shapes amortising per-chunk fixed costs, not the expansion.

## Grouped attention: correct, and a regression

`AKH_QWEN35_GROUPED_ATTENTION=1` folds the GQA repeat into the row dimension -
`(batch, kv_heads, ratio * seq, dim)` against `(batch, kv_heads, dim, positions)`
- so the expanded K and V are never built.

It is numerically sound. Exactly equal to `repeat_kv` on CPU, element for
element, on both the score and value matmuls; and cosine **1.0000000000**
against the materializing path on the full 27B model's post-prefill hidden
state, relative L2 3.03e-06.

It is also slower. Three interleaved pairs at ~5,200 tokens:

| path | mean | sd | range |
| --- | --- | --- | --- |
| materializing | 167.8 s | 5.0 | 160.7 - 171.8 |
| grouped | 184.8 s | 2.1 | 182.6 - 187.5 |

**10.1% slower, clean 3/3 separation** - every materializing run beat every
grouped run. Dropping the batch dimension from 32 heads to 8 kv-groups appears
to cost more GEMM parallelism than the avoided copy saves. `repeat_kv` looks
wasteful and buys more than it spends.

Kept behind the flag, off, with the number in its doc comment. The result is
counter-intuitive enough that it will be proposed again, and a flag plus a
measurement is cheaper than a reimplementation.

## Why the default stays 128 anyway

Peak VRAM, polled from the driver during a 10,417-token prefill - a true peak,
not the free-memory delta the gate's multiplier was derived from:

| chunk | peak | of card |
| --- | --- | --- |
| 128 | 23.17 GiB | 96.6% |
| 256 | 23.61 GiB | **98.4%** |

Chunk 256 leaves **388 MiB** on a 23.98 GiB card at a context the gate admits,
and the gate's cap is 10,816 tokens - about 400 tokens beyond this run, roughly
another 224 MiB. A compositor repaint could take the rest.

So the 1.71x is real and not safely available at the context limit: taking it
requires either a lower context cap or more headroom, and that is a trade to
decide deliberately rather than by moving a default.

`AKH_QWEN35_PREFILL_CHUNK` selects it for anyone who knows their context is
short. The measurement also **corroborates the memory gate**: at 96.6% of the
card at 10,417 tokens against a 10,816-token cap, the 4.5 cost multiplier lands
about where it should.
