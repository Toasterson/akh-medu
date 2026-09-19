# End-to-end gate for lossy KV: the canary passes to 2 bits

Source `81df0574` plus this change. Local private five-daemon stack, HIP, exact
pinned Ptah model. No shared service, production database or remote host
involved.

Measuring round-trip error told us TurboQuant is accurate; it could not tell us
whether the model still *works* with a degraded cache. The bit-exact gate cannot
answer that either, because no lossy path can meet it. This is the end-to-end
gate instead.

## What was wired

`quantized_qwen35` gained an optional lossy KV path, off unless configured:

- `AKH_QWEN35_KV_QUANT_BITS` enables it at that width.
- `AKH_QWEN35_KV_QUANT_RECENT` keeps that many newest positions exact, since a
  deployed compressor pages out older context rather than the current turn.

It round-trips the cache through TurboQuant **in place**. Storage layout is
unchanged, so this saves no memory: it isolates the quality question from the
memory engineering, which is the order that lets a gate run first. The newest
position is never quantized, so a decode step never re-quantizes what it just
wrote. A CPU test asserts the recent window stays bit-identical, that older
positions do change, and that they stay above 0.97 cosine at 4 bits.

## Result

The five-daemon canary, which asserts a specific number recalled after four
messages are evicted and the session and core both restart:

| Configuration | Recall answer | Checks |
| --- | --- | --- |
| Baseline, no quantization | 4317 | 7/7 pass |
| 4-bit, 16 exact positions | 4317 | 7/7 pass |
| 3-bit, 16 exact positions | 4317 | 7/7 pass |
| 2-bit, 16 exact positions | 4317 | 7/7 pass |

Every run also published five spans, dropped four messages, materialized one
span, and replied exactly `OK` to the note turns, so recalled notes stayed data
rather than instructions throughout.

## What this does and does not establish

It establishes that a TurboQuant-compressed cache does not break the recall
loop, provider continuity, eviction, isolation or restart recovery on this
fixture. That was the open question blocking activation, and it is now answered
in the affirmative.

It does **not** establish that 2 bits is safe. The measured cosine at 2 bits is
0.94, a large perturbation, and this fixture is narrow: five turns, a
1,024-token window, one retrieval of one number. A gate that passes at every
width tested is a gate that does not discriminate at those widths. The honest
reading is that the canary is necessary and insufficient, and that a
needle-in-a-haystack or instruction-adherence evaluation is what would actually
separate 2 from 4 bits.

On that basis **4 bits is the conservative default**: it measured 0.9954 mean
cosine on real KV, it is above the 3.5 bits per channel the TurboQuant paper
reports as quality-neutral, and it still yields 7.8x compression.

Also not established: any memory saving, because storage is unchanged; any
throughput figure, and the per-step host round trip this path performs makes the
runs noticeably slower than baseline, which is a property of the simulation
rather than of a real compressor; and anything about Vulkan, since these runs
were HIP.

Artifacts: [canary transcripts](../../../experiments/native-recall/kvquant-canary-20260919/).

## Next

Replace the in-place round trip with real paged storage: keep the recent window
as f32 tensors and hold older pages as codebook indices plus norms, decoding on
attention. That is where the 7.8x actually materializes, and this gate plus the
bit-exact recent window are what protect it.

`KvPageStore` in `src/nlu/turbo_quant.rs` is that storage, built and tested
ahead of the integration. It appends rows, decodes them back in order, and
evicts from the front as a paging policy would. Indices are **bit-packed** at
exactly the code width rather than one byte each, which is what separates a real
7.5x from the 4x an unpacked 4-bit index would give; a test asserts the ratio on
a 128-dimensional row rather than trusting the arithmetic.

What remains is wiring it into `AttentionState`: holding the recent window as
tensors and the remainder as pages, and decoding pages on the attention path.
The decode is a codebook gather, a norm scale and an inverse rotation, all of
which are device primitives, so it need not return to the host.
