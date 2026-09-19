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

**That wiring is now done.** `AttentionState` holds the recent window as f32
tensors and everything older as `KvPages`: one bit-packed store per (batch,
head) slot, so each slot's positions stay ordered across evictions. Positions
past the exact window are absorbed on append and decoded back on the attention
path. One prepared quantizer is shared by every slot and both tensors, because a
dense rotation per head per layer would cost more than the compression saves.

The in-place round trip that served as the first gate has been **removed**: real
paging supersedes it, and leaving both would have meant two code paths claiming
to do the same thing.

The five-daemon canary passes on real paged storage, at 4 and 2 bits:

| Configuration | Recall answer | Checks |
| --- | --- | --- |
| Paged 4-bit, 16 exact positions | 4317 | 7/7 pass |
| Paged 2-bit, 16 exact positions | 4317 | 7/7 pass |

Twenty-one model tests still pass, including the bit-exact ones, because the
compressed path is inert unless configured.

## Measured: host-side decode is not viable in the decode loop

Decode currently returns to the host, and that cost has now been measured rather
than assumed. Six consecutive keyed turns reaching 381 context positions:

| Configuration | Result |
| --- | --- |
| Paged KV off | completed in about 3 minutes; 17.4 to 12.2 steps/s |
| Paged KV on, 4-bit, 16 exact | **did not finish in 30 minutes**; timed out |

The GPU sat at about 10% busy during the paged run, against 94% in a normal one,
so the device is starved while the host quantizes and dequantizes. This is worse
than a slowdown of a few times: it is more than an order of magnitude, and the
run produced no per-turn figures at all before its deadline.

That reframes the earlier canary result. Those runs passed at 4, 3 and 2 bits
because the canary uses a 1,024-token window with short turns, so few positions
ever age out of the exact window. They established quality, which was their
purpose; they did not exercise the decode cost, and should not be read as
evidence that the path is usable.

**Consequence.** The paged storage is correct and its compression is real, but
it cannot be enabled for any workload with meaningful context until decode moves
onto the device.

## The device decode, specified

Op support was checked before designing, since three ops were found missing on
these backends earlier today. `index_select` **is** implemented on both: HIP
requires an f32, contiguous source; Vulkan requires **u32** indices. `matmul` is
available on both. So the decode is three device operations per slot:

1. `codebook.index_select(indices, 0)` — centroids, f32 codebook indexed by code.
2. broadcast-multiply by the per-position norms.
3. `matmul` by the rotation, which applies the inverse rotation because
   `(Pi^T y)^T = y^T Pi`.

The constraint that shapes the design is the index dtype, and checking it
changed the answer. Vulkan's `index_select` needs u32, and **u32 indices are
four bytes per value, exactly what f32 costs** — storing them that way forfeits
the entire saving.

The obvious fix, storing `u8` and converting per decode, **does not work on
either backend**:

| Cast | HIP | Vulkan |
| --- | --- | --- |
| `u8 -> u32` | absent | absent |
| `u8 -> f32` | absent | present |
| `f32 -> u32` | present | present |

HIP implements no `u8` casts at all; its only integer casts are `f32<->u32` and
`f32<->i64`. Vulkan could reach u32 in two hops through f32, but HIP cannot
reach it from u8 by any route.

So with the current backend op set, a device-resident index array must be u32 or
f32, both four bytes per value, and **saves no memory at all**. Device decode
and compression cannot both be had by composing what exists.

### Vulkan cannot save memory with `u8` at all

Checking the allocator settles it. `VulkanDevice::alloc` computes
`len.checked_mul(4)` — **four bytes per element for every dtype it supports**,
F32, U32 and U8 alike. Reads and copies use `self.len * 4` to match. A `u8`
tensor on Vulkan therefore occupies exactly what an `f32` tensor does.

HIP is different: it sizes allocations by `dtype.size_in_bytes()`, so a `u8`
tensor really is one byte per element there.

So the two backends need different work, and only one of them is small:

| Backend | Storage for `u8` | What compressed KV needs |
| --- | --- | --- |
| HIP | 1 byte per element | a `u8 -> u32` cast kernel |
| Vulkan | **4 bytes per element** | a packed representation, several codes per `u32` word, plus an unpacking gather |

On Vulkan no cast kernel helps, because the saving is lost in the allocator
before any cast happens. Getting compression there means changing how the
backend represents small integer tensors — packing four `u8` codes into each
`u32` word and teaching the gather to unpack — which is a change to the storage
model, not an added kernel.

### What this actually costs

The work is not the three-op decode; that part is straightforward. It is a
backend kernel:

- **HIP: done.** `akh_cast_u8_u32` and `akh_cast_u32_u8` are implemented in
  `vendor/candle-core/src/hip_kernels/elementwise.hip` with the existing
  `AKH_CAST_LAUNCH` macro, bound in `hip_backend.rs`, and covered by a device
  test that widens byte codes and drives `index_select` over a 256-entry
  codebook with them. 46 HIP backend tests pass. This is what lets a page hold
  one byte per code and still feed a gather that only takes u32 ids.

  `KvPages::decode_on_device` now implements the decode with those three
  primitives - `index_select` over the codebook, a broadcast multiply by the
  per-position norms, and a `matmul` by the rotation - and a test shows it
  reproduces the host decode to within 1e-4, which is f32 accumulation order
  rather than a different approximation. `TurboQuant` exposes its rotation and
  a `quantize_row` so a store can upload codes without going through the host
  page representation.

  Writing the test found two further gaps worth knowing: **F32 -> U8 and
  U8 -> F32 do not exist on HIP either**. Neither blocks the use case, because
  a page store already holds its codes as bytes on the host and uploads them
  directly, but it means u8 tensors on this backend can only be created from
  host bytes and read back as host bytes.
- **Vulkan, large**: packed small-integer storage and an unpacking gather,
  because its allocator gives `u8` no saving to begin with.

So compressed resident KV is **achievable on HIP and not on Vulkan** without
backend storage work. That asymmetry matters for planning: the feature could
ship device-side on one backend while the other keeps the f32 cache, which the
gating already allows since the path is off unless configured.

Until then the paged path stays correct, gated off by default, and unusable for
long context — which is where the earlier measurement already left it.


## Device decode measured: viable, at about a third of decode throughput

The same six-turn keyed benchmark that **did not finish in thirty minutes** with
host-side decode now completes:

| Turn | End context | No paging | Device paging, 4-bit |
| ---: | ---: | ---: | ---: |
| 0 | 58 | 17.35 | 11.60 |
| 1 | ~127 | 17.13 | 11.14 |
| 2 | ~190 | 16.25 | 11.20 |
| 3 | ~250 | 13.06 | 10.42 |
| 4 | ~318 | 8.05 | 11.21 |
| 5 | ~378 | 12.24 | 7.50 |

So the decode is no longer the obstacle: it went from starving the GPU at 10%
busy and never finishing, to completing in ordinary time. What it costs is
visible in the first three turns, which the variance analysis showed are the
stable ones: roughly **11.2 against 16.5 steps/s, about a third of decode
throughput**, in exchange for holding paged positions at one byte per code
instead of four.

Later turns are not comparable at this resolution. The baseline's 8.05 at turn 4
is one of the outliers the variance work identified, and both columns are single
runs, which is exactly the comparison that misled earlier today. The early-turn
figure is the one to quote.

Context lengths differ slightly between the columns because a lossy cache
changes the generated text, so turns tokenize to different lengths. That is
expected, and it is another reason to compare the stable early turns rather than
matching turn indices at the tail.

### A bug the canary could not have caught

The first run with real paging failed immediately: *"attention state is at
position 16, cannot append at 17"*. Once older positions page out, the state's
key tensor no longer holds every position, so the append invariant — which
compared that tensor's length against the absolute position — was wrong. It now
counts paged positions alongside resident ones.

The canary had passed at 4, 3 and 2 bits before this. It never exercised paging
at all: its 1,024-token window means nothing ages out of a 16-position exact
window in five short turns. The gate established quality, which was its purpose,
and said nothing about whether the paging mechanism worked.


## The canary, this time with paging actually exercised

The earlier canary passes used a 16-position exact window, which nothing aged
out of in five short turns. Rerun with `AKH_QWEN35_KV_QUANT_RECENT=4`, so
positions move into compressed pages on every turn:

| Configuration | Recall answer | Checks |
| --- | --- | --- |
| Device paging, 4-bit, 4 exact positions | 4317 | 7/7 pass |

Five spans published, four messages dropped, one span materialized, and the
post-restart question answered correctly — with most of the attention cache
held as one byte per code and reconstructed on the device for every step.

This is the gate the earlier runs looked like but were not. It exercises the
mechanism end to end: eviction into pages, device decode, concatenation with the
exact window, and recall across a session and core restart.

What it still does not establish is quality at length. Five short turns with a
1,024-token window is a small amount of compressed context, so this says the
mechanism is sound, not that a long conversation stays coherent under it. A
needle-in-a-haystack run remains the test that would say that.