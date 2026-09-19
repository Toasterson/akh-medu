# Vulkan compressed KV: the code layout, what it costs, and two wrong numbers

Date: 2026-09-19. Device: RADV GFX1100 via wgpu. Model: the pinned Qwen3.8-27B
Q4_K_M artifact. Evidence:
[`experiments/native-recall/kvpack-20260919-vulkan-v1`](../../../experiments/native-recall/kvpack-20260919-vulkan-v1).

## What this set out to answer

Whether compressed KV pages, already working on HIP, could be made to pay on
Vulkan without writing a new shader. The prior entry in the ledger said yes:
three 8-bit codes fit in a `u32` word, and unpacking needs only `affine`,
`floor`, a subtraction and a `stack`, all of which the backend has. Three and
not four because the unpack runs through f32, which is exact only to `2^24`.

That reasoning held. Two things around it did not.

## The byte layout never ran on Vulkan at all

The first three-arm run failed in the middle arm:

    Error: LlmGenerationFailed { reason: "session generation: Vulkan cast U8 to U32 unsupported" }

`decode_on_device` widens codes to the `u32` that `index_select` requires. The
Vulkan `to_dtype` had arms for `F32 <-> U32` and `F32 <-> U8` but none for
`U8 -> U32`, so the decode aborted the moment a position aged out of the exact
window. Compressed KV on Vulkan was not giving a nominal saving. It was failing
closed.

This had not been caught because the canary that exercised 4/3/2 bits ran a
1,024-token window against a 16-position exact window, which never ages anything
out - the same blind spot that hid the paged-position invariant bug earlier the
same day.

Both casts are now in `vendor/candle-core/src/vulkan_backend.rs`. Widening is a
copy, since both integer dtypes occupy a full word there; narrowing truncates,
to match `as u8`. A device test covers the byte range at both endpoints and the
truncation cases above it.

The packed layout sidesteps this entirely - packed words are already `u32` - so
arm C ran even before the cast existed. That is why the first run produced a
packed number and no byte number to compare it against.

## The saving, measured on a live cache

`stored_and_uncompressed_bytes` existed but was reachable only from a unit test.
The compression ratio was asserted and never measured on a running model. It is
now surfaced as `kv_page_residency()` through the backend and printed by the
probe, which is what the figures below are read from: 16 attention layers holding
592 paged positions between them.

| arm | decode warm | decode cold | KV stored | ratio |
| --- | --- | --- | --- | --- |
| no compression | 13.38 steps/s | 11.07 | - | - |
| byte layout | 8.33 | 6.49 | 4,868,608 B | **0.996x** |
| packed-triple | 7.98 | 6.07 | 1,648,128 B | **2.94x** |

Against 4,849,664 bytes of f32.

Greedy output was byte-identical across all three Vulkan arms, same stop reason
and same token count. **On HIP the same policy changes the output**: the
uncompressed run gives `Hello, it's great to meet you!` in 30 tokens, the 4-bit
paged run gives `Hello!` in 23. Both devices agree with each other uncompressed.

So token agreement on Vulkan was luck at one prompt, not a property of 4-bit
paging - and had I measured only Vulkan I would have reported it as one. This is
the failure mode the standing rule about not weakening numerical gates on greedy
agreement is aimed at, arriving from the reassuring direction: the weaker signal
said pass on the device I was working on, and the same signal says fail one
device over.

Why the two devices differ under identical codes is **unexplained**. The
quantizer is seeded identically and the paging schedule is the same, so the codes
should match; what differs is the arithmetic of the page decode. Not chased here.
HIP's behaviour is unchanged by this work - its layout, absorb and decode paths
are byte-for-byte what they were - so this is a pre-existing property of 4-bit
paging that this comparison exposed, not a regression.

## Two accounting errors, both flattering

The byte arm first reported **3.94x**. That was fiction. The accounting charged
one byte per code unconditionally, and the entire premise of this work is that a
`u8` element costs four bytes on Vulkan. So the layout that is a net loss there
was claiming the largest saving in the table.

The first fix branched the accounting on layout but left `Byte` charging one
byte per code, which is the same mistake one level in. Element width is a
property of the device, not of the dtype's name. `KvPages` now carries
`code_element_bytes` read from the device, and the corrected byte arm reports
0.996x - it stores slightly more than the f32 it replaces, as it must.

The packed figure of 2.94x was right the first time, but only because that path
was computed from the word size rather than from a dtype name. The unit test now
asserts the loss case directly: on a four-byte element the byte layout stores
more than it replaces.

## What the cost actually is

Packing is nearly free: 8.33 -> 7.98 steps/s, about 5%. Almost all of the ~40%
decode cost is the page decode both layouts share - gather the centroids, scale
by the stored norm, apply the inverse rotation. Adding the unpack to it barely
registers.

That only became measurable once the missing cast let the byte arm run. Before
that the comparison available was compression against no compression, which
cannot separate the unpack from the decode.

## Limits

- One short prompt, 592 paged positions. The decode cost is per step over the
  whole paged cache, so it grows with context and this measurement is at the
  small end. Nothing here says what 8k or 128k positions cost.
- Two rounds per arm, one process per arm. Per the variance finding, arms share
  an identical preamble, but this is not a repeat-count that supports a claim
  about a 5% difference. The 5% packing overhead is within what single runs can
  show; the 2.94x storage ratio is exact arithmetic, not a timing.
- Token-identity at one prompt is not a numerical gate, and the HIP result above
  shows it failing in practice. The 0.9999 cosine gate is unchanged and this does
  not touch it. Whether 4-bit paging is acceptable for quality is **not settled
  by anything here** and needs the cosine gate run against paged reconstruction
  on both devices.
- 4x per word is still unreachable through public tensor ops, since the f32
  route caps at three codes. A WGSL integer expression would reach it; the
  existing `elementwise` helper could express one without a new kernel file.
