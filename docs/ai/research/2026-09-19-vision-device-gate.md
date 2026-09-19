# The vision projector agrees across all three backends

Date: 2026-09-19. Evidence:
[`vision-devicegate-20260919-v1`](../../../experiments/native-recall/vision-devicegate-20260919-v1).

## What was blocked, and what was not

Vision was recorded as blocked on numerical validation, needing an external
consumer of this GGUF - llama.cpp `mtmd` or transformers - and none is installed.
That is true of **semantic** validation, and it was too broad a reading. It does
not block **cross-device agreement**, which uses the project's own 0.9999 cosine
gate and needs nothing external.

The two catch different faults, and the distinction is the whole point:

- Device agreement catches backend bugs - a wrong kernel, a transposed reshape,
  a device-specific fallback that diverges. This is the class that bit the Vulkan
  KV path earlier the same day, where a missing `U8 -> U32` cast made the feature
  fail closed on one backend without anyone noticing.
- It cannot catch a consistent misreading of the architecture. If `smart_resize`,
  the mrope section split or the patch ordering are wrong, all three devices are
  wrong together and agree to ten decimal places.

## Result

Projector `sha256-add205b7bfdb3f71...`, 64x64 deterministic input, 4 merged rows
of width 5120 - 20,480 elements - 27 blocks. One process per device, serial.

| pair | cosine | relative L2 | worst absolute |
| --- | --- | --- | --- |
| cpu vs hip | 1.0000000000 | 2.740e-06 | 1.602e-04 |
| cpu vs vulkan | 1.0000000000 | 3.728e-06 | 2.289e-04 |
| hip vs vulkan | 1.0000000000 | 4.053e-06 | 2.823e-04 |

All pairs meet the gate. That covers the substitutions the tower makes because
neither device backend implements the ops directly: patch embedding as
unfold+matmul rather than `conv2d`, and device-dispatched `softmax_last_dim` and
`gelu`. Those are precisely the places where an implementation can agree in shape
and in summary statistics while computing something else.

## Summary statistics are not the test

The probe previously reported mean, standard deviation, L2 norm and the first
eight values. Every one of those is invariant under a permutation or a sign flip
of the vector, so two devices can match on all of them and still differ. The
probe now dumps the full projection and the gate compares elementwise.

This was checked rather than assumed. A dump with **two elements swapped** -
identical mean, identical standard deviation, identical L2 norm - scores cosine
0.9998978914 and **fails**. A 5% tilt on alternate elements scores 0.9998703464
and fails. The gate rejects what it should; the passing result above is not an
artefact of a test that cannot fail.

## Two fixes this required

- The probe **did not build under HIP at all**: `Device::new_vulkan` was not
  cfg-gated, so `--features hip` failed to compile the example. Naming a device
  whose backend is absent now reports that rather than breaking the build.
- The probe had no way to emit the full projection, only its summary. Added
  `--dump`.

## What remains open

Semantic validation. Nothing here establishes that this projector computes what
Qwen3-VL's does - only that the three backends compute the same thing as each
other. That still needs an external consumer of this GGUF, and none is installed.

## Position-triple threading, since done

`mrope_pair_positions` existed with tests but was called **only from tests** -
dead in the runtime, which is what "built but not threaded" amounted to.

`PartialRope::apply` now takes `RopePositions`: `Contiguous` for text, or a
`(temporal, height, width)` triple per token for image tokens, gathered from the
rope table with `index_select` - an op both device backends already implement,
so no new kernel.

The property that makes this verifiable without an external reference is that a
triple with `t == h == w` is the text case written the long way, so it must
reproduce the contiguous path **exactly**. It does, element for element. A
genuine image triple must *not* match, or the sections would be inert and image
tokens silently placed as text - the bug that passes every smoke test. Both are
asserted.

Two refusals rather than silent fallbacks: a triple count that disagrees with the
sequence length fails, and an artifact declaring no usable section split refuses
image positions instead of treating them as text. That second path is reachable -
this artifact's split is parsed defensively and discarded when it does not
describe the model.

On the real 27B model the text path is **bit-identical** across the change:
cosine 1.0000000000, relative L2 0.000e+00, worst absolute 0.000e+00 on the
post-prefill hidden state, same prompt either side.

A first attempt at that comparison reported cosine 0.4309 and failed. The two
dumps came from **different prompts**. A cosine gate is only a gate when both
sides see identical input; otherwise it is an expensive way to confirm that two
different questions have different answers. The same mistake as this morning's
recorder-versus-direct comparison, in the opposite direction - it would have made
me reject correct work rather than accept broken work.

## The image path exists at the model level

Threading positions is not enough on its own: image rows come from the
projector, not from any token id, so the forward has to accept embeddings.

- `embed_tokens` exposes the model's own table, so a caller can build a mixed
  sequence - text rows from there, image rows from the projector.
- `prefill_embedded` takes that sequence plus a `(t, h, w)` triple per token.
  Supplied embeddings replace the table lookup wholesale.
- `rope: Option<&[[usize; 3]]>` is threaded through `Layer`, `Mixer` and
  `FullAttention`. `None` means text, which is every pre-existing caller. There
  is no inference of image layout from the token stream.

The property under test is that **the embedded path contains the text path**:
handing back exactly what the table would have produced, with degenerate
triples, reproduces the token path's logits exactly. Three refusals alongside
it - mismatched row counts, mismatched triple counts, and altered embeddings
must *change* the output, or supplied embeddings are being ignored and images
would do nothing while every test still passed.

That test caught an over-strict guard. Refusing image positions on an artifact
with no declared sections was too broad: a triple with `t == h == w` lands in
the same place whichever section it falls in, so it needs no split. The guard now
refuses only a genuine image layout on an artifact that cannot describe one -
the case that would actually mis-place image tokens as text.

The refactor touched every layer's signature, so the real 27B model was
re-checked: text output is **bit-identical**, relative L2 and worst absolute both
exactly 0.

**This is the mechanism, not a finished feature.** Nothing yet loads an image
file, turns a patch grid into triples, or reserves image placeholders in the
prompt. Those are the next slice and they are unblocked.

## The external reference exists after all

Recorded twice as "no external consumer is installed". There is no transformers,
no torch and no llama.cpp - but **ollama is installed** and has this exact model
registered as `ptah-qwen3.8`, which is where the pinned blob comes from.

Its server is not running. Starting it changes machine state and contends for the
GPU that probes need serially, so it is flagged rather than started. The accurate
statement is not "blocked, no reference exists" but "a reference exists and using
it needs a decision".
