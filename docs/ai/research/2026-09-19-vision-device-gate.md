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

Also untouched: position-triple threading for image tokens. `mrope_pair_positions`
exists and is tested to collapse to 1-D RoPE when t == h == w, but nothing yet
threads distinct temporal, height and width indices through an image's tokens.
