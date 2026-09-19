# PolarQuant on real KV: measured, and not usable as-is

Goal item 6 requires quantization quality to be measured *before* activation.
It has been. The answer is no.

Source `3be44ae5` plus this change. Local RX 7900 XTX, HIP, exact pinned Ptah
model. Read-only measurement: nothing in the live path is compressed.

## Method

`CandleBackend::kv_quantization_report` walks the resident attention caches
after a keyed call and, for every per-head, per-position vector of `head_dim`
= 256 values, runs the existing `compress_kv_vector` / `decompress_kv_vector`
PolarQuant pair and scores the round trip. Keys are post-RoPE and values are
not, so they are reported separately rather than pooled. The `std_dev_hint`
passed to decompression is that vector's own standard deviation. Zero vectors
are skipped, since cosine is undefined for them.

The probe confirms the memory model independently: exactly **16 resident
attention layers**, at indices 3, 7, 11 … 63, which is `(layer + 1) % 4 == 0`
over 64 target blocks. The MTP block held no state in this run.

## Result

45 positions, 16 layers, 4 KV heads, 2,880 vectors per tensor kind.

| | worst cosine | mean cosine | mean relative L2 |
| --- | ---: | ---: | ---: |
| Keys | 0.7374 | 0.9038 | 0.430 |
| Values | **-0.3714** | 0.7075 | 0.685 |

Per-layer keys sit between 0.877 and 0.899 mean cosine; values range from 0.453
at layer 3 to 0.683 at layer 23 and stay noisy throughout.

For scale, this project's numerical gates require **0.9999** cosine. A mean of
0.904 on keys is four orders of magnitude off in error terms, and at least one
value vector round-tripped to a *negative* cosine — anti-correlated with the
original. A 43% mean relative error on keys and 69% on values is not a
compression artifact to be tuned away; it is the wrong quantizer for this data.

## Why, and what the literature expects

PolarQuant here is 3-bit Lloyd-Max with a randomized Hadamard rotation, built
for VSA hypervectors: high-dimensional, near-isotropic, no dominant channels.
Attention KV is neither. KVQuant (arXiv:2401.18079) states plainly that existing
methods "fail to represent activations accurately in sub-4-bit precision" without
explicit outlier handling, which is exactly the regime attempted here.

The measured key/value asymmetry is the known one. KIVI (arXiv:2402.02750)
quantizes **keys per channel and values per token** precisely because the two
tensors have different outlier structure; our keys degrade gracefully while
values collapse, which is the same asymmetry from the other side. RotateKV
(IJCAI 2025) reaches 2-bit only with outlier-aware *adaptive* rotations rather
than a fixed random one, and QJL (AAAI 2025) avoids per-block constants
entirely. None of these is a drop-in for the current helper.

## Consequence

The PolarQuant helpers should not be wired into `AttentionState`. The
integration is cheap now that the model is vendored — that was the finding this
morning — but cheap integration of a quantizer this lossy would trade a
measurable 6x context ceiling for unmeasurable output damage.

The honest options, in order of increasing work:

1. **Do nothing yet.** The daemon runs at 1,024-2,048 token windows; the ceiling
   of roughly 40,000 tokens is not currently binding. Long context is a claim
   not yet exercised.
2. **Quantize only evicted pages**, keeping a recent F32 window, so the bit-exact
   recent path survives and only distant context degrades. Still needs a
   quantizer that works.
3. **Implement an outlier-aware KV quantizer** along KIVI or RotateKV lines,
   with per-channel keys and per-token values, and re-run this exact report as
   its acceptance gate.

Option 3 is the real work, and this report is the harness it would be judged by.

## Not established

One prompt, 45 positions, one device, one seed. No end-to-end generation quality
was measured, because the round-trip error is already far outside any gate this
project uses; a downstream quality run would only confirm it. A different
`std_dev_hint` calibration than the per-vector standard deviation was not
explored, and the helper documents no other. These figures are the quantizer's
error, not a statement about KV quantization in general.
