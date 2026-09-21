# TurboQuant on real KV: measured, and it works

Source `9983ccd7` plus this change. Local RX 7900 XTX, HIP, exact pinned Ptah
model, same harness and same prompt as the PolarQuant measurement earlier today,
so the numbers are directly comparable.

## Implementation

`src/nlu/turbo_quant.rs` implements Algorithm 1 (MSE-optimal) of Zandieh,
Daliri, Hadian and Mirrokni, *TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate*, arXiv:2504.19874, ICLR 2026.

A random rotation makes every coordinate of a unit vector follow the same Beta
density `f(x) ∝ (1 - x^2)^((d-3)/2)`, which tends to `N(0, 1/d)`. Coordinates
are then near-independent, so each is quantized with a scalar Lloyd-Max codebook
solved once for that density. Norms are stored in f32 and restored on
reconstruction. The rotation and codebook depend only on dimension, bit-width
and seed, never on the data, so it is valid online with no calibration pass.

Eight CPU tests, including four that check the implementation against published
values rather than against itself:

- 1-bit centroids match the paper's closed form `±sqrt(2/π)/sqrt(d)`.
- 2-bit centroids match `±0.453/sqrt(d)` and `±1.51/sqrt(d)`.
- Measured distortion on unit vectors matches Theorem 1's 0.36, 0.117, 0.03 and
  0.009 for 1 to 4 bits.
- Distortion falls about 4x per added bit, as the `4^-b` bound requires.

The rotation uses an inline SplitMix64 rather than a crate RNG, so a rotation is
reproducible from its seed regardless of which `rand` version is linked;
dequantization depends on regenerating exactly the rotation that quantized the
data. The paper's inner-product variant, which composes this with a 1-bit QJL
transform on the residual to remove estimation bias, is **not** implemented:
reconstructed vectors must not be treated as unbiased for inner products.

## Result on this model's KV cache

45 positions, 16 attention layers, 4 KV heads, 2,880 vectors per tensor kind.
Mean over layers; worst is the worst single vector.

| Quantizer | Bits | Key cos (worst/mean) | Key rel. L2 | Value cos (worst/mean) | Value rel. L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PolarQuant | 3 | -/0.9038 (worst 0.7374) | 0.430 | **-0.3714**/0.7075 | 0.685 |
| TurboQuant | 2 | 0.9181/0.9403 | 0.341 | 0.9143/0.9401 | 0.341 |
| TurboQuant | 3 | 0.9720/0.9829 | 0.184 | 0.9718/0.9830 | 0.184 |
| TurboQuant | 4 | 0.9888/0.9954 | 0.096 | 0.9891/0.9954 | 0.096 |

Three things stand out.

**TurboQuant at 2 bits beats PolarQuant at 3**, on both tensors, while storing a
third less. The comparison is not close: 0.940 against 0.904 on keys and 0.940
against 0.708 on values.

**The key/value asymmetry disappears.** PolarQuant degraded values far worse
than keys, the asymmetry KIVI addresses by quantizing them differently.
TurboQuant scores them identically to three decimal places at every bit-width,
because the random rotation makes the coordinate distribution independent of the
input's outlier structure. One code path now serves both tensors.

**No catastrophic vectors.** PolarQuant's worst value vector round-tripped to
cosine -0.371, anti-correlated with the original. TurboQuant's worst at 2 bits
is 0.914.

Measured relative L2 also matches the paper's `sqrt(D_mse)` on real KV, not just
on synthetic unit vectors: 0.341 against 0.342 at 2 bits, 0.184 against 0.173 at
3, 0.096 against 0.095 at 4. The implementation behaves on this model's data the
way the theory says it should.

## What it buys

Storage per 256-dimensional vector is `bits * 256` plus a 32-bit norm, so the
effective width is `bits + 0.125` bits per coordinate against 32 today.

| Bits | Effective width | KV per token | Compression | Resident ceiling |
| ---: | ---: | ---: | ---: | ---: |
| f32 today | 32 | 128 KiB | 1x | ~40,000 tokens |
| 4 | 4.125 | 16.5 KiB | 7.8x | ~310,000 tokens |
| 3 | 3.125 | 12.5 KiB | 10.2x | ~410,000 tokens |
| 2 | 2.125 | 8.5 KiB | 15.1x | ~600,000 tokens |

At 4 bits the advertised 262,144-token context becomes reachable on this card
for the first time, with headroom. The paper reports quality neutrality at 3.5
bits per channel and marginal degradation at 2.5, which brackets the 3-to-4 bit
range these measurements cover.

## What is still required before switching it on

Cosine 0.9954 at 4 bits is excellent for a lossy path and still far below the
project's 0.9999 bit-exact gate, which no lossy KV path can meet by
construction. Activation therefore needs an end-to-end quality gate agreed in
advance — the five-daemon canary's recall answer is the obvious candidate, since
it already asserts a specific retrieved number after eviction and restart.

Also unmeasured: the cost. Quantization is a dense `d x d` rotation per vector,
which is 65,536 multiply-adds per 256-dimensional key or value. That is the cost
Fast-TurboQuant (arXiv:2606.21448) removes with a structured Hadamard transform.
Nothing here has been run inside the decode loop, so there is no throughput
figure, and the rotation is currently CPU-side f32 rather than a device kernel.

## Not established

One prompt, 45 positions, one device, one seed, one head dimension. No
generation-quality comparison, no long-context retrieval test, no throughput
measurement, and no integration into `AttentionState`. Keys here are post-RoPE;
a quantizer applied before RoPE would face a different distribution.
