# Native text-vision stem resolution result

**Date:** 2026-09-04  
**Scope:** stem plus linear glyph readout; no language-model decoder

## Command

```bash
cargo run -q -p grug-train --example vision_stem_probe -- \
  --steps 120 --learning-rate 0.003
```

The deterministic probe classifies 32 pinned alphanumeric glyphs on a 48x48
canvas. Four small translations are used for training; two unseen translations
are held out. The readout sees the ordered patch-stem representation, not a
mean-pooled summary. The strict gate is 95% held-out-position accuracy.

## Result

| Glyph | Patch | Visual tokens | Train-position accuracy | Held-out-position accuracy | Gate |
|---:|---:|---:|---:|---:|---:|
| 8px | 8px | 36 | 76.6% | 3.1% | fail |
| 8px | 16px | 9 | 85.2% | 4.7% | fail |
| 8px | 24px | 4 | 5.5% | 3.1% | fail |
| 8px | 32px | 4 | 33.6% | 6.3% | fail |
| 8px | 48px | 1 | 36.7% | 6.3% | fail |
| 16px | 8px | 36 | 97.7% | 34.4% | fail |
| 16px | 16px | 9 | 100.0% | 29.7% | fail |
| 16px | 24px | 4 | 83.6% | 46.9% | fail |
| 16px | 32px | 4 | 75.8% | 46.9% | fail |
| 16px | 48px | 1 | 10.9% | 7.8% | fail |

A focused 16px-glyph/16px-patch run extended to 500 steps reached 100% on the
training positions but only 15.6% on held-out positions. More optimization of
the same readout therefore increased memorization rather than positional
robustness.

## Decision

The linear patch stem is retained as the Gemma-style baseline/control, but it
does not clear the bootstrap small-type/translation gate. Do not begin the
2–10M decoder rung with this stem alone. Implement and compare the already
specified compact convolutional/resampling fallback, then re-run this probe
and the 128-case OCR rung. This is representation evidence, not CER and not a
claim about production fonts.

An earlier mean-pooled version of this diagnostic stayed near chance even on
its training positions because the glyph signal was diluted across mostly
blank tokens. That probe design was rejected; its result was not attributed to
the visual architecture.

## Compact-convolution calibration

The four-stage stride-16 convolutional fallback was implemented and tested on
the same shared-decoder boundary. A 120-step 16px-glyph probe was stopped after
15 minutes because it had not completed on Burn ndarray CPU. After adding
step/loss telemetry, a four-step calibration measured:

| Measurement | Result |
|---|---:|
| parameters | 16,592 |
| first training step | 23.7 s |
| four training steps | 94.6 s |
| train + evaluation wall time | 106.8 s |
| loss | 3.5272 -> 3.4852 |

At that throughput, 120 CPU steps project to about 47 minutes. Four steps are
not an accuracy experiment, so the chance-level readout is **not** used to
reject the architecture. The conv fallback is opt-in and must run on an
accelerated backend before selection. CPU throughput is already a failed
operational gate for continuous local training, while local inference remains
unmeasured.

## Accelerated feedback loop

The compact candidate was then run through Burn 0.21's CUDA backend on one
Runpod secure-cloud NVIDIA A40 (48 GB, driver 580.159.03, CUDA/NVRTC 12.8;
container `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`). Each row keeps the
two held-out positions absent from training.

| Candidate / curriculum | Steps | Train positions per glyph | 8px held out | 16px held out | Gate |
|---|---:|---:|---:|---:|---:|
| four stride-2 convolutions | 1,000 | 4 | 1.6% | 46.9% | fail |
| full-resolution residual pre-block + strided convolutions | 1,000 | 4 | 3.1% | 57.8% | fail |
| pre-block + local convolutions/max-pool | 1,000 | 4 | 76.6% | 73.4% | fail |
| same stem, wider sparse curriculum | 1,000 | 12 | 68.8% | 65.6% | fail |
| content normalization separated from absolute position | 1,000 | 12 | 73.4% | 73.4% | fail |
| same stem, dense translation curriculum with two holes | 3,000 | 167 | **100.0%** | **100.0%** | **pass** |

The final 17,760-parameter stem first encodes strokes at full resolution,
uses four local-convolution/max-pooling reductions to an effective stride of
16, normalizes content independently, and then adds factorized row/column
positions for the shared decoder. The 167-position curriculum is the bounded
synthetic-data response to the measured stride-phase failure; it is not an
independent renderer audit. The exact two held-out offsets remain enforced by
an overlap assertion.

The final command was:

```bash
cargo run --release -p grug-train --features cuda \
  --example vision_conv_stem_probe -- \
  --steps 3000 --learning-rate 0.003 \
  > vision-conv-stem-a40.json
```

The byte-exact output is [vision-conv-stem-a40.json](vision-conv-stem-a40.json)
(SHA-256 `ac3e3f02fc3d17b55fa2d9655063f273752347470f7a1ac233fe76618f55c18e`).
This clears only the representation prerequisite and unlocks the 128-case
clean transcription rung. It is not a CER result, a font-generalization
result, or permission to claim usable OCR.
