# ADR 101: Keep a compact convolutional vision stem as a gated fallback

- **Status:** accepted for V1 bootstrap; not selected for production promotion
- **Date:** 2026-09-04
- **Supersedes:** no part of ADR 100's shared-decoder or tokenizer contract
- **Evidence:** [stem-resolution result](../../../training/results/vision-stem-resolution-RESULT.md), [V1 clean-line result](../../../training/results/vision-v1-a40-RESULT.md)

## Context

ADR 100 required a stem-only small-type probe before decoder-scale OCR. The
Gemma-style raw linear patch baseline learned the four observed positions but
failed its held-out-translation gate across every tested 8/16/24/32/48px
patch. Extending the representative 16px/16px run from 120 to 500 steps kept
training accuracy at 100% while held-out accuracy fell to 15.6%.

The same preflight showed that full screenshots and pages cannot satisfy the
visual-token budget through naive full-image patches. The plan therefore
requires an explicit compact-convolution, tiling/resampling, or deferral
decision rather than a relaxed metric.

## Decision

Implement a compact native candidate with a full-resolution two-convolution
residual pre-block followed by four 3x3 local-convolution/max-pooling stages.
Normalize visual content before adding factorized row/column embeddings. Its
effective stride is 16, so it presents the same number of visual prefix tokens
as the 16px raw-patch baseline and feeds the identical shared decoder/text ABI.
The raw linear patch stem remains the baseline/control.

The compact stem cleared the isolated representation gate on a CUDA A40 after
failure-directed refinement: a deterministic dense translation curriculum
trained 167 positions per glyph while two positions remained absent. Both 8px
and 16px glyph sets reached 100% held-out-position accuracy after 3,000 steps.
Sparse four- and twelve-position curricula remained below 77%, so the dense
curriculum is part of the bootstrap recipe rather than optional polish.

This is a stem/readout result, not OCR. It selects the compact candidate for
the 2--10M clean-line experiment only. Ordinary max pooling can still alias;
anti-aliased downsampling remains a future controlled comparison. Burn ndarray
also measured 23.7 seconds for one 32-image training step, so continuous
training uses an accelerated backend while local inference is measured
separately.

For screenshots and pages, tiling/resampling remains mandatory even if the
conv candidate wins; an effective stride of 16 still produces the same context
count as 16px patches.

## Promotion gates

The compact candidate replaces the baseline only if all hold:

1. at least 95% held-out-position accuracy on both 8px and 16px glyph probes
   (**passed: 100% / 100% on the recorded A40 run**);
2. at least 98% exact greedy transcription on the 128-case clean OCR rung
   (**passed: 100% / 100% with zero CER/WER/collapse**);
3. line/receipt token budgets remain within ADR 100 limits (**passed for this
   384x64 line: 96/512 visual positions, 18.75%; receipt preflight passed at
   the original 1,024-position target**);
4. local inference latency and memory are recorded and acceptable (**recorded:
   network-denied CPU replay 3/3 exact, 4.09 s/case, 44,253,184-byte peak RSS;
   production acceptance remains workload-specific**);
5. text-only sentinel regression remains inside its cumulative budget
   (**unrun: blocks model promotion**).

A GPU speedup without the representation gates, or representation accuracy
without a viable local inference path, is insufficient.

The compact stem is therefore selected for the next held-out OCR experiments,
but not promoted as a generally capable vision model. Different-renderer/real
audit data, unseen strings/fonts, EN/DE/FR slices, corruption/layout coverage,
metamorphic groups and text retention remain mandatory. The experimental
checkpoint may be published only with those limitations attached.
