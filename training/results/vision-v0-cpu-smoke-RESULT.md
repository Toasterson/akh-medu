# Native text-vision V0 CPU smoke result

**Date:** 2026-09-04  
**Scope:** wiring/optimization smoke only; not the V1 128-case promotion rung

## Command

```bash
cargo run -q -p grug-train --example vision_overfit_smoke -- \
  --steps 100 --cases 3 --learning-rate 0.001
```

Configuration:

- deterministic Burn ndarray seed 51;
- 201,156 parameters (`d_model=64`, 2 decoder blocks, 4 heads);
- compiled-bitmap English line images, clean rendering;
- 16x16 RGB patches (96 visual tokens for a 384x64 line);
- shared decoder with visual prefix and byte-token transcript target.

## Result

| Measurement | Result |
|---|---:|
| initial teacher-forced loss | 5.619654 |
| final teacher-forced loss | 0.092995 |
| exact greedy transcript | 0.3333 (1/3) |
| character error rate | 0.3086 |
| unique semantic prediction rate | 0.3333 |
| cross-semantic collapse rate | 1.0000 |
| `v1_gate_passed` | **false** |
| output pattern | one transcript repeated for all three images |

The rendering, patchification, shared-decoder forward pass, gradient flow,
optimizer step and greedy decode path all executed. The optimization wiring is
green: loss fell by 98.3% and gradients reach the patch projection.

The capability gate is **red**. The model learned a teacher-forced sequence
shortcut and collapsed during autoregressive decoding instead of reliably
conditioning on each image. Low loss must not be used as an OCR success proxy.
The scorer now reports `unique_semantic_prediction_rate` and
`cross_semantic_collapse_rate`, and counts cross-item collapse as a precursor.

## Steering decision

Do not start a larger decoder run from this result. Run the stem-only
glyph-resolution/readout experiment, increase distinct short-string coverage,
and compare patch sizes before the 2–10M rung. The adaptive controller keeps
precursor steering in shadow mode until multi-checkpoint lead time is measured,
but this capability matrix already prevents promotion.
