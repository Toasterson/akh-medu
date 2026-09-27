# Tecton / Kimi K3 numerical review

Tecton returned a completed review on 2026-09-22 using `kimi-k3`.
See the [exact reply](reply-tecton.json) and [runtime provenance](provenance.md).

The advice separates two acceptance questions:

1. **Cache determinism:** compare identical execution paths and resident versus
   synchronous offloading strictly. Do not loosen this gate to accommodate the
   asynchronous transfer race. Fixed-parent boundary logits and losses agree
   exactly; nonzero adapter gradients differ by at most 2.91e-11.
2. **Different computation shapes:** use the small FP32 full-prefix regression
   as evidence for algorithmic equivalence, then characterize BF16 error using
   ULP or magnitude-scaled bounds with an absolute floor. Preserve raw errors.
   A small numerical regression is supporting evidence, not a universal proof.

The suggested BF16 calibration remains **open**. Tecton's short-control advice
references a 0.5625 maximum deviation; longer historical boundary controls
recorded deviations up to 17.1005859375. Those measurements cannot be declared
acceptable merely by choosing a larger tolerance after observing them. No new
cross-shape tolerance has been adopted, and exact batched BF16 equivalence is
not claimed. Cache correctness and same-policy replay have separate passing
controls; broader cross-shape numerical qualification remains unfinished.

This review made no optimizer update and produced no trained adapter or release
promotion. The final source-pinned cache control remains unchanged.
