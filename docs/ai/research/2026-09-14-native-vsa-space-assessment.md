# Compact span projection: preserve vector-space identity

Local tool-disabled `claude-fable-5` consultation succeeded against the
[source brief](../plans/2026-09-14-native-vsa-space-fable-brief.md).
Raw response/model metadata: `/tmp/ptah-fable-vsa-space-review.json`.

## Accepted direction

The existing dense NeuralVsaBridge learns alignment to grounded symbolic
ItemMemory vectors. A new compact, untrained projection must not query that
symbolic coordinate system as if it were compatible. Evaluate same-pipeline
span-to-span retrieval first; report **neural-span similarity hits**, not
successful grounded-concept recall.

Keep any span index as an annex of the core's existing accepted-turn journal,
with the same tenant/session scope, retention and transcript-capture policy.
Key by durable accepted turn and a versioned representation identity containing
exact model/tokenizer fingerprints, tap layers and residual definition, pooling,
dimensions, projection algorithm/seed and normalization/quantization. Reject
space mismatch rather than silently mixing vectors. No second Engine in akhrtd.

A candidate vertical slice is capability-negotiated accepted-span delivery plus
a bounded session-scoped linear-scan query endpoint. Its completion would prove
write/read behavior, not yet memory materialization before decode or the full
ContextManager commit/retrieval requirement. Existing pre-dispatch context cannot
receive a query vector that has not been computed yet.

## Corrections and limits

- Fable described `polar_quant::randomized_hadamard` as SRHT with an applicable
  geometry guarantee. Actual local code only flips signs and applies a full
  normalized Hadamard transform. It neither subsamples nor binarizes nor defines
  the 5120-to-10000 mapping. A new mapping needs its own numerical and retrieval
  evaluation; the existing helper is not that evidence.
- CPU projection does not make HIP and Vulkan source activations bit-identical.
  Cross-backend projected stability remains a separate gate, particularly near
  sign thresholds. Include arithmetic-policy identity or explicitly evaluated
  compatibility; do not promise backend-independent exact bits without testing.
- Raw pooled means for later re-indexing would be extra retained sensitive data.
  Do not silently persist them by default. Any calibration retention must be
  bounded, policy-controlled and covered by deletion/retention behavior.
- The suggested 95% top-10 overlap is a proposed evaluation threshold, not a
  measured result. Neighbor agreement with f32 means alone also cannot prove
  semantic usefulness: held-out paraphrase and actual memory-answer fixtures
  are needed alongside exact/near-duplicate checks.
- Fable suggested deferring ContextManager::commit. That may delimit an initial
  index tracer, but does not remove the accepted plan's live commit obligation.

No projection, index, shared protocol change or neural retrieval route has been
implemented by this design consultation. Same-pass tap correctness is recorded
separately; symbolic-space alignment and actual virtual-context hits remain open.
