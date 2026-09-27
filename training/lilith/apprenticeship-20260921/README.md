# First frozen-weight apprenticeship experiment — 2026-09-21

Two replay-verified lessons were admitted from earlier Lilith/Ptah episodes.
The local lesson store supports attribution, immutable source/evidence envelopes,
revocation, cold reuse, text/VSA retrieval and development candidate export.
Teaching principles and query features are operator-curated. This is not yet
agent-authored curriculum or gradient training.

[Guide](../../../examples/lilith/APPRENTICESHIP.md) ·
[Protocol fixed before generation](protocol.json) · [Live result](run1/result.json)

| Arm | Independent checks | Observed turn time |
|---|---:|---:|
| No lesson | 1,512 passed | 60.17 s |
| Text retrieval | 1,512 passed | 19.06 s |
| VSA retrieval | 1,512 passed | 15.05 s |

Both retrieval arms chose the same bounded-range lesson and identical context.
There is **no measured quality improvement and no VSA superiority evidence**.
This single related-family task is a development probe, not a sealed evaluation.
The first arm paid cold-load costs; later arms used the resident model and warmed
execution paths. Do not interpret those times as a causal retrieval speedup.

The model and all caps were fixed across arms (512 output tokens, 2,048 context,
1,400-byte lesson budget, one attempt, no tools or specialist delegation). All
four project-owned daemons stopped after the run. Model weights are unchanged.

## Evidence

- `lessons.json`: exact lesson envelopes, original episode attribution and replay.
- `admitted.json`: content-deduplicated lesson IDs.
- `retrieval-checks.json`: text/VSA choices, byte-identical cold reuse, and exclusion
  after revocation/re-import. Revocation was tested in a separate temporary store.
- `candidate-train.jsonl` and `candidate-manifest.json`: one development clamp seed;
  the probe's bounded-range family is excluded. No probe solution is exported.
  This tiny candidate corpus is explicitly ineligible for adapter training.
- `run1/*-session.json`: independent model sessions and exact responses.
- `unit-tests.log`: 58 scoped Python tests pass, including admission failures,
  forged pass-flag rejection via replay, teacher-opinion separation, family
  exclusion, corruption, revocation, context budget and transfer counterexamples.
- `vsa-tests.log`: two Rust tests pass for role binding, order/dedup stability and
  rejected invalid feature sets. The real ranking helper uses existing VSA ops.

The registered verifier ID/hash covers Python reference behavior only. These
receipts do not validate C overflow, ABI, kernels or native illumos execution.
The first lesson's original clamp episode lacked a pinned contract; that fact is
preserved rather than retroactively claiming historical provenance.
