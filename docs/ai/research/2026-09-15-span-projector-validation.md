# Span projector validation checkpoint

The workspace already contained the compact projector and recall-loop wiring
when this continuation began. See [the loop description](2026-09-15-native-neural-recall-loop.md)
for that implementation; this checkpoint does not claim to have implemented it
again or to have independently verified its recorded hardware evidence.

## Change verified here

`src/nlu/span_projection.rs` exposes public `CapturePolicy` fields. A caller can
therefore bypass `for_model` and supply a hidden width larger than the fixed
Hadamard block. Previously, `zip` silently truncated those means while the
policy hash continued to describe the larger width. Duplicate and out-of-range
layers could likewise be accepted when the means matched the invalid policy.

Projection now rejects invalid policy dimensions and unordered, duplicate or
out-of-model layers before processing means. Valid policies, recipe hashes and
projection arithmetic were unchanged by this validation fix.

The subsequent specification review found that the canonical recipe omitted
the zero-derived-seed fallback and normalization precision. Projector recipe
version 2 explicitly describes those rules, the xorshift shifts/sign rule and
f32 Hadamard arithmetic. This deliberately changes `projector_sha256` so vectors
with the older incomplete recipe cannot silently enter the new query space.
The numerical projection itself is unchanged. No persisted artifacts were
migrated and no production deployment was changed.

The new regression first failed on the silent-truncation assertion, then all
eight projector tests passed with validation enabled:

```bash
CARGO_BUILD_JOBS=2 cargo test --offline --lib --no-default-features nlu::span_projection::tests -- --test-threads=1
```

The precision/seed description test separately failed before the recipe fix.
It also includes a worked positive basis-vector fixture whose zero-derived-seed
layer produces exactly 2000 zero bits under the specified fallback.
All nine projector tests passed after the recipe fix. The native three-turn
capture/recall fixture was rebuilt and rerun afterward and passed as well.

The native CPU recall fixture passed, then the full `candle-runtime` service
library suite passed (26 tests, 2 ignored) before the recipe-metadata revision:

```bash
CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-services --no-default-features --features candle-runtime --lib -- --test-threads=1
```

Review: no hard documented standards violations; two heuristic maintainability
findings (unvalidated unequal-length Hamming inputs and split policy validation
ownership). The specification review's one concrete finding was the incomplete
recipe, addressed by version 2. Broader refactoring is not part of this fix.

These are CPU interface/synthetic-vector checks, not semantic recall acceptance,
cross-backend compatibility, exact-model performance or a full project test run.

## Current step and next steps

Current step: validate the newly present compact projector and staged native
recall loop. Do not assume the older handoff's “unwired” status still describes
the working tree.

Next: independently exercise the native/session/core integration, audit complete
representation identity (including arithmetic/backend compatibility), then run
held-out exact-model retrieval and memory-answer gates. The larger roadmap
remains ContextManager materialization, vision, live KV compression, performance,
and only then deployment and legacy cleanup under the authorized scope.
