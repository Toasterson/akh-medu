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
projection arithmetic are unchanged.

The new regression first failed on the silent-truncation assertion, then all
eight projector tests passed with validation enabled:

```bash
CARGO_BUILD_JOBS=2 cargo test --offline --lib --no-default-features nlu::span_projection::tests -- --test-threads=1
```

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
