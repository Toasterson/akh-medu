# Session recall integration: stable accepted-turn retries

Current step: independently validate the session/core side of the
[native neural recall loop](2026-09-15-native-neural-recall-loop.md).

## Reproduced failure and fix

The session daemon previously reconstructed each accepted-turn request on every
publication attempt. If the core received a request containing neural captures
but its reply was lost, a fresh daemon with different capability-discovery
results could retry the same request ID without those captures. The cognitive
core's operation fingerprint would then differ, preventing an idempotent replay.

A local HTTP fake-core regression records the initial accepted request, returns
a failure in place of the receipt, changes advertised capability support, and
reconstructs SessionApp over the same Store. The retry's body equality assertion
failed before the fix: the second request omitted `span_captures`.

The complete AcceptedTurn is now persisted in a temporary outbox before its
first POST. Later attempts use the stored request, without rebuilding its
captures, tool evidence or capability decision. Transactional admission reuses
the first candidate if publishers overlap and checks for an existing receipt.
Receipt storage, the existing acceptance event, and outbox deletion are one
transaction. A second publisher does not append a duplicate acceptance event.

The outbox-cleanup assertion also failed before cleanup was implemented, then
passed. After a successful receipt a third publication makes no extra HTTP call.
The original capture staging record's broader retention lifecycle is unchanged;
this fix avoids adding a permanent extra copy of the accepted payload.

This does not solve a cognitive-core operation interrupted before its receipt:
the existing `unknown_outcome` fail-closed behavior remains. It also cannot
reconstruct the exact body of a pre-upgrade attempt that was never staged.

## Verification

```bash
CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-services --no-default-features --features session --lib -- --test-threads=1
```

All 15 session-feature library tests passed, including all seven session
lifecycle tests: scope forwarding, stale/client identity rejection, capture
delivery/publication, legacy capability compatibility, reply-loss retry and
retirement/cleanup regressions. HTTP peers and database files are test-local.
An initial filter named `server::sessions::tests` matched zero tests; the actual
module is `server::sessions::lifecycle_tests`. Zero-match runs are not evidence.
No GPU, production core, shared daemon or remote host was touched.

## Next steps

Run held-out exact-model retrieval and memory-answer gates, including the
cross-backend representation-compatibility decision and recalled-text placement.
Then connect real ContextManager materialization, implement vision and live KV
compression, measure performance, and perform scoped deployment/legacy cleanup.
The complete project remains unfinished.
