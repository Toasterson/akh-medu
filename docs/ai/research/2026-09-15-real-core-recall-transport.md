# Real-core recall transport acceptance

This follows the [native memory-answer gate](2026-09-15-native-memory-answers.md).
The bounded scope is the missing session-to-core integration, not a claim that
the complete native daemon stack or deployed agent prompt has passed.

## Required behavior

A runtime delivers a synthetic user span through the authenticated session HTTP
router. Recall must return no hit while the turn is unaccepted. After the
session publisher sends the completed turn to the real core over private
loopback HTTP, a later turn in the same session must retrieve the exact text,
turn identity, byte range and distance through the session HTTP router.
Credentials, stale turns and a different session must not expose that memory.
Repeated publication must not create duplicate hits.

## Test scope

The opt-in `core-integration-tests` feature enables the real core module without
requiring Candle or a GPU. The fixture supplies one in-memory Engine and an
explicit transcript-enabled Akh configuration through `CoreBackend`; the core
router, authentication, operation receipts, persistence and span annex are real.
The session router runs through Tower requests, while its core client crosses
an actual loopback TCP listener. Temporary redb stores and credentials are
test-owned. No PostgreSQL connection, shared daemon or remote host is used.

The scheduler state and successful model completion are fixtures. This does not
test `akhrtd`, model capture, prompt rendering, generation or context eviction.
Those still need a combined isolated native HTTP acceptance test. Existing
module tests and GPU answer tests are complementary, not substitutes for that
combined test.

```bash
CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-services --no-default-features --features core-integration-tests --lib accepted_capture_round_trips_through_real_core_http -- --nocapture
```

Run without `AKH_TRANSCRIPT_CAPTURE`; the test refuses an environment override
so the configured policy is the one actually exercised.

## Results

The negative control omitted accepted-turn publication. The test failed at the
exact expected-hit assertion: the response contained an empty hit list, with
no authentication or transport failure. After enabling the session publisher,
the focused test passed (0.39 s). Repeating publication still returned exactly
one hit; invalid credentials returned 401, a stale generation returned 409,
and another session with the same Akh and query vector returned no hits.
The captured-but-not-completed turn also returned no hits.

No runtime implementation change was needed for this transport slice. The
test feature is opt-in and adds no default production dependency activation.

The complete package suite for this feature set also passed:

```bash
CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-services --no-default-features --features core-integration-tests
```

This includes all 16 library tests and all 23 session integration tests, plus
the other enabled package targets. It is not a whole-workspace or GPU suite.
Existing compiler warnings remain; `git diff --check` passed.

## Review

Standards review: no findings. Spec review: one coverage gap, because a
generation mismatch did not independently prove completed-turn rejection.
The follow-up assertion replays the completed turn's original callback and
credential and requires 409, separately from the generation-mismatch check.
The full package suite passed again after this addition. Spec re-review
confirmed the gap closed; neither review axis has an unresolved finding for
this bounded slice.
