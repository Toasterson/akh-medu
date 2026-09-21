# Neural-span annex: first cognitive-store slice

This is a storage/protocol tracer following the Fable-guided same-space design,
not yet native projection, neural recall quality, or prompt materialization.
No shared daemon, production database or remote host was changed.

## Implemented interface

- `AcceptedTurn.span_captures` is optional/default-empty and omitted when empty.
  The old wire representation and therefore its operation fingerprint are
  unchanged. Current akhsessd explicitly sends an empty list; native producers
  must discover support before sending the new field to an older strict core.
- Authenticated `/v1/capabilities` advertises `neural_spans_v1`. Authenticated
  `/v1/spans/query` performs bounded same-space lookup using the existing core
  Store, not a second Engine or database.
- A capture pins model, tokenizer, complete capture-policy artifact and projector
  artifact. The latter recipes must include dimensions, layers, pooling,
  arithmetic, algorithm, seed, normalization and bit packing. Pin equality is a
  declared representation identity, **not producer attestation** or proof that a
  vector came from those artifacts.
- Capture text is a validated UTF-8 byte range in either the actual accepted
  user input or assistant output. Maximum four captures per turn, 8192 bytes of
  text per capture, exactly 10000 projected bits. No raw activation retention.
- Buckets are isolated by authenticated service-principal ID, Akh and session.
  At most 256 spans remain per bucket, with oldest-first eviction. Queries return
  at most 16 hits, ordered by Hamming distance then stable turn ID. A different
  representation space yields no hits. No cross-session or symbolic-index query.

## Acceptance and policy

Validation precedes operation admission. The existing accepted-turn journal
retains the canonical operation ID and accepted content, but omits projected
vectors to avoid an unbounded duplicate vector history. It rejects conflicting
turn identities before engine recording. After Engine persistence, annex writes
and the successful acceptance receipt commit in one Store transaction.

Capture-disabled Akhs neither write new spans nor query old ones. Existing data
is not retroactively purged when policy changes; erasure/redaction propagation
and re-enable behavior need an explicit later lifecycle decision. The annex
contains sensitive content and must follow any future journal-erasure workflow.

Receipt replay does not append. The pre-existing unknown-outcome contract is
deliberately retained: an operation with no receipt fails closed on retry; it
does not automatically replay Engine mutations. This is not a claim of automatic
recovery across a crash between Engine persistence and receipt commit.

## Fable follow-up

Tool-disabled local `claude-fable-5` reviewed the [implementation brief](../plans/2026-09-14-span-annex-fable-brief.md).
Raw response: `/tmp/ptah-fable-span-annex-review.json` (`is_error=false`).

It flagged an evicted-turn/new-operation/different-vector replay. Its premise
that journal comparison ignores the operation ID does not match this source:
the journal compares the entire AcceptedTurn with only vectors cleared, retaining
`request_id`. A new ID is rejected there; reusing the original ID with changed
vectors is rejected by the original operation fingerprint. An actual HTTP test
fills/evicts the 256-span bucket, then exercises the proposed attack; it passed
as recorded below. No additional unbounded vector-dedupe table was added.

Its source-side UTF-8 range and range-length concerns are covered by validation;
producer trust, erasure policy and unknown-outcome limitations are recorded above.

## Verification

Five Store tests pass: restart/ranking against known distances 0 and 10000;
principal/Akh/session/four-pin isolation; replay/conflict/invalid-UTF-8 atomic
rollback; oldest eviction/query bounds; and exact legacy wire roundtrip.
Initial storage test failed on unsupported stubs before implementation.

Initial core HTTP tests passed acceptance/query, authorization and capture policy.
Expanded actual-HTTP eviction replay and unknown-outcome tests also passed.
The core-server build exposed a pre-existing duplicate OpenTelemetry import;
only the redundant import was removed. Locked `cron 0.17.0` was fetched from the
registry because it was missing locally; no dependency version was upgraded.

Native projected-signal transport, capability negotiation in the producer,
projection calibration, pre-decode materialization, real retrieval quality and
ContextManager::commit integration remain outstanding. Tests use declared
synthetic binary vectors; they do not claim model-derived semantic recall.

Expanded core HTTP tests PASSED (2 tests, 17.09 seconds): 256 later accepted
turns evicted the first vector; a new operation ID targeting that old turn was
rejected without changing the remaining query results. Changed original
operations were rejected too. Invalid captures did not consume an operation ID,
capability discovery was authenticated, and unknown-outcome retry remained 409
without replay. The original cognitive-tool/capture-policy regression also passed.

Final checks: all 17 service-store unit tests and both shared-contract unit tests
passed; `akhsessd` checked successfully with its isolated `session` feature
(13.52 s). `git diff --check` passed. No test-owned inference/service process
remains. The core test build and this check do not replace the prior HIP/Vulkan
release bundle or install/restart any production daemon.
