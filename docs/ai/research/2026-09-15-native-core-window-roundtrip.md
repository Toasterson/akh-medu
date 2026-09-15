# Native capture, core publication and recall after window eviction

## Combined fixture

The existing real-core HTTP test now runs an additional native fixture when
both `core-integration-tests` and `candle-runtime` are enabled. Five turns use
one native continuation, a pinned tiny CPU model with a 96-token context and
four-token output reservation, and the actual `internal::run` model loop.

The first input is `Remember the word cedar`. Three twelve-word filler turns
create window pressure; the last input is `What word?`. Native capture produces
the vectors. A test Host routes capture and recall through authenticated session
HTTP handlers; the session client talks to the real core over loopback TCP.
The real session publisher sends each successful model output with its retained
capture. No canned vectors or canned hits are supplied.

Assertions require one delivered capture per turn, every previously published
span to be retrievable while the candidate pool is at most four, the first
fact's exact turn identity/text, preservation of the original durable message,
and a recalled block containing the first fact before the last question.

The fixture still supplies scheduler Running/Completed transitions and a dummy
workspace binding. Session ingress uses Tower requests, not a separate daemon
process. The Host is a test adapter, not the production runtime dispatcher.
The tiny model always emits `ok`: this is lifecycle/window acceptance, not
semantic answer quality, exact Ptah model acceptance or deployment readiness.

## Failure exposed and fix

The two-turn combined test passed. The five-turn version failed twice at the
same fact-materialization assertion (about 0.8 seconds per run). Its trace
showed that the fact was already recalled on turn four. Turn five skipped that
fact as visible, materialized other notes, then could evict the prior recalled
block while making room for those notes. Dedupe had relied on visibility in a
prompt that was subsequently changed.

The native recall path now invalidates that dedupe decision whenever recall
assembly evicts additional history. It rebuilds the current recall block from
all returned hits and budgets again. If it cannot fit, it continues with the
original prompt without persisting optional notes. A final containment guard
also prevents the materialization byte cap from dropping a returned hit that
was visible in the original prompt. The unchanged five-turn assertion passed
after the ordering fix (1.11 seconds).

This was found by combining the real native and core interfaces; neither the
earlier two-turn transport test nor the canned-hit window fixture caught it.
Future ContextManager work should keep visibility, materialization and window
selection in one decision rather than treating dedupe as independent of
eviction.

## Reproduction

```bash
CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-services --no-default-features --features core-integration-tests,candle-runtime --lib accepted_capture_round_trips -- --nocapture
CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-services --no-default-features --features core-integration-tests,candle-runtime
```

Run without `AKH_TRANSCRIPT_CAPTURE`, as required by the enclosing test.
All stores, credentials and model artifacts are temporary and local. The
loopback core task is aborted on scope exit, including assertion failures.
The native fix remains in the pre-existing uncommitted `internal_native.rs`
runtime work; the test checkpoint alone is not a complete runtime release.

The combined package suite passed after the fix and final byte-cap guard:
44 library tests passed (two exact-artifact/GPU tests ignored), all 23 session
integration tests passed, and other enabled package targets passed. Two
external Anima fixture tests remain explicitly ignored. This is not a
whole-workspace or exact-model GPU acceptance run. `git diff --check` passed.

Next: run the exact Ptah HIP/Vulkan answer gates against the new window and
recall assembly path, then exercise actual isolated daemon dispatch/restarts
with the full agent prompt. No shared services were changed.
