# Native note acknowledgment diagnostic

The [ranked recall fix](2026-09-15-ranked-recall-budget-fix.md) restored the
Atlas answer on HIP and Vulkan. Both exact five-turn runs still answered `12`
and `UNKNOWN` to the counting and color notes instead of the required `OK`.

## Feedback loop and hypotheses

The original red-capable loop is the ignored
`exact_ptah_capture_round_trips_through_real_core_http` test. Its recorded
summaries show two answer failures on each backend. Exact artifact hashing and
loading make this a minutes-long loop; a synthetic CPU model cannot reproduce
this model's instruction following.

The narrower `exact_native_note_instruction_differential` probe loads the same
pinned artifacts once. It removes capture, recall, transport and windowing,
then compares fresh and continued keyed inference on identical rendered
prompts. Both paths retain the keyed HIP f32 arithmetic guard. It also compares
the original instruction with an explicit note-versus-question instruction.
Expected answers in the original acceptance gate are not relaxed or rewritten.

Ranked predictions:

1. Instruction interpretation: the original wording fails without the removed
   layers; more explicit wording changes the answers.
2. Mutable inference state: identical fresh/continued prompts disagree.
3. Recall/window assembly: removing those layers removes the failure.

This diagnostic is not daemon acceptance and does not alter production
instructions. Its wiring, like the native runtime integration, currently lives
in the mixed worktree.

## HIP isolation result

The probe passed fresh/continued equality in 116.48 seconds including artifact
verification and loading. With the original instruction, both paths answer
`OK`, `12`, `UNKNOWN`; with the explicit instruction, both answer `OK`, `OK`,
`OK`. Continued calls actually reuse 65/93 tokens for the original wording
and 93/120 for the explicit wording; every fresh comparison reuses zero.

This reproduces both failures with capture, recall and windowing removed.
Changing only instruction wording in that reduced fixture corrects them.
It is evidence for instruction interpretation, not a cache or recall bug;
it is not an independent numerical oracle for the model implementation.
The fresh prompt still contains the same conversation history, so this is
state-reset isolation, not a claim that all history is unnecessary.

Evidence: [HIP differential](../../../experiments/native-recall/note-instruction-differential-hip-v1.jsonl),
SHA-256 `e57f3fafacc170e3b8fd69ad18b467b0d2b97c85906d6709484cc9e52ea5ca8b`.

The follow-up is a separately named
`exact_ptah_explicit_notes_round_trip_through_real_core_http` test, not a silent
replacement for the original gate. It changes only the instruction text;
the five inputs, expected answers, 160-token context, 16-token output reserve,
model pins, capture/publication checks and final recalled-fact assertion stay
the same. The original short-instruction gate remains available unchanged.

Build and run (using the same explicit model/tokenizer environment variables
documented in the [original gate](2026-09-15-exact-native-core-gate.md)):

```bash
CARGO_BUILD_JOBS=2 cargo test --offline --release -p akh-session-services --no-default-features --features core-integration-tests,hip,vulkan --lib --no-run
# For each invocation, set AKH_NATIVE_GATE_DEVICE to hip or vulkan and
# AKH_NATIVE_GATE_OUTPUT to a new JSONL path; never run GPU tests in parallel.
target/release/deps/akh_session_services-4a42cd0664cf9d14 exact_native_note_instruction_differential --ignored --nocapture --test-threads=1
target/release/deps/akh_session_services-4a42cd0664cf9d14 exact_ptah_explicit_notes_round_trip_through_real_core_http --ignored --nocapture --test-threads=1
```

The ordinary release library suite passes: 48 passed, zero failed, five ignored.
Those ignored artifact tests must be invoked explicitly; their omission is not
a GPU acceptance result.

## Full explicit-instruction gate

HIP passes in 103.31 seconds including verification/load. Answers are `OK`,
`OK`, `OK`, `OK`, `4317`, all five accepted turns publish, and final-history
recall assertions pass. The longer instruction leaves room for two ranked
notes in the final block, including the Atlas fact. No budget or output
criterion was relaxed. The changed instruction also changes captured vectors
and distances; these are contextual signals, not text-only embeddings.

Evidence: [HIP full gate](../../../experiments/native-recall/explicit-notes-core-hip-v1.jsonl),
SHA-256 `d9ed807693b25e2720ebafe06bd221e0a8b20e49cffa458f931588494c53472c`.

Vulkan passes the same full gate in 154.43 seconds including verification/load.
All five answers, capture delivery counts, retrieved texts and Hamming
distances match HIP (turn IDs differ). Its summary records zero answer failures
and `passed: true`. Both GPU test processes exited; no shared services changed.

Evidence: [Vulkan full gate](../../../experiments/native-recall/explicit-notes-core-vulkan-v1.jsonl),
SHA-256 `fd24dafa755d2db9599f65f6e648de3ad658c050caa26c0ae437a1bc5fbde97b`.

Full-gate tested release binary SHA-256:
`09278811129c0cfc2d85ff7dc56a286c6ec62c65e520995429a6cc98ef4bcc65`.

Production instructions and runtime algorithms were not changed in this step.
The old short-wording criterion remains a known model-quality failure. This
test-specific wording result is not a general instruction-following guarantee.
Next is private daemon dispatch with the real cognitive core, followed by
held-out instruction/recall tests; no shared service restart is required for
that local work.
