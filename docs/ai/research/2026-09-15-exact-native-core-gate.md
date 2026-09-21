# Exact Ptah native/core window gate

The ignored `exact_ptah_capture_round_trips_through_real_core_http` test extends
the [combined CPU fixture](2026-09-15-native-core-window-roundtrip.md) to the
pinned Ptah model. It requires explicit device, model/tokenizer paths and a
new evidence file. It cannot silently replace the ordinary CPU test through
an environment variable.

## Fixed criterion

Five inputs: the Atlas port fact (4317), three twelve-word filler notes, then
the Atlas port question. The fixed instruction requires OK for notes and only
the number for a question (UNKNOWN if unavailable). Context is 160 tokens,
output reservation 16 tokens. The first four replies must be OK; the final
reply must be 4317. Native completion itself rejects truncated output.

Every turn must deliver one real native capture. Previously published inputs
must return from the real core, under the session scope. The final durable
history must retain the original fact and contain a recalled copy immediately
before the last question. Candidate count never exceeds four; this does not
test ranking under a large distractor pool.

The production native renderer, window selector and capture/recall loop run
in-process. Session HTTP uses Tower and core HTTP uses a loopback listener.
Scheduler transitions and the Host adapter remain fixtures. The instruction is
a test prompt, not the full deployed agent prompt. This is not daemon restart,
deployment, throughput, or comprehensive semantic quality acceptance.

## Reproduction

```bash
CARGO_BUILD_JOBS=2 cargo test --offline --release -p akh-session-services --no-default-features --features core-integration-tests,hip,vulkan --lib --no-run
```

Run the emitted test executable with filter
`exact_ptah_capture_round_trips_through_real_core_http` and
`--ignored --nocapture --test-threads=1`, serially per backend. Environment:

- `AKH_NATIVE_GATE_DEVICE=hip` or `vulkan`.
- `AKH_QWEN35_TEST_MODEL`: exact model path; enforced SHA-256
  `3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`.
- `AKH_QWEN35_TEST_TOKENIZER`: exact tokenizer path; enforced SHA-256
  `e271c182ffa63eebd6d0f60a688fb90d3e87a1141c99c4d87a0d4508badfefea`.
- `AKH_NATIVE_GATE_OUTPUT`: new JSONL file, never overwritten.
- `AKH_HIP_Q8_GEMV=1`, `AKH_HIP_QWEN35_FUSED_DELTA=0`; the keyed HIP
  activation path retains its f32 correctness guard.
- No `AKH_TRANSCRIPT_CAPTURE` override.

JSONL records configuration and instructions, each completed turn's input,
answer and hits, final durable messages, and a success summary only after all
assertions and accepted-turn publications succeed. A failure may leave a
partial file. No real user content or raw hidden-state means are recorded.

This build still depends on the existing mixed native runtime worktree;
committing the harness does not establish a clean deployable release.

## Initial finding

The first HIP run failed on the counting filler: replies were `OK`, then `12`
instead of `OK`. Both turns delivered a native capture; the second retrieved
the exact first fact from the real core. The initial run stopped at that answer
assertion (101.41 s including artifact verification/load), before the final
question. Its partial evidence is retained as
`/tmp/ptah-exact-core-gate.yOot76/hip.jsonl`, SHA-256
`a3d4a8512a55a6bf76e4ae863c9bed3232e3f3a4addd70b1375c0051f0a7c48c`.

For the follow-up, the harness counts answer mismatches and continues through
all five successfully completed model turns, then still fails if any answer
was wrong. Inputs, instruction, budgets and expected answers are unchanged.
The completed model outputs are published even when they fail this test's
answer-quality criterion. Inference or structural assertion failures still
stop the run and retain partial evidence.

## Completed backend comparison: FAIL on both

HIP and Vulkan produced the identical sequence `OK`, `12`, `UNKNOWN`, `OK`,
`UNKNOWN`. All five turns delivered one capture, and all retrieved text/distance
lists matched across backends (random turn IDs differ). For the final question,
the correct fact ranked first at distance 369; the fillers were 873, 882, 1001.
The final-history assertion failed on both: no newly materialized system block
preceded the last question. A recalled block containing the port fact had been
inserted on the preceding turn. Thus this is not a missing-core-hit failure.

The shared assembly path's all-hits retry can exceed the tiny window and skip
optional recall entirely. Budget-aware selection of a ranked subset is the
next fix hypothesis to test; it must preserve the earlier dedupe/eviction
regression. The two filler acknowledgment mismatches are separate quality
failures, not waived by a future correct final answer. No runtime fix or prompt
tuning was made during this comparison, and the exact gate is not passed.

HIP completed in 103.01 s, Vulkan in 131.56 s, including model verification and
load. These are correctness-fixture runtimes, not throughput benchmarks.
Both processes exited and released their model allocations. Shared services
were untouched.

Evidence retained under `experiments/native-recall/`:

- `exact-core-hip-initial-v1.jsonl`: partial first run, hash above.
- `exact-core-hip-v1.jsonl`: SHA-256
  `80caea43a428f0b147548d164d59d8ecd85df4aeb3de48a279545e44eed75f62`.
- `exact-core-vulkan-v1.jsonl`: SHA-256
  `d6d2442f71316235795962630d466c18e9e71abcf398d3740cdc6b4f79230a33`.

The completed runs used the same executable, SHA-256
`38a43a5eb10b2c47ff502f24466022269a7b8ae980e68b227155f2a7664235a9`,
and harness source SHA-256
`d8752f2625e5594d7baa75601a6bb5b66c5bd1c4de984efd23fdc592547c7e61`.
The ordinary release library suite passed 44 tests with three explicit
artifact/GPU tests ignored; the exact ignored gate was then run explicitly as
reported above. A passing CPU suite does not override these exact-model failures.
