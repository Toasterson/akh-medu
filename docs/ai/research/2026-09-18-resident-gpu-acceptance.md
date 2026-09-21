# Exact-model resident GPU acceptance after cache bounding

Source: `9347805b` (bounded Vulkan pipeline cache). Local RX 7900 XTX only;
no shared services or remote hosts changed. Both probe processes exited normally.

## Results

| Measurement | Vulkan | HIP |
| --- | ---: | ---: |
| Model load, seconds | 30.60 | 23.92 |
| Fresh-call cold decode steps/s | 10.82 | 20.30 |
| Fresh-call warm decode steps/s | 13.52 | 20.21 |
| Unloaded baseline, W | 10.00 | 10.00 |
| Resident idle mean, W (60 samples) | 10.07 | 10.00 |
| Resident idle mean GPU busy, % | 0 | 0 |
| Final sampled fan speed, RPM | 0 | 0 |
| Peak sampled device VRAM, bytes | 19,926,855,680 | 20,198,748,160 |
| Final resident sampled device VRAM, bytes | 18,047,684,608 | 20,198,748,160 |
| Peak sampled process RSS, KiB | 1,783,840 | 582,748 |
| Total recorder duration, seconds | 170.34 | 133.27 |

Both backends produced identical text across their two fresh calls and each
other, with 30 decode steps per call. Both passed the existing keyed checks:
exact-prefix reuse (24 tokens), fresh/reused reply equivalence, full-context
cap, session eviction, cancellation, and recovery after a length stop.

The idle gate retains model ownership, waits 15 seconds after requests, then
samples once per second for 60 seconds. Required means: busy <= 5%, power <=
unloaded baseline + 20 W. Both passed. The original sustained idle utilization
symptom did **not** reproduce in either run; no additional power fix was made.
The diagnosis skill's measurement-first loop was used, without speculating
about a new failure when the measured gate was green.

VRAM is device-wide sysfs telemetry, not per-process attribution. RSS and VRAM
are sampled, not allocation high-water marks. The higher HIP retained VRAM is
an observation, not evidence by itself of a leak. Fans reaching zero is from
the final telemetry samples, not an independent fan-speed acceptance gate.

## Identity and reproduction

Build completed in 4m06s with existing feature-dependent unused-code warnings:

```sh
CARGO_BUILD_JOBS=2 cargo build --locked --offline --release \
  --no-default-features --features hip,vulkan --example candle_session_probe
```

Probe SHA-256:
`662d348d616a6aa972af8117820e6dd62ddede66e39891732057350cabf42af8`.

Model path is the pinned 16,810,714,496-byte Ptah Q4_K_M artifact
`/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec`.
The recorder checked its path/size, **not** a newly computed model digest.
Tokenizer SHA-256 was recomputed and matches
`e271c182ffa63eebd6d0f60a688fb90d3e87a1141c99c4d87a0d4508badfefea`.

The reusable recorder is `scripts/session-platform/resident-probe.py`.
Each run's `identity.json` records its exact command and explicit kernel/batching
settings. It collects stdout, stderr, one-second RSS/GPU telemetry and exit
status; it refuses to overwrite evidence and bounds execution to 30 minutes.
Its three CPU-only lifecycle tests passed: recording/non-overwrite, nonzero
exit propagation, and timeout termination of its own child.

Artifacts:

- [Vulkan identity](../../../experiments/native-recall/resident-20260918-vulkan-v1/identity.json),
  [probe output](../../../experiments/native-recall/resident-20260918-vulkan-v1/stdout.jsonl),
  [telemetry](../../../experiments/native-recall/resident-20260918-vulkan-v1/telemetry.jsonl).
- [HIP identity](../../../experiments/native-recall/resident-20260918-hip-v1/identity.json),
  [probe output](../../../experiments/native-recall/resident-20260918-hip-v1/stdout.jsonl),
  [telemetry](../../../experiments/native-recall/resident-20260918-hip-v1/telemetry.jsonl).

## Not yet established / next step

These short runs establish functional and resident-idle acceptance for this
probe binary. They are **not** a long-duration memory soak, model-wide quality
evaluation, daemon acceptance, or a controlled pre/post cache performance
comparison. Cache hit/miss/eviction counters were not collected.

Reported throughput is from fresh unkeyed calls. HIP uses Q8 activation GEMV
there; keyed Qwen continuation deliberately retains its f32 correctness guard.
Do not describe 20.21 steps/s as daemon/keyed throughput or compare it directly
with an old llama-server number from another fixture.

Next extend the probe with varied keyed sessions and per-request timings/cache
statistics, run a longer residency test, then rebuild/rerun private five-daemon
acceptance. Packaging, shared cutover and legacy cleanup remain separate gates.
