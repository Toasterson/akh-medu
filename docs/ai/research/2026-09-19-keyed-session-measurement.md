# Keyed multi-session measurement and pipeline cache counters

Source: `bb2c6a2a` plus this change. Local RX 7900 XTX only; no shared service,
daemon or remote host was touched. All three probe processes exited normally and
each passed its resident-idle gate.

This is the **first keyed throughput measurement** of the native runtime. The
2026-09-18 numbers were fresh unkeyed calls and must not be compared with these.

## What is now measured

`candle_session_probe --keyed-benchmark` drives `generate_completion_for_session`
with several distinct keys. The backend keeps exactly one resident keyed session,
so the probe runs two passes over the same conversations:

- **sequential** — each key runs its turns back to back, so a turn can reuse the
  exact prefix its predecessor forwarded.
- **interleaved** — keys rotate every turn, so every request pays a key switch
  and a full prefill.

Each request prints prompt tokens, reused prompt tokens, time to first token,
decode seconds, decode steps and steps/s. The summary aggregates by whether the
request actually reused a prefix, which is the distinction that matters, rather
than by the pass it came from. Vulkan runs additionally report shader
specialization cache counters before and after.

Configuration: 3 sessions, 3 turns per pass, 48-token cap, 18 requests per run.

## Results

| Measurement | Vulkan | HIP |
| --- | ---: | ---: |
| Model load, seconds | 30.63 | 23.87 |
| Prefix-reused requests | 3 | 3 |
| Prefix-reused decode, steps/s (aggregate) | 11.92 | 16.78 |
| Prefix-reused time to first token, s (mean) | 1.43 | 0.92 |
| Full-prefill requests | 15 | 15 |
| Full-prefill decode, steps/s (aggregate) | 11.13 | 8.95 |
| Full-prefill decode, steps/s (median) | 11.80 | 7.21 |
| Full-prefill time to first token, s (mean) | 9.89 | 6.74 |
| Benchmark wall time, seconds | 215.38 | 175.80 |
| Resident idle mean, W (60 samples) | 10.12 | 10.98 |
| Resident idle mean GPU busy, % | 0 | 0 |
| Peak sampled device VRAM, GB | 19.93 | 24.27 |

Mean prompt length was 104.3 tokens for reused requests and 196.6 for full
prefills, so the time-to-first-token columns are not a like-for-like comparison
of the same prompt; they show what a daemon actually pays in each case. Mean
reused prefix was 81.3 tokens.

Both devices passed the existing continuation checks (exact prefix reuse, fresh
reply equivalence, full-context cap, session eviction, cancellation, length
recovery) and the resident-idle gate before and after the new benchmark.

## Prefix reuse is lost after a capped reply

Reuse appeared on exactly 3 of 18 requests per device, identically on both:

| Pass | Session | Turn | Prompt tokens | Reused | Previous turn finished |
| --- | ---: | ---: | ---: | ---: | --- |
| sequential | 0 | 1 | 81 | 58 | Stop |
| sequential | 0 | 2 | 154 | 0 | Length |
| sequential | 1 | 1 | 94 | 0 | Length |
| sequential | 1 | 2 | 154 | 131 | Stop |
| sequential | 2 | 1 | 78 | 55 | Stop |
| sequential | 2 | 2 | 150 | 0 | Length |

Every zero follows a turn that hit the 48-token output cap. That is the
documented `length_recovery` behaviour — a length-truncated turn drops resident
session state — and it is not a new defect. The measurement makes its cost
concrete: with a tight output cap, a third of otherwise-consecutive turns paid a
full prefill. Interleaved requests never reused, confirming the single resident
key. Whether truncated turns should keep their state is a design question for
the ContextManager slice, not something this change alters.

## HIP keyed decode degrades over a run; context length is not the cause

HIP decode fell from 16.3–17.2 steps/s in the sequential pass to 6.5–7.4 in the
interleaved pass. Vulkan stayed within 9.1–12.9 across both. The two candidate
explanations — longer prompts in the second pass, or accumulated state — were
confounded in that run, so a third run measured them apart.

A **fresh** HIP process given a single 400-token prompt, longer than any keyed
request in the benchmark, decoded at **18.32 and 18.36 steps/s** across its two
calls with 4.2 s time to first token and peak device VRAM of 21.43 GB.

Context length therefore does not explain the collapse: the fresh process was
faster at 400 prompt tokens than the keyed run was at 207–347. The degradation
tracks time and accumulated keyed work within one process. Device telemetry over
the HIP keyed run is consistent with memory pressure rather than throttling:

| Window | GPU busy | Power | Peak VRAM |
| --- | ---: | ---: | ---: |
| 60–90 s (sequential) | 98.6 % | 302.1 W | 24.06 GB |
| 180–210 s (interleaved) | 77.8 % | 202.8 W | 24.27 GB |

Falling utilization at falling power indicates stalls, not a power or thermal
cap. Vulkan held 19.93 GB flat with 84–95 % busy at 256–284 W throughout.

This is **correlation, not an established cause**. Device VRAM is system-wide
sysfs telemetry, not per-process attribution, and no allocation counters were
collected. The established facts are: HIP keyed decode halves over a run, the
cause is not prompt length, and HIP reaches the card's 24 GB capacity while
Vulkan does not. Identifying the allocation that grows is the first task of the
long-residency step, which is where this would otherwise have been discovered.

## Shader specialization cache counters

`PipelineCache` now counts hits, misses, evictions, bypasses, compile failures
and recency resets, exposed through `VulkanDevice::pipeline_cache_stats` and
`CandleBackend::vulkan_pipeline_cache_stats`. Nine CPU tests pass, including two
new ones that separate the counters and check that a recency reset is recorded
without corrupting residency accounting.

Across the Vulkan keyed benchmark the cache served 4,326,453 hits against 24,241
misses — a 99.44 % hit rate — and performed exactly 24,241 evictions while
sitting at its 2,048-entry limit for the whole run. Every miss evicted, so the
cache is saturated and recompiling at a steady low rate rather than converging.
No shader was bypassed, no compilation failed and the recency counter never
wrapped.

Whether raising the 2,048-entry limit would reduce time to first token is
**untested**. The counters observe application-held residency and compilation
events only, not compiled pipeline bytes or the driver's own cache.

## Identity and reproduction

Probe SHA-256 `68edb5cf3c4415e93fa5b8adf7b9c287955153220c24884d5ab57116d1934052`,
built in 4m02s with the existing feature-dependent warnings:

```sh
CARGO_BUILD_JOBS=2 cargo build --locked --offline --release \
  --no-default-features --features hip,vulkan --example candle_session_probe
```

The recorder now forwards extra probe flags with `--probe-arg=--flag`, so each
run's `identity.json` records the exact measured command:

```sh
python3 scripts/session-platform/resident-probe.py --device vulkan \
  --model <pinned Q4_K_M blob> --tokenizer <pinned tokenizer> \
  --binary target/release/examples/candle_session_probe \
  --output experiments/native-recall/keyed-20260919-vulkan-v1 \
  --probe-arg=--keyed-benchmark --probe-arg=--keyed-sessions --probe-arg=3 \
  --probe-arg=--keyed-turns --probe-arg=3 --probe-arg=--keyed-tokens --probe-arg=48
```

Model path and size were checked; the model digest was not recomputed. Tokenizer
SHA-256 matches `e271c182ffa63eebd6d0f60a688fb90d3e87a1141c99c4d87a0d4508badfefea`.
Four recorder lifecycle tests pass, including a new one asserting that forwarded
arguments reach both the child process and `identity.json`.

Artifacts: [Vulkan](../../../experiments/native-recall/keyed-20260919-vulkan-v1/),
[HIP](../../../experiments/native-recall/keyed-20260919-hip-v1/),
[HIP fresh long context](../../../experiments/native-recall/keyed-20260919-hip-longctx-v1/).

## Not established

These are 18-request runs on one machine with one model. They are not a long
residency soak, a quality evaluation, a daemon acceptance pass, or a controlled
cache-size comparison. Reported HIP keyed decode retains the scoped-f32
continuation guard; do not compare these numbers with the 2026-09-18 fresh-call
figures or with an old llama-server measurement from a different fixture.
