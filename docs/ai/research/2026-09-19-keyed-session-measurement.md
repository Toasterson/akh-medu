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

`--keyed-repeats` reruns the whole benchmark from empty histories, so every
repeat issues the same prompt lengths in the same order and a rate that falls
across repeats is drift rather than a longer prompt. `--keyed-mode` selects one
pass or both.

Configuration for the two acceptance runs: 3 sessions, 3 turns per pass,
48-token cap, 18 requests per run. The soak and sweep runs below use one key.

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

## HIP keyed decode is erratic above roughly 200 context positions

HIP decode fell from 16.3-17.2 steps/s in the keyed benchmark's sequential pass
to 6.5-7.4 in its interleaved pass. Vulkan stayed within 9.1-12.9 across both.

**Correction.** The first attempt to explain this compared a fresh HIP process
given a 400-token prompt, which decoded at 18.32 and 18.36 steps/s, and
concluded that context length was excluded as the cause. That comparison is
invalid: `generate_completion` uses the unkeyed Q8 GEMV path, while
`generate_completion_for_session` on HIP uses the scoped-f32 keyed path. The two
runs therefore differed in arithmetic as well as context, and the fresh-process
figure says nothing about keyed decode. The run is retained as evidence for the
unkeyed path only. Two further runs measured the question properly, entirely
inside the keyed path.

**Drift is not the cause.** A single-key soak ran the same four turns six times
from empty histories, so every repeat issued identical prompt lengths in the
same order. Decode rate at each turn index was flat to three significant
figures across all six repeats — 17.1, 16.9, 16.6 and 8.5-11.6 — and the slow
fourth turn was slow in every repeat, including the first. Nothing degrades with
time in that run.

**VRAM saturation is not the cause either.** That soak reached 24.68 GB on the
24 GB card, higher than the keyed benchmark's 24.27 GB, while its decode rate
stayed flat. Device VRAM reaching capacity and keyed decode collapsing are
therefore separate observations, not a demonstrated mechanism. This retracts the
memory-pressure reading of the first run's telemetry.

**What the position sweep shows.** Eight consecutive keyed turns on one key, the
identical conversation on both devices:

| End context, positions | HIP steps/s | Vulkan steps/s |
| ---: | ---: | ---: |
| 58 | 17.20 | 12.35 |
| 128 | 16.94 | 11.92 |
| 191 | 16.55 | 11.64 |
| 256 | 8.20 | 12.09 |
| 322 | 12.97 | 11.95 |
| 381 | 12.96 | 11.51 |
| 440 | 7.82 | 9.41 |
| 510 | 6.89 | 9.01 |

Vulkan declines smoothly and monotonically, losing 27% from 58 to 510
positions. HIP is faster than Vulkan below 200 positions and then becomes
non-monotonic: it halves at 256, recovers to 13.0 at 322 and 381, and falls
again at 440 and 510. A monotonic cost of longer context cannot produce that
shape.

The established facts are: the effect is HIP-specific, reproducible, determined
by context position rather than elapsed time or accumulated allocations, and it
is not present on Vulkan over the same positions. The mechanism is **not
identified**. Per-step decode timings would separate a periodic expensive step,
such as a reallocation as the key/value cache grows, from a uniformly slower
kernel selection above a size threshold.

An attempt to get them from the completion's `check` callback **failed and was
removed**. The callback is not a per-decode-step hook: a turn with 3 decode
steps invoked it 20 times, so the intervals mix prefill chunks with sub-
millisecond gaps and cannot be read as step times. Proper instrumentation needs
a hook inside the decode loop of `quantized_qwen35`, reporting one duration per
sampled position. Nothing here should be inferred from the discarded numbers.
This belongs to the performance step, not to a feature slice, and it is the
reason the earlier interleaved pass looked like a collapse: those requests all
sat above 200 positions.

## Prefill warm-up, not drift

The same six-repeat soak showed time to first token for the identical 23-token
prompt growing 0.80, 0.89, 1.02, 1.10, 1.80, 1.83 seconds while that turn's
decode rate held at 17.1. A forty-repeat run has since shown this is a one-time
warm-up that flattens at about 1.92 s and then holds for 36 repeats, not
accumulating drift; the six-repeat run ended inside the transition. See the
[long residency soak](2026-09-19-long-residency-soak.md), which also found RSS
and device VRAM flat across 160 keyed requests.

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
[HIP fresh long context, unkeyed path](../../../experiments/native-recall/keyed-20260919-hip-longctx-v1/),
[HIP six-repeat soak](../../../experiments/native-recall/soak-20260919-hip-onekey-v1/),
[HIP position sweep](../../../experiments/native-recall/sweep-20260919-hip-context-v1/),
[Vulkan position sweep](../../../experiments/native-recall/sweep-20260919-vulkan-context-v1/).

The soak and sweep runs used probe SHA-256
`0e2b5b9439976fbd9ce610dca18ad39589a7477a663b230c39684dd19ef46605`, which adds
`--keyed-mode` and `--keyed-repeats` and is otherwise the same source.

## Not established

These are short runs on one machine with one model. They are not a long
residency soak, a quality evaluation, a daemon acceptance pass, or a controlled
cache-size comparison. Reported HIP keyed decode retains the scoped-f32
continuation guard; do not compare these numbers with the 2026-09-18 fresh-call
figures or with an old llama-server measurement from a different fixture.
