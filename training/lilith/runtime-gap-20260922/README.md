# Vulkan report latency investigation — 2026-09-22

Starting regression: coordination `model-run3` report completed in 452.148s.
The real system/user report is rendered in `baseline/prompt.txt`. The minimized
CLI replay uses the same pinned GGUF/tokenizer and eight generated tokens, two
resident rounds. It does not by itself certify full-report latency or quality.

## Measurement

`python3 scripts/session-platform/lilith_latency_probe.py --out NEW_DIRECTORY`
now measures both prefill (<=30s) and decode (>=15 tokens/sec). The first version
measured only prefill and incorrectly passed the slow-decode case; its frozen
result is preserved and `baseline/diagnosis.json` explicitly marks that missing
gate. First versus warm observations must remain separate.

- 773-token report: prefill 22.8/23.6s; decode 1.13/1.27 tok/s.
- 27-token short context: prefill 1.22/0.70s; decode 8.09/21.86 tok/s.
- Warm long replay had no new pipeline-cache misses. Compilation cannot alone
  explain the sustained slowdown.

The first isolated matmul probe showed sub-millisecond work; its results are in
`matmul-baseline.jsonl`. The expanded stage microbenchmark also appeared fast,
but it reused a small tensor working set. An instrumented real-model replay then
located most elapsed decode time in KV growth and repeated-head copies. The
instrumentation introduces synchronization and is diagnostic, not a throughput
oracle. Its raw code/logs are intentionally retained under `debug/`.

Ranked differential hypotheses were context-scaled attention/cache work,
shader/cache behavior, and broader GPU state. The short/long result plus warm
cache counts support the first; they do not establish a general driver fix.

## Opt-in candidates and validation

- Reusable F32 KV buffers with geometric capacity growth, copying the prefix only
  on growth and writing only new positions otherwise.
- Group query heads against shared KV heads for single-token decode, eliminating
  explicit repeated K/V tensors.
- Hoist physical matmul strides out of the reduction loop, keeping summation
  order unchanged. This supports capacity-strided cache views.

These remain opt-in through `AKH_VULKAN_REUSABLE_KV=1`,
`AKH_VULKAN_GROUPED_DECODE=1`, and `AKH_VULKAN_MATMUL_STRIDES=1` while numerical,
full-vocabulary, token, reset/growth and original-report checks are pending.
The unused parallel-GEMV sketch is retained only in `debug/`; it was never enabled.

CPU numerical/cache-growth controls passed, including shared-head groups,
nonzero-offset capacity views and growth over 256/512-token boundaries. A
same-binary reference/candidate real-model replay and full session report remain
required. Old daemon binaries and frozen experiment scores are unchanged.

## Same-binary model results

CPU and exclusive Vulkan cache/grouped-attention controls passed. Real-model
773-token replay used eight greedy output tokens and two resident rounds:

| Variant | First decode tok/s | Warm decode tok/s | Full-vocabulary parity |
|---|---:|---:|---|
| Reference | 1.465 | 1.274 | Reference |
| Reusable KV + grouped decode | 7.855 | 14.251 | All 16 step hashes identical |
| Above + physical matmul strides | 6.724 | 15.082 | All 16 step hashes identical |

The stride shader separately passed CPU/GPU matmul controls (maximum observed
absolute error 1.43e-6). Warm prefill remains roughly 23 seconds. These short
measurements support removal of repeated KV copies as the main decode gain;
they do not establish a reliable incremental stride-shader speedup. Cold shader
compilation still costs time. Timing was collected while a CPU release build was
running, so these are development observations rather than isolated performance
certification. See `decode-comparison.json` and `decode-all-comparison.json`.

The first/warm 15 tok/s gate is **not yet passed** (cold remains below it).
Full original report completion is still pending. Flags remain opt-in and no
shared runtime or model deployment has been replaced.
