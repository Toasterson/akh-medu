# Gemma Vulkan throughput tuning — 2026-09-21

A larger, still bounded shader cache improves **warm decode** on the local
RX 7900 XTX without changing the model, quantization, tensor math or output.

| Coding prompt / output cap | Reference warm TPS | Tuned warm TPS | Improvement | Tuned entry limit |
|---|---:|---:|---:|---:|
| 40 prompt tokens / 128 output tokens | 22.44 | 26.18 | 16.7% | 8,192 |
| 183 prompt tokens / 256 output tokens | 20.82 | 23.47 | 12.7% | 16,384 |

TPS here means decode steps divided by decode time, excluding prefill and model
loading. Length-limited responses have 127/255 timed decode steps respectively:
the first output token belongs to prefill. These are controlled local development
measurements, not a broad coding-quality benchmark or a universal speed promise.
The generated responses are deliberately capped and are not validated programs.

## Cause and controlled experiment

The Vulkan backend specializes shader source for tensor shapes and offsets.
Its provisional 2,048-entry cache could not retain Gemma's working set:

- The 128-token fixture needed 4,712 variants; its reference repeat compiled
  4,542 variants again. At 8,192 entries, repeats compiled zero new variants.
- The 256-token fixture needed 9,192 variants; its reference repeat compiled
  9,022 variants again. At 16,384 entries, repeats compiled zero new variants.
- Tuned runs had no evictions or compilation failures. Source-text residency
  was 2,717,389 / 5,218,956 bytes, below the unchanged 16 MiB budget.

Each comparison used A/B/B/A fresh processes with two resident generations per
process. A is the original 2,048-entry limit. Reported TPS is the mean of the
second generations. No Rust build or other model inference ran during timing.
The pass gate was fixed at at least 10% improvement, identical generated text,
and bitwise-identical prefill hidden-state dumps. Both comparisons passed.

`result.json` and `longer/result.json` contain the verdicts. `measurements.json`
and `longer/measurements.json` retain each run, cache counters and memory samples.
`command.json`, `prompt.txt`, `measure.py` and their `longer/` counterparts record
the exact workload and orchestration. Run from the repository root after copying the harness and inputs into a fresh
experiment directory without result/log files; the harness refuses existing run logs.

The initial baseline had 12.1/21.3 TPS before shader warmup. That first-pass number
is not used to inflate the improvement. Cold shader compilation and model load
remain separate costs, and changing context shapes can still miss the cache.
Cache reuse after unloading/reloading a model is not guaranteed.

## Adoption and memory tradeoff

Lilith's Vulkan `runtime.env` now selects:

```sh
AKH_GPU_MATMUL=0
AKH_GPU_QUANT=0
AKH_VULKAN_PIPELINE_CACHE_ENTRIES=16384
```

Export these variables when starting `akhrtd`; it does not automatically source
`runtime.env`. The [stack guide](../../../examples/lilith/README.md) includes an
explicit environment prefix for the review fixture. Newly prepared Vulkan stacks
receive the same setting. Other backends are unaffected. The library default
remains 2,048; unset the override or set it to 2048 to restore reference behavior.
Valid entry overrides are clamped to 1–16,384. Invalid/unparseable values fall
back to the default. The source-text budget remains fixed at 16 MiB. This bounds
application-held entries/source, not compiled pipeline bytes or driver caches.

Peak process RSS was about 1.17 GiB in both variants. Tuned final device usage
was about 2 MiB higher for the short fixture and 262 MiB higher for the longer
fixture. Those are observed driver/sysfs samples, not memory guarantees; more
retained pipelines trade residency for fewer compilations.

Nine cache regression tests passed, covering entry/byte bounds, LRU behavior,
failed compilation, retained handles, zero capacity and counter wrap. Seven
stack-preparation tests passed, including emission of the Vulkan tuning setting.
Generated text and hidden dumps matched exactly in all measured A/B runs.

The next structural optimization would reduce shape-specific shader variants
(e.g. passing dimensions/offsets as data), which could improve first-use costs
without growing cache residency. This change only tunes the existing bounded
cache; it does not claim to solve that broader problem.

The final `akhrtd` release build and corrected Lilith config check passed.
Binary/source hashes are recorded in `provenance.json`. Services remain stopped.
