# Vulkan adapter transfer — 22 September 2026

This is an explicit scratch-adapter experiment. No deployed model file or daemon
configuration is changed. The candidate has **no measured quality improvement**
in the BF16 baseline/SFT/KLPO experiment and must not be promoted.

The exporter accepts only the pinned training parent's rank-2, alpha-2 last-layer
query projection. It verifies the experiment receipt and exact source file hash,
rejects unsupported PEFT variants, and writes two F32 tensors with an exact reload
check. Output is staged and renamed only after successful validation.

The runtime adds `x @ A.T @ B.T` to the existing quantized query projection before
query normalization/rotary embedding. Installing or clearing an adapter resets
KV state; the base quantized tensors remain unchanged. The probe checks zero-B
identity, nonzero learned effect, fresh tensor reload, and base rollback on the
actual Vulkan device. Greedy decode timings are descriptive single observations,
not a throughput optimization claim.

The training parent uses BF16 while deployed Gemma uses quantized GGUF. Mechanical
compatibility does not establish quality transfer. The experimental API is not
wired into automatic daemon activation or promotion.

## Evidence

- `export-tests.log`: exporter contract and failed-write cleanup tests.
- `export-result.json`, `klpo-export/manifest.json`: exact candidate provenance.
- `rust-unit-tests.log`: additive tensor math and zero-adapter CPU controls.
- Actual GPU probe receipts are added after the sequential model comparison.

## Measured result

Both fresh Vulkan processes passed. Zero-adapter, removal/rollback and reloaded
adapter full-vocabulary maximum error were exactly zero. The learned adapter
changed at least one logit by `0.0043854117`; greedy tokens happened to remain
unchanged for this prompt. Base and adapted full-vocabulary F32 hashes and both
generated token sequences matched exactly across the two fresh processes.

| Probe | Base decode TPS | Adapter decode TPS | Model load |
|---|---:|---:|---:|
| Fresh process 1 | 18.68 | 11.44 | 16.76 s |
| Fresh process 2 | 18.22 | 15.17 | 16.19 s |

These 32-token observations always time the adapter first, then the base; kernel
warm-up and order can bias the difference. They do not isolate steady-state LoRA
overhead and are not directly comparable to the earlier session-stack throughput
runs. There is no throughput improvement claim. Warm, alternating paired trials
would be required before treating this path as deployment-ready.

`probe-protocol.json` freezes the driver, binary, runtime/example source, manifest
and experiment receipt hashes. `probe-{1,2}.json` contain the controls, timings and
logit hashes; `probe-{1,2}-process.json` durably record command, exit and elapsed
time; `cold-reload.json` records exact fresh-process comparisons. Eight exporter
unit tests and two CPU tensor controls passed.

The [independent provenance audit](independent-provenance-audit.md) verifies the
export-to-probe hash chain and records the remaining process-receipt limitations.
