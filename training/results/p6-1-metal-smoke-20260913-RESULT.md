# Phase 51 P6.1 — grug-train `metal` feature + Apple M2 throughput/memory smoke (2026-09-13)

**Verdict: local lane is GO at context 2048 (and 4096 with gradient checkpointing); NO-GO at 8192 on this box.**

Scope: ADR 106 / P6 plan (`docs/ai/plans/2026-09-13-phase51-p6-local-training-neurosymbolic.md`), Anima WI P6.1
(`01a09ac7-8638-71d1-a6ac-69857951e042`). Session-driven; no runner involved.

## What landed

- `grug-train` feature `metal = ["burn/metal"]` — Burn 0.21's `burn-wgpu` CubeCL backend with the Metal
  Shading Language target. Under `--features metal`, `GpuBackend = burn::backend::Metal<f32>`,
  `GpuAutodiffBackend` honours `grad_checkpointing`, and `gpu_device()` returns `WgpuDevice::default()`.
  `metal` is mutually exclusive with `gpu` (ROCm) and `cuda` (`compile_error!`), same as those two are with each other.
- Every example/binary that selected `Cpu…`/`Gpu…` by `any(feature = "gpu", feature = "cuda")` now
  also selects the GPU alias under `metal` (13 examples + `knowledge_runtime`/`multimodal_runtime`/`transformer` tests).
- `train_foundation --context <n>`: packed training/eval context for the run, capped at the shape's
  `max_seq_len`. Lets `small_8k` (RoPE, chunked attention 1024) train at 2048/4096 without changing the
  checkpointed shape. Needed for every number below and for the remediation-v4 recipe in the P6 plan.

## Setup

| item | value |
|---|---|
| machine | Apple M2, 16 GB unified memory, Metal 4, macOS 25.5 |
| model | `small_8k` (d=512, 8 heads, 8 layers, ff 2048, max_seq_len 8192, RoPE θ=10k, attention chunk 1024), vocab 2000 (smoke tokenizer) |
| corpus | synthetic 2 000/100-record foundation corpus (92 922 / 4 646 tokens); 3 000/600 for the 8192 run |
| build | `cargo build --release -p grug-train --features metal[,grad_checkpointing] --example train_foundation` |
| run | 5–6 optimizer steps, `--eval-every` = steps, `--eval-batches 1`; `step_ms` from `train.jsonl`, footprint from `/usr/bin/time -l` "peak memory footprint" |

Step 0 includes CubeCL kernel JIT; steady-state = median of steps 1..4.

## Measurements

| context | batch | grad ckpt | steady step | tokens/s | peak footprint | outcome |
|---:|---:|:-:|---:|---:|---:|---|
| 2048 | 1 | no  | 1.35 s | ≈1 500 | 4.7 GB | OK |
| 2048 | 1 | yes | 1.41 s | ≈1 450 | 3.5 GB | OK |
| 2048 | 2 | yes | 2.60 s | ≈1 575 | 7.6 GB | OK |
| 2048 | 4 | no  | ≈31 s  | ≈265   | 16.6 GB | swapping — unusable |
| 4096 | 1 | no  | ≈17 s  | ≈240   | 16.3 GB | swapping — unusable |
| 4096 | 1 | yes | 4.20 s | ≈975   | 11.9 GB | OK (tight) |
| 4096 | 2 | no  | —      | —      | 29 GB   | killed by the OS |
| 8192 | 1 | no  | —      | —      | 40 GB   | killed by the OS |
| 8192 | 1 | yes | ≈205 s | ≈40    | >21 GB swap | swapping — stopped by hand |

Losses match across backends/settings for the same seed (eval@4 = 4.2736 at 2048×1 with and without
checkpointing), i.e. checkpointing is numerically neutral here as it should be.

## Reading

- **Throughput ceiling ≈ 1.5k tokens/s** for this 33.4M model on the M2 GPU regardless of batch shape once
  memory is not the limiter. For scale: the P5.5 pod lane moved 65 536 tokens/step; one such step is ≈45 s here.
  A 10M-token SFT/remediation stage ≈ 2 h; 100M tokens ≈ 19 h; the full 493k-step foundation run is not a local job.
- **Memory grows ~3.4× per doubling of context**, not 2×: the chunked attention (1024) bounds the forward
  working set but the autodiff graph still retains per-chunk score tensors, so the quadratic term survives
  under `Autodiff`. `grad_checkpointing` (Burn `BalancedCheckpointing`) cuts 4096×1 from 16.3 GB (swapping)
  to 11.9 GB at only ~1.05× step cost — it is the default for the local lane from now on.
- **Usable envelope on this box:** 2048 context up to batch 2 (7.6 GB), or 4096 context at batch 1 (11.9 GB),
  always with `grad_checkpointing`. Leave ≥3 GB for the OS; the 16 GB is unified, so the GPU and the process share it.
- **8192 is not reachable locally** without a memory change in the model (fused/flash-style attention that
  never materialises chunk scores under autodiff, or fp16/bf16 activations on wgpu). That is exactly the
  P6.6 conditional: either an `mlx-rs`-backed Burn backend fork with fused attention, or a capped pod for the
  8192 stages only. RoPE lets the 2048/4096 stages run locally and only the final context-extension stage move.

## Decision (feeds P6.2–P6.5)

1. Remediation v4 (from candidate 401,577) runs locally at `--context 2048 --batch-size 2` or
   `--context 4096 --batch-size 1`, `--features metal,grad_checkpointing`; budget the stage at ≤10M tokens.
2. Any stage that must see 8192 tokens is a P6.6 decision (operator: pod cap or backend fork), not a local run.
3. Next measurement worth taking before P6.5: fp16 activations on the wgpu backend (`Metal<f16>`) — halves the
   activation term if the loss stays numerically stable at this scale.

## Reproduce

```
cargo build --release -p grug-train --features metal,grad_checkpointing --example train_foundation
target/release/examples/train_foundation --manifest <corpus>/foundation-corpus.json \
  --artifacts-dir <run> --model small_8k --context 4096 --batch-size 1 --steps 5 \
  --eval-every 5 --eval-batches 1 --ckpt-every 100 --warmup-steps 2
python3 -c "import json;print([round(json.loads(l)['step_ms']) for l in open('<run>/train.jsonl')])"
```

The synthetic corpus generator used for the smoke was a throwaway (not committed); any
`FoundationCorpusManifest`-conformant corpus with > 8 194 eval tokens reproduces the table.
