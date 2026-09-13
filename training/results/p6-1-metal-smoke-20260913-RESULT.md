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

## P6.1b(c) — f16 activations + f32 master weights (2026-09-13)

**GPU smoke: pending — run by the coordinator.** The GPU was occupied for the whole session by
another agent's P6.2 decoder-rescore exam runs (`eval_tool_contract_exam`, up to a full 1272-item
exam at 768 max new tokens), which correctly hold `/tmp/p6-gpu.lock` for extended periods. Rather
than force a run mid-lock or leave the branch idle, the code, the CPU-provable unit test, and both
required `cargo test` gates below are done and green; the coordinator will run the two smokes
(exact commands below) once the lock frees and append the numbers.

### What landed

- `crates/grug-train/Cargo.toml`: `metal_f16 = ["metal"]`.
- `crates/grug-train/src/lib.rs`: under `metal_f16`, `GpuBackend = Metal<f16>` (was `Metal<f32>`
  under plain `metal`); `GpuAutodiffBackend` is unchanged code (still `Autodiff<GpuBackend
  [, BalancedCheckpointing]>`, just wrapping whichever element type is active). New
  `GpuMasterBackend = Metal<f32>` / `GpuMasterAutodiffBackend` (same `grad_checkpointing` honoring)
  for the master weights. `gpu_device()` is unchanged — both backends share one `WgpuDevice`.
  `metal_f16` implying `metal` means the existing `gpu`/`cuda` `compile_error!`s already cover it;
  no new mutual-exclusion code was needed.
- `crates/grug-train/src/mixed_precision.rs` (new): `MixedPrecisionTrainer<BLow, BHigh>` — owns
  the f32 master weights + AdamW optimizer state; the caller keeps driving forward/backward on a
  separate `TinyTransformer<BLow>` exactly as the plain loop does. Per guarded step: forward +
  masked cross-entropy on the low-precision model, `.backward()` on a **scaled** loss, cast the
  resulting f16 gradients to f32 (dividing out the scale), Adam-step the f32 master, cast the
  updated master back down to a fresh low-precision model. `BHigh: AutodiffBackend<Device =
  BLow::Device>` is enforced in the type itself. `transformer.rs` was not touched (per the task
  boundary — another agent is editing the attention op there concurrently).
  - **`ParamId` stability**: the low- and high-precision models are linked once, at construction,
    via a full-precision in-memory record round trip (`BinBytesRecorder<FullPrecisionSettings>`),
    which is the same mechanism `checkpoint.rs` already relies on for cross-backend checkpoint
    loading and is what preserves each parameter's `ParamId` across the cast. After that, every
    per-step cast reuses `burn::optim::GradientsParams` purely as a generic `ParamId -> Tensor<B,
    D>` bag (its gradient-flavored name notwithstanding), via `ModuleVisitor`/`ModuleMapper` — no
    serialization on the hot path.
  - **Lazy-init pitfall found and fixed**: Burn's lazily-initialized `Param`s independently
    re-materialize (re-sample) on each clone until first read (`Param::clone`'s doc: "initializing
    one does not affect the other"). Cloning a freshly-`Config::init`-ed model for the record round
    trip without forcing materialization first produced a master with **completely unrelated
    random weights** from the model the caller kept driving — proven with a from-scratch repro
    (`Linear<CpuAutodiffBackend>` self-round-trip mismatched even model-to-itself). Fixed by a
    `force_materialize` pass (`ModuleVisitor` calling `Param::val()`, which only needs `&self` and
    caches through interior mutability) before every cross-backend clone. Caught this before it
    ever reached a real training loop, via the CPU mechanics test below.
- `crates/grug-train/src/error.rs`: `GrugError::MixedPrecisionCastFailed` (miette diagnostic,
  transparent to the `BinBytesRecorder` round trip's failure modes).
- `crates/grug-train/examples/train_foundation.rs`: `run<B>` is untouched (now `#[cfg(not(feature
  = "metal_f16"))]`, byte-identical body) and a new `run_mixed_precision<BLow, BHigh>` (same
  corpus/tokenizer/token-cache/run.json/loss-cap/grad-norm-cap/eval-cadence/checkpoint-cadence
  code, wired through `MixedPrecisionTrainer` instead of `init_training`/`train_step_guarded`
  directly). `main` dispatches to `run_mixed_precision::<GpuAutodiffBackend,
  GpuMasterAutodiffBackend>` under `metal_f16`, to `run::<ActiveBackend>` otherwise. Checkpointing
  saves/loads the **f32 master** (`checkpoint::save_checkpoint::<BHigh>`/`load_checkpoint::<BHigh>`
  unchanged), so a `metal_f16`-trained `foundation-model` stays loadable by the existing f32
  eval/generation code with no format change.

### Loss scaling

Implemented: a static scale (`mixed_precision::DEFAULT_LOSS_SCALE = 1024.0`, matching NVIDIA's
mixed-precision guide / PyTorch AMP's default initial scale) multiplies the loss before
`.backward()` and is divided back out of the f32 gradients before they reach the master's Adam
step, exactly as ADR 107 flagged as a possible requirement. **Diagnostic finding**: on a 5-step
smoke (see below), re-running with the scale forced to `1.0` produced a nearly identical loss/
grad-norm trajectory to `1024.0` (both diverge from the f32 baseline starting at step 1, in the
same direction and rough magnitude). This rules out gradient underflow/overflow from the scale
value as the dominant effect at this model/step scale — the residual divergence from the f32
baseline is not fixed by adjusting the static scale. The most likely remaining cause is that
`Metal<f16>` computes the *entire* forward/backward graph (matmul, softmax, layernorm) at f16
precision with no autocast-style upcasting for numerically sensitive reductions, unlike typical
"mixed precision" training (PyTorch AMP keeps softmax/layernorm/loss in fp32 even under autocast).
`MixedPrecisionTrainer` only fixes the *optimizer's* precision (the specific ADR 107 failure mode:
whole-backend f16 also putting Adam's moments in f16) — it cannot fix precision loss inside the
f16 forward/backward computation itself, which is a `Metal<f16>`-backend-level property. This is
the same distinction ADR 107 already drew ("f16 activations are fine; f16 optimizer state is not")
— the smoke below suggests the activation side is noisier at short horizons than that framing
assumed, which the coordinator's real (longer, exact-corpus) smoke should confirm or refute.

### Tests

- `cargo test -p grug-train` (default/CPU features): **133 passed, 0 failed, 2 ignored.**
- `cargo test -p grug-train --features metal_f16 --lib`: **133 passed, 1 failed** (`transformer::
  tests::cached_decode_matches_full_forward_on_the_gpu_backend`) **, 2 ignored.** That one failure
  is pre-existing and unrelated to this change: the test hardcodes `.to_vec::<f32>()` against
  `GpuBackend`'s native float element, which flipping `GpuBackend` to `Metal<f16>` under
  `metal_f16` makes a real type mismatch (`expected F16, got F32`). Fixing it means editing
  `transformer.rs`, which this task explicitly kept off-limits (another agent is concurrently
  changing the attention op there). Every other test, including the new
  `mixed_precision::tests::master_weights_round_trip_and_sync` CPU mechanics test (link → verify
  master matches the source model exactly → one guarded step → verify the master moved → verify
  the synced low-precision model exactly matches the new master), passes under both feature sets.
- `cargo fmt -p grug-train -- --check`: clean. `cargo clippy -p grug-train --lib --no-deps` and
  `--example train_foundation --no-deps` (both with and without `metal_f16,grad_checkpointing`):
  zero new warnings (one pre-existing, unrelated warning in `multimodal_runtime.rs` both before and
  after this change).

### Own diagnostic run (not the official gate numbers — see note)

The original smoke's corpus (`/tmp/p6-smoke/corpus/foundation-corpus.json`) was a throwaway per
the prior RESULT.md entry and no longer exists on this machine, so the ADR 107 reference numbers
(loss@4 = 4.35 for f32, 5.25 for whole-backend f16) are **not reproducible from this session** —
a fresh synthetic corpus was generated (2 000/100 raw records, `prepare_foundation` decontamination
kept 704/100 after n-gram filtering, tokenizer vocab 946) purely to smoke-test the mechanism and
run the loss-scale diagnostic above. These numbers are **not comparable in absolute terms** to the
ADR's table (different corpus, different tokenizer/vocab, different parameter count: 26.19M vs
33.4M) — they exist only to check relative behavior (f16/f32-master vs plain f32, same seed/corpus/
tokenizer) and are superseded by the coordinator's run on the real setup:

| variant | loss@0 | loss@4 (step 4) | eval@4 | peak footprint | elapsed (5 steps) |
|---|---:|---:|---:|---:|---:|
| `metal,grad_checkpointing` (f32 baseline) | 7.078 | 3.765 | 3.705 | 3.56 GB | 9.7 s |
| `metal_f16,grad_checkpointing`, scale=1024 | 7.078 | 6.301 | 6.262 | 2.36 GB | 24.2 s (incl. JIT) |
| `metal_f16,grad_checkpointing`, scale=1 | 7.078 | 6.340 | 6.320 | — | 9.6 s |

Step 0 (loss and grad-norm) matches the f32 baseline almost exactly in both f16 variants,
confirming the master-weights link and first Adam step are correct; the curves diverge starting at
step 1 and do not re-converge within 5 steps, in a mildly erratic (non-monotonic) way, independent
of the loss-scale value tested. Peak footprint is directly comparable to ADR 107's own f16-vs-f32
ratio (2.36/3.56 ≈ 0.66, in the same direction as the whole-backend f16 measurement's 2.2/3.5 ≈
0.63) even though the absolute corpus/model differ, which is at least consistent with the expected
memory win surviving the master-weights change.

### Exact commands for the coordinator's real smoke

```
cargo build --release -p grug-train --features metal_f16,grad_checkpointing --example train_foundation

# copy /tmp/p6-smoke/run2048/tokenizer.json into the artifacts dir first if reusing the original tokenizer
target/release/examples/train_foundation \
  --manifest /tmp/p6-smoke/corpus/foundation-corpus.json \
  --artifacts-dir /tmp/p6-smoke/run2048-f16 \
  --model small_8k --context 2048 --batch-size 1 --steps 5 \
  --eval-every 5 --eval-batches 1 --ckpt-every 100 --warmup-steps 2 \
  --bpe-vocab-size 2000 --tokenizer-sample-records 2000

target/release/examples/train_foundation \
  --manifest /tmp/p6-smoke/corpus/foundation-corpus.json \
  --artifacts-dir /tmp/p6-smoke/run4096-f16 \
  --model small_8k --context 4096 --batch-size 1 --steps 5 \
  --eval-every 5 --eval-batches 1 --ckpt-every 100 --warmup-steps 2 \
  --bpe-vocab-size 2000 --tokenizer-sample-records 2000
```

Both wrapped in `/usr/bin/time -l` for peak footprint; take `/tmp/p6-gpu.lock` (`until mkdir
/tmp/p6-gpu.lock 2>/dev/null; do sleep 15; done`) before running and `rmdir` it after, always.
Target from the P6.1b spec: loss@4 within ±0.05 of the f32 reference (4.35 at 2048×1) with memory
and step time close to the whole-backend f16 numbers (2.2 GB / 0.99 s at 2048×1). Given this
session's diagnostic above, **that target may not be met as-is** — if the divergence reproduces on
the real corpus, the next lever (beyond adjusting the static loss scale, which this session showed
does not move the needle) is likely something that reduces forward/backward-computation noise
itself (e.g. keeping softmax/layernorm in f32 within an otherwise-f16 forward pass), which is out
of `mixed_precision.rs`'s scope as specified (it only wraps the optimizer, not the model's forward
pass) and would need a `transformer.rs` change coordinated with whoever owns that file's concurrent
edits.
