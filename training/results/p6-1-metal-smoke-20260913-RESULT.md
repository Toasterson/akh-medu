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

## P6.1b — recompute-backward attention (2026-09-13)

**Verdict: 8192 context now fits and trains, both at batch 1 and batch 2, on the same 16 GB M2 — the
NO-GO above is superseded.** ADR 107 (`docs/ai/decisions/107-training-memory-turboquant-vs-recompute-attention.md`)
diagnosed the blocker as the retained `[chunk x kv_len]` attention-score tensor under `Autodiff`, not
anything `grad_checkpointing` could fix (`Matmul` is unconditionally `ComputeBound` in `burn-autodiff`,
so its output is never recomputed regardless of checkpointing strategy). P6.1b landed two changes in
`crates/grug-train/src/transformer.rs`:

- **(a) Causal-prefix K/V slicing** in `chunked_causal_attention`: chunk *i* now scores only keys/values
  `[0, kv_offset + end)` instead of the full `kv_len` (`kv_offset = kv_len - query_len`, 0 outside cached
  decode). Exact, not approximate — masked-out columns beyond that point could only ever contribute `-inf`.
- **(b) A custom recompute-backward op**, `AttentionBackend::chunked_causal_attention` / `RecomputeAttention`,
  implemented via `burn_autodiff`'s own `ops::Backward<B, N>` extension mechanism (the same one `Matmul`/`Add`
  use internally). Forward keeps only the attended output `O` and each row's log-sum-exp; backward recomputes
  `P = exp(S - LSE)` per chunk and derives `dQ`/`dK`/`dV` from it (FlashAttention's backward, Dao et al. 2022).
  Unit-tested against the naive autodiff-differentiated chunked path to 1e-4 (forward + dQ/dK/dV) at
  `L=256`/`chunk=64` and `L=200` (not a multiple of chunk), plus an end-to-end gradient check through a real
  `TinyTransformer` block's attention projections (`cargo test -p grug-train --lib`, both CPU and
  `--features metal`).

**Deviation from the Burn-book "custom autodiff" pattern as literally described**: `burn-autodiff` 0.21.0's
`graph`/`tensor` modules (which own `AutodiffTensor<B>`, `NodeRef`, etc. — the types the *public*
`ops::Backward`/`ops::OpsPrep` API already takes and returns) are declared `pub(crate)`, so none of it is
actually reachable from a downstream crate even though every field/method on them is individually `pub`.
Fixed with a two-line vendor patch (`vendor/burn-autodiff/src/lib.rs`, wired via the workspace's existing
`[patch.crates-io]` — see the root `Cargo.toml` comment): widen those two module declarations from
`pub(crate)` to `pub`. No item added, removed, or changed; zero behavior change for every existing caller
(only `grug-train` depends on `burn` in this workspace).

### Setup

Same machine/model as the P6.1 table above. Corpus: the `run_id: p6-1b-smoke` synthetic manifest prepared for
this task (704 training / 100 evaluation records, `/tmp/p6-smoke/corpus`) — sufficient for the 2048 run, but
its 100-record eval split BPE-encodes to only 6 055 tokens, under the `context+1` packing floor for 8192.
**Deviation**: rather than a hand-edited/fabricated corpus, `evaluation/part-00000.jsonl` was mechanically
extended to 400 records (the original 100 plus 3 duplicated-with-new-`id` copies) via a throwaway example
using the crate's own `FoundationRecord`/`update_foundation_fingerprint`/`FoundationCorpusManifest` public
API (so the manifest is a genuinely valid, self-`verify()`-ing corpus, not hand-faked hashes), written to
`/tmp/p6-smoke/corpus8k` and used only for the two 8192 runs; the 2048 run below used the original corpus.
Also **no pre-existing `run2048/tokenizer.json`** was present in this environment (the referenced path was
from a since-cleared prior session); a same-corpus tokenizer was instead read from a concurrent session's
`/tmp/p6-smoke/run2048-f16/tokenizer.json` (946-token BPE vocab over the identical `p6-1b-smoke` manifest,
seed 51004) and reused across all three runs below for consistency, rather than retraining fresh each time.
Because of the different corpus/tokenizer, **the loss numbers below are not directly comparable to ADR 107's
own 4.35 baseline** (different vocab), and 8192's peak footprint is not directly comparable to a same-corpus
`recompute` vs. `naive` A/B (the naive path OOMs/swaps at 8192 by design, per the P6.1 table above, so there
is no same-run baseline to diff against locally) — the improvement is read qualitatively (fits vs. does not
fit) rather than as a matched-pair percentage.

Command (identical shape to the P6.1 reproduce block above):
```
cargo build --release -p grug-train --features metal,grad_checkpointing --example train_foundation
target/release/examples/train_foundation --manifest <corpus>/foundation-corpus.json \
  --artifacts-dir <run> --model small_8k --context <ctx> --batch-size <batch> --steps 5 \
  --eval-every 5 --eval-batches 1 --ckpt-every 100 --warmup-steps 2 \
  --bpe-vocab-size 2000 --tokenizer-sample-records 2000
```
No run swapped (all three completed well under the 300 s watchdog); GPU access was serialized against a
concurrent session via `/tmp/p6-gpu.lock`.

### Measurements

| context | batch | steady step | tokens/s | peak footprint | loss@step4 | outcome |
|---:|---:|---:|---:|---:|---:|---|
| 2048 | 1 | 1.11 s | ≈1 840 | 1.68 GB | 3.76 | OK (`grad_checkpointing`, same shape as the 1.41 s / 3.5 GB P6.1 row above) |
| 8192 | 1 | 7.45 s | ≈1 100 | 4.67 GB | 3.67 | **OK — was killed/`>21GB swap` pre-P6.1b** |
| 8192 | 2 | 15.80 s | ≈1 040 | 7.02 GB | 3.60 | **OK — untested pre-P6.1b (8192×1 alone already OOM'd)** |

Steady step = mean of steps 1..4 (step 0 pays one-time CubeCL kernel JIT). Peak footprint is `/usr/bin/time
-l`'s "peak memory footprint" line. All three runs used `--features metal,grad_checkpointing`.

### Reading

- **8192 moved from "not reachable locally" (ADR 107 decision 4) to fitting comfortably at batch 1 *and*
  batch 2** with headroom to spare on a 16 GB machine (7.02 GB peak at batch 2, vs. the ADR's own >21 GB/killed
  pre-P6.1b measurements and its ≈2.7 GB *analytic* attention-only estimate — the gap is the ≈4 GB of
  weights/Adam/other activations the analytic table already itemized separately).
- **2048 got faster and lighter too, not just 8192**: 1.11 s/1.68 GB here vs. the P6.1 baseline's 1.41 s/3.5 GB
  at the same shape — consistent with (a) alone already roughly halving average per-chunk score-tensor size
  (causal-prefix slicing), stacked with (b) no longer retaining any of it for backward. This is a different
  corpus/tokenizer/build than the P6.1 row, so treat the specific delta as directional, not a controlled A/B;
  the 8192 result (impossible → comfortably fits) cannot be explained by that confound.
- **Throughput at 8192 (≈1.0-1.1k tok/s) is in the same ballpark as the P6.1 ≈1.5k tok/s ceiling** measured at
  2048/4096, i.e. the recompute (one extra forward-shaped pass per backward chunk) does not appear to be a
  large multiplicative tax at this shape — the run stayed in the same 1-2k tok/s regime the box was already at,
  rather than dropping by the ~2x a naive "redo the forward" accounting might suggest.
- Per P6.1b's own gate ("Go/no-go for the GPU box moves behind P6.1b... only a *measured* miss after P6.1b
  (throughput, not memory) triggers P6.6"): memory is now a non-issue at 8192; throughput (≈1.0k tok/s) is the
  remaining number to weigh against a remediation-v4 stage's token budget when P6.2+ schedules the 8192 stage.
