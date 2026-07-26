# `training/` — Phase 51 Grug-Brain GPU training workspace (H100 cloud handoff)

Everything needed to take the Phase-51 reasoning organ from the measured 11M
ablations to the **350M foundation run**, on a rented NVIDIA box (H100), from a
clean `git clone`. Designed for **many-days, on-and-off** training: every stage
is checkpoint-resumable and writes results as small committable files.

- **What's already measured** (don't re-run these): see
  [`../docs/ai/plans/2026-07-11-phase51-p4-350m-foundation.md`](../docs/ai/plans/2026-07-11-phase51-p4-350m-foundation.md)
  §"Slice-2 ablation results". Short version: the rule gate is **met** at 11M
  (in-dist `chain` 0.918), typing is **kept** by decision, and the
  length-generalization wall is **architectural, not capacity-gated**
  (`chain`=0.000 on unseen depths at both 11M and 50M).
- **The one tuning lesson that must carry forward:** `lr 3e-4` works at 11M and
  **destabilizes 50M**. See [LR at scale](#lr-at-scale) — do not reuse the 11M
  recipe verbatim at 350M.

## Layout

```
training/
  README.md            <- you are here: the runbook
  scripts/
    setup-cuda.sh      one-shot cloud-box bootstrap (toolchain + build check)
    run-350m.sh        the 350M foundation run (resumable, sequential)
    resume.sh          resume any run from its newest checkpoint
    status.sh          honest status of any run (live-verified, no stale-log traps)
  configs/
    350m.env           the 350M hyperparameters (single source of truth)
    11m-repro.env      the measured 11M ablation config (regression baseline)
  results/             small committable artifacts: eval JSONL, RESULT.md, tables
```

Model checkpoints and raw logs live in `runs/` (gitignored — they're GBs).
**`training/results/` is for the small stuff you want in git.**

## Quick start on a fresh H100 box

```bash
git clone https://code.aopc.cloud/toasterson/akh-medu.git
cd akh-medu && git checkout develop
bash training/scripts/setup-cuda.sh      # verifies toolchain + builds the CUDA binaries
bash training/scripts/run-350m.sh        # detached, resumable; writes runs/ + training/results/
bash training/scripts/status.sh p350m    # honest live status any time
```

## Backend: CUDA vs ROCm

The GPU backend is a cargo feature; pick exactly one (enabling both is a
compile error, by design):

| box | feature | build |
|---|---|---|
| local AMD 7900 XTX | `gpu` (ROCm) | `source scripts/train-env.sh && cargo build --features gpu --release` |
| **cloud NVIDIA H100** | **`cuda`** | `cargo build --features cuda --release` |

Example/binary code is **vendor-agnostic** — it calls `grug_train::gpu_device()`
rather than naming `RocmDevice`/`CudaDevice`, so the same source builds for both.

## LR at scale

The measured failure mode, and the reason the 350M config differs from 11M's:

| model | lr | outcome |
|---|---|---|
| 11M | 3e-4 | converged by ~step 7500, `chain` 0.918 ✅ |
| 50M | 3e-4 | trace-validity depressed, `chain` still climbing at step 15000 ❌ undertrained/unstable |
| **350M** | **~1e-4 + warmup + cosine** | see `configs/350m.env` |

Bigger models also cost much more per step (11M ~235 ms, 50M ~726 ms on a 7900
XTX; H100 will be faster but 350M is ~7× the 50M). **Budget days, not hours**,
and rely on `--ckpt-every` + `resume.sh`.

## Hard-won operational rules (violating these cost real time)

1. **Never run two GPU training processes concurrently.** On the local AMD box
   this *hung the card* and required a reboot. Run stages **sequentially** —
   `run-350m.sh` does this for you. (Also: don't run `eval_rule` while a
   training job is live.)
2. **Verify a launch by process + log mtime, never by log content.** The corpus
   is deterministic-seeded, so a *stale* log from a previous run looks identical
   to a fresh one. `status.sh` does this correctly.
3. **Eval is expensive** — `generate` has no KV-cache, so an eval over a large
   holdout can cost more than the training between evals. Keep `EVAL_EVERY`
   high and `HOLDOUT` small *during* training; run the precision eval
   (`eval_rule --n 512`) **at the end**, when the GPU is free.
4. **Quote the best checkpoint, not the final one.** High-LR runs dip and
   recover; the final step is not reliably the best.
