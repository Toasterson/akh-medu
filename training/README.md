# `training/` — Phase 51 Grug-Brain GPU training workspace (H100 cloud handoff)

**Current R1 direction (2026-09-11):** use the [integrated Akh plan](../docs/ai/plans/2026-09-11-akh-350m-graph-grounded-release-plan.md) and [unfamiliar-homelab release exam](../docs/ai/plans/2026-09-11-akh-homelab-release-exam.md) to coordinate 350M–1.1B capacity, graph learning, High Grug, all Dojo forms, emotional intelligence, Portuguese basics and Mac deployment. Historical recipes and results below remain references. Broad Portuguese training is no longer a dependency of consultation; no old training command is an automatic continuation of this proposal.

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
cd akh-medu && git checkout main
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

## Combined preparation: Portuguese and process consultation

Recorded at Till's request on 2026-09-10: prepare corpus acquisition and the two
pending training stages as one coordinated batch. Both stages are deferred until
an eligible parent has completed its existing terminal gates. The checklist
below is the implementation handoff; acquisition and training are still pending.

**Corpus selection decision:** use accessible corpus components regardless of
license category. Apply quality, language coverage, deduplication, and evaluation
separation checks. Retain source IDs, revisions, and hashes for reproducibility.

| Stage | Corpus preparation | Implementation still needed |
| --- | --- | --- |
| [Portuguese](../docs/ai/plans/2026-09-10-portuguese-language-stage.md) | Acquire the listed Portuguese prose and instruction sources; author Portuguese knowledge/tool/Dojo episodes; prepare independent language and task holdouts. | Source exporters, tokenizer audit, Portuguese workflow/generator coverage, and candidate/retention evaluation. |
| [Process consultation](../docs/ai/manuals/2026-09-01-babyllm-dojo-curriculum.md#10-process-consultation-curriculum-added-2026-09-08) | Assemble and author complete episodes for all six kata, including Portuguese alongside English/German/French; preserve corrections, outcomes, and participant ownership. | Separate episode-preserving converter, consultation evaluator, and scenario-family holdouts. |

Combined execution checklist:

- [ ] Record the actual parent run outcome and exact checkpoint, configuration,
  tokenizer, and tool-contract hashes with its gate evidence.
- [ ] Prepare both corpora in one acquisition/authoring pass. Record source
  revisions, stable IDs, processing configuration, seeds, file hashes, and
  record/token counts in a shared acquisition inventory. Keep a separate
  versioned corpus manifest and output directory for each stage.
- [ ] Deduplicate across both corpora and existing replay data; freeze held-out
  document/scenario families before training. Keep translated equivalents in
  the same partition and future episode outcomes out of earlier model inputs.
- [ ] Audit the selected parent's tokenizer on Portuguese and complete both
  stages' converters, generators, evaluators, and targeted implementation checks.
- [ ] Freeze token budgets, mixture weights, stop conditions, evaluation sample
  sizes, promotion thresholds, and retention tolerances for both stages.
- [ ] Execute sequentially. The planned order is Portuguese language adaptation
  and instruction/task training, then consultation training from the eligible
  Portuguese candidate. Pass the applicable gates between stages and record the
  exact parent selected for each; a failed candidate does not advance the chain.
- [ ] Evaluate the final candidate on both Portuguese capability and consultation
  behavior, plus English/German/French, knowledge, tool, and Dojo retention.
  Record artifact hashes, results, and failure slices before certification.

Preparation can be coordinated while checkpoints, corpus manifests, and results
remain separately inspectable. This handoff launches no acquisition or training
process and changes no active run. It is not a runnable combined launcher yet.

## Deferred tail stage: process consultation dojo

Added at Till's request on 2026-09-08: train the
[process consultation curriculum](../docs/ai/manuals/2026-09-01-babyllm-dojo-curriculum.md#10-process-consultation-curriculum-added-2026-09-08)
after the currently running LLM run and its already-planned terminal gates.
This is a documented follow-on stage, not an automatic job appended to a live
supervisor. Preserve that run's pinned corpus, checkpoint lineage, tokenizer,
configuration, and stop conditions. A failed or stopped run is not a successful
parent merely because its process has exited.

Handoff for the follow-on stage:

1. Record the completed run's actual outcome and select an eligible parent by
   exact checkpoint/config/tokenizer/contract hashes and gate evidence. For
   the tool-native continual adaptation path, use the certified parent required
   by [ADR 102](../docs/ai/decisions/102-tool-native-learning-model.md) and the
   [operator runbook](../docs/ai/manuals/2026-09-04-adaptive-exams-native-vision.md).
   If that parent is unavailable, keep this stage pending and record the gap.
2. Prepare a versioned consultation corpus and independent held-out suite:
   six forms, complete episodes, mistaken first interpretations, correction and
   repair, differing participant needs, and useful non-intervention. Freeze
   scenario-family splits, provenance, scoring rubric, sample sizes, and
   promotion thresholds before training.
3. Add the episode-preserving converter and consultation evaluator. The existing
   character converter does not ingest this specification. Keep new outputs in
   a separate stage directory; do not overwrite the current datasets/adapters.
4. Train a new candidate sequentially from the selected compatible parent,
   retaining the original dojo, knowledge, and tool capabilities through replay
   and retention evaluation. Pin the mixture and budget in the stage manifest.
5. Compare parent/candidate consultation results, pass the applicable existing
   release gates again, and record exact artifact hashes and failure slices.
   Certification, activation, and publication retain their separate boundaries.

Preparation status: curriculum and handoff recorded; new corpus, converter,
evaluator, training, and promotion evidence are pending. No active-run status
is inferred from this note, and no training process is launched by it.

## Deferred language stage: Portuguese

The [Portuguese language-stage plan](../docs/ai/plans/2026-09-10-portuguese-language-stage.md)
records candidate corpora from the Manacá/Tucano ecosystem, source-level
provenance, tokenizer checks, and Portuguese instruction/tool/Dojo coverage.
PT-BR is the initial working target. Preparation and training remain pending;
follow the [combined preparation checklist](#combined-preparation-portuguese-and-process-consultation)
for acquisition, parent selection, and the planned order of both stages.
