# p350m — Phase 51 P4 narrow-first 350M foundation run — RESULT

- **Date:** 2026-09-01 → 2026-09-02 (run 4 of 4; runs 1–3 died, see §3)
- **Verdict: GATE 5 PASSED.** The narrow deductive-rule recipe reproduces —
  and slightly exceeds — the 11M organ's competence at 347M params:
  in-dist `chain` **0.935** (n=512) vs the 11M gate of 0.918. Length
  generalization stays at **chain 0.000** — 31× capacity buys zero length
  transfer, confirming the architectural wall is not a capacity artifact.
- **Hardware:** Runpod A100 SXM 80GB (secure, US-WA-1), $1.59/hr,
  `runpod-torch-v240` image + rustup, CUDA 12.4, Burn 0.21 `--features cuda`.
  Wall clock 17.1h train + ~1.5h evals; ~$30 total across all four runs.
- **Config:** `training/configs/350m.env` — `p1a_350m` (341.9M actual),
  batch 16, ctx 1024, 30,000 steps (12 epochs × 40k examples), **LR 5e-5**
  (halved from the planned 1e-4, see §3), warmup 2000 + cosine → 5e-6,
  seed 42, BPE 3000 → 355 effective.

## 1. Definitive numbers (eval_rule --n 512, checkpoint step_00028000)

| axis | ans | exec | valid | select | bind | fire | chain |
|---|---|---|---|---|---|---|---|
| in-dist (d2,3,4) | 0.928 | 0.992 | 0.928 | 1.000 | 1.000 | 1.000 | **0.935** |
| len-gen (d5,6) | 0.494 | 0.328 | 0.076 | 0.425 | 1.000 | 0.135 | **0.000** |

Per-depth in-dist answers: d2 **1.000**, d3 **1.000**, d4 **0.796** — the
entire in-dist error mass sits at depth 4. Checkpoints 24000/26000/28000
score identically to 3 decimals (fully converged plateau); 28000 quoted.
Training-time n=128 evals showed 1.000 across the board from step 22500 —
the n=512 numbers are the honest ones (small-sample flattery at n=128).

Scaling context: 11M needed ~7500 steps to chain 0.918; 350M passed it by
step ~5000 (chain 0.689 at eval@5000, 0.967 at 15000, saturated after).
Loss floored at ~0.5 by step 600 and ~0.01 by step 2500 — **the final ~25k
steps bought polish at depth ≤4 and nothing at depth ≥5.**

## 2. What this decides (per the P4 plan)

1. **Recipe scales**: warmup+cosine + the §3 guard stack trains 350M
   stably where the naive recipe (constant LR, loss-only NaN check) died
   twice before step 900.
2. **Length wall confirmed at scale**: chain 0.000 at unseen depths for
   11M, 50M, and now 350M. Capacity is exonerated; the P4 plan's full
   mixture / curriculum / architecture work is the only live lever.
3. Runs shorter than 30k steps suffice at this corpus size — by step
   ~7500 everything measurable had converged. Post-convergence training
   mostly manufactured instability exposure (§3).

## 3. The instability saga (runs 1–4) — read this before any future run

Four runs, same stream (seed 42), three outcomes:

| run | config | outcome |
|---|---|---|
| 1 | LR 1e-4 warmup+cosine, loss-NaN abort every 10 steps | **dead @ step 880**: healthy 0.5 loss → single-step NaN → weights gone |
| 2 | + skip non-finite-loss batches | **dead @ step ~879–929**: finite-loss (5.8) batches with NaN *gradients* stepped anyway; weights died, 51-skip flood |
| 3 | + loss-spike gate (EMA cap), LR 5e-5 | killed at ~35min to deploy run 4's telemetry (healthy while it ran) |
| 4 | + **global pre-clip grad-norm gate** + gnorm telemetry | **survived 17.1h**: 501 poisoned updates refused, zero weight damage, ended at step 29,593/30,000 via the 500-skip cap |

**Confirmed kill mechanism (runs 1–2):** a batch produces finite loss but
non-finite gradients; Burn's `Norm(1.0)` clipping is **per parameter
tensor** (burn-optim source), so an inf-norm tensor is scaled by
1/inf → inf·0 = **NaN written into weights in one optimizer step**. The
loss-only guard cannot see this coming. The global-norm gate
(`train_step_guarded`, `crates/grug-train/src/train.rs`) checks before
`optim.step` and refuses the update.

**Storm phenomenology (run 4, 501 gated events):** storms are bursts of
3–12 consecutive batches, loss pinned at ~5.82–5.86 ≈ ln(355) = uniform,
gnorm NaN. They are **epoch-periodic**: the same stream offsets
(mod 2500: ~874–882, ~1220–1231, ~2084–2095 + scattered singles) fire
every epoch, across different random inits (runs 1, 2, 4 all stormed at
~874–880) and at every LR from 4.4e-5 down to 5e-6.

**Forensics — two hypotheses tested, both negative (tools committed):**
- *Context truncation* (`examples/forensic_batches.rs`): FALSIFIED. Max
  sequence in the whole 40k stream is 677 tokens vs ctx 1024; zero
  over-ctx, zero target-cut, zero empty-mask anywhere; trigger windows
  are statistically ordinary in depth/length/label mix.
- *Batch-content logit divergence* (`examples/probe_logits.rs`, z-loss
  motivation per Wortsman et al. 2309.14322): NOT SUPPORTED. Trigger
  batches through ckpt 28000 max out at |logit| ≈ 25.2 vs controls 25.1 —
  identical, and ~3.5× under fp32 exp overflow. Caveat: storm-moment
  weights were never checkpointed, so a transient state can't be fully
  excluded.

**What stands:** the PaLM conclusion (2204.02311 §5.1) — spikes arise
from the *combination* of specific data positions with the optimizer
state, amplified here by second-moment collapse on a fully-converged task
(Adam's v → 0, effective step 1/√v explodes; cf. Adaptive Edge of
Stability, 2207.14484, and the Adafactor RMS(update) analysis,
1804.04235). The data is ordinary; the encounter is not.

## 4. Recommendations for future runs (P5+)

1. **Keep the guard stack permanently** (loss-spike gate + global
   grad-norm gate + warmup/cosine). It is cheap (~one scalar readback per
   step) and it is the difference between run 1 and run 4.
2. **Capture storm-moment state**: on the first skip of a burst, dump the
   batch indices and write an emergency checkpoint (bounded, e.g. max 3
   per run). This is the only way to close the state×data question the
   forensics above could not.
3. **Add RMS(update) telemetry** (Adafactor gauge: RMS of g/√v̂ ≈ 1 when
   healthy) — gnorm-NaN arrives with zero lead time; the update-ratio
   drifts early. Consider raising Adam ε / β₂ for any long
   post-convergence run.
4. **z-loss / QK-layernorm as cheap insurance** at ≥350M — no longer
   claimed as this run's root cause, but literature-standard and
   harmless.
5. **Don't train 50× past the floor.** Size steps to the corpus:
   everything here was decided by step ~7500. If long runs are needed,
   expect storms and budget the skip cap accordingly (~15–18 skips/1000
   steps here; the 500 cap tripped at 98.6% — set it from measured rate).
6. Ops: `pkill`/`pgrep -f` over SSH must use the `[b]racket` pattern (a
   plain pattern matches the remote shell itself — cost us a deploy and
   made early watcher liveness checks lie); `runpodctl` v2.12 has no
  `--terminate-after` (cost guard = monitoring + volume checkpoints).

## 5. Artifacts

- `training/results/p350m-eval.jsonl` — 12-point training-time eval series
- `training/results/p350m-eval_rule.json` — definitive n=512 eval (ckpt 28000)
- Volume `akh-medu-babyllm-p350m` (US-WA-1, 150GB): checkpoints
  step_00002000…step_00028000 (model+optim), tokenizer, all four runs'
  logs (`p350m-run1-nan.log`, `p350m-run2-skipflood.log`,
  `p350m-run3-pregate.log`, `p350m.log`)
- Forensic tools: `crates/grug-train/examples/forensic_batches.rs`,
  `crates/grug-train/examples/probe_logits.rs`,
  `training/scripts/monitor_precursors.py` (precursor watcher analyzer)
- The dojo character corpus (`data/phase51/character/`) rode the branch to
  the volume; it plays no role in this run's training — it is staged for
  the character-SFT stage (see `docs/ai/manuals/2026-09-01-babyllm-dojo-curriculum.md`).
