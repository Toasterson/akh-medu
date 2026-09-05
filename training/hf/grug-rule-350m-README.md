---
license: mit
language:
  - en
tags:
  - burn
  - rust
  - from-scratch
  - reasoning
  - deductive-reasoning
  - training-stability
  - research-artifact
pipeline_tag: text-generation
library_name: burn
---

# grug-rule-350m — a from-scratch 350M deduction organ (research artifact)

⚠️ **This is not a usable chat model.** It is a 341.9M-parameter decoder-only
transformer trained **from scratch** on a synthetic deductive-reasoning DSL
(effective vocab: 355 BPE tokens). It speaks nothing else. Its value is as a
**reproducible research artifact** for two results and one training-stability
case study — and as a checkpoint you can re-evaluate yourself for ~$0.30.

Part of
[akh-medu at source commit `053c6651`](https://code.aopc.cloud/toasterson/akh-medu/commit/053c66516074c5cdc13438df230c4a92bc7a1125)
/ Project Grug-Brain: the bet that a tiny model can hold *reasoning* while an
external knowledge engine holds *facts*.

## The two results

1. **Narrow deductive competence scales cleanly.** On held-out
   modus-ponens chains over in-prompt rules (depths 2–4), n=512:
   answers **0.928**, trace validity 0.928, and the strict
   `select/bind/fire/chain` decomposition scores **1.000/1.000/1.000/0.935**
   — exceeding the 11M reference model (chain 0.918) that gated this run.
   This competence is **learned, not latent**: a generic 600M pretrained
   model (Qwen3-0.6B), given few-shot exemplars, scores *below the
   majority-class baseline* on the same test (it imitates the trace
   vocabulary and copies exemplar label frequencies — format mimicry, not
   derivation).
2. **Capacity does not buy length generalization.** At unseen depths 5–6:
   chain **0.000** — identical to the 11M and 50M models. A 31× parameter
   increase moved this number not at all. The wall is architectural.

## The training-stability case study

The run survived what killed two predecessors: **epoch-periodic gradient
storms** — bursts of batches with finite loss (~ln(vocab), i.e. collapsed-
to-uniform logits) but **non-finite gradients**, recurring at the *same
stream positions every epoch*, across different random inits, at learning
rates from 4.4e-5 down to 5e-6. Two content-side root causes were tested and
**falsified** (context-length truncation: max sequence 677 vs ctx 1024;
batch-content logit divergence: trigger batches produce logits ≈25,
identical to controls). What stands is a data-position × optimizer-state
interaction on a fully-converged task (Adam second-moment collapse; cf.
Adaptive Edge of Stability, arXiv 2207.14484; PaLM §5.1's batch-skip
experience, arXiv 2204.02311).

The kill mechanism, confirmed in source: per-parameter-tensor Norm(1.0)
gradient clipping divides an inf-norm gradient by infinity → **NaN weights
in a single optimizer step**. The mitigation stack that let this run finish:

- linear warmup + cosine decay (peak 5e-5),
- a loss-spike gate (skip updates > 4× loss-EMA),
- a **global pre-clip grad-norm gate** (skip non-finite or > 5× gnorm-EMA).

501 poisoned updates refused over 17.1h on one A100; zero weight damage;
~$28 of compute. Because the corpus is deterministically seeded, **the
storms are byte-reproducible** — they arrive at the same steps if you rerun
the recipe, which makes this checkpoint's lineage a rare controlled testbed
for training-instability research.

## Files

- `model.mpk` — Burn 0.21 checkpoint (step 28000; best; the three final
  checkpoints score identically to 3 decimals)
- `tokenizer.json` — byte-level BPE trained in-process (vocab 3000,
  saturates at 355 on this templated corpus)
- `eval_rule.json` — the definitive n=512 evaluation output
- `eval_rule.jsonl` — 12-point training-time eval series
- `logs/` — all four runs' training logs, including both fatal NaN runs
  (`p350m-run1-nan.log`, `p350m-run2-skipflood.log`) — the storms and the
  gate refusals are visible line by line

## Reproduce (from scratch, ~$30, or any 24GB+ GPU)

```bash
git clone https://code.aopc.cloud/toasterson/akh-medu.git
cd akh-medu && git checkout 053c66516074c5cdc13438df230c4a92bc7a1125
bash training/scripts/setup-cuda.sh   # rust + CUDA build + 60-step smoke
bash training/scripts/run-350m.sh     # detached, resumable, guarded
bash training/scripts/status.sh p350m
```

No dataset download: the corpus (40,000 examples, seed 42) and the tokenizer
are generated in-process. Evaluate this checkpoint without training:
`eval_rule --n 512 --resume <checkpoint dir>` (see
`training/results/README.md`). The cross-model exam harness
(`dump_rule_eval` / `score_rule_jsonl` + an MLX runner) lets you sit any
model against the same scorer.

Full run report: `training/results/p350m-RESULT.md` in the repo.

## Architecture

d_model 1024 · 24 layers · 16 heads · d_ff 4864 · ctx 1024 · learned
positional embeddings · fp32 training (Burn 0.21 / CubeCL, CUDA) ·
Adam + per-tensor Norm(1.0) clip + the gate stack above.

## Native multimodal development status (2026-09-04)

This published `model.mpk` is **text-only**. It contains neither the new visual
stem nor multimodal training and must not be advertised as an OCR model.

The companion research artifacts under `research/adaptive-vision-v1/` document
an implemented continuous-exam controller and two native visual-prefix
candidates sharing the BabyLLM decoder. The first CPU smoke reduced
teacher-forced loss from 5.6197 to 0.0930 but emitted the same transcript for
all three images (`cross_semantic_collapse_rate=1.0`), and the raw-patch stem
failed held-out translations. Those negative results remain published.

Failure-directed refinement selected a compact stride-16 convolutional stem.
Its 3.31M shared-decoder checkpoint passed the deliberately narrow V1
training-overfit gate: 128/128 exact clean-line transcripts, zero CER/WER/blank
and zero cross-semantic collapse. It also loaded into the local CPU backend and
produced 3/3 exact outputs under an OS network-deny sandbox. The experimental
checkpoint, Adam state and exact compatibility tuple are under
`research/adaptive-vision-v1/checkpoints/compact-clean-line-v1/`; they do not
replace root `model.mpk`.

This is not a usable general OCR claim. The checkpoint saw every scored string
during training. Different-renderer/real images, unseen strings/fonts,
EN/DE/FR, corruptions/layouts, metamorphic groups and text-retention gates are
still unrun or red. Those boundaries are part of the published model card,
results and checkpoint README.

## License

MIT (published weights, tokenizer, evaluation outputs, and logs). The linked
training/evaluation source code is GPL-3.0-only.
