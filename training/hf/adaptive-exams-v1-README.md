# Adaptive exams v1: public bootstrap artifacts

This directory is the public, reproducible bootstrap for BabyLLM's continuous
exam loop and native text-recognition work.

## What is here

- `demo-training-sibling/`: 30 semantic items rendered under five deterministic
  variants (150 PNGs), plus exact transcript/provenance manifests and checksums;
- `compact-clean-line-training/`: the exact 128-image corpus consumed by the
  first compact shared-decoder run, plus its corpus metadata;
- `schemas/`: JSON Schemas for exam plans, counted observations and OCR
  predictions;
- `docs/`: research, accepted ADRs, execution plan, runbook and reviewer record;
- `results/`: captured token-budget, raw/compact stem, decoder-smoke, corpus
  quarantine, A40 V1 and local-offline outcomes.

The demo images use generator `babyllm-vision-text-v1` and compiled renderer
`font8x8-unicode-0.3/babyllm-raster-v1`. Host fonts, network access and cloud
OCR are not involved.

## Important boundaries

- This public image set is a **training-sibling demo**, not a sentinel.
- A fresh seed from the same renderer is still not an independent audit.
- Audit promotion requires a different renderer or real/human-verified data.
- Exact and near-duplicate exam cases must never enter training.
- Predictive precursors are shadow-only until lead time is proven and do not
  alter family priority while disabled; separate catastrophic blank/invalid
  counts can still stop a run.
- Per-family sentinel difficulty is pinned at registration, so frontier
  adaptation cannot change the frozen sentinel case set.
- EN/DE/FR here means the initial Latin-script scope, not broad multilingual OCR.
- The V1 clean-line **training-overfit** gate is green (128/128 exact,
  CER/WER/collapse 0), but general OCR capability is red. These artifacts prove
  wiring, replay, bounded steering and narrow capacity—not unseen-image reading.

## Reproduce locally

From the corresponding `akh-medu` source tree:

```bash
cargo run -p grug-data --example generate_vision_exam -- \
  --out demo-training-sibling \
  --seed 740059131 --count 30 --variants 5 --difficulty 2 \
  --domain training-sibling
```

Score predictions with `score_vision_exam`, feed its counted outputs to
`adaptive_exam_controller`, and run `audit_vision_sibling` before admitting any
new sibling corpus. See `docs/operator-runbook.md` for the full command flow.

Regenerate the exact V1 training corpus with:

```bash
cargo run -p grug-data --example generate_vision_exam -- \
  --out compact-clean-line-training \
  --seed 51100 --count 128 --index-stride 15 --variants 1 \
  --difficulty 1 --domain training-sibling
```

The matching experimental checkpoint and compatibility tuple live under
`research/adaptive-vision-v1/checkpoints/compact-clean-line-v1/` in
`OpenFlowLabs/grug-rule-350m`.

## Licences

The generated data and documentation bundle is MIT. The linked Rust generator,
training and evaluation source remains GPL-3.0-only. The compiled `font8x8`
dependency retains its own upstream licence.
