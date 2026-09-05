# Compact native text-vision V1 A40 result

**Date:** 2026-09-04  
**Verdict:** **V1 clean-line overfit gate passed; model promotion remains held**

## Question and frozen gate

Can a 2--10M-parameter BabyLLM learn a native pixel-to-byte path through a
compact visual stem and the same causal decoder used for text?

The gate was declared before the run:

- at least 128 deterministic `training_sibling` cases;
- greedy exact transcript rate at least 0.98;
- cross-semantic collapse rate at most 0.01;
- teacher-forced loss must decrease;
- visual tokens must remain below 25% of context.

This is an overfit/capacity gate. It is not an unseen-string, font,
multilingual, corruption, layout, or independent-audit result.

## Configuration

```bash
cargo run --release -p grug-train --features cuda \
  --example vision_compact_overfit -- \
  --steps 4000 --cases 128 --batch-size 32 \
  --learning-rate 0.001 --min-learning-rate 0.00001 \
  --artifacts-dir /workspace/vision-v1-checkpoint-cosine
```

| Item | Value |
|---|---:|
| model parameters | 3,310,340 |
| decoder | d=192, 6 layers, 6 heads, FF=768 |
| compact stem | 24 hidden channels, effective stride 16 |
| image geometry | 384x64 RGB |
| visual prefix | 96 tokens |
| decoder context | 512 tokens |
| visual context fraction | 18.75% |
| corpus | 128 English, clean, line-layout training siblings |
| corpus seed/index stride | 51,100 / 15 |
| schedule | cosine, 1e-3 -> 1e-5 |

The A40 environment, reduced CUDA workspace lockfile and exact source hashes
are preserved in [`vision-v1-run-bundle/`](vision-v1-run-bundle/). The exact
run source is SHA-256
`36431471aee29913369f9e213b0782a6b6c8941ec489c30f662efffd209f6615`.

## Failure-directed intervention ledger

| Attempt | Stop point | Signal | Bounded response |
|---|---:|---|---|
| 1 | step 450 | no checkpoint destination was configured | stop; add artifact destination rather than risk an unrepeatable long run |
| 2 | step 950 | 96 visual tokens consumed 37.5% of a 256-token context | stop; restore the declared 512-token context, making the fraction 18.75% |
| 3 | step 1,250 | fixed-rate loss alternated around 0.019/0.034 for repeated windows | stop; retain log and use one predeclared cosine schedule |
| 4 | step 4,000 | scheduled run broke through the floor and remained finite | freeze and run greedy evaluation |

The maintained training example now emits complete model+Adam snapshots every
500 steps when an artifact directory is supplied. A separate two-step CUDA
smoke proved that a progress snapshot writes `model.mpk`, `optim.mpk`, then
`meta.json`, and that the final checkpoint is independently marked complete.
The live run predates that hardening, so its byte-exact source remains in the
bundle rather than being rewritten after the fact.

## Result

| Measurement | Result | Gate |
|---|---:|---:|
| initial loss | 5.659536 | -- |
| final loss | 0.000052 | decreased: pass |
| training time | 2,543.93 s (42m 23.93s) | recorded |
| A40 greedy inference | 19.63 s / 128 (0.153 s/case) | recorded |
| exact transcript | **1.0000 (128/128)** | >= 0.98: **pass** |
| CER / WER | **0.0000 / 0.0000** | recorded |
| blank rate | **0.0000** | recorded |
| unique semantic prediction rate | **1.0000** | recorded |
| cross-semantic collapse | **0.0000** | <= 0.01: **pass** |
| visual context fraction | **0.1875** | < 0.25: **pass** |
| `v1_gate_passed` | **true** | **pass** |

`metamorphic_consistency` is `0.0` because this rung deliberately has one
render per semantic item. There is no metamorphic denominator, so it is
**unmeasured**, not a failure and not a pass. Multi-render consistency remains
a later promotion gate.

The raw result is
[`vision-v1-clean-line-a40.json`](vision-v1-clean-line-a40.json), SHA-256
`308eb3dd4694732c83f4c5fd0cd029f375f3da750b3a306869e0f1bf5891566b`.

## Corpus and quarantine evidence

The exact 128 images were materialized separately for publication. Their
manifest has SHA-256
`c1456f2701eaa012aa6d0ae6788e7e51989f6e00659e7c230a375b390c89527a`
and contains 128 unique normalized transcripts. Identity scoring was 128/128
exact with zero CER/WER/collapse.

A temporary domain-separated exam was used only for the sibling-admission
audit, then removed from the working path. The retained aggregate report shows:

- exact transcript overlap: 0;
- exact pixel overlap: 0;
- max trigram Jaccard: 0.3721 (limit 0.80);
- minimum foreground perceptual-hash distance: 19 (floor 10);
- held-out axis: content bank;
- admission: true.

This is a same-renderer quarantine check, not the generator-independent audit
required for promotion.

## Checkpoint and local-offline replay

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `model.mpk` | 13,250,031 | `95a131de80436d5ba3dd8b3c1cb3fa6a87cd446389e272cec8d8fbb9406c2dec` |
| `optim.mpk` | 26,499,107 | `4a31ec01182f69e7c43425a72555084c39a68873ac59e1043c474b8611a40e4c` |
| `meta.json` | 713 | `7fb4ec488b0616ab389aeb1ffbc2e29dc133b7f9e55fa824603c945f4f042562` |

The checkpoint was then loaded through Burn ndarray on the local Mac and run
inside `sandbox-exec` with all network operations denied. Three cases were
3/3 exact with zero CER/WER/collapse in 12.27 model seconds (4.09 s/case).
`/usr/bin/time -l` recorded 44,253,184 bytes peak RSS and zero messages sent or
received. This proves the provider-free runtime path and cross-backend load;
the sample is too small to establish a latency distribution.

## Decision

Publish the checkpoint only under
`research/adaptive-vision-v1/checkpoints/compact-clean-line-v1/`, labelled
`experimental clean-line training overfit`, with its optimizer, compatibility
tuple, exact corpus and all failure boundaries. Do **not** overwrite or relabel
the existing text-only 350M checkpoint.

Do not advance to the 10--30M or 350M multimodal rungs yet. The independent
renderer/real-image audit, unseen-string/font EN/DE/FR evaluation, corruption
and layout slices, metamorphic groups, and text-only retention comparison are
still red or unrun. Passing this rung selects the compact shared-decoder path
for those experiments; it does not establish usable OCR.

The disposable secure-cloud A40 pod ran for 6,201 seconds at $0.49/hour
(observed estimate: $0.8440) and was deleted after artifact retrieval. Its
task-specific account SSH key and local private key were also removed; the
unrelated pre-existing account key was preserved.
