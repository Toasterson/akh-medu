# Lilith quality slice v2 — 2026-09-22

This versioned slice separates artifact extraction from semantic correctness and evaluates fresh logic, bounds, and error-handling tasks. The bounded GPU comparison completed successfully, but **neither one-step SFT nor one-step KLPO improved the measured primary score**. The scratch KLPO adapter is mechanically valid and cold-reloads exactly; it is not recommended for promotion on quality evidence.

## Frozen contract and task split

The [primary protocol](frozen-protocol.json) and [task manifest](tasks.json) were frozen before generation. A response may be raw Python or one complete enclosing Markdown code fence. The fence is extracted without changing its contents. The scalar reward is binary semantic correctness of that extracted artifact; `presentation` and `formatting_compliant` are recorded separately and do not affect reward.

Four training tasks cover `bounded_normalization` and `alignment`. Four development tasks use disjoint `packed_fields` and `range_windows` families. References pass all finite cases and all authored negative controls fail. These are public finite checks, not proofs or sealed benchmarks.

## Primary ROCm comparison

[Run 2](live/run2/result.json) loaded the pinned Gemma 4 parent, reset the same rank-2 final-query-projection LoRA for each arm, and used paired full-support samples. SFT and KLPO each performed one optimizer step. KLPO consumed four completed training trajectories with rewards `[1, 1, 0, 0]`. The eight-position log-probability drift was `0.13850489`; the predeclared one-position fallback reduced it to `3.15e-7` before updates.

| Arm | Train | Disjoint dev | Complete | Overall |
|---|---:|---:|---:|---:|
| Initial | 2/4 | 2/4 | 7/8 | 4/8 |
| SFT | 2/4 | 2/4 | 7/8 | 4/8 |
| KLPO | 2/4 | 2/4 | 7/8 | 4/8 |

The [independent replay](independent-report.json) reproduced every reward and provenance check. All three arms generated the same observed task outcomes under paired RNG:

- Two training artifacts passed. `align-up-power2/2` was semantically plausible but used constant-tuple membership, which is valid Python outside the frozen grammar. `aligned-offset-valid/2` evaluated `a % b` before validating `b`, causing division by zero at `b=0`.
- Two development artifacts passed. `field-fits-unsigned/2` omitted the required `1 <= b <= 16` validation and incorrectly accepted width zero. `windows-overlap/2` exhausted the generation budget and remained a syntax-invalid truncation.
- The [unscored grammar diagnostic](unscored-diagnostics.json) expands only the constant-tuple membership into admitted equality comparisons and passes all 374 finite cases. This does not change the official grammar failure or scores.

The run took 1,155.1 seconds and peaked at 24,297,685,504 GPU bytes. The initial, SFT, and KLPO tensor fingerprints are recorded in the result. The KLPO candidate is:

`live/run2/klpo-adapter/adapter_model.safetensors`

File SHA-256: `80be5b0fbeb264292f84f0e757adf84259a213c4c3654dc87bd3c448fb1ff808`

Tensor fingerprint: `c583780e5eca2e517464d2e50a53ebfb5d874affc1a856834e2ef262e7cf3db3`

## Retention and cold reload

The [retention protocol](retention-manifest-run2.json) was frozen before retention generation. It evaluates four previously exposed training-family tasks with the same artifact extraction and generation budget, never using them for updates.

The [retention run](live/retention-run2/result.json) scored **4/4 for initial, SFT, and KLPO**, with all 12 trajectories completed. The [independent retention replay](retention-independent-report.json) verifies each reward, task, policy fingerprint, input hash, candidate file hash, and aggregate. No retention regression was observed on this small pack. In PID 56551, distinct from training PID 44108, the KLPO artifact reproduced the stored full-vocabulary fixed-prefix log probabilities exactly.

## Failures and verification

The first primary loader attempt is retained under `live/run1`: an incompatible gfx10 override on the native gfx1100 host exited 139 before parent loading completed, generation, or updates. The first retention entry point under `live/retention-run1` exited before parent loading because the reused loader lacked its module-level Torch binding. Both have explicit `failure.json` receipts; run 2 is the sole collecting attempt for each phase.

The v2 task and unchanged trajectory-core suite has [20 passing tests](unit-tests.log). The independent primary replay has [four adversarial tests](report-tests.log), covering forged reward evidence, duplicate rows, and provenance drift. Original 2026-09-21 data, deployed weights, and services were not modified.

This experiment supports the artifact/reward contract, meaningful failure capture, one-step mechanics, retention on four prior tasks, and exact HF/ROCm reload. It does not establish a coding-quality gain, statistical superiority, native illumos behavior, or a promotion candidate.
