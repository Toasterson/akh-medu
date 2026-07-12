# Phase 51 P3-R — Reasoning-Trace Training Results (trace vs answer-only A/B)

**Date:** 2026-07-11
**Status:** MEASURED — all four gates scored
**Plan:** [P3-R reasoning corpus](plans/2026-07-11-phase51-p3r-reasoning-corpus.md); **language:** [ADR 092](decisions/092-high-grug-v1-reasoning-register.md); **probe context:** [GWT emergence probes](2026-07-11-phase51-gwt-emergence-probes.md)
**Setup:** two identical 11.7M runs (seed 42, 40k examples × 8 epochs = 20k steps, per-run BPE, grad-clip, ~176 min trace / ~51 min answer on the RX 7900 XTX), differing ONLY in target rendering: full ADR-092 traces vs bare `call answer | value <x>`. Training stream mixes 20% broken-chain abstention examples (added after the first launch measured abstention P/R = 0.00 — gap emission is only learnable in-distribution; P3c's lesson re-confirmed). Definitive numbers from `eval_reason` n=256/axis on the matched step-16000 checkpoints (both arms peak ~15–16k and decline slightly by 20k; the JSONL time series shows the trajectory).

## Headline numbers (step 16000, n=256 per axis)

| axis | trace arm | answer-only arm |
|---|---|---|
| **axis-B answer accuracy** (held-out subjects, depths 2–4) | **0.840** | 0.516 |
| axis-B per-depth 2 / 3 / 4 | 0.848 / 0.822 / 0.846 | 0.500 / 0.493 / 0.549 |
| axis-B executability (engine replay) | 0.800 | — (no trace) |
| axis-B trace validity | 0.770 | — |
| **axis-A answer accuracy** (held-out depths 5–6) | 0.039 | **0.105** |
| axis-A per-depth 5 / 6 | 0.056 / 0.023 | 0.169 / 0.045 |
| axis-A trace validity | **0.000** | — |
| abstention precision / recall | 1.000 / 0.992 | 1.000 / 0.984 |

## Gate verdicts

1. **Format gate (trace_valid + exe ≥ ~0.9 on axis-B): NEAR-MISS.** Executability 0.800, trace validity 0.770 — clearly learnable (from 0 at step 0), but under the 0.9 bar. Error structure: accuracy 0.840 > exe 0.800 > valid 0.770, i.e. a slice of correct answers rides on imperfect traces. The syntax itself is not the blocker (abstention traces hit 0.984 validity); the residual is reasoning slips inside otherwise well-formed traces.
2. **Reasoning gate: MET.** 0.840 on held-out-subject multi-hop is far above both the same-relation guessing prior and the answer-only arm's 0.516. Notably **flat across depths 2–4** (0.85/0.82/0.85) — within the trained regime there is no per-hop decay; supervised steps cure error accumulation exactly as the GWT #3 literature said.
3. **Supervision differential on axis-A: NULL — and structurally negative.** Traces do not buy depth generalization: 0.039 vs the answer arm's 0.105 (the answer arm is *better* at depth 5, 0.169 vs 0.056). Cause is visible in the structure: **axis-A trace validity is 0.000** — a model trained on 2–4-step traces cannot render a 5-step trace at all, so the trace format acts as a length straitjacket out-of-depth, while the answer-only arm can still guess an entity from the prompt. This is the length-generalization wall with its mechanism exposed.
4. **Crossover migration: PARTIAL POSITIVE.** Per-checkpoint `probe_lens --reason` (2k grid, `runs/p3r-trace/probes/`): axis-B crossover migrates **5 → 4** (at step ~6000, stable after), abstention **4 → 3**, while axis-A stays pinned at **6** all run. Against the tool-calling corpora's flat 5–6 baseline, trace supervision pulls verbalization one layer down the stack — but only on axes the model actually masters. The imposed workspace forms where competence forms, and nowhere else.

## What this decides

- **The ADR-092 bet pays in-distribution: +0.32 answer accuracy (0.516 → 0.840) from supervising intermediate workspace states, at identical model/data/compute.** The trace arm's reasoning is also engine-checkable (0.80 replay rate) — the executability law working as designed.
- **Iteration/depth stays a System-2 property.** Four independent results now agree (four tool-calling levers, the emergence probes, Phase 41's scaling law, and this axis-A null): the 6-layer organ does not extrapolate structure it hasn't seen; the VSA loop owns unbounded depth. Per the plan's trigger (slot fires only if gate 4 flat AND gate 3 negative): gate 4 moved, so **the Goyal/Bengio architectural slot stays in the back pocket**.
- **Abstention is fully solved in-distribution** (P=1.000, R≈0.99 both arms) once the stream contains it — the gap token is format-independent.
- **Next levers, in information-per-cost order:** (a) mixed-depth curriculum including depth 5–6 in training to separate "can't render long traces" from "can't reason long" (cheap, one run); (b) error-structure pass on the 0.04 exe-vs-valid gap and the 0.16 acc-vs-exe gap (inspection, no GPU); (c) System-2 handoff — the organ emits trace-until-gap, the VSA loop continues (P6 seam, design); (d) the true intra-forward read-by-many Stage S on trace prompts (harness exists).

## Artifacts

`runs/p3r-trace/`, `runs/p3r-answer/` (checkpoints, per-500-step eval JSONL, `eval_reason.json` definitive evals, per-checkpoint probes under `runs/p3r-trace/probes/`). Harness commits: `3498587` (corpus + trace harness), `d487bc6` (abstention-stream fix). One first-launch abort is recorded in `d487bc6`'s message.

---

## Appendix: Mixed-Depth Curriculum (measured same day)

**Setup:** identical A/B rerun with `--train-depths 2,3,4,5,6 --axis-a-depths 7,8` (`runs/p3r-mixed-trace`, `runs/p3r-mixed-answer`; depth-8 sequences fit ctx: max 370 of 640). Definitive `eval_reason` n=256/axis: trace @ step 19999 (best 0.969 @ 19500, no late decline this time), answer @ step 16000.

| axis | trace | answer-only |
|---|---|---|
| axis-B acc (depths 2–6) | **0.957** | 0.477 |
| axis-B per-depth 2/3/4/5/6 | .980/.949/.981/**.940**/**.930** | .333/.475/.453/.560/.581 |
| axis-B exe / trace_valid | **0.956 / 0.949** | — |
| axis-A acc (depths 7–8) | 0.004 | 0.199 |
| axis-A per-step trace replay | 0.418 @ d7 / 0.277 @ d8 (trace_valid 0.000) | — |
| abstention P / R (depths 2–6) | 1.000 / 0.977 | 1.000 / 0.961 |

**Verdict on the gate-3 null: it was the render straitjacket, fully.** Depths 5–6 go 0.056/0.023 (held-out) → **0.940/0.930** (in-curriculum); reasoning is depth-flat across 2–6 at 11.7M. The wall moves exactly to the new curriculum edge (7–8 ≈ 0), so curriculum coverage — not architecture — buys any *bounded* depth; unbounded depth remains System-2's job, now measured rather than assumed (the P6 trace-until-gap handoff premise). Two supporting shifts: the **format gate clears** on the harder corpus (0.956/0.949 vs 0.800/0.770 — length diversity improved format discipline), and the **supervision gap widens** to +0.48 (0.957 vs 0.477; the answer arm gained nothing from the extra depths). Partial extrapolation nuance: at depth 7 the trace arm now replays 41.8% of steps before derailing (was: nothing valid at all out-of-depth) — length generalization degrades gradually at the step level even though whole-trace validity is 0.

**Crossover correction (probe sweep, matched depth-2–4 probe corpus, `runs/p3r-mixed-trace/probes/`):** the mixed model's axis-B crossover is **pinned at layer 5** all run and its mid-stack curve sits *lower* than the easy-curriculum model's (layer-4 top1_final 0.74 vs 0.93) — the opposite of the baseline's 5→4 migration. Reading: **crossover position tracks spare capacity relative to trained task difficulty, not workspace formation as such** — deeper chains consume more of the 6-layer stack, deferring verbalization. The ADR-092 "downward-migration = imposed workspace" signature is therefore only interpretable at *matched curriculum difficulty*; across difficulties it inverts. Both facts stand: supervision moves verbalization earlier at fixed difficulty; difficulty moves it later at fixed supervision.
