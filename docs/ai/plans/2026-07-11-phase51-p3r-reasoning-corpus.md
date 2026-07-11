# Phase 51 P3-R — Format-First Reasoning Corpus + Intermediate Supervision

**Date:** 2026-07-11
**Status:** Active
**ADR:** [092 — High Grug v1 reasoning register](../decisions/092-high-grug-v1-reasoning-register.md)
**Parent:** [Phase 51 plan](2026-07-05-phase51-grug-brain-baby-llm.md) P3 + §2a GWT amendment #3; probe evidence in [2026-07-11 results](../2026-07-11-phase51-gwt-emergence-probes.md)

## Question

Can the 13M organ *reason* — chain prompt-supplied facts through an imposed High-Grug workspace — and does supervising the intermediate workspace states (vs the answer alone) buy correctness and length generalization, as GWT #3 claims? Phase 41's VSA loop (depth 4→32 at budgets 100→1000) is the System-2 baseline the traces must beat or complement.

## Corpus (grug-data, `reason` module)

- **Synthetic KG, deterministic from seed:** typed entity namespaces (`p__` person, `c__` city, `r__` region, `k__` country, `o__` org …), functional relations forming chains (`lives_in: p→c`, `part_of: c→r`, `in_country: r→k`, `works_for: p→o`, `based_in: o→c` …). Composed query relations (`region_of = part_of ∘ lives_in`, etc.) generated procedurally to depth d.
- **Example =** prompt: shuffled `fact <s> <rel> <o>` lines = the gold chain's facts + distractor facts (same relations, other entities; count is a knob) + `goal <query> <subject>`; target: ADR-092 v1 trace (`step n | recall …` → `derive … | by i j` → `call answer | value <o>`), or `gap` abstention when the chain is deliberately broken.
- **Splits/axes (P1/P2 discipline):**
  - **Train:** depths d ∈ {2,3,4}, entity vocab A, ~40k examples.
  - **Axis-B (in-distribution):** held-out examples at trained depths, unseen entity *combinations*.
  - **Axis-A (length generalization):** depths d ∈ {5,6} — never trained; the intermediate-supervision claim lives or dies here.
  - **Abstention axis:** broken-chain goals → gap (reusing the P3c mechanism).
- **A/B arm:** identical examples, target = `call answer | value <o>` only (no trace) — the "answer-only" control.
- **Scoring:** `TraceReplayer` executability rate (per-op replay against prompt facts, `by`-reference validity), answer accuracy, per-depth breakdown, abstention precision/recall, anti-collapse (distinct-answer ratio, entropy).

## Training (grug-train)

`train_reason` + `eval_reason` examples; identical recipe discipline to P2/P3a: 13M `TinyTransformerConfig::p1a`, fresh per-run BPE, seed 42, 20k steps, batch 16, lr 3e-4, Adam grad-clip Norm 1.0, PROMPT-LEN guard, `--ckpt-every 2000`, eval JSONL, frozen holdouts. Two runs: `p3r-trace` and `p3r-answer` (A/B). GPU, setsid-detached, NaN-covering watcher.

## Gates

1. **Format gate:** trace validity + executability ≥ ~0.9 on axis-B (the format is learnable at 13M) — else the v1 syntax needs rework before anything else is interpretable.
2. **Reasoning gate:** axis-B answer accuracy meaningfully above the relation-prior baseline; report per-depth.
3. **Supervision differential:** trace arm vs answer arm on axis-A (depths 5–6) — the headline number.
4. **Crossover migration:** `probe_lens` on p3r-trace checkpoints (2k-step grid) — does the verbalization crossover move down the stack relative to the flat 5–6 baseline?

Nulls are reportable results (house rule): if the trace arm shows no axis-A lift, that is a real finding about imposed workspaces at 13M and gates P1b/architecture-slot decisions.

## Out of scope

Gloss bilinguality (P1b), TMS/epistemic ops (v2 corpora), VSA-projector entities (Phase 26 gate), the Goyal/Bengio architectural slot (fires only if gate 4 is flat AND gate 3 is negative), RL (post-SFT per ADR 087 curriculum).
