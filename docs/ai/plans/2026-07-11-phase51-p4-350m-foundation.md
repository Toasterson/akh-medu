# Phase 51 P4 — The 350M Foundation: High-Grug-v2 Reasoning Organ at Scale

**Date:** 2026-07-11
**Status:** Planned (design ratified via user grilling; build gated on 11M ablations)
**Language:** [ADR 092-v2](../decisions/092-high-grug-v1-reasoning-register.md) (Q1–Q8)
**Curriculum:** [Phi memo](../2026-07-11-phi-curriculum-350m-memo.md); **corpus base:** [Apertus scoping](../2026-07-11-phase51-p1b-apertus-scoping.md)
**Premises (measured):** [P3-R + mixed-depth](../2026-07-11-phase51-p3r-results.md), [ADR 093 handoff](../decisions/093-dual-process-trace-handoff.md)

## Goal

Scale the reasoning organ 11.7M → **350M**, thinking in the High-Grug-v2 register, on a **sovereign Phi-informed curriculum** — the first model that (a) reasons deductively (rule-based, not just graph-walking), (b) offloads *facts* to the KG and *iteration* to System-2, and (c) is trained with no cloud dependency in its shipped recipe. This is the bet of ADR 087, at the scale where it either holds or breaks.

## Corpus mixture (Phi two-phase, sovereign)

| Phase | Share | Source | Sovereign? |
|---|---|---|---|
| **1 — comprehension** | ~40% | FineWeb-Edu `sample-10BT` + FineWeb2-HQ `deu_Latn` (reservoir-sampled) + apertus-pretrain-swiss | yes (ODC-By/Apache-2.0; inherits Phi's classifier-filter via FineWeb-Edu) |
| **2b — engine reasoning** | ~35% | thinking-engine-generated High-Grug fluency + gloss↔trace pairs + multi-hop/rule traces | **yes — zero teacher, correct by construction** |
| **2a — synthetic reasoning prose** *(optional, measured arm)* | ~15% | Ollama-Cloud teacher (experiment) → local open-weight if it wins (shipped) | experiment-only; **provenance-flagged** |
| repeats over 2 | to budget | — | phi-4: epochs-over-synthetic beat fresh web for reasoning |

## Lighter knobs (recommendations — confirm/adjust)

- **Token budget:** **7–10B as a checkpoint-gate**, extend only if the eval curve is still climbing (Phi data-optimal + measure-then-scale). Not 50B upfront on one RX 7900 XTX.
- **Narrow-first:** run a **phi-1-small-style narrow 350M** (engine reasoning corpus only) as the first 350M checkpoint — phi-1-small@350M (45% HumanEval) is our only primary-source datapoint at scale; validate the recipe narrow before the full mixture.
- **Decontamination:** 13+7-gram + embedding near-dup vs our own eval holdouts before training (phi-1/phi-4).
- **Tokenizer:** definitive NL BPE (~8–32k, byte-fallback, ~75:25 EN:DE) + the ADR-092-v2 reserved governed region (operators/classifiers/markers + dark SymbolId block) + slots.

## Staged build order (each gate is a number)

1. **Corpus arms** (`grug-data`): propositional + rule-based generators; rule-KG + modus-ponens trace rendering; classifier-typed slots; `select/bind/fire/chain` component scorer in `reason::replay`; High-Grug-v2 renderer (reserved-token IDs).
2. **11M ablations → capacity ladder 11M→50M→100M** (`grug-train`): rule arm (Q4), classifier arm (Q2), pretrain-fluency arm (Q7), optional teacher arm (Q8) — all component-decomposed, both-arm ladder (Q5).
3. **Gate readout** → decide exactly what enters the 350M corpus (only what clears its 11M gate; capacity slope tells fix-it from scale-it).
4. **Definitive tokenizer** built on the chosen corpus.
5. **350M pretrain: narrow-first → full mixture** (the big compute, gated behind 2–3). Burn/ROCm, checkpoint-every-N, NaN-covering watcher, setsid-detached.
6. **Reasoning-SFT at 350M** + re-measure the ADR-093 handoff economics at scale (does +0.48 supervision and the neural-fraction hold at 25×?).

The 350M run sits at step 5, **behind** the cheap ablations — the same discipline that turned every prior hunch into a number. No big compute on an unmeasured curriculum choice.

## Gates

1. **Rule-based gate (11M):** `select/bind/fire` off the floor; a **positive capacity slope** on `chain` (11→100M) = "scale it"; flat-at-floor with floored `select` = "broken, fix the corpus." A partial-yes at 11M is the awesome result.
2. **Classifier gate (11M):** typed slots cut off-path derailment vs bare (else classifiers are dead tokens — drop them).
3. **Pretrain-fluency gate (11M/50M):** engine-fluency-in-pretrain beats cold-SFT downstream (else drop to embedding-init only).
4. **Teacher-arm gate (11M):** teacher prose beats the teacher-free base (else stay teacher-free; if it wins, reproduce locally before shipping).
5. **Narrow-350M gate:** recipe reproduces phi-1-small-shaped competence before committing the full mixture.
6. **Scale gate (350M):** the P3-R supervision differential and handoff neural-fraction survive 25× scale-up.

Nulls are results (house rule): a flat capacity slope on rules is a real finding about deductive reasoning's floor at ≤100M and reshapes the 350M target rather than wasting its compute.

## Slice-2 ablation results (2026-07-18) → 350M GO, with an LR caveat

The 11M ablations ran (full data + tables in `runs/p4-ablation-RESULT-2026-07-13.md`, gitignored). Verdicts:

- **Q4 — rule-based gate: MET, strong.** The 11M organ does near-perfect in-distribution first-order modus ponens over *in-prompt* rules: `select`/`fire` → 1.000, **`chain` 0.918**, answer 0.992 (converged ~step 7500). The capability jump past graph-walking is real at 11M.
- **Q2 — sort determinative (typed `x1·SORT` vs bare `x1`): NULL under full information; typing KEPT by decision.** Matched runs are statistically identical (typed chain 0.918 vs bare 0.934, bare marginally ahead). Cause understood: in a fully-known corpus the coarse sort is *redundant with the fine predicate*, so the type-gate has nothing to prune. Typing is retained as a designed uncertainty-prior (its value is in unknown/open-world regimes this corpus doesn't exercise); **no retest** — decided on principle.
- **Q5 — capacity ladder (11M→50M; 100M skipped): the length-generalization wall HOLDS.** len-gen `chain` = 0.000 at both 11M and 50M — capacity does not crack unseen-depth composition (fourth replication of "bounded depth learnable, unbounded = System-2's job"). Depth is architectural, not capacity-gated. *In-dist* slope is inconclusive: 50M was undertrained + LR-unstable (chain still climbing at step 15000, trace-validity depressed) — **not** a "bigger is worse" result.
- **★ LR-AT-SCALE CAVEAT (the load-bearing carry-forward):** lr `3e-4` works at 11M but destabilizes 50M's trace-validity. **The 350M run must use a lower LR (~1e-4 or below) + warmup + cosine decay + more steps** — reusing 11M's recipe verbatim will reproduce the 50M instability. Bigger models are much slower/step (50M ~726ms vs 11M ~235ms) → plan **many-days on-and-off, checkpoint-resumable** training.

**Status: cleared to build the 350M foundation in a fresh session.** Corpus/register/harness are proven (slice 1–2); gates Q2/Q4/Q5 understood; the only new tuning the scale-up needs is the LR schedule above. Operational rules learned: launch-verify by process+mtime (not deterministic log content); never run two GPU train jobs concurrently (hangs the card); eval has no KV-cache (keep eval-every high during training, precision eval at end).

## Out of scope (P5+)

P6-proper broadcast-read + dual-substrate cost ledger (ADR 093); TMS/epistemic corpora (ADR-092 deferred set); RL post-SFT (ADR 087 curriculum); inter-akh Grug.

## Open decisions for the human

Token-budget point in 7–10B; the 2a:2b ratio (15:35 is a first guess); whether the teacher arm runs in the first ablation wave or after the sovereign base is proven; the 1.5-Pints follow-up ([arXiv:2408.03506], numbers unverified).
