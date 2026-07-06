# Phase 51 P0→P1a-2 — BabyLLM Tool-Calling Scaling Results

- **Date:** 2026-07-05
- **Status:** Empirical results (reproducible). Companion to [Phase 51 plan](plans/2026-07-05-phase51-grug-brain-baby-llm.md) + [ADR 087](decisions/087-grug-brain-baby-llm-training.md).
- **Author:** Fable 5 via Claude Code, fleet-executed, for Till.
- **Artifacts:** `crates/grug-train` + `crates/grug-data` on develop; run logs under `runs/p1a/` + `runs/p1a2/` (gitignored). Commits: P0 `774181c`, P1a `8de8161`, P1a-2 `6ce6e7b`+`3bd37c8`.

---

## 0. Headline

Training a from-scratch tiny transformer (13M params, Burn/ROCm, AMD 7900 XTX) to emit JSON tool calls, evaluated on two honest holdout axes, produced a clean three-step finding:

1. **Scale fixes JSON *syntax*.** A 496K byte-level model (P0) never emitted valid JSON (0.000). A 13M BPE model (P1a) emits it ~1.0.
2. **Scale does *not* fix schema-grounding.** At 56 tools the 13M model memorized: argument-correctness plateaued at ~0.10 and generalization to unseen tools *collapsed* (one canned call for everything). More training did nothing — a flat plateau across 5 epochs.
3. **A data-distribution change fixes in-distribution grounding.** Exploding the catalog to 1000 procedural tools (few examples each, so memorization is impossible) took the *same 13M model* to **arg-correct 0.69 / schema 0.87** on held-out phrasings — a 7× jump — with no extra parameters. Zero-shot to *entirely unseen* tools improved off the floor but stays weak (~2–4%): the next wall.

**The methodological win:** the pre-committed rule "if scale is insufficient, reach for frontier technique, not brute force" was triggered by a measured plateau and paid off — a data-distribution move, not more parameters, delivered the capability.

## 1. Setup

- **Framework:** Burn 0.21, `default-features=false, features=["ndarray","autodiff","std"]` (CPU) + `Rocm` CubeCL backend (GPU, opt-in `gpu` feature). No burn-dataset (dodges the libsqlite3-sys conflict). Hand-rolled Adam SFT loop.
- **Hardware:** AMD RX 7900 XTX (gfx1100), ROCm 6.3.4. ~80–105 ms/step at 13M. (Needed a `hipconfig` shim — `scripts/hipconfig-shim/` — on PATH; cubecl-hip-sys calls it at build *and* runtime.)
- **Model:** `TinyTransformer<B>` — token embedding + sinusoidal positions + N decoder blocks (causal self-attention behind an `Attention<B>` trait seam + MLP + LayerNorm) + LM head. P1a/P1a-2 config: d_model 384, 6 layers, 6 heads, ctx 640 → ~13M params.
- **Objective:** single-stage teacher-forced SFT. Prompt = serialized offered tools + user request; target = expected-call JSON; prompt tokens masked from the loss.
- **Data (`grug-data`):** programmatic generator, deterministic. Two orthogonal, structurally-leak-proof holdout axes:
  - **Axis A — whole-tool holdout:** tools reserved entirely from training → tests in-context schema-reading (zero-shot to unseen tools).
  - **Axis B — held-out phrasing:** for *trained* tools, request phrasings (templates, not re-fuzzed args) reserved for eval → tests generalization to unseen wordings of known tools.
- **Eval:** greedy-generate on frozen n=128 holdout batches per axis every N steps; score valid-JSON / schema-adherence / argument-correctness + an anti-collapse diversity/entropy metric.

## 2. P0 — feasibility (496K, byte-level)

R1 ✅ Burn compiles + trains on CPU. R2 ✅ a transformer trains (loss 5.79→0.14). R3 ✅ trains on the 7900 XTX via ROCm/CubeCL (~1.8–2× CPU at this toy scale, dispatch-bound). **Valid-JSON = 0.000 flat** — the expected floor at 496K/byte-level/128-examples/300-steps. Feasibility proven; capability deferred.

## 3. P1a — scale-up (13M, BPE, 56 tools)

20k examples × 5 epochs (6250 steps), BPE vocab 2810, train loss → 0.15.

| Axis | valid-JSON | schema | arg-correct | diversity |
|---|---|---|---|---|
| B (held-out phrasing) | ~1.00 | ~0.34 | **~0.10** | ~0.32 |
| A (whole-tool holdout) | ~0.94 | **0.00** | 0.00 | **0.008** |

**Reading:** valid-JSON *solved* by scale (0.000 → 1.0). But arg-correctness plateaued at ~0.10 by step 500 and never moved across the remaining ~5700 steps; axis-A schema stayed dead 0.00 with diversity pinned at 1/128 (one memorized call for every unseen tool). **Diagnosis:** with only 56 tools, memorizing training tools is easier than learning the general "read schema → emit correct call" skill, so the model took the shortcut. The flat plateau proves this is a capability ceiling at this *data distribution*, not under-training — so more parameters is the wrong lever.

## 4. P1a-2 — frontier move: explode the catalog (13M, 1000 procedural tools)

Same model. 1000 procedural tools (850 train / 150 whole-tool holdout), ~47 examples/tool (vs P1a's ~500), 40k examples × 4 epochs (10k steps), procedural phrasings that embed every required argument value verbatim (so copying is *learnable* by construction), BPE vocab 2953, train loss → 0.073.

| Axis | valid-JSON | schema | arg-correct | diversity | vs P1a |
|---|---|---|---|---|---|
| B (held-out phrasing) | 0.99 | **0.87** | **0.69** | 0.88 | arg 7×, schema 2.6× |
| A (whole-tool holdout) | 0.99 | 0.02–0.04 | ~0.02 | ~0.03 | off the dead floor |

**Reading:** denying memorization *forced* schema-grounded argument production. On trained tools (axis B) the same 13M model went from arg-correct 0.10 → **0.69** and schema 0.34 → **0.87** — a real, working in-distribution tiny tool-caller that **essentially meets the P1a gate** (arg 0.69 vs target ≥0.7, still rising at run end). On entirely-unseen tools (axis A) the collapse *broke* — schema off 0.00, diversity off 1/128, occasional correct zero-shot calls — but true generalization to novel schemas stays weak (~2–4%).

## 5. Interpretation

- **Three distinct capabilities, three distinct levers.** JSON *syntax* ← model scale. In-distribution schema-*grounding* ← data distribution (tool diversity / anti-memorization). Zero-shot *schema-generalization* ← still open (candidate levers: many more tools, contrastive/hard-negative tool sampling, an explicit copy/pointer bias, or more capacity — TBD, and the point where "frontier vs brute force" gets re-litigated).
- **Frontier-over-brute-force, validated.** Scale did nothing for grounding (measured flat plateau); a data change delivered a 7× jump on the same parameters. This is the project's stated principle earning its keep.
- **Apertus was correctly *not* on this path.** Argument-copying on synthetic templates is not a natural-language-comprehension gap, so foundation pretraining wasn't the fix — consistent with the P1a/P1b split.

## 6. Caveats (honest)

- **Arg-correct 0.69 is a hair under the 0.7 gate** — reported as "essentially met, still climbing," not "passed."
- **The tokenizer lever never actually pulled.** A bug caps the BPE training corpus at `min(train_examples, 5000)` examples regardless of data size, so vocab hit 2953 instead of the requested 8000. There is likely real headroom on arg-correctness from fixing this.
- **n=128 eval noise** — axis-B schema bounces ~0.70–0.87 across late checkpoints; treat single numbers as ±.
- **Process hygiene:** 3 of 4 fleet agents committed to `develop` directly despite worktree-branch instructions (code tested/correct each time; develop integrity verified after each). Likely the Agent worktree checking out `develop` itself — a risk to watch.

## 7. Reproducibility

- Data: `Dataset::new_sized(seed, n_tools, holdout_frac)` (procedural) / `Dataset::new(seed)` (56 hand-authored). Deterministic by seed.
- Train (GPU): `source scripts/train-env.sh && cargo run -p grug-train --example train_p1a --release --features gpu -- --n-tools 1000 --holdout-frac 0.15 --train-examples 40000 --epochs 4 --batch-size 16 --learning-rate 3e-4 --eval-every 500 --ckpt-every 1000 --seed 42 --artifacts-dir runs/p1a2`.
- Eval time-series: `runs/<run>/eval_axes.jsonl` (rows tagged by `axis`). Checkpoints: `runs/<run>/checkpoints/step_*/` (Burn `Recorder`, resumable via `--resume`).

## 8. Open next steps (for a fresh decision)

1. **Close the gate cheaply:** fix the BPE-corpus cap + a little more training → push arg-correct decisively past 0.7.
2. **Zero-shot (axis A):** the real remaining frontier — more tools / contrastive negatives / copy-pointer bias.
3. **P1b (Apertus):** in-distribution tool-calling works; bring in foundation pretraining for natural-language comprehension.
