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

## 8. P1a-3 — a three-lever ablation of the remaining walls (2026-07-05)

After P1a-2 left two gaps — arg-correct just under the 0.7 gate, and axis-A (zero-shot to unseen tools) near zero — three levers were tested, each isolated:

**Step 1 — more training (gate close).** 4→8 epochs (same 13M model + data). Arg-correct was *still climbing* at 4 epochs, not plateaued; 8 epochs took axis-B **arg 0.69 → 0.78, schema 0.90** — gate met. (The BPE-corpus-cap "fix" was a red herring: the procedural corpus is diversity-limited, so 5k→60k examples only moved vocab 2953→3045; more training was the real lever.)

**Step 2 — harder data (force schema-reading).** Same-family, similarity-ranked "hard" distractors (`DistractorConfig`) to force in-context schema discrimination. Result: **net-negative.** Axis-A zero-shot rose only ~0.03 → ~0.05–0.07 (peak 0.10, collapse broken) while axis-B *regressed* (arg 0.78 → 0.52) — the harder task made the model worse at what it was good at. Two real bugs surfaced + fixed: silent target-truncation when prompt+target overflows the 640 ctx (`build_batch`'s `ids.truncate`; now guarded by a startup PROMPT-LEN check + capping to 4 distractors), and **NaN divergence** at ~step 100 (unclipped Adam on the harder/longer task; fixed with gradient-norm clipping at 1.0).

**Step 3 — more capacity.** 13M → **~28.7M** (`p1a_large`: d_model 512, 8 layers/heads), same easy-distractor data as Step 1, 8 epochs. Result: **no help on either axis.** Axis-B arg **0.73** / schema 0.84 ≈ 13M's 0.78/0.90 (flat; train loss cratered to 0.045 = fit training faster, no better generalization). Axis-A zero-shot **0.008** ≈ 13M's ~0.03 (flat at the floor). **2.2× capacity broke neither ceiling.**

### Conclusion — the remaining wall is *architectural*

A clean ablation ruled out the obvious levers: **more data (Step 2) and more capacity (Step 3) both fail** to move zero-shot-to-unseen-tools. In-distribution grounding is also saturated for this architecture (30M ≈ 13M). Therefore the remaining gap is **not** data-, epoch-, or capacity-bound — the model lacks the *mechanism* to copy novel param names from an unseen in-prompt schema into its call. The indicated lever is **architectural**: a copy/pointer mechanism (bias generation toward copying spans from the offered schema), *not* more parameters, more data, or Apertus foundation pretraining. (Apertus buys natural-language comprehension, which was never the bottleneck for these synthetic templates.)

## 9. Open next steps (for a fresh decision)

1. **Copy/pointer mechanism** — the evidence-backed lever for zero-shot: let the model copy param-name/value spans directly from the offered schema. Architectural change to `TinyTransformer`.
2. **Organ-scale + High Grug/KG** — accept that true zero-shot generalization belongs to the larger organ (P3+) where schema knowledge comes from the KG via the gap-token, not from weights.
3. **P1b (Apertus)** — still open, but de-prioritized: it addresses comprehension, which this ablation shows is not the current bottleneck.

## 10. P1a-4 Step A — schema-constrained decoding (2026-07-07)

The P1a-3 ablation concluded the zero-shot wall is **architectural**: the model lacks a mechanism to emit param names it only saw in an unseen in-prompt schema. Step A is the cheap, inference-only probe of that claim (no retraining): constrain greedy decoding so every emitted token keeps the output on a path to a call that is valid for *some offered tool*, then re-eval the existing 13M arg-0.78 checkpoint (`runs/p1a2-ep8/step_00019999`) on the **same** frozen 128×2 holdout.

**Method.** A byte-level DFA acceptor (`grug_data::constrain::ToolCallGrammar`) over the exact training target — which, because `serde_json`'s default `Map` is a `BTreeMap` (no `preserve_order` in this workspace), is `{"arguments":{<sorted params>},"name":"<tool>"}` (**arguments before name, name last**). A golden test pins this against 300 real generator targets. Since the name comes *last*, the acceptor carries a shrinking set of **viable tools** (those consistent with every param key emitted so far) and requires the final name to lie in it — so the constraint is "consistent with a single offered tool", not the weaker "each key belongs to *some* tool". Disallowed tokens' logits are masked to −∞ before the argmax (`generate_constrained`); `EOS` is offered only at a complete call. **Structure + param names + tool name are constrained; values are left free** — the deliberate isolation of "invalid names" from "wrong values". Per-token byte pieces come from `Vocab::token_pieces` (GPT-2 byte↔unicode inverse for the BPE surface; a test proves the pieces concatenate byte-exactly to `decode`).

**Baseline reproduced the training run exactly** (load-fidelity + config check): axis-B arg **0.781** / schema 0.898, axis-A schema **0.031** — identical to the `p1a2-ep8` step-19999 eval row. So the deltas below are real, not artifacts of a mismatched holdout.

| Axis | metric | baseline | constrained | Δ |
|---|---|---|---|---|
| **B** (held-out phrasing, in-distribution) | valid_json | 0.992 | 0.992 | — |
| | schema | 0.898 | **0.961** | +0.063 |
| | arg_correct | 0.781 | **0.797** | +0.016 |
| | distinct_call_ratio | 0.906 | 0.969 | +0.063 |
| **A** (whole-tool holdout, zero-shot) | valid_json | 0.984 | 0.969 | −0.015 |
| | schema | **0.031** | **0.219** | **+0.188 (≈7×)** |
| | arg_correct | **0.031** | **0.125** | **+0.094 (≈4×)** |
| | distinct_call_ratio | 0.039 | 0.227 | +0.188 |

(Artifact: `runs/p1a2-ep8/eval_constrained.json`. Baseline eval 122s, constrained 72s incl. per-step CPU masking, on the 7900 XTX.)

**Reading.**
- **In-distribution is not harmed — slightly helped.** Constraining cleaned up the occasional invalid key (schema 0.898→0.961) with no valid-JSON or arg regression. So schema-constrained decoding is a safe default even where the model already generalizes.
- **Zero-shot schema-adherence lifts ~7× (0.031→0.219)** — a real, cheap floor-raiser. A meaningful fraction of the zero-shot failure *was* invalid param names, exactly as the P1a-3 diagnosis predicted; forcing valid, schema-consistent names recovers it.
- **But it stays low in absolute terms, and arg-correct (0.125) trails schema (0.219).** Two residuals remain, both *values*-shaped, both what Step A cannot fix: (a) even a schema-valid call needs the right *argument values*, which must be **copied** from the request (constraining names can't supply them); (b) the acceptor does not force required-key coverage or the correct tool *selection*, so the model can still name the wrong tool or drop a required key. The gap between constrained-schema and constrained-arg is precisely the copy problem.

**Verdict — Step A confirms the Step-B thesis.** Constrained decoding is worth keeping as a free inference-time win (ship it as the default decode for tool-calling; it strictly helps in-distribution and 7× on zero-shot schema), but the residual zero-shot gap is a **value-copying** problem, not a name-validity one. That is the evidence-backed motivation for **Step B — the copy/pointer mechanism** (let the model copy novel param *names and values* directly from the offered schema/request), now with a quantified floor to beat: constrained zero-shot schema 0.219 / arg 0.125.

**Caveats (honest).** The grammar constrains names/structure only, not required-coverage or values, so constrained-schema 0.219 mixes "valid names" with residual "missing-required / wrong-tool" failures — a tighter required-coverage constraint would separate these further but was out of Step A's cheap-probe scope. n=128 per axis → treat single points as ±. Distractor config = easy (2 uniform), matching the `p1a2-ep8` training distribution.

## 11. P1a-4 Step B — pointer-generator copy mechanism (2026-07-07)

Step A localised the residual zero-shot gap to **value copying** and predicted a copy/pointer mechanism as the lever. Step B built it (See et al. 2017, adapted to the decoder-only trunk): a single-head causal **copy attention** over input positions, scattered onto the vocabulary; a learned `p_gen ∈ [0,1]` gate from `[hidden ; context]`; mixed next-token distribution `P = p_gen·P_vocab + (1−p_gen)·P_copy`, trained by NLL over the mixture (marginal likelihood — generate-vs-copy is learned, not supervised). Same 13M trunk, same data/tokenizer/schedule as `p1a2-ep8` (seed 42, 1000 tools, 8 epochs, easy 2-distractors), batch 16, 96 min on the 7900 XTX. Artifacts: `runs/p1a4-ptr/`. Wiring verified before the run: a unit test confirms the mixed distribution normalises to 1.0, and an autodiff train-step test confirms loss decreases on a fixed batch (scatter backward + copy-path gradients + optimizer all work).

| Decode | Axis | metric | non-pointer (Step A) | **pointer (Step B)** |
|---|---|---|---|---|
| **plain greedy** | B (in-dist) | arg_correct | 0.781 | **0.711** |
| | A (zero-shot) | schema | 0.031 | **0.016** |
| | A (zero-shot) | arg_correct | 0.031 | **0.016** |
| **constrained** | B (in-dist) | schema / arg | 0.961 / 0.797 | **0.930 / 0.758** |
| | A (zero-shot) | schema | 0.219 | **0.250** |
| | A (zero-shot) | arg_correct | 0.125 | **0.094** |

(Pointer axis-A plain schema across 41 evals: mean 0.026, max 0.062 — flat at the baseline floor the whole run.)

### Verdict — Step B did **not** clear the gate

The gate was zero-shot schema ≥ 0.3–0.5 without regressing in-distribution arg below ~0.75. Neither held:
- **Zero-shot is unchanged.** Plain axis-A schema 0.016 ≈ the non-pointer baseline 0.031 (flat the entire run). The copy mechanism produced *no* zero-shot lift.
- **In-distribution mildly regressed.** Plain axis-B arg 0.711 vs 0.781 (below the 0.75 floor); constrained axis-B arg 0.758 vs 0.797.
- **The copy path is vestigial even under constraint.** Pointer+constrained axis-A schema 0.250 ≈ Step A's grammar-only 0.219 (within n=128 noise), and arg is *worse* (0.094 vs 0.125). So the constrained zero-shot signal comes entirely from the **grammar mask**, not the copy pathway — adding copy contributes nothing and slightly hurts.

### Why — the mechanism is necessary but not sufficient; the *training signal* is the real lever

Leading hypothesis (behaviourally strongly supported, not yet instrumented): **the training objective never incentivises copying.** On *trained* tools, generating each value from memory works — every required value is learnable, so there is no gradient pressure to route probability through the copy path. `p_gen` drifts toward "generate" and the copy attention stays undertrained; at zero-shot time that untrained copy path gives no benefit. The mild in-distribution regression is consistent with a partially-on copy path injecting noise into an otherwise-competent LM. This refines the P1a-3/Step-A picture: the wall is not only that a vanilla LM *lacks* a copy mechanism, but that **marginal-likelihood training on a memorisable distribution won't *learn* to use one** even when it's present.

The indicated next levers (in order of directness), for a fresh decision — **not started**:
1. **Force copying during training** — a coverage/copy-supervision loss (mark target tokens that appear verbatim in the prompt, push `p_gen` toward copy + copy-attention onto the matching source position), and/or **anti-memorisation data** (per-example novel/randomised values so generating from memory fails and copying is the only route). This directly trains the pathway Step B left dormant. Cheap confirming step first: **instrument `p_gen` / copy-attention entropy** to verify the gate-collapse hypothesis before investing in the loss.
2. **Accept zero-shot belongs to the organ (Step C, P3+)** — schema knowledge arrives from the KG via the High-Grug gap-token, not from weights. The two negative results (more capacity/data in P1a-3; the learned copy mechanism here) together are a strong argument that a 13M weights-only model on synthetic templates is the wrong vehicle for open-world zero-shot.

### What ships regardless

**Schema-constrained decoding (Step A) is the concrete win from P1a-4** — a free inference-time 7× on zero-shot schema (0.031→0.219) with no in-distribution cost, on the *non-pointer* 13M checkpoint. Step B is a documented negative that redirects the effort from "more architecture" to "the training signal / the organ". n=128 → single points ±; one seed.

### Step B2 — copy-supervision (2026-07-07): gate-collapse confirmed, still no zero-shot

`p_gen` instrumentation on the Step B checkpoint showed **mean `p_gen` ≈ 0.96** (~95% of target positions >0.9) — **gate collapse confirmed**: the copy pathway trains but is never used. An auxiliary BCE supervising `p_gen` toward copy (`train_step_ptr_supervised`, weight λ) *does* move the gate (`p_gen` 0.96→~0.77 zero-shot by step 1000) but fails to clear the gate for two reasons: (1) **grad-norm-clip caps λ** — λ=1 and λ=4 gave *identical* `p_gen`=0.767 at step 1000 despite different losses/checkpoints; (2) even with the gate opening, in-distribution copying recovers only slowly (axis-B ~0.72/0.52 at step 12k) while **zero-shot plateaus at schema ~0.06** (axis-A, flat across 12k steps). The copy *attention* does not learn accurate in-context pointing at this scale.

**The deeper lesson → format pivot (P2).** Three model-machinery levers now fail on zero-shot — capacity (P1a-3), the copy mechanism (Step B), copy-supervision (Step B2) — and constrained decoding only yields valid-but-wrong. The common thread is that a tiny model emitting **JSON** spends scarce capacity on the format, not the task (the small-model "format tax"). **[P2](plans/2026-07-07-phase51-p2-format-first-corpus.md) (ADR 088)** abandons JSON for a minimal High-Grug format and tests it on an otherwise-identical pipeline — the format, not more model machinery, is the next hypothesis.

## 12. P2 (format) + P3b (KG-repair) — the format is a wash on accuracy; the *organ* is the zero-shot lever (2026-07-07)

**P2 format A/B** (`runs/p2-grug/`, base 13M, seed 42, only the boundary format varies): High-Grug v0 **halves** encoded sequence length (median 109 vs JSON's ~345) and makes valid-format trivial, but **in-distribution accuracy is a wash** (Grug final arg 0.65 vs JSON 0.78, marginally below) and **zero-shot is unchanged** (axis-A schema mean **0.030** vs JSON 0.031, flat the whole run). Format was the fourth lever to leave zero-shot at ~0.03. It is locked in ([ADR 088](decisions/088-high-grug-boundary-format.md)) for efficiency/reliability/High-Grug alignment, **not** as a zero-shot fix.

**P3b KG-backed repair** ([ADR 089](decisions/089-organ-kg-zeroshot-path.md) mechanism 2) is the first thing to move zero-shot — and it moves it a lot. `grug_data::repair::repair_call` grounds the neural proposal onto the offered schemas (stand-in KG): pick the tool that best explains the call (name similarity + emitted-param overlap), snap each arg name to the nearest schema param (Levenshtein floor; `region`→`region_name`), coerce values to declared types. No new training — a symbolic post-process on the *same* checkpoints.

| model | axis | metric | raw (neural) | **repaired (KG-grounded)** | Δ |
|---|---|---|---|---|---|
| **Grug** 13M | B (in-dist) | schema / arg | 0.812 / 0.641 | 0.852 / 0.656 | +0.04 / +0.02 |
| | A (zero-shot) | schema / arg | 0.023 / 0.023 | **0.664 / 0.555** | **+0.64 / +0.53** |
| **JSON** 13M | B (in-dist) | schema / arg | 0.898 / 0.781 | 0.914 / 0.781 | +0.02 / +0.00 |
| | A (zero-shot) | schema / arg | 0.031 / 0.031 | **0.727 / 0.602** | **+0.70 / +0.57** |

**Reading — this validates ADR 089 empirically.** Zero-shot arg-correct jumps **~0.03 → ~0.55–0.60 (≈20×)** on *both* independently-trained models, format-independent. In-distribution barely moves (the model is already right there). The interpretation is decisive: **the neural model, even zero-shot, already produces the right argument *values* — it just names the wrong tool / slightly-off params**, which the symbolic layer fixes trivially against the schema. Zero-shot correctness was never a weights problem (four negatives proved that); it was a *grounding* problem, and grounding is System-2's job. The "four levers fail" story and this "one symbolic pass wins" story are the same story told from both sides.

**Caveats (honest).** n=128, one seed, easy 2-distractor offered sets (3 tools) — tool-grounding is easier with a small offered set; a larger/harder set stresses it, which is exactly why the organ's *retrieval* step (ADR 089 mechanism 1) keeps the offered set small and relevant. The "KG" here is the offered procedural schemas; similarity is Levenshtein (the real KG will use SymbolId/VSA similarity). Repair does not invent values for params the model never emitted (that is the retrieval/gap-token layer). But the core claim stands and is robust across two models: **a tiny neural proposer + a symbolic KG-grounding pass does zero-shot tool-calling that neither does alone.**

## 13. P3c — gap-token abstention: learnable, precise, but low-recall at 13M (2026-07-08)

The third organ mechanism (ADR 089 §3): teach the model to emit a **gap token** — the bare word `gap` in High-Grug — when *no offered tool fits* the request, rather than forcing a (necessarily wrong) call. ADR 089's claim is that this is categorically unlike open-world zero-shot: abstention ("none of these match") is an **in-distribution classification the weights *can* learn**, whereas "produce a correct call for an unseen tool" is not.

**Setup** (`runs/p3c-gap/`, base 13M, seed 42, 1000 tools, High-Grug, easy-2 distractors, 20000 steps — the p2 recipe with one change): **20% of training examples are abstention** examples (request phrased for tool `T`, offered bundle excludes `T`, target = `gap`). Frozen holdout confusion matrix (n=128 should-call held-out-phrasing + n=128 should-abstain): **recall** (of should-abstain, fraction that emitted gap), **false-abstention** (of should-call, fraction that wrongly emitted gap), **precision**.

| metric | mature plateau (10k–20k, 21 evals) | final (20k) | max over run |
|---|---|---|---|
| abstain **recall** | **0.241** | 0.273 | 0.320 |
| abstain **precision** | **0.923** | 0.833 | 1.000 |
| **false-abstention** | **0.023** | 0.055 | — |
| call arg-correct (axis B) | 0.541 | 0.570 | 0.633 |

**Reading — qualitatively confirms ADR 089, quantitatively modest.** The decisive contrast is with §10–§12: *four* levers (capacity, copy, copy-supervision, format) left open-world zero-shot pinned at schema ≈ 0.03 — **nothing moved it**. Abstention **moves**: recall climbs off zero to a real ~0.24 plateau at **high precision (~0.92)** and **near-zero false-abstention (~0.02)**, while call accuracy tracks the p2 baseline (minus a modest ~0.05–0.08 mix tax from spending 20% of the corpus on abstention). So the weights genuinely *learn* "none of these fit" — the ADR 089 partition (zero-shot → organ; abstention → weights) holds. But a 13M model is a **conservative, precise, low-recall abstainer**: it catches only ~1 in 4 no-fit cases. When it does abstain it is almost always right, and it does not cannibalise real calls — a *precise under-firing safety valve*, not a reliable "always know when to abstain" classifier.

**Why recall is low (hypotheses, untested).** (1) The 80/20 call/abstain prior teaches "there is usually a matching tool"; a higher `--abstain-frac` should raise recall (at some call cost). (2) Easy-2 distractors make the "none fits" boundary soft — hard (similar) distractors would sharpen the discrimination, likely at some false-abstention cost. (3) 13M capacity.

**Where it fits.** A precise, low-recall gap-token is *usable as the mechanism-3 fallback* precisely because it sits behind mechanism 1 (retrieval keeps the offered set relevant → genuine "nothing fits" is rare) and mechanism 2 (KG-repair fixes near-misses): it fires on the clearest no-fit cases with few false alarms; retrieval/synthesis handles the rest. Raising it to a *primary* abstention signal needs the abstain-frac / hard-distractor sweep or more capacity. Artifacts: `runs/p3c-gap/{eval_gap.jsonl,checkpoints/step_00019999}`; harness `crates/grug-train/examples/train_gap.rs`. **(§14 shows M1's retrieval scoping is exactly that sharper signal — it lifts this same abstention recall from 0.24 to ~0.998.)**

## 14. P3a — retrieval-scoped tool universe: the premise holds, and it rescues abstention (2026-07-08)

The first organ mechanism (ADR 089 §1): there is no genuine open-world zero-shot in normal operation. The tool **registry** is closed and the model trains on all of it; at runtime a **retriever** surfaces the top-K request-relevant tools and only those are offered. So the whole-tool-holdout scenario (the §10–§13 neural floor at ~0.03) is one the organ never faces — zero-shot is *dissolved by scoping*.

**Setup** (`runs/p3a-registry/`, 13M, seed 42, High-Grug, 20000 steps): 850-tool **registry** (trained) + 150 **out-of-registry** tools (genuine-novelty source). The offered set per request is a lexical retriever's top-8 over the registry — a faithful stand-in for the Phase-28 VSA/semantic router, since the procedural requests embed each tool's `verb_noun` (`verb_phrase_for`: `fetch_ledger` → "…fetch the ledger…"), so name-token overlap *is* the relevance signal, and many tools share one token → the top-8 is a set of request-keyed *confusables* (not random distractors). The **label is a function of the offered set**: call when retrieval surfaced the target, gap when it did not (a retrieval miss, or an out-of-registry request) — coupling M1 to M3. 20% of the corpus is genuine out-of-registry must-abstain examples.

| metric | mature plateau (14k–20k, 13 evals) | final (20k) |
|---|---|---|
| **retrieval recall@8** (model-independent) | **1.000** | 1.000 |
| registry call schema / arg | 0.909 / 0.611 | 0.891 / 0.633 |
| **abstain recall** | **0.998** | 1.000 |
| abstain false-abstention | 0.001 | 0.008 |
| abstain precision | 0.999 | 0.992 |

**Finding 1 — the premise holds decisively (recall@8 = 1.000).** With a retriever, the offered set contains the target essentially always, so the model never faces the whole-tool-holdout scenario in normal operation. Open-world zero-shot is dissolved by *scoping*, not demanded from the weights — exactly ADR 089's claim. The whole-tool-holdout axis (§10–§13) is retired as a target and kept only as the neural-floor diagnostic it always was.

**Finding 2 — M1 rescues M3.** Abstention recall jumps from §13's **~0.24** (random-distractor offered sets) to **~0.998** here, at near-zero false-abstention. The reason is structural, not more capacity: under retrieval scoping a *callable* request's offered top-8 always contains the exact verb+noun match, while a *genuine-novelty* request's offered set contains only one-token near-misses — so "is the right tool actually present?" becomes a **crisp, learnable pattern** instead of §13's no-signal "3 random tools". The mechanisms compose synergistically: retrieval makes both the call and the abstain decision sharp. And registry call reliability (schema 0.91 / arg 0.61) matches the easy-2 baseline (p2 0.90 / 0.65) *even though* the retrieved offered set — 8 request-keyed confusables — is a harder discrimination than 3 random tools.

**Caveat (honest).** The lexical retriever and the call/gap label share the same name-token signal, so "exact match present vs only near-misses" is an especially clean cue in this harness; the production router is semantic (VSA over `goal ↔ tool`) and the schema read is semantic, so abstention in the real system is a semantic judgment, not token-matching — the *structure* of the result should transfer, but likely not as pristine as 0.998. Artifacts: `runs/p3a-registry/{eval_registry.jsonl,eval_registry_repair.json,checkpoints/step_00019999}`; harness `crates/grug-train/examples/{train_registry,eval_registry_repair}.rs`, retriever `crates/grug-data/src/retriever.rs`.

**Finding 3 — the full stack, and where the residual actually lives.** Running M2 KG-repair on the final checkpoint (n=256, `eval_registry_repair.json`), grounding each proposal against the *retrieved* offered set: registry call arg **0.594 raw → 0.594 repaired (Δ +0.000; schema 0.914 → 0.938)**. M2's ≈20× lever from §12 does not transfer to the in-distribution retrieval regime. An **error-structure inspection** (`examples/inspect_registry.rs`, n=400) says why — and *refutes an earlier guess that the gap was tool-selection error*:

> Of 160 arg-wrong (parsed-as-call) cases, **0 chose the wrong tool.** Every one is right-tool-wrong-args. Breakdown: **value-only 68%** (right tool + right param names, a *value* differs — generation fidelity, partly synthetic float-format noise like `1.66` vs `1.6600000000000001`), **presence-only 22%** (valid param names, a different optional-inclusion subset than the RNG chose — partly ill-posed), **invalid-name 10%** (a non-schema param name — the only M2-repairable slice).

So the §12-vs-§14 contrast is **naming vs value**: §12's zero-shot error was right-values-wrong-names → grounding recovers ~0.53; §14's in-distribution residual is value fidelity + optional presence, which grounding cannot touch (tool and param names are already right). M2 had only ~10% of the residual to work on, and fixing a name rarely flips a value-wrong call to exact-correct → Δ +0.000 arg. **Tool selection is *already solved* (M1 retrieval + the model: 0/160 wrong-tool), so a re-ranker is a no-op; the remaining ceiling is the model's own argument-*value* fidelity (capacity / value-grounding), not selection or naming.**

**Finding 4 — the metric is fair, and the true value-fidelity ceiling is higher than the whole-call number says.** A scorer-based inspection (`inspect_registry.rs` via `score_output_fmt` + `ArgMismatch`, n=400) tested whether the residual is a measurement artifact: **float-formatting = 0 scored errors** (the scorer's 1e-6 numeric tolerance already equates `1.66` and `1.6600000000000001`), **empty-value artifacts = 0**. So the proposed "canonicalize floats / relax optional-presence" cleanup was *unwarranted* — the metric is fair, and the residual (130/400: value-only 106, presence-only 14, mixed 10) is genuine. The real improvement is a *finer lens*, not a fix: **`EvalReport.arg_value_accuracy`** (micro-averaged per argument-value slot) alongside the strict whole-call `arg_correct_rate`. On `step_00019999`: **per-value 0.748 vs whole-call 0.594** — the model gets ~3 of every 4 argument values right; whole-call exact-match undersells it because one wrong value fails an entire multi-param call. (Repaired per-value is identical, 0.748 — M2 fixes names, not values.) So the honest last-mile number is *value fidelity ≈ 0.75*, a capacity / value-grounding problem.

**Finding 5 — value-grounding closes ~half the value residual.** The value errors (§Finding 4) have a decisive property: an inspection (`inspect_registry.rs`, n=400) found the **correct value is present verbatim in the request 100% of the time**, and ~70% of errors are numeric (mis-copied multi-digit numbers `637.82`→`637.39`, or misaligned similar numbers `350` grabbed from `350.39`). So values can be re-grounded against the request's own value inventory. `grug_data::ground::ground_values` (composed *after* M2 repair): snaps each emitted numeric arg not already in the request to its nearest *unused* request number (a nearest-value assignment fixes both corruption and misalignment without two params colliding on one number), and snaps a wrong enum to the single declared variant present in the request; strings/booleans and already-correct values are left untouched, so it never regresses a correct arg. **Measured (`step_00019999`, n=256): whole-call arg 0.594 → 0.738 (+0.145, ~24% relative), per-value 0.748 → 0.824 (+0.076), schema 0.914 → 0.945.** The grounded registry model now *beats* the easy-2 baseline (whole-call 0.65). The whole-call lift exceeds the per-value lift because fixing the one wrong value in an otherwise-correct call flips the whole call. The residual grounding does *not* recover: misalignment where the wrong value is itself a request value (left untouched by design), string/boolean errors (no unambiguous re-grounding signal), and rare nearest-assignment mis-picks — the rest is genuine model value-fidelity.

**Arc close (§10–§15).** Four levers could not make a 13M weights-only model do open-world zero-shot (§10–§13). The organ does it — now measured end-to-end as a *four-stage* last-mile stack, each stage with a distinct, non-substitutable job: **M1 retrieve** so the target is present (recall@8 = 1.0), **propose** (the neural model — near-perfect tool selection, 0 wrong-tool), **M2 repair** the near-miss *name* against the schema (≈20× on zero-shot naming; noop in-distribution where naming is already right), **value-ground** the mis-copied *values* against the request (+0.145 whole-call), and **M3 abstain** when nothing fits (≈0.998). Zero-shot was never the weights' job; and the residual after all four is the model's own value-copy fidelity on the errors symbolic grounding can't disambiguate — a genuine capacity signal, no longer inflated by presence, naming, selection, or metric artifacts.

## 15. P3a follow-up levers — string grounding (null), HDC retriever (caveat probed), organ gap-response (built) (2026-07-09)

Four follow-up levers, each built and *measured*, two of which land honestly as no-ops on this corpus (the same "measure, don't assume" discipline as the refuted re-ranker in §Finding 3):

**15a — String value-grounding: correct, but null on this corpus.** `ground_values` was extended to re-ground `String` args (the value is a verbatim request span, like numbers): an ungrounded string snaps to the request word-window most similar by normalized Levenshtein, threshold-gated so a confidently-wrong emission with no close span is left alone. Booleans are **declined** (a single bit, no unambiguous bag-of-words polarity — snapping would coin-flip a correct value). Re-running the full-stack eval on `step_00019999`: grounded numbers are **byte-identical** (whole-call 0.738, per-value 0.824) — string grounding fired ~0 times because this corpus's string values are a tiny closed bank of short place names/phrases the model copies correctly (the residual is numeric/enum, per §Finding 3). The lever is unit-tested and works when strings *are* corrupted; it is standing capability for real tools with free-form string args, and a measured no-op here.

**15b — Semantic (HDC) retriever probes the lexical caveat.** A binary-hypervector char-trigram retriever (`grug_data::retriever::SemanticRetriever`, Hamming — a *surface-semantic* step toward the real VSA router, not a learned embedding) vs the exact-token lexical retriever, over the 850-tool registry (`examples/gap_organ.rs`, n=500):

| requests | lexical (exact-token) recall@8 | HDC (char-trigram) recall@8 |
|---|---|---|
| clean | **1.000** | **0.980** |
| + name typos | **0.000** | **0.588** |

On clean requests both are high — the caveat made concrete: the *only* tool-side signal on this corpus is the name (descriptions merely echo it), so any surface retriever keys on the verbatim embedded name. They diverge only under **name typos**: exact-token overlap zeroes out (recall 0.000) while the HDC retriever holds 0.588 — **robustness** is what the HDC/VSA layer buys over exact matching. Neither is a *learned* semantic embedding, so **neither survives genuine paraphrase** (synonyms with no shared surface); that is untestable on this name-only corpus and stays the learned Phase-28 router's job. So the §14 abstention transfers structurally and gains typo-robustness, but the pristine 0.998 still owes something to surface coupling — the honest caveat stands.

**15c — The organ's gap-response, built + measured.** The runtime half of M3 (`grug_data::organ`): `GapResponse{Widen,Synthesize}` + `respond_to_gap` + `GapRecoveryReport`. On a fired gap the organ widens retrieval along `[16,64,…,|registry|]`; the first width surfacing a **covering** tool (`Retriever::covers` = *all* name tokens present, not one-token overlap) → `Widen` and re-offer; else → `Synthesize`. Measured (`examples/gap_organ.rs`): (1) **retrieval is not the bottleneck** — the two-token match is unique enough that recall saturates even at k=3, so **0 natural retrieval-miss gaps** exist to rescue (the widen mechanism is proven by a synthetic test forcing a covering tool outside the offer); (2) **genuine novelty terminates correctly** — 500 out-of-registry requests → **500 Synthesize (1.000), 0 false-widen**: the all-tokens coverage test is never fooled by a partial confusable. On this corpus M3's runtime value is entirely synthesis-gating; the widen branch is standing capacity for the harder retrieval regime a semantic router faces. Synthesis authoring itself stays a KG/agent-layer seam.

**15d — Confirmatory run (seed 43): n=2, and grounding is a variance absorber.** The §14 numbers were n=1; a fresh-seed run (`runs/p3a-registry-s43/`, seed 43, otherwise identical — 20000 steps, 65 min, final loss 0.051) gives the second sample. Both evals use their own seed's registry split (n=256 on `step_00019999`).

| metric | seed 42 | seed 43 |
|---|---|---|
| retrieval recall@8 | **1.000** | **1.000** |
| raw neural — whole-call / per-value | 0.594 / 0.748 | **0.465 / 0.680** |
| M2 repaired — whole-call | 0.594 | 0.477 |
| **grounded** — whole-call / per-value | **0.738 / 0.824** | **0.727 / 0.811** |
| grounded schema | 0.945 | 0.957 |
| abstain recall / false-abstain / precision | ~0.998 / 0.001 / 0.999 | 0.977 / 0.012 / 0.988 |

**The finding:** the premise (recall@8) is seed-invariant at 1.000, and abstention is stable (~0.98–1.0). But the decisive observation is the last-mile stack's behaviour under neural variance: the **raw** neural whole-call number swings **0.13** across seeds (0.594 → 0.465 — seed 43 drew a weaker model), yet the **grounded** number converges to within **0.011** (0.738 vs 0.727). The grounded Δ is correspondingly larger on the weaker seed (+0.262 vs +0.145). So value-grounding does not merely raise the mean — it **absorbs the model's training-seed variance**, pulling a noisier proposer to the same ~0.73 grounded outcome. This is exactly the ADR 089 thesis sharpened: the correctness that matters lives in the symbolic organ, and the organ makes the shipped number robust to which particular weights the neural System-1 happened to land on. Mean grounded whole-call over n=2 ≈ **0.733 ± 0.006**; raw ≈ 0.53 ± 0.065 — an order of magnitude more spread.

## 16. P3d-1 — the recall matrix: recall@8 = 1.000 was *entirely* surface coupling to the name (2026-07-09)

Every §14–15 headline carried the asterisk that the corpus embeds the tool name verbatim in the request and lets the description echo the name. P3d ([plan](plans/2026-07-09-phase51-p3d-semantic-retrieval-corpus.md)) builds the corpus that removes it: a **gloss lexicon** (`grug_data::semantic`, `VERB_GLOSS`/`NOUN_GLOSS`, each verb/noun given primary + alt synonyms with *zero whole-word overlap with the name*, tested) drives three `SignalMode` tiers — **D0** name (current), **D1** description (request + description use the primary gloss, name absent), **D2** paraphrase (request uses the *alt* gloss, description the *primary* — surface-disjoint). Argument values are never glossed (they stay verbatim), so value-grounding is a control.

P3d-1 is the model-independent recall matrix (`examples/recall_matrix.rs`, seed 42, 1000 tools, k=8, n=500): recall@8 for a lexical and an HDC retriever, each keyed on the name or the description.

| tier | lex@name | lex@desc | hdc@name | hdc@desc |
|---|---|---|---|---|
| **D0** name | **1.000** | 1.000 | 0.990 | 0.924 |
| **D1** description | **0.002** | **0.990** | 0.010 | **1.000** |
| **D2** paraphrase | 0.002 | **0.012** | 0.008 | **0.018** |

*(one tool, transformed: D0 request "resume the endpoint" → D1 "continue the api address" (desc "Continue the api address.") → D2 "carry on the service url" vs the same primary-gloss description.)*

**Two caveats, now numbers.** (1) **How much of recall@8 = 1.000 was surface coupling? All of it.** The instant the name leaves the request (D1), lexical-on-**name** collapses **1.000 → 0.002** — the P3a premise was the verbatim name, nothing more. (2) **Does the signal survive when moved to the description?** At **D1, completely** — a description-keyed retriever recovers it (lex 0.990, HDC **1.000**): retrieval is fine keyed on *meaning-bearing text the retriever actually reads*, as the real system does, **provided request and description share vocabulary**. At **D2 (genuine paraphrase), no surface retriever can** — lexical- and HDC-on-description floor at **0.012 / 0.018**, because request (alt gloss) and description (primary gloss) share no content surface. That ~0.01–0.02 is the **surface-retrieval floor under true paraphrase**, and it is exactly where a *learned* embedding must earn its keep (P3d-2, Seshat MiniLM). D0 hdc@desc 0.924 < 1.0 is HDC noise on the echo-description; immaterial.

**Consequence for the arc.** The P3a premise (recall = 1.0) is *not* a semantic result — it is a surface artifact that vanishes under D1. The good news is D1 shows the premise is *recoverable* by reading the description, so the organ's retrieval story is sound wherever the request and the capability description share vocabulary (the common case). The open question is D2 — genuine paraphrase — which surface methods cannot touch and which P3d-2/P3d-3 test with a learned embedder and an abstention retrain. This is the honest sharpening the whole P3d phase was for: the caveat is now a measured floor, not a hand-wave.
