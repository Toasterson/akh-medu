# Phase 51 P1a-4 — Next Steps: Zero-Shot via a Copy/Pointer Mechanism

- **Status:** Steps A + B **DONE** (2026-07-07). **Step A cleared its probe** (constrained decoding: zero-shot schema 0.031→0.219, [§10](../2026-07-05-phase51-p1-scaling-results.md)); **Step B did NOT clear the gate** — the learned copy/pointer mechanism gave no zero-shot lift and mildly regressed in-distribution ([§11](../2026-07-05-phase51-p1-scaling-results.md)). Diagnosis: marginal-likelihood training on a memorisable distribution doesn't *learn* to copy (mechanism present but untrained). Next lever = training **signal** (copy-supervision loss / anti-memorisation data) or the organ (Step C), NOT more architecture. Self-contained continuation doc.
- **Date:** 2026-07-05 (Steps A+B results 2026-07-07)
- **Depends on / continues:** [P0→P1a-3 scaling results](../2026-07-05-phase51-p1-scaling-results.md) (§8 ablation), [Phase 51 plan](2026-07-05-phase51-grug-brain-baby-llm.md), [ADR 087](../decisions/087-grug-brain-baby-llm-training.md).

---

## 0. Where we are (state recap — read this first after a compact)

Phase 51 P0→P1a-3 is **complete and on develop** (unsigned commits, merged). A from-scratch **13M transformer** (`crates/grug-train`, Burn/ROCm on the AMD 7900 XTX gfx1100) trained on a **1000-tool procedural catalog** (`crates/grug-data`) does **in-distribution JSON tool-calling well** — held-out-phrasing arg-correct **0.78**, schema 0.90.

**The one remaining wall — proven by ablation:** zero-shot to *unseen* tools (whole-tool holdout) is stuck at schema ~0.01–0.03. We ruled out the obvious levers:
- More data / harder distractors (P1a-3 Step 2) → **net-negative**.
- 2.2× capacity, 13M→30M (P1a-3 Step 3) → **no help on either axis**.

**Conclusion: the wall is architectural.** The model has no mechanism to *copy a novel param name from an unseen in-prompt schema* into its call. A vanilla LM can only emit vocab it learned; unseen tools' param names live only in the prompt.

**Key files / how to run (unchanged):**
- Model + loop + eval: `crates/grug-train/{src/transformer.rs (TinyTransformer, Attention<B> seam, p1a / p1a_large configs), src/train.rs (SFT loop + Adam grad-clip Norm(1.0) + build_batch), src/generate.rs (greedy decode), src/eval.rs, examples/train_p1a.rs}`.
- Data + eval scorer: `crates/grug-data/{src/generator.rs (DistractorConfig, offer_bundle), procedural.rs, dataset.rs (new_sized / new_sized_with_distractors), eval.rs}`.
- **GPU run recipe:** `source scripts/train-env.sh` (puts the `hipconfig` shim on PATH — REQUIRED at build *and* runtime), then `cargo run -p grug-train --example train_p1a --release --features gpu -- <flags>`. Detach long runs with `setsid`. Watch with a NaN-guarded poll loop (a progress-only watcher once stayed silent through a 67-min NaN divergence).
- Eval time-series → `runs/<name>/eval_axes.jsonl` (rows tagged `axis`: `held_out_phrasing` = in-distribution, `whole_tool_holdout` = zero-shot). Checkpoints via Burn `Recorder`, resumable with `--resume`.
- ~80 ms/step (13M) to ~160 ms/step (30M / long prompts). 20k-step (8-epoch, 40k-example) run ≈ 60–100 min incl. eval overhead.

## 1. The plan — cheapest-first, two experiments

### Step A — Schema-constrained decoding (cheap, inference-only, do first) — ✅ DONE 2026-07-07
> **Outcome:** built as a byte-level DFA (`grug_data::constrain::ToolCallGrammar`) + `generate_constrained` + `Vocab::token_pieces` + `examples/eval_constrained.rs`. Baseline reproduced the training run exactly; constrained decoding lifted **zero-shot schema 0.031→0.219 (~7×)** and **arg 0.031→0.125 (~4×)** with **no in-distribution regression** (axis-B schema 0.898→0.961). Confirms the residual zero-shot gap is a **value-copying** problem → Step B is the indicated lever, with a floor to beat (constrained zero-shot schema 0.219 / arg 0.125). Full result: [scaling results §10](../2026-07-05-phase51-p1-scaling-results.md).

Constrain greedy decode so that at param-name positions the model may only emit tokens present in the *offered tool's schema* (grammar-/schema-constrained decoding). No retraining — apply to an existing 13M checkpoint (`runs/p1a2-ep8/checkpoints/step_00019999`, arg-0.78 model).
- **What it tests:** does forcing *valid* param names lift zero-shot schema-adherence? It cannot fix arg *values* (those must be copied from the request), so expect schema↑ but arg-correct only partial.
- **Where:** `crates/grug-train/src/generate.rs` — mask the logits at param-key positions to the schema's param-name token sequences.
- **Effort:** ~half a day (1 agent to implement the constrained decoder + a re-eval pass on the existing checkpoint; no training run).
- **Value:** a cheap floor-raiser and a clean diagnostic — isolates "invalid param names" from "wrong values."

### Step B — Copy/pointer mechanism (the real, evidence-backed lever) — ❌ DONE 2026-07-07, GATE NOT MET
> **Outcome:** built (`crate::pointer::PointerTransformer` — causal copy attention scattered onto vocab + `p_gen` gate + mixed log-prob distribution; NLL loss; `generate_ptr`; `train_p1a_ptr`; committed `cc00b55`). Trained same config as `p1a2-ep8`. **Result: no zero-shot lift** (plain axis-A schema 0.016 ≈ baseline 0.031, flat all run) and **mild in-distribution regression** (plain axis-B arg 0.711 vs 0.781). Pointer+constrained ≈ grammar-only (axis-A schema 0.250 vs 0.219), i.e. the copy path is vestigial. **Diagnosis:** on trained tools, generating memorised values suffices, so nothing pressures `p_gen`/copy — the pathway stays undertrained. The mechanism is necessary but *not sufficient*; the lever is the training **signal**. Full result: [§11](../2026-07-05-phase51-p1-scaling-results.md). Next: (1) instrument `p_gen` to confirm gate-collapse, then a copy-supervision / anti-memorisation-data loop; or (2) Step C (organ/KG).

Add a **pointer-generator** (See et al. 2017, "Get To The Point") to `TinyTransformer`: at each decode step, mix a vocab distribution with a **copy distribution over input positions** (attention over the prompt), gated by a learned `p_gen ∈ [0,1]`. Final `P = p_gen·P_vocab + (1-p_gen)·P_copy`. This lets the model copy novel param names *and* arg values directly from the offered schema / request.
- **Where:** `transformer.rs` (add a copy-attention head over the input + a `p_gen` gate from the decoder state); `train.rs` (marginal-likelihood loss over generate-or-copy); `generate.rs` (mix at decode). The `Attention<B>` seam already exists.
- **Effort:** ~1.5–2 days — architectural change + loss change in Burn (young framework, hand-written), then 2–3 training runs (~1 hr GPU each) + debugging/tuning. Reuse the same data (easy 2-distractor, `--n-tools 1000 --epochs 8`) so it's directly comparable to the 13M baseline (arg 0.78 / zero-shot 0.03).
- **Success gate:** zero-shot (whole-tool holdout) schema **moves decisively off ~0.03** (target ≥0.3–0.5) *without* regressing in-distribution arg-correct below ~0.75.
- **Risk:** pointer-generator in Burn is non-trivial; budget debugging time. If it works, this is the headline result.

### Step C — Organ integration (deferred, P3+)
If A/B confirm the copy lever, the honest long-game is that *true* open-world zero-shot belongs to the **organ** (P3+): schema knowledge arrives from the KG via the High-Grug **gap-token**, not from weights. Blocked on High Grug language design (its own plan). Not in scope for P1a-4.

**De-prioritized:** P1b Apertus foundation pretrain — the ablation shows comprehension is *not* the bottleneck for synthetic tool-calling, so it's poor value against the copy mechanism.

## 2. Time estimate

| Step | Work | Estimate |
|---|---|---|
| A — schema-constrained decoding | 1 agent implement + re-eval an existing checkpoint (no training) | **~0.5 day** |
| B — copy/pointer mechanism | model+loss+decode change (1–2 agents, iterative) + 2–3 GPU runs (~1 hr each) + debug/tune | **~1.5–2 days** |
| **A + B total (the actionable next chunk)** | | **~2–2.5 days** of active work |
| C — organ integration | blocked on High Grug design | weeks; deferred |

"Days" = active coordinator + fleet-agent time, not wall-clock idle. GPU runs are local (no API cost); the cost is agent/coordination tokens + ~1 hr GPU per training experiment. Wall-clock for A+B is likely **3–5 calendar days** allowing for run turnaround and iteration.

## 3. Decision points (for whoever resumes)
1. Do Step A first (cheap diagnostic) or jump straight to B? Rec: A first — it's half a day and de-risks/scopes B.
2. If B's pointer-generator proves hard in Burn, fall back to schema-constrained decoding (A) as the shipped zero-shot story + defer learned copying to the organ.
3. Everything is on `develop`; runs write to `runs/` (gitignored). Verify `git rev-parse --abbrev-ref HEAD` before committing (branches churned under concurrent Phase 50 fleets during P1a-3).
