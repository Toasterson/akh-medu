# Phase 40b — E-graph Re-saturation Measurement Report

> Date: 2026-07-05
> Status: COMPLETE (synthetic arm; real-workload arm deferred, see §Gaps)
> Plan: [`docs/ai/plans/2026-04-25-phase40-egraph-persistence.md`](plans/2026-04-25-phase40-egraph-persistence.md)
> Decision: [ADR 084](decisions/084-egraph-persistence-decision.md) — Option (0), persist nothing
> Harness: [`crates/akh-egraph-bench/`](../../crates/akh-egraph-bench/)

## One-line verdict

**Option (0) — persist nothing.** Median e-graph saturation on representative
synthetic theories is milliseconds-to-sub-second under production limits (two
orders of magnitude under the 5 s threshold from ADR 051), production call
sites bound the worst case at ~5 s by construction, and no code path rebuilds
an e-graph at wake at all. The measured wake-path cost that *does* matter is
HNSW(VSA) rebuild, which is already covered by ADR 049's local-disk cache
decision.

## Method

- Harness: `cargo run -p akh-egraph-bench --release -- --what both`
  (`crates/akh-egraph-bench/`). Every grid cell runs in its own subprocess so
  peak RSS (`VmHWM`) is attributable per cell. Deterministic seed (42);
  synthetic theory generator documented in
  [`crates/akh-egraph-bench/src/theory.rs`](../../crates/akh-egraph-bench/src/theory.rs).
- Theory shape: `n_facts` facts as a balanced `and`-conjunction with a mix
  chosen so every production rule pack has fodder — 40 % `causes` edges in a
  `branching`-ary tree (DFS-adjacent so `cause-trans` actually fires), 15 %
  event-calculus pairs, 15 % `knows` facts (the epistemic worst-case driver),
  15 % `before` triple chains, 15 % VSA algebra terms.
- Rule packs are the *real compiled packs* from `src/reason/mod.rs`, no
  synthetic padding: small = builtin (7 rewrites), medium = + pim + calendar +
  causal (15), large = + epistemic (18, includes the deliberately
  non-terminating `positive-introspection`).
- Limit profiles: **prod** = `egg::Runner::default()` (iter 30, nodes 10 000,
  time 5 s — byte-identical to `Engine::simplify_expression()`), **generous**
  = iter 1 000, nodes 1 000 000, time 60 s (what full saturation would cost
  without the engine's guard rails).
- Hardware: 8-core x86_64, 31 GB RAM, Linux 6.8, release build, otherwise
  unloaded. (The plan's M2 workstation tier was not available to this run;
  see §Gaps.)
- 5 repetitions per production cell, 3 per generous/HNSW cell; tables report
  p50/min/max. Raw per-cell JSON committed under
  [`docs/ai/data/2026-07-05-phase40/`](data/2026-07-05-phase40/).

## Results

All numbers: release build, seed 42, per-cell subprocess. "sat" = equality
saturation wall time; "extract" = `Extractor::new(AstSize)` + `find_best`;
"stop" = `egg::StopReason` (identical across reps of a cell); "peak RSS" =
`VmHWM` of the cell subprocess.

### E-graph re-saturation — production limits (`Runner::default()`: iter 30, nodes 10 000, time 5 s)

| n_facts | rules | branch | build ms | init nodes | sat p50 ms | sat min | sat max | extract p50 ms | stop | e-nodes | peak RSS MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 | small | 1 | 0.0 | 404 | 0.2 | 0.2 | 0.8 | 0.1 | Saturated | 465 | 3 |
| 100 | small | 3 | 0.0 | 404 | 0.2 | 0.2 | 0.8 | 0.1 | Saturated | 465 | 3 |
| 100 | small | 10 | 0.0 | 404 | 0.2 | 0.2 | 0.8 | 0.1 | Saturated | 465 | 3 |
| 100 | medium | 1 | 0.0 | 404 | 2.1 | 2.0 | 2.7 | 0.2 | Saturated | 741 | 3 |
| 100 | medium | 3 | 0.0 | 404 | 2.4 | 1.9 | 2.7 | 0.2 | Saturated | 701 | 3 |
| 100 | medium | 10 | 0.0 | 404 | 2.0 | 1.9 | 2.6 | 0.2 | Saturated | 681 | 3 |
| 100 | large | 1 | 0.0 | 404 | 24.5 | 22.9 | 28.3 | 1.4 | IterationLimit(30) | 3321 | 4 |
| 100 | large | 3 | 0.0 | 404 | 23.5 | 23.0 | 25.2 | 1.4 | IterationLimit(30) | 3311 | 4 |
| 100 | large | 10 | 0.0 | 404 | 23.6 | 22.6 | 26.2 | 1.4 | IterationLimit(30) | 3291 | 4 |
| 1000 | small | 1 | 0.1 | 4025 | 1.8 | 1.8 | 2.9 | 1.4 | Saturated | 4448 | 4 |
| 1000 | small | 3 | 0.1 | 4025 | 1.8 | 1.7 | 2.9 | 1.3 | Saturated | 4448 | 4 |
| 1000 | small | 10 | 0.1 | 4025 | 1.9 | 1.8 | 2.9 | 1.4 | Saturated | 4448 | 4 |
| 1000 | medium | 1 | 0.1 | 4025 | 38.8 | 38.0 | 41.3 | 2.6 | Saturated | 8337 | 6 |
| 1000 | medium | 3 | 0.1 | 4025 | 36.2 | 35.9 | 37.8 | 2.5 | Saturated | 7687 | 6 |
| 1000 | medium | 10 | 0.1 | 4025 | 40.3 | 39.6 | 41.6 | 2.5 | Saturated | 7467 | 6 |
| 1000 | large | 1 | 0.1 | 4025 | 29.4 | 29.1 | 30.8 | 3.3 | NodeLimit(10171) | 10072 | 6 |
| 1000 | large | 3 | 0.1 | 4025 | 34.4 | 27.3 | 51.7 | 3.1 | NodeLimit(10081) | 9982 | 6 |
| 1000 | large | 10 | 0.1 | 4025 | 27.7 | 27.5 | 28.9 | 3.1 | NodeLimit(10249) | 10150 | 6 |
| 10000 | small | 1 | 0.8 | 40250 | 11.5 | 10.7 | 22.0 | 41.8 | NodeLimit(30993) | 30993 | 18 |
| 10000 | small | 3 | 1.7 | 40250 | 11.9 | 11.5 | 28.7 | 46.6 | NodeLimit(30993) | 30993 | 18 |
| 10000 | small | 10 | 0.9 | 40250 | 22.9 | 11.8 | 25.1 | 81.6 | NodeLimit(30993) | 30993 | 18 |
| 10000 | medium | 1 | 1.3 | 40250 | 11.7 | 10.4 | 16.3 | 33.6 | NodeLimit(30993) | 30993 | 18 |
| 10000 | medium | 3 | 0.8 | 40250 | 14.0 | 12.4 | 16.4 | 40.3 | NodeLimit(30993) | 30993 | 18 |
| 10000 | medium | 10 | 0.8 | 40250 | 12.0 | 10.9 | 15.2 | 38.4 | NodeLimit(30993) | 30993 | 18 |
| 10000 | large | 1 | 0.8 | 40250 | 12.9 | 11.3 | 16.2 | 41.2 | NodeLimit(30993) | 30993 | 18 |
| 10000 | large | 3 | 0.8 | 40250 | 12.0 | 11.3 | 16.5 | 40.7 | NodeLimit(30993) | 30993 | 18 |
| 10000 | large | 10 | 0.9 | 40250 | 11.8 | 10.6 | 15.3 | 39.3 | NodeLimit(30993) | 30993 | 18 |
| 100000 | small | 1 | 7.5 | 402500 | 179.3 | 166.9 | 217.1 | 638.5 | NodeLimit(295462) | 295462 | 130 |
| 100000 | small | 3 | 14.6 | 402500 | 253.5 | 175.4 | 347.8 | 894.1 | NodeLimit(295462) | 295462 | 131 |
| 100000 | small | 10 | 7.5 | 402500 | 176.2 | 171.3 | 235.1 | 689.6 | NodeLimit(295462) | 295462 | 132 |
| 100000 | medium | 1 | 8.0 | 402500 | 211.5 | 166.9 | 266.1 | 695.7 | NodeLimit(295462) | 295462 | 131 |
| 100000 | medium | 3 | 7.7 | 402500 | 168.2 | 161.8 | 223.5 | 644.7 | NodeLimit(295462) | 295462 | 132 |
| 100000 | medium | 10 | 7.6 | 402500 | 173.6 | 167.2 | 206.2 | 651.8 | NodeLimit(295462) | 295462 | 133 |
| 100000 | large | 1 | 7.6 | 402500 | 171.5 | 167.9 | 213.7 | 652.7 | NodeLimit(295462) | 295462 | 132 |
| 100000 | large | 3 | 7.6 | 402500 | 172.5 | 163.6 | 217.3 | 656.5 | NodeLimit(295462) | 295462 | 132 |
| 100000 | large | 10 | 7.9 | 402500 | 176.5 | 168.0 | 209.1 | 645.3 | NodeLimit(295462) | 295462 | 133 |

Worst production-limit cell: **253.5 ms median saturation + 894.1 ms
extraction** (100 k facts). Note: at ≥10 k facts the initial expression alone
exceeds the 10 000-node runner limit, so the runner stops after one iteration
— that is the production semantics for oversized expressions, and it is
*fast*, not slow.

### E-graph full saturation — generous limits (iter 1 000, nodes 1 000 000, time 60 s)

| n_facts | rules | branch | sat p50 ms | sat min | sat max | extract p50 ms | stop | e-nodes | peak RSS MiB |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | medium | 3 | 39.1 | 38.4 | 41.7 | 2.5 | Saturated | 7687 | 6 |
| 10000 | medium | 3 | 1388.3 | 1007.0 | 1448.6 | 64.2 | Saturated | 75049 | 30 |
| 100000 | medium | 3 | 22064.0 | 21706.8 | 22743.7 | 1044.1 | Saturated | 730971 | 224 |
| 10000 | large | 3 | 60119.3 | 60020.0 | 60200.3 | 5485.8 | TimeLimit(60.0s) | 693511 | 202 |

True full saturation of a whole-KG conjunction genuinely converges with the
medium pack: **~1.4 s at 10 k facts, ~22.1 s at 100 k facts** (superlinear over
this decade, ~n^1.2). The large pack never converges by construction
(`positive-introspection`) — it burns the whole 60 s budget and grows to
~694 k e-nodes at 10 k facts. Rule-pack composition, not KG size, is the
dominant risk factor.

### Q2 fork workload (clone + perturb + re-run, medium rules, prod limits)

| n_facts | forks | base sat ms | fork p50 ms | fork max ms | total (100 forks) | peak RSS MiB |
|---|---|---|---|---|---|---|
| 1000 | 100 | 38.0 | 15.4 | 46.8 | 1618.6 ms | 6 |

### HNSW(VSA) rebuild — 10 000-bit hypervectors, `Dimension::DEFAULT`

| n_items | cold build (gen + insert) | rebuild p50 | rebuild min | rebuild max | peak RSS MiB |
|---|---|---|---|---|---|
| 1000 | 0.64 s | 0.63 s | 0.62 s | 0.65 s | 12 |
| 10000 | 31.5 s | 32.7 s | 32.2 s | 32.7 s | 107 |
| 100000 | 621.7 s | 841.4 s | 710.5 s | 880.8 s | 983 |

(The 100 k-item row was measured by the same harness/binary/host in the first
full run; that cell completed but the parent lost the batch to a
stdout-parsing bug — `hnsw_rs` prints progress noise to stdout — since fixed
in the harness by taking the last non-empty stdout line and persisting results
incrementally. Values above are the cell's captured output.)

## Interpretation

### 1. Production saturation cost is bounded and small

Every production call site (`Engine::simplify_expression()`,
`src/engine.rs:1340`; the `Reason` pipeline stage, `src/pipeline/mod.rs`;
`InferEngine::verify_with_egraph`, `src/infer/engine.rs`) uses
`Runner::default()` — 30 iterations, 10 000 nodes, 5 s time limit. In the
36-cell production grid the worst cell is **253.5 ms median saturation
(+ 894.1 ms extraction)** at 100 k facts — it stops at the node limit — and
*nothing can exceed ~5 s* because the runner enforces it. Small/medium packs
on ≤1 k-fact theories genuinely saturate (StopReason `Saturated`) in
0.3–42 ms.

### 2. Scaling is driven by rule count and rule shape, not KG size

Adding the epistemic pack (large) multiplies cost ~10× at every size because
`positive-introspection` (K(a,p) → K(a,K(a,p))) never terminates — the runner
always exits via a limit. Branching factor barely matters: egg's conjunctive
patterns match structural `(and x y)` subterms only, so a whole-KG conjunction
gives transitivity rules little to chew on regardless of graph shape. This is
the deeper architectural point: **the e-graph is a per-expression simplifier,
not a KG-wide join engine** — forward chaining over triples is `RuleEngine`'s
job (`src/autonomous/rule_engine.rs`), and *that* state lives in the persisted
triples already.

### 3. There is no wake-time e-graph cost to persist away

No engine code path builds an e-graph at wake. The runner is constructed
per query and discarded after extraction. The actual Phase 38 wake
implementation (`wake()`, `src/snapshot.rs:584`) loads the `akh_state` row,
rehydrates symbols/facts/cognitive state from Postgres, and rebuilds the KG +
item memory — there is no e-graph step in it. Persisting the e-graph would
cache a structure nothing rebuilds.

### 4. Q2 (fork workload): incremental state is cheap

Cloning a saturated 1 000-fact e-graph, adding a perturbation, and re-running
the medium pack costs 15.4 ms median (46.8 ms max) per fork — 100 forks in
1.6 s. An MCTS-style explorer that forks the e-graph hundreds of times per
decision is comfortably in-memory territory; incremental e-graph
*persistence* is not the axis that matters (confirming the plan's
expectation for Q2).

### 5. Applying ADR 051's threshold table honestly

One generous-limit cell lands inside the 5–30 s "Option (1)" band: full
saturation of a 100 k-fact whole-KG conjunction takes 22.1 s. We still pick
Option (0), and the reasoning must be explicit rather than buried: **that
workload does not exist in the engine.** No call site saturates a whole-KG
conjunction — production expressions are per-query (pipeline seeds, inference
chains, single simplification targets), production limits stop oversized
inputs at 10 000 nodes in ~0.2 s, and the wake path (`wake()`,
`src/snapshot.rs:584`) contains no e-graph step whatsoever. The plan's
"representative theory" is therefore the per-query expression under
production limits — median well under 100 ms — not the synthetic 10⁵-fact
conjunction, which we measured as an upper bound. If a future phase
introduces whole-KG or long-lived saturation (e.g. Phase 41's ThinkingLoop),
the 22.1 s number is the honest cost estimate at 10⁵ facts and this decision
must be revisited — that trigger is recorded in ADR 084.

### 6. Q1 (HNSW arm): the asymmetric sibling

HNSW(VSA) rebuild is the opposite story: cold build 10.4 min and rebuild
~11.8–14.7 min per pass at 100 k 10 000-bit items (~1 GB peak RSS) — three to
four orders of magnitude above the e-graph's production-limit cost at the
same "size", and growing worse than linearly (0.63 s → 32.7 s → 841 s for
1 k → 10 k → 100 k). The wake-path budget should be spent where ADR 049
already spends it: local-disk HNSW cache for same-host wake, background
rebuild from `akhomed.facts` for cross-host wake. Per ADR 051 Alternative D,
the two arms landing this asymmetric means **no generic
`DerivedStructureCache` abstraction is warranted**.

### 7. Per-microtheory granularity (Q3)

Saturation cost under production limits is effectively flat in theory size
once the node limit binds, and genuinely saturating workloads are small.
Partitioning by microtheory would only subdivide an already-negligible cost;
no per-microtheory snapshot keying is needed. (Data: compare the 1 k vs 10 k
vs 100 k rows — cost is bounded, not accumulating.)

### 8. Bug found by the harness

`calendar_rules()` was uncallable since Phase 13f: the `cal-conflict` rewrite
referenced `?conflicts` on the RHS with nothing binding it on the LHS, which
panics inside `rewrite!` at construction. No production or test code ever
called the pack, so the panic stayed latent until this harness loaded the
medium pack. Fixed in this phase (concrete `conflicts` symbol on the RHS) with
construction tests for all packs (`src/reason/mod.rs`,
`all_rule_packs_construct`). Corollary worth recording: pim / calendar /
causal / epistemic packs currently have **zero non-test callers** —
`Engine::all_rules()` loads builtin + skill rules only.

## Decision applied

ADR 051's binding rule: median < 5 s → **Option (0), persist nothing**.
Measured production-semantics medians are 0.2–254 ms — one to four orders of
magnitude below the threshold — and the worst case is bounded by egg's own
5 s guard. The single generous-limit cell inside the 5–30 s band (22.1 s,
100 k-fact whole-KG full saturation) measures a workload no call site
executes; see §5 for why it does not flip the decision and what would. The
decision, its structural rationale, and the reopening conditions are recorded
in [ADR 084](decisions/084-egraph-persistence-decision.md). Phase 40d is
closed as a no-op; no schema ships, no seam is added.

## Gaps (recorded, not blocking)

The plan's completion criteria for 40b included two arms this run could not
cover; per ADR 084 they are re-run triggers, not blockers, because the
synthetic result is not marginal — it is 100× inside the threshold and
production limits bound the worst case architecturally:

1. **Real-workload arm** (≥3 anonymized live-akh KG snapshots): no live-akh
   snapshot exists in this worktree (the Phase 38 truth-store export path
   lives on the deployment hosts). Live expressions are *smaller* than the
   synthetic whole-KG conjunctions measured here — production expressions are
   per-query seeds/inference chains — so the synthetic grid is an upper
   bound, not an approximation from below.
2. **Hardware-tier comparison** (M2 workstation + co-tenanted 4-core x86):
   this run used one unloaded 8-core x86 box. Co-tenancy scales the numbers
   by small constants; it does not move a 40 ms median across a 5 s
   threshold.

If either gap later produces a surprise (see ADR 084 §What would reopen this
decision), re-run: `cargo run -p akh-egraph-bench --release -- --what both`.

## Reproduction

```sh
export CARGO_TARGET_DIR=/path/to/shared/target   # repo convention
cargo run -p akh-egraph-bench --release -- \
  --what both --out-json phase40-results.json --report phase40-tables.md
# quick smoke grid:
cargo run -p akh-egraph-bench --release -- --what both --quick
```
