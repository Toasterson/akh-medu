# Phase 41f — Thinking Loop Scaling-Law Measurement

> Date: 2026-07-05
> Status: Measured (release build, deterministic seed 41)
> Plan: [2026-04-27-phase41-thinking-engine.md](plans/2026-04-27-phase41-thinking-engine.md) (sub-phase 41f)
> ADRs: [052 — Unified Thinking Loop](decisions/052-unified-thinking-loop.md) (design), [085 — Thinking Engine Findings](decisions/085-thinking-engine-findings.md) (interpretation)
> Harness: `src/thinking/bench.rs`, runner `examples/bench_thinking.rs`
> Raw data: `target/bench-results/thinking/{runs,summary}.csv` (regenerate with `cargo run --release --example bench_thinking`)

## Setup

One in-memory engine (`Dimension::TEST` = 1,000-bit hypervectors), 45 seeded
tasks, 270 runs total (45 tasks × 5 budgets fixed + 45 adaptive), ~24 s wall
for the whole sweep on an 8-core Linux box. Every run is deterministic: all 5
instances of every (family, budget) bucket agree on success/steps.

Task families (goals are regular Entity symbols with `thinking:*` query
relations; the loop uses the 41b goal-shape scheduler, 41c composite
verifier, and — in adaptive mode — the 41e VOC controller):

1. **Multi-hop KG queries** (depths 2, 4, 8, 16, 32; 5 instances each): find
   the unique answer-typed node at the end of a relation chain; every chain
   node also sprouts 6 dead-end branches with 3 leaf children each (25 nodes
   of chaff per ring), created before the chain node so activation ties break
   *toward* the chaff. Discovery requires real beam-limited frontier search.
2. **Analogical retrieval** (10 instances): `A : B :: C : ?` over structured
   item-memory bindings (`B = A ⊗ R`, `D = C ⊗ R`; the ADR 002 encoding)
   with 8 distractor symbols each — resolvable in one exact VSA operation.
3. **Factual evidence queries** (10 instances): `(S, R, ?)` with one
   candidate at confidence ~0.9 and four at 0.15–0.40; argumentation +
   Dempster-Shafer fusion must pick the well-supported one.

The plan's fourth benchmark (Raven's Progressive Matrices) needs external
data files and was out of scope for this run.

## Results — accuracy vs. step budget (fixed mode)

| family | 100 | 300 | 1000 | 3000 | 10000 | steps-to-answer |
|---|---|---|---|---|---|---|
| multi-hop-d2  | **100%** | 100% | 100% | 100% | 100% | 22 |
| multi-hop-d4  | **100%** | 100% | 100% | 100% | 100% | 58 |
| multi-hop-d8  | 0% | **100%** | 100% | 100% | 100% | 157 |
| multi-hop-d16 | 0% | **100%** | 100% | 100% | 100% | 283 |
| multi-hop-d32 | 0% | 0% | **100%** | 100% | 100% | 607 |
| multi-hop (aggregate) | 40% | 80% | **100%** | 100% | 100% | — |
| analogy       | 100% | 100% | 100% | 100% | 100% | 4 |
| factual       | 100% | 100% | 100% | 100% | 100% | 17 |

Wall time per solved multi-hop task: d2 ≈ 15 ms, d8 ≈ 100 ms, d16 ≈ 190 ms,
d32 ≈ 470–560 ms (release, includes verifier runs every 4 steps).

**The scaling law on the search family**: solvable depth grows monotonically
with budget — 100 steps buys depth 4, 300 buys depth 16, 1,000 buys depth 32.
Steps-to-answer is linear in depth (`≈ 18.7·d + 20`, r² > 0.99 over the five
depths), so aggregate accuracy vs. **log** budget rises 40% → 80% → 100%
across 100/300/1000 — the frontier-LLM-shaped curve, produced by explicit
budgeted search instead of token sampling. Above the knee the curve is
perfectly flat: once every task's search completes, additional budget is
simply not spent (the verifier or convergence halts the loop; budget 10,000
never binds).

**The negative results the plan asked for** (41f acceptance: at least one
family must NOT scale):

- **Analogy is one-shot.** Exact VSA arithmetic answers in the first
  spreading step (4 steps to sufficiency, belief 0.85); accuracy is flat at
  100% from budget 100 up. Test-time compute buys nothing when the substrate
  has an O(1) algebraic answer — the interesting inversion of the LLM story.
  (One earlier sweep saw 9/10 here: a single HNSW approximate-recall miss in
  a then-4k-symbol item memory, not a budget effect; it vanished when task
  layout changed. Exact-match cleanup should not rely on approximate NN — see
  ADR 085.)
- **Factual evidence fusion saturates in ~1.5 rotations** (17 steps): argue →
  combine → verify converges immediately at every budget.

## Halt-reason distribution (270 runs)

| halt reason | fixed | adaptive |
|---|---|---|
| converged | 155 | — |
| verifier-sufficient | 50 | 10 |
| budget-exhausted | 20 (all = the failed low-budget multi-hop runs) | — |
| diminishing-returns | — | 35 |

## Adaptive (41e VOC) mode, budget 10,000

| family | accuracy | steps (vs fixed) |
|---|---|---|
| multi-hop d2/d4/d8/d16/d32 | 100% | 16/48/128/272/576 (−27%/−17%/−18%/−4%/−5%) |
| analogy | 100% | 4 (=) |
| factual | 100% | 16 (−6%) |

The controller halts *after* the answer is in hand, when verifier-score
deltas flatten — economic halting with zero accuracy cost on this suite.

**But this took a fix that is itself a headline finding.** The first sweep
ran the controller exactly as planned (halt when Phase 11g marginal value <
0) and it **reproduced the Apple illusion-of-thinking failure it was built to
prevent**: during frontier search the verifier score is flat (no hypothesis
yet → nothing to score), the flat score read as "diminishing returns", and
every multi-hop task deeper than 2 was killed at 16 steps (0% accuracy with
9,984 steps of budget left). The fix: the diminishing-returns rule only
applies to the *refinement* regime — it never fires while unexplored frontier
remains and no hypothesis has formed. Full analysis in ADR 085 §"The VOC
trap".

## Fixes the measurement forced (methodological yield)

1. **Premature convergence** (`4b563ed`): spreading steps that pop only leaf
   chaff push zero focus deltas; the stability window fired mid-search 22
   steps in. Convergence now additionally requires an answer in hand or an
   exhausted frontier.
2. **The VOC trap** (`d6c61a3`): above.

Both bugs were invisible to unit tests and to the smoke config — only the
full-size adversarial task set exposed them. The benchmark paid for itself
before producing a single curve.

## Reproduction

```sh
cargo run --release --example bench_thinking          # full sweep, ~25 s
AKH_BENCH_QUICK=1 cargo run --release --example bench_thinking  # smoke
```

CSV columns: `task_id, family, depth, budget, mode, success, steps_used,
wall_ms, voc_used, halt_reason, belief`.
