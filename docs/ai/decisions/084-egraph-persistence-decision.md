# ADR 084 — E-graph Persistence: Option (0), Persist Nothing

> Date: 2026-07-05
> Status: ACCEPTED
> Phase: 40 (closes 40b/40c/40d)
> Supersedes: amends [ADR 051](051-egraph-persistence.md) (`Status: PROPOSED —
>             pending measurement` → decided here; 051's context, alternatives,
>             and threshold table remain the record of *why* we measured first)
> Plan: [`docs/ai/plans/2026-04-25-phase40-egraph-persistence.md`](../plans/2026-04-25-phase40-egraph-persistence.md)
> Findings: [`docs/ai/2026-07-05-phase40-egraph-resaturation.md`](../2026-07-05-phase40-egraph-resaturation.md)
> Harness: [`crates/akh-egraph-bench/`](../../../crates/akh-egraph-bench/)

## Decision

**Option (0): persist nothing.** The e-graph remains a derived, per-query,
in-memory structure. It is rebuilt by `egg::Runner::default()` on every
`Engine::simplify_expression()` / `Reason`-stage call, exactly as today. No
snapshot table, no serialization format, no invalidation logic, no seam. Phase
40d is closed as a no-op.

ADR 051's binding rule was: median time-to-saturation on a representative
theory < 5 s → Option (0). The measured numbers (2026-07-05, 8-core x86_64,
31 GB RAM, release build, `crates/akh-egraph-bench`, seed 42; full tables in
the findings report):

| Workload (medium rule pack, `branching = 3`) | Median saturation |
|---|---|
| 1 000-fact theory, production `Runner::default()` limits | 38.1 ms (Saturated) |
| 10 000-fact theory, production limits | 11.8 ms (NodeLimit after 1 iteration) |
| 100 000-fact theory, production limits | 176.4 ms + 670.9 ms extraction (NodeLimit) |
| Worst cell in the whole 36-cell production grid | 178.7 ms (architecturally ≤ 5 s: egg's own time limit) |
| 10 000-fact theory, *generous* limits (1 M nodes / 60 s) | 0.97 s (genuinely Saturated) |
| 100 000-fact theory, generous limits | 19.5 s (genuinely Saturated) |
| 10 000-fact theory, *large* pack, generous limits | 60 s (TimeLimit — `positive-introspection` never terminates) |

The production-semantics median is one to four orders of magnitude below the
5 s threshold. One generous-limit cell — 100 k-fact whole-KG full saturation
at 19.5 s — falls inside ADR 051's 5–30 s "Option (1)" band. It does not
flip the decision because no call site executes that workload (finding 1
below, findings-report §5); it stands as the honest cost estimate if one
ever does, and the reopening triggers below cover that case. The
deliberately pathological epistemic arm is bounded by the runner's own
limits — a bound persistence could not improve, since the "saturated
product" it would cache never exists for a non-terminating rule set.

## Why the measurement was decisive beyond the raw numbers

Three structural findings from the harness matter more than the timings:

1. **There is no wake-time e-graph rebuild to amortize.** The engine never
   builds a KG-wide e-graph. `Engine::simplify_expression()`
   (`src/engine.rs:1340`) and the `Reason` pipeline stage
   (`src/pipeline/mod.rs`) each construct a fresh `Runner` around a *single
   query expression* and discard it after extraction. The Phase 38 wake
   implementation (`wake()`, `src/snapshot.rs:584`) rehydrates symbols,
   facts, cognitive state, KG, and item memory from Postgres — it contains no
   e-graph step. A persistence layer would be caching a structure that no
   code path ever rebuilds at wake.

2. **Production saturation cost is architecturally bounded.** Every production
   call site uses `egg::Runner::default()`: 30 iterations, 10 000 nodes, 5 s
   time limit. The wake-/query-path cost therefore *cannot* exceed ~5 s per
   query regardless of KG size — the 100 k-fact cells hit the node limit in
   ~0.18 s (+ ~0.67 s extraction). Persistence cannot improve a bound that
   the runner already enforces.

3. **Conjunctive rules don't do KG-wide joins anyway.** egg pattern rules
   match structural `(and x y)` subterms; a whole-KG conjunction only fires
   rules where related facts happen to be adjacent in the and-tree. Whole-KG
   saturation is not just unnecessary — it is not even the right tool. That
   role belongs to `RuleEngine` (forward chaining over `KnowledgeGraph`
   triples, `src/autonomous/rule_engine.rs`), which operates on the persisted
   triples directly.

Additional supporting data:

- **Q2 (incremental/fork workloads):** cloning a saturated 1 000-fact e-graph,
  perturbing, and re-running costs 15.3 ms median (51 ms max) per fork — 100
  forks in 1.6 s. MCTS-style exploration does not need incremental e-graph
  persistence.
- **Memory:** peak RSS for the largest production-limit cells stayed at
  ~132 MiB; the generous 100 k cell at 224 MiB. Nothing here pressures a
  snapshot-and-restore design.
- **HNSW(VSA) is the real wake-path cost** (Q1 arm): rebuild at 100 k
  10 000-bit items is 841 s median (~14 min, ~1 GB peak RSS) on this
  hardware, versus 0.18 s for the e-graph under production limits at the
  same scale. This confirms ADR 049's already-decided
  mitigation (local-disk HNSW cache for same-host wake, rebuild from
  `akhomed.facts` cross-host) is aimed at the right structure, and answers
  ADR 051 Alternative D: the two measurements are *not* symmetric, so no
  generic `DerivedStructureCache` abstraction is warranted.

## What would reopen this decision

Re-run `cargo run -p akh-egraph-bench --release -- --what egraph` and
re-apply ADR 051's threshold table if any of these change:

- A feature starts building **long-lived or KG-wide e-graphs** (e.g. a future
  Reason stage that saturates whole microtheories instead of query
  expressions, or Phase 41's ThinkingLoop holding an e-graph across turns).
- The **active rule count grows by an order of magnitude** (skill-loaded rule
  packs at ~100+ rewrites), since per-iteration matching cost scales with
  rules × nodes.
- Production call sites **raise the default runner limits** materially.
- The real-workload arm (deferred, see Findings §Gaps) surfaces live-akh
  expressions that behave qualitatively unlike the synthetic grid.

Until one of those happens, any e-graph persistence code is complexity
without a payer.

## Consequences

- **Positive:** zero new schema, zero invalidation logic, zero egg-version
  coupling. Wake-time SLA budgeting (Phase 38 / ADR 049) can treat the
  e-graph contribution as 0 and focus on HNSW, which the data says is the
  actual cost.
- **Positive (side effect):** the measurement harness is permanent wake-path
  infrastructure (`crates/akh-egraph-bench`, arms: `egraph`, `hnsw`, `fork`),
  and building it surfaced and fixed a latent production panic —
  `calendar_rules()`'s `cal-conflict` rewrite referenced an unbound RHS
  variable (`?conflicts`) and would have crashed any caller since Phase 13f;
  it now derives the concrete `conflicts` symbol and all packs have
  construction tests.
- **Negative:** none beyond the sunk measurement cost (~1 000 lines of
  harness + this documentation), which ADR 051 accepted up-front.
- **Neutral:** the decision is reversible; the harness plus the threshold
  table make re-deciding a one-day exercise.

## References

- ADR 051 (measure-before-deciding rationale and threshold table)
- ADR 049 (relocatable akh — wake protocol; HNSW local-disk cache decision)
- Findings report: [`docs/ai/2026-07-05-phase40-egraph-resaturation.md`](../2026-07-05-phase40-egraph-resaturation.md)
- Production e-graph call sites: `src/engine.rs` (`simplify_expression`),
  `src/pipeline/mod.rs` (Reason stage), `src/infer/engine.rs`
  (`verify_with_egraph`)
- Anima ticket: `019dc59a-513a-74a1-a088-d1524d891095`
