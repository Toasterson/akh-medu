# ADR 085 — Thinking Engine Findings (Phase 41 scaling-law results)

> Date: 2026-07-05
> Status: Accepted
> Phase: 41 (The Thinking Engine) — closes sub-phase 41g
> Supersedes: nothing; records the empirical outcome of the bet made in [ADR 052](052-unified-thinking-loop.md)
> Plan: [docs/ai/plans/2026-04-27-phase41-thinking-engine.md](../plans/2026-04-27-phase41-thinking-engine.md)
> Measurement report: [docs/ai/2026-07-05-phase41f-scaling-law.md](../2026-07-05-phase41f-scaling-law.md)
> Code: `src/thinking/` (loop, scheduler, verifier, VOC budget, bench harness)
> Anima ticket: `019dd0c5-9f4b-74e3-ad5c-3bc0a22e0ef9`

## Context

ADR 052 made a falsifiable bet: composing akh-medu's existing reasoning
primitives (spreading activation, e-graph saturation, MCTS, argumentation,
counterfactual fork, Dempster-Shafer fusion, superposition merge) into one
budgeted `ThinkingLoop` would demonstrate a test-time-compute scaling law —
the property that defines frontier LLM reasoning architectures — on a
substrate where every step is faithful and auditable.

Phase 41a–41c, 41e, and 41f are now built and measured (41d persistence is a
seam only, deferred with Phase 38 WAL integration). This ADR records what the
data actually says.

## Findings

### 1. The scaling law holds — on the family where compute is search

On multi-hop KG queries (find the answer-typed node at the end of a relation
chain buried in ~25 chaff nodes per ring), solvable depth grows monotonically
with step budget:

| step budget | 100 | 300 | 1,000+ |
|---|---|---|---|
| max solvable depth | 4 | 16 | 32 |
| aggregate accuracy | 40% | 80% | 100% |

Steps-to-answer is linear in depth (≈ 18.7·d + 20), so accuracy against
**log budget** is the LLM-shaped rising curve. This is the "single dial
labelled think harder": `engine.deliberate(goal, ThinkingBudget::steps(n))`
converts budget into solved depth, deterministically, with the full
`ThoughtStep` trace as evidence.

The mechanism is honest and mundane: the budget buys frontier expansions.
There is no emergent magic — which is precisely the point. Where an LLM
converts extra tokens into implicit search, the loop converts extra steps
into explicit search, and the exchange rate is measurable.

### 2. The law does NOT hold where the substrate has an O(1) answer

- **Analogy** (`A : B :: C : ?` over structured bindings): flat at 100% from
  budget 100 — one exact `⊗`/cleanup resolves it in the first spreading
  step. Extra compute buys *nothing*.
- **Factual evidence fusion**: flat at 100%, saturating in ~17 steps
  (argumentation verdict + one Dempster combination round).

The plan demanded at least one non-scaling family as the informative
negative; we got two, and they invert the LLM narrative in an interesting
way: **test-time compute is a substitute for algebraic structure, not a
complement to it.** Where the VSA/KG substrate already encodes the answer
(bind/unbind arithmetic, direct triples), the scaling dial is dead weight;
where the answer requires traversal the substrate hasn't precomputed, the
dial is the whole game. A budget-vs-accuracy curve is therefore a *diagnostic
of representation quality*: a steep curve on a task family says the KG lacks
a shortcut (index, schema, materialised relation) for it.

### 3. The VOC trap: we reproduced the illusion-of-thinking failure ourselves

The 41e controller as designed (halt when Phase 11g marginal value < 0,
improvement signal = verifier-score deltas) **killed every deep search at 16
steps** on the first full sweep — 0% on depth ≥ 8 with 99.8% of the budget
unspent. The verifier score is flat while no hypothesis exists, a flat score
reads as diminishing returns, and the controller gave up exactly the way
[Apple's *Illusion of Thinking*](https://arxiv.org/abs/2506.06941) describes
reasoning LLMs giving up on hard problems.

The lesson generalises: **a value-of-computation signal derived from answer
quality is blind to exploration progress.** Search has two regimes —
*exploration* (no answer yet; progress = novelty/frontier consumption) and
*refinement* (answer in hand; progress = verifier-score delta) — and VOC
halting is only sound in the second. The shipped rule: diminishing-returns
never fires while unexplored frontier remains and no hypothesis has formed.
With that guard, adaptive mode matches fixed-budget accuracy on every family
while saving 4–27% of steps (halting once the answer stops improving).

Follow-up (not built): feed an explicit exploration-progress term (novelty
rate, frontier consumption rate) into the improvement history so the
controller can also detect *futile* exploration — currently a dead-end goal
with a huge frontier will spend its whole fixed budget.

### 4. The verifier needed the same regime split

The convergence sub-verifier (focus-vector stability) also mistook
"traversal through chaff" for convergence: spreading steps that pop only
dead-end leaves move the focus by ~0, and four quiet deltas halted the loop
mid-search. Fixed the same way: focus stability only counts as convergence
with an answer in hand or an exhausted frontier. **Both mid-flight bugs were
in the halting policy, not in the primitives or their composition** — the
integration bet itself held up; deciding when to *stop* integrating is where
the design risk actually lived.

### 5. What composition cost, and what it bought

- **No new reasoning algorithms were needed.** Every primitive ran through
  its existing public API (`Engine::infer` with per-step seeding for
  granularity, `Engine::argue`, `MassFunction::combine`, egg `Runner`,
  `MctsPlanner::plan`, `counterfactual_query`, `VsaOps::bundle`). The
  ADR 052 risk "primitives may need step-wise APIs" did not materialise for
  this suite — one-frontier-ring-per-step falls out of `max_depth: 1` +
  loop-owned frontier.
- **The trace is the computation.** Every run yields a `Vec<ThoughtStep>`
  with per-primitive payloads (novelty, belief intervals, verdicts, verifier
  reasons); the 270-run sweep is fully deterministic (all 5 instances of
  every bucket agree), which made both halting bugs one-look diagnosable
  from the trace. This is the faithfulness property ADR 052 promised, doing
  real debugging work on day one.
- **MCTS and counterfactual contributed nothing on this suite** (no action
  schemas in the bench KGs → recorded no-op steps). The goal-shape scheduler
  correctly weights them to zero for factual/analogy/exploratory goals, so
  they cost nothing here — but their composition is exercised only by unit
  tests, not by the benchmark. A planning-family benchmark (goals with
  Phase 15 action schemas) is the missing measurement.

### 6. Smaller empirical notes

- Item-memory cleanup for hypothesis resolution uses HNSW approximate NN;
  one sweep configuration lost 1/10 analogy tasks to a recall miss despite
  the exact answer vector being present. Exact-match cleanup deserves an
  exact path (or a verify-then-fallback re-rank) before analogy accuracy is
  quoted anywhere load-bearing.
- Verifier cadence (full composite every 4 steps, cheap convergence check
  every step) kept verification under ~15% of wall time; the plan's
  hierarchical-verifier mitigation was enough.
- Wall costs are small: depth-32 search (607 steps, ~840 KG nodes touched)
  ≈ 0.5 s in release at 1,000-bit test dimension.

## Decision

1. **Keep the Thinking Loop as an additive surface** (as ADR 052 specified):
   `engine.deliberate()` is now the deliberation entry point; existing
   `infer()`/`argue()`/`mcts.plan()` call sites are untouched.
2. **Adopt the two-regime halting policy** (exploration vs refinement) as a
   design rule for every future halting/valuation signal in the loop,
   including the planned learned rotation (41h) and any Phase 20 active-
   inference integration.
3. **41d (Postgres `thinking_traces` persistence) stays deferred** behind the
   `ThoughtPersistence` seam until it can ride the Phase 38 WAL; nothing in
   the measurements argued for pulling it forward.
4. **Extension worth planning** (in this order of expected value):
   a. exploration-progress term in the VOC controller (close the futile-
      exploration gap);
   b. a planning-family benchmark exercising MCTS/counterfactual composition;
   c. exact-match cleanup path in item memory;
   d. Raven's-matrices family once data files are curated (NVSA comparison).
5. **Do not** chase scaling on the flat families. The right response to a
   flat curve is representation work (indexes, schemas, materialised
   relations), not more thinking steps — finding 2 is the operating manual.

## Consequences

- Akh-medu now demonstrably converts a compute budget into accuracy on
  search-shaped reasoning, with a faithful, replayable trace per thought —
  the empirical core of the "frontier architecture" thesis, at the scale of
  a synthetic-but-adversarial benchmark. The claim to publish is scoped
  accordingly: *scaling law on explicit-search families over a KG substrate*,
  not general reasoning parity.
- The scaling curve doubles as a KG-quality diagnostic (finding 2), which no
  token-based reasoner offers.
- The illusion-of-thinking reproduction (finding 3) is the strongest external
  story from this phase: the failure mode is architecture-independent — any
  system that judges continued thinking by answer-quality deltas will give up
  during exploration — and the structural fix is two lines once the trace
  makes the regime visible.

## References

- [ADR 052 — Unified Thinking Loop](052-unified-thinking-loop.md)
- [Phase 41 plan](../plans/2026-04-27-phase41-thinking-engine.md)
- [41f measurement report](../2026-07-05-phase41f-scaling-law.md)
- [Research synthesis: thinking in LLMs vs VSA](../research/2026-04-27-thinking-in-llms-vs-vsa.md)
- [Apple, *The Illusion of Thinking* (NeurIPS 2025)](https://arxiv.org/abs/2506.06941)
- [Snell et al., test-time compute (2024)](https://arxiv.org/html/2408.03314v1)
