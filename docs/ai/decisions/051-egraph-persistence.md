# ADR 051 — E-graph Persistence: Measure Before Deciding

> Date: 2026-04-25
> Status: ACCEPTED (2026-07-05) — measurement ran; Option (0) selected.
>         Decision record: [ADR 084](084-egraph-persistence-decision.md).
>         Data: [findings report](../2026-07-05-phase40-egraph-resaturation.md).
> Phase: 40
> Depends on: Phase 38 (Relocatable Session-Based Akh — wake protocol this
>             would hook into)
> Related: ADR 037 (LLM-VSA deep integration — sibling wake-path cost)
> Plan: [`docs/ai/plans/2026-04-25-phase40-egraph-persistence.md`](../plans/2026-04-25-phase40-egraph-persistence.md)

## Context

Phase 38 (Relocatable Session-Based Akh) defines a three-tier sleep model —
light, deep, hibernated — and a postgres-backed truth store from which the akh
wakes onto a fresh workstation. Wake means rehydrating every **derived**
runtime structure: the knowledge graph from triples, the VSA item memory and
HNSW index, the spreading-activation caches, and — the subject of this ADR —
the e-graph saturation product produced by `egg`.

Today the e-graph is rebuilt fresh on every call. `Engine::simplify_expression()`
([`src/engine.rs:1174-1184`](../../src/engine.rs)) and the `Reason` pipeline
stage ([`src/pipeline/mod.rs:430-447`](../../src/pipeline/mod.rs)) both
construct a `Runner`, call `run(&rules)` against a freshly-parsed expression,
and discard the result after extraction. There is no long-lived `EGraph`
inside `Engine` and nothing on disk. The rule packs themselves
([`src/reason/mod.rs`](../../src/reason/mod.rs): builtin, pim, calendar,
causal, epistemic, plus skill-loaded rules) are pure Rust functions returning
`Vec<Rewrite<AkhLang, ()>>`.

The wake-path tension is straightforward:

- If re-saturation against the typical KG is fast (sub-second), persistence is
  **harmful**: it adds storage, invalidation logic, and an egg-version coupling
  surface for no benefit.
- If re-saturation is slow (tens of seconds), persistence is **necessary**:
  the Phase 38 wake-time SLA cannot be met without caching the saturation
  product.

We do not currently know which regime we are in. `egg` is not generally
known as "slow" — saturation on small theories is millisecond-scale — but
akh-medu loads many rule packs into a single global e-graph (microtheory
rules, predicate-hierarchy rules, second-order rules from
[`src/reason/second_order.rs`](../../src/reason/second_order.rs), causal,
epistemic), and the KG can grow to 10⁵+ triples on a long-running akh. The
combination is unmeasured.

There is a deeper architectural question buried in this: **the e-graph is a
derived structure, not a primary one.** The primary state of the akh is the
knowledge graph (triples + provenance, persisted in the truth store) plus
the VSA item memory (binary hypervectors, also persisted). Everything else —
the HNSW index, the spreading-activation caches, the e-graph — is a function
of those primaries plus the active rule set. Caching a derived structure
across a sleep boundary is a **performance optimization**, not a correctness
requirement. It is the kind of optimization that should be justified by
measurement, not asserted on intuition.

This ADR records the decision **not to commit** to an implementation now. It
sets up the measurement work and binds future-us to a decision rule based on
the data. The shape of the ADR is deliberately amenable to amendment: the
threshold table in §Decision is the binding rule, and Phase 40c amends a
single subsection once measurement is in.

## Decision

### Build a measurement harness, run it on representative workloads, then amend this ADR with the chosen path.

Concretely:

1. Phase 40a builds a benchmark crate (`crates/akh-egraph-bench/`) with a
   synthetic theory generator (parameterized by fact count, rule-set size,
   and branching factor) and a real-workload loader (anonymized live-akh KG
   snapshots).
2. Phase 40b runs the harness across the cross product 10²..10⁵ facts ×
   small/medium/large rule sets × multiple branching factors, plus the real
   workloads, and produces a markdown report with time-to-saturation, e-class
   count, and memory-peak distributions.
3. Phase 40c amends this ADR (Status: PROPOSED → Status: ACCEPTED, Decision:
   refined to one of the three options below) using the binding rule:

| Median time-to-saturation, representative theory | Choice |
|---|---|
| < 5 s | **Option (0)** — persist nothing, re-saturate on wake |
| 5 s – 30 s | **Option (1)** — persist canonicalized e-class roots only |
| > 30 s | **Option (2)** — persist full e-graph state |

### Selected (2026-07-05): Option (0) "rebuild from facts"

Measurement ran on 2026-07-05 (`crates/akh-egraph-bench`, 8-core x86_64,
release; findings in
[`docs/ai/2026-07-05-phase40-egraph-resaturation.md`](../2026-07-05-phase40-egraph-resaturation.md)).
Median saturation under production `Runner::default()` semantics was
**38.1 ms at 1 000 facts, 11.8 ms at 10 000, 176.4 ms at 100 000** (medium
pack) — far below the 5 s threshold — and full whole-KG saturation under
generous limits reached 19.5 s only at 100 000 facts, a workload no call
site executes. Option (0) is selected; details, the honest treatment of the
one in-band generous cell, and reopening triggers are in
[ADR 084](084-egraph-persistence-decision.md).

The pre-measurement reasons for leaning (0), all confirmed by the data:

- The e-graph is a **derived structure**. Persisting derived state always
  carries an invalidation cost; persisting derived state that is also
  *cheap to recompute* is a net loss.
- `egg` 0.11's internal types (`EGraph`, `EClass`, `UnionFind`, the
  `Analysis` slot) are not part of a stable serialization contract. Any
  persistence solution must serialize a canonical form we own, which is
  itself non-trivial work — work we'd be doing on the assumption that the
  cost it amortizes is real.
- Wake latency is a **UX concern, not a correctness concern**. Even with
  Option (0), a slow first-saturation after wake degrades responsiveness but
  doesn't break anything. We can ship Option (0), measure user-visible wake
  latency in production, and revisit if it's a problem in practice rather
  than a problem in theory.
- Premature optimization in this space is **expensive in both directions**:
  the bytes on disk, the postgres schema, and the cognitive load on every
  future contributor who has to reason about whether the cached e-graph is
  stale relative to the current facts.

This was "lean toward (0)" not "commit to (0)" until the data came in; the
threshold table above bound the choice, measurement came in under 5 s, and
the lean became the decision (ADR 084).

## Alternatives Considered

### Alternative A — Pick Option (0) without measuring

Rejected. We don't have data on real saturation cost on representative
KGs. "It's probably fast" is a guess; the akh-medu rule set is large
enough (~50 rewrites across the built-in packs alone) that the guess could
be wrong. The cost of being wrong is shipping a wake-time SLA that fails on
the first user with a non-trivial KG.

### Alternative B — Pick Option (2) without measuring

Rejected for the opposite reason. Building full-state persistence is on the
order of 1500 lines of code plus a postgres schema, plus invalidation logic,
plus a canonical serialization format. If Option (0) turns out to be
adequate, all of that is dead weight that future contributors have to
maintain and reason about. The complexity is permanent; the imagined latency
problem may not be real.

### Alternative C — Pick Option (1) as a hedge

Rejected. Option (1) is a real engineering choice with real complexity
(canonicalization, partial invalidation, replay protocol) — it is **not** a
"safe middle". If Option (0) is right, Option (1) is over-built. If Option
(2) is right, Option (1) under-delivers and we end up doing the work twice.
Hedging without data points to neither end usefully.

### Alternative D — Build a generic "derived structure cache" abstraction

Rejected for now. The argument for a generic cache (covering e-graph,
HNSW(VSA), spreading-activation, etc.) only becomes real if multiple
independent measurements all land in the "yes, persist" regime. Until at
least two of those are in, designing a unifying abstraction is speculative.
This is flagged in the plan as a follow-up worth revisiting if Phase 40b's
HNSW arm (see Q1 in the plan) lands symmetric to e-graph.

### Alternative E — Persist incrementally, append-only

Rejected. The motivation (avoid full-rebuild on every fact mutation) is
real but solves a problem we have not shown to exist. If Option (1) wins,
it already gives us narrow invalidation; if Option (2) wins, the union-find
is global and incremental persistence is incoherent. Append-only persistence
adds replay semantics on top of either, which compounds complexity without
addressing the underlying "do we need persistence at all?" question.

### Why measure-and-decide beats picking blindly

All four "pick now" alternatives above share one failure mode: they commit
to a complexity level before establishing whether the complexity is paid
for. The measurement phase itself is **cheap** — a single workspace crate,
no production deployment, reusable for HNSW(VSA) and other wake-path
measurements as a side benefit. The cost of measure-first is a
documentation phase delay; the cost of pick-first is potentially a wrong
implementation that survives in the codebase forever.

## Consequences

### Positive

- **Avoids premature commitment.** No code is written that the data
  doesn't support.
- **Lets data drive the choice.** Future readers of the codebase see "we
  ship Option X because median saturation was Y seconds on real workloads"
  in the amended ADR, not "we guessed Option X felt right".
- **Builds reusable measurement infrastructure.** The `crates/akh-egraph-bench/`
  harness is the obvious place to add HNSW(VSA) rebuild benches, spreading-
  activation benches, and any other wake-path measurement we need going
  forward. The first user gets the harness; later phases reuse it.
- **Decouples from egg internals up-front.** By committing in advance to
  "if we persist, we own the format", we avoid the trap of serializing
  egg's private types and then being held hostage by an egg version bump.

### Negative

- **Introduces a measurement phase before implementation.** Phase 40a/40b
  is real work — ~700 lines for the harness, plus the run time and report.
  If Option (0) wins, that work produces no shipping code; only a report
  and an ADR amendment.
- **May delay the wake-time SLA we want.** Phase 38 may want a concrete
  e-graph wake guarantee. We cannot give one until 40b is in. If Phase 38
  ships before Phase 40, it ships with "wake includes a possibly-slow
  e-graph rebuild on first reasoning call" as a known caveat.
- **Decision rule is brittle to bimodal distributions.** If saturation
  times split into "most theories: 1 s, a few theories: 60 s", a single
  threshold-based pick is dishonest about the long tail. The plan handles
  this by reporting p50/p90/p99 explicitly, but the *decision* is still a
  single option — we accept that some workloads will be on the slow path.

### Neutral

- **Egg version coupling stays loose.** Whichever option wins, the
  serialization layer is ours. Egg minor-version bumps remain free, because
  the on-disk format never references egg's private types directly.
- **Phase 38 truth store gains a new optional table.** `akhomed.egraph_snapshots`
  is created only if 40c picks (1) or (2). The schema migration is keyed
  off the chosen option, not landed speculatively.
- **The bench harness is permanent infrastructure regardless of outcome.**
  Even in the Option (0) branch, `crates/akh-egraph-bench/` stays in the
  repo as a wake-path measurement tool. Future phases that touch the wake
  path (HNSW(VSA) compression, spreading-activation cache invalidation,
  microtheory partition restructuring) reuse it.
- **Decision is reversible.** Picking (0) today does not foreclose (1) or
  (2) tomorrow. If user-visible wake latency turns out to be a problem in
  production, we re-run the harness on production-shape KGs and re-decide.
  The threshold table in §Decision is the rule each time.

## Implementation Note

This ADR is **explicitly amenable** to amendment by 40c. The structure —
Context, Decision (with the threshold table as the binding rule), Alternatives,
Consequences — should survive the amendment unchanged. The amendment edits
the Status line, replaces the "Default heading: Option (0)" subsection with
"Selected: Option (X), median saturation was Y s on workload Z", and leaves
everything else.

If a reader of this ADR after 2026-Q3 still sees `Status: PROPOSED — pending
measurement`, that is itself a finding: either Phase 40b never ran, or it ran
and the amendment was forgotten. Both are bugs worth investigating.

## References

- Phase 40 plan: [`docs/ai/plans/2026-04-25-phase40-egraph-persistence.md`](../plans/2026-04-25-phase40-egraph-persistence.md)
- Anima source ticket: `019dc59a-513a-74a1-a088-d1524d891095`
- E-graph use sites in akh-medu:
  - [`src/engine.rs:1174-1184`](../../src/engine.rs) — `Engine::simplify_expression()`
  - [`src/pipeline/mod.rs:430-447`](../../src/pipeline/mod.rs) — `Reason` pipeline stage
- Rule packs:
  - [`src/reason/mod.rs`](../../src/reason/mod.rs) — `builtin_rules`,
    `pim_rules`, `calendar_rules`, `causal_rules`, `epistemic_rules`
  - [`src/reason/second_order.rs`](../../src/reason/second_order.rs) — second-order
    quantification rules (Phase 9n)
  - [`src/autonomous/rules.rs`](../../src/autonomous/rules.rs) — forward-chaining
    rule data type (sibling, not e-graph)
- Bench precedent: [`benches/phase26.rs`](../../benches/phase26.rs),
  [`benches/vsa_ops.rs`](../../benches/vsa_ops.rs)
- Workspace crate precedent: [`crates/seshat/`](../../crates/seshat/) (see
  [`Cargo.toml:2`](../../Cargo.toml))
- Egg crate: pinned at `egg = "0.11"` ([`Cargo.toml:27`](../../Cargo.toml));
  GitHub: <https://github.com/egraphs-good/egg>
- Sibling wake-path concern: HNSW(VSA) rebuild, addressed in
  [ADR 037](037-llm-vsa-deep-integration.md) and Phase 26 plan
- Phase 38 wake protocol: Anima ticket `019dc599-81c5-7ce2-a4e3-1a84ad6001e6`
  (plan not yet authored as of 2026-04-25)

## Resolutions log

### 2026-04-25

- **P40Q1** (HNSW measurement flag): confirmed. The `--what hnsw|egraph|both` CLI flag for the measurement harness is a Phase 40a deliverable from day one. Tracked in the plan; the ADR's measurement contract is unchanged.
