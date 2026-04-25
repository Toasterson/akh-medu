# Phase 40 — E-graph Persistence (Measure + Report + Decide)

> Date: 2026-04-25
> Status: PROPOSED — measurement pending
> Phase: 40
> Depends on: Phase 38 (Relocatable Session-Based Akh — provides the snapshot/replay
>             framework this hooks into, or doesn't, depending on the outcome)
> Related: Phase 9 (microtheory rules), Phase 15 (causal rewrites),
>          Phase 19 (epistemic rewrites), Phase 26 (HNSW(VSA) is a sibling
>          wake-path cost worth measuring alongside)
> ADR: [051-egraph-persistence](../decisions/051-egraph-persistence.md)

## Executive Summary

Phase 38 introduces a three-tier sleep model (light → deep → hibernated) and a
postgres-backed truth store from which the akh wakes onto a fresh workstation.
On wake we have to rebuild any **derived** runtime structures: the HNSW(VSA)
index, the e-graph saturation product, the spreading-activation caches, and so
on. Of these, the e-graph is the most expensive to rebuild **in principle** and
the one we know the least about **in practice**.

The e-graph (egg) saturates a knowledge expression against the active rule set
and produces an equivalence-class structure. Today this happens lazily on every
`simplify_expression()` call and inside the `Reason` pipeline stage — the
saturation product is never persisted; it is recomputed each time. If wake-time
re-saturation across a representative KG is fast (sub-second), persistence is a
distraction: complexity cost for no benefit, with the additional risk of stale
snapshots when rules change. If it is slow (tens of seconds), persistence is a
hard requirement for the wake-time SLA Phase 38 wants to meet.

We do not currently know which regime we are in. Therefore Phase 40 is
explicitly a **measure-first phase**: we build a benchmark harness, run it on
synthetic and real workloads, write up the results, then amend ADR 051 with
the chosen path. The default heading is **option (0): persist nothing** — the
e-graph stays a derived structure and is recomputed on wake. We only deviate
from that if measurement says we must.

This plan covers four sub-phases (40a–40d) with the implementation work in 40d
explicitly contingent on the outcome of 40b.

---

## Why This Isn't Ready to Commit to an Implementation

The e-graph in akh-medu is a **derived structure**. It has no canonical
identity; the same set of facts and rules will, modulo non-determinism in egg's
union-find ordering, yield equivalent equivalence classes. Concretely:

- The `egg::Runner` type is rebuilt fresh on every call to
  `Engine::simplify_expression()` ([`src/engine.rs:1174-1184`](../../src/engine.rs))
  and inside the `Reason` pipeline stage
  ([`src/pipeline/mod.rs:430-447`](../../src/pipeline/mod.rs)).
- The rewrite rules themselves live in compiled Rust code in
  [`src/reason/mod.rs`](../../src/reason/mod.rs) — `builtin_rules()`,
  `pim_rules()`, `calendar_rules()`, `causal_rules()`, `epistemic_rules()` —
  plus runtime-loaded skill and microtheory rules.
- `RuleEngine` (the forward-chaining cousin in
  [`src/autonomous/rule_engine.rs`](../../src/autonomous/rule_engine.rs)) is
  similarly stateless; it runs over `KnowledgeGraph` triples.
- The `Engine` does not currently expose any method that returns a long-lived
  e-graph handle. There is no place a snapshot would naturally come from.

Persisting a derived structure is always a tradeoff:

| Pro | Con |
|---|---|
| Avoid recomputation cost on wake | Storage cost, both bytes and complexity |
| | Invalidation logic (when does the cache go stale?) |
| | Library version coupling (egg 0.11 internals are not a stable format) |
| | Consistency surface — caller sees a snapshot not derivable from current facts |

The only reason to pay that complexity is if the recomputation is actually slow.
We have not measured it. Building a persistence layer first and measuring later
is exactly the wrong order: we'd be solving an imagined problem and accepting
real costs.

---

## Three Candidate Persistence Shapes

The phase considers exactly three options. We pick **one** based on the data,
not a blend.

### Option (0) — Persist nothing; re-saturate on wake

**What's stored:** Nothing. The e-graph is rebuilt from facts + rules every
time it is needed, which is what happens today.

**Pros:**
- Simplest. Zero new code, zero schema changes, zero invalidation logic.
- No staleness risk — the e-graph is always coherent with current facts.
- Egg version bumps are free. No on-disk format to migrate.
- Aligns with how the e-graph is used today (one-shot saturation per query).

**Cons:**
- Pays full re-saturation cost on every wake (and every query that needs an
  e-graph today, but that cost already exists).
- If saturation is slow on representative KGs, wake-time SLA suffers.

**Choose if:** Median saturation time for representative theories is **< 5 s**.
Wake latency budget can absorb that.

### Option (1) — Persist canonicalized e-class roots only

**What's stored:** A canonical, version-independent representation of the
equivalence classes the e-graph would converge to: a set of e-class roots and
the rewrite bindings between them, serialized in a format we own.

**Pros:**
- Smaller than full state — only the union-find result, not the full e-node
  graph.
- Replay is cheaper than cold saturation: the rewrites that produced each class
  can be skipped because the class identities are pre-known. New facts merge
  into existing classes via a single targeted re-run.
- Egg internals don't appear in the snapshot. We can survive an egg version
  bump without a format migration.

**Cons:**
- Still pays some wake cost — incremental rebuild, not zero rebuild.
- Canonicalization is non-trivial: e-class IDs in egg are union-find handles
  with no semantic identity; we have to choose a representative term per class
  and hash it stably.
- New invalidation logic, even if narrower than Option (2).

**Choose if:** Median saturation time is **5–30 s**. Worth caching the answer,
but not the full intermediate state.

### Option (2) — Persist the full e-graph state

**What's stored:** A full serialization of the egg `EGraph<AkhLang, ()>` —
e-classes, e-nodes, parent pointers, the union-find, scheduler state, the
analysis (currently `()`).

**Pros:**
- Fastest possible wake — the e-graph is ready to answer queries immediately.
- Caller-facing semantics on wake match steady-state semantics exactly: any
  query that would have hit the e-graph before sleep hits the same e-graph
  after wake.

**Cons:**
- Largest snapshot. E-graphs blow up super-linearly during saturation; a 10⁵
  fact theory may produce a multi-GB e-graph.
- Hardest to invalidate. *Any* fact retraction or rule change requires
  discarding the snapshot in full (the union-find is global state).
- Egg version coupling. egg 0.11 is not a stable serialization target — fields
  on `EGraph`, `EClass`, `Analysis` may move between versions and there is no
  `serde` impl in the public API. We would have to write our own canonical
  serializer, defeating part of the "full state" advantage.
- Highest staleness risk. A miscalibrated invalidation rule means the akh
  reasons against an out-of-date saturation product without realising it.

**Choose if:** Median saturation time is **> 30 s**. The complexity is paid for.

---

## Decision Criteria

We commit, before measurement, to the following thresholds. They will be
written into ADR 051 as the binding rule once data is in.

| Median time-to-saturation, representative theory | Choice |
|---|---|
| < 5 s | **Option (0)** — persist nothing |
| 5 s – 30 s | **Option (1)** — class roots only |
| > 30 s | **Option (2)** — full state |

"Representative theory" is defined in §Measurement Plan below. "Median" is
chosen over mean because we expect a long tail driven by pathological
combinations of rules and facts; the goal is to size the wake-path budget for
the typical case, not the worst.

If results are bimodal — e.g., most theories saturate in 1 s, but ~10% take
60 s — we will still pick a single option, but the ADR will note which
workloads we are knowingly leaving on the slow path.

---

## Measurement Plan

The harness must answer one question reliably: *what is the cold-start
saturation cost as a function of theory size and rule-set size, for both
synthetic and real workloads?*

### 40a — Benchmark harness (`crates/akh-egraph-bench/`)

A new workspace crate `crates/akh-egraph-bench/` containing:

1. A **synthetic theory generator** parameterized by:
   - `n_facts`: number of triple facts to assert
   - `rule_set`: small (≤10) / medium (≤50) / large (≤200) rule pack
   - `branching_factor`: how often a rule fires per fact (controls e-graph blowup)
   - `seed`: RNG seed for reproducibility
2. A **harness binary** (`akh-egraph-bench`) using `criterion` (already a dev
   dep — see [`Cargo.toml:154`](../../Cargo.toml)) that:
   - Builds the synthetic KG via `Engine::add_triple()`
   - Runs `simplify_expression()` and the `Reason` pipeline stage on a
     representative query expression
   - Records: wall-clock time-to-saturation, e-class count at saturation,
     resident-set memory peak (via `jemalloc-ctl` or platform-specific RSS),
     `egg::StopReason` (Saturated / IterationLimit / NodeLimit / TimeLimit)
3. A **real-workload loader** that ingests an anonymized KG snapshot from a
   live akh and re-runs the same measurements. Anonymization replaces label
   strings with hashes; only structural edges and rule patterns matter for
   saturation time.
4. CLI flags to dump per-rule-set stats as JSON for the report.

Crate placement: matches the precedent set by `crates/seshat/` (see
[`Cargo.toml:2`](../../Cargo.toml)). Keeps the bench out of the main
binary's compile graph.

### 40b — Measurement run

Run the harness across the cross product:

| Dimension | Values |
|---|---|
| `n_facts` | 10², 10³, 10⁴, 10⁵ |
| `rule_set` | small / medium / large |
| `branching_factor` | 1, 3, 10 |
| `repetitions` | 5 (criterion's default warmup + sample) |

Plus, for each akh we have a snapshot of:
- The full live KG (one anonymized export per akh)
- The actual rule set that akh has loaded (from compiled `reason::*` packs +
  any skill-loaded rules)

Hardware: M2 Pro 32 GB RAM as the canonical workstation tier; a 4-core x86
container as the deployment tier. We measure both because Phase 38 supports
relocation between them.

Output: a markdown report committed alongside this plan as
`docs/ai/plans/2026-04-25-phase40-measurement-report.md` (created by 40b),
with:
- Time-to-saturation distribution per cell, plotted as box-and-whisker
- E-class count growth as a function of `n_facts × branching_factor`
- Memory peak per cell
- Saturation success rate (did egg actually converge, or hit a limit?)
- A one-line verdict: which option the data points to

### 40c — ADR amendment

Once 40b is in, ADR 051 is amended from `Status: PROPOSED — pending
measurement` to `Status: ACCEPTED` with the chosen option recorded explicitly
in the Decision section. This is a one-paragraph edit; the whole point of the
ADR is to be amenable to that edit without rewriting context or alternatives.

### 40d — Implementation (conditional)

Only happens if 40c selects (1) or (2). If (0) wins, 40d is closed as
"no-op — the chosen path is already what we do today".

---

## Implementation Outline (IF Persistence Is Chosen)

This is **sketch-level only**. The real work doesn't happen until 40b is in
and 40c picks an option. Listed here so readers know roughly what 40d looks
like in either branch.

### Schema (option 1 or 2)

A single new postgres table owned by the akhomed truth store (Phase 38):

```sql
CREATE TABLE akhomed.egraph_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    akh_id          UUID NOT NULL REFERENCES akhomed.akhs(id),
    snapshot_kind   TEXT NOT NULL,          -- 'class_roots' | 'full_state'
    rule_set_hash   BYTEA NOT NULL,         -- BLAKE3 of compiled rule pack
    fact_set_hash   BYTEA NOT NULL,         -- BLAKE3 of (s, p, o) tuples
    egg_version     TEXT NOT NULL,          -- e.g., '0.11.2'
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    payload         BYTEA NOT NULL,         -- our canonical format
    payload_size    BIGINT NOT NULL,
    UNIQUE (akh_id, rule_set_hash, fact_set_hash, egg_version)
);
```

The unique constraint enforces "one snapshot per (rules, facts, egg version)
tuple" — implicit invalidation by hash rather than by trigger.

### Serialization format

We **own the format**. egg's internal types (`EGraph`, `EClass`, `EClassData`,
the union-find `UnionFind`) do not implement `serde::Serialize` and are not
part of the egg public API contract. Coupling our on-disk format to egg's
private types means an egg minor-version bump can corrupt the cache.

For Option (1), the payload is a CBOR-encoded list of:

```rust
struct ClassRecord {
    canonical_term: String,       // smallest expression in the class, .to_string()-ed
    member_terms: Vec<String>,    // other expressions known equal
    confidence: f32,
}
```

For Option (2), we serialize a normalized form: every e-node is a tuple of
`(op_name, Vec<class_canonical_term>)`, classes are referenced by their
canonical term. Replay rebuilds the egg `EGraph` by `add_expr`-ing each
canonical term then `union`-ing on the recorded equivalences. This gives us
the "full state" benefit without coupling to egg's internals, at the cost of
a non-trivial replay step (still expected cheaper than cold saturation,
because the rules don't have to fire).

### Invalidation triggers

Discard the cached snapshot on:

1. **Microtheory schema change** — a new microtheory loaded, or
   `ctx:specializes` edges modified. Detected by hashing the active set of
   microtheory roots.
2. **Rule-set change** — a skill loaded/unloaded changes the active rule pack.
   Detected by hashing the concatenated `egg::Rewrite` source strings.
3. **Fact mutation** — any `add_triple` / `remove_triple` whose subject or
   predicate appears in the cached e-graph. Detected by intersecting the
   cache's recorded `fact_set_hash` with the post-mutation fact set.
4. **Egg version bump** — the `egg_version` column doesn't match the running
   binary. Hard discard, no migration.

For Option (1), invalidation can be partial: only classes that touch a mutated
fact are re-saturated; the rest stay. For Option (2), invalidation is total —
the union-find is global state.

### Replay protocol on wake

Phase 38 wake sequence becomes:

1. Reattach to truth store (postgres).
2. Rehydrate KG triples into `KnowledgeGraph`.
3. Rehydrate VSA item memory + HNSW (already a sibling concern; see open
   questions below).
4. **(NEW)** Compute `(rule_set_hash, fact_set_hash, egg_version)` and look
   up `egraph_snapshots`. On hit, deserialize and replay; on miss, schedule a
   lazy cold saturation on first use.
5. Resume OODA.

Crucially, the wake path **does not block on e-graph rebuild** even with
persistence. The replay (or cold saturation) runs lazily on first
`simplify_expression()` / `Reason`-stage call, with a wake-time progress
indicator if it's actually slow. Persistence is a latency improvement, not a
correctness requirement.

---

## Sub-phase Breakdown

| Sub-phase | Description | Output | Estimated effort |
|---|---|---|---|
| 40a | Benchmark harness | `crates/akh-egraph-bench/` + synthetic generator + criterion harness + real-workload loader | ~700 lines |
| 40b | Measurement run | Markdown report with distributions per cell, real-akh data, hardware-tier comparison | data + ~200 lines of report |
| 40c | ADR amendment | ADR 051 status: ACCEPTED, with the chosen option | ~50 lines diff to ADR |
| 40d | Implementation (conditional) | If (0): no-op. If (1): canonical-roots persistence + invalidation. If (2): full-state persistence + invalidation. | (0) ~0, (1) ~800, (2) ~1500 lines |

Total scope, worst case (Option 2 wins): ~2,400 lines. Best case (Option 0
wins): ~900 lines, all in the bench harness, which is reusable for HNSW(VSA)
and other wake-path measurements.

---

## Dependency Graph

```
40a (harness) ──┐
                ├──→ 40b (measurement) ──→ 40c (ADR amendment) ──→ 40d (implementation, conditional)
[real-akh KG    ┘
 anonymized snapshots
 from Phase 38 hibernated truth store]
```

40a can begin immediately; the synthetic generator does not need real data.
40b cannot complete without at least one anonymized live KG snapshot, which
in turn requires Phase 38 to be at least partially landed (the postgres truth
store specifically). 40c is a documentation change. 40d is gated on 40c.

---

## Open Questions

These are explicitly left for the plan to resolve during 40a/40b. They are
not blockers for starting the work.

### Q1: Should we measure HNSW(VSA) rebuild time alongside e-graph re-saturation?

The wake-path budget is shared. A 4-second HNSW rebuild plus a 4-second
e-graph re-saturation is an 8-second wake; either alone is acceptable, both
together are not. If we are going to build a benchmark harness for one, we
should sweep the other in the same run — they share the same workloads (the
KG facts and the VSA item memory are co-resident). The harness in 40a should
have a `--what hnsw|egraph|both` flag from day one.

The HNSW path is reasonably well-understood — `hnsw_rs` index build is O(n
log n) with measured constants in the existing `vsa_ops` bench
([`benches/vsa_ops.rs`](../../benches/vsa_ops.rs)) — but rebuild from a
serialized item-memory snapshot has not been measured at the workstation
scale of items we expect. **Tentative answer: yes, fold it in.** The
incremental cost of running both benches together is small.

### Q2: Are there workloads that need partial e-graph state mid-reasoning?

Active inference's inner loop (Phase 20), the MCTS planning tree (Phase 16),
and counterfactual reasoning (Phase 15c) all do iterative exploration over
modified KGs. If any of them holds an e-graph open across many internal
steps and mutates the underlying facts during exploration, then persistence
of *incremental* state — not just steady-state snapshots — becomes the real
question, and Options (0)/(1)/(2) above are the wrong axis.

We do not currently believe this is the case — e-graph saturation in akh-medu
is one-shot per query — but 40a should include at least one synthetic
workload that mimics an MCTS-style "fork the e-graph 100 times with small
perturbations" pattern, so we can rule it out empirically.

### Q3: What's the per-microtheory granularity?

Microtheories (Phase 9a, [`src/compartment/microtheory.rs`](../../src/compartment/microtheory.rs))
partition the KG. A naive global e-graph mixes all of them; a more
sophisticated approach saturates per microtheory. If per-microtheory
saturation is what we're caching, the snapshot keying changes (one row per
microtheory, not one row per akh), and invalidation can be much narrower
(mutate microtheory M, only invalidate M's snapshot). 40a should report
per-microtheory saturation cost to inform this — even if we end up with a
global snapshot, the data is useful.

### Q4: Cross-architecture portability of snapshots

Phase 38 explicitly supports relocation between an M2 workstation and an x86
deployment node. If a snapshot is bytewise architecture-dependent (e.g.,
endian-sensitive bincode), wake-after-relocate has to discard it. CBOR is
endian-stable; bincode is not. The serialization format choice in 40d
should default to CBOR for that reason, even if it is slightly larger than
bincode.

### Q5: Should this become a generic "derived structure" cache?

If the answer is "persist nothing" for e-graphs and "persist everything" for
HNSW, we have two ad hoc systems. If both end up in the middle, there's an
argument for a single `DerivedStructureCache` abstraction with `(kind, hash)
→ blob` semantics. This is out of scope for Phase 40 but worth flagging if
the Q1 measurement comes out symmetric.

---

## Risks

The actual risk of this phase is **not measurement** — measurement is cheap
and contained in a separate crate that doesn't ship to users. The risk is
**making the wrong decision under-measured**:

- **Synthetic benchmarks don't cover the real distribution.** Mitigation:
  the real-workload arm of 40b is non-optional. If we don't have at least
  three anonymized live-akh KG snapshots, the report is incomplete and 40c
  is blocked.
- **Measurement on idle hardware overstates wake-time performance.**
  Mitigation: 40b runs on both an unloaded M2 and a co-tenanted x86 with
  background load (simulating a deployed akh sharing a node). The variance
  goes in the report.
- **The "median" criterion hides a tail that matters.** If 95th-percentile
  saturation is 60 s but median is 2 s, the user-visible wake experience is
  the 95th. Mitigation: the report includes p50, p90, p99, and we revisit
  the threshold table if the distribution is heavy-tailed.
- **Egg version churn.** egg has historically had API churn between minor
  versions (we are pinned at `egg = "0.11"` —
  [`Cargo.toml:27`](../../Cargo.toml)). Any persistence solution must
  decouple from internal types. Mitigation: the implementation outline above
  explicitly serializes a canonical form, not egg's structures. This is
  enforced by code review at 40d.
- **Premature lock-in via the harness itself.** The benchmark crate is
  permanent infrastructure. A poorly-chosen harness API could lead to all
  future wake-path measurements being awkward. Mitigation: 40a's
  generator/recorder split is explicit, and the recorder writes JSON, not
  Rust-specific binary, so the report can be regenerated by other tools.

The smallest-blast-radius risk is what shapes this plan: we accept a delay
to the Phase 38 wake-time SLA target rather than commit to an implementation
that turns out to be the wrong shape.

---

## Verification

This plan is verified when:

1. `cargo build -p akh-egraph-bench` succeeds (40a complete).
2. `cargo run -p akh-egraph-bench -- --what egraph --workload synthetic
   --report report.md` produces a report covering the full cross product
   in §Measurement Plan (40b synthetic complete).
3. The same command with `--workload real --akh-snapshot <path>` produces
   per-akh stats for at least three anonymized live KGs (40b real complete).
4. ADR 051 is amended with `Status: ACCEPTED` and a Decision section that
   names exactly one of options (0)/(1)/(2), with the median-time number
   from the report cited (40c complete).
5. If 40c selects (1) or (2): `akhomed.egraph_snapshots` table exists,
   wake-path tests show that a snapshot from a recorded sleep is correctly
   replayed on wake, and invalidation tests show that all four invalidation
   triggers cause a discard (40d complete).
6. If 40c selects (0): 40d is closed as no-op; no schema change ships.

---

## References

- ADR 051: [E-graph persistence: measure before deciding](../decisions/051-egraph-persistence.md)
- Source of e-graph usage: [`src/engine.rs:1174-1184`](../../src/engine.rs),
  [`src/pipeline/mod.rs:430-447`](../../src/pipeline/mod.rs)
- Rule packs: [`src/reason/mod.rs`](../../src/reason/mod.rs) (builtin, pim,
  calendar, causal, epistemic), [`src/autonomous/rules.rs`](../../src/autonomous/rules.rs)
- Existing bench precedent: [`benches/phase26.rs`](../../benches/phase26.rs),
  [`benches/vsa_ops.rs`](../../benches/vsa_ops.rs)
- Workspace crate precedent: [`crates/seshat/`](../../crates/seshat/)
- Phase 38 (wake protocol that this hooks into): see Anima ticket
  `019dc599-81c5-7ce2-a4e3-1a84ad6001e6`
- Sibling wake-path concern: HNSW(VSA) rebuild — Phase 26 plan
  [`docs/ai/plans/2026-03-26-phase26-llm-vsa-deep-integration.md`](2026-03-26-phase26-llm-vsa-deep-integration.md)
