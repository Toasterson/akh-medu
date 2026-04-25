# ADR 049 — Relocatable Session-Based Akh: Postgres-backed truth with in-process cache

> Date: 2026-04-25
> Status: Proposed
> Phase: 38
> Depends on: Phase 14 (identity, provenance), Phase 17 (evidence intervals),
>             ADR-047 (Seshat / Postgres precedent), ADR-029 (MCP server)
> Enhances: Phase 36 (Delegation), Phase 37 (Seshat's Archive)
> Anima ticket: `019dc599-81c5-7ce2-a4e3-1a84ad6001e6`
> Plan: [`docs/ai/plans/2026-04-25-phase38-relocatable-akh.md`](../plans/2026-04-25-phase38-relocatable-akh.md)

## Context

Akh-medu's daemon (`akhomed`, see `src/bin/akhomed.rs`) is today a
single-host process that owns local files. `WorkspaceManager`
(`src/workspace.rs:206`) materialises each workspace as a directory under
`$XDG_DATA_HOME/akh-medu/workspaces/{name}/`, holding a per-workspace
oxigraph (via `SparqlStore`, `src/graph/sparql.rs:23`), redb
(`src/store/durable.rs:20`), compartments, scratch, and an HNSW item
memory rebuilt fresh each boot
(`Hnsw<'static, u32, DistHamming>` at `src/vsa/item_memory.rs:40`).

Three architectural pressures push against this:

1. **Multi-device deployment.** The next deployment milestone spans the
   Talos cluster (CNPG Postgres-ha already up, used by Seshat's Archive
   in Phase 37), an M2 Mac Mini in the cellar as a companion host, and
   ad-hoc inference backends. Identity must outlive any single host. Today
   it does not — kill the akhomed VM, lose the disk, lose the Akh.

2. **Dataloss risk on crash.** The agent's working state — `WorkingMemory`,
   episode buffer, OODA cycle counter, `Psyche` mutations
   (`src/compartment/psyche.rs:19`) — is persisted only on the 60 s
   `persist_interval` in `src/agent/daemon.rs::DaemonConfig`. A crash
   between persist ticks loses up to 60 s of cognition. For an agent
   doing OODA cycles every 30 s, that is an unbounded number of lost
   decisions across hosts and time.

3. **External read story.** Other tools (visualisers, debuggers, future
   federation, Phase 22 Akhipedia rendering) want to read the KG
   without speaking to a live akhomed. The current architecture has no
   answer.

The opportunity: Phase 37 already established SeaORM + sqlx-postgres in
the project (`crates/seshat/Cargo.toml`), the cluster has CNPG with
pgvector, and adding the AGE extension and an `akhomed` schema is
incremental infrastructure rather than greenfield.

The Anima ticket explicitly frames the goal as **relocatable, not
stateless** — a stateless daemon would break OODA continuity, active
inference (Phase 20), and the affective system (Phase 23), all of which
depend on continuity of in-process state across cycles.

## Decision

Move the *truth* of an Akh's state to Postgres while keeping the *cache*
in-process. The akhomed daemon continues to own oxigraph, petgraph, the
HNSW(VSA) item memory, and the e-graph in memory for hot-path reasoning.
Every mutation streams to a Postgres write-ahead log within seconds; full
snapshots run periodically and on deep-sleep transitions. Akh identity
is a UUID stored in `akhomed.akh`, not a path on a disk; when a host
dies, the same Akh wakes elsewhere from the latest snapshot plus the
WAL tail.

In bullet form:

- **Postgres holds the truth.** `akhomed.facts`, `akhomed.mutation_log`,
  `akhomed.snapshots`, `akhomed.psyche`, `akhomed.provenance`,
  `akhomed.episodes`, plus `workspaces.workspace`,
  `workspaces.membership`, `workspaces.active_microtheories`.
- **The daemon holds the cache.** Oxigraph stays as a write-through
  read cache. Petgraph, HNSW, and the e-graph stay in-process.
- **Continuous WAL.** `Engine::add_triple` (`src/engine.rs:448`) and its
  retract / microtheory siblings append to an in-process buffer that
  flushes to `akhomed.mutation_log` every N seconds or M asserts (target
  ≤5 s). Bounded dataloss on crash.
- **Periodic snapshots.** Every ~5 min and on signal (SIGTERM, relocate,
  memory pressure). Snapshots truncate the WAL behind their watermark.
- **Three-tier sleep.** Running ↔ Idle-in-memory are CPU/perf states
  (WAL keeps flushing in both); Deep is a durability/relocation state
  (caches dropped, only Postgres holds the data).
- **HNSW is in-process only.** Local-disk cache for same-host fast wake;
  rebuild from `akhomed.facts` for cross-host wake. Never persisted to
  Postgres.
- **AGE is an external read view.** Apache AGE projects `akhomed.facts`
  into a Cypher-queryable graph for non-Akh consumers. Akhs continue to
  read SPARQL through the in-process oxigraph cache.
- **Idempotent batched writes.** Asserts are content-addressed and
  deduplicated at the cache boundary; WAL replay is a no-op past first
  application.
- **Multi-Akh sharing.** Two Akhs in the same workspace see each other's
  committed assertions via WAL plus pg_notify. Cross-host coordination
  uses the existing Phase 36 delegation protocol.
- **Akh-to-host binding is advisory.** `akh.home_host` is a hint; any
  akhomed pointed at the same Postgres can wake the Akh after the
  current host has stopped updating its `last_seen` heartbeat.

## Alternatives considered

### Fully stateless akhomed (Postgres-only, no in-process cache)

**Why considered.** Maximally relocatable; zero recovery work; fits a
12-factor-style deployment.

**Why rejected.** Breaks OODA continuity, active inference (Phase 20),
and the affective system (Phase 23). The OODA loop in
`src/agent/ooda.rs` and the daemon ticks in `src/agent/daemon.rs` keep
in-process state across cycles — working memory, drives, the psyche, and
all derived caches. Round-tripping every read to Postgres would burn
latency budgets that today land in microseconds. And the HNSW item
memory in `src/vsa/item_memory.rs` is fundamentally an in-process data
structure (`Hnsw<'static, …>`) — there is no usable form of it that
lives in a database.

### AGE as the primary store (no in-process cache)

**Why considered.** AGE turns Postgres into a property graph; if it's
fast enough, we get persistent + queryable + relocatable in one move.

**Why rejected.** Hot-path KG traversals (spreading activation,
predicate-hierarchy walks, contradiction detection in
`src/graph/contradiction.rs`) need sub-millisecond scans. AGE's Cypher
planner has known query-planner limitations for deep traversals;
cross-network round-trip latency dominates even when the query is fast.
We need a write-through cache regardless. Once we have one, AGE adds
value as a *read view* for outside consumers, not as the primary path.

### Snapshot-only durability (no WAL)

**Why considered.** Simpler. One operation (snapshot), one timer.

**Why rejected.** Bounds dataloss only by snapshot frequency. A 5 min
snapshot interval means up to 5 min of mutations lost on crash —
unacceptable for an agent where each OODA cycle is a decision and each
external interaction is a user-visible event. We want bounded dataloss
in **seconds**. The combination snapshot + WAL is the standard answer
because no single mechanism gives both bounded RPO and bounded recovery
time.

### File-based persistence (redb across sessions)

**Why considered.** redb is already a dependency
(`src/store/durable.rs`). Persisting more state to redb is a small step.

**Why rejected.** Doesn't solve multi-Akh sharing (every Akh has its
own redb), doesn't solve the external-read story (no Cypher / SQL access
without an akhomed running), and doesn't survive disk loss. The
Anima ticket's "M2 in cellar dies" scenario specifically rules this
out: we need the Akh to wake on a different host.

### Postgres-backed snapshots + redb-based WAL

**Why considered.** Hybrid: minimise Postgres write rate by keeping the
WAL local.

**Why rejected.** Re-introduces the multi-host problem the WAL was
supposed to solve: a redb WAL is host-local, so a host crash that
loses the disk loses the WAL. The whole point is to make Postgres the
authoritative ledger.

### Per-Akh dedicated Postgres database

**Why considered.** Strong isolation; one DB per Akh, no shared schema
risk.

**Why rejected.** Operational overhead grows linearly with Akh count.
CNPG is provisioned for one cluster; per-Akh dedicated DBs blow up the
backup story. Multi-Akh sharing in one workspace becomes hard
(cross-database queries). Schemas + a workspace_id column give us the
isolation we need within one cluster.

## Consequences

### Positive

- **Relocatable.** Any akhomed pointed at the same Postgres can wake any
  Akh from its latest snapshot plus WAL tail.
- **Crash safety with bounded dataloss.** Maximum RPO is one
  flush_interval (target ≤5 s), independent of sleep level.
- **Multi-Akh shared workspaces.** Two Akhs in one workspace see each
  other's committed assertions within one flush interval.
- **External Cypher access.** AGE views give visualisers, debuggers,
  and future tooling a direct read path without involving an akhomed.
- **No per-fact provenance loss across relocation.** The provenance
  ledger lives in `akhomed.provenance`; it survives whatever the host
  does.
- **Path to Phase 39** (network inference backends): once "where state
  lives" is decoupled from "where compute happens", routing inference
  to the 7900 XTX desktop is a separate, smaller change.
- **Snapshot-driven before-upgrade safety.** Operators can write a
  snapshot before any risky upgrade and roll back to it.

### Negative

- **Write amplification.** Every assert now writes to oxigraph,
  petgraph, HNSW (in-process) and `akhomed.facts` + `akhomed.mutation_log`
  (Postgres, asynchronously). Throughput is bounded by WAL flush time
  for sustained bursts.
- **Snapshot/WAL boundary complexity.** Truncating the WAL behind a
  snapshot watermark is the kind of edge-case-rich code that requires
  careful tests for crash-during-truncate, snapshot-during-flush, etc.
  The 38g harness is not optional.
- **Cache drift risk.** Local oxigraph and Postgres `akhomed.facts` can
  drift if a write hits one but not the other. Mitigation: write-through
  with WAL flush is the only write path; a periodic
  `akhomed verify` job samples and reconciles.
- **AGE adoption risk.** AGE is a CNPG extension with a lightly-loaded
  contributor base. If a future CNPG release drops AGE support, we
  lose external Cypher; we don't lose Akh state, but the AGE-dependent
  consumers need a fallback (direct SPARQL via akhomed HTTP).
- **More moving parts on the operator side.** A Postgres outage now
  affects akhomed (it falls back to in-memory-only mode and refuses
  new asserts when the buffer fills). Today the daemon is unaffected
  by Postgres state.
- **Identity churn during migration.** The first time a workspace is
  promoted to Postgres, every Akh in it gets a UUID in `akhomed.akh`.
  Existing local-only deployments need a one-time migration step
  (handled by 38g playbook).
- **Schema lock-in.** `akhomed.facts` is now part of the public-ish
  contract for AGE consumers; schema changes need migration plus
  projection refresh. Mitigated by sea-orm-migration and CI coverage.

### Risks (from the plan)

- Postgres unreachable for hours → buffered, then degraded mode.
- WAL flush stalls → backpressure as audit, not as hang.
- Cross-host wake races → `last_seen` heartbeat plus Postgres advisory
  lock keyed on `akh_id`.
- HNSW rebuild bottlenecks cold wake → local-disk cache mitigates
  same-host; cross-host budget measured before deciding on Phase 40.

See the plan's risk table for the full list.

## Schema sketch

Two schemas in the existing CNPG cluster, alongside Phase 37's `seshat.*`:

- `akhomed.akh` — Akh identities (UUID primary key, advisory `home_host`
  hint, `last_seen` heartbeat for split-brain detection).
- `akhomed.facts` — every triple, indexed by workspace, subject,
  predicate, object; `(workspace_id, subject, predicate, object)` is
  unique so concurrent asserts dedupe in Postgres.
- `akhomed.mutation_log` — the streaming WAL; `BIGSERIAL` seq, payload
  as JSONB, kind taxonomy (`assert`, `retract`, `microtheory_load`,
  `microtheory_unload`, `compartment_create`, `compartment_destroy`).
- `akhomed.snapshots` — `working_mem` and `episodes` as bincode bytea,
  `state_blob` for catch-all engine state, `psyche` as JSONB; each row
  carries the `watermark` it covers.
- `akhomed.psyche` — current row per Akh (the snapshots table holds
  history; this one is the live mirror for AGE / external read).
- `akhomed.provenance` — mirrors `src/provenance.rs` with the same
  `DerivationKind` taxonomy serialized as JSONB.
- `akhomed.episodes` — episodic memory persisted across sleep cycles.
- `workspaces.workspace`, `workspaces.membership`,
  `workspaces.active_microtheories` — workspace metadata, member Akhs,
  active microtheory set.

AGE graph `akhomed_kg` is a periodic projection of `akhomed.facts`,
created via `SELECT create_graph('akhomed_kg')` in a migration step that
is a no-op when the AGE extension is unavailable. Full DDL sketch lives
in the plan doc.

## How it interacts with existing subsystems

- **`Engine::add_triple` (`src/engine.rs:448`).** Signature unchanged.
  After the in-process insert succeeds, the same call appends a
  mutation entry. Phase-9 contradiction / constraint logic is untouched.
- **`SparqlStore` (`src/graph/sparql.rs:23`).** Stays as a write-through
  cache. Reads come from oxigraph; writes flow through the WAL.
- **`ItemMemory` (`src/vsa/item_memory.rs:35`).** Gains a
  `dump_to_disk` / `load_or_rebuild` pair. The HNSW itself is never
  serialised — only the bipolar vectors and a fingerprint.
- **`SleepCycle` (`src/agent/sleep.rs:92`).** The existing
  `ConsolidationPhase` enum (Wake / Consolidation / Reorganization /
  DreamState) is the cognitive sub-phase during a sleep cycle. The new
  `SleepLevel` (Running / IdleInMemory / Deep) sits orthogonally on top
  for durability state.
- **`Psyche` (`src/compartment/psyche.rs:19`).** Already
  `Serialize`/`Deserialize`; lands in JSONB without code changes.
- **`AgentDaemon` (`src/agent/daemon.rs`).** `DaemonConfig` gains
  `wal_flush_interval`, `snapshot_interval`,
  `idle_threshold_cycles`, `memory_pressure_threshold`. Three new
  background tasks: WAL flusher, snapshot writer, AGE projection
  refresher.
- **`WorkspaceManager` (`src/workspace.rs:206`).** Postgres-backed,
  with TOML files retained as a bootstrap fixture.
- **`ServerState` (`src/bin/akhomed.rs`).** Carries the SeaORM
  connection pool; daemon startup wires per-workspace WAL/snapshot
  tasks.

## Implementation

Seven sub-phases (38a–38g), ~4,200 lines. The migration is staged so
that the daemon remains runnable at every step:

1. 38a + 38b: layer the WAL on top with Postgres optional. No boot-path
   change. A Postgres outage degrades cleanly to today's behaviour.
2. 38c: add snapshots; same-host crash recovery works.
3. 38d: AGE projections come online; akhomed-internal behaviour
   unchanged.
4. 38e: change the boot path to Postgres-first. Behind a config flag
   for one release cycle.
5. 38f: workspaces themselves move to Postgres rows; multi-Akh sharing
   ships.
6. 38g: codify the cross-host playbook with a CI-enforced harness.

See `docs/ai/plans/2026-04-25-phase38-relocatable-akh.md` for the
sub-phase breakdown and acceptance criteria.

## References

- Plan: `docs/ai/plans/2026-04-25-phase38-relocatable-akh.md`
- Anima ticket: `019dc599-81c5-7ce2-a4e3-1a84ad6001e6`
- ADR 047 — Seshat's Archive (Postgres + SeaORM precedent)
- ADR 037 — LLM-VSA Deep Integration (the multi-host inference story
  Phase 38 enables)
- Phase 36 plan — Delegation + External Scheduling (cross-homeserver
  coordination)
- Phase 39 (queued) — Network inference backends, unblocked by 38
- Phase 40 (queued) — E-graph durability, if rebuild proves too slow
- Apache AGE: https://age.apache.org/
- CloudNativePG (CNPG): https://cloudnative-pg.io/
- redb: `src/store/durable.rs`
- oxigraph: `src/graph/sparql.rs`
- hnsw_rs: `src/vsa/item_memory.rs`
