# Phase 38 — Relocatable Session-Based Akh: Workspaces + 3-Tier Sleep + Postgres Truth

> Date: 2026-04-25
> Status: Planned
> Phase: 38
> Depends on: Phase 14 (identity, provenance), Phase 17 (evidence intervals), Postgres-ha CNPG cluster (shipped)
> Enhances: Phase 36 (Delegation), Phase 37 (Seshat's Archive)
> Blocks (cleanly unblocks): Phase 39 (network inference backends), Phase 40 (e-graph durability)
> ADR: [049-relocatable-akh](../decisions/049-relocatable-akh.md)
> Anima ticket: `019dc599-81c5-7ce2-a4e3-1a84ad6001e6`

## Executive summary

Phase 38 restructures the akh-medu daemon (`akhomed`) so that the **truth of an
Akh's state lives in Postgres** and the **process is relocatable**. The daemon
keeps an in-process cache of oxigraph, petgraph, the HNSW(VSA) item memory, and
the e-graph for hot-path reasoning. Mutations stream to Postgres as a continuous
write-ahead log; full snapshots land periodically and on deep-sleep transitions.
An Akh's identity is bound to its Postgres data, not to any particular host —
when a host dies, the Akh wakes elsewhere with the same identity and resumes
from the last committed mutation.

This is **not** a stateless rewrite. A stateless akhomed would break the OODA
loop, active inference (Phase 20), and the affective system (Phase 23) — all of
which depend on continuity of working memory, episodic memory, and psyche state
across cycles. The right framing is *relocatable*, not stateless.

The phase ships in seven sub-phases (38a–38g), ordered by dependency, each
shippable independently. The first three add durability without changing the
boot path; the next three change the boot path; the last formalises the
multi-Akh sharing semantics. At every intermediate sub-phase the daemon still
runs end-to-end.

## Motivation

### The multi-device deployment story

Akh-medu was built as a single-host daemon that owns local files. The
`WorkspaceManager` in `src/workspace.rs` materialises each workspace as a
directory tree under `$XDG_DATA_HOME/akh-medu/workspaces/{name}/`, with
`kg/` holding oxigraph and redb, `compartments/` holding compartment data,
and `scratch/` for agent scratch. Identity is implicit in *whichever akhomed
opened that directory last*. This is a fine model for a workstation, but the
deployment plan now spans:

- **The Talos cluster** running CNPG Postgres-ha (already deployed; the
  Seshat Archive in Phase 37 already reads/writes pgvector against it).
- **An always-on primary akhomed** in the cluster that Claude talks to.
- **An M2 Mac Mini in the cellar** as a companion host for additional Akhs.
- **Optional Raspberry Pis / 7900 XTX desktop** as inference backends or
  short-lived hosts.

In this topology, "the Akh's data" must outlive any one host. Today it does
not: kill the akhomed VM, lose the disk, lose the Akh.

### The dataloss-on-crash concern

The current daemon writes to oxigraph (`src/graph/sparql.rs`) and redb
(`src/store/durable.rs`) on local disk. Both are durable on commit, but the
agent's *working state* — working memory, episode buffer, OODA cycle counter,
psyche-state mutations — is persisted only on the 60-second `persist_interval`
in `src/agent/daemon.rs::DaemonConfig`. A crash between persist ticks loses up
to 60 seconds of cognition. For an autonomous agent doing OODA cycles every
30 seconds, that is an unbounded number of lost decisions.

We want **bounded dataloss** on crash, measured in seconds, irrespective of
sleep level. That requires a continuous WAL.

### The "M2 in cellar" plan

The next deployment milestone is to host most Akhs on an M2 Mac Mini in the
cellar while the cluster runs only the Claude-facing primary. For this to
work, an Akh on the M2 must:

1. Be able to crash-recover from cluster Postgres without rebuilding from
   scratch. (The M2 doesn't get its own Postgres.)
2. Be reachable from Claude in the cluster — which means the cluster akhomed
   needs to be able to wake an Akh on behalf of the M2 (or vice versa) by
   activating its workspace.
3. Survive the M2 going down (cellar power, broken fan, hardware refresh)
   without ceremony — wake the Akh in the cluster, run there until the M2 is
   back.

None of this is possible with today's local-disk-only persistence. Phase 38
is the prerequisite.

### Why now and not earlier

Phase 37 (Seshat's Archive) ships the first first-party use of CNPG inside
akh-medu. SeaORM, sqlx-postgres, and the migration runner are now established
dependencies (`crates/seshat/Cargo.toml`). The cluster has Postgres with
pgvector; adding the AGE extension and an `akhomed` schema is incremental
infrastructure rather than greenfield.

## Scope and non-goals

### In scope

- A continuous WAL of every assert / retract / microtheory mutation, flushed
  to Postgres within seconds.
- Periodic full snapshots that bound replay distance.
- A three-tier sleep state machine (Running / Idle-in-memory / Deep) with
  continuous WAL across all three.
- AGE-backed read views over `akhomed.facts` for *external* (non-Akh)
  consumers — visualizers, debugging tools, future federated queries.
- Workspace abstraction extended: workspaces become first-class Postgres
  rows, and an Akh can host multiple at once.
- Multi-Akh sharing semantics: two Akhs in the same workspace see each
  other's assertions live; Akhs across hosts coordinate via the Phase 36
  delegation protocol.
- Akh identity as primary key, not host-bound.
- Migration playbook for the dead-host case.

### Explicitly out of scope

- **E-graph durability.** The e-graph (egg) is rebuilt from facts on wake.
  If that becomes the wake-latency bottleneck, see Phase 40.
- **Replacing Seshat.** Seshat owns the corpus schema (`seshat.*`); Phase 38
  adds `akhomed.*` and `workspaces.*` alongside it. They share the cluster
  but not the schema.
- **Live multi-master concurrency.** Two Akhs in the same workspace see each
  other's *committed* assertions, not in-flight cache state. Hot conflicts
  resolve via Phase 17 evidence intervals on the next read, not via
  distributed locking.
- **HNSW persistence in Postgres.** The HNSW index is in-process only. It
  rebuilds on cold wake from `akhomed.facts`. See "HNSW wake protocol" below.
- **Replacing oxigraph.** Oxigraph stays in-process as a write-through cache;
  AGE is a *read view* for outside-the-Akh consumers, not the primary write
  path.

## Architecture

### Backing-store topology

```
┌─────────────────────────────────────────────────────────────────────┐
│                Postgres-ha (CNPG, shared, durable)                   │
│                                                                       │
│  ┌────────────────┐  ┌──────────────────┐  ┌──────────────────────┐ │
│  │ seshat.*       │  │ akhomed.*        │  │ workspaces.*         │ │
│  │ corpus +       │  │ triples (AGE),   │  │ workspace metadata,  │ │
│  │ embeddings     │  │ provenance,      │  │ membership, shared   │ │
│  │ (Phase 37)     │  │ episodes,        │  │ microtheory refs     │ │
│  │                │  │ microtheories,   │  │                      │ │
│  │                │  │ psyche jsonb,    │  │                      │ │
│  │                │  │ snapshots,       │  │                      │ │
│  │                │  │ mutation_log     │  │                      │ │
│  └────────────────┘  └──────────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │  WAL flush every N seconds or M asserts
                              │  Snapshot every M minutes, on deep-sleep,
                              │  on relocation request.
                              │
┌─────────────────────────────┼───────────────────────────────────────┐
│              Akh daemon (relocatable; M2 / Talos / RPi)             │
│                              │                                       │
│  In-memory:                                                          │
│    oxigraph cache (write-through)                                    │
│    petgraph index                                                    │
│    HNSW(VSA) item memory                                             │
│    e-graph (rebuilt on wake)                                         │
│    working memory + episode buffer                                   │
│                                                                       │
│  Local disk (per-host, ephemeral):                                   │
│    HNSW(VSA) cache for same-host fast wake                          │
│    oxigraph LSM files (write-through cache; can be discarded)       │
│                                                                       │
│  Wire to Postgres:                                                   │
│    akhomed-wal: per-mutation streaming flusher                       │
│    akhomed-snap: periodic + on-signal full snapshot writer           │
│    akhomed-replay: snapshot + WAL-tail loader on wake                │
└─────────────────────────────────────────────────────────────────────┘
```

The clean rule: **Postgres holds the truth, the daemon holds the cache.**
Read-heavy workloads stay fast (in-process oxigraph is what answers SPARQL).
Crash safety comes from the WAL, not from the cache.

### Workspace abstraction

A **Workspace** is the unit of cognitive scope:

```
Workspace = (
  active microtheory set,
  member Akh identities,
  shared goal frame,
  conversation history,
)
```

In code today, `WorkspaceConfig` (`src/workspace.rs:82`) is a per-workspace
TOML file with a name, dimension, encoding, language, max-memory, seed packs,
and shared partitions. After Phase 38:

- One Akh has multiple workspaces and switches between them
  (browser-tab semantics).
- Multiple Akhs can join the same workspace simultaneously — knowledge
  sharing, not lockstep execution. They run independent OODA loops over a
  shared microtheory set.
- Loading a workspace = activating its microtheories on whatever daemon
  hosts the session. Microtheory contents stream from Postgres into the
  local in-process cache on first activation.
- Workspace metadata lives in `workspaces.*` Postgres tables. The on-disk
  TOML in `$XDG_CONFIG_HOME/akh-medu/workspaces/` becomes a
  *bootstrap-only* artifact; Postgres is authoritative.
- Phase 14 identity covers per-fact provenance — *which* Akh asserted *what*
  inside a shared workspace.

### Three-tier sleep model with continuous durability

| Level             | In memory | Continuously persisted   | Snapshot frequency | Wake cost                   |
|-------------------|-----------|--------------------------|--------------------|-----------------------------|
| Running           | full      | every assert via WAL     | every N min (≈5)   | 0                           |
| Idle-in-memory    | full      | every assert via WAL     | every N min (≈5)   | 0                           |
| Deep sleep        | freed     | already written to Postgres | full snapshot at transition | replay + HNSW rebuild |

**Crash safety.** At any moment, Postgres holds the full Akh state up to the
most recent WAL flush (seconds-old). Crash recovery replays from the latest
snapshot watermark forward through the WAL tail. Maximum dataloss is one
flush interval (target ≤5 s).

**Idle ≠ crash-unsafe.** Idle-in-memory is a CPU/perf state, not a
durability state. The WAL keeps flushing whether the agent is busy or not;
flushes piggyback on assertion activity, with a heartbeat for empty
intervals.

**Deep transition is on signal, not timer.** Idle → Deep happens on:

- Host shutdown (SIGTERM with grace period).
- Explicit relocation request (operator command, or scheduler decision).
- Severe memory pressure (e.g. >90 % of `max_memory_mb`).

It is *not* triggered by elapsed wall time. The warm-Akh experience stays
snappy across conversation gaps; we never burn wake-cost just because a
human walked away from the chat for 10 minutes. Crash safety remains
independent of the deep transition.

### Continuous WAL durability protocol

- Every assertion / retraction / microtheory-change passes through
  `Engine::add_triple` (`src/engine.rs:448`) and its siblings. Today the
  function returns after Phase-9 checks + in-process insert + (optional)
  oxigraph sync.
- Phase 38 wraps the same call: after the in-process insert succeeds, the
  mutation is appended to an in-process WAL ring buffer
  (`akhomed-wal-buffer`).
- The WAL buffer flushes to `akhomed.mutation_log` (Postgres) every N
  seconds **or** M asserts (whichever first); target ≤5 s flush interval
  under normal load. Default: N=2, M=64.
- Flushes are batched in a single transaction. If the transaction fails the
  buffer holds the pending entries and retries on the next tick (idempotent
  inserts; see "Idempotent batched writes" below).
- A **periodic full snapshot** runs every ~5 min (or on signal). The
  snapshot writer captures: (1) the current mutation watermark
  `(snapshot_id, mutation_log_seq)`, (2) the working-memory and
  episode-buffer slices, (3) the psyche state (already a single jsonb in
  `Psyche` at `src/compartment/psyche.rs:19`). The snapshot truncates the
  WAL behind the watermark on success.
- On wake/recovery: load latest snapshot, then replay any
  `mutation_log` rows newer than the snapshot watermark.
- The watermark + WAL is what makes this crash-safe with bounded dataloss,
  *independent* of the sleep-level transition.

### Snapshot/replay protocol

Snapshot is a row in `akhomed.snapshots`:

```
(snapshot_id BIGSERIAL,
 akh_id      UUID,
 created_at  TIMESTAMPTZ,
 watermark   BIGINT,    -- mutation_log seq this snapshot covers up to
 working_mem BYTEA,     -- bincode-encoded WorkingMemory (existing serde type)
 episodes    BYTEA,     -- bincode-encoded recent episode buffer
 psyche      JSONB,     -- current Psyche (already a Serialize type)
 state_blob  BYTEA      -- catch-all for sleep_cycle + ooda counter + drives
)
```

On replay:

1. `SELECT … FROM akhomed.snapshots WHERE akh_id = $1 ORDER BY created_at DESC LIMIT 1` → load the latest snapshot.
2. Decode `working_mem`, `episodes`, `psyche`, `state_blob` into the in-process
   structs. (These already ship `Serialize`/`Deserialize` impls — see
   `src/agent/sleep.rs:92` for `SleepCycle`, `src/compartment/psyche.rs:19`
   for `Psyche`.)
3. `SELECT … FROM akhomed.mutation_log WHERE seq > $watermark ORDER BY seq` →
   stream the WAL tail back through `Engine::add_triple` (idempotent — repeat
   asserts are no-ops at the cache boundary).
4. Mark the daemon Running.

Snapshot writes happen on:

- The 5-minute periodic timer.
- Idle → Deep transition (mandatory).
- An explicit relocation command (e.g. `akh akh relocate <id>`).
- Operator-requested checkpoint via the daemon HTTP API (useful for
  before-upgrade safety).

### HNSW wake protocol

HNSW(VSA) is **in-process only** — never persisted to Postgres. Reasons:

- The HNSW index from `hnsw_rs` is built around `Hnsw<'static, u32, DistHamming>`
  (`src/vsa/item_memory.rs:40`). It has no first-class serialization and its
  `'static` lifetime ties it to the process. Re-deriving it from the
  bipolar item-memory vectors is the only sane path.
- Index size is roughly proportional to symbol count × log(symbol count) ×
  bits per node. Pushing it through Postgres adds write amplification with
  no payoff: external readers (AGE consumers) don't need it.

Wake protocol:

| Wake type        | HNSW source                                    | Latency budget    |
|------------------|------------------------------------------------|-------------------|
| Same-host warm   | already in-memory                              | 0                 |
| Same-host cold   | local-disk cache file                          | sub-second        |
| Relocated cold   | rebuild from `akhomed.facts` rows              | ~hundreds of ms for 10K vecs, ~30–60 s for 1M vecs |

The local-disk cache is best-effort: if the file is missing or stale (the
schema fingerprint stored alongside the index doesn't match the current
build), the daemon falls back to rebuild-from-Postgres without complaint.

If a workload exceeds the rebuild budget, we measure first and tune second
(see "Open questions"). E-graph hits the same wall and is deferred to
Phase 40.

### AGE for outside-the-Akh read

Apache AGE (PostgreSQL extension) lets us expose `akhomed.facts` as a
property graph queryable in Cypher. Akhs themselves work on the in-process
oxigraph (fast, single-writer, SPARQL). AGE views over the same underlying
rows give *other* applications query access without needing to talk to the
akhomed daemon at all:

- KG visualisers (web frontends, debugger UIs)
- External federation tools that prefer Cypher to SPARQL
- Future Phase 22 Akhipedia rendering

This is read-only by design — writes flow only through an Akh's WAL. AGE is
queried over the same rows the WAL writes; there is no second write path to
keep in sync.

### Idempotent batched writes

- Asserts are content-addressed and deduplicated at the cache boundary
  (`Engine::add_triple` already short-circuits if the triple is present).
- Read-heavy workload assumed; only dedicated jobs write back. The numbers
  from Phase 37's analysis hold: a typical Akh writes ~10²–10³ triples/min
  during active learning, ~0 during idle.
- Write-back to Postgres is batched on a flush trigger (every N seconds or
  M asserts, whichever first).
- Assertion idempotence means we can safely retry batched writes on
  transient failure. The mutation_log primary key is `(akh_id, seq)`; the
  client-side seq is monotonic per-Akh; conflicting `(akh_id, seq)` rows on
  retry are absorbed by `ON CONFLICT DO NOTHING`.
- Same idempotence makes WAL replay safe (re-applying mutations through
  `Engine::add_triple` is a no-op past the first).

### Multi-Akh sharing

Two regimes:

**Shared homeserver.** Two Akhs running on the same akhomed in the same
workspace see each other's assertions live. Both read the same Postgres
rows; both have their own in-process cache. Cache invalidation is
*pull-on-read*: when either Akh's WAL flushes, a small `pg_notify` channel
publishes the new watermark; the other Akh, on its next OODA tick, fetches
any rows it doesn't yet have. Active coordination still goes through the
existing `AgentProtocolMessage` (Phase 12g).

**Cross-homeserver.** Two Akhs on different akhomeds (cluster + M2). The
Phase 36 delegation protocol covers Akh ↔ Akh across networks for active
coordination. For passive knowledge sharing — "I want a copy of your
microtheory on Egyptian mathematics" — workspace contents replicate via
`.akhskill`-style archives (Phase 31) for off-network sync. This avoids
demanding that every M2-Akh have a live Postgres connection to the cluster
24/7; the M2 can pull a microtheory once and run offline.

### Akh-to-host binding

- Akhs are statically stuck to one akhomed by default. The relocation path
  is a recovery mechanism, not a routine load-balancer.
- Default home: M2 in cellar (when deployed) hosts most Akhs; the cluster
  hosts the always-on primary that Claude talks to.
- 7900 XTX desktop is *not* an Akh host — it's an inference backend that
  any Akh can route to (see Phase 39).
- Migration is a recovery mechanism: host dies → wake elsewhere with same
  identity, picking up from the latest snapshot + WAL tail.

## Postgres schema additions

DDL is sketched here for the plan; final SQL ships in 38a as a
`sea-orm-migration` runner alongside the Seshat one.

### `akhomed.*` schema

```sql
CREATE SCHEMA IF NOT EXISTS akhomed;

-- Akh identities (primary key, not host-bound)
CREATE TABLE akhomed.akh (
    id            UUID PRIMARY KEY,
    name          TEXT NOT NULL,
    home_host     TEXT,           -- preferred akhomed host id; advisory only
    awakened_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen     TIMESTAMPTZ
);

-- The fact table — every triple ever asserted by any Akh
-- backs both the in-process oxigraph cache and the AGE view
CREATE TABLE akhomed.facts (
    fact_id       BIGSERIAL PRIMARY KEY,
    akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
    workspace_id  UUID NOT NULL,
    subject       BIGINT NOT NULL,    -- SymbolId.get()
    predicate     BIGINT NOT NULL,
    object        BIGINT NOT NULL,
    confidence    REAL NOT NULL DEFAULT 1.0,
    asserted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    retracted_at  TIMESTAMPTZ,
    provenance_id BIGINT,             -- foreign key into akhomed.provenance
    compartment   TEXT,
    CONSTRAINT facts_unique UNIQUE (workspace_id, subject, predicate, object)
);

CREATE INDEX facts_by_workspace ON akhomed.facts(workspace_id);
CREATE INDEX facts_by_subject   ON akhomed.facts(workspace_id, subject);
CREATE INDEX facts_by_predicate ON akhomed.facts(workspace_id, predicate);
CREATE INDEX facts_by_object    ON akhomed.facts(workspace_id, object);

-- The streaming WAL — every assert/retract/microtheory mutation
-- gets a row here within seconds of happening
CREATE TABLE akhomed.mutation_log (
    seq           BIGSERIAL PRIMARY KEY,
    akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
    workspace_id  UUID NOT NULL,
    kind          TEXT NOT NULL,    -- 'assert' | 'retract' | 'microtheory_load' | …
    payload       JSONB NOT NULL,
    applied_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX mutation_log_by_akh ON akhomed.mutation_log(akh_id, seq);

-- Periodic snapshot table; one row per snapshot
CREATE TABLE akhomed.snapshots (
    snapshot_id   BIGSERIAL PRIMARY KEY,
    akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    watermark     BIGINT NOT NULL,   -- mutation_log seq this covers up to
    working_mem   BYTEA NOT NULL,
    episodes      BYTEA NOT NULL,
    state_blob    BYTEA NOT NULL,
    psyche        JSONB
);

CREATE INDEX snapshots_by_akh ON akhomed.snapshots(akh_id, created_at DESC);

-- Per-Akh psyche, mirrored as a current-row table for AGE / external read
-- (the snapshots table holds historical psyche; this is the live one)
CREATE TABLE akhomed.psyche (
    akh_id        UUID PRIMARY KEY REFERENCES akhomed.akh(id),
    persona       JSONB NOT NULL,
    shadow        JSONB NOT NULL,
    archetypes    JSONB NOT NULL,
    self_integration JSONB NOT NULL,
    awakened      BOOLEAN NOT NULL DEFAULT false,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Provenance ledger; mirrors src/provenance.rs
CREATE TABLE akhomed.provenance (
    provenance_id BIGSERIAL PRIMARY KEY,
    akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
    workspace_id  UUID NOT NULL,
    subject       BIGINT NOT NULL,
    derivation    JSONB NOT NULL,    -- DerivationKind serialised
    inputs        BIGINT[] NOT NULL,
    confidence    REAL NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Episodic memory — survives sleep cycles
CREATE TABLE akhomed.episodes (
    episode_id    BIGSERIAL PRIMARY KEY,
    akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
    workspace_id  UUID NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL,
    ended_at      TIMESTAMPTZ,
    payload       JSONB NOT NULL    -- the episode struct
);
```

### `workspaces.*` schema

```sql
CREATE SCHEMA IF NOT EXISTS workspaces;

CREATE TABLE workspaces.workspace (
    id            UUID PRIMARY KEY,
    name          TEXT NOT NULL UNIQUE,
    dimension     INTEGER NOT NULL DEFAULT 10000,
    encoding      TEXT NOT NULL DEFAULT 'bipolar',
    language      TEXT NOT NULL DEFAULT 'auto',
    seed_packs    TEXT[] NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Membership: which Akhs are members of which workspaces
CREATE TABLE workspaces.membership (
    workspace_id  UUID NOT NULL REFERENCES workspaces.workspace(id),
    akh_id        UUID NOT NULL REFERENCES akhomed.akh(id),
    joined_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    role          TEXT NOT NULL DEFAULT 'member',
    PRIMARY KEY (workspace_id, akh_id)
);

-- Microtheories activated in this workspace
CREATE TABLE workspaces.active_microtheories (
    workspace_id  UUID NOT NULL REFERENCES workspaces.workspace(id),
    microtheory_id TEXT NOT NULL,
    loaded_by     UUID NOT NULL REFERENCES akhomed.akh(id),
    loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (workspace_id, microtheory_id)
);
```

### AGE view definition

AGE creates a graph by name and exposes it through Cypher. The wiring is:

```sql
CREATE EXTENSION IF NOT EXISTS age;
LOAD 'age';
SET search_path = ag_catalog, public;
SELECT create_graph('akhomed_kg');

-- Nightly job (or trigger) projects akhomed.facts into AGE vertices/edges:
--   Each distinct symbol → a vertex labelled 'Symbol' with id = subject
--   Each row in akhomed.facts → an edge labelled with the predicate IRI
-- The exact projection script ships in 38d.
```

External readers query:

```cypher
MATCH (s:Symbol)-[r]->(o:Symbol)
WHERE r.workspace_id = '...'
RETURN s, r, o
LIMIT 100;
```

We do **not** treat AGE as the source of truth. It's a periodically
refreshed projection over `akhomed.facts`. Akhs read SPARQL from the
in-process oxigraph cache, which is itself populated from `akhomed.facts`.
AGE is a read view for non-Akh consumers, full stop.

## Wake/sleep state machine

```
                 ┌──────────────────────────────────────────┐
                 │                                          │
                 │           ┌──── WAL flush ────┐          │
                 ▼           │                   │          │
            ┌─────────┐   ┌──┴───────────────────┴──┐      │
   wake ──▶ │ Running │ ◀▶│   Idle-in-memory       │      │
            └────┬────┘   └────┬───────────────────┘      │
                 │             │                           │
                 │ explicit signal:                        │
                 │ - SIGTERM with grace period             │
                 │ - operator relocate                     │
                 │ - severe memory pressure                │
                 ▼                                         │
            ┌──────────────────────────────────┐           │
            │   Deep sleep                      │ ── crash, host dies, etc.
            │   - in-memory caches dropped     │           │
            │   - last snapshot is current     │           │
            │   - only Postgres holds state    │           │
            └────────┬─────────────────────────┘           │
                     │                                     │
                     │  wake on:                          │
                     │  - operator command                ─┘
                     │  - workspace activation request
                     │  - inbound message via Phase 36
                     ▼
              (relocatable to any akhomed)
```

### Transitions and actions

| From               | To                | Trigger                                                                 | Actions                                                                                                      |
|--------------------|-------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| (cold)             | Running           | wake command on a host                                                  | Load snapshot → replay WAL tail → rebuild HNSW (from disk cache or Postgres) → mark akh.last_seen → emit AwakeAudit |
| Running            | Idle-in-memory    | OODA cycle ran without external input for K cycles                      | Park OODA loop; daemon background tasks continue; WAL keeps flushing                                         |
| Idle-in-memory     | Running           | external input (HTTP, WS, AgentProtocolMessage)                         | Resume OODA loop; no I/O cost                                                                                |
| Running / Idle     | Deep sleep        | SIGTERM with grace, relocate command, memory pressure ≥0.9              | (1) Force WAL flush (2) Write fresh snapshot (3) Drop in-process oxigraph/petgraph/HNSW/e-graph (4) Update akh.last_seen (5) Emit DeepSleepAudit |
| Deep sleep         | Running           | wake command on any host                                                | Cold wake protocol (load snapshot → replay tail → rebuild HNSW)                                              |
| Running / Idle     | Crashed (kill -9) | kernel killed the process, host died                                    | None — by definition                                                                                          |
| Crashed            | Running           | next wake on any host                                                   | Cold wake from latest snapshot + replay any WAL rows newer than the snapshot watermark                       |

The same recovery path serves both "scheduled deep sleep" and "kernel
killed us" — the daemon doesn't need to distinguish. Kernel kills lose at
most one flush interval; deep sleep loses zero.

## Sub-phase breakdown

The order is dependency-driven. Each sub-phase ends with a working daemon
plus a smaller, isolated test harness.

### 38a — Postgres schema + SeaORM models (~700 lines)

**Depends on:** Phase 37 (SeaORM dependency, migration runner pattern)

**What ships:**

- New crate `crates/akhomed-store/` with SeaORM entities for all
  `akhomed.*` and `workspaces.*` tables.
- `sea-orm-migration` migrations that create the schemas, tables, and
  indices (no AGE yet).
- A tiny CLI `akhomed-migrate` that runs migrations against a configured
  Postgres URL (mirrors Seshat's pattern in `crates/seshat/`).
- Config: a new `[postgres]` section in `akhomed.toml` with `url`,
  `pool_size`, `connect_timeout`, `migrate_on_start`.
- Connection pool plumbed into `ServerState` in `src/bin/akhomed.rs`.

**Acceptance:**

1. `akhomed-migrate` against a fresh Postgres creates all tables + indices
   with no errors.
2. The pool initializes on daemon start when `[postgres]` is set; daemon
   logs a clear warning and continues in legacy local-only mode when not.
3. SeaORM model unit tests cover round-trip serialisation of every column
   type (especially the JSONB columns for psyche / payload / derivation).

### 38b — Mutation WAL (~600 lines)

**Depends on:** 38a

**What ships:**

- New module `src/wal.rs` implementing `MutationLog`:
  - In-process `Vec<MutationEntry>` ring buffer (ring is the wrong data
    structure if we have to retry; use a `VecDeque` and bounded retain
    instead).
  - Background flusher task spawned per workspace, parameterised by
    `(flush_interval, batch_size, max_buffer_bytes)`.
  - Flush implementation: `INSERT … ON CONFLICT DO NOTHING` against
    `akhomed.mutation_log`.
- `Engine::add_triple` and `Engine::retract_triple` gain a hook that
  appends to the workspace's `MutationLog`. Hook is no-op when no
  Postgres pool is configured (so 38b ships without breaking local-only).
- Mutation kind taxonomy: `assert`, `retract`, `microtheory_load`,
  `microtheory_unload`, `compartment_create`, `compartment_destroy`. The
  `payload` JSONB carries kind-specific data.
- Metrics on `MutationLog`: pending count, last flush latency, last flush
  size, dropped-on-overflow counter.

**Acceptance:**

1. With Postgres configured, every `add_triple` results in a row in
   `akhomed.mutation_log` within `flush_interval` seconds. Verified via
   integration test that asserts a triple, sleeps 6 s, queries Postgres.
2. `kill -9` the daemon mid-write, restart, the WAL queries show the
   pre-kill rows committed and the in-flight ones lost — and the count of
   lost rows is bounded by `flush_interval × write_rate`.
3. Postgres unreachable for 30 s then reconnects: no rows lost (buffer
   absorbs), error path doesn't deadlock the daemon. Buffer cap is
   honoured; on overflow we drop oldest non-`assert` mutations first and
   log a `BufferOverflow` audit entry.

### 38c — Snapshot writer + replay loader (~700 lines)

**Depends on:** 38b

**What ships:**

- New module `src/snapshot.rs`:
  - `SnapshotWriter::write_snapshot(akh_id, engine, agent) -> Result`:
    serialises `WorkingMemory`, episode buffer, `SleepCycle`, drives,
    psyche, OODA cycle counter; computes `watermark` (the current max
    seq from `mutation_log` for this akh); inserts a row in
    `akhomed.snapshots`; truncates `akhomed.mutation_log` rows ≤ that
    watermark.
  - `SnapshotLoader::wake(akh_id) -> Result<EngineBootstrap>`: loads
    latest snapshot, decodes into struct shells, returns a payload the
    daemon merges into a fresh `Engine`.
  - `Replayer::apply_wal_tail(engine, watermark)`: streams
    `akhomed.mutation_log WHERE seq > watermark` and re-applies through
    `Engine::add_triple` (idempotent).
- New daemon background task: `snapshot_interval` (default 5 min) on
  `DaemonConfig`. Snapshot also fires on graceful shutdown.
- A new CLI subcommand `akh akh snapshot <name>` for manual triggering.

**Acceptance:**

1. Five-minute timer test: assert 100 triples, wait 6 min, assert 100
   more, kill -9, restart on the *same* host. The first 100 are loaded
   from snapshot, the second 100 from WAL replay. Final triple count
   matches expectation.
2. Snapshot truncation test: post-snapshot, the `mutation_log` count
   for that akh dropped by exactly the number of pre-snapshot rows.
3. Cross-host wake test: shut down host A, point host B at the same
   Postgres, run the same `akh akh wake` command. The Akh comes up
   with identical state (asserted via deep equality of the in-process
   `KnowledgeGraph` triple set).

### 38d — AGE views + projection job (~500 lines)

**Depends on:** 38a

**What ships:**

- A migration step in 38a's `akhomed-migrate` that runs
  `CREATE EXTENSION age` and `SELECT create_graph('akhomed_kg')` if AGE
  is available; logs and skips otherwise.
- A `src/projection/age_project.rs` module that reads `akhomed.facts`
  and emits AGE vertex/edge upserts. Two modes: full rebuild (idempotent,
  for first projection) and incremental (since-watermark, for ongoing
  refresh).
- An `akhomed projection refresh` CLI command and a daemon timer
  (default: 30 min, configurable).
- A short read-side example in the repo: a Cypher snippet that lists the
  most-connected concepts in a workspace.

**Acceptance:**

1. After ingesting 1 K triples, AGE Cypher query returns the expected
   vertex/edge count.
2. Removing a triple via `Engine::retract_triple` then refreshing the
   projection drops the corresponding edge.
3. The daemon runs cleanly when AGE is *not* installed (the projection
   job logs a warning and exits 0; nothing else fails).

### 38e — Three-tier sleep state machine (~600 lines)

**Depends on:** 38c

**What ships:**

- New type in `src/agent/sleep.rs`: extend `ConsolidationPhase` (which is
  today {Wake, Consolidation, Reorganization, DreamState}) with a new
  enum `SleepLevel { Running, IdleInMemory, Deep }` orthogonal to the
  consolidation-phase enum. (Sleep level is about durability; the
  existing enum is about the cognitive sub-phase during a sleep cycle.)
- State-machine transitions wired into `AgentDaemon`:
  - Running ↔ Idle: `K` cycles without external input; resumes on input.
  - * → Deep: SIGTERM grace, `akh akh sleep <name>`, memory pressure.
  - Deep → Running: external wake (CLI, HTTP wake endpoint, peer RPC).
- HNSW local-disk cache + invalidation logic in `src/vsa/item_memory.rs`:
  - `ItemMemory::dump_to_disk(path)` serialises the bipolar vectors and
    a small schema fingerprint (dimension, encoding, item count, hash
    of stable seed) — *not* the HNSW graph itself.
  - On wake, `ItemMemory::load_or_rebuild(path, ops)` either replays the
    bipolar vectors and reinserts into a fresh HNSW (sub-second), or
    rebuilds from `akhomed.facts` when the dump is missing/mismatched.

**Acceptance:**

1. The daemon on SIGTERM writes one final snapshot, one final WAL flush,
   and exits cleanly; on restart, the same Akh comes back up with no
   delta in triple count.
2. Memory-pressure deep sleep: drive the daemon to >90% of
   `max_memory_mb`; verify it transitions to Deep, frees in-process
   structures, and remains responsive to wake.
3. Same-host warm wake from Deep is sub-second; cold wake from
   `akhomed.facts` for a 10 K-triple workspace is under 5 s.

### 38f — Workspace-as-Postgres-row + multi-Akh sharing (~700 lines)

**Depends on:** 38c

**What ships:**

- `WorkspaceManager` (`src/workspace.rs:206`) gets a Postgres-backed
  variant. Existing TOML loader stays as a bootstrap path: on first
  startup, missing workspaces in Postgres are seeded from local TOML
  files; thereafter Postgres is authoritative.
- New CLI: `akh workspace join <id>` adds an Akh to a workspace
  (membership row); `akh workspace leave`; `akh workspace list-members`.
- pg_notify channel `akhomed_wal_<workspace_id>`: WAL flushers
  `NOTIFY` on commit; daemons subscribed to that workspace `LISTEN` for
  the new watermark and pull rows newer than what they've cached.
- `AgentProtocolMessage` (existing Phase 12g type) gains a `WorkspaceJoin`
  / `WorkspaceLeave` variant so peer Akhs can notice each other's
  arrival/departure proactively (rather than discovering it via a stale
  membership read).

**Acceptance:**

1. Two daemons, two Akhs, one shared workspace, one shared Postgres:
   Akh A asserts; within ≤5 s + WAL tick, Akh B's in-process oxigraph
   cache contains the new triple (verified via SPARQL).
2. Akh A and Akh B both assert against the same `(s, p, o)` simultaneously:
   one row in `akhomed.facts` (deduplicated by the unique constraint),
   both Akhs converge to identical caches.
3. Akh A retracts a triple Akh B is using to support a derivation: the
   TMS in Akh B receives the retract via WAL replay and the dependent
   inferences are cascaded out.

### 38g — Migration playbook + crash-recovery harness (~400 lines)

**Depends on:** 38e, 38f

**What ships:**

- An end-to-end test harness `tests/relocation.rs`:
  - Spin up two ephemeral akhomed processes against a shared Postgres.
  - Drive the first one to a known state.
  - SIGKILL the first; bring up the second pointing at the same akh_id.
  - Assert deep equality of triple set, working memory, episode buffer,
    psyche.
- **Postgres advisory-lock split-brain guard** (hard requirement, not a
  follow-up): the wake/relocation path acquires
  `pg_try_advisory_lock(hashtext('akh:' || akh_id))` before touching
  Akh state, in addition to the `last_seen` heartbeat check. Advisory
  locks are session-scoped, so they survive clock skew between hosts —
  the `last_seen` guard catches the routine race, the advisory lock
  catches the clock-skew race. Wake fails fast with a clear error if
  either guard rejects.
- A short operator runbook in `docs/ops/relocate-akh.md`:
  1. Identify the dead host.
  2. Pick a recovery host (cluster or M2).
  3. Run `akh akh wake --akh-id <id>` on the recovery host. The wake
     command refuses if `akh.last_seen` was updated within the last
     30 s by another host (split-brain guard) **or** if the advisory
     lock for this `akh_id` is held by another session.
  4. Verify the recovered Akh responds to `akh status`.
- A property test: random sequences of asserts/retracts, randomly
  interleaved with kill -9, always recover with bounded dataloss.

**Acceptance:**

1. The kill-mid-write recovery test passes 100 times in CI without flake.
2. Cross-host wake test succeeds with the same identity (UUID) and
   identical in-memory state hash.
3. Split-brain guard: attempting to wake an Akh on host B while host A is
   still updating `last_seen` returns a clear error and exits non-zero.

## Estimated effort

| Sub-phase | Description                                          | Lines  |
|-----------|------------------------------------------------------|--------|
| 38a       | Postgres schema + SeaORM models                      | ~700   |
| 38b       | Mutation WAL                                         | ~600   |
| 38c       | Snapshot writer + replay loader                      | ~700   |
| 38d       | AGE views + projection job                           | ~500   |
| 38e       | Three-tier sleep state machine                       | ~600   |
| 38f       | Workspace-as-Postgres-row + multi-Akh sharing        | ~700   |
| 38g       | Migration playbook + crash-recovery harness          | ~400   |
| **Total** |                                                      | **~4,200** |

## Migration path

Today's akhomed is in-process oxigraph + redb on a per-workspace local
directory. The transition is staged so the daemon stays runnable at every
step:

1. **Layer the WAL on top first (38a + 38b).** Postgres is optional. With
   no `[postgres]` config, the daemon behaves exactly as today. With it
   set, every assert also lands in `akhomed.mutation_log` — but oxigraph
   is still the read source. This is risk-free: a Postgres outage
   degrades to local-only behaviour.
2. **Add snapshots (38c).** Now we can recover *across* host crashes,
   provided the same host comes back. Cross-host wake is technically
   possible but not yet polished.
3. **Add AGE projections (38d).** External read consumers come online.
   Akh-internal behaviour unchanged.
4. **Change the boot path (38e).** The daemon stops booting from local
   oxigraph by default and instead boots from Postgres snapshot + WAL.
   Local oxigraph becomes a write-through cache, rebuilt on cold wake.
   This is the riskiest transition; it ships behind a config flag
   `[postgres] authoritative = true` and keeps the local-first path as a
   fallback for one release.
5. **Promote workspaces (38f).** Workspace metadata moves to Postgres.
   The TOML files stay as bootstrap fixtures, ignored after first sync.
6. **Codify the playbook (38g).** Operators get a clear procedure for the
   "M2 died, wake on cluster" case.

The intent is that no single sub-phase requires an existential rewrite of
the engine. `Engine::add_triple` keeps its signature; the WAL and AGE
projection are layered around it. The boot path moves last, when we
already trust the WAL and snapshot machinery.

## Verification across the phase

- **WAL liveness.** Continuous Postgres write rate ≤ N rows/s during
  normal operation, ≥0 even during idle (heartbeat).
- **Crash recovery.** Random kill -9 during writes loses ≤ flush_interval
  seconds of mutations; nothing else.
- **Cross-host equivalence.** A snapshotted Akh, woken on a fresh host,
  produces a triple set hash equal to the source host's.
- **Multi-Akh convergence.** Two Akhs in one workspace converge within
  ≤ flush_interval seconds after each assertion.
- **External read.** AGE Cypher query results match the equivalent SPARQL
  result on the same data within projection-refresh window.
- **No regression on local-only mode.** With `[postgres]` unset, the
  daemon behaves exactly as today (existing tests pass unmodified).

## Open questions

These come from the Anima ticket plus what surfaced while reading the
code.

1. **Snapshot / WAL flush tuning.** 5 s flush + 5 min snapshot are
   starting points. What's the actual cost at typical write rates?
   Specifically: how many `mutation_log` rows per active hour for a
   typical Akh, what's the dominant payload size (assert vs retract vs
   microtheory_load), and where does retention bite — at the
   `mutation_log` table or at the `snapshots` table? We want a
   measurement-first tuning pass after 38c lands, before 38e.

2. **Workspace ↔ microtheory composition.** A workspace activates a *set*
   of microtheories, but microtheories themselves can compose
   (`ctx:specializes` chains via `src/compartment/microtheory.rs`). When
   two workspaces sharing an Akh transitively activate microtheories
   that conflict, what wins? Options: (a) workspace-of-most-recent-load
   wins, (b) Phase 17 evidence intervals decide per-fact, (c) the union
   of microtheories is taken and conflicts surface as TMS contradictions.
   Plan: pick (b) as the default and leave (c) as a per-workspace
   policy override; revisit after 38f.

3. **E-graph state on wake.** Rebuild from facts (slow on large
   theories) or persist (own phase, see Phase 40)? At current scale —
   a few thousand triples — rebuild is cheap and correct. The decision
   point is: if a workspace exceeds ~50 K triples or its e-graph
   rewrite-saturation latency exceeds the cold-wake budget, durability
   becomes worth it. Phase 38 proceeds with rebuild; Phase 40 is queued
   if the metrics demand it.

4. **HNSW local-disk cache: shared between processes?** *Resolved
   (2026-04-25):* akhomed runs alone per host/container in production.
   No sibling-process sharing concern. HNSW local-disk cache is
   single-writer with no cross-process locking layer.

5. **Split-brain on relocation.** *Resolved (2026-04-25):* the
   Postgres advisory lock keyed on `akh_id` (in addition to the
   `last_seen` guard) is accepted as a hard requirement, not a
   follow-up. Advisory locks are session-scoped and survive clock
   skew. See sub-phase 38g deliverables — the lock guard ships with
   the wake/relocation path.

6. **AGE adoption risk.** *Resolved (2026-04-25):* AGE availability
   is guaranteed by the custom CNPG image build (homelab repo). The
   image bakes the AGE extension at known versions. AGE is still
   treated as a projection over `akhomed.facts` so that a future
   migration off AGE doesn't lose Akh state — but the deploy-side
   commitment makes the runtime risk effectively zero. If AGE
   upstream stops being maintained, we vendor it; we do not lose it
   via dependency rot.

## Risks and mitigations

| Risk                                                 | Likelihood | Impact   | Mitigation                                                                  |
|------------------------------------------------------|-----------|----------|-----------------------------------------------------------------------------|
| Postgres unreachable for hours                       | Medium    | High     | WAL buffer absorbs up to `max_buffer_bytes`; on overflow, daemon enters degraded mode and refuses new asserts (clear error to caller). |
| WAL flush stalls (Postgres slow)                     | Medium    | Medium   | Flush is on a background task; OODA loop never blocks on it. Backpressure surfaces as an audit entry, not a hang. |
| AGE Cypher planner is slow                          | High      | Low      | AGE is a read-side convenience; if it's slow, callers fall back to direct SPARQL through the akhomed HTTP API. |
| AGE upstream stops being maintained                  | Low       | Low      | AGE availability is guaranteed by the custom CNPG image build (homelab repo) which bakes the extension at known versions. If upstream rots we vendor it. AGE is still treated as a projection over `akhomed.facts` so a future migration off AGE doesn't lose Akh state. |
| Cross-host wake races (split-brain)                 | Low       | High     | `last_seen` heartbeat + Postgres advisory lock; CLI refuses to wake within 30 s of another host's heartbeat. |
| Snapshot blob size grows unbounded                   | Medium    | Medium   | Snapshots include only working memory + episodes (recent-N), not full KG; full KG is in `akhomed.facts`. Episode count capped per snapshot. |
| HNSW rebuild bottlenecks cold wake                  | Medium    | Medium   | Local-disk cache mitigates same-host. Cross-host falls back to rebuild; if it exceeds budget on real workloads, defer to Phase 40. |
| Idempotency assumption violated by some future op    | Low       | High     | Every mutation kind ships with a contract test that asserts WAL-replay safety. Adding a new kind requires checking idempotence. |
| Postgres + local oxigraph drift                     | Medium    | High     | A nightly `akhomed verify` job samples N facts per workspace and asserts oxigraph cache matches Postgres. Drift triggers a full cache rebuild. |
| Operator confusion during migration                  | High      | Medium   | The 38g playbook is mandatory. Snapshot + WAL is opt-in via config flag for the first release; legacy local-only path stays available. |
| Schema migration failure mid-deploy                  | Low       | High     | `sea-orm-migration` runs migrations transactionally; failed migration leaves the schema in the previous state. CI runs the full migration sequence on every PR. |

## Dependencies (existing components touched)

- `src/engine.rs` — `Engine::add_triple` (`:448`),
  `Engine::add_triple_with_support` (`:1879`), retract paths.
- `src/graph/sparql.rs` — `SparqlStore` becomes a write-through cache
  rather than the source of truth.
- `src/store/durable.rs` — redb stays for local-only metadata; not the
  cross-host source of truth.
- `src/agent/daemon.rs` — `DaemonConfig` gains `wal_flush_interval`,
  `snapshot_interval`, `idle_threshold_cycles`, `memory_pressure_threshold`.
  Background tasks gain WAL flush + snapshot + AGE projection refresh.
- `src/agent/sleep.rs` — `ConsolidationPhase` stays as the
  cognitive-sub-phase enum; new `SleepLevel` enum stacks on top for
  durability state.
- `src/workspace.rs` — `WorkspaceManager` learns to read/write
  `workspaces.workspace`, `workspaces.membership`,
  `workspaces.active_microtheories`.
- `src/compartment/psyche.rs` — `Psyche` is already `Serialize`/`Deserialize`,
  ready to live in a JSONB column.
- `src/vsa/item_memory.rs` — local-disk cache + rebuild from
  `akhomed.facts`.
- `src/bin/akhomed.rs` — `ServerState` carries the SeaORM connection pool;
  startup wires WAL/snapshot tasks per workspace.

## References

- Anima ticket: `019dc599-81c5-7ce2-a4e3-1a84ad6001e6`
- ADR 049 — Relocatable Session-Based Akh: Postgres-backed truth with
  in-process cache (sibling document)
- ADR 047 — Seshat's Archive (the precedent for SeaORM-backed Postgres
  integration)
- ADR 037 — LLM-VSA Deep Integration (sets the multi-host inference
  story Phase 38 enables)
- Phase 14 plan — identity, provenance (provides per-fact provenance
  schema we mirror into `akhomed.provenance`)
- Phase 17 plan — evidence intervals (used to resolve cross-Akh fact
  conflicts in shared workspaces)
- Phase 36 plan — delegation + external scheduling (the cross-homeserver
  coordination protocol)
- Phase 39 (queued) — network inference backends (Phase 38 separates
  "where state lives" from "where compute happens")
- Phase 40 (queued) — e-graph durability, if rebuild exceeds wake budget

## Resolutions log

### 2026-04-25

- **P38Q4** (HNSW local-disk cache shared between sibling akhomeds): NO. akhomed runs alone per host/container; single-writer with no cross-process locking layer.
- **P38Q5** (Split-brain on relocation under clock skew): YES, Postgres advisory lock keyed on `akh_id` is a hard requirement in 38g, in addition to the `last_seen` guard.
- **P38Q6** (AGE adoption risk): AGE availability is guaranteed by the custom CNPG image build (homelab repo). Runtime risk is effectively zero; AGE remains a projection so migrating off it would not lose Akh state.
