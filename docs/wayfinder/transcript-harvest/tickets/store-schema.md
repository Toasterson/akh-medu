---
type: grilling
status: closed
flag-cleared: 2026-09-19 — own cluster with TimescaleDB (see end)
resolved: 2026-09-19
blocked-by: [session-identity.md, event-normal-form.md]
---

## Question

What is the physical schema, and which Postgres does it live in?

1. **Which cluster.** The akh-pg CNPG cluster in Talos k8s (LAN
   192.168.1.16:5432) already has TimescaleDB, so the extension question is
   settled there — but this store is deliberately *not* the akh's, per
   [the substrate decision](substrate.md). Same cluster with its own database or
   schema, or a separate cluster? Blast radius and backup policy decide it.
1b. **The key is settled.** [Session identity](session-identity.md) fixed it as
   `(source, session_id, stream_id, seq)`, with harvesting machine as an
   attribute rather than a key column. Records are append-only, so writes are
   idempotent upserts on that key. Note `stream_id` is a *new* level this ticket
   did not originally anticipate — a session has many streams.
2. **Hypertable layout.** Partition column (record timestamp vs harvest time —
   they differ, and backfilling an old session must not fight the chunk
   boundaries), chunk interval, and whether the derived tables are hypertables
   too or plain tables keyed into the raw one.
3. **Compression.** Which policy, after what age, and segment-by what. Imitate
   `crates/akhomed-store/src/migration/m20260620_000002_mutation_log_hypertable.rs`
   — including its guard that no-ops when the extension is absent, so the schema
   still works on a plain Postgres for testing.
4. **Indexes.** What answers `get_lines` cheaply, what answers grep (GIN trigram
   vs tsvector), and what answers "all tool_use across all sources last week".
   **Amended 2026-09-19:** embeddings now definitely live here, so **pgvector is a
   hard requirement of the chosen cluster, not an option** — confirm the extension
   is available alongside TimescaleDB in the target image *before* settling the
   cluster question in item 1. Index type and dimensionality belong to
   [the embedding pipeline](embedding-pipeline.md); this ticket owns where the
   vector table sits relative to the hypertable.
5. **Migrations.** SeaORM entities and migrations, per project convention — never
   raw SQL. Advisory-locked like `akhomed_store::run_migrations_locked` if more
   than one machine can start a harvester concurrently.
6. **Retention.** ADR 091 chose keep-raw-indefinitely for the akh's own turns.
   Does the same hold here, given the corpus grows with every session on every
   machine?

## Resolution (2026-09-19)

### 1. Placement — own database in the existing akh-pg cluster

A separate **database** in the CNPG cluster already running in Talos k8s
(192.168.1.16:5432). Reuses backup, monitoring and operator ops; a separate
database keeps the blast radius off akh truth, satisfying
[the substrate decision](substrate.md)'s independence requirement without
standing up another cluster for ~3 GB/year.

**Gated on extension availability.** `scripts/pg-standalone.sh` documents that the
base image `ghcr.io/cloudnative-pg/postgresql:16` ships **neither** extension:
pgvector arrives via a wrapper image and TimescaleDB is optional. Both are
currently *guarded to no-op when absent* across `akhomed-store` migrations. That
was fine when they were nice-to-have; pgvector is now load-bearing for
[the read surface](read-mcp-surface.md)'s semantic half. New ticket:
[extension verification](extension-verification.md).

### 2. Physical layout — raw is the hypertable, with carry-forward time

**Chosen against the standing recommendation** (raw plain / events hypertable).
The accepted trade-off: columnar compression applies to the bulk of the data,
at the price of a partly-synthetic column inside the primary key.

- **Partition column `observed_at`**, and therefore
  **PK = `(observed_at, source, session_id, stream_id, seq)`** — TimescaleDB
  requires the partition column in every unique index, which is why the natural
  key from [session identity](session-identity.md) cannot stand alone. The
  existing `m20260620_000002_mutation_log_hypertable.rs` hit exactly this and is
  the pattern to copy.
- **`observed_at_synthetic boolean`** marks filled values. Never silently.

#### The carry-forward rule, and why idempotency survives it

Because no unique index may omit `observed_at`, **idempotency now depends on
carry-forward being deterministic**. The rule is therefore forward-only:

1. The record's own timestamp, when present.
2. Otherwise the timestamp of the nearest **preceding** record in the same stream.
3. Never a *following* record. Backward-fill would change a record's
   `observed_at` between harvest runs as the stream grows — changing its primary
   key and silently inserting a duplicate row. This is the single sharpest hazard
   in the layout and the reason the rule is one-directional.
4. A record with no preceding timestamp in its stream is **quarantined, not
   guessed** — consistent with [the normal form](event-normal-form.md)'s
   never-guess stance.

Forward-fill is sound because stream order is provably topological (0 violations
in 15,861 parent links) and the sources are append-only, so rule 1–2 depend only
on content that can never change.

**Measured exposure:** 11.1% of Claude Code records carry no timestamp
(`last-prompt` 793, `mode` 651, `ai-title` 513, `atis-latch` 234,
`started`/`result` 8); Codex is at 100% coverage. Of 46 streams, exactly **one**
has timestamp-less records before any timestamped one — and it is not a
transcript (see below). So rule 4 fires essentially never on real streams.

#### The workflow journal is a distinct stream kind

`<slug>/<sessionId>/subagents/workflows/wf_<id>/journal.jsonl` holds **only**
`started`/`result` records — no timestamps anywhere, and **no `sessionId` in the
content**: its session is derivable only from its path. It is the home of every
spawn-link record the normal form's reading order depends on. Consequences:

- Its `observed_at` comes from its parent session's stream, which is immutable
  and therefore deterministic — not from file mtime, which is not.
- Discovery must classify it rather than treat it as a transcript:
  [source file lifecycle](source-file-lifecycle.md).
- Stream identity is **path-derived** for this kind: noted on
  [session identity](session-identity.md).

### 3. Derived tables

Normalized events are a second hypertable sharing `observed_at` and the same key
columns, so chunk boundaries align and joins stay local. Render text is a column
on the normalized row, not a third table. Vector storage belongs to
[the embedding pipeline](embedding-pipeline.md); this ticket only fixes that it
lives in the same database and keys back to `(source, session_id, stream_id, seq)`.

### 4. Compression and retention

- **Keep raw indefinitely.** ADR 091's precedent, and here it is stronger: the
  source files rotate to `.jsonl.zst` and are eventually pruned, so our copy
  becomes the only copy. Re-derivation — the whole reason raw exists — depends
  on it.
- **Compress on age**, segmented by `(source, session_id)` and ordered by `seq`,
  so a session's rows land in the same compressed batch and whole-session reads
  stay cheap. The generic filesystem already achieves ~2.5x on this JSONL
  (178 MB logical → 70 MB on disk), which is the floor columnar compression
  should beat.
- **Volume context:** 66,084 records / 178 MB over 77 days ≈ **0.85 GB/year per
  machine, 3.4 GB/year across four**. Small. Nothing here is driven by scale
  pressure; the layout is chosen for query shape and re-derivability.

### 4b. Addendum 2026-09-19 — columns the derivation worker needs

[The derivation worker](derivation-worker.md) settled discovery as an
**anti-join**, which adds requirements here:

- Normalized rows carry **`derivation_version`**; pending is defined against the
  current value, so bumping it re-derives the corpus.
- The anti-join **must be index-backed** on the normalized key — unindexed it
  degrades badly at the 66k-record backfill, which is the very first run.
- A failed normalization still writes a row marked `degraded`, so a poison record
  satisfies the anti-join instead of re-pending forever.
- Raw rows are never written back to. The insert-only grant and hypertable
  compression both depend on that, which is why the Seshat-style marker column on
  raw was rejected.

### 5. Indexes

- PK `(observed_at, source, session_id, stream_id, seq)` — exact addressing.
- Session listing: `(repo, observed_at DESC)` and `(source, session_id)`.
- Kind filtering: `(kind, observed_at)`, serving "every `tool_call` last week".
- Text search: GIN over the normalized text for grep.
- Drift counters: a small table keyed `(source, type, schema_marker)` holding
  unknown- and degraded-record counts, which is what
  [the format research](source-format-stability.md)'s alarm thresholds read.

### 6. Migrations

SeaORM entities and `sea-orm-migration` — **never raw SQL** (project convention),
excepting the `create_hypertable`/compression calls that have no SeaORM
equivalent, exactly as `akhomed-store` already does. Advisory-locked in the
manner of `run_migrations_locked` (`crates/akhomed-store/src/lib.rs:96`), since
more than one machine may start a harvester concurrently.

**Degradation is asymmetric and deliberate:**

- *TimescaleDB absent* → the table is created with the identical PK and simply
  stays plain. Everything works; compression does not. Warn, continue — the
  existing guard pattern.
- *pgvector absent* → exact addressing works, semantic search does not. This must
  **fail loudly at startup** rather than warn, because the read surface advertises
  a capability it could not deliver. A miette diagnostic naming the wrapper image
  is the right shape here.

## Flagged 2026-09-19 — TimescaleDB is not available in the target cluster

[Extension verification](extension-verification.md) queried the live akh-pg
cluster: **`vector` is available; `timescaledb` is not present at all.**

Decision 2 traded "a partly-synthetic column inside the primary key" for columnar
compression. The compression is unavailable, so the trade currently returns
nothing — the design carries `observed_at` in the PK, and therefore carries the
carry-forward determinism requirement, for a benefit that cannot be enabled.

The degradation path written into decision 6 still holds exactly as specified: the
table is created with an identical PK and stays plain, everything works, warn and
continue. So **nothing is broken** — the question is only whether the complexity
is still worth carrying.

**Not reversed unilaterally.** Options and their trade-offs are recorded on
[extension verification](extension-verification.md); the recommendation there is
to drop `observed_at` from the key, which is what session 5's rejected option
would have produced. That is the user's call.

## Flag cleared 2026-09-19 — own cluster, with TimescaleDB

The fleet-wide check found **no cluster has TimescaleDB**
([extension verification](extension-verification.md)), so decision 1's placement
("a separate database in the existing akh-pg cluster") is **superseded**: the
store gets **its own CNPG cluster**, `harvest-pg`, running a custom image with
TimescaleDB 2.25.1 on the CNPG 18 base. That was option 2 of decision 1's
original three, rejected at the time for costing "another cluster to run for ~3
GB/year" — the calculus changed once the alternative became a rolling image swap
on the database the akhs depend on.

**Decision 2 therefore stands unchanged and regains its justification.**
`observed_at` in the primary key buys columnar compression again, so the
partly-synthetic key column is paid for. The forward-only carry-forward rule
remains a frozen contract, for the reasons in
[the derivation worker](derivation-worker.md).

Artifacts: `assets/deploy/Dockerfile`, `assets/deploy/harvest-pg-cluster.yaml`
(validated with `kubectl apply --dry-run=server`), `assets/deploy/README.md`.
Not built, not pushed, not applied.
