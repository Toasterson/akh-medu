# ADR 120 — Per-akh SQLite databases for the local backing store

- **Status:** Accepted (implemented on branch `feat/per-akh-sqlite`; the Roman
  core switches only when the operator runs the CUTOVER.md "Per-akh databases"
  sequence)
- **Date:** 2026-09-29
- **Relates:** [ADR 049](049-relocatable-akh.md) (residency, WAL, snapshot),
  [ADR 113](113-full-akh-always.md) (full akhs on the Mac Roman core),
  [ADR 116](116-lilith-chief-of-staff-guided-rl.md) (Lilith is the fourth
  writer), [ADR 118](118-shared-store-migration-compatibility.md) (schema is a
  deployment boundary)
- **Runbook:** `deploy/roman-mini/CUTOVER.md` §13 "Per-akh databases"

## Context

The Roman core is one `akhomed` hosting janus, minerva, vesta and lilith on a
single SQLite file, `state/core/akh.sqlite3` (`pool_size = 1`,
`snapshot_interval_secs = 30`, `wal_flush_interval_secs = 1`). The core's log
recorded periodic snapshot failures that got worse when Lilith became the
fourth writer, all reported with the ADR-049 text "check whether the
split-brain guard was bypassed". The log and a probe showed three causes, none
a split-brain:

1. **In-process pool starvation** (the most frequent: 143 in the current log).
   Every akh's snapshot, harvest, librarian pass and recall read shares the one
   pooled connection. A snapshot persist took up to 100 s, and the others
   gave up with `Connection pool timed out`.
2. **SQLite write-lock contention that busy_timeout cannot absorb** (36 WAL
   flushes and 2 snapshots). Each akh's WAL writes through its own residency
   connection. `publish_snapshot` and the WAL append both open a deferred
   transaction, read, then write. A connection that already holds a read
   snapshot cannot wait for the write lock: SQLite returns
   `(code: 5) database is locked` at once (measured: 28 µs with a 5 s
   busy_timeout), or `SQLITE_BUSY_SNAPSHOT` (517) when another writer committed
   in between.
3. An unrelated `UNIQUE constraint failed: akhomed_sleep_metrics` (162 in the
   current log). It is not addressed here and needs its own work item.

PostgreSQL has neither problem: rows are locked individually and the pool has
several connections.

## Decision

**Under SQLite, an opt-in `[database] layout = "per_akh"` puts each akh's
tables in their own file with one connection. Host-level tables stay in the
configured file. PostgreSQL and the default `"shared"` layout are unchanged.**

1. **Table split** (`akhomed_store::layout::TABLE_HOMES`, the single source of
   truth):
   - **Host** (`[database] url`): `workspaces_workspace`,
     `workspaces_workspace_seed_pack`, `workspaces_membership`,
     `workspaces_active_microtheories`, `akhomed_schema_registry`,
     `seaql_migrations`. These tables are shared across akhs (membership is
     many-to-many; packs and activations are workspace/host-wide).
   - **Akh** (`<host dir>/akh.<name>.sqlite3`): `akhomed_akh` (identity,
     `residency_epoch`, `last_seen`), `akhomed_akh_state`, `akhomed_symbols`,
     `akhomed_mutation_log`, `akhomed_episodes`, `akhomed_provenance`,
     `akhomed_provenance_snapshot`, every cognitive table (keyed by `akh_id`),
     plus `akhomed_facts` and `akhomed_akh_config` (keyed by the akh's own
     `workspace_id`) and `akhomed_provenance_input` (through its
     `akhomed_provenance` row). The fencing epoch sits in the same file as the
     rows it fences, so the epoch check in `publish_snapshot` and the WAL append
     stays one local transaction.
2. **Addressing by akh name.** Both store ids are UUIDv5 of the workspace
   name (`akh_id_for`, `workspace_id_for`, now owned by the store crate). An
   akh file holds rows under either id, so only the name identifies the whole
   file. Names are already restricted to `[a-z0-9][a-z0-9_-]{0,62}`
   (`AkhStoreName` re-checks this), which is safe as a file-name component. An
   operator can also read `state/core/` directly. A name/id mismatch in the
   source is refused by the split.
3. **One connection per akh file.** The akh store is a one-connection pool.
   Under `per_akh`, the residency session (`StablePgSession`) shares that
   connection instead of opening its own. Snapshot, WAL, harvest, librarian
   and recall reads of one akh therefore queue on the pool (acquire timeout =
   `connect_timeout_secs`), and no transaction can hit the deferred-upgrade
   `SQLITE_BUSY`. Different akhs never share a lock.
4. **Residency lock files stay beside the host file**
   (`<host>.akh-<uuid>.lock`), in the file the deployment configures. The split
   tool holds them for the whole copy.
5. **Migrations per file.** Every file gets the full schema from the one
   `Migrator`, under its own `.migrate.lock` (`run_migrations_locked`). A table
   whose home is elsewhere just stays empty. A per-akh file is migrated when
   the daemon first opens it, and by `_akh/migrate` for every opened akh. There
   is no new migration: an ADR-118 schema bump would also apply to PostgreSQL.
6. **Fail closed.** `per_akh` with a non-SQLite URL is refused at startup
   (`akhomed_store::layout::requires_sqlite`). A per-akh daemon whose host file
   still holds the akh's `akhomed_akh` row (an unsplit shared file) refuses to
   open that akh (`akhomed_store::layout::unmigrated`) instead of starting it
   from an empty file.
7. **Offline split tool:** `akhomed store split-per-akh --source <shared>
   --host <new host> [--dry-run] [--json] [--allow-orphans] [--evidence P]`.
   It is an `akhomed` subcommand, so the deployed stable-path binary can run it
   with no second artifact. The tool:
   - opens the source read-only (`immutable` when it has no pending WAL) and
     refuses while any lock file beside the source is held;
   - refuses existing targets, unknown tables, migrations this build does not
     know, and rows that belong to no akh (`--allow-orphans` leaves them in the
     source and lists them in the evidence);
   - copies value-for-value (SQLite storage classes preserved) in one
     transaction per target file, then re-reads every table and compares the
     row count and a SHA-256 content digest per table per akh;
   - hashes the source (and its `-wal`) before and after, and writes a 0600
     evidence JSON (`akhomed.store.split_per_akh.v1`);
   - removes the targets it created on any failure, and never modifies or
     deletes the source.
8. **Diagnostics name the real cause.** `SQLITE_BUSY`/`SQLITE_LOCKED`
   (classified by the driver's typed error code, never by message text) become
   `StoreError::LockContention` (`akhomed_store::sqlite_busy`). Its help names
   the per-akh layout and busy_timeout, and says when busy_timeout cannot help.
   On SQLite, the periodic-snapshot log reports "SQLite lock contention (not a
   split-brain …)" for busy errors and pool timeouts. Residency refusals name
   the host-local lock file instead of "another host". The ADR-049 split-brain
   wording stays exactly as before for PostgreSQL, where advisory-lock
   residency makes it true.
9. **Renderer opt-in: a switch file.** `deploy/roman-mini/configure_core.py`
   renders the per-akh layout only when `config/per-akh-databases.json`
   (`{"evidence": "<split evidence>"}`) exists. The render also requires a
   verified split of `state/core/akh.sqlite3` into `state/core/host.sqlite3`
   and an existing host file. The switch then moves both `[database] url` and
   the plist's `DATABASE_URL`, which overrides the url, to `host.sqlite3`. A
   switch file follows the existing operator switch pattern
   (`janus-delegation.json`, `lilith.json`), and gating on the evidence means a
   render cannot point the core at an empty host file.

## Consequences

- One akh's slow snapshot no longer delays or fails another akh's. On the
  measured live copy, the split took 4.3 s (debug build) for 126 k rows and
  was verified.
- Within one akh, writers wait on its connection instead of failing
  immediately. A persist longer than `connect_timeout_secs` still delays that
  akh's WAL flush (retried next tick; the buffer holds the rows).
- There is no reverse split. A rollback to the shared file loses what the
  akhs wrote after the cutover, unless it is recovered by hand from the
  per-akh files, which the runbook keeps.
- A binary without ADR 120 ignores `layout` (the host config has no
  unknown-key refusal) and would treat `host.sqlite3` as an empty shared
  store. The runbook requires the ADR-120 binary first, and the renderer
  gates on the evidence the new binary writes.
- Every per-akh file carries empty copies of the host tables and vice versa;
  the cost is a few pages per file.
- Relocation across hosts is unaffected: SQLite was never relocatable.

## Alternatives considered

- **`BEGIN IMMEDIATE` for every SQLite write transaction.** This would fix
  cause 2 inside one file, but SeaORM 1.1 exposes no transaction mode for
  SQLite, and cause 1 (one pool for four akhs) would remain.
- **A larger shared pool.** More connections to one file turn pool timeouts
  into cause 2.
- **A URL template (`akh.{akh}.sqlite3`).** This is less explicit than a
  typed `layout`, and a binary without the feature would open a literal
  `{akh}` file.
- **Addressing by UUID.** The akh id and workspace id differ, so either one
  names only part of the file's rows, and UUID file names are not readable in
  an operator runbook.
