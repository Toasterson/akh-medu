# ADR 077 — Postgres backing-store connection resilience (retry + runtime reconnect)

> Date: 2026-06-23
> Status: Accepted (implemented)
> Related: ADR 049 (relocatable akh — Postgres-backed truth), ADR 072 (normalized
>          akhomed-store), Phase 38b/38c (mutation WAL + periodic snapshot),
>          `src/wal.rs` (buffered flusher), `src/bin/akhomed.rs`
>          (`init_postgres`, `ServerState::spawn_pg_reconnector`),
>          memory: `macos-launchd-local-network-tcc`

## Context

Phase 38 makes Postgres the durable truth store: each akh's mutations stream to
`akhomed_mutation_log` (WAL, flushed every ~2s) and a full relational snapshot is
written periodically (every ~300s). On startup `init_postgres` opened the pool
**once**; on failure it logged *"PostgreSQL unavailable — falling back to
local-only mode"* and set `pg_pool = None` for the entire process lifetime.

Two real failures exposed this as too brittle (observed 2026-06-23):

1. **Boot-time race.** launchd starts `akhomed` at login, often *before* the LAN
   route to `akh-pg` is up. The single connect attempt failed with
   `Network is unreachable (os error 51)`, the daemon went local-only, and an
   entire working session's mutations (a full OpenClaw self-import — 38 journals
   + 7 skills) lived only in RAM. `akh akh snapshot` then failed with *"no
   Postgres backing store configured"* because the runtime pool was `None`.
   The data was unrecoverable to PG without re-running the import.

2. **No path back.** Even though the network recovered seconds later (a plain
   client could reach `akh-pg` over mTLS), the running daemon had no way to
   reconnect. `config set` does not rebuild the pool — it is captured once at
   construction.

The WAL itself is already resilient to a *mid-session* drop once wired: a failed
flush re-buffers the batch (front-inserted) and retries next tick, bounded by
`wal_max_buffer_mb` (`src/wal.rs`). The gap is strictly the **never-connected**
case: no pool ⇒ no WAL wired ⇒ mutations are dropped, not buffered.

## Decisions

### 1. Bounded startup retry/backoff
`init_postgres` now retries the connect (`PG_STARTUP_ATTEMPTS = 5`, backoff
1→2→4→8s) before falling back. This alone absorbs the common boot race, so the
daemon usually comes up `postgres=true` without any akh ever loading local-only.

### 2. Swappable pool slot
`ServerState.pg_pool` changes from `Option<Arc<DatabaseConnection>>` to
`std::sync::RwLock<Option<Arc<DatabaseConnection>>>`. A sync accessor
`pg_pool() -> Option<Arc<…>>` returns a cloned snapshot (poison-tolerant via
`into_inner`); all data-plane call sites read through it, so the pool can be
swapped in at runtime with no API change to the ~12 consumers. `std::sync::RwLock`
(not `tokio`) keeps the accessor synchronous — the lock is never held across an
`.await`.

### 3. Background reconnector that re-enables durability without restart
When startup ends local-only but a `[postgres]` section is configured, `main`
spawns `spawn_pg_reconnector`. It retries with capped backoff (2→…→60s),
re-reading config each attempt (so a `config set` can fix a bad URL live). On
success it swaps the live pool in and calls `rewire_after_reconnect`, which for
**every already-loaded akh**:

1. wires the WAL (`wire_wal`) so subsequent mutations stream to PG, then
2. writes a full snapshot (`snapshot_workspace`) so the state accumulated **while
   local-only** is materialised into the relational tables — nothing is lost.

The task self-terminates once connected. A restart is no longer required to
recover from a boot-time miss.

## Why not

- **Restart-to-recover.** A supervisor that restarts the process once PG is
  reachable would wake each akh from the *last snapshot*, discarding the
  in-memory delta — exactly the data loss we are fixing. Rejected.
- **`ArcSwapOption`.** Marginally faster lock-free reads, but `arc-swap` would
  become a direct dep and the ownership ergonomics are identical. `RwLock` is
  sufficient for a low-frequency slot. (It is already a transitive dep.)
- **Hot-reload via `config set`.** Considered, but reconnection should be
  automatic and continuous, not operator-driven.

## Consequences

- A daemon that boots before its network is up self-heals: it logs retries, then
  `reconnected to PostgreSQL backing store — re-enabling durability` and
  snapshots each loaded akh.
- Operational gotcha retired: no more "kickstart twice after reboot."
- Snapshot-vs-WAL nuance (learned during recovery): `akhomed_symbols`/`_facts`
  reflect the periodic **snapshot**, while `akhomed_mutation_log` reflects live
  deltas. Tooling that verifies persistence should consult the WAL (or force a
  snapshot) rather than assume the symbol tables are current.
- Mid-session PG loss after a successful wire remains handled by the WAL's
  re-buffer/retry; this ADR does not change that path.
