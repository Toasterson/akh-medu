# ADR 052 — Heka's Hall: Postgres+pgvector Long-Term Memory Foundation

- **Status:** Accepted
- **Date:** 2026-04-27
- **Deciders:** Tecton (autonomous); Till to ratify
- **Inputs:** [Phase 42 plan](../plans/2026-04-27-phase42-hekas-hall.md); Seshat Phase 37 pattern (`crates/seshat/`); research `docs/ai/research/2026-04-26-otlp-observability.md` (shared infra context)
- **Relates:** Phase 42a (this ADR's foundation), Phase 42i (contradict tool, later), Phase 42j (metrics, later)

## Context

Heka is the long-term-memory subsystem for akh-medu. Named after the Egyptian god of magic and writing — the personification of the creative power of words — it stores timestamped "drawers" (memory entries) per akh, with project-scoped bindings and pgvector-backed semantic retrieval.

The existing Seshat crate (`crates/seshat/`, Phase 37) established the pattern for Postgres+pgvector in this codebase: SeaORM 1.x entities, raw-SQL migrations for pgvector types, miette diagnostics, and a `seshd` daemon. Heka mirrors that pattern deliberately — same dependency set, same migration strategy, same error-handling conventions — so that the two services share operational muscle and the `heka_*` tables can live in the same database as `pa_*` if desired.

## Decision

**Stand up `crates/heka/` as a Postgres+pgvector foundation mirroring Seshat, with 5 tables and an HNSW cosine index, behind a `hekad migrate` CLI.**

### Schema (5 tables)

| Table                     | Purpose                                          |
|---------------------------|--------------------------------------------------|
| `heka_akhs`               | Registered akh instances (per-workspace identity)|
| `heka_project_bindings`   | Akh ↔ project associations (unique per pair)     |
| `heka_drawers`            | Top-level memory entries (a "drawer" of memories) |
| `heka_drawer_chunks`      | Text chunks within a drawer (for embedding)       |
| `heka_drawer_embeddings`  | 384-dim pgvector embeddings + HNSW cosine index   |

### Key choices

1. **pgvector `vector(384)` + HNSW `vector_cosine_ops`** — matches Seshat's Phase 37 migration `m20260425_000001_switch_to_hnsw`. HNSW over IVFFlat because:
   - No training data needed — works correctly on small/empty tables (cold start).
   - Better recall at low data volumes.
   - Incremental updates are cheap (no re-clustering).
   - pgvector defaults (`m=16`, `ef_construction=64`); tuning deferred to Phase 42j.

2. **SeaORM 1.x entities** — same feature set as Seshat: `sqlx-postgres`, `runtime-tokio-rustls`, `macros`, `with-uuid`, `with-chrono`, `with-json`. The `embedding` column maps to `String` in SeaORM (pgvector's `vector` type is not natively supported); actual vector queries use parameterized SQL, following `crates/seshat/src/query.rs`.

3. **Raw-SQL migration executed via `execute_unprepared`** — SeaORM's `ColumnDef` cannot express `vector(384)` or HNSW indexes, so the migration reads `migrations/20260427_000001_init.sql` and runs it raw. This mirrors Seshat's `m20260409_000001_create_seshat.rs`.

4. **`hekad migrate` CLI** — loads `heka.toml`, connects via SeaORM, runs `Migrator::up`. Matches `seshd migrate`. The full `hekad` daemon (MCP server, HTTP API) is Phase 42b; this slice only ships the migration runner.

5. **`HekaConfig` parser** — TOML config with env-var overrides (`DATABASE_URL`, `HEKA_BIND`). Same pattern as `SeshatConfig`.

6. **`HekaError` with miette diagnostics** — every variant carries a diagnostic code and help string. Mirrors `SeshatError`.

### Idempotency

All DDL uses `IF NOT EXISTS`. Re-running `hekad migrate` is safe. The `CREATE EXTENSION vector` requires superuser on the cluster; the migration enables it per-database.

## Consequences

- **Positive:** Same operational pattern as Seshat — shared tooling, shared debugging muscle, optional shared database. HNSW gives correct recall from day one (no IVFFlat cold-start trap). The `hekad migrate` CLI is independently runnable for CI/CD.
- **Negative:** The `embedding` column is `String` in SeaORM — vector I/O requires raw SQL (Phase 42b+). pgvector must be installed cluster-side before `hekad migrate` runs.
- **Neutral:** 384 dimensions matches `all-MiniLM-L6-v2` (the default embedding model). A model swap requires a schema migration to change the dimension.

## Alternatives considered

- **redb (local key-value store)** — rejected. Heka needs SQL-grade querying (project bindings, tag filtering) and pgvector's HNSW index, which redb cannot provide. A previous (wrong) Phase 42i attempt used redb; it was replaced entirely.
- **IVFFlat instead of HNSW** — rejected. Seshat learned this lesson in Phase 37: IVFFlat returns zero results on small tables because `lists=100` with `probes=1` visits empty clusters. HNSW has no training-data failure mode.
- **Separate database from Seshat** — deferred. The schema is designed to coexist with `pa_*` tables in the same database, but `heka.toml` can point to an isolated DB if operational needs change.

## Acceptance criteria (Phase 42a)

- `cargo test -p heka` passes
- `psql -c "\dt heka_*"` shows 5 tables
- HNSW index visible in `\di+`
- `cargo run --bin hekad migrate` creates schema on test DB