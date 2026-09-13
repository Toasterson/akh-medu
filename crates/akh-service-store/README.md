# akh-service-store

Transactional durable records for independently owned Akh services. One
`Store` per daemon (`akhsessd`, `akhrtd`, `akhexecd`, `akhsbxd`, `akheventd`),
opened once at startup and shared behind an `Arc`.

The public API — `Store::get`/`Store::scan`, `Store::transaction` and the
`Transaction::get`/`put`/`delete`/`scan` it hands to the closure, and the
`key(namespace, parts)` helper — is identical regardless of which backend a
`Store` was opened with. No call site needs to know or care.

## Backends

### redb (default)

`Store::open(path)` opens (creating if needed) a single redb file on local
disk. This is the original implementation; every test in this crate and in
`akh-session-services` uses it, and every deployment predating the Postgres
backend still does.

In production this file lived on an `openebs-hostpath` PVC — on the Raspberry
Pi cluster, a PVC backed by an SD card. I/O pressure there stalls the owning
daemon's `/healthz` and can wedge reconciliation, which is why the Postgres
backend below exists.

### Postgres

`Store::open_postgres(PostgresConfig { url, schema, kind, legacy_redb_path })`
opens a store against a Postgres cluster instead. Configured from
`akh-session-services` via the daemon's config file:

```json
{
  "store": {
    "kind": "postgres",
    "url_env": "AKH_STORE_URL",
    "schema": "akh_sessions"
  }
}
```

- `url_env` names the environment variable holding the `postgres://`
  connection string. **The URL is never read from the config file** — only
  the environment, so credentials never need to appear in a
  SOPS-decrypted-or-not config. In the homelab cluster this is
  `AKH_STORE_URL=postgres://akhomed:<pw>@akh-pg-rw.akh-medu.svc.cluster.local:5432/akhomed?sslmode=require`
  (CloudNativePG serves TLS with a cluster CA the pods don't have mounted, so
  `sslmode=require` — encrypted, not certificate-verified — is intentional,
  not a shortcut).
- `schema` is a Postgres schema (default `akh_sessions`) the store creates
  (`CREATE SCHEMA IF NOT EXISTS`) and owns a single table in
  (`akh_store(kind, key, value, updated_at)`, primary key `(kind, key)`). The
  database itself (`akhomed`) is shared with the `akhomed` daemon; every
  object this store creates lives inside its own schema so the two never
  collide.
- Every daemon kind shares that one table, partitioned by the `kind` column
  (`"akhsessd"`, `"akhrtd"`, ...), rather than getting its own table — the
  schema is intentionally as small as the KV model it replaces.

Absent a `store` section, a daemon keeps using its redb file under
`data_dir` exactly as before — this is opt-in, not a breaking change.

## Design notes (Postgres backend)

See the doc comments at the top of `src/postgres.rs` for the full reasoning;
summarized here:

- **Sync facade over an async driver.** The public API is synchronous
  because every existing call site is. sqlx is async-only, so a small
  dedicated multi-thread Tokio runtime (not the caller's ambient runtime,
  and never called into recursively) drives every query to completion
  before the calling `fn` returns. This works identically whether the
  caller is a `#[tokio::main]` daemon or a plain `#[test]` with no runtime
  at all.
- **Locking.** redb allows one writer at a time per file, holding a
  consistent snapshot for the transaction's duration. Postgres emulates this
  per `kind` with `pg_advisory_xact_lock(hashtextextended(kind, 0))`, taken
  at the start of every `Store::transaction` and released automatically on
  commit or rollback.
- **Cache.** Each Postgres-backed `Store` keeps a process-local
  read-through `HashMap` cache, populated on a `Store::get` miss and written
  through only after a transaction commits. A rolled-back transaction's
  writes are simply never handed to the cache — there is nothing to
  invalidate. Reads *inside* a transaction always go straight to Postgres
  (never the cache), so they see the transaction's own uncommitted writes
  via ordinary MVCC. **This is only safe because production runs a single
  process per store `kind` at a time** (a `Deployment` with `strategy:
  Recreate`); if that ever changes, the cache needs cross-process
  invalidation (e.g. `LISTEN`/`NOTIFY`) to stay correct.
- **Prefix scans.** `key BYTEA` with primary key `(kind, key)` supports an
  efficient `key >= $lo AND key < $hi` range for `scan(prefix)`; `$hi` is the
  prefix's byte-lexicographic upper bound (`prefix_upper_bound` in
  `postgres.rs`), falling back to an open-ended scan (filtered client-side by
  `starts_with`, exactly like the redb backend already does) when no finite
  upper bound exists.

## One-shot redb → Postgres migration

`Store::open_postgres` always checks, before returning, whether this `kind`
has already recorded a migration (a reserved marker key, `\0` +
`b"akh_store_migrated_from_redb"` — never producible by `key()`, which only
emits printable ASCII, so it can never collide with an application key).

- If the marker is already present, nothing happens — the store never
  re-imports, even if the legacy redb file still exists or changes on disk.
- If it is absent and `legacy_redb_path` points at a redb file, every
  `(key, value)` pair in that file is imported once (via `INSERT ... ON
  CONFLICT DO NOTHING`, inside the same advisory-locked transaction as the
  marker write, so a crash mid-import can safely retry), and the count is
  logged (`tracing::info!`).
- If it is absent and there is no legacy file (a fresh install), the marker
  is written anyway with `imported = 0`, so the check never runs twice.

This lets a live deployment switch a daemon from redb to Postgres without
losing in-flight sessions: point `store.kind` at `postgres`, keep `data_dir`
where it already is, and the first startup after the switch carries
everything across exactly once.

## Testing

- `cargo test -p akh-service-store` runs the redb-backed suite plus
  Postgres-backend unit tests (prefix bound math, cache semantics) that need
  no database.
- The Postgres integration suite (`tests/postgres.rs`) is `#[ignore]`d by
  default and gated on `AKH_TEST_DATABASE_URL`:

  ```sh
  docker run -d --rm --name akh-store-pg -p 55433:5432 \
    -e POSTGRES_PASSWORD=x -e POSTGRES_USER=akh -e POSTGRES_DB=akh \
    postgres:18-alpine
  AKH_TEST_DATABASE_URL=postgres://akh:x@127.0.0.1:55433/akh \
    cargo test -p akh-service-store --test postgres -- --ignored
  docker stop akh-store-pg
  ```
