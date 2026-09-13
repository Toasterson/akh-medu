//! Postgres-backed implementation of the `Store`/`Transaction` API.
//!
//! # Sync facade over an async driver
//!
//! `Store`'s public API is synchronous (plain `fn`s, no `.await`), because
//! the redb backend it originally shipped with is synchronous and every call
//! site in `akh-session-services` invokes it that way, often from inside a
//! `#[tokio::main]` daemon. sqlx is async-only, so this module bridges the
//! two with a small, dedicated, lazily-started multi-thread Tokio runtime
//! (see [`pg_runtime`]) that is *not* the caller's ambient runtime.
//!
//! Every Postgres operation below is exactly one top-level
//! `pg_runtime().block_on(...)` call that runs to completion before
//! returning. None of these calls are ever nested inside one another (a
//! transaction's `get`/`put`/`scan` each issue their own independent
//! `block_on`, only after the transaction's opening `block_on` has already
//! returned control to plain sync code). That is deliberate: Tokio panics
//! with "Cannot start a runtime from within a runtime" if `block_on` is
//! reentered on a thread that is already inside a `block_on`/task poll for
//! *that same runtime* — using a private runtime instance side-steps the
//! ambient-runtime version of that hazard, and never calling `block_on`
//! recursively side-steps the same-runtime version. The alternative
//! (`tokio::task::block_in_place` + `Handle::current().block_on`) would only
//! work when the caller happens to be on a multi-thread runtime, and would
//! panic outright in a plain `#[test]` with no runtime at all; the
//! dedicated-runtime approach works identically in both places, which is why
//! it was chosen over that alternative.
//!
//! # Locking model
//!
//! redb allows only one write transaction at a time per database file, and
//! that transaction sees a consistent snapshot for its whole duration. To
//! give `Store::transaction` the same "one writer at a time, per store kind"
//! semantics on Postgres (where many connections share one cluster), every
//! transaction opens by taking `pg_advisory_xact_lock` keyed by the store's
//! `kind` (e.g. `"akhsessd"`). The lock is scoped to the SQL transaction and
//! is released automatically on commit or rollback, so it cannot leak. This
//! serializes concurrent `Store::transaction` calls for the same kind
//! (whether from one process or several), matching the redb behavior it
//! replaces, without requiring per-row `SELECT ... FOR UPDATE` bookkeeping.
//!
//! # Cache
//!
//! Each Postgres-backed `Store` keeps an in-memory `HashMap` read cache. It
//! is populated on a `Store::get` cache miss and written through after a
//! `Store::transaction` commits successfully; a rolled-back transaction
//! never touches it, so there is nothing to invalidate on rollback beyond
//! "don't apply the pending writes." Reads *inside* a transaction always go
//! straight to Postgres (never the cache), so they see the transaction's own
//! uncommitted writes via ordinary Postgres MVCC. This cache is safe only
//! because production deploys run a single process per store `kind` at a
//! time (`Deployment` with `strategy: Recreate`); if that ever changes to
//! multiple concurrent readers/writers per kind, the cache would need
//! invalidation across processes (e.g. `LISTEN`/`NOTIFY`) to stay correct.
use crate::{Fault, Result, failure};
use serde::{Serialize, de::DeserializeOwned};
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::{
    cell::RefCell,
    collections::HashMap,
    path::{Path, PathBuf},
    sync::RwLock,
};

/// Never produced by [`crate::key`] (which only ever emits printable ASCII),
/// so this can never collide with an application key.
const MIGRATION_MARKER_KEY: &[u8] = b"\0akh_store_migrated_from_redb";

/// Inputs needed to open (and, if applicable, one-shot migrate into) a
/// Postgres-backed [`crate::Store`]. The connection URL is expected to have
/// already been resolved from an environment variable by the caller — this
/// crate never reads environment variables itself.
pub struct PostgresConfig {
    pub url: String,
    pub schema: String,
    /// Logical store name (the daemon kind, e.g. `"akhsessd"`). Rows for
    /// every kind sharing a Postgres database can live in the same table,
    /// partitioned by this column.
    pub kind: String,
    /// Path to the legacy per-kind redb file, if one might still exist from
    /// before the switch to Postgres. When present and the Postgres store
    /// has not already recorded a migration for this `kind`, every key is
    /// imported once.
    pub legacy_redb_path: Option<PathBuf>,
}

pub(crate) struct PostgresInner {
    pool: PgPool,
    /// Fully-quoted `"schema".akh_store`, precomputed once at open time.
    table: String,
    kind: String,
    cache: ReadCache,
}

/// Process-local read-through cache for a single Postgres-backed store
/// (i.e. a single `kind`). Deliberately not shared across kinds or
/// processes; see the module docs for why that is safe here. Kept as its
/// own type so its commit/rollback semantics can be unit tested without a
/// database.
#[derive(Default)]
struct ReadCache(RwLock<HashMap<String, Vec<u8>>>);

impl ReadCache {
    fn get(&self, key: &str) -> Option<Vec<u8>> {
        self.0
            .read()
            .expect("akh-service-store Postgres cache lock poisoned")
            .get(key)
            .cloned()
    }

    fn insert(&self, key: String, value: Vec<u8>) {
        self.0
            .write()
            .expect("akh-service-store Postgres cache lock poisoned")
            .insert(key, value);
    }

    /// Applies a transaction's staged writes/deletes after it has committed.
    /// Never call this for a rolled-back transaction — that is the entire
    /// "invalidation" story: a rollback simply never reaches this method, so
    /// the cache keeps whatever it held before the transaction began.
    fn apply_committed(&self, pending: Vec<(String, Option<Vec<u8>>)>) {
        let mut cache = self
            .0
            .write()
            .expect("akh-service-store Postgres cache lock poisoned");
        for (key, value) in pending {
            match value {
                Some(bytes) => {
                    cache.insert(key, bytes);
                }
                None => {
                    cache.remove(&key);
                }
            }
        }
    }
}

pub(crate) struct PgTxState<'a> {
    inner: &'a PostgresInner,
    tx: RefCell<Option<sqlx::Transaction<'static, sqlx::Postgres>>>,
    pending: RefCell<Vec<(String, Option<Vec<u8>>)>>,
}

/// A small, dedicated Tokio runtime used only to drive sqlx futures to
/// completion synchronously. See the module docs for why this is not the
/// caller's ambient runtime.
fn pg_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("akh-service-store-pg")
            .enable_all()
            .build()
            .expect("failed to start the dedicated Postgres store runtime")
    })
}

/// A Postgres identifier is restricted to `[a-z_][a-z0-9_]*` here (stricter
/// than Postgres itself allows) because `schema` is interpolated directly
/// into DDL/DML text — sqlx has no bind-parameter form for identifiers. Every
/// other value (kind, key, value) is always passed as a bound parameter.
fn validate_identifier(name: &str) -> Result<()> {
    let mut chars = name.chars();
    let starts_ok = chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c == '_');
    let rest_ok = chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    if !starts_ok || !rest_ok || name.len() > 63 {
        return Err(Fault::invalid(format!(
            "'{name}' is not a valid Postgres schema identifier"
        )));
    }
    Ok(())
}

/// Computes the exclusive upper bound of a byte-string prefix range, i.e.
/// the smallest byte string that is greater than every string starting with
/// `prefix`. Returns `None` when no finite upper bound exists (the prefix is
/// empty, or every byte is already `0xFF`); callers must then scan
/// open-ended and rely on a `starts_with` filter for correctness.
pub(crate) fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut upper = prefix.to_vec();
    while let Some(&last) = upper.last() {
        if last == 0xFF {
            upper.pop();
        } else {
            *upper.last_mut().expect("checked non-empty above") += 1;
            return Some(upper);
        }
    }
    None
}

async fn ensure_schema(pool: &PgPool, schema: &str, table: &str) -> sqlx::Result<()> {
    sqlx::query(&format!("CREATE SCHEMA IF NOT EXISTS \"{schema}\""))
        .execute(pool)
        .await?;
    sqlx::query(&format!(
        "CREATE TABLE IF NOT EXISTS {table} (\
            kind TEXT NOT NULL, \
            key BYTEA NOT NULL, \
            value BYTEA NOT NULL, \
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), \
            PRIMARY KEY (kind, key))"
    ))
    .execute(pool)
    .await?;
    Ok(())
}

async fn advisory_lock(
    tx: &mut sqlx::Transaction<'static, sqlx::Postgres>,
    kind: &str,
) -> sqlx::Result<()> {
    sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
        .bind(kind)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

async fn migrate_from_redb_if_needed(
    pool: &PgPool,
    table: &str,
    kind: &str,
    legacy_path: Option<&Path>,
) -> sqlx::Result<()> {
    let mut tx = pool.begin().await?;
    advisory_lock(&mut tx, kind).await?;
    let already_migrated: Option<i32> = sqlx::query_scalar(&format!(
        "SELECT 1 FROM {table} WHERE kind = $1 AND key = $2"
    ))
    .bind(kind)
    .bind(MIGRATION_MARKER_KEY)
    .fetch_optional(&mut *tx)
    .await?;
    if already_migrated.is_some() {
        tx.rollback().await?;
        return Ok(());
    }
    let mut imported: i64 = 0;
    if let Some(path) = legacy_path
        && path.is_file()
    {
        match crate::redb_backend::scan_all_raw(path) {
            Ok(entries) => {
                for (key, value) in entries {
                    sqlx::query(&format!(
                        "INSERT INTO {table} (kind, key, value) VALUES ($1, $2, $3) \
                         ON CONFLICT (kind, key) DO NOTHING"
                    ))
                    .bind(kind)
                    .bind(key.as_bytes())
                    .bind(&value)
                    .execute(&mut *tx)
                    .await?;
                    imported += 1;
                }
            }
            Err(error) => {
                tracing::error!(
                    kind,
                    path = %path.display(),
                    %error,
                    "could not read legacy redb store for one-shot Postgres import; leaving it unmigrated"
                );
            }
        }
    }
    sqlx::query(&format!(
        "INSERT INTO {table} (kind, key, value) VALUES ($1, $2, $3) \
         ON CONFLICT (kind, key) DO NOTHING"
    ))
    .bind(kind)
    .bind(MIGRATION_MARKER_KEY)
    .bind(imported.to_string().into_bytes())
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    tracing::info!(kind, imported, "imported legacy redb store into Postgres");
    Ok(())
}

async fn fetch_value<'e, E>(executor: E, table: &str, kind: &str, key: &[u8]) -> sqlx::Result<Option<Vec<u8>>>
where
    E: sqlx::Executor<'e, Database = sqlx::Postgres>,
{
    sqlx::query_scalar(&format!("SELECT value FROM {table} WHERE kind = $1 AND key = $2"))
        .bind(kind)
        .bind(key)
        .fetch_optional(executor)
        .await
}

async fn fetch_scan<'e, E>(
    executor: E,
    table: &str,
    kind: &str,
    prefix: &[u8],
) -> sqlx::Result<Vec<(Vec<u8>, Vec<u8>)>>
where
    E: sqlx::Executor<'e, Database = sqlx::Postgres>,
{
    let upper = prefix_upper_bound(prefix);
    let rows: Vec<(Vec<u8>, Vec<u8>)> = sqlx::query_as(&format!(
        "SELECT key, value FROM {table} \
         WHERE kind = $1 AND key >= $2 AND ($3::bytea IS NULL OR key < $3) \
         ORDER BY key ASC"
    ))
    .bind(kind)
    .bind(prefix)
    .bind(upper)
    .fetch_all(executor)
    .await?;
    // Server-side range narrows the scan; this is the same defense-in-depth
    // `starts_with` check the redb backend already does after `range(prefix..)`.
    Ok(rows
        .into_iter()
        .filter(|(key, _)| key.starts_with(prefix))
        .collect())
}

impl PostgresInner {
    pub(crate) fn open(config: PostgresConfig) -> Result<Self> {
        validate_identifier(&config.schema)?;
        let table = format!("\"{}\".akh_store", config.schema);
        let pool = pg_runtime()
            .block_on(PgPoolOptions::new().max_connections(5).connect(&config.url))
            .map_err(failure)?;
        pg_runtime()
            .block_on(ensure_schema(&pool, &config.schema, &table))
            .map_err(failure)?;
        pg_runtime()
            .block_on(migrate_from_redb_if_needed(
                &pool,
                &table,
                &config.kind,
                config.legacy_redb_path.as_deref(),
            ))
            .map_err(failure)?;
        Ok(Self {
            pool,
            table,
            kind: config.kind,
            cache: ReadCache::default(),
        })
    }

    pub(crate) fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        if let Some(bytes) = self.cache.get(key) {
            return serde_json::from_slice(&bytes).map(Some).map_err(failure);
        }
        let row = pg_runtime()
            .block_on(fetch_value(&self.pool, &self.table, &self.kind, key.as_bytes()))
            .map_err(failure)?;
        match row {
            Some(bytes) => {
                self.cache.insert(key.to_string(), bytes.clone());
                serde_json::from_slice(&bytes).map(Some).map_err(failure)
            }
            None => Ok(None),
        }
    }

    pub(crate) fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        let rows = pg_runtime()
            .block_on(fetch_scan(&self.pool, &self.table, &self.kind, prefix.as_bytes()))
            .map_err(failure)?;
        rows.into_iter()
            .map(|(_, value)| serde_json::from_slice(&value).map_err(failure))
            .collect()
    }

    pub(crate) fn begin(&self) -> Result<PgTxState<'_>> {
        let tx = pg_runtime()
            .block_on(async {
                let mut tx = self.pool.begin().await?;
                advisory_lock(&mut tx, &self.kind).await?;
                Ok::<_, sqlx::Error>(tx)
            })
            .map_err(failure)?;
        Ok(PgTxState {
            inner: self,
            tx: RefCell::new(Some(tx)),
            pending: RefCell::new(Vec::new()),
        })
    }

    pub(crate) fn finish<T>(&self, state: PgTxState<'_>, result: Result<T>) -> Result<T> {
        let sqlx_tx = state
            .tx
            .into_inner()
            .expect("transaction handle consumed twice");
        match result {
            Ok(value) => {
                pg_runtime().block_on(sqlx_tx.commit()).map_err(failure)?;
                self.cache.apply_committed(state.pending.into_inner());
                Ok(value)
            }
            Err(err) => {
                // Best-effort: the connection returning to the pool without an
                // explicit commit already rolls back on drop, but doing it
                // explicitly surfaces failures instead of swallowing them.
                let _ = pg_runtime().block_on(sqlx_tx.rollback());
                Err(err)
            }
        }
    }
}

impl PgTxState<'_> {
    pub(crate) fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let mut guard = self.tx.borrow_mut();
        let tx = guard.as_mut().expect("transaction already finished");
        let row = pg_runtime()
            .block_on(fetch_value(&mut **tx, &self.inner.table, &self.inner.kind, key.as_bytes()))
            .map_err(failure)?;
        row.map(|bytes| serde_json::from_slice(&bytes).map_err(failure))
            .transpose()
    }

    pub(crate) fn put<T: Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        let bytes = serde_json::to_vec(value).map_err(failure)?;
        {
            let mut guard = self.tx.borrow_mut();
            let tx = guard.as_mut().expect("transaction already finished");
            pg_runtime()
                .block_on(
                    sqlx::query(&format!(
                        "INSERT INTO {} (kind, key, value, updated_at) VALUES ($1, $2, $3, now()) \
                         ON CONFLICT (kind, key) DO UPDATE SET value = excluded.value, updated_at = now()",
                        self.inner.table
                    ))
                    .bind(&self.inner.kind)
                    .bind(key.as_bytes())
                    .bind(&bytes)
                    .execute(&mut **tx),
                )
                .map_err(failure)?;
        }
        self.pending
            .borrow_mut()
            .push((key.to_string(), Some(bytes)));
        Ok(())
    }

    pub(crate) fn delete(&mut self, key: &str) -> Result<()> {
        {
            let mut guard = self.tx.borrow_mut();
            let tx = guard.as_mut().expect("transaction already finished");
            pg_runtime()
                .block_on(
                    sqlx::query(&format!(
                        "DELETE FROM {} WHERE kind = $1 AND key = $2",
                        self.inner.table
                    ))
                    .bind(&self.inner.kind)
                    .bind(key.as_bytes())
                    .execute(&mut **tx),
                )
                .map_err(failure)?;
        }
        self.pending.borrow_mut().push((key.to_string(), None));
        Ok(())
    }

    pub(crate) fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        let mut guard = self.tx.borrow_mut();
        let tx = guard.as_mut().expect("transaction already finished");
        let rows = pg_runtime()
            .block_on(fetch_scan(&mut **tx, &self.inner.table, &self.inner.kind, prefix.as_bytes()))
            .map_err(failure)?;
        rows.into_iter()
            .map(|(_, value)| serde_json::from_slice(&value).map_err(failure))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_upper_bound_increments_last_byte() {
        assert_eq!(prefix_upper_bound(b"abc"), Some(b"abd".to_vec()));
    }

    #[test]
    fn prefix_upper_bound_trims_trailing_ff_bytes() {
        assert_eq!(prefix_upper_bound(&[b'a', 0xFF]), Some(vec![b'b']));
    }

    #[test]
    fn prefix_upper_bound_is_none_for_all_ff() {
        assert_eq!(prefix_upper_bound(&[0xFF, 0xFF]), None);
    }

    #[test]
    fn prefix_upper_bound_is_none_for_empty_prefix() {
        assert_eq!(prefix_upper_bound(&[]), None);
    }

    #[test]
    fn identifiers_reject_anything_that_is_not_snake_case() {
        assert!(validate_identifier("akh_sessions").is_ok());
        assert!(validate_identifier("Akh").is_err());
        assert!(validate_identifier("akh sessions").is_err());
        assert!(validate_identifier("akh;drop table x;--").is_err());
        assert!(validate_identifier("").is_err());
    }

    #[test]
    fn cache_populates_on_insert_and_serves_hits() {
        let cache = ReadCache::default();
        assert_eq!(cache.get("a"), None);
        cache.insert("a".to_string(), b"1".to_vec());
        assert_eq!(cache.get("a"), Some(b"1".to_vec()));
    }

    #[test]
    fn cache_apply_committed_writes_through_puts_and_deletes() {
        let cache = ReadCache::default();
        cache.insert("kept".to_string(), b"old".to_vec());
        cache.apply_committed(vec![
            ("a".to_string(), Some(b"1".to_vec())),
            ("kept".to_string(), None),
        ]);
        assert_eq!(cache.get("a"), Some(b"1".to_vec()));
        assert_eq!(cache.get("kept"), None);
    }

    #[test]
    fn cache_is_untouched_when_a_transaction_never_reaches_apply_committed() {
        // This is the entire "invalidated on rollback" story: a rolled-back
        // transaction's pending writes are simply dropped by the caller
        // instead of being passed to `apply_committed`, so the cache never
        // observes them.
        let cache = ReadCache::default();
        cache.insert("a".to_string(), b"before".to_vec());
        let pending = vec![("a".to_string(), Some(b"after".to_vec()))];
        drop(pending); // simulates `Store::transaction`'s rollback path
        assert_eq!(cache.get("a"), Some(b"before".to_vec()));
    }
}
