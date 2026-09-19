//! Postgres-backed implementation of the `Store`/`Transaction` API.
//!
//! # Sync facade over an async driver, without nesting runtimes
//!
//! `Store`'s public API is synchronous (plain `fn`s, no `.await`), because
//! the redb backend it originally shipped with is synchronous and every call
//! site in `akh-session-services` invokes it that way — often from inside a
//! `#[tokio::main]` daemon's async request handlers. sqlx is async-only, so
//! this module has to bridge the two.
//!
//! An earlier version of this module did that by calling
//! `.block_on(...)` on a private, lazily-started multi-thread Tokio runtime
//! from whichever thread the caller happened to be on. That is unsound: it
//! panics with "Cannot start a runtime from within a runtime" the moment the
//! caller thread is *already* a worker thread of some other Tokio runtime
//! (e.g. every `#[tokio::main]` daemon's request-handling path) — Tokio's
//! "already inside a runtime" detection is a thread-local flag checked
//! regardless of which `Runtime` object `block_on` is called on, so a
//! second, distinct runtime instance does not side-step it. This crashed
//! every daemon at startup in production (see the crate's `CHANGES`/incident
//! notes) the moment `Store::open_postgres` (or any `get`/`scan`/
//! `transaction` call) executed on a Tokio worker thread.
//!
//! The fix used here instead runs all Postgres/sqlx work on a **dedicated
//! OS thread that Tokio never owns**, so nothing ever calls `block_on` while
//! standing on a runtime's worker thread:
//!
//! - [`db_thread`] lazily spawns one plain `std::thread` (named
//!   `akh-store-pg`) that builds its own `tokio::runtime::Builder::
//!   new_current_thread()` runtime and then just loops: receive a job from an
//!   `std::sync::mpsc::Receiver<Job>`, `runtime.block_on` it, repeat. That
//!   thread is never entered by any other code, so `block_on` there is
//!   always the outermost (and only) runtime frame on that thread — the
//!   hazard the old code hit cannot occur here.
//! - Every Postgres operation (`open`, `get`, `scan`, `begin`, a
//!   transaction's `get`/`put`/`delete`/`scan`, `commit`/`rollback`) is
//!   packaged by [`run_on_db_thread`] into one such job: a boxed closure
//!   that produces the future to run, plus a `std::sync::mpsc::sync_channel`
//!   of capacity 1 used exactly once to send back the result. The calling
//!   thread — whatever it is — sends the job and then blocks waiting for
//!   that one reply.
//! - How the calling thread blocks depends on what it *is*:
//!   - If it is a worker thread of a multi-thread Tokio runtime
//!     (`Handle::try_current().runtime_flavor() ==
//!     RuntimeFlavor::MultiThread`), the wait is wrapped in
//!     `tokio::task::block_in_place`, which tells that runtime "this worker
//!     is about to block; hand its other ready tasks to another worker (or
//!     spawn one)". This is exactly the situation `block_in_place` exists
//!     for, and it is what every daemon in this workspace hits — all five
//!     (`akhsessd`, `akhrtd`, `akhexecd`, `akhsbxd`, `akheventd`) use a bare
//!     `#[tokio::main]`, which defaults to the multi-thread scheduler.
//!   - If there is no ambient runtime at all (a plain `#[test]`, or a call
//!     from a bare `std::thread`), the calling thread just blocks on
//!     `Receiver::recv()` directly — nothing else needed that thread, so
//!     blocking it is free.
//!   - If the ambient runtime exists but is `current_thread` flavor,
//!     `block_in_place` would itself panic ("this runtime doesn't support
//!     blocking operations"), so this case also falls back to a plain
//!     `Receiver::recv()`. That is only safe because it stalls that
//!     runtime's *only* thread for the duration of one DB round trip — any
//!     other task scheduled on it (a concurrent request, a health check)
//!     would not make progress until the store call returns. No binary in
//!     this workspace uses `current_thread` today (grepping for
//!     `tokio::main`/`tokio::test` turns up only the multi-thread default,
//!     confirmed for every `akhsessd`/`akhrtd`/`akhexecd`/`akhsbxd`/
//!     `akheventd` binary), so this branch is a documented safety net, not
//!     a path anything currently takes. If a `current_thread` daemon is ever
//!     added, revisit this — it would need either a runtime-flavor change
//!     or a different blocking strategy.
//!
//! All Postgres work is funneled through the single `akh-store-pg` thread
//! and processed strictly one job at a time (the thread's loop does not
//! start the next job until the previous one's future resolves). That is a
//! deliberate simplification, not an oversight: `Store` is already meant to
//! be a single-writer-at-a-time, low-QPS durability layer (see the locking
//! and cache sections below, and the "single process per store kind"
//! invariant they depend on), so serializing its Postgres round trips onto
//! one thread trades a small amount of theoretical throughput for a much
//! simpler, obviously-correct implementation. If this ever becomes a
//! bottleneck, the jobs could be `tokio::spawn`ed onto the `akh-store-pg`
//! runtime instead of `block_on`-ed one at a time, letting them interleave
//! on that same single OS thread — but nothing measured so far calls for
//! that complexity.
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
//! Each Postgres-backed `Store` keeps a bounded in-memory LRU read cache. It
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
//!
//! The cache is capped by total cached bytes (keys plus values) and evicts
//! least-recently-used entries once over budget. It was previously an
//! unbounded `HashMap`, which made it a read-through record of every key the
//! process had ever touched: `akhsessd` grew roughly a gigabyte a day until
//! it hit its container limit and was OOM-killed, about every one to two
//! days. Eviction is always safe here because the cache is read-through — a
//! miss simply re-reads the row from Postgres.
use crate::{Fault, Result, failure};
use lru::LruCache;
use serde::{Serialize, de::DeserializeOwned};
use sqlx::postgres::{PgPool, PgPoolOptions};
use std::{
    cell::RefCell,
    future::Future,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{Arc, Mutex, MutexGuard, OnceLock, mpsc},
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
    /// Ceiling on the in-memory read cache, in bytes of keys plus values.
    /// `None` uses [`DEFAULT_CACHE_CAPACITY_BYTES`]; `Some(0)` disables
    /// caching entirely, sending every read to Postgres.
    pub cache_capacity_bytes: Option<usize>,
}

pub(crate) struct PostgresInner {
    pool: PgPool,
    /// Fully-quoted `"schema".akh_store`, precomputed once at open time.
    table: String,
    kind: String,
    cache: ReadCache,
    /// Serialises transactions within this process; see
    /// [`PostgresInner::transaction_gate`].
    tx_gate: Mutex<()>,
}

/// Default ceiling for [`ReadCache`], in bytes of keys plus values.
///
/// The cache exists to spare Postgres round-trips, not to mirror the store,
/// so this is deliberately far below any daemon's container limit: 64 MiB
/// leaves the previous unbounded growth no room to recur while still
/// absorbing the hot working set.
pub const DEFAULT_CACHE_CAPACITY_BYTES: usize = 64 * 1024 * 1024;

/// Process-local read-through cache for a single Postgres-backed store
/// (i.e. a single `kind`). Deliberately not shared across kinds or
/// processes; see the module docs for why that is safe here. Kept as its
/// own type so its commit/rollback and eviction semantics can be unit tested
/// without a database.
///
/// Bounded by total cached bytes rather than entry count, because entries
/// are opaque serialised values whose sizes differ by orders of magnitude —
/// an entry cap would bound the count while leaving memory unbounded, which
/// is the bug this type exists to prevent.
struct ReadCache(Mutex<CacheState>);

struct CacheState {
    entries: LruCache<String, Vec<u8>>,
    /// Sum of `key.len() + value.len()` over `entries`.
    bytes: usize,
    capacity_bytes: usize,
}

impl Default for ReadCache {
    fn default() -> Self {
        Self::with_capacity_bytes(DEFAULT_CACHE_CAPACITY_BYTES)
    }
}

impl ReadCache {
    fn with_capacity_bytes(capacity_bytes: usize) -> Self {
        Self(Mutex::new(CacheState {
            // Unbounded by count: `CacheState::admit` enforces the byte
            // budget, which is the only bound that maps to memory.
            entries: LruCache::unbounded(),
            bytes: 0,
            capacity_bytes,
        }))
    }

    fn state(&self) -> MutexGuard<'_, CacheState> {
        self.0
            .lock()
            .expect("akh-service-store Postgres cache lock poisoned")
    }

    fn get(&self, key: &str) -> Option<Vec<u8>> {
        // `LruCache::get` records the access, so this needs the write side of
        // the lock; that is why this is a `Mutex` and not an `RwLock`.
        self.state().entries.get(key).cloned()
    }

    fn insert(&self, key: String, value: Vec<u8>) {
        self.state().admit(key, value);
    }

    /// Applies a transaction's staged writes/deletes after it has committed.
    /// Never call this for a rolled-back transaction — that is the entire
    /// "invalidation" story: a rollback simply never reaches this method, so
    /// the cache keeps whatever it held before the transaction began.
    fn apply_committed(&self, pending: Vec<(String, Option<Vec<u8>>)>) {
        let mut state = self.state();
        for (key, value) in pending {
            match value {
                Some(bytes) => state.admit(key, bytes),
                None => state.evict(&key),
            }
        }
    }
}

impl CacheState {
    fn entry_bytes(key: &str, value: &[u8]) -> usize {
        key.len() + value.len()
    }

    /// Inserts `key`, then evicts least-recently-used entries until the
    /// budget is met.
    ///
    /// A value too large to ever fit is not cached at all, and any previous
    /// entry for that key is dropped rather than left behind: serving a
    /// superseded value would be a correctness bug, where a miss is only a
    /// Postgres round-trip.
    fn admit(&mut self, key: String, value: Vec<u8>) {
        self.evict(&key);
        if Self::entry_bytes(&key, &value) > self.capacity_bytes {
            return;
        }
        self.bytes += Self::entry_bytes(&key, &value);
        self.entries.put(key, value);
        while self.bytes > self.capacity_bytes {
            match self.entries.pop_lru() {
                Some((evicted_key, evicted_value)) => {
                    self.bytes = self
                        .bytes
                        .saturating_sub(Self::entry_bytes(&evicted_key, &evicted_value));
                }
                // Unreachable while `bytes` tracks `entries`, but looping
                // forever on an accounting drift would be worse than
                // tolerating it.
                None => {
                    self.bytes = 0;
                    break;
                }
            }
        }
    }

    fn evict(&mut self, key: &str) {
        if let Some(previous) = self.entries.pop(key) {
            self.bytes = self.bytes.saturating_sub(Self::entry_bytes(key, &previous));
        }
    }
}

/// A transaction's `sqlx::Transaction`, shared between the caller thread
/// (which owns the `PgTxState`) and whichever job the `akh-store-pg` thread
/// is currently running against it. Only ever touched by one job at a time
/// — the caller always waits for one job's reply before issuing the next —
/// so the `Mutex` is never actually contended; it exists so the handle can
/// be `Send`/`'static` and moved into each job's closure by reference-
/// counted clone.
type TxHandle = Arc<Mutex<Option<sqlx::Transaction<'static, sqlx::Postgres>>>>;

pub(crate) struct PgTxState<'a> {
    inner: &'a PostgresInner,
    tx: TxHandle,
    pending: RefCell<Vec<(String, Option<Vec<u8>>)>>,
}

/// One unit of work for the `akh-store-pg` thread: a closure that, when
/// called (on that thread), produces the future to drive to completion. See
/// [`run_on_db_thread`] for how the result gets back to the caller.
type Job = Box<dyn FnOnce() -> Pin<Box<dyn Future<Output = ()> + Send>> + Send>;

struct DbThread {
    sender: mpsc::Sender<Job>,
}

/// Lazily spawns (once per process) the dedicated `akh-store-pg` OS thread
/// and its private single-threaded Tokio runtime. See the module docs for
/// why this thread — and never the caller's own thread — is what ever calls
/// `block_on`.
fn db_thread() -> &'static DbThread {
    static THREAD: OnceLock<DbThread> = OnceLock::new();
    THREAD.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("akh-store-pg".to_string())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to start the dedicated akh-store-pg runtime");
                // One job at a time, start to finish, before pulling the
                // next: see the module docs for why that is an intentional
                // simplification rather than an accident.
                while let Ok(job) = receiver.recv() {
                    runtime.block_on(job());
                }
            })
            .expect("failed to spawn the akh-store-pg thread");
        DbThread { sender }
    })
}

/// Runs `work` to completion on the dedicated `akh-store-pg` thread and
/// blocks the calling thread for the result, using whichever waiting
/// strategy is safe for the caller's own runtime context. See the module
/// docs for the three cases this distinguishes.
fn run_on_db_thread<T, F, Fut>(work: F) -> T
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = T> + Send + 'static,
{
    let (reply_tx, reply_rx) = mpsc::sync_channel::<T>(1);
    let job: Job = Box::new(move || {
        Box::pin(async move {
            let result = work().await;
            // `send` only fails if the receiving end was already dropped,
            // which only happens if `block_for_reply` below already gave up
            // (e.g. panicked) — nothing useful to do with that here.
            let _ = reply_tx.send(result);
        })
    });
    db_thread()
        .sender
        .send(job)
        .expect("the akh-store-pg worker thread has terminated; cannot submit more work");
    block_for_reply(reply_rx)
}

/// Blocks the calling thread for a `run_on_db_thread` reply, choosing the
/// wait strategy that is actually safe given whatever runtime (if any) owns
/// this thread. See the module docs for the reasoning behind each branch.
fn block_for_reply<T>(reply_rx: mpsc::Receiver<T>) -> T {
    let recv = move || {
        reply_rx
            .recv()
            .expect("akh-store-pg worker thread dropped a reply sender without responding")
    };
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            // Hands this worker off so the runtime can keep making progress
            // on other tasks while we block on the DB thread's reply.
            tokio::task::block_in_place(recv)
        }
        // No ambient runtime (plain thread/`#[test]`), or a `current_thread`
        // runtime where `block_in_place` would itself panic: just block.
        // See the module docs for why the latter is safe only because no
        // binary in this workspace currently runs a `current_thread` daemon.
        _ => recv(),
    }
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

/// `advisory_lock`'s `seed` for the per-`kind` transaction lock (see
/// [`PostgresInner::begin`] and [`migrate_from_redb_if_needed`]).
const KIND_TRANSACTION_LOCK_SEED: i64 = 0;
/// `advisory_lock`'s `seed` for the schema-bootstrap lock (see
/// [`ensure_schema`]). Distinct from the per-`kind` seed above so a `kind`
/// string that happened to equal the schema name would not contend for the
/// same lock as an unrelated bootstrap.
const SCHEMA_BOOTSTRAP_LOCK_SEED: i64 = 1;

/// Every daemon (akhsessd, akheventd, akhrtd, akhexecd, akhsbxd) opens its
/// store — and therefore calls this — concurrently right after a rollout,
/// all against the same shared `schema`. Without serializing it, two
/// connections can both pass `CREATE SCHEMA IF NOT EXISTS`/`CREATE TABLE IF
/// NOT EXISTS`'s existence check before either commits, and one loses with a
/// duplicate-key error against the catalog (`pg_namespace_nspname_index` or
/// `pg_type_typname_nsp_index`). Taking a dedicated advisory lock keyed by
/// the schema name — so every daemon contends for the exact same lock,
/// regardless of `kind` — serializes the bootstrap; the 23505 retry is a
/// belt-and-braces fallback for any interleaving the lock doesn't cover.
async fn ensure_schema(pool: &PgPool, schema: &str, table: &str) -> sqlx::Result<()> {
    for attempt in 0..2 {
        let mut tx = pool.begin().await?;
        advisory_lock(&mut tx, schema, SCHEMA_BOOTSTRAP_LOCK_SEED).await?;
        let bootstrap: sqlx::Result<()> = async {
            sqlx::query(&format!("CREATE SCHEMA IF NOT EXISTS \"{schema}\""))
                .execute(&mut *tx)
                .await?;
            sqlx::query(&format!(
                "CREATE TABLE IF NOT EXISTS {table} (\
                    kind TEXT NOT NULL, \
                    key BYTEA NOT NULL, \
                    value BYTEA NOT NULL, \
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), \
                    PRIMARY KEY (kind, key))"
            ))
            .execute(&mut *tx)
            .await?;
            Ok(())
        }
        .await;
        match bootstrap {
            Ok(()) => {
                tx.commit().await?;
                return Ok(());
            }
            Err(sqlx::Error::Database(ref db_error))
                if attempt == 0 && db_error.code().as_deref() == Some("23505") =>
            {
                let _ = tx.rollback().await;
            }
            Err(error) => return Err(error),
        }
    }
    // Unreachable in practice (the loop above always returns from its first
    // iteration unless it hits a retryable 23505, and the second iteration
    // always returns), spelled out so a future edit that changes the retry
    // count can't silently fall through without a result.
    unreachable!("ensure_schema either returns Ok or propagates an error within two attempts")
}

async fn advisory_lock(
    tx: &mut sqlx::Transaction<'static, sqlx::Postgres>,
    key: &str,
    seed: i64,
) -> sqlx::Result<()> {
    sqlx::query(&format!(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, {seed}))"
    ))
    .bind(key)
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
    advisory_lock(&mut tx, kind, KIND_TRANSACTION_LOCK_SEED).await?;
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

async fn fetch_value<'e, E>(
    executor: E,
    table: &str,
    kind: &str,
    key: &[u8],
) -> sqlx::Result<Option<Vec<u8>>>
where
    E: sqlx::Executor<'e, Database = sqlx::Postgres>,
{
    sqlx::query_scalar(&format!(
        "SELECT value FROM {table} WHERE kind = $1 AND key = $2"
    ))
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
        let PostgresConfig {
            url,
            schema,
            kind,
            legacy_redb_path,
            cache_capacity_bytes,
        } = config;
        let table_for_setup = table.clone();
        let kind_for_setup = kind.clone();
        let pool = run_on_db_thread(move || async move {
            let pool = PgPoolOptions::new()
                .max_connections(5)
                .connect(&url)
                .await?;
            ensure_schema(&pool, &schema, &table_for_setup).await?;
            migrate_from_redb_if_needed(
                &pool,
                &table_for_setup,
                &kind_for_setup,
                legacy_redb_path.as_deref(),
            )
            .await?;
            Ok::<_, sqlx::Error>(pool)
        })
        .map_err(failure)?;
        Ok(Self {
            pool,
            table,
            kind,
            cache: ReadCache::with_capacity_bytes(
                cache_capacity_bytes.unwrap_or(DEFAULT_CACHE_CAPACITY_BYTES),
            ),
            tx_gate: Mutex::new(()),
        })
    }

    /// Admits one transaction at a time per process, the way redb's single
    /// write transaction did. This is not an optimisation but a deadlock
    /// guard: the per-kind advisory lock in [`PostgresInner::begin`] is
    /// taken on the `akh-store-pg` thread, which runs one job start to
    /// finish. If a second transaction's `begin` were allowed to queue while
    /// a first one is open, it would block that thread on the advisory lock
    /// held by the first transaction, whose own next statement is then stuck
    /// behind it in the job queue — both hang forever and every plain read
    /// in the process with them. akhsessd and akheventd did exactly that
    /// within seconds of starting on 2026-09-13 (two backends per pod in
    /// `pg_stat_activity`: one `idle in transaction`, one waiting on
    /// `advisory`). Waiting here, on the caller's thread, keeps the DB
    /// thread free to serve the open transaction. Uses `block_in_place` on
    /// a multi-thread runtime for the same reason [`block_for_reply`] does.
    pub(crate) fn transaction_gate(&self) -> MutexGuard<'_, ()> {
        let lock = || {
            self.tx_gate
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        };
        match tokio::runtime::Handle::try_current() {
            Ok(handle) if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
                tokio::task::block_in_place(lock)
            }
            _ => lock(),
        }
    }

    pub(crate) fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        if let Some(bytes) = self.cache.get(key) {
            return serde_json::from_slice(&bytes).map(Some).map_err(failure);
        }
        let pool = self.pool.clone();
        let table = self.table.clone();
        let kind = self.kind.clone();
        let key_bytes = key.as_bytes().to_vec();
        let row =
            run_on_db_thread(
                move || async move { fetch_value(&pool, &table, &kind, &key_bytes).await },
            )
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
        let pool = self.pool.clone();
        let table = self.table.clone();
        let kind = self.kind.clone();
        let prefix_bytes = prefix.as_bytes().to_vec();
        let rows = run_on_db_thread(move || async move {
            fetch_scan(&pool, &table, &kind, &prefix_bytes).await
        })
        .map_err(failure)?;
        rows.into_iter()
            .map(|(_, value)| serde_json::from_slice(&value).map_err(failure))
            .collect()
    }

    pub(crate) fn begin(&self) -> Result<PgTxState<'_>> {
        let pool = self.pool.clone();
        let kind = self.kind.clone();
        let sqlx_tx = run_on_db_thread(move || async move {
            let mut tx = pool.begin().await?;
            advisory_lock(&mut tx, &kind, KIND_TRANSACTION_LOCK_SEED).await?;
            Ok::<_, sqlx::Error>(tx)
        })
        .map_err(failure)?;
        Ok(PgTxState {
            inner: self,
            tx: Arc::new(Mutex::new(Some(sqlx_tx))),
            pending: RefCell::new(Vec::new()),
        })
    }

    pub(crate) fn finish<T>(&self, state: PgTxState<'_>, result: Result<T>) -> Result<T> {
        let sqlx_tx = state
            .tx
            .lock()
            .expect("akh-store-pg tx lock poisoned")
            .take()
            .expect("transaction handle consumed twice");
        match result {
            Ok(value) => {
                run_on_db_thread(move || async move { sqlx_tx.commit().await }).map_err(failure)?;
                self.cache.apply_committed(state.pending.into_inner());
                Ok(value)
            }
            Err(err) => {
                // Best-effort: the connection returning to the pool without an
                // explicit commit already rolls back on drop, but doing it
                // explicitly surfaces failures instead of swallowing them.
                let _ = run_on_db_thread(move || async move { sqlx_tx.rollback().await });
                Err(err)
            }
        }
    }
}

/// Takes `handle`'s transaction out, handing back both the owned
/// transaction and a guard-free "put it back" closure. A
/// `std::sync::MutexGuard` is not `Send`, so it cannot be held across an
/// `.await` inside a future that must itself be `Send` (required by
/// [`run_on_db_thread`]); every call site below takes the transaction out
/// before its query's `.await`, and restores it with
/// [`restore_transaction`] right after. This is sound because only one job
/// touches a given transaction at a time (the caller always waits for one
/// job's reply before issuing the next), so there is never a second taker
/// while a transaction is checked out.
fn take_transaction(handle: &TxHandle) -> sqlx::Transaction<'static, sqlx::Postgres> {
    handle
        .lock()
        .expect("akh-store-pg tx lock poisoned")
        .take()
        .expect("transaction already finished")
}

fn restore_transaction(handle: &TxHandle, tx: sqlx::Transaction<'static, sqlx::Postgres>) {
    *handle.lock().expect("akh-store-pg tx lock poisoned") = Some(tx);
}

impl PgTxState<'_> {
    pub(crate) fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let tx = self.tx.clone();
        let table = self.inner.table.clone();
        let kind = self.inner.kind.clone();
        let key_bytes = key.as_bytes().to_vec();
        let row = run_on_db_thread(move || async move {
            let mut sqlx_tx = take_transaction(&tx);
            let result = fetch_value(&mut *sqlx_tx, &table, &kind, &key_bytes).await;
            restore_transaction(&tx, sqlx_tx);
            result
        })
        .map_err(failure)?;
        row.map(|bytes| serde_json::from_slice(&bytes).map_err(failure))
            .transpose()
    }

    pub(crate) fn put<T: Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        let bytes = serde_json::to_vec(value).map_err(failure)?;
        let tx = self.tx.clone();
        let table = self.inner.table.clone();
        let kind = self.inner.kind.clone();
        let key_bytes = key.as_bytes().to_vec();
        let bytes_for_write = bytes.clone();
        run_on_db_thread(move || async move {
            let mut sqlx_tx = take_transaction(&tx);
            let result = sqlx::query(&format!(
                "INSERT INTO {table} (kind, key, value, updated_at) VALUES ($1, $2, $3, now()) \
                 ON CONFLICT (kind, key) DO UPDATE SET value = excluded.value, updated_at = now()"
            ))
            .bind(&kind)
            .bind(&key_bytes)
            .bind(&bytes_for_write)
            .execute(&mut *sqlx_tx)
            .await;
            restore_transaction(&tx, sqlx_tx);
            result
        })
        .map_err(failure)?;
        self.pending
            .borrow_mut()
            .push((key.to_string(), Some(bytes)));
        Ok(())
    }

    pub(crate) fn delete(&mut self, key: &str) -> Result<()> {
        let tx = self.tx.clone();
        let table = self.inner.table.clone();
        let kind = self.inner.kind.clone();
        let key_bytes = key.as_bytes().to_vec();
        run_on_db_thread(move || async move {
            let mut sqlx_tx = take_transaction(&tx);
            let result = sqlx::query(&format!("DELETE FROM {table} WHERE kind = $1 AND key = $2"))
                .bind(&kind)
                .bind(&key_bytes)
                .execute(&mut *sqlx_tx)
                .await;
            restore_transaction(&tx, sqlx_tx);
            result
        })
        .map_err(failure)?;
        self.pending.borrow_mut().push((key.to_string(), None));
        Ok(())
    }

    pub(crate) fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        let tx = self.tx.clone();
        let table = self.inner.table.clone();
        let kind = self.inner.kind.clone();
        let prefix_bytes = prefix.as_bytes().to_vec();
        let rows = run_on_db_thread(move || async move {
            let mut sqlx_tx = take_transaction(&tx);
            let result = fetch_scan(&mut *sqlx_tx, &table, &kind, &prefix_bytes).await;
            restore_transaction(&tx, sqlx_tx);
            result
        })
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
    fn cache_evicts_least_recently_used_once_over_its_byte_budget() {
        // Budget fits exactly two 10-byte entries (1-byte key + 9-byte value).
        let cache = ReadCache::with_capacity_bytes(20);
        cache.insert("a".to_string(), vec![b'x'; 9]);
        cache.insert("b".to_string(), vec![b'x'; 9]);
        // Touch "a" so "b" becomes the least recently used.
        assert_eq!(cache.get("a"), Some(vec![b'x'; 9]));
        cache.insert("c".to_string(), vec![b'x'; 9]);
        assert_eq!(
            cache.get("b"),
            None,
            "least recently used should be evicted"
        );
        assert_eq!(cache.get("a"), Some(vec![b'x'; 9]));
        assert_eq!(cache.get("c"), Some(vec![b'x'; 9]));
    }

    #[test]
    fn cache_accounting_survives_overwrites_and_deletes() {
        let cache = ReadCache::with_capacity_bytes(20);
        // Overwriting must not double-count the old value's bytes, otherwise
        // the budget drifts down until the cache holds nothing.
        for _ in 0..50 {
            cache.insert("a".to_string(), vec![b'x'; 9]);
        }
        cache.insert("b".to_string(), vec![b'x'; 9]);
        assert_eq!(cache.get("a"), Some(vec![b'x'; 9]));
        assert_eq!(cache.get("b"), Some(vec![b'x'; 9]));
        assert_eq!(cache.state().bytes, 20);

        cache.apply_committed(vec![("a".to_string(), None)]);
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.state().bytes, 10);
    }

    #[test]
    fn cache_refuses_a_value_too_large_to_ever_fit_and_drops_the_stale_one() {
        let cache = ReadCache::with_capacity_bytes(20);
        cache.insert("a".to_string(), vec![b'o'; 9]);
        // Too big for the budget: it must not be cached, and the superseded
        // value must not survive to be served in its place.
        cache.insert("a".to_string(), vec![b'n'; 100]);
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.state().bytes, 0);
    }

    #[test]
    fn cache_can_be_disabled_entirely() {
        let cache = ReadCache::with_capacity_bytes(0);
        cache.insert("a".to_string(), b"1".to_vec());
        assert_eq!(cache.get("a"), None);
        assert_eq!(cache.state().bytes, 0);
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

    /// Regression test for the production incident: calling `run_on_db_thread`
    /// synchronously from inside an async function running on a multi-thread
    /// Tokio runtime worker must not panic with "Cannot start a runtime from
    /// within a runtime". This does not touch Postgres — it only exercises
    /// the `block_for_reply`/`block_in_place` bridging, which is exactly what
    /// crashed every daemon at startup before this module's rewrite.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_on_db_thread_does_not_panic_from_a_multi_thread_runtime_worker() {
        let value = run_on_db_thread(|| async { 42u32 });
        assert_eq!(value, 42);
    }

    #[test]
    fn run_on_db_thread_works_with_no_ambient_runtime() {
        let value = run_on_db_thread(|| async { "ok".to_string() });
        assert_eq!(value, "ok");
    }
}
