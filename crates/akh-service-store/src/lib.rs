//! One durable database per daemon. Multi-record decisions commit atomically.
//!
//! `Store` has two interchangeable backends, selected at construction time:
//!
//! - **redb** (default, via [`Store::open`]): one local file per store kind.
//!   This is the original implementation and remains what every test and
//!   every existing call site uses unless it opts into Postgres.
//! - **Postgres** (via [`Store::open_postgres`]): a shared cluster, used in
//!   production so daemon state survives node/PVC loss instead of living on
//!   a single Raspberry Pi SD card. See the `postgres` module docs for the
//!   sync/async bridging, locking, cache and migration design.
//!
//! Both backends implement exactly the same public API (`get`/`scan` on
//! `Store`, `get`/`put`/`delete`/`scan` on `Transaction`, the same `key()`
//! helper), so no call site needs to know or care which one it is talking
//! to.
mod postgres;
mod redb_backend;

pub(crate) use akh_session_types::{Fault, Result};
pub use postgres::{DEFAULT_CACHE_CAPACITY_BYTES, PostgresConfig};
use postgres::{PgTxState, PostgresInner};
use redb_backend::{RedbInner, RedbTxState};
use serde::{Serialize, de::DeserializeOwned};
use std::{path::Path, sync::Arc};
use tokio::sync::Notify;

pub(crate) fn failure(error: impl std::fmt::Display) -> Fault {
    Fault::new("storage_error", error.to_string())
}

enum Inner {
    Redb(RedbInner),
    Postgres(PostgresInner),
}

struct Shared {
    inner: Inner,
    /// Signalled after every committed transaction; see [`Store::changes`].
    changes: Arc<Notify>,
}

#[derive(Clone)]
pub struct Store(Arc<Shared>);

enum TransactionInner<'a> {
    Redb(RedbTxState<'a>),
    Postgres(PgTxState<'a>),
}

pub struct Transaction<'a> {
    inner: TransactionInner<'a>,
}

impl Store {
    /// Opens (creating if needed) a redb-backed store at `path`. Unchanged
    /// since before the Postgres backend existed — every existing call site
    /// and every test keeps using this.
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self::wrap(Inner::Redb(RedbInner::open(path)?)))
    }

    /// Opens a Postgres-backed store, importing a legacy redb file's
    /// contents once if `config.legacy_redb_path` points at one and this
    /// `kind` has not already been migrated. See the `postgres` module docs
    /// for the full design (locking, cache, sync bridging).
    pub fn open_postgres(config: PostgresConfig) -> Result<Self> {
        Ok(Self::wrap(Inner::Postgres(PostgresInner::open(config)?)))
    }

    fn wrap(inner: Inner) -> Self {
        Self(Arc::new(Shared {
            inner,
            changes: Arc::new(Notify::new()),
        }))
    }

    /// A handle notified after every transaction that commits on this store.
    ///
    /// Lets a poller react to writes instead of re-scanning on a timer. The
    /// sessions dispatcher previously scanned every session record once a
    /// second regardless of activity, which on a store holding only closed
    /// sessions was ~19 MB deserialised per second for no work — most of its
    /// CPU and all of its memory churn. Because every write already goes
    /// through [`Store::transaction`], signalling here cannot miss a mutation
    /// path the way per-call-site wakeups can.
    ///
    /// Semantics are those of [`Notify::notify_one`]: a signal with no waiter
    /// is kept as a single permit, so a write that lands while the consumer is
    /// busy is not lost, and a burst of writes coalesces into one wakeup. A
    /// rolled-back transaction never signals, and neither does a committed
    /// transaction that only read: idempotent request replays (a client
    /// re-POSTing a repository or credential it already has) run their
    /// validation inside a transaction and return early without writing, and
    /// waking the dispatcher for those had it re-scanning every session on
    /// every replay. Waiters see "something changed", not what; they are
    /// expected to re-read.
    pub fn changes(&self) -> Arc<Notify> {
        self.0.changes.clone()
    }

    pub fn transaction<T>(
        &self,
        action: impl FnOnce(&mut Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        match &self.0.inner {
            Inner::Redb(inner) => {
                let write = inner.database().begin_write().map_err(failure)?;
                let (result, wrote) = {
                    let table = write.open_table(redb_backend::RECORDS).map_err(failure)?;
                    let mut tx = Transaction {
                        inner: TransactionInner::Redb(RedbTxState::new(table)),
                    };
                    let result = action(&mut tx);
                    (result, tx.wrote())
                };
                // Mirrors the original behavior exactly: only a successful
                // action commits. Dropping `write` without committing aborts
                // the redb transaction, so a failed action's writes never
                // become visible — this is what the rollback test asserts.
                match result {
                    Ok(value) => {
                        write.commit().map_err(failure)?;
                        if wrote {
                            self.0.changes.notify_one();
                        }
                        Ok(value)
                    }
                    Err(err) => Err(err),
                }
            }
            Inner::Postgres(inner) => {
                // Held until `finish` returns: one transaction per process
                // at a time (see `PostgresInner::transaction_gate`).
                let _gate = inner.transaction_gate();
                let state = inner.begin()?;
                let mut tx = Transaction {
                    inner: TransactionInner::Postgres(state),
                };
                let result = action(&mut tx);
                let wrote = tx.wrote();
                let TransactionInner::Postgres(state) = tx.inner else {
                    unreachable!("postgres transaction stayed postgres")
                };
                let outcome = inner.finish(state, result);
                if outcome.is_ok() && wrote {
                    self.0.changes.notify_one();
                }
                outcome
            }
        }
    }

    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match &self.0.inner {
            Inner::Redb(inner) => inner.get(key),
            Inner::Postgres(inner) => inner.get(key),
        }
    }

    pub fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        match &self.0.inner {
            Inner::Redb(inner) => inner.scan(prefix),
            Inner::Postgres(inner) => inner.scan(prefix),
        }
    }
}

impl Transaction<'_> {
    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match &self.inner {
            TransactionInner::Redb(state) => state.get(key),
            TransactionInner::Postgres(state) => state.get(key),
        }
    }
    pub fn put<T: Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        match &mut self.inner {
            TransactionInner::Redb(state) => state.put(key, value),
            TransactionInner::Postgres(state) => state.put(key, value),
        }
    }
    pub fn delete(&mut self, key: &str) -> Result<()> {
        match &mut self.inner {
            TransactionInner::Redb(state) => state.delete(key),
            TransactionInner::Postgres(state) => state.delete(key),
        }
    }
    pub fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        match &self.inner {
            TransactionInner::Redb(state) => state.scan(prefix),
            TransactionInner::Postgres(state) => state.scan(prefix),
        }
    }
    fn wrote(&self) -> bool {
        match &self.inner {
            TransactionInner::Redb(state) => state.wrote(),
            TransactionInner::Postgres(state) => state.wrote(),
        }
    }
}

/// Length-prefix tuples so caller-controlled IDs cannot collide at a separator.
pub fn key(namespace: &str, parts: &[&str]) -> String {
    let mut key = format!("{namespace}/");
    for part in parts {
        key.push_str(&format!("{}:{part}", part.len()));
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_multi_record_admission_is_rolled_back_and_commits_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.redb");
        let store = Store::open(&path).unwrap();
        let result: Result<()> = store.transaction(|tx| {
            tx.put("session/x", &1)?;
            Err(Fault::conflict("abort"))
        });
        assert!(result.is_err());
        assert_eq!(store.get::<u32>("session/x").unwrap(), None);
        store
            .transaction(|tx| {
                tx.put("session/x", &2)?;
                tx.put("dedupe/x", &"session/x")
            })
            .unwrap();
        drop(store);
        let store = Store::open(&path).unwrap();
        assert_eq!(store.get::<u32>("session/x").unwrap(), Some(2));
        assert_eq!(
            store.get::<String>("dedupe/x").unwrap().as_deref(),
            Some("session/x")
        );
    }
    /// `changes()` must fire on commit and stay quiet on rollback. The
    /// sessions dispatcher now scans only when signalled, so a missed signal
    /// delays work until its fallback and a spurious one on rollback would
    /// make it scan for nothing. The retained-permit and coalescing
    /// behaviours are asserted too, because the dispatcher relies on both:
    /// a write that lands while it is mid-scan must still wake it, and a
    /// burst of writes must not turn into a burst of scans.
    #[tokio::test]
    async fn changes_signals_commits_and_not_rollbacks() {
        use std::time::Duration;
        use tokio::time::timeout;
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(&dir.path().join("changes.redb")).unwrap();
        let changes = store.changes();

        // Nothing has committed yet: no permit is pending.
        assert!(
            timeout(Duration::from_millis(100), changes.notified())
                .await
                .is_err()
        );

        store.transaction(|tx| tx.put("a/1", &1u8)).unwrap();
        // Signalled after the fact: the permit is retained, so a consumer
        // that was busy during the commit still wakes.
        assert!(
            timeout(Duration::from_secs(1), changes.notified())
                .await
                .is_ok()
        );

        // A rolled-back transaction must not signal.
        let failed: Result<()> = store.transaction(|tx| {
            tx.put("a/2", &2u8)?;
            Err(Fault::new("test", "abort"))
        });
        assert!(failed.is_err());
        assert!(
            timeout(Duration::from_millis(100), changes.notified())
                .await
                .is_err()
        );

        // A committed transaction that only read must not signal either:
        // nothing a waiter could observe has changed.
        let read: Option<u8> = store.transaction(|tx| tx.get("a/1")).unwrap();
        assert_eq!(read, Some(1));
        assert!(
            timeout(Duration::from_millis(100), changes.notified())
                .await
                .is_err()
        );

        // A burst coalesces into a single wakeup, not one per write.
        for i in 0..5u8 {
            store.transaction(|tx| tx.put("b", &i)).unwrap();
        }
        assert!(
            timeout(Duration::from_secs(1), changes.notified())
                .await
                .is_ok()
        );
        assert!(
            timeout(Duration::from_millis(100), changes.notified())
                .await
                .is_err()
        );
    }

    #[test]
    fn tuple_ids_do_not_alias() {
        assert_ne!(key("x", &["a/b", "c"]), key("x", &["a", "b/c"]));
    }
}
