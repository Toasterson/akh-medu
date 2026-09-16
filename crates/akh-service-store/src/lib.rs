//! One durable database per daemon. Multi-record decisions commit atomically.
//!
//! `Store` has three interchangeable backends, selected at construction time:
//!
//! - **redb** (default, via [`Store::open`]): one local file per store kind.
//!   This is the original implementation and remains what every test and
//!   every existing call site uses unless it opts into Postgres.
//! - **Postgres** (via [`Store::open_postgres`]): a shared cluster, used in
//!   production so daemon state survives node/PVC loss instead of living on
//!   a single Raspberry Pi SD card. See the `postgres` module docs for the
//!   sync/async bridging, locking, cache and migration design.
//!
//! SQLite is available through `Store::open_sqlite` for local, offline deployments.
//!
//! All backends implement exactly the same public API (`get`/`scan` on
//! `Store`, `get`/`put`/`delete`/`scan` on `Transaction`, the same `key()`
//! helper), so no call site needs to know or care which one it is talking
//! to.
mod postgres;
mod redb_backend;
mod sqlite_backend;

pub(crate) use akh_session_types::{Fault, Result};
pub use postgres::PostgresConfig;
use postgres::{PgTxState, PostgresInner};
use redb_backend::{RedbInner, RedbTxState};
use serde::{Serialize, de::DeserializeOwned};
use std::{path::Path, sync::Arc};

pub(crate) fn failure(error: impl std::fmt::Display) -> Fault {
    Fault::new("storage_error", error.to_string())
}

enum Inner {
    Sqlite(sqlite_backend::SqliteInner),
    Redb(RedbInner),
    Postgres(PostgresInner),
}

#[derive(Clone)]
pub struct Store(Arc<Inner>);

enum TransactionInner<'a> {
    Sqlite(&'a rusqlite::Connection),
    Redb(RedbTxState<'a>),
    Postgres(PgTxState<'a>),
}

pub struct Transaction<'a> {
    inner: TransactionInner<'a>,
}

impl Store {
    /// Opens a private, local SQLite database using WAL and full synchronous commits.
    pub fn open_sqlite(path: &Path) -> Result<Self> {
        Ok(Self(Arc::new(Inner::Sqlite(
            sqlite_backend::SqliteInner::open(path)?,
        ))))
    }

    /// Opens (creating if needed) a redb-backed store at `path`. Unchanged
    /// since before the Postgres backend existed — every existing call site
    /// and every test keeps using this.
    pub fn open(path: &Path) -> Result<Self> {
        Ok(Self(Arc::new(Inner::Redb(RedbInner::open(path)?))))
    }

    /// Opens a Postgres-backed store, importing a legacy redb file's
    /// contents once if `config.legacy_redb_path` points at one and this
    /// `kind` has not already been migrated. See the `postgres` module docs
    /// for the full design (locking, cache, sync bridging).
    pub fn open_postgres(config: PostgresConfig) -> Result<Self> {
        Ok(Self(Arc::new(Inner::Postgres(PostgresInner::open(
            config,
        )?))))
    }

    pub fn transaction<T>(
        &self,
        action: impl FnOnce(&mut Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        match self.0.as_ref() {
            Inner::Sqlite(inner) => {
                let mut connection = inner.lock()?;
                let write = connection
                    .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                    .map_err(failure)?;
                let result = action(&mut Transaction {
                    inner: TransactionInner::Sqlite(&write),
                })?;
                write.commit().map_err(failure)?;
                Ok(result)
            }
            Inner::Redb(inner) => {
                let write = inner.database().begin_write().map_err(failure)?;
                let result = {
                    let table = write.open_table(redb_backend::RECORDS).map_err(failure)?;
                    let mut tx = Transaction {
                        inner: TransactionInner::Redb(RedbTxState::new(table)),
                    };
                    action(&mut tx)
                };
                // Mirrors the original behavior exactly: only a successful
                // action commits. Dropping `write` without committing aborts
                // the redb transaction, so a failed action's writes never
                // become visible — this is what the rollback test asserts.
                match result {
                    Ok(value) => {
                        write.commit().map_err(failure)?;
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
                let TransactionInner::Postgres(state) = tx.inner else {
                    unreachable!("postgres transaction stayed postgres")
                };
                inner.finish(state, result)
            }
        }
    }

    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match self.0.as_ref() {
            Inner::Sqlite(inner) => sqlite_backend::get(&*inner.lock()?, key),
            Inner::Redb(inner) => inner.get(key),
            Inner::Postgres(inner) => inner.get(key),
        }
    }

    pub fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        match self.0.as_ref() {
            Inner::Sqlite(inner) => sqlite_backend::scan(&*inner.lock()?, prefix),
            Inner::Redb(inner) => inner.scan(prefix),
            Inner::Postgres(inner) => inner.scan(prefix),
        }
    }
}

impl Transaction<'_> {
    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        match &self.inner {
            TransactionInner::Sqlite(state) => sqlite_backend::get(state, key),
            TransactionInner::Redb(state) => state.get(key),
            TransactionInner::Postgres(state) => state.get(key),
        }
    }
    pub fn put<T: Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        match &mut self.inner {
            TransactionInner::Sqlite(state) => sqlite_backend::put(state, key, value),
            TransactionInner::Redb(state) => state.put(key, value),
            TransactionInner::Postgres(state) => state.put(key, value),
        }
    }
    pub fn delete(&mut self, key: &str) -> Result<()> {
        match &mut self.inner {
            TransactionInner::Sqlite(state) => sqlite_backend::delete(state, key),
            TransactionInner::Redb(state) => state.delete(key),
            TransactionInner::Postgres(state) => state.delete(key),
        }
    }
    pub fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        match &self.inner {
            TransactionInner::Sqlite(state) => sqlite_backend::scan(state, prefix),
            TransactionInner::Redb(state) => state.scan(prefix),
            TransactionInner::Postgres(state) => state.scan(prefix),
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
    fn sqlite_rollback_reopen_and_literal_prefixes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.sqlite3");
        let store = Store::open_sqlite(&path).unwrap();
        let result: Result<()> = store.transaction(|tx| {
            tx.put("session/x", &1)?;
            tx.put("dedupe/x", &2)?;
            Err(Fault::conflict("abort"))
        });
        assert!(result.is_err());
        assert_eq!(store.get::<u32>("session/x").unwrap(), None);
        assert_eq!(store.get::<u32>("dedupe/x").unwrap(), None);
        store
            .transaction(|tx| {
                tx.put("a%/2", &2)?;
                tx.put("a%/1", &1)?;
                tx.put("abc/3", &3)?;
                assert_eq!(tx.scan::<u32>("a%/")?, vec![1, 2]);
                tx.delete("a%/1")?;
                Ok(())
            })
            .unwrap();
        drop(store);
        let store = Store::open_sqlite(&path).unwrap();
        assert_eq!(store.scan::<u32>("a%/").unwrap(), vec![2]);
        assert_eq!(store.scan::<u32>("").unwrap(), vec![2, 3]);
        assert_eq!(&std::fs::read(&path).unwrap()[..16], b"SQLite format 3\0");
    }

    #[test]
    fn sqlite_independent_connections_serialize_admission() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.sqlite3");
        let stores = [
            Store::open_sqlite(&path).unwrap(),
            Store::open_sqlite(&path).unwrap(),
        ];
        let workers: Vec<_> = stores
            .into_iter()
            .map(|store| {
                std::thread::spawn(move || {
                    for _ in 0..30 {
                        store
                            .transaction(|tx| {
                                let n: u32 = tx.get("counter")?.unwrap_or_default();
                                tx.put("counter", &(n + 1))
                            })
                            .unwrap();
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(
            Store::open_sqlite(&path)
                .unwrap()
                .get::<u32>("counter")
                .unwrap(),
            Some(60)
        );
    }

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
    #[test]
    fn tuple_ids_do_not_alias() {
        assert_ne!(key("x", &["a/b", "c"]), key("x", &["a", "b/c"]));
    }
}
