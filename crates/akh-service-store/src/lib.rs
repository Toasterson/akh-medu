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
pub mod neural_spans;
mod redb_backend;

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
    Redb(RedbInner),
    Postgres(PostgresInner),
}

#[derive(Clone)]
pub struct Store(Arc<Inner>);

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
            Inner::Redb(inner) => inner.get(key),
            Inner::Postgres(inner) => inner.get(key),
        }
    }

    pub fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        match self.0.as_ref() {
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
    #[test]
    fn tuple_ids_do_not_alias() {
        assert_ne!(key("x", &["a/b", "c"]), key("x", &["a", "b/c"]));
    }
}
