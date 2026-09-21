//! redb-backed implementation of the `Store`/`Transaction` API. This is the
//! original (and still default) backend: one file on disk per store kind.
use crate::{Result, failure};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Serialize, de::DeserializeOwned};
use std::path::Path;

pub(crate) const RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("records_v1");

pub(crate) struct RedbInner(Database);

pub(crate) struct RedbTxState<'a> {
    table: redb::Table<'a, &'static str, &'static [u8]>,
    wrote: bool,
}

impl<'a> RedbTxState<'a> {
    pub(crate) fn new(table: redb::Table<'a, &'static str, &'static [u8]>) -> Self {
        Self {
            table,
            wrote: false,
        }
    }

    /// Whether any `put` or `delete` ran in this transaction; see
    /// [`crate::Store::changes`].
    pub(crate) fn wrote(&self) -> bool {
        self.wrote
    }
}

impl RedbInner {
    /// Exposes the underlying database so `Store::transaction` can drive the
    /// write/commit lifecycle itself (needed so it can hand out a `&mut
    /// RedbTxState` with a single natural lifetime, rather than going
    /// through a generic closure boundary that would need two independent
    /// lifetime parameters on the public `Transaction<'a>` type — which
    /// would break every call site's `Transaction<'_>` elision).
    pub(crate) fn database(&self) -> &Database {
        &self.0
    }

    pub(crate) fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(failure)?;
        }
        let db = Database::create(path).map_err(failure)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(failure)?;
        }
        let inner = Self(db);
        inner.transaction(|_| Ok(()))?;
        Ok(inner)
    }

    pub(crate) fn transaction<T>(
        &self,
        action: impl FnOnce(&mut RedbTxState<'_>) -> Result<T>,
    ) -> Result<T> {
        let write = self.0.begin_write().map_err(failure)?;
        let result = {
            let table = write.open_table(RECORDS).map_err(failure)?;
            action(&mut RedbTxState::new(table))?
        };
        write.commit().map_err(failure)?;
        Ok(result)
    }

    pub(crate) fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let read = self.0.begin_read().map_err(failure)?;
        let table = read.open_table(RECORDS).map_err(failure)?;
        table
            .get(key)
            .map_err(failure)?
            .map(|v| serde_json::from_slice(v.value()).map_err(failure))
            .transpose()
    }

    pub(crate) fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        let read = self.0.begin_read().map_err(failure)?;
        let table = read.open_table(RECORDS).map_err(failure)?;
        let mut values = Vec::new();
        for entry in table.range(prefix..).map_err(failure)? {
            let (key, value) = entry.map_err(failure)?;
            if !key.value().starts_with(prefix) {
                break;
            }
            values.push(serde_json::from_slice(value.value()).map_err(failure)?);
        }
        Ok(values)
    }
}

impl RedbTxState<'_> {
    pub(crate) fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        self.table
            .get(key)
            .map_err(failure)?
            .map(|v| serde_json::from_slice(v.value()).map_err(failure))
            .transpose()
    }
    pub(crate) fn put<T: Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        let data = serde_json::to_vec(value).map_err(failure)?;
        self.table.insert(key, data.as_slice()).map_err(failure)?;
        self.wrote = true;
        Ok(())
    }
    pub(crate) fn delete(&mut self, key: &str) -> Result<()> {
        self.table.remove(key).map_err(failure)?;
        self.wrote = true;
        Ok(())
    }
    pub(crate) fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
        let mut values = Vec::new();
        for entry in self.table.range(prefix..).map_err(failure)? {
            let (key, value) = entry.map_err(failure)?;
            if !key.value().starts_with(prefix) {
                break;
            }
            values.push(serde_json::from_slice(value.value()).map_err(failure)?);
        }
        Ok(values)
    }
}

/// Reads every raw `(key, value)` pair out of a redb store file without
/// deserializing the value (the file holds records of many different
/// serde types keyed by namespace, so there is no single `T` to deserialize
/// to). Used only for the one-shot import into the Postgres backend.
pub(crate) fn scan_all_raw(path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let db = Database::open(path).map_err(failure)?;
    let read = db.begin_read().map_err(failure)?;
    let table = read.open_table(RECORDS).map_err(failure)?;
    let mut values = Vec::new();
    for entry in table.range::<&str>(..).map_err(failure)? {
        let (key, value) = entry.map_err(failure)?;
        values.push((key.value().to_string(), value.value().to_vec()));
    }
    Ok(values)
}
