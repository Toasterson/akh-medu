//! One durable database per daemon. Multi-record decisions commit atomically.
use crate::types::{Fault, Result};
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Serialize, de::DeserializeOwned};
use std::{path::Path, sync::Arc};

const RECORDS: TableDefinition<&str, &[u8]> = TableDefinition::new("records_v1");

#[derive(Clone)]
pub struct Store(Arc<Database>);

pub struct Transaction<'a> {
    table: redb::Table<'a, &'static str, &'static [u8]>,
}

fn failure(error: impl std::fmt::Display) -> Fault {
    Fault::new("storage_error", error.to_string())
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
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
        let store = Self(Arc::new(db));
        store.transaction(|_| Ok(()))?;
        Ok(store)
    }

    pub fn transaction<T>(
        &self,
        action: impl FnOnce(&mut Transaction<'_>) -> Result<T>,
    ) -> Result<T> {
        let write = self.0.begin_write().map_err(failure)?;
        let result = {
            let table = write.open_table(RECORDS).map_err(failure)?;
            action(&mut Transaction { table })?
        };
        write.commit().map_err(failure)?;
        Ok(result)
    }

    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let read = self.0.begin_read().map_err(failure)?;
        let table = read.open_table(RECORDS).map_err(failure)?;
        table
            .get(key)
            .map_err(failure)?
            .map(|v| serde_json::from_slice(v.value()).map_err(failure))
            .transpose()
    }

    pub fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
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

impl Transaction<'_> {
    pub fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        self.table
            .get(key)
            .map_err(failure)?
            .map(|v| serde_json::from_slice(v.value()).map_err(failure))
            .transpose()
    }
    pub fn put<T: Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        let data = serde_json::to_vec(value).map_err(failure)?;
        self.table.insert(key, data.as_slice()).map_err(failure)?;
        Ok(())
    }
    pub fn delete(&mut self, key: &str) -> Result<()> {
        self.table.remove(key).map_err(failure)?;
        Ok(())
    }
    pub fn scan<T: DeserializeOwned>(&self, prefix: &str) -> Result<Vec<T>> {
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
