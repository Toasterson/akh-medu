//! ACID-durable key-value store backed by redb.
//!
//! Used for metadata, provenance records, and any data that must survive
//! crashes. Provides full transactional guarantees.

use std::path::Path;
use std::sync::Arc;

use redb::{Database, TableDefinition};

use crate::error::StoreError;
use crate::store::StoreResult;

/// Table for general metadata (string keys → binary values).
const META_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("meta");

/// ACID-durable store using redb.
///
/// All writes go through transactions. Reads use MVCC snapshots.
pub struct DurableStore {
    db: Arc<Database>,
}

impl DurableStore {
    /// Open or create a durable store in the given directory.
    pub fn open(data_dir: &Path) -> StoreResult<Self> {
        std::fs::create_dir_all(data_dir).map_err(|e| StoreError::Io { source: e })?;
        let db_path = data_dir.join("akh-medu.redb");
        let db = Database::create(&db_path).map_err(|e| StoreError::Redb {
            message: format!("failed to open redb at {}: {e}", db_path.display()),
        })?;
        Ok(Self { db: Arc::new(db) })
    }

    /// Store a key-value pair with full ACID guarantees.
    pub fn put(&self, key: &[u8], value: &[u8]) -> StoreResult<()> {
        let txn = self.db.begin_write().map_err(|e| StoreError::Redb {
            message: format!("begin_write failed: {e}"),
        })?;
        {
            let mut table = txn.open_table(META_TABLE).map_err(|e| StoreError::Redb {
                message: format!("open_table failed: {e}"),
            })?;
            table.insert(key, value).map_err(|e| StoreError::Redb {
                message: format!("insert failed: {e}"),
            })?;
        }
        txn.commit().map_err(|e| StoreError::Redb {
            message: format!("commit failed: {e}"),
        })?;
        Ok(())
    }

    /// Read a value by key. Returns `Ok(None)` if the key doesn't exist.
    pub fn get(&self, key: &[u8]) -> StoreResult<Option<Vec<u8>>> {
        let txn = self.db.begin_read().map_err(|e| StoreError::Redb {
            message: format!("begin_read failed: {e}"),
        })?;
        let table = txn.open_table(META_TABLE).map_err(|e| StoreError::Redb {
            message: format!("open_table failed: {e}"),
        })?;
        let result = table.get(key).map_err(|e| StoreError::Redb {
            message: format!("get failed: {e}"),
        })?;
        Ok(result.map(|guard| guard.value().to_vec()))
    }

    /// Delete a key. Returns whether the key existed.
    pub fn remove(&self, key: &[u8]) -> StoreResult<bool> {
        let txn = self.db.begin_write().map_err(|e| StoreError::Redb {
            message: format!("begin_write failed: {e}"),
        })?;
        let existed = {
            let mut table = txn.open_table(META_TABLE).map_err(|e| StoreError::Redb {
                message: format!("open_table failed: {e}"),
            })?;
            let result = table.remove(key).map_err(|e| StoreError::Redb {
                message: format!("remove failed: {e}"),
            })?;
            result.is_some()
        };
        txn.commit().map_err(|e| StoreError::Redb {
            message: format!("commit failed: {e}"),
        })?;
        Ok(existed)
    }

    /// Store multiple key-value pairs in a single transaction.
    pub fn put_batch(&self, entries: &[(&[u8], &[u8])]) -> StoreResult<()> {
        let txn = self.db.begin_write().map_err(|e| StoreError::Redb {
            message: format!("begin_write failed: {e}"),
        })?;
        {
            let mut table = txn.open_table(META_TABLE).map_err(|e| StoreError::Redb {
                message: format!("open_table failed: {e}"),
            })?;
            for (key, value) in entries {
                table.insert(*key, *value).map_err(|e| StoreError::Redb {
                    message: format!("insert failed: {e}"),
                })?;
            }
        }
        txn.commit().map_err(|e| StoreError::Redb {
            message: format!("commit failed: {e}"),
        })?;
        Ok(())
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &[u8]) -> StoreResult<bool> {
        self.get(key).map(|v| v.is_some())
    }

    /// Scan all keys with the given prefix, returning `(key, value)` pairs.
    ///
    /// Uses redb range queries with prefix bounds to efficiently iterate
    /// only the matching key space.
    pub fn scan_prefix(&self, prefix: &[u8]) -> StoreResult<Vec<(Vec<u8>, Vec<u8>)>> {
        let txn = self.db.begin_read().map_err(|e| StoreError::Redb {
            message: format!("begin_read failed: {e}"),
        })?;
        let table = txn.open_table(META_TABLE).map_err(|e| StoreError::Redb {
            message: format!("open_table failed: {e}"),
        })?;

        // Compute the exclusive upper bound by incrementing the last byte.
        // If the last byte is 0xFF, try the second-to-last, etc.
        // If all bytes are 0xFF, we use an unbounded range (scan to end).
        let end_bound = increment_prefix(prefix);

        let mut results = Vec::new();

        let iter = match &end_bound {
            Some(end) => {
                table
                    .range::<&[u8]>(prefix..end.as_slice())
                    .map_err(|e| StoreError::Redb {
                        message: format!("range scan failed: {e}"),
                    })?
            }
            None => {
                table
                    .range::<&[u8]>(prefix..)
                    .map_err(|e| StoreError::Redb {
                        message: format!("range scan failed: {e}"),
                    })?
            }
        };

        for entry in iter {
            let (k, v) = entry.map_err(|e| StoreError::Redb {
                message: format!("range iteration failed: {e}"),
            })?;
            let key_bytes = k.value().to_vec();
            // Double-check prefix match (for the unbounded case).
            if !key_bytes.starts_with(prefix) {
                break;
            }
            results.push((key_bytes, v.value().to_vec()));
        }

        Ok(results)
    }

    /// Get a reference to the underlying database (for custom table operations).
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Get a shared handle to the underlying database.
    ///
    /// Used by the provenance ledger to open its own tables on the same DB.
    pub fn database_arc(&self) -> Arc<Database> {
        Arc::clone(&self.db)
    }
}

/// Compute the exclusive upper bound for a prefix scan.
///
/// Increments the last byte of the prefix. If it overflows (0xFF),
/// truncates and carries. Returns `None` if all bytes are 0xFF (scan to end).
fn increment_prefix(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return Some(end);
        }
        end.pop();
    }
    None
}

impl std::fmt::Debug for DurableStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DurableStore").finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn put_get_remove() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();

        store.put(b"hello", b"world").unwrap();
        assert_eq!(store.get(b"hello").unwrap(), Some(b"world".to_vec()));
        assert!(store.contains(b"hello").unwrap());

        assert!(store.remove(b"hello").unwrap());
        assert!(!store.contains(b"hello").unwrap());
        assert_eq!(store.get(b"hello").unwrap(), None);
    }

    #[test]
    fn overwrite_value() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();

        store.put(b"key", b"val1").unwrap();
        store.put(b"key", b"val2").unwrap();
        assert_eq!(store.get(b"key").unwrap(), Some(b"val2".to_vec()));
    }

    #[test]
    fn persistence_across_reopens() {
        let dir = TempDir::new().unwrap();

        {
            let store = DurableStore::open(dir.path()).unwrap();
            store.put(b"persist_key", b"persist_val").unwrap();
        }

        let store = DurableStore::open(dir.path()).unwrap();
        assert_eq!(
            store.get(b"persist_key").unwrap(),
            Some(b"persist_val".to_vec())
        );
    }

    #[test]
    fn remove_nonexistent_key() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();
        assert!(!store.remove(b"nonexistent").unwrap());
    }

    #[test]
    fn scan_prefix_returns_matching_keys() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();

        store.put(b"sym_meta:1", b"data1").unwrap();
        store.put(b"sym_meta:2", b"data2").unwrap();
        store.put(b"sym_meta:10", b"data10").unwrap();
        store.put(b"other_key", b"other").unwrap();

        let results = store.scan_prefix(b"sym_meta:").unwrap();
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|(k, _)| k.starts_with(b"sym_meta:")));
    }

    #[test]
    fn scan_prefix_empty_result() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();

        store.put(b"other_key", b"data").unwrap();
        let results = store.scan_prefix(b"sym_meta:").unwrap();
        assert!(results.is_empty());
    }
}
