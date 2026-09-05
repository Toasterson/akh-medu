//! ACID-durable key-value store backed by redb.
//!
//! Used for metadata, provenance records, and any data that must survive
//! crashes. Provides full transactional guarantees.
//!
//! ## Concurrency
//!
//! The underlying redb `Database` is wrapped in `Arc<Mutex<Database>>` to
//! serialize write access at the Rust Mutex level. This prevents thread
//! pile-up on redb's internal file lock, which on macOS can exhaust kernel
//! resources and cause SIGKILL when many concurrent MCP calls hit the same
//! database.
//!
//! Reads lock the mutex only long enough to create a read transaction (MVCC
//! snapshot), then release it so other reads can proceed concurrently.
//! Writes hold the mutex for the full transaction lifetime.
//!
//! ## Calling from async contexts
//!
//! All write methods (`put`, `remove`, `put_batch`, `remove_batch`) acquire
//! a `std::sync::Mutex`, which **blocks the calling thread**.  Callers from
//! an async Tokio context must therefore be inside either
//! `tokio::task::spawn_blocking` or `tokio::task::block_in_place`; calling
//! them directly on an async task thread starves the Tokio runtime and can
//! prevent MCP handlers from receiving new requests.

use std::path::Path;
use std::sync::{Arc, Mutex};

use redb::{Database, ReadableTable, TableDefinition};

use crate::error::StoreError;
use crate::store::StoreResult;

/// Table for general metadata (string keys → binary values).
const META_TABLE: TableDefinition<&[u8], &[u8]> = TableDefinition::new("meta");

/// ACID-durable store using redb.
///
/// All writes go through transactions. Reads use MVCC snapshots.
///
/// The database is wrapped in `Arc<Mutex<Database>>` to serialize concurrent
/// write attempts at the Rust level, avoiding OS-level lock contention.
pub struct DurableStore {
    db: Arc<Mutex<Database>>,
}

impl DurableStore {
    /// Open or create a durable store in the given directory.
    pub fn open(data_dir: &Path) -> StoreResult<Self> {
        std::fs::create_dir_all(data_dir).map_err(|e| StoreError::Io { source: e })?;
        let db_path = data_dir.join("akh-medu.redb");
        let db = Database::create(&db_path).map_err(|e| StoreError::Redb {
            message: format!("failed to open redb at {}: {e}", db_path.display()),
        })?;
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
        })
    }

    /// Store a key-value pair with full ACID guarantees.
    pub fn put(&self, key: &[u8], value: &[u8]) -> StoreResult<()> {
        let db = self.db.lock().expect("redb mutex poisoned");
        let txn = db.begin_write().map_err(|e| StoreError::Redb {
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

    /// Atomically replace `key` only when its current bytes equal `expected`.
    ///
    /// The comparison and insert share one redb write transaction, so this is
    /// safe across threads and across processes opening the same database.
    pub fn compare_and_swap(
        &self,
        key: &[u8],
        expected: Option<&[u8]>,
        value: &[u8],
    ) -> StoreResult<bool> {
        let db = self.db.lock().expect("redb mutex poisoned");
        let txn = db.begin_write().map_err(|e| StoreError::Redb {
            message: format!("begin_write failed: {e}"),
        })?;
        let swapped = {
            let mut table = txn.open_table(META_TABLE).map_err(|e| StoreError::Redb {
                message: format!("open_table failed: {e}"),
            })?;
            let current = table
                .get(key)
                .map_err(|e| StoreError::Redb {
                    message: format!("get failed: {e}"),
                })?
                .map(|guard| guard.value().to_vec());
            if current.as_deref() != expected {
                false
            } else {
                table.insert(key, value).map_err(|e| StoreError::Redb {
                    message: format!("insert failed: {e}"),
                })?;
                true
            }
        };
        if swapped {
            txn.commit().map_err(|e| StoreError::Redb {
                message: format!("commit failed: {e}"),
            })?;
        }
        Ok(swapped)
    }

    /// Read a value by key. Returns `Ok(None)` if the key doesn't exist
    /// or if the table hasn't been created yet (fresh database).
    ///
    /// Locks the mutex only long enough to begin the read transaction;
    /// the MVCC snapshot is then used without holding the lock.
    pub fn get(&self, key: &[u8]) -> StoreResult<Option<Vec<u8>>> {
        let db = self.db.lock().expect("redb mutex poisoned");
        let txn = db.begin_read().map_err(|e| StoreError::Redb {
            message: format!("begin_read failed: {e}"),
        })?;
        drop(db); // release lock — read txn is independent
        let table = match txn.open_table(META_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(None),
            Err(e) => {
                return Err(StoreError::Redb {
                    message: format!("open_table failed: {e}"),
                });
            }
        };
        let result = table.get(key).map_err(|e| StoreError::Redb {
            message: format!("get failed: {e}"),
        })?;
        Ok(result.map(|guard| guard.value().to_vec()))
    }

    /// Delete a key. Returns whether the key existed.
    pub fn remove(&self, key: &[u8]) -> StoreResult<bool> {
        let db = self.db.lock().expect("redb mutex poisoned");
        let txn = db.begin_write().map_err(|e| StoreError::Redb {
            message: format!("begin_write failed: {e}"),
        })?;
        let existed = {
            let mut table = match txn.open_table(META_TABLE) {
                Ok(t) => t,
                Err(redb::TableError::TableDoesNotExist(_)) => return Ok(false),
                Err(e) => {
                    return Err(StoreError::Redb {
                        message: format!("open_table failed: {e}"),
                    });
                }
            };
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
        let db = self.db.lock().expect("redb mutex poisoned");
        let txn = db.begin_write().map_err(|e| StoreError::Redb {
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

    /// Remove multiple keys in a single transaction.
    ///
    /// Returns the number of keys that were actually removed (existed and
    /// were deleted). Keys that don't exist are silently skipped.
    pub fn remove_batch(&self, keys: &[&[u8]]) -> StoreResult<usize> {
        let db = self.db.lock().expect("redb mutex poisoned");
        let txn = db.begin_write().map_err(|e| StoreError::Redb {
            message: format!("begin_write failed: {e}"),
        })?;
        let count = {
            let mut table = match txn.open_table(META_TABLE) {
                Ok(t) => t,
                Err(redb::TableError::TableDoesNotExist(_)) => return Ok(0),
                Err(e) => {
                    return Err(StoreError::Redb {
                        message: format!("open_table failed: {e}"),
                    });
                }
            };
            let mut removed = 0usize;
            for key in keys {
                let result = table.remove(*key).map_err(|e| StoreError::Redb {
                    message: format!("remove failed: {e}"),
                })?;
                if result.is_some() {
                    removed += 1;
                }
            }
            removed
        };
        txn.commit().map_err(|e| StoreError::Redb {
            message: format!("commit failed: {e}"),
        })?;
        Ok(count)
    }

    /// Atomically replace every entry under `prefix` with `entries`.
    ///
    /// Snapshot restore uses this for authoritative namespaces such as the
    /// symbol registry.  A scan followed by independent remove/put
    /// transactions leaves a crash window where readers can observe a partial
    /// registry; keeping the delete and insert phases in one redb write
    /// transaction makes the replacement all-or-nothing.
    pub fn replace_prefix(&self, prefix: &[u8], entries: &[(&[u8], &[u8])]) -> StoreResult<()> {
        if entries.iter().any(|(key, _)| !key.starts_with(prefix)) {
            return Err(StoreError::Serialization {
                message: "replacement entry is outside the requested metadata prefix".into(),
            });
        }

        let db = self.db.lock().expect("redb mutex poisoned");
        let txn = db.begin_write().map_err(|e| StoreError::Redb {
            message: format!("begin_write failed: {e}"),
        })?;
        {
            let mut table = txn.open_table(META_TABLE).map_err(|e| StoreError::Redb {
                message: format!("open_table failed: {e}"),
            })?;
            let end_bound = increment_prefix(prefix);
            let stale_keys = {
                let iter = match &end_bound {
                    Some(end) => table.range::<&[u8]>(prefix..end.as_slice()).map_err(|e| {
                        StoreError::Redb {
                            message: format!("range scan failed: {e}"),
                        }
                    })?,
                    None => table
                        .range::<&[u8]>(prefix..)
                        .map_err(|e| StoreError::Redb {
                            message: format!("range scan failed: {e}"),
                        })?,
                };
                let mut keys = Vec::new();
                for entry in iter {
                    let (key, _) = entry.map_err(|e| StoreError::Redb {
                        message: format!("range iteration failed: {e}"),
                    })?;
                    let key = key.value().to_vec();
                    if !key.starts_with(prefix) {
                        break;
                    }
                    keys.push(key);
                }
                keys
            };
            for key in stale_keys {
                table.remove(key.as_slice()).map_err(|e| StoreError::Redb {
                    message: format!("remove failed: {e}"),
                })?;
            }
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
        let db = self.db.lock().expect("redb mutex poisoned");
        let txn = db.begin_read().map_err(|e| StoreError::Redb {
            message: format!("begin_read failed: {e}"),
        })?;
        drop(db); // release lock — read txn is independent
        let table = match txn.open_table(META_TABLE) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(Vec::new()),
            Err(e) => {
                return Err(StoreError::Redb {
                    message: format!("open_table failed: {e}"),
                });
            }
        };

        // Compute the exclusive upper bound by incrementing the last byte.
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
            None => table
                .range::<&[u8]>(prefix..)
                .map_err(|e| StoreError::Redb {
                    message: format!("range scan failed: {e}"),
                })?,
        };

        for entry in iter {
            let (k, v) = entry.map_err(|e| StoreError::Redb {
                message: format!("range iteration failed: {e}"),
            })?;
            let key_bytes = k.value().to_vec();
            if !key_bytes.starts_with(prefix) {
                break;
            }
            results.push((key_bytes, v.value().to_vec()));
        }

        Ok(results)
    }

    /// Get a shared handle to the database mutex.
    ///
    /// Used by provenance ledger, audit ledger, and other subsystems
    /// that write to their own tables on the same database.
    pub fn database_arc(&self) -> Arc<Mutex<Database>> {
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
    fn remove_batch_basic() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();

        store.put(b"k1", b"v1").unwrap();
        store.put(b"k2", b"v2").unwrap();
        store.put(b"k3", b"v3").unwrap();
        store.put(b"k4", b"v4").unwrap();

        let removed = store.remove_batch(&[b"k1", b"k3"]).unwrap();
        assert_eq!(removed, 2);
        assert_eq!(store.get(b"k1").unwrap(), None);
        assert_eq!(store.get(b"k2").unwrap(), Some(b"v2".to_vec()));
        assert_eq!(store.get(b"k3").unwrap(), None);
        assert_eq!(store.get(b"k4").unwrap(), Some(b"v4".to_vec()));
    }

    #[test]
    fn replace_prefix_removes_stale_keys_and_preserves_other_namespaces() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();
        store.put(b"sym_meta:1", b"old-one").unwrap();
        store.put(b"sym_meta:99", b"stale").unwrap();
        store.put(b"agent:state", b"keep").unwrap();

        store
            .replace_prefix(
                b"sym_meta:",
                &[(b"sym_meta:1", b"new-one"), (b"sym_meta:2", b"new-two")],
            )
            .unwrap();

        assert_eq!(store.get(b"sym_meta:1").unwrap(), Some(b"new-one".to_vec()));
        assert_eq!(store.get(b"sym_meta:2").unwrap(), Some(b"new-two".to_vec()));
        assert_eq!(store.get(b"sym_meta:99").unwrap(), None);
        assert_eq!(store.get(b"agent:state").unwrap(), Some(b"keep".to_vec()));
    }

    #[test]
    fn remove_batch_partial_nonexistent() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();

        store.put(b"exists", b"data").unwrap();

        let removed = store.remove_batch(&[b"exists", b"nope"]).unwrap();
        assert_eq!(removed, 1);
        assert_eq!(store.get(b"exists").unwrap(), None);
    }

    #[test]
    fn remove_batch_empty() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();

        store.put(b"key", b"val").unwrap();
        let removed = store.remove_batch(&[]).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(store.get(b"key").unwrap(), Some(b"val".to_vec()));
    }

    #[test]
    fn remove_batch_all_nonexistent() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();
        let removed = store.remove_batch(&[b"ghost1", b"ghost2"]).unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn scan_prefix_empty_result() {
        let dir = TempDir::new().unwrap();
        let store = DurableStore::open(dir.path()).unwrap();

        store.put(b"other_key", b"data").unwrap();
        let results = store.scan_prefix(b"sym_meta:").unwrap();
        assert!(results.is_empty());
    }

    /// Verify that concurrent writes from multiple threads are serialized
    /// correctly by the Arc<Mutex<Database>> wrapper. No data should be lost
    /// and no "multiple active write transactions" error should occur.
    #[test]
    fn concurrent_writes_are_serialized() {
        use std::sync::Arc;
        use std::thread;

        let dir = TempDir::new().unwrap();
        let store = Arc::new(DurableStore::open(dir.path()).unwrap());

        let handles: Vec<_> = (0u8..8)
            .map(|i| {
                let s = Arc::clone(&store);
                thread::spawn(move || {
                    let key = [b'k', i];
                    let val = [b'v', i];
                    s.put(&key, &val).expect("concurrent put should not fail");
                })
            })
            .collect();

        for h in handles {
            h.join().expect("thread panicked");
        }

        // All 8 keys must be present.
        for i in 0u8..8 {
            let key = [b'k', i];
            let val = [b'v', i];
            assert_eq!(
                store.get(&key).unwrap(),
                Some(val.to_vec()),
                "key {i} missing after concurrent writes",
            );
        }
    }
}
