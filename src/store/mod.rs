//! Tiered storage system for akh-medu.
//!
//! Three storage tiers serve different access patterns:
//!
//! - [`MemStore`] — hot data in concurrent hashmaps (DashMap)
//! - [`MmapStore`] — warm data in memory-mapped files (memmap2)
//! - [`DurableStore`] — cold/metadata in ACID transactions (redb)
//!
//! [`TieredStore`] composes these tiers with automatic promotion/demotion.

pub mod durable;
pub mod mem;
pub mod mmap;

use crate::error::StoreError;
use crate::symbol::SymbolId;

/// Result type for store operations.
pub type StoreResult<T> = std::result::Result<T, StoreError>;

/// Reference to stored data — either borrowed (zero-copy) or owned.
#[derive(Debug)]
pub enum StoredRef<'a> {
    /// Zero-copy reference into a memory map or cache.
    Borrowed(&'a [u8]),
    /// Owned copy (e.g., from redb or after deserialization).
    Owned(Vec<u8>),
}

impl<'a> AsRef<[u8]> for StoredRef<'a> {
    fn as_ref(&self) -> &[u8] {
        match self {
            StoredRef::Borrowed(b) => b,
            StoredRef::Owned(v) => v,
        }
    }
}

impl<'a> StoredRef<'a> {
    /// Convert to an owned `Vec<u8>`, copying if necessary.
    pub fn into_owned(self) -> Vec<u8> {
        match self {
            StoredRef::Borrowed(b) => b.to_vec(),
            StoredRef::Owned(v) => v,
        }
    }
}

/// Composable tiered store: hot (mem) → warm (mmap) → cold (durable).
///
/// Reads check tiers in order; writes go to the hot tier and can be
/// flushed down. The durable tier is used for metadata and provenance.
pub struct TieredStore {
    pub hot: mem::MemStore,
    pub warm: Option<mmap::MmapStore>,
    pub durable: Option<durable::DurableStore>,
}

impl TieredStore {
    /// Create a memory-only tiered store (no persistence).
    pub fn memory_only() -> Self {
        Self {
            hot: mem::MemStore::new(),
            warm: None,
            durable: None,
        }
    }

    /// Create a fully tiered store with all backends.
    pub fn with_persistence(data_dir: &std::path::Path, mmap_name: &str) -> StoreResult<Self> {
        let warm = mmap::MmapStore::open_or_create(data_dir, mmap_name)?;
        let durable = durable::DurableStore::open(data_dir)?;
        Ok(Self {
            hot: mem::MemStore::new(),
            warm: Some(warm),
            durable: Some(durable),
        })
    }

    /// Store a value in the hot tier.
    pub fn put(&self, key: SymbolId, value: Vec<u8>) {
        self.hot.put(key, value);
    }

    /// Get a value, checking hot → warm tiers.
    pub fn get(&self, key: SymbolId) -> Option<Vec<u8>> {
        if let Some(v) = self.hot.get(key) {
            return Some(v);
        }
        if let Some(warm) = &self.warm {
            if let Some(v) = warm.get(key) {
                // Promote to hot on read
                self.hot.put(key, v.clone());
                return Some(v);
            }
        }
        None
    }

    /// Check if a key exists in any tier.
    pub fn contains(&self, key: SymbolId) -> bool {
        self.hot.contains(key) || self.warm.as_ref().is_some_and(|w| w.contains(key))
    }

    /// Number of entries in the hot tier.
    pub fn hot_len(&self) -> usize {
        self.hot.len()
    }

    /// Store metadata in the durable tier.
    pub fn put_meta(&self, key: &[u8], value: &[u8]) -> StoreResult<()> {
        match &self.durable {
            Some(d) => d.put(key, value),
            None => Ok(()), // no-op without durable backend
        }
    }

    /// Get metadata from the durable tier.
    pub fn get_meta(&self, key: &[u8]) -> StoreResult<Option<Vec<u8>>> {
        match &self.durable {
            Some(d) => d.get(key),
            None => Ok(None),
        }
    }

    /// Scan all keys with the given prefix from the durable tier.
    ///
    /// Returns `(key, value)` pairs. Returns an empty vec if no durable
    /// backend is configured.
    pub fn scan_prefix(&self, prefix: &[u8]) -> StoreResult<Vec<(Vec<u8>, Vec<u8>)>> {
        match &self.durable {
            Some(d) => d.scan_prefix(prefix),
            None => Ok(vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiered_store_memory_only() {
        let store = TieredStore::memory_only();
        let id = SymbolId::new(1).unwrap();
        store.put(id, vec![1, 2, 3]);
        assert!(store.contains(id));
        assert_eq!(store.get(id), Some(vec![1, 2, 3]));
        assert_eq!(store.hot_len(), 1);
    }

    #[test]
    fn tiered_store_missing_key() {
        let store = TieredStore::memory_only();
        let id = SymbolId::new(999).unwrap();
        assert!(!store.contains(id));
        assert_eq!(store.get(id), None);
    }

    #[test]
    fn stored_ref_into_owned() {
        let data = vec![10, 20, 30];
        let borrowed = StoredRef::Borrowed(&data);
        assert_eq!(borrowed.as_ref(), &[10, 20, 30]);
        let owned = StoredRef::Owned(vec![40, 50]);
        assert_eq!(owned.into_owned(), vec![40, 50]);
    }
}
