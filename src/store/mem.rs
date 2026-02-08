//! In-memory hot storage backed by DashMap.
//!
//! Provides the fastest possible lookups for frequently accessed symbols
//! and their hypervectors. All data is lost on process exit.

use dashmap::DashMap;

use crate::symbol::SymbolId;

/// Concurrent in-memory store using a sharded hashmap.
#[derive(Debug)]
pub struct MemStore {
    data: DashMap<SymbolId, Vec<u8>>,
}

impl MemStore {
    /// Create an empty in-memory store.
    pub fn new() -> Self {
        Self {
            data: DashMap::new(),
        }
    }

    /// Create a store with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: DashMap::with_capacity(capacity),
        }
    }

    /// Insert or replace a value.
    pub fn put(&self, key: SymbolId, value: Vec<u8>) {
        self.data.insert(key, value);
    }

    /// Get a clone of the stored value.
    pub fn get(&self, key: SymbolId) -> Option<Vec<u8>> {
        self.data.get(&key).map(|v| v.value().clone())
    }

    /// Check if a key exists.
    pub fn contains(&self, key: SymbolId) -> bool {
        self.data.contains_key(&key)
    }

    /// Remove a key and return its value.
    pub fn remove(&self, key: SymbolId) -> Option<Vec<u8>> {
        self.data.remove(&key).map(|(_, v)| v)
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Iterate over all keys (snapshot — not a consistent view under concurrent writes).
    pub fn keys(&self) -> Vec<SymbolId> {
        self.data.iter().map(|entry| *entry.key()).collect()
    }
}

impl Default for MemStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_and_get() {
        let store = MemStore::new();
        let id = SymbolId::new(1).unwrap();
        store.put(id, vec![10, 20]);
        assert_eq!(store.get(id), Some(vec![10, 20]));
    }

    #[test]
    fn overwrite() {
        let store = MemStore::new();
        let id = SymbolId::new(1).unwrap();
        store.put(id, vec![1]);
        store.put(id, vec![2]);
        assert_eq!(store.get(id), Some(vec![2]));
    }

    #[test]
    fn remove() {
        let store = MemStore::new();
        let id = SymbolId::new(1).unwrap();
        store.put(id, vec![1]);
        assert_eq!(store.remove(id), Some(vec![1]));
        assert!(!store.contains(id));
    }

    #[test]
    fn concurrent_access() {
        use std::sync::Arc;
        let store = Arc::new(MemStore::new());
        let handles: Vec<_> = (1..=100)
            .map(|i| {
                let store = Arc::clone(&store);
                std::thread::spawn(move || {
                    let id = SymbolId::new(i).unwrap();
                    store.put(id, vec![i as u8]);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(store.len(), 100);
    }
}
