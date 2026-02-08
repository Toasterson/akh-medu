//! Symbol registry: bidirectional label ↔ ID mapping.
//!
//! The [`SymbolRegistry`] provides O(1) lookups in both directions
//! using two `DashMap`s. Labels are normalized to lowercase for
//! case-insensitive matching.

use dashmap::DashMap;

use crate::error::{AkhResult, SymbolError};
use crate::store::TieredStore;
use crate::symbol::{SymbolId, SymbolMeta};

/// Bidirectional symbol registry mapping IDs to metadata and labels to IDs.
pub struct SymbolRegistry {
    /// Forward map: SymbolId → SymbolMeta (source of truth).
    id_to_meta: DashMap<SymbolId, SymbolMeta>,
    /// Reverse map: normalized label (lowercase) → SymbolId.
    label_to_id: DashMap<String, SymbolId>,
}

impl SymbolRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            id_to_meta: DashMap::new(),
            label_to_id: DashMap::new(),
        }
    }

    /// Register a symbol. Errors if the label is already taken.
    pub fn register(&self, meta: SymbolMeta) -> AkhResult<()> {
        let normalized = meta.label.to_lowercase();

        // Check for duplicate label.
        if let Some(existing) = self.label_to_id.get(&normalized) {
            return Err(SymbolError::DuplicateLabel {
                label: meta.label.clone(),
                existing_id: existing.value().get(),
            }
            .into());
        }

        self.label_to_id.insert(normalized, meta.id);
        self.id_to_meta.insert(meta.id, meta);
        Ok(())
    }

    /// Look up symbol metadata by ID.
    pub fn get(&self, id: SymbolId) -> Option<SymbolMeta> {
        self.id_to_meta.get(&id).map(|r| r.value().clone())
    }

    /// Look up a symbol ID by label (case-insensitive).
    pub fn lookup(&self, label: &str) -> Option<SymbolId> {
        let normalized = label.to_lowercase();
        self.label_to_id.get(&normalized).map(|r| *r.value())
    }

    /// Look up symbol metadata by label (case-insensitive).
    pub fn lookup_meta(&self, label: &str) -> Option<SymbolMeta> {
        let id = self.lookup(label)?;
        self.get(id)
    }

    /// Return all registered symbols.
    pub fn all(&self) -> Vec<SymbolMeta> {
        self.id_to_meta.iter().map(|r| r.value().clone()).collect()
    }

    /// Number of registered symbols.
    pub fn len(&self) -> usize {
        self.id_to_meta.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.id_to_meta.is_empty()
    }

    /// Resolve a label to a human-readable string, falling back to `sym:{id}`.
    pub fn resolve_label(&self, id: SymbolId) -> String {
        self.get(id)
            .map(|m| m.label.clone())
            .unwrap_or_else(|| format!("sym:{}", id.get()))
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    /// Persist all registry entries to the durable store.
    ///
    /// Each entry is written as `b"sym_meta:<id>"` → bincode-encoded `SymbolMeta`.
    /// Uses a single batch transaction instead of N individual writes.
    pub fn persist(&self, store: &TieredStore) -> AkhResult<()> {
        if let Some(ref durable) = store.durable {
            let mut keys = Vec::with_capacity(self.id_to_meta.len());
            let mut values = Vec::with_capacity(self.id_to_meta.len());
            for entry in self.id_to_meta.iter() {
                let meta = entry.value();
                let key = format!("sym_meta:{}", meta.id.get());
                let encoded = bincode::serialize(meta).map_err(|e| {
                    crate::error::StoreError::Serialization {
                        message: format!("failed to serialize symbol meta: {e}"),
                    }
                })?;
                keys.push(key);
                values.push(encoded);
            }
            let entries: Vec<(&[u8], &[u8])> = keys
                .iter()
                .zip(values.iter())
                .map(|(k, v)| (k.as_bytes(), v.as_slice()))
                .collect();
            durable.put_batch(&entries)?;
        }
        Ok(())
    }

    /// Restore the registry from the durable store by scanning `sym_meta:` prefix.
    pub fn restore(store: &TieredStore) -> AkhResult<Self> {
        let registry = Self::new();

        if let Some(ref durable) = store.durable {
            let entries = durable.scan_prefix(b"sym_meta:")?;
            for (_key, value) in entries {
                let meta: SymbolMeta = bincode::deserialize(&value).map_err(|e| {
                    crate::error::StoreError::Serialization {
                        message: format!("failed to deserialize symbol meta: {e}"),
                    }
                })?;
                let normalized = meta.label.to_lowercase();
                registry.label_to_id.insert(normalized, meta.id);
                registry.id_to_meta.insert(meta.id, meta);
            }
        }

        Ok(registry)
    }
}

impl Default for SymbolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SymbolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymbolRegistry")
            .field("count", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolKind;

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    fn meta(id: u64, label: &str) -> SymbolMeta {
        SymbolMeta::new(sym(id), SymbolKind::Entity, label)
    }

    #[test]
    fn register_and_lookup() {
        let reg = SymbolRegistry::new();
        let m = meta(1, "Sun");
        reg.register(m.clone()).unwrap();

        // Forward lookup by ID.
        let got = reg.get(sym(1)).unwrap();
        assert_eq!(got.label, "Sun");

        // Reverse lookup by label.
        let id = reg.lookup("Sun").unwrap();
        assert_eq!(id, sym(1));

        // Lookup meta by label.
        let got_meta = reg.lookup_meta("Sun").unwrap();
        assert_eq!(got_meta.id, sym(1));
    }

    #[test]
    fn case_insensitive_lookup() {
        let reg = SymbolRegistry::new();
        reg.register(meta(1, "Sun")).unwrap();

        assert_eq!(reg.lookup("sun"), Some(sym(1)));
        assert_eq!(reg.lookup("SUN"), Some(sym(1)));
        assert_eq!(reg.lookup("sUn"), Some(sym(1)));
    }

    #[test]
    fn duplicate_label_error() {
        let reg = SymbolRegistry::new();
        reg.register(meta(1, "Sun")).unwrap();

        let result = reg.register(meta(2, "sun"));
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("duplicate label"));
    }

    #[test]
    fn get_by_id() {
        let reg = SymbolRegistry::new();
        reg.register(meta(42, "Moon")).unwrap();

        let got = reg.get(sym(42)).unwrap();
        assert_eq!(got.label, "Moon");
        assert_eq!(got.kind, SymbolKind::Entity);

        assert!(reg.get(sym(999)).is_none());
    }

    #[test]
    fn all_symbols() {
        let reg = SymbolRegistry::new();
        reg.register(meta(1, "Sun")).unwrap();
        reg.register(meta(2, "Moon")).unwrap();
        reg.register(meta(3, "Star")).unwrap();

        let all = reg.all();
        assert_eq!(all.len(), 3);
        assert_eq!(reg.len(), 3);
        assert!(!reg.is_empty());
    }

    #[test]
    fn persist_and_restore() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = TieredStore::with_persistence(dir.path(), "test").unwrap();

        let reg = SymbolRegistry::new();
        reg.register(meta(1, "Sun")).unwrap();
        reg.register(meta(2, "Moon")).unwrap();
        reg.register(meta(3, "Star")).unwrap();

        reg.persist(&store).unwrap();

        // Restore from the same store.
        let restored = SymbolRegistry::restore(&store).unwrap();
        assert_eq!(restored.len(), 3);
        assert_eq!(restored.lookup("Sun"), Some(sym(1)));
        assert_eq!(restored.lookup("Moon"), Some(sym(2)));
        assert_eq!(restored.lookup("Star"), Some(sym(3)));

        // Verify metadata is preserved.
        let sun = restored.get(sym(1)).unwrap();
        assert_eq!(sun.label, "Sun");
        assert_eq!(sun.kind, SymbolKind::Entity);
    }
}
