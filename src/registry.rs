//! Symbol registry: bidirectional label ↔ ID mapping.
//!
//! The [`SymbolRegistry`] provides O(1) lookups in both directions
//! using two `DashMap`s. Labels are normalized to lowercase with
//! apostrophes stripped for deduplication-safe, case-insensitive matching.

use dashmap::DashMap;

use crate::error::{AkhResult, SymbolError};
use crate::store::TieredStore;
use crate::symbol::{SymbolId, SymbolMeta};

/// Normalize a label into a deduplication key: lowercase and strip apostrophe
/// variants so that "Maat", "Ma'at" (ASCII U+0027), and "Ma\u{2019}at"
/// (Unicode RIGHT SINGLE QUOTATION MARK) all produce the same key "maat".
///
/// Stripped characters: U+0027 APOSTROPHE, U+2018 LEFT SINGLE QUOTATION MARK,
/// U+2019 RIGHT SINGLE QUOTATION MARK, U+02BC MODIFIER LETTER APOSTROPHE.
pub(crate) fn normalize_label_key(label: &str) -> String {
    label
        .to_lowercase()
        .replace(['\u{0027}', '\u{2018}', '\u{2019}', '\u{02BC}'], "")
}

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
        let normalized = normalize_label_key(&meta.label);

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

    /// Look up a symbol ID by label (case-insensitive, apostrophe-normalised).
    pub fn lookup(&self, label: &str) -> Option<SymbolId> {
        let normalized = normalize_label_key(label);
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

    /// Authoritatively replace the durable symbol namespace with this
    /// registry in one transaction.
    ///
    /// Normal persistence is append/update oriented because a live registry
    /// only grows.  Cold restore is different: the relational snapshot is the
    /// source of truth, so stale `sym_meta:*` keys from a reused host must be
    /// removed atomically with publishing the restored entries.
    pub fn replace_persisted(&self, store: &TieredStore) -> AkhResult<()> {
        let mut keys = Vec::with_capacity(self.id_to_meta.len());
        let mut values = Vec::with_capacity(self.id_to_meta.len());
        for entry in self.id_to_meta.iter() {
            let meta = entry.value();
            keys.push(format!("sym_meta:{}", meta.id.get()).into_bytes());
            values.push(bincode::serialize(meta).map_err(|e| {
                crate::error::StoreError::Serialization {
                    message: format!("failed to serialize symbol meta: {e}"),
                }
            })?);
        }
        let entries: Vec<(&[u8], &[u8])> = keys
            .iter()
            .zip(values.iter())
            .map(|(key, value)| (key.as_slice(), value.as_slice()))
            .collect();
        store.replace_meta_prefix(b"sym_meta:", &entries)?;
        Ok(())
    }

    /// Restore the registry from the durable store by scanning `sym_meta:` prefix.
    pub fn restore(store: &TieredStore) -> AkhResult<Self> {
        let registry = Self::new();

        if let Some(ref durable) = store.durable {
            let entries = durable.scan_prefix(b"sym_meta:")?;
            let mut seen_ids = std::collections::HashSet::with_capacity(entries.len());
            let mut seen_labels = std::collections::HashMap::with_capacity(entries.len());
            for (key, value) in entries {
                let key_text = std::str::from_utf8(&key).map_err(|error| {
                    crate::error::StoreError::Serialization {
                        message: format!("invalid symbol metadata key encoding: {error}"),
                    }
                })?;
                let raw_key_id = key_text
                    .strip_prefix("sym_meta:")
                    .and_then(|suffix| suffix.parse::<u64>().ok())
                    .ok_or_else(|| crate::error::StoreError::Serialization {
                        message: format!("invalid symbol metadata key {key_text:?}"),
                    })?;
                let key_id = SymbolId::new(raw_key_id).ok_or_else(|| {
                    crate::error::StoreError::Serialization {
                        message: "symbol metadata key contains reserved zero ID".into(),
                    }
                })?;
                if key != format!("sym_meta:{}", key_id.get()).as_bytes() {
                    return Err(crate::error::StoreError::Serialization {
                        message: format!("non-canonical symbol metadata key {key_text:?}"),
                    }
                    .into());
                }
                let meta: SymbolMeta = bincode::deserialize(&value).map_err(|e| {
                    crate::error::StoreError::Serialization {
                        message: format!("failed to deserialize symbol meta: {e}"),
                    }
                })?;
                if meta.id != key_id {
                    return Err(crate::error::StoreError::Serialization {
                        message: format!(
                            "symbol metadata key ID {} does not match embedded ID {}",
                            key_id.get(),
                            meta.id.get()
                        ),
                    }
                    .into());
                }
                if matches!(
                    meta.id.get(),
                    crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL
                        | crate::provenance::RULE_APPLICATION_V2_SENTINEL
                ) {
                    return Err(crate::error::StoreError::Serialization {
                        message: format!(
                            "symbol metadata uses reserved provenance marker ID {}",
                            meta.id.get()
                        ),
                    }
                    .into());
                }
                if !seen_ids.insert(meta.id) {
                    return Err(crate::error::StoreError::Serialization {
                        message: format!("duplicate durable symbol ID {}", meta.id.get()),
                    }
                    .into());
                }
                let normalized = normalize_label_key(&meta.label);
                if let Some(existing_id) = seen_labels.insert(normalized, meta.id) {
                    return Err(SymbolError::DuplicateLabel {
                        label: meta.label,
                        existing_id: existing_id.get(),
                    }
                    .into());
                }
                registry.register(meta)?;
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

    #[test]
    fn authoritative_restore_removes_reused_host_symbols() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = TieredStore::with_persistence(dir.path(), "test").unwrap();

        let stale = SymbolRegistry::new();
        stale.register(meta(1, "Sun")).unwrap();
        stale.register(meta(99, "StaleHostOnly")).unwrap();
        stale.persist(&store).unwrap();

        let restored = SymbolRegistry::new();
        restored.register(meta(1, "Sun")).unwrap();
        restored.register(meta(2, "Moon")).unwrap();
        restored.replace_persisted(&store).unwrap();

        let reopened = SymbolRegistry::restore(&store).unwrap();
        assert_eq!(reopened.len(), 2);
        assert_eq!(reopened.lookup("Sun"), Some(sym(1)));
        assert_eq!(reopened.lookup("Moon"), Some(sym(2)));
        assert!(reopened.lookup("StaleHostOnly").is_none());
        assert!(reopened.get(sym(99)).is_none());
    }

    #[test]
    fn restore_rejects_key_and_embedded_id_mismatch() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = TieredStore::with_persistence(dir.path(), "test").unwrap();
        let encoded = bincode::serialize(&meta(2, "Moon")).unwrap();
        store.put_meta(b"sym_meta:1", &encoded).unwrap();

        let error = SymbolRegistry::restore(&store).unwrap_err();
        assert!(format!("{error}").contains("does not match embedded ID"));
    }

    #[test]
    fn restore_rejects_duplicate_normalized_labels() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = TieredStore::with_persistence(dir.path(), "test").unwrap();
        let first = bincode::serialize(&meta(1, "Ma'at")).unwrap();
        let second = bincode::serialize(&meta(2, "Maat")).unwrap();
        store.put_meta(b"sym_meta:1", &first).unwrap();
        store.put_meta(b"sym_meta:2", &second).unwrap();

        let error = SymbolRegistry::restore(&store).unwrap_err();
        assert!(format!("{error}").contains("duplicate label"));
    }

    #[test]
    fn restore_rejects_noncanonical_key() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = TieredStore::with_persistence(dir.path(), "test").unwrap();
        let encoded = bincode::serialize(&meta(1, "Sun")).unwrap();
        store.put_meta(b"sym_meta:01", &encoded).unwrap();

        let error = SymbolRegistry::restore(&store).unwrap_err();
        assert!(format!("{error}").contains("non-canonical"));
    }
}
