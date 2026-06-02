//! Heka drawer storage.
//!
//! A "drawer" is a single memory entry: a timestamped claim with optional
//! source, confidence, and topic tags. Drawers are persisted in a redb
//! database keyed by a monotonically increasing u64 ID.

use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::error::{HekaError, HekaResult};

// ── Storage schema ────────────────────────────────────────────────────────

const DRAWER_TABLE: TableDefinition<u64, &[u8]> =
    TableDefinition::new("heka_drawers");

const NEXT_ID_TABLE: TableDefinition<&str, u64> =
    TableDefinition::new("heka_meta");

const NEXT_ID_KEY: &str = "next_drawer_id";

// ═══════════════════════════════════════════════════════════════════════
// Drawer record
// ═══════════════════════════════════════════════════════════════════════

/// A single memory entry in Heka.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawerRecord {
    /// Numeric ID assigned at insertion.
    pub id: u64,
    /// Short label for the subject entity (e.g. "temperature").
    pub subject: String,
    /// Full claim text (e.g. "temperature is 37 degrees").
    pub claim: String,
    /// Source label, if known (e.g. "thermometer-A", "alice").
    pub source: Option<String>,
    /// Tool that created this drawer (e.g. "browser", "manual", "mcp").
    pub tool: Option<String>,
    /// Stated confidence [0.0, 1.0].
    pub confidence: f32,
    /// Unix timestamp (seconds) when the drawer was recorded.
    pub timestamp: u64,
    /// Optional freeform topic tags.
    pub topic_ids: Vec<String>,
}

impl DrawerRecord {
    pub fn new(
        id: u64,
        subject: impl Into<String>,
        claim: impl Into<String>,
    ) -> Self {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            id,
            subject: subject.into(),
            claim: claim.into(),
            source: None,
            tool: None,
            confidence: 0.8,
            timestamp: ts,
            topic_ids: Vec::new(),
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tool = Some(tool.into());
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DrawerStore
// ═══════════════════════════════════════════════════════════════════════

/// Redb-backed persistent store for drawer records.
pub struct DrawerStore {
    db: Database,
    next_id: AtomicU64,
}

impl DrawerStore {
    /// Open (or create) a drawer store at `path`.
    pub fn open(path: &Path) -> HekaResult<Self> {
        let db = Database::create(path)?;

        // Ensure tables exist.
        {
            let txn = db.begin_write()?;
            txn.open_table(DRAWER_TABLE)?;
            txn.open_table(NEXT_ID_TABLE)?;
            txn.commit()?;
        }

        // Restore next_id from persistent storage.
        let next_id = {
            let txn = db.begin_read()?;
            let table = txn.open_table(NEXT_ID_TABLE)?;
            table
                .get(NEXT_ID_KEY)?
                .map(|v| v.value())
                .unwrap_or(1)
        };

        Ok(Self {
            db,
            next_id: AtomicU64::new(next_id),
        })
    }

    /// Open an in-memory (ephemeral) store for testing.
    pub fn in_memory() -> HekaResult<Self> {
        let db = Database::builder()
            .create_with_backend(redb::backends::InMemoryBackend::new())?;
        {
            let txn = db.begin_write()?;
            txn.open_table(DRAWER_TABLE)?;
            txn.open_table(NEXT_ID_TABLE)?;
            txn.commit()?;
        }
        Ok(Self {
            db,
            next_id: AtomicU64::new(1),
        })
    }

    /// Insert a new drawer. Returns the assigned ID.
    pub fn insert(
        &self,
        subject: impl Into<String>,
        claim: impl Into<String>,
        source: Option<String>,
        confidence: f32,
        timestamp: Option<u64>,
        tool: Option<String>,
        topic_ids: Vec<String>,
    ) -> HekaResult<u64> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let ts = timestamp.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        });

        let mut record = DrawerRecord::new(id, subject, claim)
            .with_confidence(confidence)
            .with_timestamp(ts);
        if let Some(src) = source {
            record = record.with_source(src);
        }
        if let Some(t) = tool {
            record = record.with_tool(t);
        }
        record.topic_ids = topic_ids;

        let bytes = bincode::serialize(&record)?;
        let txn = self.db.begin_write()?;
        {
            let mut table = txn.open_table(DRAWER_TABLE)?;
            table.insert(id, bytes.as_slice())?;
            let mut meta = txn.open_table(NEXT_ID_TABLE)?;
            meta.insert(NEXT_ID_KEY, id + 1)?;
        }
        txn.commit()?;

        Ok(id)
    }

    /// Retrieve a single drawer by ID.
    pub fn get(&self, id: u64) -> HekaResult<DrawerRecord> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DRAWER_TABLE)?;
        let bytes = table
            .get(id)?
            .ok_or_else(|| HekaError::DrawerNotFound { id: id.to_string() })?;
        let record: DrawerRecord = bincode::deserialize(bytes.value())?;
        Ok(record)
    }

    /// List all drawers ordered by ID ascending.
    pub fn list(&self) -> HekaResult<Vec<DrawerRecord>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DRAWER_TABLE)?;
        let mut records = Vec::new();
        for entry in table.iter()? {
            let (_, v) = entry?;
            let record: DrawerRecord = bincode::deserialize(v.value())?;
            records.push(record);
        }
        Ok(records)
    }

    /// Return drawers whose subject contains `query` (case-insensitive).
    pub fn search_by_subject(&self, query: &str) -> HekaResult<Vec<DrawerRecord>> {
        let q = query.to_lowercase();
        let all = self.list()?;
        Ok(all
            .into_iter()
            .filter(|r| r.subject.to_lowercase().contains(&q))
            .collect())
    }

    /// Return the total number of drawers.
    pub fn count(&self) -> HekaResult<u64> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DRAWER_TABLE)?;
        Ok(table.len()?)
    }
}

/// Wrap the store in an `Arc` for shared state across async tasks.
pub type SharedDrawerStore = Arc<DrawerStore>;

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_store() -> DrawerStore {
        DrawerStore::in_memory().expect("in-memory store")
    }

    #[test]
    fn insert_and_get() {
        let store = test_store();
        let id = store
            .insert("sky", "sky is blue", None, 0.9, None, None, vec![])
            .unwrap();
        let record = store.get(id).unwrap();
        assert_eq!(record.claim, "sky is blue");
        assert_eq!(record.confidence, 0.9);
    }

    #[test]
    fn ids_are_sequential() {
        let store = test_store();
        let id1 = store
            .insert("x", "x is a", None, 0.8, None, None, vec![])
            .unwrap();
        let id2 = store
            .insert("x", "x is b", None, 0.8, None, None, vec![])
            .unwrap();
        assert_eq!(id2, id1 + 1);
    }

    #[test]
    fn get_nonexistent_returns_error() {
        let store = test_store();
        let result = store.get(99_999);
        assert!(matches!(result, Err(HekaError::DrawerNotFound { .. })));
    }

    #[test]
    fn list_returns_all_records() {
        let store = test_store();
        store
            .insert("a", "a is x", None, 0.5, None, None, vec![])
            .unwrap();
        store
            .insert("b", "b is y", None, 0.7, None, None, vec![])
            .unwrap();
        let all = store.list().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn search_by_subject_case_insensitive() {
        let store = test_store();
        store
            .insert("Temperature", "Temperature is 37", None, 0.9, None, None, vec![])
            .unwrap();
        store
            .insert("sky", "sky is blue", None, 0.9, None, None, vec![])
            .unwrap();
        let results = store.search_by_subject("temperature").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].subject, "Temperature");
    }
}
