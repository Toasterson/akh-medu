//! Provenance ledger: tracks how knowledge was derived.
//!
//! Every inference, reasoning step, or extraction produces a provenance record
//! that links the derived knowledge back to its sources. Records are persisted
//! via redb with multiple indices for efficient lookup.

use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use redb::{Database, MultimapTableDefinition, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::error::{ProvenanceError, ProvenanceResult, StoreError};
use crate::symbol::SymbolId;

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// Primary table: provenance_id → bincode-encoded ProvenanceRecord.
const PROVENANCE_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("provenance");

/// Index: derived_symbol_id → provenance_ids that derive that symbol.
const DERIVED_INDEX: MultimapTableDefinition<u64, u64> =
    MultimapTableDefinition::new("provenance_derived_idx");

/// Reverse index: source_symbol_id → provenance_ids that depend on that source.
const SOURCE_INDEX: MultimapTableDefinition<u64, u64> =
    MultimapTableDefinition::new("provenance_source_idx");

/// Index: derivation_kind_tag → provenance_ids with that kind.
const KIND_INDEX: MultimapTableDefinition<u8, u64> =
    MultimapTableDefinition::new("provenance_kind_idx");

// ---------------------------------------------------------------------------
// DerivationKind
// ---------------------------------------------------------------------------

/// How a piece of knowledge was derived.
///
/// Unified superset covering extraction, inference, reasoning, and aggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DerivationKind {
    /// Directly extracted from source material.
    Extracted,
    /// Seed symbol provided by the user at query time.
    Seed,
    /// Inferred by following a graph edge.
    GraphEdge {
        from: SymbolId,
        predicate: SymbolId,
    },
    /// Inferred via VSA unbind + cleanup recovery.
    VsaRecovery {
        from: SymbolId,
        predicate: SymbolId,
        similarity: f32,
    },
    /// Inferred via analogy (A:B :: C:?).
    Analogy {
        a: SymbolId,
        b: SymbolId,
        c: SymbolId,
    },
    /// Recovered as a role-filler via unbind.
    FillerRecovery {
        subject: SymbolId,
        predicate: SymbolId,
    },
    /// Derived via symbolic reasoning (e-graph rewriting).
    Reasoned,
    /// Aggregated from multiple sources.
    Aggregated,
}

impl DerivationKind {
    /// Numeric tag for indexing by kind.
    pub fn tag(&self) -> u8 {
        match self {
            Self::Extracted => 0,
            Self::Seed => 1,
            Self::GraphEdge { .. } => 2,
            Self::VsaRecovery { .. } => 3,
            Self::Analogy { .. } => 4,
            Self::FillerRecovery { .. } => 5,
            Self::Reasoned => 6,
            Self::Aggregated => 7,
        }
    }
}

// ---------------------------------------------------------------------------
// ProvenanceId
// ---------------------------------------------------------------------------

/// Niche-optimized provenance record identifier.
///
/// `Option<ProvenanceId>` is the same size as `ProvenanceId` thanks to `NonZeroU64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ProvenanceId(NonZeroU64);

impl ProvenanceId {
    /// Create from a raw `u64`. Returns `None` if raw is zero.
    pub fn new(raw: u64) -> Option<Self> {
        NonZeroU64::new(raw).map(ProvenanceId)
    }

    /// Get the underlying value.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl std::fmt::Display for ProvenanceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "prov:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ProvenanceRecord
// ---------------------------------------------------------------------------

/// A single provenance record linking derived knowledge to its sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    /// Unique ID, assigned on persistence. `None` until stored.
    pub id: Option<ProvenanceId>,
    /// The symbol that was derived / activated.
    pub derived_id: SymbolId,
    /// Source symbols this derivation depends on.
    pub sources: Vec<SymbolId>,
    /// How this was derived.
    pub kind: DerivationKind,
    /// Confidence in the derivation.
    pub confidence: f32,
    /// Inference depth at which the symbol was activated.
    pub depth: usize,
    /// Timestamp (seconds since UNIX epoch).
    pub timestamp: u64,
}

impl ProvenanceRecord {
    /// Create a new record with default values.
    pub fn new(derived_id: SymbolId, kind: DerivationKind) -> Self {
        Self {
            id: None,
            derived_id,
            sources: Vec::new(),
            kind,
            confidence: 1.0,
            depth: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Set the source symbols.
    pub fn with_sources(mut self, sources: Vec<SymbolId>) -> Self {
        self.sources = sources;
        self
    }

    /// Set the confidence.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set the depth.
    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }
}

// ---------------------------------------------------------------------------
// ProvenanceLedger
// ---------------------------------------------------------------------------

/// Persistent provenance ledger backed by redb.
///
/// Stores provenance records with multiple indices for efficient querying
/// by derived symbol, source symbol, and derivation kind.
pub struct ProvenanceLedger {
    db: Arc<Database>,
    next_id: AtomicU64,
}

impl std::fmt::Debug for ProvenanceLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProvenanceLedger")
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

fn redb_err(msg: impl std::fmt::Display) -> ProvenanceError {
    ProvenanceError::Store(StoreError::Redb {
        message: msg.to_string(),
    })
}

impl ProvenanceLedger {
    /// Open a provenance ledger on an existing database.
    ///
    /// Recovers the next ID counter by scanning the primary table.
    pub fn open(db: Arc<Database>) -> ProvenanceResult<Self> {
        // Ensure tables exist by opening a write txn.
        {
            let txn = db.begin_write().map_err(|e| redb_err(e))?;
            // Opening the tables creates them if absent.
            txn.open_table(PROVENANCE_TABLE)
                .map_err(|e| redb_err(e))?;
            txn.open_multimap_table(DERIVED_INDEX)
                .map_err(|e| redb_err(e))?;
            txn.open_multimap_table(SOURCE_INDEX)
                .map_err(|e| redb_err(e))?;
            txn.open_multimap_table(KIND_INDEX)
                .map_err(|e| redb_err(e))?;
            txn.commit().map_err(|e| redb_err(e))?;
        }

        // Recover max ID from the primary table.
        let max_id = {
            let txn = db.begin_read().map_err(|e| redb_err(e))?;
            let table = txn
                .open_table(PROVENANCE_TABLE)
                .map_err(|e| redb_err(e))?;
            let mut max = 0u64;
            let iter = table.iter().map_err(|e| redb_err(e))?;
            for entry in iter {
                let (key_guard, _val_guard) = entry.map_err(|e| redb_err(e))?;
                let key: u64 = key_guard.value();
                if key > max {
                    max = key;
                }
            }
            max
        };

        Ok(Self {
            db,
            next_id: AtomicU64::new(max_id + 1),
        })
    }

    /// Store a single record, assigning it a new ID.
    pub fn store(&self, record: &mut ProvenanceRecord) -> ProvenanceResult<ProvenanceId> {
        let raw_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let prov_id =
            ProvenanceId::new(raw_id).expect("provenance ID counter should never reach zero");
        record.id = Some(prov_id);

        let encoded = bincode::serialize(&record).map_err(|e| {
            ProvenanceError::Store(StoreError::Serialization {
                message: format!("provenance record serialize: {e}"),
            })
        })?;

        let txn = self
            .db
            .begin_write()
            .map_err(|e| redb_err(e))?;
        {
            let mut table = txn
                .open_table(PROVENANCE_TABLE)
                .map_err(|e| redb_err(e))?;
            table
                .insert(raw_id, encoded.as_slice())
                .map_err(|e| redb_err(e))?;
        }
        {
            let mut idx = txn
                .open_multimap_table(DERIVED_INDEX)
                .map_err(|e| redb_err(e))?;
            idx.insert(record.derived_id.get(), raw_id)
                .map_err(|e| redb_err(e))?;
        }
        {
            let mut idx = txn
                .open_multimap_table(SOURCE_INDEX)
                .map_err(|e| redb_err(e))?;
            for src in &record.sources {
                idx.insert(src.get(), raw_id)
                    .map_err(|e| redb_err(e))?;
            }
        }
        {
            let mut idx = txn
                .open_multimap_table(KIND_INDEX)
                .map_err(|e| redb_err(e))?;
            idx.insert(record.kind.tag(), raw_id)
                .map_err(|e| redb_err(e))?;
        }
        txn.commit().map_err(|e| redb_err(e))?;

        Ok(prov_id)
    }

    /// Store a batch of records in a single transaction.
    pub fn store_batch(
        &self,
        records: &mut [ProvenanceRecord],
    ) -> ProvenanceResult<Vec<ProvenanceId>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let mut ids = Vec::with_capacity(records.len());
        let mut encoded_batch = Vec::with_capacity(records.len());

        // Assign IDs and serialize before opening the transaction.
        for record in records.iter_mut() {
            let raw_id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let prov_id =
                ProvenanceId::new(raw_id).expect("provenance ID counter should never reach zero");
            record.id = Some(prov_id);
            ids.push(prov_id);

            let encoded = bincode::serialize(&record).map_err(|e| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: format!("provenance record serialize: {e}"),
                })
            })?;
            encoded_batch.push((raw_id, encoded));
        }

        let txn = self
            .db
            .begin_write()
            .map_err(|e| redb_err(e))?;
        {
            let mut table = txn
                .open_table(PROVENANCE_TABLE)
                .map_err(|e| redb_err(e))?;
            for (raw_id, encoded) in &encoded_batch {
                table
                    .insert(*raw_id, encoded.as_slice())
                    .map_err(|e| redb_err(e))?;
            }
        }
        {
            let mut derived_idx = txn
                .open_multimap_table(DERIVED_INDEX)
                .map_err(|e| redb_err(e))?;
            let mut source_idx = txn
                .open_multimap_table(SOURCE_INDEX)
                .map_err(|e| redb_err(e))?;
            let mut kind_idx = txn
                .open_multimap_table(KIND_INDEX)
                .map_err(|e| redb_err(e))?;

            for (i, (raw_id, _)) in encoded_batch.iter().enumerate() {
                let record = &records[i];
                derived_idx
                    .insert(record.derived_id.get(), *raw_id)
                    .map_err(|e| redb_err(e))?;
                for src in &record.sources {
                    source_idx
                        .insert(src.get(), *raw_id)
                        .map_err(|e| redb_err(e))?;
                }
                kind_idx
                    .insert(record.kind.tag(), *raw_id)
                    .map_err(|e| redb_err(e))?;
            }
        }
        txn.commit().map_err(|e| redb_err(e))?;

        Ok(ids)
    }

    /// Get a provenance record by its ID.
    pub fn get(&self, id: ProvenanceId) -> ProvenanceResult<ProvenanceRecord> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| redb_err(e))?;
        let table = txn
            .open_table(PROVENANCE_TABLE)
            .map_err(|e| redb_err(e))?;
        let guard = table
            .get(id.get())
            .map_err(|e| redb_err(e))?
            .ok_or(ProvenanceError::NotFound { id: id.get() })?;
        let record: ProvenanceRecord =
            bincode::deserialize(guard.value()).map_err(|e| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: format!("provenance record deserialize: {e}"),
                })
            })?;
        Ok(record)
    }

    /// Find all provenance records for a given derived symbol.
    pub fn by_derived(&self, symbol: SymbolId) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        self.records_from_multimap_index(&DERIVED_INDEX, symbol.get())
    }

    /// Find all provenance records that depend on a given source symbol.
    ///
    /// This answers "what was derived from X?" — the reverse dependency lookup.
    pub fn by_source(&self, symbol: SymbolId) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        self.records_from_multimap_index(&SOURCE_INDEX, symbol.get())
    }

    /// Find all provenance records of a given derivation kind.
    pub fn by_kind(&self, kind: &DerivationKind) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        self.records_from_kind_index(kind.tag())
    }

    /// Total number of provenance records.
    pub fn len(&self) -> ProvenanceResult<usize> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| redb_err(e))?;
        let table = txn
            .open_table(PROVENANCE_TABLE)
            .map_err(|e| redb_err(e))?;
        Ok(table.len().map_err(|e| redb_err(e))? as usize)
    }

    /// Whether the ledger has no records.
    pub fn is_empty(&self) -> ProvenanceResult<bool> {
        self.len().map(|n| n == 0)
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Retrieve records given a u64-keyed multimap index.
    fn records_from_multimap_index(
        &self,
        table_def: &MultimapTableDefinition<u64, u64>,
        key: u64,
    ) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| redb_err(e))?;
        let idx = txn
            .open_multimap_table(*table_def)
            .map_err(|e| redb_err(e))?;
        let primary = txn
            .open_table(PROVENANCE_TABLE)
            .map_err(|e| redb_err(e))?;

        let values = idx.get(key).map_err(|e| redb_err(e))?;

        let mut records = Vec::new();
        for entry in values {
            let raw_id = entry.map_err(|e| redb_err(e))?.value();
            let guard = primary
                .get(raw_id)
                .map_err(|e| redb_err(e))?
                .ok_or(ProvenanceError::NotFound { id: raw_id })?;
            let record: ProvenanceRecord = bincode::deserialize(guard.value()).map_err(|e| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: format!("provenance record deserialize: {e}"),
                })
            })?;
            records.push(record);
        }
        Ok(records)
    }

    /// Retrieve records given a u8-keyed multimap index (kind).
    fn records_from_kind_index(&self, tag: u8) -> ProvenanceResult<Vec<ProvenanceRecord>> {
        let txn = self
            .db
            .begin_read()
            .map_err(|e| redb_err(e))?;
        let idx = txn
            .open_multimap_table(KIND_INDEX)
            .map_err(|e| redb_err(e))?;
        let primary = txn
            .open_table(PROVENANCE_TABLE)
            .map_err(|e| redb_err(e))?;

        let values = idx.get(tag).map_err(|e| redb_err(e))?;

        let mut records = Vec::new();
        for entry in values {
            let raw_id = entry.map_err(|e| redb_err(e))?.value();
            let guard = primary
                .get(raw_id)
                .map_err(|e| redb_err(e))?
                .ok_or(ProvenanceError::NotFound { id: raw_id })?;
            let record: ProvenanceRecord = bincode::deserialize(guard.value()).map_err(|e| {
                ProvenanceError::Store(StoreError::Serialization {
                    message: format!("provenance record deserialize: {e}"),
                })
            })?;
            records.push(record);
        }
        Ok(records)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Arc<Database> {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.redb");
        // Leak the TempDir to keep the directory alive for the test.
        std::mem::forget(dir);
        Arc::new(Database::create(path).unwrap())
    }

    fn sym(id: u64) -> SymbolId {
        SymbolId::new(id).unwrap()
    }

    #[test]
    fn store_and_retrieve() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let mut record = ProvenanceRecord::new(sym(1), DerivationKind::Seed)
            .with_confidence(1.0)
            .with_depth(0);

        let id = ledger.store(&mut record).unwrap();
        assert!(record.id.is_some());

        let retrieved = ledger.get(id).unwrap();
        assert_eq!(retrieved.derived_id, sym(1));
        assert_eq!(retrieved.kind, DerivationKind::Seed);
        assert_eq!(retrieved.confidence, 1.0);
        assert_eq!(retrieved.depth, 0);
    }

    #[test]
    fn auto_incrementing_ids() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let mut r1 = ProvenanceRecord::new(sym(1), DerivationKind::Seed);
        let mut r2 = ProvenanceRecord::new(sym(2), DerivationKind::Extracted);
        let mut r3 = ProvenanceRecord::new(sym(3), DerivationKind::Reasoned);

        let id1 = ledger.store(&mut r1).unwrap();
        let id2 = ledger.store(&mut r2).unwrap();
        let id3 = ledger.store(&mut r3).unwrap();

        assert_eq!(id1.get(), 1);
        assert_eq!(id2.get(), 2);
        assert_eq!(id3.get(), 3);
    }

    #[test]
    fn by_derived_index() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let target = sym(10);
        let mut r1 = ProvenanceRecord::new(target, DerivationKind::Seed);
        let mut r2 = ProvenanceRecord::new(target, DerivationKind::Reasoned);
        let mut r3 = ProvenanceRecord::new(sym(20), DerivationKind::Extracted);

        ledger.store(&mut r1).unwrap();
        ledger.store(&mut r2).unwrap();
        ledger.store(&mut r3).unwrap();

        let results = ledger.by_derived(target).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.derived_id == target));
    }

    #[test]
    fn by_source_reverse_lookup() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let source = sym(5);
        let mut r1 = ProvenanceRecord::new(sym(10), DerivationKind::GraphEdge {
            from: source,
            predicate: sym(100),
        })
        .with_sources(vec![source]);
        let mut r2 = ProvenanceRecord::new(sym(20), DerivationKind::Reasoned)
            .with_sources(vec![source, sym(6)]);

        ledger.store(&mut r1).unwrap();
        ledger.store(&mut r2).unwrap();

        let dependents = ledger.by_source(source).unwrap();
        assert_eq!(dependents.len(), 2);
    }

    #[test]
    fn by_kind_index() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let mut r1 = ProvenanceRecord::new(sym(1), DerivationKind::Seed);
        let mut r2 = ProvenanceRecord::new(sym(2), DerivationKind::Seed);
        let mut r3 = ProvenanceRecord::new(sym(3), DerivationKind::Extracted);
        let mut r4 = ProvenanceRecord::new(sym(4), DerivationKind::Reasoned);

        ledger.store(&mut r1).unwrap();
        ledger.store(&mut r2).unwrap();
        ledger.store(&mut r3).unwrap();
        ledger.store(&mut r4).unwrap();

        let seeds = ledger.by_kind(&DerivationKind::Seed).unwrap();
        assert_eq!(seeds.len(), 2);
        let extracted = ledger.by_kind(&DerivationKind::Extracted).unwrap();
        assert_eq!(extracted.len(), 1);
        let reasoned = ledger.by_kind(&DerivationKind::Reasoned).unwrap();
        assert_eq!(reasoned.len(), 1);
    }

    #[test]
    fn store_batch_single_transaction() {
        let db = test_db();
        let ledger = ProvenanceLedger::open(db).unwrap();

        let mut records: Vec<ProvenanceRecord> = (1..=5)
            .map(|i| ProvenanceRecord::new(sym(i), DerivationKind::Seed))
            .collect();

        let ids = ledger.store_batch(&mut records).unwrap();
        assert_eq!(ids.len(), 5);
        assert_eq!(ledger.len().unwrap(), 5);

        // All records should have IDs assigned.
        for record in &records {
            assert!(record.id.is_some());
        }
    }

    #[test]
    fn persistence_across_reopens() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("persist.redb");
        let db = Arc::new(Database::create(&path).unwrap());

        let stored_id = {
            let ledger = ProvenanceLedger::open(Arc::clone(&db)).unwrap();
            let mut record = ProvenanceRecord::new(sym(42), DerivationKind::Extracted)
                .with_confidence(0.95);
            ledger.store(&mut record).unwrap()
        };

        // Drop and reopen.
        drop(db);
        let db2 = Arc::new(Database::create(&path).unwrap());
        let ledger2 = ProvenanceLedger::open(db2).unwrap();

        let retrieved = ledger2.get(stored_id).unwrap();
        assert_eq!(retrieved.derived_id, sym(42));
        assert_eq!(retrieved.confidence, 0.95);

        // Next ID should continue past the stored one.
        let mut r2 = ProvenanceRecord::new(sym(43), DerivationKind::Seed);
        let id2 = ledger2.store(&mut r2).unwrap();
        assert!(id2.get() > stored_id.get());
    }

    #[test]
    fn provenance_id_niche_optimization() {
        assert_eq!(
            std::mem::size_of::<Option<ProvenanceId>>(),
            std::mem::size_of::<ProvenanceId>()
        );
    }
}
