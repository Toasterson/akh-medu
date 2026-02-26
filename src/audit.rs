//! Append-only audit ledger: tracks tool calls, content ingestion, and agent log messages.
//!
//! The audit ledger records every significant action in the akh-medu system,
//! providing a persistent, tamper-evident activity trace. Records are persisted
//! via redb with a kind index for efficient filtered queries.
//!
//! **Critical invariant: this ledger is append-only. No delete, update, or
//! truncate operations exist.**

use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use miette::Diagnostic;
use redb::{Database, MultimapTableDefinition, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::error::StoreError;
use crate::message::{AkhMessage, MessageSink};

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

/// Primary table: audit_id → bincode-encoded AuditEntry.
const AUDIT_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("audit");

/// Index: kind_tag → audit_ids with that kind.
const AUDIT_KIND_INDEX: MultimapTableDefinition<u8, u64> =
    MultimapTableDefinition::new("audit_kind_idx");

// ---------------------------------------------------------------------------
// AuditId
// ---------------------------------------------------------------------------

/// Niche-optimized audit record identifier.
///
/// `Option<AuditId>` is the same size as `AuditId` thanks to `NonZeroU64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct AuditId(NonZeroU64);

impl AuditId {
    /// Create from a raw `u64`. Returns `None` if raw is zero.
    pub fn new(raw: u64) -> Option<Self> {
        NonZeroU64::new(raw).map(AuditId)
    }

    /// Get the underlying value.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl std::fmt::Display for AuditId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "audit:{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// AuditKind
// ---------------------------------------------------------------------------

/// What kind of activity this audit entry records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AuditKind {
    /// A tool was invoked by the agent.
    ToolInvocation {
        tool_name: String,
        params_summary: String,
        success: bool,
        output_summary: String,
        duration_ms: u64,
    },
    /// Content was ingested into the library.
    ContentIngestion {
        document_title: String,
        source: String,
        format: String,
        chunk_count: usize,
        triple_count: usize,
    },
    /// The agent voluntarily logged a message.
    AgentLog {
        level: LogLevel,
        message: String,
    },
}

impl AuditKind {
    /// Numeric tag for kind-indexed queries.
    pub fn tag(&self) -> u8 {
        match self {
            Self::ToolInvocation { .. } => 0,
            Self::ContentIngestion { .. } => 1,
            Self::AgentLog { .. } => 2,
        }
    }

    /// Human-readable kind label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::ToolInvocation { .. } => "tool",
            Self::ContentIngestion { .. } => "ingest",
            Self::AgentLog { .. } => "log",
        }
    }

    /// One-line summary of the entry.
    pub fn summary(&self) -> String {
        match self {
            Self::ToolInvocation {
                tool_name,
                success,
                duration_ms,
                ..
            } => {
                let status = if *success { "ok" } else { "FAIL" };
                format!("{tool_name} [{status}] ({duration_ms}ms)")
            }
            Self::ContentIngestion {
                document_title,
                chunk_count,
                triple_count,
                ..
            } => {
                format!("\"{document_title}\" ({chunk_count} chunks, {triple_count} triples)")
            }
            Self::AgentLog { level, message } => {
                format!("[{level:?}] {message}")
            }
        }
    }
}

/// Log level for agent-initiated audit messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Info,
    Warn,
    Debug,
}

// ---------------------------------------------------------------------------
// AuditEntry
// ---------------------------------------------------------------------------

/// A single audit record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique ID, assigned on persistence. `None` until stored.
    pub id: Option<AuditId>,
    /// What happened.
    pub kind: AuditKind,
    /// Which workspace this occurred in.
    pub workspace: String,
    /// Timestamp (seconds since UNIX epoch).
    pub timestamp: u64,
}

impl AuditEntry {
    /// Create a new entry with the current timestamp.
    pub fn new(kind: AuditKind, workspace: impl Into<String>) -> Self {
        Self {
            id: None,
            kind,
            workspace: workspace.into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Convert to an AkhMessage for TUI/WS rendering.
    pub fn to_message(&self) -> AkhMessage {
        AkhMessage::AuditLog {
            id: self.id.map(|id| id.get()).unwrap_or(0),
            kind: self.kind.label().to_string(),
            summary: self.kind.summary(),
            timestamp: self.timestamp,
        }
    }
}

// ---------------------------------------------------------------------------
// AuditPage
// ---------------------------------------------------------------------------

/// Paginated query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditPage {
    pub entries: Vec<AuditEntry>,
    pub total: usize,
    pub has_more: bool,
}

// ---------------------------------------------------------------------------
// AuditError
// ---------------------------------------------------------------------------

/// Errors from audit ledger operations.
#[derive(Debug, Error, Diagnostic)]
pub enum AuditError {
    #[error("audit entry not found: id {id}")]
    #[diagnostic(
        code(akh::audit::not_found),
        help("No audit entry exists with this ID. Verify the audit ID is correct.")
    )]
    NotFound { id: u64 },

    #[error("audit ledger requires persistence — no durable store configured")]
    #[diagnostic(
        code(akh::audit::no_persistence),
        help(
            "The audit ledger needs a data directory for storage. \
             Pass --data-dir to enable persistence."
        )
    )]
    NoPersistence,

    #[error(transparent)]
    #[diagnostic(transparent)]
    Store(#[from] StoreError),
}

/// Result type for audit operations.
pub type AuditResult<T> = std::result::Result<T, AuditError>;

// ---------------------------------------------------------------------------
// AuditLedger
// ---------------------------------------------------------------------------

/// Persistent, append-only audit ledger backed by redb.
///
/// Stores audit entries with a kind index for filtered queries.
/// Supports optional broadcast (for WS streaming) and sink (for TUI live push).
///
/// **No delete, update, or truncate operations exist.**
pub struct AuditLedger {
    db: Arc<Database>,
    next_id: AtomicU64,
    #[cfg(feature = "daemon")]
    broadcast: std::sync::Mutex<Option<tokio::sync::broadcast::Sender<AuditEntry>>>,
    sink: std::sync::Mutex<Option<Arc<dyn MessageSink>>>,
}

impl std::fmt::Debug for AuditLedger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuditLedger")
            .field("next_id", &self.next_id.load(Ordering::Relaxed))
            .finish()
    }
}

fn redb_err(msg: impl std::fmt::Display) -> AuditError {
    AuditError::Store(StoreError::Redb {
        message: msg.to_string(),
    })
}

impl AuditLedger {
    /// Open an audit ledger on an existing database.
    ///
    /// Recovers the next ID counter by scanning the primary table.
    pub fn open(db: Arc<Database>) -> AuditResult<Self> {
        // Ensure tables exist by opening a write txn.
        {
            let txn = db.begin_write().map_err(redb_err)?;
            txn.open_table(AUDIT_TABLE).map_err(redb_err)?;
            txn.open_multimap_table(AUDIT_KIND_INDEX)
                .map_err(redb_err)?;
            txn.commit().map_err(redb_err)?;
        }

        // Recover max ID from the primary table.
        let max_id = {
            let txn = db.begin_read().map_err(redb_err)?;
            let table = txn.open_table(AUDIT_TABLE).map_err(redb_err)?;
            let mut max = 0u64;
            let iter = table.iter().map_err(redb_err)?;
            for entry in iter {
                let (key_guard, _val_guard) = entry.map_err(redb_err)?;
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
            #[cfg(feature = "daemon")]
            broadcast: std::sync::Mutex::new(None),
            sink: std::sync::Mutex::new(None),
        })
    }

    /// Append an entry, assigning it a new ID.
    ///
    /// After storage, broadcasts to WS subscribers and emits to the TUI sink
    /// if either is attached.
    pub fn append(&self, entry: &mut AuditEntry) -> AuditResult<AuditId> {
        let raw_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let audit_id = AuditId::new(raw_id).expect("audit ID counter should never reach zero");
        entry.id = Some(audit_id);

        let encoded = bincode::serialize(&entry).map_err(|e| {
            AuditError::Store(StoreError::Serialization {
                message: format!("audit entry serialize: {e}"),
            })
        })?;

        let txn = self.db.begin_write().map_err(redb_err)?;
        {
            let mut table = txn.open_table(AUDIT_TABLE).map_err(redb_err)?;
            table
                .insert(raw_id, encoded.as_slice())
                .map_err(redb_err)?;
        }
        {
            let mut idx = txn
                .open_multimap_table(AUDIT_KIND_INDEX)
                .map_err(redb_err)?;
            idx.insert(entry.kind.tag(), raw_id)
                .map_err(redb_err)?;
        }
        txn.commit().map_err(redb_err)?;

        // Broadcast to WS subscribers (best-effort).
        #[cfg(feature = "daemon")]
        {
            if let Ok(guard) = self.broadcast.lock()
                && let Some(ref tx) = *guard
            {
                let _ = tx.send(entry.clone());
            }
        }

        // Emit to TUI sink (best-effort).
        if let Ok(guard) = self.sink.lock()
            && let Some(ref sink) = *guard
        {
            sink.emit(&entry.to_message());
        }

        Ok(audit_id)
    }

    /// Get a single entry by ID.
    pub fn get(&self, id: AuditId) -> AuditResult<AuditEntry> {
        let txn = self.db.begin_read().map_err(redb_err)?;
        let table = txn.open_table(AUDIT_TABLE).map_err(redb_err)?;
        let guard = table
            .get(id.get())
            .map_err(redb_err)?
            .ok_or(AuditError::NotFound { id: id.get() })?;
        deserialize_entry(guard.value())
    }

    /// Paginated range scan over all entries.
    pub fn list_page(&self, offset: u64, limit: usize) -> AuditResult<AuditPage> {
        let txn = self.db.begin_read().map_err(redb_err)?;
        let table = txn.open_table(AUDIT_TABLE).map_err(redb_err)?;
        let total = table.len().map_err(redb_err)? as usize;

        let mut entries = Vec::with_capacity(limit);
        let iter = table.iter().map_err(redb_err)?;
        let mut skipped = 0u64;

        for result in iter {
            let (_, val_guard) = result.map_err(redb_err)?;
            if skipped < offset {
                skipped += 1;
                continue;
            }
            if entries.len() >= limit {
                break;
            }
            entries.push(deserialize_entry(val_guard.value())?);
        }

        let has_more = (offset as usize) + entries.len() < total;
        Ok(AuditPage {
            entries,
            total,
            has_more,
        })
    }

    /// Paginated filtered query by kind tag.
    pub fn list_by_kind(&self, kind_tag: u8, offset: u64, limit: usize) -> AuditResult<AuditPage> {
        let txn = self.db.begin_read().map_err(redb_err)?;
        let idx = txn
            .open_multimap_table(AUDIT_KIND_INDEX)
            .map_err(redb_err)?;
        let primary = txn.open_table(AUDIT_TABLE).map_err(redb_err)?;

        // Count total matching entries.
        let total = {
            let values = idx.get(kind_tag).map_err(redb_err)?;
            values.count()
        };

        let mut entries = Vec::with_capacity(limit);
        let values = idx.get(kind_tag).map_err(redb_err)?;
        let mut skipped = 0u64;

        for result in values {
            let raw_id = result.map_err(redb_err)?.value();
            if skipped < offset {
                skipped += 1;
                continue;
            }
            if entries.len() >= limit {
                break;
            }
            let guard = primary
                .get(raw_id)
                .map_err(redb_err)?
                .ok_or(AuditError::NotFound { id: raw_id })?;
            entries.push(deserialize_entry(guard.value())?);
        }

        let has_more = (offset as usize) + entries.len() < total;
        Ok(AuditPage {
            entries,
            total,
            has_more,
        })
    }

    /// Get entries after a given ID (for streaming catch-up).
    pub fn since(&self, after_id: u64, limit: usize) -> AuditResult<Vec<AuditEntry>> {
        let txn = self.db.begin_read().map_err(redb_err)?;
        let table = txn.open_table(AUDIT_TABLE).map_err(redb_err)?;

        let mut entries = Vec::with_capacity(limit);
        let range = table.range((after_id + 1)..).map_err(redb_err)?;

        for result in range {
            if entries.len() >= limit {
                break;
            }
            let (_, val_guard) = result.map_err(redb_err)?;
            entries.push(deserialize_entry(val_guard.value())?);
        }

        Ok(entries)
    }

    /// Total number of audit entries.
    pub fn len(&self) -> AuditResult<usize> {
        let txn = self.db.begin_read().map_err(redb_err)?;
        let table = txn.open_table(AUDIT_TABLE).map_err(redb_err)?;
        Ok(table.len().map_err(redb_err)? as usize)
    }

    /// Whether the ledger has no entries.
    pub fn is_empty(&self) -> AuditResult<bool> {
        self.len().map(|n| n == 0)
    }

    /// Attach a broadcast sender for WS streaming.
    #[cfg(feature = "daemon")]
    pub fn set_broadcast(&self, tx: tokio::sync::broadcast::Sender<AuditEntry>) {
        if let Ok(mut guard) = self.broadcast.lock() {
            *guard = Some(tx);
        }
    }

    /// Attach a TUI message sink for live push.
    pub fn set_sink(&self, sink: Arc<dyn MessageSink>) {
        if let Ok(mut guard) = self.sink.lock() {
            *guard = Some(sink);
        }
    }
}

/// Deserialize a bincode-encoded audit entry.
fn deserialize_entry(bytes: &[u8]) -> AuditResult<AuditEntry> {
    bincode::deserialize(bytes).map_err(|e| {
        AuditError::Store(StoreError::Serialization {
            message: format!("audit entry deserialize: {e}"),
        })
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> Arc<Database> {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("test.redb");
        std::mem::forget(dir);
        Arc::new(Database::create(path).unwrap())
    }

    #[test]
    fn audit_id_niche_optimization() {
        assert_eq!(
            std::mem::size_of::<Option<AuditId>>(),
            std::mem::size_of::<AuditId>()
        );
    }

    #[test]
    fn append_and_retrieve() {
        let db = test_db();
        let ledger = AuditLedger::open(db).unwrap();

        let mut entry = AuditEntry::new(
            AuditKind::AgentLog {
                level: LogLevel::Info,
                message: "hello world".into(),
            },
            "default",
        );

        let id = ledger.append(&mut entry).unwrap();
        assert!(entry.id.is_some());

        let retrieved = ledger.get(id).unwrap();
        assert_eq!(retrieved.workspace, "default");
        assert!(matches!(retrieved.kind, AuditKind::AgentLog { .. }));
    }

    #[test]
    fn auto_incrementing_ids() {
        let db = test_db();
        let ledger = AuditLedger::open(db).unwrap();

        let mut e1 = AuditEntry::new(
            AuditKind::AgentLog {
                level: LogLevel::Info,
                message: "one".into(),
            },
            "ws",
        );
        let mut e2 = AuditEntry::new(
            AuditKind::AgentLog {
                level: LogLevel::Warn,
                message: "two".into(),
            },
            "ws",
        );

        let id1 = ledger.append(&mut e1).unwrap();
        let id2 = ledger.append(&mut e2).unwrap();

        assert_eq!(id1.get(), 1);
        assert_eq!(id2.get(), 2);
    }

    #[test]
    fn list_page_pagination() {
        let db = test_db();
        let ledger = AuditLedger::open(db).unwrap();

        for i in 0..5 {
            let mut entry = AuditEntry::new(
                AuditKind::AgentLog {
                    level: LogLevel::Info,
                    message: format!("msg {i}"),
                },
                "ws",
            );
            ledger.append(&mut entry).unwrap();
        }

        let page = ledger.list_page(0, 3).unwrap();
        assert_eq!(page.entries.len(), 3);
        assert_eq!(page.total, 5);
        assert!(page.has_more);

        let page2 = ledger.list_page(3, 3).unwrap();
        assert_eq!(page2.entries.len(), 2);
        assert!(!page2.has_more);
    }

    #[test]
    fn list_by_kind_filter() {
        let db = test_db();
        let ledger = AuditLedger::open(db).unwrap();

        let mut e1 = AuditEntry::new(
            AuditKind::AgentLog {
                level: LogLevel::Info,
                message: "log".into(),
            },
            "ws",
        );
        let mut e2 = AuditEntry::new(
            AuditKind::ToolInvocation {
                tool_name: "kg_query".into(),
                params_summary: "symbol=Sun".into(),
                success: true,
                output_summary: "found 3 triples".into(),
                duration_ms: 12,
            },
            "ws",
        );
        let mut e3 = AuditEntry::new(
            AuditKind::AgentLog {
                level: LogLevel::Debug,
                message: "debug".into(),
            },
            "ws",
        );

        ledger.append(&mut e1).unwrap();
        ledger.append(&mut e2).unwrap();
        ledger.append(&mut e3).unwrap();

        // Filter: agent logs only (tag=2).
        let page = ledger.list_by_kind(2, 0, 10).unwrap();
        assert_eq!(page.entries.len(), 2);
        assert_eq!(page.total, 2);

        // Filter: tool invocations only (tag=0).
        let page = ledger.list_by_kind(0, 0, 10).unwrap();
        assert_eq!(page.entries.len(), 1);
    }

    #[test]
    fn since_streaming_catchup() {
        let db = test_db();
        let ledger = AuditLedger::open(db).unwrap();

        for i in 0..5 {
            let mut entry = AuditEntry::new(
                AuditKind::AgentLog {
                    level: LogLevel::Info,
                    message: format!("msg {i}"),
                },
                "ws",
            );
            ledger.append(&mut entry).unwrap();
        }

        let entries = ledger.since(3, 10).unwrap();
        assert_eq!(entries.len(), 2); // IDs 4 and 5
    }

    #[test]
    fn persistence_across_reopens() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("persist.redb");
        let db = Arc::new(Database::create(&path).unwrap());

        let stored_id = {
            let ledger = AuditLedger::open(Arc::clone(&db)).unwrap();
            let mut entry = AuditEntry::new(
                AuditKind::ContentIngestion {
                    document_title: "Test Doc".into(),
                    source: "file://test.md".into(),
                    format: "markdown".into(),
                    chunk_count: 3,
                    triple_count: 12,
                },
                "ws",
            );
            ledger.append(&mut entry).unwrap()
        };

        // Drop and reopen.
        drop(db);
        let db2 = Arc::new(Database::create(&path).unwrap());
        let ledger2 = AuditLedger::open(db2).unwrap();

        let retrieved = ledger2.get(stored_id).unwrap();
        assert!(matches!(retrieved.kind, AuditKind::ContentIngestion { .. }));

        // Next ID should continue past the stored one.
        let mut e2 = AuditEntry::new(
            AuditKind::AgentLog {
                level: LogLevel::Info,
                message: "after reopen".into(),
            },
            "ws",
        );
        let id2 = ledger2.append(&mut e2).unwrap();
        assert!(id2.get() > stored_id.get());
    }

    #[test]
    fn len_and_is_empty() {
        let db = test_db();
        let ledger = AuditLedger::open(db).unwrap();

        assert!(ledger.is_empty().unwrap());
        assert_eq!(ledger.len().unwrap(), 0);

        let mut entry = AuditEntry::new(
            AuditKind::AgentLog {
                level: LogLevel::Info,
                message: "test".into(),
            },
            "ws",
        );
        ledger.append(&mut entry).unwrap();

        assert!(!ledger.is_empty().unwrap());
        assert_eq!(ledger.len().unwrap(), 1);
    }

    #[test]
    fn kind_summary_formatting() {
        let tool = AuditKind::ToolInvocation {
            tool_name: "kg_query".into(),
            params_summary: "".into(),
            success: true,
            output_summary: "".into(),
            duration_ms: 42,
        };
        assert!(tool.summary().contains("kg_query"));
        assert!(tool.summary().contains("42ms"));

        let ingest = AuditKind::ContentIngestion {
            document_title: "My Doc".into(),
            source: "".into(),
            format: "".into(),
            chunk_count: 5,
            triple_count: 20,
        };
        assert!(ingest.summary().contains("My Doc"));
        assert!(ingest.summary().contains("5 chunks"));
    }
}
