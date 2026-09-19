//! Entity: `th_raw_records` — the verbatim system of record.
//!
//! Identity is `(source, session_id, stream_id, seq)`
//! ([session-identity ticket](../../../docs/wayfinder/transcript-harvest/tickets/session-identity.md)),
//! but the physical primary key also carries `observed_at` because TimescaleDB
//! requires the partition column in every unique index
//! ([store-schema ticket](../../../docs/wayfinder/transcript-harvest/tickets/store-schema.md)).
//!
//! `observed_at` is **forward-fill only** — the record's own timestamp, else
//! the nearest *preceding* timestamp in the same stream, never a following one,
//! else quarantined. `observed_at_synthetic` marks a filled value; it is never
//! set silently. This carry-forward rule is a **frozen contract**
//! ([derivation-worker ticket](../../../docs/wayfinder/transcript-harvest/tickets/derivation-worker.md)):
//! changing it changes the primary key and can silently insert a duplicate row.
//!
//! Raw rows are never written back to (insert-only grant, per
//! [ingest-transport](../../../docs/wayfinder/transcript-harvest/tickets/ingest-transport.md)) —
//! writers use `INSERT ... ON CONFLICT DO NOTHING`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "th_raw_records")]
pub struct Model {
    /// Partition column (TimescaleDB hypertable), forward-fill only. Part of
    /// the primary key alongside the natural `(source, session_id, stream_id,
    /// seq)` key.
    #[sea_orm(primary_key, auto_increment = false)]
    pub observed_at: DateTimeUtc,
    /// `claude_code` | `codex` | `opencode` — [`crate::model::Source::as_str`].
    #[sea_orm(primary_key, auto_increment = false)]
    pub source: String,
    #[sea_orm(primary_key, auto_increment = false)]
    pub session_id: String,
    #[sea_orm(primary_key, auto_increment = false)]
    pub stream_id: String,
    #[sea_orm(primary_key, auto_increment = false)]
    pub seq: i64,

    /// [`crate::model::StreamKind`] as a string, e.g. `main`, `subagent`,
    /// `workflow_journal`, `event_log`.
    pub stream_kind: String,
    /// The machine that harvested this record. Provenance, not identity —
    /// the same session harvested from two hosts converges onto one row per
    /// `(source, session_id, stream_id, seq)` rather than duplicating.
    pub machine: String,
    /// The exact source bytes (one JSONL line, or a serialised SQLite row).
    /// `text`, not `jsonb` — jsonb normalises whitespace/key order, which
    /// would break "verbatim" as a promise.
    pub raw: String,
    /// Parsed form when `raw` is JSON; `None` if it did not parse.
    #[sea_orm(column_type = "JsonBinary", nullable)]
    pub payload: Option<Json>,
    /// The source's own top-level type marker, for `(source, record_type,
    /// schema_marker)` dispatch and drift counting.
    #[sea_orm(nullable)]
    pub record_type: Option<String>,
    #[sea_orm(nullable)]
    pub schema_marker: Option<String>,
    /// File path for JSONL sources; the SQLite path for OpenCode.
    pub origin_path: String,
    /// True when `observed_at` was forward-filled rather than read from the
    /// record itself.
    pub observed_at_synthetic: bool,
    /// When the harvester pushed this row, distinct from `observed_at`
    /// (the record's own, possibly filled, timestamp).
    pub harvested_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
