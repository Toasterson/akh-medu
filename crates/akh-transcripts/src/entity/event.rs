//! Entity: `th_events` — the normalized cross-tool layer.
//!
//! One of the ten closed [`crate::model::EventKind`]s, produced from a
//! `th_raw_records` row by the derivation worker
//! ([normal-form ticket](../../../docs/wayfinder/transcript-harvest/tickets/event-normal-form.md)).
//!
//! Deliberately a **plain table**, not a hypertable: TimescaleDB would force
//! `observed_at` into the unique index, but the index this table actually
//! needs is `(derivation_version, source, session_id, stream_id, seq)` — the
//! natural key plus the version the derivation worker's anti-join discovery
//! depends on
//! ([derivation-worker ticket](../../../docs/wayfinder/transcript-harvest/tickets/derivation-worker.md)).
//! `observed_at` is carried as an ordinary column for read-surface ordering.
//!
//! Re-derivation bumps `derivation_version` rather than deleting rows: the
//! anti-join finds raw rows with no normalized row *at the current version*,
//! so old rows become simply unreferenced and can be dropped lazily.
//!
//! `degraded` (with `degraded_reason`) marks a partially-recognised shape —
//! this is what stops a poison record from re-pending forever, since a
//! degraded row still satisfies the anti-join.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "th_events")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub derivation_version: i32,
    #[sea_orm(primary_key, auto_increment = false)]
    pub source: String,
    #[sea_orm(primary_key, auto_increment = false)]
    pub session_id: String,
    #[sea_orm(primary_key, auto_increment = false)]
    pub stream_id: String,
    #[sea_orm(primary_key, auto_increment = false)]
    pub seq: i64,

    /// The raw record's `observed_at` (carried, not part of this table's key).
    pub observed_at: DateTimeUtc,

    /// One of the ten closed [`crate::model::EventKind`]s.
    pub kind: String,
    /// Finer classification within a kind, e.g. `context`'s compaction
    /// subtype. Not a second dispatch axis — the ten kinds stay closed.
    #[sea_orm(nullable)]
    pub subkind: Option<String>,
    /// Speaker/role where the source distinguishes one, e.g. `user`,
    /// `assistant`, `developer`.
    #[sea_orm(nullable)]
    pub role: Option<String>,
    /// Readable text content, when the event carries any (prose kinds).
    #[sea_orm(nullable)]
    pub text: Option<String>,
    #[sea_orm(nullable)]
    pub tool_name: Option<String>,
    #[sea_orm(nullable)]
    pub tool_call_id: Option<String>,
    /// Subagent identity, for `tool_call`/`lifecycle` spawn links
    /// (reading-order splicing, per the normal-form ticket).
    #[sea_orm(nullable)]
    pub agent_id: Option<String>,
    #[sea_orm(nullable)]
    pub parent_agent_id: Option<String>,

    /// Compact one-line render (lossy by design) — a derived column, not a
    /// separate table.
    #[sea_orm(nullable)]
    pub render: Option<String>,

    /// True when this row was derived from a partially-recognised shape.
    /// Carries `degraded_reason`; never silent.
    pub degraded: bool,
    #[sea_orm(nullable)]
    pub degraded_reason: Option<String>,

    /// Secret-detector findings (named-pattern matches), if any.
    #[sea_orm(column_type = "JsonBinary", nullable)]
    pub secret_flags: Option<Json>,

    /// Canonical project identity (git remote/root), distinct from
    /// `workspace` (the directory the stream actually ran in).
    #[sea_orm(nullable)]
    pub repo: Option<String>,
    #[sea_orm(nullable)]
    pub workspace: Option<String>,
    #[sea_orm(nullable)]
    pub branch: Option<String>,

    pub derived_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
