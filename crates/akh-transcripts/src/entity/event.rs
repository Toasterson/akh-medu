//! Entity: `th_events` — the normalized cross-tool layer.
//!
//! One of the ten closed [`crate::model::EventKind`]s, produced from a
//! `th_raw_records` row by the derivation worker
//! ([normal-form ticket](../../../docs/wayfinder/transcript-harvest/tickets/event-normal-form.md)).
//!
//! Deliberately a **plain table**, not a hypertable: TimescaleDB would force
//! `observed_at` into the unique index, but the index this table actually
//! needs is `(derivation_version, source, session_id, stream_id, seq,
//! subseq)` — the natural key, `subseq` (below), and the version the
//! derivation worker's anti-join discovery depends on
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
//!
//! `subseq` and `disposition` were added by
//! `m20260920_000001_events_subseq_disposition` once the derivation worker
//! ran into the two gaps increment 1 left open: one raw record can yield
//! several events (a Claude `user` record with three `tool_result` blocks —
//! `subseq` is the block index, `0` for a record that yields at most one
//! event), and a record that yields **no** event
//! ([`crate::normalize::Outcome::Noise`], `::Undetermined`, or a hard
//! normalizer failure) must still write a row or the anti-join re-pends it
//! forever — `disposition` (`event | noise | undetermined | failed`, a
//! Postgres `CHECK` constraint) records which case this row is, and `kind`
//! is `NULL` for every disposition except `event` — never an invented
//! pseudo-kind, per the normal-form ticket's closed-ten-kinds stance.

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
    /// Distinguishes multiple events derived from one raw record (the block
    /// index inside e.g. a Claude `user` record's content array). `0` for a
    /// record that yields at most one event — every existing row before
    /// this column was added satisfies that, which is why `0` rather than a
    /// nullable column is the migration's default.
    #[sea_orm(primary_key, auto_increment = false)]
    pub subseq: i32,

    /// The raw record's `observed_at` (carried, not part of this table's key).
    pub observed_at: DateTimeUtc,

    /// One of the ten closed [`crate::model::EventKind`]s. `NULL` unless
    /// `disposition = "event"`.
    #[sea_orm(nullable)]
    pub kind: Option<String>,
    /// `event | noise | undetermined | failed` (Postgres `CHECK`-enforced).
    /// Which arm of [`crate::normalize::Outcome`] produced this row —
    /// including the two arms that produce no event at all, which still
    /// need a row here so the derivation worker's anti-join does not
    /// re-pend them forever.
    pub disposition: String,
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

    /// True when this row was derived from a partially-recognised shape, or
    /// (derivation worker) when normalization panicked outright
    /// (`disposition = "failed"`). Carries `degraded_reason`; never silent.
    pub degraded: bool,
    /// Human-readable explanation. Populated for `degraded = true` rows as
    /// documented above; the derivation worker also repurposes this column
    /// for the `noise` and `undetermined` dispositions' explanation (why a
    /// record has no event) rather than adding a second reason column for
    /// what is, in every case, "why this row does not look like the others".
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
