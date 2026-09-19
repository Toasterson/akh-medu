//! Entity: `th_sessions` — derived per-session metadata.
//!
//! One row per `(source, session_id)`, rolled up from that session's
//! `th_events` rows: title, project grouping, span, and stream/record counts
//! for the read surface's `list_sessions`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "th_sessions")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub source: String,
    #[sea_orm(primary_key, auto_increment = false)]
    pub session_id: String,

    /// Human-legible title. Claude Code's repeated `ai-title` collapses here
    /// rather than being re-emitted as 513 identical events (normal-form
    /// ticket, decision 10).
    #[sea_orm(nullable)]
    pub title: Option<String>,
    /// Canonical project identity (git remote/root); sessions group by repo.
    #[sea_orm(nullable)]
    pub repo: Option<String>,
    /// The directory a stream actually ran in — may differ from `repo` for a
    /// worktree or a subagent.
    #[sea_orm(nullable)]
    pub workspace: Option<String>,
    #[sea_orm(nullable)]
    pub branch: Option<String>,

    pub first_seen: DateTimeUtc,
    pub last_seen: DateTimeUtc,
    pub stream_count: i32,
    pub record_count: i64,
    pub derivation_version: i32,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
