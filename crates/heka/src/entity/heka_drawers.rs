//! Entity: `heka_drawers` — top-level memory entries.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "heka_drawers")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Foreign key to `heka_akhs.id`.
    pub akh_id: Uuid,

    /// Optional project scope (e.g. Anima project UUID).
    pub project_id: Option<String>,

    /// Human-readable title.
    pub title: String,

    /// One-sentence summary of the drawer contents.
    pub summary: Option<String>,

    /// Domain tags for categorization (stored as JSONB).
    #[sea_orm(column_type = "JsonBinary")]
    pub tags: Json,

    /// Arbitrary metadata (JSONB).
    #[sea_orm(column_type = "JsonBinary")]
    pub metadata: Json,

    // --- Phase 42c: drawer ingest path columns ---

    /// SHA-256 hex digest of the drawer body. Used for idempotent re-ingest:
    /// the UNIQUE(akh_id, body_hash) index ensures re-capturing the same
    /// session produces no duplicate rows.
    pub body_hash: Option<String>,

    /// Claude Code session identifier (from the stop-hook payload).
    /// Links this drawer to the session that produced it.
    pub session_id: Option<String>,

    /// Speaker who produced this turn: "user", "assistant", or "system".
    pub speaker: Option<String>,

    /// Freeform contact label (e.g. "claude-code@laptop", "tecton@worktree-3").
    pub contact: Option<String>,

    /// Topic tags extracted from the turn (JSONB array of strings).
    #[sea_orm(column_type = "JsonBinary")]
    pub topics: Json,

    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::heka_akhs::Entity",
        from = "Column::AkhId",
        to = "super::heka_akhs::Column::Id"
    )]
    Akh,

    #[sea_orm(has_many = "super::heka_drawer_chunks::Entity")]
    Chunks,
}

impl Related<super::heka_akhs::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Akh.def()
    }
}

impl Related<super::heka_drawer_chunks::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Chunks.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}