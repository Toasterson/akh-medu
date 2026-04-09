//! Entity: `pa_documents` — ingested document metadata.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "pa_documents")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Human-readable title of the document.
    pub title: String,

    /// URL-safe unique slug for referencing.
    #[sea_orm(unique)]
    pub slug: String,

    /// Original file path or URL.
    pub source_path: Option<String>,

    /// Document format: "pdf", "epub", "mobi", "text", "html".
    pub format: String,

    /// Domain tags for categorization (stored as JSONB).
    #[sea_orm(column_type = "JsonBinary")]
    pub tags: Json,

    /// Arbitrary metadata: author, language, keywords, etc. (JSONB).
    #[sea_orm(column_type = "JsonBinary")]
    pub metadata: Json,

    /// Number of chunks this document was split into.
    pub chunk_count: i32,

    /// When the document was ingested.
    pub ingested_at: DateTimeUtc,

    /// Identifier of the pipeline that ingested this document.
    pub ingested_by: Option<String>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::pa_chunk::Entity")]
    Chunks,
}

impl Related<super::pa_chunk::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Chunks.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
