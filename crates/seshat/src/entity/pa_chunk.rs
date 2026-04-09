//! Entity: `pa_chunks` — document text chunks.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "pa_chunks")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Foreign key to `pa_documents.id`.
    pub document_id: Uuid,

    /// Sequential index of this chunk within the document.
    pub chunk_index: i32,

    /// The actual text content of the chunk.
    #[sea_orm(column_type = "Text")]
    pub text: String,

    /// Word count of the chunk text.
    pub word_count: i32,

    /// Chapter number (if applicable).
    pub chapter: Option<i32>,

    /// Section number within chapter (if applicable).
    pub section: Option<i32>,

    /// Section/chapter heading text.
    pub heading: Option<String>,

    /// When this chunk was embedded. NULL means embedding is pending.
    pub embedded_at: Option<DateTimeUtc>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::pa_document::Entity",
        from = "Column::DocumentId",
        to = "super::pa_document::Column::Id"
    )]
    Document,

    #[sea_orm(has_many = "super::pa_embedding::Entity")]
    Embeddings,
}

impl Related<super::pa_document::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Document.def()
    }
}

impl Related<super::pa_embedding::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Embeddings.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
