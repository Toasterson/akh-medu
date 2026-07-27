//! Entity: `heka_drawer_chunks` — text chunks within a drawer.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "heka_drawer_chunks")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Foreign key to `heka_drawers.id`.
    pub drawer_id: Uuid,

    /// Sequential index of this chunk within the drawer.
    pub chunk_index: i32,

    /// The actual text content of the chunk.
    #[sea_orm(column_type = "Text")]
    pub text: String,

    /// Word count of the chunk text.
    pub word_count: i32,

    /// When this chunk was embedded. NULL means embedding is pending.
    pub embedded_at: Option<DateTimeUtc>,

    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::heka_drawers::Entity",
        from = "Column::DrawerId",
        to = "super::heka_drawers::Column::Id"
    )]
    Drawer,

    #[sea_orm(has_many = "super::heka_drawer_embeddings::Entity")]
    Embeddings,
}

impl Related<super::heka_drawers::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Drawer.def()
    }
}

impl Related<super::heka_drawer_embeddings::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Embeddings.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}