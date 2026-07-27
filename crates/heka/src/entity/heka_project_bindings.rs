//! Entity: `heka_project_bindings` — akh ↔ project associations.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "heka_project_bindings")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Foreign key to `heka_akhs.id`.
    pub akh_id: Uuid,

    /// External project identifier (e.g. Anima project UUID).
    pub project_id: String,

    /// Human-readable project name (denormalized for display).
    pub project_name: Option<String>,

    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(
        belongs_to = "super::heka_akhs::Entity",
        from = "Column::AkhId",
        to = "super::heka_akhs::Column::Id"
    )]
    Akh,
}

impl Related<super::heka_akhs::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Akh.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}