//! Entity: `heka_akhs` — registered akh instances.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "heka_akhs")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Unique akh name (e.g. "tecton", "thoth").
    #[sea_orm(unique)]
    pub name: String,

    /// Human-readable description.
    pub description: Option<String>,

    /// Personality / prompt-flavour text.
    pub personality: Option<String>,

    /// Avatar URL (optional).
    pub avatar_url: Option<String>,

    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::heka_project_bindings::Entity")]
    ProjectBindings,

    #[sea_orm(has_many = "super::heka_drawers::Entity")]
    Drawers,
}

impl Related<super::heka_project_bindings::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::ProjectBindings.def()
    }
}

impl Related<super::heka_drawers::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Drawers.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}