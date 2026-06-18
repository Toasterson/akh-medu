//! Entity: `workspaces.active_microtheories` — microtheories activated in a
//! workspace, and which Akh loaded each. Composite primary key
//! `(workspace_id, microtheory_id)`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(schema_name = "workspaces", table_name = "active_microtheories")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub workspace_id: Uuid,
    #[sea_orm(primary_key, auto_increment = false)]
    pub microtheory_id: String,

    pub loaded_by: Uuid,
    pub loaded_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
