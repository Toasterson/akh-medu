//! Entity: `workspaces.workspace_seed_pack` — seed packs applied on first
//! activation of a workspace.
//!
//! Normalises the former `workspace.seed_packs` JSON `Vec<String>` into child
//! rows, one per pack, ordered by `position`. Keyed by `workspace_id`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "workspaces_workspace_seed_pack")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub workspace_id: Uuid,
    /// Ordinal of this pack within the workspace's seed-pack list.
    #[sea_orm(primary_key, auto_increment = false)]
    pub position: i32,
    pub pack_name: String,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
