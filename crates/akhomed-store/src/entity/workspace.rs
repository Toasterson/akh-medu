//! Entity: `workspaces.workspace` — workspace metadata as a first-class row.
//!
//! After Phase 38f, Postgres is authoritative for workspace config; the on-disk
//! TOML in `$XDG_CONFIG_HOME/akh-medu/workspaces/` becomes a bootstrap-only
//! artifact. The seed-pack list is normalised into
//! [`crate::entity::workspace_seed_pack`] (ordered child rows).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "workspaces_workspace")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    #[sea_orm(unique)]
    pub name: String,

    pub dimension: i32,
    pub encoding: String,
    pub language: String,

    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
