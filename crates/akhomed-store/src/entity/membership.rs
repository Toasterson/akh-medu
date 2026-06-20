//! Entity: `workspaces.membership` — which Akhs belong to which workspaces.
//!
//! Multiple Akhs can join one workspace (knowledge sharing, not lockstep
//! execution). Composite primary key `(workspace_id, akh_id)`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "workspaces_membership")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub workspace_id: Uuid,
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,

    pub joined_at: DateTimeUtc,

    /// Membership role, e.g. `member` | `owner`.
    pub role: String,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
