//! Entity: `akhomed.goal_dependencies` — "blocked-by" goal edges.
//!
//! Normalises the former `goals.blocked_by` JSON list into edge rows: a goal is
//! blocked by each `blocking_goal_id`. One row per `(akh_id, goal_id,
//! blocking_goal_id)`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_goal_dependencies")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// The blocked goal's `SymbolId.get()`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub goal_id: i64,
    /// The blocking goal's `SymbolId.get()`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub blocking_goal_id: i64,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
