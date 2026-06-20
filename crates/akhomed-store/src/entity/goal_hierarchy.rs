//! Entity: `akhomed.goal_hierarchy` — parent→child goal edges.
//!
//! Normalises the former `goals.children` JSON list into proper edge rows so the
//! goal DAG is queryable and the child ordering is explicit (`position`). One row
//! per `(akh_id, parent_id, child_id)`; `position` preserves the live ordering of
//! a parent's children for lossless round-trip.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_goal_hierarchy")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// Parent goal `SymbolId.get()`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub parent_id: i64,
    /// Child goal `SymbolId.get()`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub child_id: i64,
    /// Ordinal of this child within the parent's children list.
    pub position: i32,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
