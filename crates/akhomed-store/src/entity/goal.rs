//! Entity: `akhomed.goals` — agent goals. Queryable scalars as columns; the
//! nested optional structures (priority rationale, justification, effort
//! estimate, reformulation, full source) ride in `detail` JSON for fidelity.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_goals")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// Goal `SymbolId.get()`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: i64,
    pub description: String,
    /// `GoalStatus` as text.
    pub status: String,
    pub priority: i16,
    pub success_criteria: String,
    pub parent: Option<i64>,
    pub created_at: i64,
    pub cycles_worked: i32,
    pub last_progress_cycle: i64,
    /// `GoalSource` discriminant as text (full source in `detail`).
    pub source: Option<String>,
    #[sea_orm(column_type = "Json")]
    pub children: Json,
    #[sea_orm(column_type = "Json")]
    pub blocked_by: Json,
    /// Nested optionals: priority_rationale, justification, estimated_effort,
    /// reformulated_from, full source — serialised as one JSON object.
    #[sea_orm(column_type = "Json", nullable)]
    pub detail: Option<Json>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
