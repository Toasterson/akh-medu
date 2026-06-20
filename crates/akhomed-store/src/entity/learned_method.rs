//! Entity: `akhomed.learned_methods` — procedurally-learned HTN methods. The
//! HNSW method index is rebuilt from these rows on wake (`MethodIndex::from_methods`).
//!
//! The ordered steps are normalised out into
//! [`crate::entity::learned_method_step`] (one child row per step).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_learned_methods")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// Method `SymbolId.get()`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: i64,
    pub name: String,
    pub precondition_pattern: String,
    /// VSA precondition vector bytes.
    pub precondition_vector: Vec<u8>,
    pub confidence: f32,
    pub usage_count: i32,
    pub success_count: i32,
    pub failure_count: i32,
    pub learned_at: i64,
    pub last_used: i64,
    pub source_goal_description: String,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
