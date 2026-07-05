//! Entity: `akhomed.learned_method_step` — ordered steps of a learned HTN method.
//!
//! Normalises the former `learned_methods.steps` JSON list (`Vec<GeneralizedStep>`)
//! into child rows, one per step, ordered by `step_index`. Mirrors
//! `crate::agent::chunking::GeneralizedStep` (tool / arg_pattern / expected_outcome).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_learned_method_step")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// Owning method `SymbolId.get()`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub method_id: i64,
    /// Ordinal of this step within the method.
    #[sea_orm(primary_key, auto_increment = false)]
    pub step_index: i32,
    pub tool: String,
    pub arg_pattern: String,
    pub expected_outcome: String,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
