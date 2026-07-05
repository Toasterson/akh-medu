//! Entity: `akhomed.failure_cases` — CBR failure records. The HNSW failure index
//! is rebuilt from these rows on wake (`FailureIndex::from_cases`).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_failure_cases")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// Per-Akh ordinal (assigned at snapshot; rows are full-replaced).
    #[sea_orm(primary_key, auto_increment = false)]
    pub seq: i32,
    pub goal_description: String,
    pub tool_name: String,
    pub error_message: String,
    pub cycle: i64,
    pub resolved: bool,
    pub resolution_tool: Option<String>,
    /// Bipolar VSA vector bytes (rebuilds the HNSW index on load).
    pub vector: Vec<u8>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
