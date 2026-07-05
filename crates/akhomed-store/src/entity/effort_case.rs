//! Entity: `akhomed.effort_cases` — CBR effort-estimation records. The HNSW
//! effort index is rebuilt from these rows on wake (`EffortIndex::from_cases`).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_effort_cases")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    #[sea_orm(primary_key, auto_increment = false)]
    pub seq: i32,
    pub description: String,
    /// VSA goal vector bytes.
    pub goal_vector: Vec<u8>,
    pub cycles_consumed: i32,
    pub succeeded: bool,
    /// `HashMap<String, u32>` tool usage (string keys → JSON object).
    #[sea_orm(column_type = "Json")]
    pub tool_usage: Json,
    pub initial_coverage: i32,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
