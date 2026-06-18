//! Entity: `akhomed.episodes` — episodic memory that survives sleep cycles.
//!
//! The snapshot carries only a recent episode buffer; this table is the durable
//! long-tail so episodic recall works after relocation or a long deep sleep.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(schema_name = "akhomed", table_name = "episodes")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub episode_id: i64,

    pub akh_id: Uuid,
    pub workspace_id: Uuid,

    pub started_at: DateTimeUtc,
    pub ended_at: Option<DateTimeUtc>,

    /// Serialised episode struct.
    #[sea_orm(column_type = "JsonBinary")]
    pub payload: Json,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
