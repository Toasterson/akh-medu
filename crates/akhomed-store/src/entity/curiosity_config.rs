//! Entity: `akhomed.curiosity_config` — directed-curiosity tuning (1 row/Akh).
//!
//! The persisted form is the config only; live curiosity *targets* are recomputed.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_curiosity_config")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    pub max_targets: i32,
    pub proximal_low: f32,
    pub proximal_high: f32,
    pub peak_fill_ratio: f32,
    pub beyond_curiosity_factor: f32,
    pub known_curiosity_factor: f32,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
