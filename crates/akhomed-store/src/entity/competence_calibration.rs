//! Entity: `akhomed.competence_calibration` — calibration history (Brier score).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_competence_calibration")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// Per-Akh ordinal (full-replaced at snapshot).
    #[sea_orm(primary_key, auto_increment = false)]
    pub seq: i32,
    pub predicted: f32,
    /// 0.0 or 1.0.
    pub actual: f32,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
