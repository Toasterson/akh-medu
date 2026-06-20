//! Entity: `akhomed.drives` — the four-drive motivational system (4 rows/Akh).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_drives")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// `DriveKind` label: `curiosity` | `coherence` | `completeness` | `efficiency`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub kind: String,
    pub strength: f32,
    pub last_computed: i64,
    pub last_triple_count: i64,
    pub last_symbol_count: i64,
    /// Activation threshold for this drive.
    pub threshold: f32,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
