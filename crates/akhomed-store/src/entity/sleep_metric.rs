//! Entity: `akhomed.sleep_metrics` — per-sleep-cycle consolidation outcomes
//! (a time series; `phase`/`last_sleep_cycle` live on `akh_state`).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_sleep_metrics")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// OODA cycle at which this sleep episode occurred.
    #[sea_orm(primary_key, auto_increment = false)]
    pub cycle: i64,
    pub episodes_replayed: i32,
    pub duplicates_merged: i32,
    pub orphans_pruned: i32,
    pub dream_connections_found: i32,
    pub equivalences_discovered: i32,
    pub duration_cycles: i64,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
