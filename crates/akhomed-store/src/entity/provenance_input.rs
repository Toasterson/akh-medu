//! Entity: `akhomed.provenance_input` — input symbols of a provenance record.
//!
//! Normalises the former `provenance.inputs` JSON `Vec<i64>` into child rows,
//! one per input symbol, ordered by `position`. Keyed by `provenance_id` (the
//! parent ledger row's PK), not `akh_id`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_provenance_input")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub provenance_id: i64,
    /// Ordinal of this input within the record's input list.
    #[sea_orm(primary_key, auto_increment = false)]
    pub position: i32,
    /// Input `SymbolId.get()`.
    pub input_symbol_id: i64,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
