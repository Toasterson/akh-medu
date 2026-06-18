//! Entity: `akhomed.psyche` — the live, current-row psyche per Akh.
//!
//! `akhomed.snapshots.psyche` holds historical psyche tied to a watermark;
//! this table holds the single current row, kept current for AGE / external
//! read consumers. Mirrors `crate::compartment::psyche::Psyche` in the engine.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(schema_name = "akhomed", table_name = "psyche")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,

    #[sea_orm(column_type = "JsonBinary")]
    pub persona: Json,
    #[sea_orm(column_type = "JsonBinary")]
    pub shadow: Json,
    #[sea_orm(column_type = "JsonBinary")]
    pub archetypes: Json,
    #[sea_orm(column_type = "JsonBinary")]
    pub self_integration: Json,

    pub awakened: bool,
    pub updated_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
