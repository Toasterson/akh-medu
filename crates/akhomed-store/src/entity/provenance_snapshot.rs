//! Entity: `akhomed_provenance_snapshot` — portable local-ledger rows.
//!
//! The engine's redb provenance IDs are scoped to one Akh, so the relational
//! primary key is `(akh_id, provenance_id)` rather than the legacy global
//! `akhomed_provenance.provenance_id`.  `sources` is an ordered JSON array;
//! `derivation` is the tagged `DerivationKind` payload.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_provenance_snapshot")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    #[sea_orm(primary_key, auto_increment = false)]
    pub provenance_id: i64,

    pub workspace_id: Uuid,
    pub derived_symbol: i64,
    #[sea_orm(column_type = "Json")]
    pub sources: Json,
    #[sea_orm(column_type = "Json")]
    pub derivation: Json,
    pub confidence: f32,
    pub depth: i64,
    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
