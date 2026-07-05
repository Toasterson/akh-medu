//! Entity: `akhomed.provenance` — the provenance ledger.
//!
//! Mirrors `crate::provenance` in the engine: each derived fact records the
//! `DerivationKind` (as JSONB) and a confidence. The input symbol ids that
//! produced it are normalised into [`crate::entity::provenance_input`] (ordered
//! child rows keyed by `provenance_id`).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_provenance")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub provenance_id: i64,

    pub akh_id: Uuid,
    pub workspace_id: Uuid,

    /// `SymbolId.get()` of the derived subject.
    pub subject: i64,

    /// Serialised `DerivationKind`.
    #[sea_orm(column_type = "Json")]
    pub derivation: Json,

    pub confidence: f32,
    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
