//! Entity: `akhomed.provenance` — the provenance ledger.
//!
//! Mirrors `crate::provenance` in the engine: each derived fact records the
//! `DerivationKind` (as JSONB), the input symbol ids that produced it, and a
//! confidence. `inputs` is a Postgres `BIGINT[]` mapped to `Vec<i64>`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(schema_name = "akhomed", table_name = "provenance")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub provenance_id: i64,

    pub akh_id: Uuid,
    pub workspace_id: Uuid,

    /// `SymbolId.get()` of the derived subject.
    pub subject: i64,

    /// Serialised `DerivationKind`.
    #[sea_orm(column_type = "JsonBinary")]
    pub derivation: Json,

    /// Input `SymbolId`s (as `i64`) that produced this fact.
    pub inputs: Vec<i64>,

    pub confidence: f32,
    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
