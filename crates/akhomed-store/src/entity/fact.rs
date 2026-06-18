//! Entity: `akhomed.facts` — every triple ever asserted by any Akh.
//!
//! Backs both the in-process oxigraph cache (loaded on wake) and the AGE read
//! view (sub-phase 38d). Soft-deleted via `retracted_at` rather than row
//! removal so the WAL and AGE projection stay append-mostly.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(schema_name = "akhomed", table_name = "facts")]
pub struct Model {
    /// Surrogate key (BIGSERIAL).
    #[sea_orm(primary_key)]
    pub fact_id: i64,

    /// Asserting Akh.
    pub akh_id: Uuid,

    /// Owning workspace.
    pub workspace_id: Uuid,

    /// `SymbolId.get()` of the subject.
    pub subject: i64,
    /// `SymbolId.get()` of the predicate.
    pub predicate: i64,
    /// `SymbolId.get()` of the object.
    pub object: i64,

    /// Assertion confidence in `[0, 1]`.
    pub confidence: f32,

    pub asserted_at: DateTimeUtc,

    /// Set when the fact is retracted; `None` while live.
    pub retracted_at: Option<DateTimeUtc>,

    /// Optional link into `akhomed.provenance`.
    pub provenance_id: Option<i64>,

    /// Optional compartment / microtheory tag.
    pub compartment: Option<String>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
