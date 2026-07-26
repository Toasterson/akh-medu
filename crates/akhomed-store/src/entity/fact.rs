//! Entity: `akhomed.facts` — every triple ever asserted by any Akh.
//!
//! Backs both the in-process oxigraph cache (loaded on wake) and the AGE read
//! view (sub-phase 38d). Soft-deleted via `retracted_at` rather than row
//! removal so the WAL and AGE projection stay append-mostly.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_facts")]
pub struct Model {
    /// Owning workspace (part of the content-addressed primary key).
    #[sea_orm(primary_key, auto_increment = false)]
    pub workspace_id: Uuid,
    /// `SymbolId.get()` of the subject.
    #[sea_orm(primary_key, auto_increment = false)]
    pub subject: i64,
    /// `SymbolId.get()` of the predicate.
    #[sea_orm(primary_key, auto_increment = false)]
    pub predicate: i64,
    /// `SymbolId.get()` of the object.
    #[sea_orm(primary_key, auto_increment = false)]
    pub object: i64,

    /// Asserting Akh.
    pub akh_id: Uuid,

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
