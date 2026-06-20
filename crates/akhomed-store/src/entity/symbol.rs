//! Entity: `akhomed.symbols` — the symbol registry as relational rows.
//!
//! Replaces the `sym_meta:` redb blob. Restoring these on a fresh host is what
//! makes the numeric ids in `akhomed.facts` line up (well-known predicates keep
//! their ids), so `AgentPredicates::init` resolves consistently.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_symbols")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// `SymbolId.get()`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: i64,
    /// `SymbolKind` discriminant as text (`Entity` | `Relation` | `Composite` | `Glyph`).
    pub kind: String,
    /// Unicode codepoint for `Glyph` symbols; `None` otherwise.
    pub glyph_codepoint: Option<i32>,
    /// Human-readable label.
    pub label: String,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
