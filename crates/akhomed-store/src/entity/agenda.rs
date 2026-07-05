//! Entity: `akhomed.agenda` — ordered project list with priorities.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_agenda")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,
    /// Project `SymbolId.get()`.
    #[sea_orm(primary_key, auto_increment = false)]
    pub project_id: i64,
    pub priority: i16,
    /// Whether this is the currently active project.
    pub is_active: bool,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
