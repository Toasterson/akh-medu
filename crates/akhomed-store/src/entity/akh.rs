//! Entity: `akhomed.akh` — Akh identities.
//!
//! Identity is the primary key, **not** the host. When a host dies the Akh
//! wakes elsewhere with the same `id` and resumes from its latest snapshot +
//! WAL tail. `home_host` is advisory only.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(schema_name = "akhomed", table_name = "akh")]
pub struct Model {
    /// Stable Akh identity (host-independent).
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: Uuid,

    /// Human-readable Akh name (e.g. "tecton").
    pub name: String,

    /// Preferred akhomed host id. Advisory; relocation can override it.
    pub home_host: Option<String>,

    /// When this Akh was first awakened.
    pub awakened_at: DateTimeUtc,

    /// Heartbeat: last time any host reported this Akh alive. Part of the
    /// split-brain guard on relocation (see sub-phase 38g).
    pub last_seen: Option<DateTimeUtc>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
