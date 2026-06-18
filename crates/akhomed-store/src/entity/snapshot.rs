//! Entity: `akhomed.snapshots` — periodic full snapshots of working state.
//!
//! A snapshot bounds WAL replay distance. It captures the bincode-encoded
//! working memory, episode buffer, and a catch-all state blob (sleep cycle,
//! OODA counter, drives), plus the current psyche as JSONB. The KG itself is
//! **not** snapshotted here — it lives in `akhomed.facts`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(schema_name = "akhomed", table_name = "snapshots")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub snapshot_id: i64,

    pub akh_id: Uuid,

    pub created_at: DateTimeUtc,

    /// `mutation_log.seq` this snapshot covers up to. Replay applies rows with
    /// `seq > watermark`.
    pub watermark: i64,

    /// bincode-encoded `WorkingMemory`.
    pub working_mem: Vec<u8>,
    /// bincode-encoded recent episode buffer.
    pub episodes: Vec<u8>,
    /// bincode-encoded catch-all (sleep cycle, OODA counter, drives).
    pub state_blob: Vec<u8>,

    /// Current psyche at snapshot time.
    #[sea_orm(column_type = "JsonBinary", nullable)]
    pub psyche: Option<Json>,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
