//! Entity: `akhomed.akh_state` — the per-Akh runtime-state row (one per Akh, upserted).
//!
//! Replaces 38a's blob-bearing `snapshots` table. Holds the small scalar
//! working-state plus the recovery boundary; the larger cognitive state lives in
//! the dedicated cognitive tables, and a few misc subsystems ride in `extra`.

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_akh_state")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub akh_id: Uuid,

    /// `mutation_log.seq` covered by this state — wake replays rows `seq > watermark`.
    pub watermark: i64,
    /// The symbol-id allocator counter (`sym_allocator_next`) at snapshot time.
    pub allocator_next: i64,
    /// OODA cycle counter.
    pub cycle_count: i64,
    /// Cycle at the start of the persisted session.
    pub session_start_cycle: i64,

    /// Current sleep phase (`ConsolidationPhase` as text).
    pub sleep_phase: String,
    /// Cycle of the last sleep episode.
    pub last_sleep_cycle: i64,

    /// Current psyche document.
    #[sea_orm(column_type = "Json", nullable)]
    pub psyche: Option<Json>,
    /// Misc JSON state not worth its own table (method_stats, audience,
    /// session_summary, improvement_history, residual reasoning subsystems).
    #[sea_orm(column_type = "Json", nullable)]
    pub extra: Option<Json>,

    pub updated_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
