//! Entity: `akhomed.mutation_log` — the streaming write-ahead log.
//!
//! Every assert / retract / microtheory mutation lands here within seconds
//! (sub-phase 38b). On wake, rows newer than the latest snapshot watermark are
//! replayed through `Engine::add_triple` (idempotent).

use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "akhomed_mutation_log")]
pub struct Model {
    /// Monotonic per-store sequence (BIGSERIAL); also the snapshot watermark unit.
    #[sea_orm(primary_key)]
    pub seq: i64,

    pub akh_id: Uuid,
    pub workspace_id: Uuid,

    /// Mutation kind: `assert` | `retract` | `microtheory_load` |
    /// `microtheory_unload` | `compartment_create` | `compartment_destroy` |
    /// `turn`. `turn` (ADR 091) is raw transcript-capture staging, not a KG
    /// mutation — wake-replay explicitly skips it (see the main crate's
    /// `snapshot.rs::apply_mutation`); the sleep-cycle harvest phase reads
    /// `kind='turn'` rows to mint `episode:turn:*` indices.
    pub kind: String,

    /// Kind-specific payload.
    #[sea_orm(column_type = "Json")]
    pub payload: Json,

    pub applied_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
