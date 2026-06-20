//! Typed snapshot/restore API for the akh's relational state (Phase 38c).
//!
//! All SeaORM access lives here, inside the leaf crate that owns the schema. The
//! daemon's `snapshot` module constructs entity `Model`s from live agent state
//! and calls these helpers, so it needs no direct ORM dependency.
//!
//! Per-Akh cognitive tables are **full-replaced** (DELETE-by-Akh + `insert_many`
//! in a transaction) so the row set always mirrors live state, including
//! removals. `akh_state` is a single upserted row.

use sea_orm::sea_query::OnConflict;
use sea_orm::{
    ColumnTrait, DatabaseConnection, EntityTrait, IntoActiveModel, QueryFilter, QueryOrder,
    QuerySelect, TransactionTrait,
};
use uuid::Uuid;

// The macro-driven helpers fully-qualify their entity modules; only these are
// referenced directly by the hand-written helpers below.
use crate::entity::{akh_state, fact, mutation_log, provenance_input, workspace_seed_pack};
use crate::error::StoreResult;

/// Generate `replace_*` (full-replace for an Akh) and `load_*` helpers for an
/// `akh_id`-scoped entity module.
macro_rules! akh_scoped_ops {
    ($module:ident, $replace:ident, $load:ident) => {
        /// Replace all rows for `akh_id` with `rows` (DELETE + insert, one txn).
        pub async fn $replace(
            db: &DatabaseConnection,
            akh_id: Uuid,
            rows: Vec<crate::entity::$module::Model>,
        ) -> StoreResult<()> {
            use crate::entity::$module::{Column, Entity};
            let txn = db.begin().await?;
            Entity::delete_many()
                .filter(Column::AkhId.eq(akh_id))
                .exec(&txn)
                .await?;
            if !rows.is_empty() {
                Entity::insert_many(rows.into_iter().map(IntoActiveModel::into_active_model))
                    .exec(&txn)
                    .await?;
            }
            txn.commit().await?;
            Ok(())
        }

        /// Load all rows for `akh_id`.
        pub async fn $load(
            db: &DatabaseConnection,
            akh_id: Uuid,
        ) -> StoreResult<Vec<crate::entity::$module::Model>> {
            use crate::entity::$module::{Column, Entity};
            Ok(Entity::find().filter(Column::AkhId.eq(akh_id)).all(db).await?)
        }
    };
}

akh_scoped_ops!(symbol, replace_symbols, load_symbols);
akh_scoped_ops!(drive, replace_drives, load_drives);
akh_scoped_ops!(agenda, replace_agenda, load_agenda);
akh_scoped_ops!(curiosity_config, replace_curiosity, load_curiosity);
akh_scoped_ops!(goal, replace_goals, load_goals);
akh_scoped_ops!(goal_hierarchy, replace_goal_hierarchy, load_goal_hierarchy);
akh_scoped_ops!(goal_dependency, replace_goal_dependencies, load_goal_dependencies);
akh_scoped_ops!(failure_case, replace_failure_cases, load_failure_cases);
akh_scoped_ops!(effort_case, replace_effort_cases, load_effort_cases);
akh_scoped_ops!(learned_method, replace_learned_methods, load_learned_methods);
akh_scoped_ops!(learned_method_step, replace_learned_method_steps, load_learned_method_steps);
akh_scoped_ops!(competence_tool, replace_competence_tools, load_competence_tools);
akh_scoped_ops!(competence_category, replace_competence_categories, load_competence_categories);
akh_scoped_ops!(competence_calibration, replace_competence_calibration, load_competence_calibration);
akh_scoped_ops!(sleep_metric, replace_sleep_metrics, load_sleep_metrics);

/// Replace all facts for a workspace (DELETE-by-workspace + insert, one txn).
pub async fn replace_workspace_facts(
    db: &DatabaseConnection,
    workspace_id: Uuid,
    rows: Vec<fact::Model>,
) -> StoreResult<()> {
    let txn = db.begin().await?;
    fact::Entity::delete_many()
        .filter(fact::Column::WorkspaceId.eq(workspace_id))
        .exec(&txn)
        .await?;
    if !rows.is_empty() {
        fact::Entity::insert_many(rows.into_iter().map(IntoActiveModel::into_active_model))
            .exec(&txn)
            .await?;
    }
    txn.commit().await?;
    Ok(())
}

/// Load all facts for a workspace.
pub async fn load_workspace_facts(
    db: &DatabaseConnection,
    workspace_id: Uuid,
) -> StoreResult<Vec<fact::Model>> {
    Ok(fact::Entity::find()
        .filter(fact::Column::WorkspaceId.eq(workspace_id))
        .all(db)
        .await?)
}

/// Replace all input rows for a provenance record (DELETE-by-`provenance_id` +
/// insert, one txn). `provenance_input` is keyed by `provenance_id`, not
/// `akh_id`, so it cannot use the `akh_scoped_ops!` macro.
pub async fn replace_provenance_inputs(
    db: &DatabaseConnection,
    provenance_id: i64,
    rows: Vec<provenance_input::Model>,
) -> StoreResult<()> {
    let txn = db.begin().await?;
    provenance_input::Entity::delete_many()
        .filter(provenance_input::Column::ProvenanceId.eq(provenance_id))
        .exec(&txn)
        .await?;
    if !rows.is_empty() {
        provenance_input::Entity::insert_many(rows.into_iter().map(IntoActiveModel::into_active_model))
            .exec(&txn)
            .await?;
    }
    txn.commit().await?;
    Ok(())
}

/// Load all input rows for a provenance record, ordered by `position`.
pub async fn load_provenance_inputs(
    db: &DatabaseConnection,
    provenance_id: i64,
) -> StoreResult<Vec<provenance_input::Model>> {
    Ok(provenance_input::Entity::find()
        .filter(provenance_input::Column::ProvenanceId.eq(provenance_id))
        .order_by_asc(provenance_input::Column::Position)
        .all(db)
        .await?)
}

/// Replace all seed-pack rows for a workspace (DELETE-by-`workspace_id` +
/// insert, one txn). `workspace_seed_pack` is keyed by `workspace_id`.
pub async fn replace_workspace_seed_packs(
    db: &DatabaseConnection,
    workspace_id: Uuid,
    rows: Vec<workspace_seed_pack::Model>,
) -> StoreResult<()> {
    let txn = db.begin().await?;
    workspace_seed_pack::Entity::delete_many()
        .filter(workspace_seed_pack::Column::WorkspaceId.eq(workspace_id))
        .exec(&txn)
        .await?;
    if !rows.is_empty() {
        workspace_seed_pack::Entity::insert_many(
            rows.into_iter().map(IntoActiveModel::into_active_model),
        )
        .exec(&txn)
        .await?;
    }
    txn.commit().await?;
    Ok(())
}

/// Load all seed-pack rows for a workspace, ordered by `position`.
pub async fn load_workspace_seed_packs(
    db: &DatabaseConnection,
    workspace_id: Uuid,
) -> StoreResult<Vec<workspace_seed_pack::Model>> {
    Ok(workspace_seed_pack::Entity::find()
        .filter(workspace_seed_pack::Column::WorkspaceId.eq(workspace_id))
        .order_by_asc(workspace_seed_pack::Column::Position)
        .all(db)
        .await?)
}

/// Upsert the single `akh_state` row for an Akh.
pub async fn upsert_akh_state(db: &DatabaseConnection, model: akh_state::Model) -> StoreResult<()> {
    use akh_state::Column;
    akh_state::Entity::insert(model.into_active_model())
        .on_conflict(
            OnConflict::column(Column::AkhId)
                .update_columns([
                    Column::Watermark,
                    Column::AllocatorNext,
                    Column::CycleCount,
                    Column::SessionStartCycle,
                    Column::SleepPhase,
                    Column::LastSleepCycle,
                    Column::Psyche,
                    Column::Extra,
                    Column::UpdatedAt,
                ])
                .to_owned(),
        )
        .exec(db)
        .await?;
    Ok(())
}

/// Load the `akh_state` row for an Akh, if present.
pub async fn load_akh_state(
    db: &DatabaseConnection,
    akh_id: Uuid,
) -> StoreResult<Option<akh_state::Model>> {
    Ok(akh_state::Entity::find_by_id(akh_id).one(db).await?)
}

/// The highest `mutation_log.seq` for an Akh (0 if none) — the snapshot watermark.
pub async fn current_watermark(db: &DatabaseConnection, akh_id: Uuid) -> StoreResult<i64> {
    let max: Option<Option<i64>> = mutation_log::Entity::find()
        .select_only()
        .column_as(mutation_log::Column::Seq.max(), "m")
        .filter(mutation_log::Column::AkhId.eq(akh_id))
        .into_tuple()
        .one(db)
        .await?;
    Ok(max.flatten().unwrap_or(0))
}

/// A replayable mutation from the WAL tail.
#[derive(Debug, Clone)]
pub struct TailMutation {
    pub kind: String,
    pub payload: serde_json::Value,
}

/// Load mutation-log rows with `seq > watermark`, ordered, for wake-time replay.
pub async fn load_mutation_tail(
    db: &DatabaseConnection,
    akh_id: Uuid,
    watermark: i64,
) -> StoreResult<Vec<TailMutation>> {
    let rows = mutation_log::Entity::find()
        .filter(mutation_log::Column::AkhId.eq(akh_id))
        .filter(mutation_log::Column::Seq.gt(watermark))
        .order_by_asc(mutation_log::Column::Seq)
        .all(db)
        .await?;
    Ok(rows
        .into_iter()
        .map(|m| TailMutation {
            kind: m.kind,
            payload: m.payload,
        })
        .collect())
}
