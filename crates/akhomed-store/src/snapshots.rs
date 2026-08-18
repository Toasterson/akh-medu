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
    ActiveValue, ColumnTrait, DatabaseConnection, EntityTrait, IntoActiveModel, QueryFilter,
    QueryOrder, QuerySelect, TransactionTrait,
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
///
/// The `akhomed_facts` primary key is `(workspace_id, subject, predicate,
/// object)`, so duplicate triples in the captured snapshot would otherwise
/// raise a unique-key violation on every write after the first. We
/// deduplicate the input here as a belt-and-suspenders guard and use
/// `ON CONFLICT DO NOTHING` at the database boundary.
pub async fn replace_workspace_facts(
    db: &DatabaseConnection,
    workspace_id: Uuid,
    rows: Vec<fact::Model>,
) -> StoreResult<()> {
    let input_len = rows.len();
    let mut seen = std::collections::HashSet::with_capacity(input_len);
    let mut unique_rows = Vec::with_capacity(input_len);
    for row in rows {
        let key = (row.subject, row.predicate, row.object);
        if seen.insert(key) {
            unique_rows.push(row);
        }
    }
    if unique_rows.len() < input_len {
        tracing::debug!(
            workspace_id = %workspace_id,
            total = input_len,
            unique = unique_rows.len(),
            "deduplicated duplicate facts before snapshot insert"
        );
    }

    let txn = db.begin().await?;
    fact::Entity::delete_many()
        .filter(fact::Column::WorkspaceId.eq(workspace_id))
        .exec(&txn)
        .await?;
    if !unique_rows.is_empty() {
        fact::Entity::insert_many(
            unique_rows.into_iter().map(IntoActiveModel::into_active_model),
        )
        .on_conflict(
            OnConflict::columns([
                fact::Column::WorkspaceId,
                fact::Column::Subject,
                fact::Column::Predicate,
                fact::Column::Object,
            ])
            .do_nothing()
            .to_owned(),
        )
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
///
/// **Clobber guard (ADR 091 / ADR 081 slice 4):** `harvest_watermark` and
/// `librarian_watermark` are deliberately ABSENT from `update_columns` below.
/// A full-state snapshot upsert runs on every sleep tick / relocation and
/// would otherwise overwrite them with whatever stale value the caller's
/// in-memory `akh_state::Model` happens to carry — resetting progress and
/// causing the sleep-cycle harvest phase / daily-Librarian pass to re-walk
/// (and re-mint episodes / re-extract triples for) turns they already
/// processed. Each column must be written ONLY by its dedicated setter
/// ([`set_harvest_watermark`] / [`set_librarian_watermark`]), so on conflict
/// both are left untouched; the values in `model` only matter for the
/// first-ever insert of a row (see those setters' docs for how callers keep
/// them correct).
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
                    // NOTE: Column::HarvestWatermark / Column::LibrarianWatermark
                    // intentionally omitted — see the clobber-guard doc comment above.
                ])
                .to_owned(),
        )
        .exec(db)
        .await?;
    Ok(())
}

/// Read the sleep-cycle harvest watermark for an Akh (ADR 091). `0` if no
/// `akh_state` row exists yet (nothing harvested).
pub async fn get_harvest_watermark(db: &DatabaseConnection, akh_id: Uuid) -> StoreResult<i64> {
    Ok(akh_state::Entity::find_by_id(akh_id)
        .select_only()
        .column(akh_state::Column::HarvestWatermark)
        .into_tuple::<i64>()
        .one(db)
        .await?
        .unwrap_or(0))
}

/// Advance the sleep-cycle harvest watermark for an Akh (ADR 091).
///
/// A targeted `UPDATE` of just `harvest_watermark` (+ `updated_at`) — never
/// routes through [`upsert_akh_state`], so it can never be clobbered by (and
/// never clobbers) the rest of the snapshot state. If no `akh_state` row exists
/// yet, inserts a minimal default row carrying only this watermark; the next
/// full snapshot upsert fills in the rest of the columns without touching this
/// one (see the clobber-guard note on [`upsert_akh_state`]).
pub async fn set_harvest_watermark(db: &DatabaseConnection, akh_id: Uuid, seq: i64) -> StoreResult<()> {
    use akh_state::Column;
    let now = chrono::Utc::now();
    let existing = akh_state::Entity::find_by_id(akh_id).one(db).await?;
    match existing {
        Some(row) => {
            let mut active = row.into_active_model();
            active.harvest_watermark = ActiveValue::Set(seq);
            active.updated_at = ActiveValue::Set(now);
            akh_state::Entity::update(active)
                .filter(Column::AkhId.eq(akh_id))
                .exec(db)
                .await?;
        }
        None => {
            // No row yet: insert a minimal default carrying only the watermark.
            // A later full snapshot upsert fills in the rest via `update_columns`
            // (which excludes `harvest_watermark`, so this value survives).
            let model = akh_state::Model {
                akh_id,
                watermark: 0,
                allocator_next: 0,
                cycle_count: 0,
                session_start_cycle: 0,
                sleep_phase: "Wake".to_owned(),
                last_sleep_cycle: 0,
                psyche: None,
                extra: None,
                harvest_watermark: seq,
                librarian_watermark: 0,
                updated_at: now,
            };
            akh_state::Entity::insert(model.into_active_model())
                .on_conflict(
                    OnConflict::column(Column::AkhId)
                        .update_columns([Column::HarvestWatermark, Column::UpdatedAt])
                        .to_owned(),
                )
                .exec(db)
                .await?;
        }
    }
    Ok(())
}

/// Read the daily-Librarian watermark for an Akh (ADR 081 slice 4). `0` if no
/// `akh_state` row exists yet (nothing extracted). Independent of
/// [`get_harvest_watermark`] — see the field doc on
/// [`akh_state::Model::librarian_watermark`].
pub async fn get_librarian_watermark(db: &DatabaseConnection, akh_id: Uuid) -> StoreResult<i64> {
    Ok(akh_state::Entity::find_by_id(akh_id)
        .select_only()
        .column(akh_state::Column::LibrarianWatermark)
        .into_tuple::<i64>()
        .one(db)
        .await?
        .unwrap_or(0))
}

/// Advance the daily-Librarian watermark for an Akh (ADR 081 slice 4).
///
/// A targeted `UPDATE` of just `librarian_watermark` (+ `updated_at`) — never
/// routes through [`upsert_akh_state`], mirroring [`set_harvest_watermark`]'s
/// clobber-guard pattern exactly (see that function's doc). If no `akh_state`
/// row exists yet, inserts a minimal default row carrying only this watermark.
pub async fn set_librarian_watermark(db: &DatabaseConnection, akh_id: Uuid, seq: i64) -> StoreResult<()> {
    use akh_state::Column;
    let now = chrono::Utc::now();
    let existing = akh_state::Entity::find_by_id(akh_id).one(db).await?;
    match existing {
        Some(row) => {
            let mut active = row.into_active_model();
            active.librarian_watermark = ActiveValue::Set(seq);
            active.updated_at = ActiveValue::Set(now);
            akh_state::Entity::update(active)
                .filter(Column::AkhId.eq(akh_id))
                .exec(db)
                .await?;
        }
        None => {
            // No row yet: insert a minimal default carrying only the watermark.
            // A later full snapshot upsert fills in the rest via `update_columns`
            // (which excludes `librarian_watermark`, so this value survives).
            let model = akh_state::Model {
                akh_id,
                watermark: 0,
                allocator_next: 0,
                cycle_count: 0,
                session_start_cycle: 0,
                sleep_phase: "Wake".to_owned(),
                last_sleep_cycle: 0,
                psyche: None,
                extra: None,
                harvest_watermark: 0,
                librarian_watermark: seq,
                updated_at: now,
            };
            akh_state::Entity::insert(model.into_active_model())
                .on_conflict(
                    OnConflict::column(Column::AkhId)
                        .update_columns([Column::LibrarianWatermark, Column::UpdatedAt])
                        .to_owned(),
                )
                .exec(db)
                .await?;
        }
    }
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
