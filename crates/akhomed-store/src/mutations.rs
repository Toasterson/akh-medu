//! Typed write API for the mutation WAL (Phase 38b).
//!
//! All SeaORM usage for appending mutations lives here, inside the leaf crate
//! that owns the schema. The daemon's `wal` module calls these functions with
//! plain data, so it never takes a direct ORM dependency.

use sea_orm::ActiveValue::{NotSet, Set};
use sea_orm::sea_query::OnConflict;
use sea_orm::{DatabaseConnection, EntityTrait};
use uuid::Uuid;

use crate::entity::{akh, mutation_log};
use crate::error::StoreResult;

/// A mutation row destined for `akhomed.mutation_log`. `seq` and `applied_at`
/// are assigned by Postgres on insert.
#[derive(Debug, Clone)]
pub struct MutationRow {
    /// `MutationKind::as_str()` value: `assert` | `retract` | ...
    pub kind: String,
    /// Kind-specific JSONB payload.
    pub payload: serde_json::Value,
}

/// Ensure an `akhomed.akh` identity row exists (idempotent).
///
/// Required before appending mutations: `mutation_log.akh_id` has a foreign key
/// to `akh.id`. Uses `INSERT ... ON CONFLICT (id) DO NOTHING`; an existing row
/// is treated as success.
pub async fn ensure_akh(
    db: &DatabaseConnection,
    akh_id: Uuid,
    name: &str,
    home_host: Option<&str>,
) -> StoreResult<()> {
    let model = akh::ActiveModel {
        id: Set(akh_id),
        name: Set(name.to_owned()),
        home_host: Set(home_host.map(str::to_owned)),
        awakened_at: NotSet, // DB default now()
        last_seen: NotSet,
    };
    let result = akh::Entity::insert(model)
        .on_conflict(OnConflict::column(akh::Column::Id).do_nothing().to_owned())
        .exec(db)
        .await;
    match result {
        Ok(_) => Ok(()),
        // `do_nothing` against an existing row reports RecordNotInserted; that
        // is the success case for an idempotent ensure.
        Err(sea_orm::DbErr::RecordNotInserted) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Append a batch of mutations in a single multi-row insert.
///
/// Returns the number of rows submitted. Idempotency of *replay* is guaranteed
/// at the engine boundary (re-asserting a present triple is a no-op), so the
/// WAL uses at-least-once delivery: a batch is removed from the in-memory
/// buffer only after this call commits.
pub async fn append_mutations(
    db: &DatabaseConnection,
    akh_id: Uuid,
    workspace_id: Uuid,
    rows: &[MutationRow],
) -> StoreResult<u64> {
    if rows.is_empty() {
        return Ok(0);
    }
    let models: Vec<mutation_log::ActiveModel> = rows
        .iter()
        .map(|r| mutation_log::ActiveModel {
            seq: NotSet,
            akh_id: Set(akh_id),
            workspace_id: Set(workspace_id),
            kind: Set(r.kind.clone()),
            payload: Set(r.payload.clone()),
            applied_at: NotSet,
        })
        .collect();
    mutation_log::Entity::insert_many(models).exec(db).await?;
    Ok(rows.len() as u64)
}
