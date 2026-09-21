//! Typed write API for the mutation WAL (Phase 38b).
//!
//! All SeaORM usage for appending mutations lives here, inside the leaf crate
//! that owns the schema. The daemon's `wal` module calls these functions with
//! plain data, so it never takes a direct ORM dependency.

use sea_orm::ActiveValue::{NotSet, Set};
use sea_orm::prelude::DateTimeUtc;
use sea_orm::sea_query::OnConflict;
use sea_orm::sqlx::Connection as _;
use sea_orm::{
    ColumnTrait, ConnectionTrait, DatabaseBackend, DatabaseConnection, EntityTrait, Iterable,
    QueryFilter, QueryOrder, QuerySelect, TransactionTrait,
};
use uuid::Uuid;

use crate::entity::{akh, mutation_log};
use crate::error::{StoreError, StoreResult};
use crate::pool::StablePgSession;

/// Postgres extended-query protocol bind-parameter limit.
///
/// A single prepared statement cannot carry more than 65,535 bind parameters,
/// so any `insert_many` whose `rows × columns` exceeds this must be chunked.
const PG_MAX_BIND_PARAMS: usize = 65_535;

/// A `kind='turn'` WAL row — one captured transcript turn (ADR 091). The
/// daily-log is the projection `WHERE kind='turn' ORDER BY seq`; these rows are
/// never replayed into the KG (see the `apply_mutation` skip in the main
/// crate's `snapshot.rs`).
#[derive(Debug, Clone)]
pub struct TurnRow {
    /// `mutation_log.seq` — also the harvest watermark unit.
    pub seq: i64,
    /// The turn payload, per the locked schema in ADR 091 / the transcript
    /// capture plan (`{v, ts, workspace, model, user_prompt, agent_answer, tools}`).
    pub payload: serde_json::Value,
    pub applied_at: DateTimeUtc,
}

/// A mutation row destined for `akhomed.mutation_log`. `seq` and `applied_at`
/// are assigned by Postgres on insert.
#[derive(Debug, Clone)]
pub struct MutationRow {
    /// `MutationKind::as_str()` value: `assert` | `retract` | ...
    pub kind: String,
    /// Kind-specific JSONB payload.
    pub payload: serde_json::Value,
}

/// Ensure an `akhomed_akh` identity row exists (idempotent).
///
/// Recorded before appending mutations so the Akh identity is known. Uses
/// `INSERT ... ON CONFLICT (id) DO NOTHING`; an existing row is treated as success.
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
        // Timestamps set app-side (no DB defaults) for backend portability.
        awakened_at: Set(chrono::Utc::now()),
        last_seen: NotSet,
        residency_epoch: NotSet,
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
    let now = chrono::Utc::now();
    let limit = if db.get_database_backend() == DatabaseBackend::Sqlite {
        32_766
    } else {
        PG_MAX_BIND_PARAMS
    };
    let chunk_size = (limit / mutation_log::Column::iter().count()).max(1);
    let mut inserted = 0u64;
    let txn = db.begin().await?;
    for chunk in rows.chunks(chunk_size) {
        let models: Vec<mutation_log::ActiveModel> = chunk
            .iter()
            .map(|r| mutation_log::ActiveModel {
                seq: NotSet,
                akh_id: Set(akh_id),
                workspace_id: Set(workspace_id),
                kind: Set(r.kind.clone()),
                payload: Set(r.payload.clone()),
                applied_at: Set(now),
            })
            .collect();
        mutation_log::Entity::insert_many(models).exec(&txn).await?;
        inserted += chunk.len() as u64;
    }
    txn.commit().await?;
    Ok(inserted)
}

/// Append a WAL batch through the exact PostgreSQL session that owns the Akh's
/// residency advisory lock, rejecting a stale fencing generation atomically.
///
/// The workspace publication advisory lock orders this transaction against
/// [`StablePgSession::claim_akh_residency`]. Consequently an old owner can
/// either commit before the handoff advances the epoch, or observe the newer
/// epoch and fail without inserting any row; it cannot straddle the handoff.
pub async fn append_mutations_fenced(
    session: &StablePgSession,
    akh_id: Uuid,
    workspace_id: Uuid,
    residency_epoch: i64,
    rows: &[MutationRow],
) -> StoreResult<u64> {
    if rows.is_empty() {
        return Ok(0);
    }

    if let Some(local) = &session.sqlite {
        return local
            .lock()
            .await
            .append(akh_id, workspace_id, residency_epoch, rows)
            .await;
    }

    // The state mutex covers both the local lock marker and the physical
    // connection for the whole transaction. Explicit release therefore cannot
    // race between the ownership check and commit.
    let mut state = session.lock_state().await;
    if state.locked_akh != Some(akh_id) {
        return Err(StoreError::Conflict {
            message: format!(
                "refusing WAL append for akh {akh_id}: the pinned session does not own its residency lock"
            ),
        });
    }

    let connection = state
        .connection
        .as_mut()
        .ok_or_else(|| StoreError::Conflict {
            message: format!("refusing WAL append for akh {akh_id}: pinned session is closed"),
        })?;
    let mut transaction = connection.begin().await.map_err(sqlx_query_error)?;
    sea_orm::sqlx::query("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))")
        .bind(format!("snapshot:{workspace_id}"))
        .execute(&mut *transaction)
        .await
        .map_err(sqlx_query_error)?;

    let current_epoch: Option<i64> =
        sea_orm::sqlx::query_scalar("SELECT residency_epoch FROM akhomed_akh WHERE id = $1")
            .bind(akh_id)
            .fetch_optional(&mut *transaction)
            .await
            .map_err(sqlx_query_error)?;
    let Some(current_epoch) = current_epoch else {
        transaction.rollback().await.map_err(sqlx_query_error)?;
        return Err(StoreError::Conflict {
            message: format!("cannot append WAL for unknown akh {akh_id}"),
        });
    };
    if current_epoch != residency_epoch {
        transaction.rollback().await.map_err(sqlx_query_error)?;
        return Err(StoreError::Conflict {
            message: format!(
                "stale WAL owner for workspace {workspace_id}: writer residency epoch \
                 {residency_epoch}, current owner epoch {current_epoch}"
            ),
        });
    }

    let applied_at = chrono::Utc::now();
    for row in rows {
        sea_orm::sqlx::query(
            "INSERT INTO akhomed_mutation_log \
             (akh_id, workspace_id, kind, payload, applied_at) \
             VALUES ($1, $2, $3, $4, $5)",
        )
        .bind(akh_id)
        .bind(workspace_id)
        .bind(&row.kind)
        .bind(&row.payload)
        .bind(applied_at)
        .execute(&mut *transaction)
        .await
        .map_err(sqlx_query_error)?;
    }
    transaction.commit().await.map_err(sqlx_query_error)?;
    Ok(rows.len() as u64)
}

fn sqlx_query_error(source: sea_orm::sqlx::Error) -> StoreError {
    StoreError::Db {
        source: sea_orm::DbErr::Query(sea_orm::RuntimeErr::SqlxError(source)),
    }
}

/// Load `kind='turn'` rows for an Akh with `seq > after_seq`, ordered ascending
/// (the sleep-cycle harvest phase's read of the daily-log tail; ADR 091).
pub async fn load_turns_since(
    db: &DatabaseConnection,
    akh_id: Uuid,
    after_seq: i64,
    limit: u64,
) -> StoreResult<Vec<TurnRow>> {
    let rows = mutation_log::Entity::find()
        .filter(mutation_log::Column::AkhId.eq(akh_id))
        .filter(mutation_log::Column::Kind.eq("turn"))
        .filter(mutation_log::Column::Seq.gt(after_seq))
        .order_by_asc(mutation_log::Column::Seq)
        .limit(limit)
        .all(db)
        .await?;
    Ok(rows
        .into_iter()
        .map(|m| TurnRow {
            seq: m.seq,
            payload: m.payload,
            applied_at: m.applied_at,
        })
        .collect())
}

/// Load a single `kind='turn'` row's payload by exact `seq` (the recall-path
/// body-paging read; ADR 091 decision 5). `None` if no such row exists, or the
/// row at that `seq` is not a `turn` row.
pub async fn load_turn_by_seq(
    db: &DatabaseConnection,
    akh_id: Uuid,
    seq: i64,
) -> StoreResult<Option<serde_json::Value>> {
    let row = mutation_log::Entity::find()
        .filter(mutation_log::Column::AkhId.eq(akh_id))
        .filter(mutation_log::Column::Kind.eq("turn"))
        .filter(mutation_log::Column::Seq.eq(seq))
        .one(db)
        .await?;
    Ok(row.map(|m| m.payload))
}
