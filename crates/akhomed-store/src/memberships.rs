//! Typed API for workspace membership + the split-brain advisory lock (Phase 38f/38g).
//!
//! All SeaORM access lives here, inside the leaf crate that owns the schema, so
//! the daemon never takes a direct ORM dependency. Membership rows
//! (`workspaces_membership`) record which Akhs belong to which workspace —
//! multiple Akhs can join one workspace for knowledge sharing (composite PK
//! `(workspace_id, akh_id)`), distinct from the one-Akh-per-workspace execution
//! identity. The advisory lock ([`try_acquire_akh_lock`]) is the 38g guard that
//! keeps two hosts from waking/relocating the *same* Akh at once.

use sea_orm::sea_query::OnConflict;
use sea_orm::{
    ColumnTrait, DatabaseBackend, DatabaseConnection, EntityTrait, FromQueryResult, QueryFilter,
    Statement,
};
use uuid::Uuid;

use crate::entity::membership;
use crate::error::StoreResult;

/// Add (or update) a membership row: Akh `akh_id` joins workspace `workspace_id`.
///
/// Idempotent — re-joining updates the `role`/`joined_at` of the existing row
/// rather than erroring (composite PK `(workspace_id, akh_id)`). `joined_at` is
/// passed in by the caller (the daemon supplies `chrono::Utc::now()`).
pub async fn join_workspace(
    db: &DatabaseConnection,
    workspace_id: Uuid,
    akh_id: Uuid,
    role: &str,
    joined_at: chrono::DateTime<chrono::Utc>,
) -> StoreResult<()> {
    let model = membership::ActiveModel {
        workspace_id: sea_orm::ActiveValue::Set(workspace_id),
        akh_id: sea_orm::ActiveValue::Set(akh_id),
        role: sea_orm::ActiveValue::Set(role.to_owned()),
        joined_at: sea_orm::ActiveValue::Set(joined_at),
    };
    let result = membership::Entity::insert(model)
        .on_conflict(
            OnConflict::columns([membership::Column::WorkspaceId, membership::Column::AkhId])
                .update_columns([membership::Column::Role, membership::Column::JoinedAt])
                .to_owned(),
        )
        .exec(db)
        .await;
    match result {
        Ok(_) => Ok(()),
        // An update-on-conflict that changes nothing can surface as
        // RecordNotInserted; that is still the success case for an idempotent join.
        Err(sea_orm::DbErr::RecordNotInserted) => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Remove a membership row. Returns `true` if a row was deleted, `false` if the
/// Akh was not a member of the workspace (so callers can report a no-op clearly).
pub async fn leave_workspace(
    db: &DatabaseConnection,
    workspace_id: Uuid,
    akh_id: Uuid,
) -> StoreResult<bool> {
    let result = membership::Entity::delete_many()
        .filter(membership::Column::WorkspaceId.eq(workspace_id))
        .filter(membership::Column::AkhId.eq(akh_id))
        .exec(db)
        .await?;
    Ok(result.rows_affected > 0)
}

/// List all membership rows for a workspace.
pub async fn list_members(
    db: &DatabaseConnection,
    workspace_id: Uuid,
) -> StoreResult<Vec<membership::Model>> {
    Ok(membership::Entity::find()
        .filter(membership::Column::WorkspaceId.eq(workspace_id))
        .all(db)
        .await?)
}

/// One-column boolean projection for the `pg_try_advisory_lock` result.
#[derive(Debug, FromQueryResult)]
struct LockedRow {
    locked: bool,
}

/// Derive the 64-bit advisory-lock key for an Akh from its UUID.
///
/// Postgres advisory locks key on a `bigint`; we hash a stable
/// `akh:<uuid>` string with `hashtext` inside the query so the same Akh always
/// maps to the same lock across hosts. Kept as a helper so the namespace prefix
/// (`akh:`) is defined in exactly one place.
fn akh_lock_subject(akh_id: Uuid) -> String {
    format!("akh:{akh_id}")
}

/// Try to acquire the session-scoped Postgres advisory lock for an Akh (38g
/// split-brain guard). Returns `Ok(true)` if this connection now holds the lock,
/// `Ok(false)` if another host already holds it.
///
/// `pg_try_advisory_lock` is non-blocking; the lock is held for the lifetime of
/// the *connection* (SeaORM may hand the next statement a different pooled
/// connection, so this is a best-effort guard for the duration of one wake, not
/// a durable mutex — release it explicitly with [`release_akh_lock`] when the
/// workspace is evicted). Raw `Statement` is used because `pg_try_advisory_lock`
/// is a DB function with no SeaORM entity — the idiomatic SeaORM escape hatch.
pub async fn try_acquire_akh_lock(db: &DatabaseConnection, akh_id: Uuid) -> StoreResult<bool> {
    let stmt = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "SELECT pg_try_advisory_lock(hashtext($1)) AS locked",
        [akh_lock_subject(akh_id).into()],
    );
    let row = LockedRow::find_by_statement(stmt)
        .one(db)
        .await?
        .ok_or_else(|| {
            sea_orm::DbErr::Custom(
                "pg_try_advisory_lock returned no row (expected exactly one)".to_string(),
            )
        })?;
    Ok(row.locked)
}

/// Release the session-scoped advisory lock for an Akh. Returns `Ok(true)` if a
/// lock held by this connection was released, `Ok(false)` if none was held.
///
/// Symmetric counterpart to [`try_acquire_akh_lock`]; called when a workspace is
/// evicted/slept so a later wake on this host can re-acquire cleanly.
pub async fn release_akh_lock(db: &DatabaseConnection, akh_id: Uuid) -> StoreResult<bool> {
    let stmt = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "SELECT pg_advisory_unlock(hashtext($1)) AS locked",
        [akh_lock_subject(akh_id).into()],
    );
    let row = LockedRow::find_by_statement(stmt).one(db).await?;
    Ok(row.map(|r| r.locked).unwrap_or(false))
}

/// Update an Akh's `last_seen` heartbeat to `now` (38g). Best-effort presence
/// marker read by the split-brain guard; the row must already exist
/// (`ensure_akh` is called first on every relocate/snapshot path).
pub async fn touch_last_seen(
    db: &DatabaseConnection,
    akh_id: Uuid,
    now: chrono::DateTime<chrono::Utc>,
) -> StoreResult<()> {
    use crate::entity::akh;
    let model = akh::ActiveModel {
        id: sea_orm::ActiveValue::Set(akh_id),
        last_seen: sea_orm::ActiveValue::Set(Some(now)),
        ..Default::default()
    };
    // Only update the heartbeat column; the row exists (ensure_akh ran first).
    akh::Entity::update(model)
        .filter(akh::Column::Id.eq(akh_id))
        .exec(db)
        .await?;
    Ok(())
}

/// Clear an Akh's `last_seen` heartbeat — it is no longer resident anywhere.
///
/// `last_seen` answers "was this akh alive recently?", and the guard refuses a
/// wake while it is under 60s old. After a **graceful** sleep/evict that answer
/// is definitively *no*: the snapshot is written, the WAL flushed, the engine
/// dropped and the advisory lock released. Leaving a fresh timestamp behind
/// makes the akh un-wakeable for a minute — including by the very host that
/// just put it down, which is the ordinary config-reload path (`akh sleep`
/// then `akh wake`).
///
/// Clearing here does not weaken the guard: it *sharpens* it. The 60s window
/// exists for hosts that died without releasing anything, and those never reach
/// this call, so their stale heartbeat still blocks a competing wake.
pub async fn clear_last_seen(db: &DatabaseConnection, akh_id: Uuid) -> StoreResult<()> {
    use crate::entity::akh;
    let model = akh::ActiveModel {
        id: sea_orm::ActiveValue::Set(akh_id),
        last_seen: sea_orm::ActiveValue::Set(None),
        ..Default::default()
    };
    akh::Entity::update(model)
        .filter(akh::Column::Id.eq(akh_id))
        .exec(db)
        .await?;
    Ok(())
}

/// Read an Akh's `last_seen` heartbeat (ADR-049 split-brain guard). Returns
/// `None` if the row has no heartbeat (never touched) or the row doesn't exist.
///
/// Used by the lazy-load guard in `get_agent` to check whether another host
/// recently heartbeated for this akh — a fresh `last_seen` means another host
/// owns the akh and a cold lazy-load must refuse to construct a write-capable
/// agent (WAL + snapshot ticker) to prevent split-brain.
pub async fn get_last_seen(
    db: &DatabaseConnection,
    akh_id: Uuid,
) -> StoreResult<Option<chrono::DateTime<chrono::Utc>>> {
    use crate::entity::akh;
    let row = akh::Entity::find_by_id(akh_id).one(db).await?;
    Ok(row.and_then(|m| m.last_seen))
}
