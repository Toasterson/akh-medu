//! `akhomed-store` — the Postgres-backed durable store for relocatable Akhs.
//!
//! Phase 38 restructures `akhomed` so the truth of an Akh's state lives in
//! Postgres while the daemon keeps an in-process cache for hot-path reasoning.
//! This crate is sub-phase **38a**: the schema, the SeaORM entities, the
//! migration runner, and the connection-pool builder. Later sub-phases layer
//! the mutation WAL (38b), snapshot/replay (38c), and the workspace/multi-Akh
//! sharing semantics (38f) on top of these tables.
//!
//! The crate is deliberately a leaf: it depends on nothing in the main
//! `akh-medu` crate so it can be linked into the daemon binary, exercised by
//! the standalone `akhomed-migrate` CLI, and unit-tested without booting an
//! Engine.
//!
//! See `docs/ai/plans/2026-04-25-phase38-relocatable-akh.md` and ADR 049.

pub mod entity;
pub mod error;
pub mod memberships;
pub mod migration;
pub mod mutations;
pub mod schema_registry;
pub mod snapshots;
mod pool;

pub use error::{StoreError, StoreResult};
pub use memberships::{
    join_workspace, leave_workspace, list_members, release_akh_lock, touch_last_seen,
    try_acquire_akh_lock,
};
pub use migration::Migrator;
pub use mutations::{
    MutationRow, TurnRow, append_mutations, ensure_akh, load_turn_by_seq, load_turns_since,
};
pub use pool::{PoolOptions, connect};
pub use schema_registry::{
    SchemaPackRow, admit_pack, get_pack, latest_pack, list_active_microtheories, list_packs,
    record_pack_activation,
};
pub use snapshots::{TailMutation, get_harvest_watermark, set_harvest_watermark};

// Re-export the connection type, the `sea_orm` crate, and `uuid` so downstream
// crates (the `akhomed` daemon, its tests) can name the pool, run raw queries,
// and build Akh/workspace ids without taking direct `sea-orm`/`uuid` deps.
pub use sea_orm::DatabaseConnection;
pub use {chrono, sea_orm, uuid};

use sea_orm::{ConnectionTrait, DatabaseBackend, Statement};
use sea_orm_migration::MigratorTrait;

/// Run all pending `akhomed-store` migrations against the given connection.
///
/// Migrations run transactionally per-step on Postgres, so a failure leaves the
/// schema in its previous state. Returns [`StoreError::Migrate`] on failure with
/// a diagnostic pointing the operator at the underlying error.
///
/// **Concurrency caveat:** this takes a *shared* pool, whose statements may each
/// run on a different pooled connection — so it is not safe to run from several
/// processes against the *same fresh* database simultaneously (they race on
/// creating `seaql_migrations`). Use [`run_migrations_locked`] for the daemon
/// start-up / CLI paths where multiple hosts may migrate at once. This plain
/// variant is for callers that already hold a connection and guarantee a single
/// migrator (e.g. serial tests).
pub async fn run_migrations(db: &DatabaseConnection) -> StoreResult<()> {
    Migrator::up(db, None)
        .await
        .map_err(|source| StoreError::Migrate { source })
}

/// Advisory-lock key serializing concurrent migrators.
///
/// Multiple daemons may start against the same fresh cluster database at once —
/// the multi-host homing scenario where one akh per host points at the shared
/// Postgres. A naive concurrent `Migrator::up` races on creating
/// `seaql_migrations` and fails with a duplicate-key error on `pg_type`. The
/// value is arbitrary but must stay stable and distinct from the per-akh
/// `hashtext('akh:'||id)` keys used in [`memberships`]. Mnemonic ASCII "AKMDMIGR".
const MIGRATION_LOCK_KEY: i64 = 0x414B_4D44_4D49_4752_u64 as i64;

/// Run migrations under a Postgres session **advisory lock**, serializing
/// concurrent migrators across processes and hosts.
///
/// Unlike [`run_migrations`] (whose pooled statements may each land on a
/// different connection, so a session lock could not span the whole migration),
/// this opens a dedicated **single-connection** pool: every statement
/// `Migrator::up` issues reuses the one connection that holds the lock, and the
/// session lock auto-releases when that connection closes — on success or error.
/// Other migrators block on the lock until the first finishes, then observe the
/// migrations already applied and no-op. On non-Postgres backends there is
/// nothing to serialize and this behaves like [`run_migrations`].
pub async fn run_migrations_locked(opts: &PoolOptions) -> StoreResult<()> {
    // Dedicated single connection so the advisory lock and every migration
    // statement share one backend session.
    let mut single = opts.clone();
    single.max_connections = 1;
    let db = connect(&single).await?;

    let backend = db.get_database_backend();
    if backend == DatabaseBackend::Postgres {
        db.execute(Statement::from_string(
            backend,
            format!("SELECT pg_advisory_lock({MIGRATION_LOCK_KEY})"),
        ))
        .await
        .map_err(|source| StoreError::Migrate { source })?;
    }

    let result = Migrator::up(&db, None)
        .await
        .map_err(|source| StoreError::Migrate { source });

    // Explicit unlock so the lock frees promptly; dropping `db` also releases it
    // when the connection closes. Best-effort — the migration result is what we
    // report.
    if backend == DatabaseBackend::Postgres {
        let _ = db
            .execute(Statement::from_string(
                backend,
                format!("SELECT pg_advisory_unlock({MIGRATION_LOCK_KEY})"),
            ))
            .await;
    }
    result
}
