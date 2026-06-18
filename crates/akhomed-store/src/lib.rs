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
pub mod migration;
pub mod mutations;
mod pool;

pub use error::{StoreError, StoreResult};
pub use migration::Migrator;
pub use mutations::{MutationRow, append_mutations, ensure_akh};
pub use pool::{PoolOptions, connect};

// Re-export the connection type, the `sea_orm` crate, and `uuid` so downstream
// crates (the `akhomed` daemon, its tests) can name the pool, run raw queries,
// and build Akh/workspace ids without taking direct `sea-orm`/`uuid` deps.
pub use sea_orm::DatabaseConnection;
pub use {sea_orm, uuid};

use sea_orm_migration::MigratorTrait;

/// Run all pending `akhomed-store` migrations against the given connection.
///
/// Migrations run transactionally per-step on Postgres, so a failure leaves the
/// schema in its previous state. Returns [`StoreError::Migrate`] on failure with
/// a diagnostic pointing the operator at the underlying error.
pub async fn run_migrations(db: &DatabaseConnection) -> StoreResult<()> {
    Migrator::up(db, None)
        .await
        .map_err(|source| StoreError::Migrate { source })
}
