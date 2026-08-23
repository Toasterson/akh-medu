//! Database migrations for the Heka pgvector schema.
//!
//! Uses SeaORM's migration runner. The actual DDL is in the raw SQL files
//! under `crates/heka/migrations/` — SeaORM's `ColumnDef` does not support
//! pgvector types, so we execute the SQL directly via `execute_unprepared`,
//! following the Seshat pattern.

use sea_orm::DatabaseConnection;
use sea_orm_migration::prelude::*;

mod m20260427_000001_init;
mod m20260604_000002_drawer_ingest;
mod m20260604_000003_embedding_idempotency;
mod m20260604_000004_repair_legacy_title_hashes;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20260427_000001_init::Migration),
            Box::new(m20260604_000002_drawer_ingest::Migration),
            Box::new(m20260604_000003_embedding_idempotency::Migration),
            Box::new(m20260604_000004_repair_legacy_title_hashes::Migration),
        ]
    }
}

/// Run all pending migrations on the given connection.
///
/// Convenience wrapper around `Migrator::up` for use by `hekad` and tests.
pub async fn run_migrations(db: &DatabaseConnection) -> Result<(), DbErr> {
    Migrator::up(db, None).await
}
