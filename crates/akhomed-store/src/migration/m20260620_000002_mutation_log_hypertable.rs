//! Migration: convert `akhomed_mutation_log` into a TimescaleDB hypertable.
//!
//! **Postgres-only and graceful.** On non-Postgres backends (SQLite/libSQL-Turso)
//! it is a no-op — the WAL stays a plain SeaORM table. On Postgres it runs only
//! if the `timescaledb` extension is available (baked into the CNPG image);
//! otherwise it logs and leaves the plain table. This is the single place raw
//! SQL is used, guarded by a backend + extension check.

use sea_orm::{ConnectionTrait, DbBackend, Statement};
use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        if manager.get_database_backend() != DbBackend::Postgres {
            return Ok(()); // portable backends: WAL stays a plain table
        }
        let db = manager.get_connection();

        let available = db
            .query_one(Statement::from_string(
                DbBackend::Postgres,
                "SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'".to_owned(),
            ))
            .await?
            .is_some();
        if !available {
            tracing::warn!(
                "timescaledb extension not available; akhomed_mutation_log stays a plain table"
            );
            return Ok(());
        }

        db.execute_unprepared("CREATE EXTENSION IF NOT EXISTS timescaledb").await?;
        // TimescaleDB requires the partition column in every unique index, so the
        // primary key must include applied_at.
        db.execute_unprepared(
            "ALTER TABLE akhomed_mutation_log DROP CONSTRAINT IF EXISTS akhomed_mutation_log_pkey",
        )
        .await?;
        db.execute_unprepared(
            "ALTER TABLE akhomed_mutation_log ADD PRIMARY KEY (applied_at, seq)",
        )
        .await?;
        db.execute_unprepared(
            "SELECT create_hypertable('akhomed_mutation_log', 'applied_at', migrate_data => true)",
        )
        .await?;
        tracing::info!("akhomed_mutation_log converted to a TimescaleDB hypertable");
        Ok(())
    }

    async fn down(&self, _manager: &SchemaManager) -> Result<(), DbErr> {
        // No-op: dropping the table (prior migration's down) removes the hypertable too.
        Ok(())
    }
}
