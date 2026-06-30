//! Database migrations for the akhomed store (Phase 38).
//!
//! All migrations use the SeaORM schema builder (backend-portable: Postgres now,
//! SQLite / libSQL-Turso later). Table names are flat and prefixed (no schemas,
//! which SQLite lacks). The only Postgres-specific step — the TimescaleDB
//! hypertable — is guarded and no-ops on other backends.

use sea_orm_migration::prelude::*;

mod m20260618_000001_create_akhomed;
mod m20260618_000002_create_workspaces;
mod m20260620_000001_cognitive_tables;
mod m20260620_000002_mutation_log_hypertable;
mod m20260621_000001_akh_config;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20260618_000001_create_akhomed::Migration),
            Box::new(m20260618_000002_create_workspaces::Migration),
            Box::new(m20260620_000001_cognitive_tables::Migration),
            Box::new(m20260620_000002_mutation_log_hypertable::Migration),
            Box::new(m20260621_000001_akh_config::Migration),
        ]
    }
}
