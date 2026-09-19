//! Database migrations for the `harvest-pg` transcript store (ADR 108).
//!
//! Built with the SeaORM schema builder, raw SQL reserved for what SeaORM
//! cannot express: `CREATE EXTENSION`, TimescaleDB's `create_hypertable` and
//! compression policy, and pgvector's column type / HNSW index. Table names
//! are flat and prefixed `th_`.

use sea_orm_migration::prelude::*;

mod m20260919_000001_create_raw_records;
mod m20260919_000002_create_events;
mod m20260919_000003_create_sessions;
mod m20260919_000004_create_chunks;
mod m20260919_000005_create_drift_counters;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20260919_000001_create_raw_records::Migration),
            Box::new(m20260919_000002_create_events::Migration),
            Box::new(m20260919_000003_create_sessions::Migration),
            Box::new(m20260919_000004_create_chunks::Migration),
            Box::new(m20260919_000005_create_drift_counters::Migration),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_counters_is_the_latest_migration() {
        let migrations = Migrator::migrations();
        assert_eq!(
            migrations.last().expect("at least one migration").name(),
            "m20260919_000005_create_drift_counters"
        );
    }
}
