//! Database migrations for Seshat's pgvector schema.

use sea_orm_migration::prelude::*;

mod m20260409_000001_create_seshat;
mod m20260425_000001_switch_to_hnsw;
mod m20260521_000001_purge_tecton_garbage;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20260409_000001_create_seshat::Migration),
            Box::new(m20260425_000001_switch_to_hnsw::Migration),
            Box::new(m20260521_000001_purge_tecton_garbage::Migration),
        ]
    }
}
