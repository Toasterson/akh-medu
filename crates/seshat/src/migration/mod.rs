//! Database migrations for Seshat's pgvector schema.

use sea_orm_migration::prelude::*;

mod m20260409_000001_create_seshat;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![Box::new(
            m20260409_000001_create_seshat::Migration,
        )]
    }
}
