//! Database migrations for the akhomed Postgres backing store (Phase 38a).
//!
//! Migrations run in chronological order. `akhomed` is created before
//! `workspaces` because `workspaces.membership` and
//! `workspaces.active_microtheories` reference `akhomed.akh`.
//!
//! AGE (the read-side property-graph projection) is intentionally **not** set
//! up here — that ships in sub-phase 38d behind an availability check.

use sea_orm_migration::prelude::*;

mod m20260618_000001_create_akhomed;
mod m20260618_000002_create_workspaces;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20260618_000001_create_akhomed::Migration),
            Box::new(m20260618_000002_create_workspaces::Migration),
        ]
    }
}
