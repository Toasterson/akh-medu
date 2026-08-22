//! Migration 000002: drawer ingest path — add content-hash idempotency + session metadata.
//!
//! The DDL lives in `crates/heka/migrations/20260604_000002_drawer_ingest.sql`.
//! We execute it raw via `execute_unprepared` because the migration includes
//! a DO block for backfilling `body_hash` that SeaORM's `ColumnDef` cannot express.

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260604_000002_drawer_ingest"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let sql = include_str!("../../migrations/20260604_000002_drawer_ingest.sql");
        manager.get_connection().execute_unprepared(sql).await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        db.execute_unprepared("DROP INDEX IF EXISTS idx_heka_drawers_speaker")
            .await?;
        db.execute_unprepared("DROP INDEX IF EXISTS idx_heka_drawers_session")
            .await?;
        db.execute_unprepared("DROP INDEX IF EXISTS uq_heka_drawers_akh_body_hash")
            .await?;

        db.execute_unprepared(
            "ALTER TABLE heka_drawers \
             DROP COLUMN IF EXISTS topics, \
             DROP COLUMN IF EXISTS contact, \
             DROP COLUMN IF EXISTS speaker, \
             DROP COLUMN IF EXISTS session_id, \
             DROP COLUMN IF EXISTS body_hash",
        )
        .await?;

        db.execute_unprepared(
            "ALTER TABLE heka_drawer_chunks \
             DROP COLUMN IF EXISTS heading",
        )
        .await?;

        Ok(())
    }
}
