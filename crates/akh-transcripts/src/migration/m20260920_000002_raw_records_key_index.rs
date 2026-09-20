//! Index `th_raw_records` on its natural key `(source, session_id, stream_id, seq)`.
//!
//! The derivation worker's anti-join (`th_raw_records` NOT EXISTS `th_events`)
//! could only merge on `(source, session_id, stream_id)` and applied
//! `e.seq = r.seq` as a post-filter — quadratic within a stream. Measured live
//! on 75k rows: 210,751,599 rows removed by the join filter, 43.6 s per pass;
//! with this index the planner merges on all four columns: 80 ms.
//!
//! `CONCURRENTLY` is not supported on hypertables, so the build takes a brief
//! lock; the table is small enough (≈1 GB/year/machine) that this is fine.

use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared(
                "CREATE INDEX IF NOT EXISTS th_raw_records_key_idx \
                 ON th_raw_records (source, session_id, stream_id, seq)",
            )
            .await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared("DROP INDEX IF EXISTS th_raw_records_key_idx")
            .await?;
        Ok(())
    }
}
