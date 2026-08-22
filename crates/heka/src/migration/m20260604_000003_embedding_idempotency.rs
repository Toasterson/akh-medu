//! Migration 000003: enforce one canonical embedding per chunk.

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260604_000003_embedding_idempotency"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let sql = include_str!("../../migrations/20260604_000003_embedding_idempotency.sql");
        manager.get_connection().execute_unprepared(sql).await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared("DROP INDEX IF EXISTS uq_heka_embeddings_chunk")
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    const MIGRATION_SQL: &str =
        include_str!("../../migrations/20260604_000003_embedding_idempotency.sql");

    #[test]
    fn duplicates_are_repaired_before_uniqueness_is_enforced() {
        let lock = MIGRATION_SQL
            .find("LOCK TABLE heka_drawer_embeddings IN SHARE ROW EXCLUSIVE MODE")
            .unwrap();
        let dedupe = MIGRATION_SQL
            .find("DELETE FROM heka_drawer_embeddings AS duplicate")
            .unwrap();
        let unique = MIGRATION_SQL
            .find("CREATE UNIQUE INDEX IF NOT EXISTS uq_heka_embeddings_chunk")
            .unwrap();

        assert!(lock < dedupe);
        assert!(dedupe < unique);
        assert!(MIGRATION_SQL[dedupe..unique].contains("duplicate.chunk_id = canonical.chunk_id"));
        assert!(MIGRATION_SQL[dedupe..unique].contains("duplicate.id > canonical.id"));
        assert!(MIGRATION_SQL[unique..].contains("ON heka_drawer_embeddings(chunk_id)"));
    }
}
