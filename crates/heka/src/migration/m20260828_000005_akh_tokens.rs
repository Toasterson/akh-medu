//! Migration 000005: per-akh bearer token hashes for the MCP tenant route.

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260828_000005_akh_tokens"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let sql = include_str!("../../migrations/20260828_000005_akh_tokens.sql");
        manager.get_connection().execute_unprepared(sql).await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager
            .get_connection()
            .execute_unprepared(
                "DROP INDEX IF EXISTS uq_heka_akhs_token_hash;\n\
                 ALTER TABLE heka_akhs DROP COLUMN IF EXISTS token_hash;",
            )
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    const MIGRATION_SQL: &str = include_str!("../../migrations/20260828_000005_akh_tokens.sql");

    #[test]
    fn token_hash_column_and_lookup_index_are_idempotent_and_unique() {
        assert!(MIGRATION_SQL.contains("ADD COLUMN IF NOT EXISTS token_hash TEXT"));
        let create_index = MIGRATION_SQL
            .find("CREATE UNIQUE INDEX IF NOT EXISTS uq_heka_akhs_token_hash")
            .unwrap();
        assert!(MIGRATION_SQL[create_index..].contains("ON heka_akhs(token_hash)"));
        assert!(MIGRATION_SQL[create_index..].contains("WHERE token_hash IS NOT NULL"));
        // The column exists before the index that depends on it.
        let add_column = MIGRATION_SQL.find("ADD COLUMN IF NOT EXISTS token_hash").unwrap();
        assert!(add_column < create_index);
    }
}
