//! Initial migration: create pgvector extension and all Heka tables.
//!
//! The DDL lives in `crates/heka/migrations/20260427_000001_init.sql`. We
//! execute it raw via `execute_unprepared` because SeaORM's `ColumnDef`
//! cannot express pgvector's `vector(384)` type or HNSW indexes.
//!
//! This mirrors the Seshat pattern (`crates/seshat/src/migration/m20260409_000001_create_seshat.rs`).

use sea_orm_migration::prelude::*;

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260427_000001_init"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let sql = include_str!("../../migrations/20260427_000001_init.sql");
        manager.get_connection().execute_unprepared(sql).await?;
        Ok(())
    }

    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        // Drop in reverse dependency order.
        db.execute_unprepared("DROP TABLE IF EXISTS heka_drawer_embeddings CASCADE")
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS heka_drawer_chunks CASCADE")
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS heka_drawers CASCADE")
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS heka_project_bindings CASCADE")
            .await?;
        db.execute_unprepared("DROP TABLE IF EXISTS heka_akhs CASCADE")
            .await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    const MIGRATION_SQL: &str = include_str!("../../migrations/20260427_000001_init.sql");

    #[test]
    fn untagged_index_uses_a_jsonb_native_empty_array_predicate() {
        let normalized = MIGRATION_SQL
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        assert!(normalized.contains(
            "CREATE INDEX IF NOT EXISTS idx_heka_drawers_untagged ON heka_drawers(akh_id) WHERE tags = '[]'::jsonb;"
        ));
        assert!(!normalized.contains("tags::text[]"));
        assert!(!normalized.contains("array_length(tags"));
    }
}
