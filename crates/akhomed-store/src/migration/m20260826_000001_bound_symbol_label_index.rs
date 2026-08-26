//! Replace the unbounded symbol-label B-tree with a fixed-size hash index.
//!
//! PostgreSQL B-tree index entries are limited to roughly one third of a page.
//! Symbol labels are arbitrary knowledge text, so the original `(akh_id, label)`
//! index made an otherwise valid snapshot fail once a label exceeded that limit.
//! A hash index stores a fixed-size hash and still serves the only useful access
//! pattern for a full label: exact equality. The table primary key already
//! provides the `akh_id` index prefix.
//!
//! This is Postgres-only. SQLite does not have PostgreSQL's B-tree tuple limit,
//! so its existing portable index remains valid.

use sea_orm::{ConnectionTrait, DbBackend};
use sea_orm_migration::prelude::*;

const DROP_LABEL_INDEX: &str = "DROP INDEX IF EXISTS idx_akhomed_symbols_label";
const CREATE_HASH_LABEL_INDEX: &str =
    "CREATE INDEX idx_akhomed_symbols_label ON akhomed_symbols USING hash (label)";

#[derive(DeriveMigrationName)]
pub struct Migration;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        if manager.get_database_backend() != DbBackend::Postgres {
            return Ok(());
        }

        let db = manager.get_connection();
        db.execute_unprepared(DROP_LABEL_INDEX).await?;
        db.execute_unprepared(CREATE_HASH_LABEL_INDEX).await?;
        tracing::info!("replaced unbounded akhomed symbol-label B-tree with a hash index");
        Ok(())
    }

    async fn down(&self, _manager: &SchemaManager) -> Result<(), DbErr> {
        // Deliberately retain the safe index. Recreating the prior B-tree can
        // fail immediately when the table already contains a long label; the
        // hash index is compatible with the older application's equality reads.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacement_index_never_embeds_unbounded_label_in_a_btree() {
        assert_eq!(
            DROP_LABEL_INDEX,
            "DROP INDEX IF EXISTS idx_akhomed_symbols_label"
        );
        assert!(CREATE_HASH_LABEL_INDEX.contains("USING hash (label)"));
        assert!(!CREATE_HASH_LABEL_INDEX.contains("akh_id, label"));
    }
}
