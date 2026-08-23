//! Migration 000004: repair legacy hashes written by the rejected 000002.
//!
//! The original Phase 42c migration populated pre-existing drawers with
//! `sha256(title)`. Besides depending on pgcrypto, that value could collide for
//! two distinct legacy drawers. Editing 000002 only fixes fresh installs, so
//! this follow-up also repairs databases that already recorded the rejected
//! migration.

use sea_orm::{ConnectionTrait, DatabaseBackend, Statement};
use sea_orm_migration::prelude::*;
use sha2::{Digest, Sha256};
use uuid::Uuid;

const REJECTED_MIGRATION: &str = "m20260604_000002_drawer_ingest";
const LOCK_DRAWERS_SQL: &str = "LOCK TABLE heka_drawers IN SHARE ROW EXCLUSIVE MODE";
const REPAIR_BATCH_SIZE: usize = 256;
const LEGACY_CANDIDATES_SQL: &str = "SELECT d.id, d.title, d.body_hash \
       FROM heka_drawers AS d \
       JOIN seaql_migrations AS m ON m.version = $1 \
      WHERE d.body_hash IS NOT NULL \
        AND (d.xmin = m.xmin OR d.created_at < to_timestamp(m.applied_at)) \
        AND ($2::uuid IS NULL OR d.id > $2) \
      ORDER BY d.id \
      LIMIT $3";
const REPAIR_HASH_SQL: &str = "UPDATE heka_drawers \
      SET body_hash = $1 \
    WHERE id = $2 AND body_hash = $3 \
      AND EXISTS ( \
          SELECT 1 FROM seaql_migrations AS m \
           WHERE m.version = $4 \
             AND (heka_drawers.xmin = m.xmin \
                  OR heka_drawers.created_at < to_timestamp(m.applied_at)) \
      )";

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260604_000004_repair_legacy_title_hashes"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        let db = manager.get_connection();

        // PostgreSQL migrations are run transactionally by SeaORM. This lock
        // prevents a live writer from inserting or changing a candidate
        // between selection and the conditional update below.
        db.execute_unprepared(LOCK_DRAWERS_SQL).await?;

        let mut after_id: Option<Uuid> = None;
        loop {
            let rows = db
                .query_all(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    LEGACY_CANDIDATES_SQL,
                    [
                        REJECTED_MIGRATION.into(),
                        after_id.into(),
                        (REPAIR_BATCH_SIZE as i64).into(),
                    ],
                ))
                .await?;
            if rows.is_empty() {
                break;
            }

            for row in &rows {
                let id: Uuid = row.try_get("", "id")?;
                let title: String = row.try_get("", "title")?;
                let body_hash: String = row.try_get("", "body_hash")?;
                after_id = Some(id);

                let Some(replacement) = repaired_legacy_hash(id, &title, &body_hash) else {
                    continue;
                };

                db.execute(Statement::from_sql_and_values(
                    DatabaseBackend::Postgres,
                    REPAIR_HASH_SQL,
                    [
                        replacement.into(),
                        id.into(),
                        body_hash.into(),
                        REJECTED_MIGRATION.into(),
                    ],
                ))
                .await?;
            }

            if rows.len() < REPAIR_BATCH_SIZE {
                break;
            }
        }

        Ok(())
    }

    async fn down(&self, _manager: &SchemaManager) -> Result<(), DbErr> {
        // Data repair is intentionally irreversible. Reconstructing the bad
        // title hash would reintroduce collisions. SeaORM will remove this
        // migration's ledger row, and a later `up` safely rechecks the data.
        Ok(())
    }
}

fn repaired_legacy_hash(id: Uuid, title: &str, body_hash: &str) -> Option<String> {
    (body_hash == title_hash(title)).then(|| format!("legacy:{id}"))
}

fn title_hash(title: &str) -> String {
    let digest = Sha256::digest(title.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        let _ = write!(hex, "{byte:02x}");
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_rejected_title_hash_is_repaired_without_pgcrypto() {
        let id = Uuid::parse_str("018f55b2-1df7-7b0a-b9ca-2a880942782b").unwrap();
        let rejected = title_hash("same legacy title");

        assert_eq!(
            repaired_legacy_hash(id, "same legacy title", &rejected),
            Some(format!("legacy:{id}"))
        );
        assert!(!LEGACY_CANDIDATES_SQL.contains("digest("));
        assert!(!REPAIR_HASH_SQL.contains("digest("));
    }

    #[test]
    fn repair_is_idempotent_and_does_not_touch_other_hashes() {
        let id = Uuid::parse_str("018f55b2-1df7-7b0a-b9ca-2a880942782b").unwrap();
        let repaired = format!("legacy:{id}");

        assert_eq!(repaired_legacy_hash(id, "title", &repaired), None);
        assert_eq!(repaired_legacy_hash(id, "title", &title_hash("body")), None);
        assert!(REPAIR_HASH_SQL.contains("id = $2 AND body_hash = $3"));
    }

    #[test]
    fn candidate_scope_matches_the_backfill_transaction_or_older_rows() {
        assert!(LEGACY_CANDIDATES_SQL.contains("d.xmin = m.xmin"));
        assert!(LEGACY_CANDIDATES_SQL.contains("d.created_at < to_timestamp(m.applied_at)"));
        assert!(LEGACY_CANDIDATES_SQL.contains("ORDER BY d.id"));
        assert!(LEGACY_CANDIDATES_SQL.contains("LIMIT $3"));
        assert!(REPAIR_HASH_SQL.contains("heka_drawers.xmin = m.xmin"));
        assert_eq!(
            LOCK_DRAWERS_SQL,
            "LOCK TABLE heka_drawers IN SHARE ROW EXCLUSIVE MODE"
        );
    }
}
