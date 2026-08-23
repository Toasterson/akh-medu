//! Migration 000004: register the fail-closed legacy-hash repair boundary.
//!
//! The original Phase 42c migration populated pre-existing drawers with
//! `sha256(title)`. Besides depending on pgcrypto, that value could collide for
//! two distinct legacy drawers. There is no durable automatic discriminator
//! between such a row and a legitimate post-migration one-line drawer whose
//! title equals its body. Migration 000004 therefore performs no heuristic data
//! rewrite. Operators repair only positively identified drawer UUIDs through
//! `hekad repair-legacy-title-hashes --drawer-id <UUID>`; that command verifies
//! the rejected legacy shape, updates all requested rows atomically, and writes
//! a durable audit marker into drawer metadata.

use std::collections::HashSet;

use sea_orm::{ConnectionTrait, DatabaseBackend, Statement, TransactionTrait};
use sea_orm_migration::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{HekaError, HekaResult};

const MAX_EXPLICIT_REPAIRS: usize = 256;
const AUDIT_KEY: &str = "heka_phase42c_legacy_title_hash_repair";
const REPAIRED_MARKER: &str = "repaired:m20260604_000002_drawer_ingest";
const SELECT_EXPLICIT_DRAWER_SQL: &str = "SELECT title, body_hash, session_id, speaker, contact, topics, metadata \
       FROM heka_drawers \
      WHERE id = $1 \
      FOR UPDATE";
const REPAIR_EXPLICIT_DRAWER_SQL: &str = "UPDATE heka_drawers \
      SET body_hash = $1, \
          metadata = jsonb_set(metadata, '{heka_phase42c_legacy_title_hash_repair}', to_jsonb($2::text), true), \
          updated_at = clock_timestamp() \
    WHERE id = $3 \
      AND body_hash = $4 \
      AND session_id IS NULL \
      AND speaker IS NULL \
      AND contact IS NULL \
      AND topics = '[]'::jsonb";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LegacyTitleHashRepairStats {
    pub repaired: usize,
    pub already_repaired: usize,
}

struct ExplicitRepairRow {
    id: Uuid,
    rejected_hash: String,
}

enum RepairDisposition {
    Repair(ExplicitRepairRow),
    AlreadyRepaired,
}

pub struct Migration;

impl MigrationName for Migration {
    fn name(&self) -> &str {
        "m20260604_000004_repair_legacy_title_hashes"
    }
}

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, _manager: &SchemaManager) -> Result<(), DbErr> {
        // Deliberately no automatic data rewrite. See the module-level
        // operator contract and repair_rejected_legacy_title_hashes().
        Ok(())
    }

    async fn down(&self, _manager: &SchemaManager) -> Result<(), DbErr> {
        // This ledger migration has no schema or automatic data side effects.
        // Explicit operator repairs are intentionally irreversible.
        Ok(())
    }
}

/// Atomically repair an explicit, bounded set of drawer UUIDs positively
/// identified by an operator as products of the rejected 000002 backfill.
pub async fn repair_rejected_legacy_title_hashes(
    db: &sea_orm::DatabaseConnection,
    drawer_ids: &[Uuid],
) -> HekaResult<LegacyTitleHashRepairStats> {
    validate_explicit_ids(drawer_ids)?;
    // A stable lock order prevents two concurrent, overlapping explicit
    // repair requests from deadlocking when their callers supplied IDs in
    // opposite orders.
    let mut ordered_ids = drawer_ids.to_vec();
    ordered_ids.sort_unstable();
    let transaction = db.begin().await?;
    let result = repair_in_transaction(&transaction, &ordered_ids).await;
    match result {
        Ok(stats) => {
            transaction.commit().await?;
            Ok(stats)
        }
        Err(error) => {
            transaction.rollback().await?;
            Err(error)
        }
    }
}

fn validate_explicit_ids(drawer_ids: &[Uuid]) -> HekaResult<()> {
    if drawer_ids.is_empty() {
        return Err(HekaError::Config(
            "legacy title-hash repair requires at least one explicit drawer UUID".to_string(),
        ));
    }
    if drawer_ids.len() > MAX_EXPLICIT_REPAIRS {
        return Err(HekaError::Config(format!(
            "legacy title-hash repair accepts at most {MAX_EXPLICIT_REPAIRS} drawer UUIDs per atomic request"
        )));
    }
    let mut unique = HashSet::with_capacity(drawer_ids.len());
    if drawer_ids.iter().any(|id| !unique.insert(*id)) {
        return Err(HekaError::Config(
            "legacy title-hash repair drawer UUIDs must be unique".to_string(),
        ));
    }
    Ok(())
}

async fn repair_in_transaction(
    transaction: &sea_orm::DatabaseTransaction,
    drawer_ids: &[Uuid],
) -> HekaResult<LegacyTitleHashRepairStats> {
    let mut pending = Vec::with_capacity(drawer_ids.len());
    let mut already_repaired = 0usize;
    for id in drawer_ids {
        let row = transaction
            .query_one(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                SELECT_EXPLICIT_DRAWER_SQL,
                [(*id).into()],
            ))
            .await?
            .ok_or_else(|| HekaError::DrawerNotFound {
                drawer_id: id.to_string(),
            })?;
        let title: String = row.try_get("", "title")?;
        let body_hash: Option<String> = row.try_get("", "body_hash")?;
        let session_id: Option<String> = row.try_get("", "session_id")?;
        let speaker: Option<String> = row.try_get("", "speaker")?;
        let contact: Option<String> = row.try_get("", "contact")?;
        let topics: Value = row.try_get("", "topics")?;
        let metadata: Value = row.try_get("", "metadata")?;

        match classify_explicit_repair(
            *id,
            &title,
            body_hash.as_deref(),
            session_id.as_deref(),
            speaker.as_deref(),
            contact.as_deref(),
            &topics,
            &metadata,
        )? {
            RepairDisposition::Repair(row) => pending.push(row),
            RepairDisposition::AlreadyRepaired => already_repaired += 1,
        }
    }

    for row in &pending {
        let replacement = format!("legacy:{}", row.id);
        let updated = transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                REPAIR_EXPLICIT_DRAWER_SQL,
                [
                    replacement.into(),
                    REPAIRED_MARKER.into(),
                    row.id.into(),
                    row.rejected_hash.clone().into(),
                ],
            ))
            .await?;
        if updated.rows_affected() != 1 {
            return Err(HekaError::Config(format!(
                "drawer {} changed after legacy repair validation; no repairs were committed",
                row.id
            )));
        }
    }

    Ok(LegacyTitleHashRepairStats {
        repaired: pending.len(),
        already_repaired,
    })
}

#[allow(clippy::too_many_arguments)]
fn classify_explicit_repair(
    id: Uuid,
    title: &str,
    body_hash: Option<&str>,
    session_id: Option<&str>,
    speaker: Option<&str>,
    contact: Option<&str>,
    topics: &Value,
    metadata: &Value,
) -> HekaResult<RepairDisposition> {
    let expected_repaired_hash = format!("legacy:{id}");
    let marker_value = metadata.get(AUDIT_KEY);
    let marker = marker_value.and_then(Value::as_str);
    if body_hash == Some(expected_repaired_hash.as_str()) && marker == Some(REPAIRED_MARKER) {
        return Ok(RepairDisposition::AlreadyRepaired);
    }
    if marker_value.is_some() {
        return Err(invalid_explicit_candidate(
            id,
            "reserved repair audit marker has an unexpected state",
        ));
    }
    let rejected_hash = title_hash(title);
    if body_hash != Some(rejected_hash.as_str()) {
        return Err(invalid_explicit_candidate(
            id,
            "body_hash is not the rejected sha256(title) value",
        ));
    }
    if session_id.is_some()
        || speaker.is_some()
        || contact.is_some()
        || topics != &serde_json::json!([])
    {
        return Err(invalid_explicit_candidate(
            id,
            "drawer contains post-000002 ingest metadata",
        ));
    }
    if !metadata.is_object() {
        return Err(invalid_explicit_candidate(
            id,
            "metadata is not an object and cannot hold the audit marker",
        ));
    }
    Ok(RepairDisposition::Repair(ExplicitRepairRow {
        id,
        rejected_hash,
    }))
}

fn invalid_explicit_candidate(id: Uuid, reason: &str) -> HekaError {
    HekaError::Config(format!(
        "drawer {id} is not a verified rejected-000002 legacy candidate: {reason}"
    ))
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
    fn exact_rejected_title_hash_requires_an_explicit_bounded_request() {
        let id = Uuid::parse_str("018f55b2-1df7-7b0a-b9ca-2a880942782b").unwrap();
        let rejected = title_hash("same legacy title");

        assert!(matches!(
            classify_explicit_repair(
                id,
                "same legacy title",
                Some(&rejected),
                None,
                None,
                None,
                &serde_json::json!([]),
                &serde_json::json!({}),
            )
            .unwrap(),
            RepairDisposition::Repair(_)
        ));
        assert!(validate_explicit_ids(&[]).is_err());
        assert!(validate_explicit_ids(&[id, id]).is_err());
        assert!(validate_explicit_ids(&vec![id; MAX_EXPLICIT_REPAIRS + 1]).is_err());
        assert!(!REPAIR_EXPLICIT_DRAWER_SQL.contains("digest("));
        assert!(!REPAIR_EXPLICIT_DRAWER_SQL.contains("to_timestamp"));
        assert!(!REPAIR_EXPLICIT_DRAWER_SQL.contains("xmin"));
    }

    #[test]
    fn completed_audit_marker_makes_explicit_repair_idempotent() {
        let id = Uuid::parse_str("018f55b2-1df7-7b0a-b9ca-2a880942782b").unwrap();
        let metadata = serde_json::json!({
            "heka_phase42c_legacy_title_hash_repair": REPAIRED_MARKER
        });
        assert!(matches!(
            classify_explicit_repair(
                id,
                "title",
                Some(&format!("legacy:{id}")),
                None,
                None,
                None,
                &serde_json::json!([]),
                &metadata,
            )
            .unwrap(),
            RepairDisposition::AlreadyRepaired
        ));
        assert!(REPAIR_EXPLICIT_DRAWER_SQL.contains("jsonb_set(metadata"));
        assert!(REPAIR_EXPLICIT_DRAWER_SQL.contains("id = $3"));
        assert!(REPAIR_EXPLICIT_DRAWER_SQL.contains("body_hash = $4"));
    }

    #[test]
    fn ambiguous_or_post_ingest_rows_fail_closed_even_when_explicitly_named() {
        let id = Uuid::parse_str("018f55b2-1df7-7b0a-b9ca-2a880942782b").unwrap();
        let rejected = title_hash("title");
        assert!(
            classify_explicit_repair(
                id,
                "title",
                Some(&rejected),
                Some("new-session"),
                None,
                None,
                &serde_json::json!([]),
                &serde_json::json!({}),
            )
            .is_err()
        );
        assert!(
            classify_explicit_repair(
                id,
                "title",
                Some(&title_hash("different body")),
                None,
                None,
                None,
                &serde_json::json!([]),
                &serde_json::json!({}),
            )
            .is_err()
        );
        assert!(
            classify_explicit_repair(
                id,
                "title",
                Some(&rejected),
                None,
                None,
                None,
                &serde_json::json!([]),
                &serde_json::json!({
                    "heka_phase42c_legacy_title_hash_repair": null
                }),
            )
            .is_err()
        );
    }
}
