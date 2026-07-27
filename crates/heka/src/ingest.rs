//! Drawer ingestion: content-hash idempotent write + paragraph-aware chunking.
//!
//! `Ingest::write_drawer` is the verbatim drawer write path for Claude Code
//! session turns. It:
//!
//! 1. Computes a SHA-256 hash of the turn body.
//! 2. Inserts a `heka_drawers` row with `INSERT ... ON CONFLICT (akh_id,
//!    body_hash) DO NOTHING` — re-capturing the same session is a no-op.
//! 3. Chunks the body using the paragraph-aware chunker (target 1500 chars,
//!    MiniLM 512-token-friendly).
//! 4. Inserts `heka_drawer_chunks` rows with `embedded_at = NULL`, enqueuing
//!    them for the background `embed_worker`.
//!
//! All writes happen inside a single transaction so a chunk-insert failure
//! rolls back the drawer row.

use sea_orm::{
    ActiveModelTrait, ColumnTrait, ConnectionTrait, DatabaseBackend, DatabaseConnection,
    EntityTrait, QueryFilter, Set, Statement, TransactionTrait,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::chunker::{chunk_paragraph_aware, DEFAULT_TARGET_CHUNK_CHARS};
use crate::entity::{heka_drawer_chunks, heka_drawers};
use crate::error::{HekaError, HekaResult};

/// Summary returned from a successful `write_drawer` call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawerIngestSummary {
    /// The drawer UUID (existing if duplicated, new if freshly inserted).
    pub drawer_id: Uuid,
    /// Number of chunks created (0 if the drawer was a duplicate).
    pub chunk_count: usize,
    /// Total word count across all chunks.
    pub word_count: usize,
    /// True if the drawer already existed (content-hash duplicate).
    pub duplicated: bool,
    /// The SHA-256 hex digest of the body.
    pub body_hash: String,
}

/// Ingest a single Claude Code session turn as a drawer.
///
/// Content-hash idempotent: if a drawer with the same `(akh_id, body_hash)`
/// already exists, the call is a no-op and returns `duplicated: true` with
/// the existing drawer's ID and `chunk_count: 0`.
///
/// # Arguments
///
/// * `db` — SeaORM database connection.
/// * `akh_id` — The akh this drawer belongs to (FK to `heka_akhs.id`).
/// * `session_id` — Claude Code session identifier (from stop-hook payload).
/// * `speaker` — Who produced this turn: "user", "assistant", or "system".
/// * `body` — The full text content of the turn.
/// * `contact` — Freeform contact label (e.g. "claude-code@laptop").
/// * `topics` — Topic tags extracted from the turn (stored as JSONB array).
pub async fn write_drawer(
    db: &DatabaseConnection,
    akh_id: Uuid,
    session_id: Option<&str>,
    speaker: Option<&str>,
    body: &str,
    contact: Option<&str>,
    topics: &[String],
) -> HekaResult<DrawerIngestSummary> {
    if body.trim().is_empty() {
        return Err(HekaError::Config(
            "write_drawer: body must not be empty".to_string(),
        ));
    }

    // SHA-256 content hash for idempotency.
    let body_hash = hex_hash(body);

    // Check for an existing drawer with the same (akh_id, body_hash).
    // The UNIQUE partial index `uq_heka_drawers_akh_body_hash` enforces this
    // at the DB level, but we check first so we can return the existing ID
    // rather than silently swallowing the conflict.
    if let Some(existing) = heka_drawers::Entity::find()
        .filter(heka_drawers::Column::AkhId.eq(akh_id))
        .filter(heka_drawers::Column::BodyHash.eq(&body_hash))
        .one(db)
        .await?
    {
        return Ok(DrawerIngestSummary {
            drawer_id: existing.id,
            chunk_count: 0,
            word_count: 0,
            duplicated: true,
            body_hash,
        });
    }

    // Chunk the body — paragraph-aware, target 1500 chars.
    let chunks = chunk_paragraph_aware(body, DEFAULT_TARGET_CHUNK_CHARS, "markdown");
    if chunks.is_empty() {
        return Err(HekaError::Config(
            "write_drawer: chunker produced zero chunks — is body non-empty whitespace only?"
                .to_string(),
        ));
    }

    let total_words: usize = chunks.iter().map(|c| c.word_count as usize).sum();
    let chunk_count = chunks.len();
    let drawer_id = Uuid::new_v4();
    let now = chrono::Utc::now();

    let topics_json = serde_json::to_value(topics)?;
    let title = derive_title(body, speaker);

    // Transaction so a chunk-insert failure rolls back the drawer row.
    let txn = db.begin().await?;

    // Insert the drawer row. We use raw SQL with ON CONFLICT DO NOTHING
    // as a belt-and-suspenders idempotency guard — the pre-check above
    // handles the common case, but a concurrent insert between the check
    // and the insert would otherwise error.
    let stmt = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "INSERT INTO heka_drawers \
         (id, akh_id, title, tags, metadata, body_hash, session_id, speaker, contact, topics, \
          created_at, updated_at) \
         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) \
         ON CONFLICT (akh_id, body_hash) WHERE body_hash IS NOT NULL DO NOTHING \
         RETURNING id",
        [
            drawer_id.into(),
            akh_id.into(),
            title.into(),
            serde_json::json!([]).into(), // tags
            serde_json::json!({}).into(),  // metadata
            body_hash.clone().into(),
            session_id.map(|s| s.to_string()).into(),
            speaker.map(|s| s.to_string()).into(),
            contact.map(|s| s.to_string()).into(),
            topics_json.into(),
            now.into(),
            now.into(),
        ],
    );
    let result = txn.query_one(stmt).await?;

    // If ON CONFLICT fired (concurrent insert), result is non-null with
    // the existing ID — but since we did DO NOTHING without RETURNING on
    // conflict, the row is None. Re-query to get the actual ID.
    let actual_drawer_id = match result {
        Some(row) => row.try_get_by_index::<Uuid>(0).map_err(|e| {
            HekaError::Database(sea_orm::DbErr::RecordNotFound(format!(
                "failed to read drawer id from RETURNING: {e}"
            )))
        })?,
        None => {
            // Concurrent insert won the race — fetch the existing row.
            let existing = heka_drawers::Entity::find()
                .filter(heka_drawers::Column::AkhId.eq(akh_id))
                .filter(heka_drawers::Column::BodyHash.eq(&body_hash))
                .one(&txn)
                .await?
                .ok_or_else(|| {
                    HekaError::Database(sea_orm::DbErr::RecordNotFound(
                        "ON CONFLICT DO NOTHING but no existing row found".to_string(),
                    ))
                })?;
            txn.commit().await?;
            return Ok(DrawerIngestSummary {
                drawer_id: existing.id,
                chunk_count: 0,
                word_count: 0,
                duplicated: true,
                body_hash,
            });
        }
    };

    // Insert chunk rows with embedded_at = NULL (enqueued for embed_worker).
    let chunk_models: Vec<heka_drawer_chunks::ActiveModel> = chunks
        .into_iter()
        .enumerate()
        .map(|(i, c)| heka_drawer_chunks::ActiveModel {
            id: Set(Uuid::new_v4()),
            drawer_id: Set(actual_drawer_id),
            chunk_index: Set(i as i32),
            text: Set(c.text),
            word_count: Set(c.word_count),
            embedded_at: Set(None),
            created_at: Set(now.into()),
        })
        .collect();

    heka_drawer_chunks::Entity::insert_many(chunk_models)
        .exec(&txn)
        .await?;

    txn.commit().await?;

    Ok(DrawerIngestSummary {
        drawer_id: actual_drawer_id,
        chunk_count,
        word_count: total_words,
        duplicated: false,
        body_hash,
    })
}

/// Compute the SHA-256 hex digest of a string.
fn hex_hash(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    let bytes = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(hex, "{b:02x}");
    }
    hex
}

/// Derive a short title from the body and speaker.
///
/// Takes the first non-empty line, truncated to 120 chars. Falls back
/// to a speaker + timestamp placeholder if the body has no content lines.
fn derive_title(body: &str, speaker: Option<&str>) -> String {
    let first_line = body
        .lines()
        .map(|l| l.trim())
        .find(|l| !l.is_empty())
        .unwrap_or("");

    let title = if first_line.is_empty() {
        format!(
            "{} turn",
            speaker.unwrap_or("unknown")
        )
    } else {
        let chars: Vec<char> = first_line.chars().collect();
        if chars.len() > 120 {
            chars[..120].iter().collect::<String>() + "…"
        } else {
            first_line.to_string()
        }
    };

    title
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_hash_is_stable() {
        let h1 = hex_hash("hello world");
        let h2 = hex_hash("hello world");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
        assert_ne!(hex_hash("hello world"), hex_hash("hello world!"));
    }

    #[test]
    fn derive_title_truncates_long_lines() {
        let long = "a".repeat(200);
        let title = derive_title(&long, Some("user"));
        assert!(title.ends_with('…'));
        assert!(title.chars().count() <= 121); // 120 + ellipsis
    }

    #[test]
    fn derive_title_uses_speaker_fallback() {
        let title = derive_title("   \n  \n  ", Some("assistant"));
        assert_eq!(title, "assistant turn");
    }
}