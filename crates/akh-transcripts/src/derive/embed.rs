//! Embedding phase: fill `th_chunks.embedding` for every chunk still
//! `NULL`, batch by batch, until none remain.
//!
//! `prompt`/`reply` chunk scope and 384-dim/HNSW/cosine choices:
//! [embedding-pipeline ticket](../../../docs/wayfinder/transcript-harvest/tickets/embedding-pipeline.md).
//! Mirrors `crates/seshat/src/embed_worker.rs` — including running inference
//! on a blocking thread — with one deliberate correction: the batch query
//! here pushes `LIMIT` into SQL via [`sea_orm::QuerySelect::limit`] rather
//! than `embed_pending_batch_impl`'s `.all(db)` + `.take(batch_size)`, which
//! loads every pending chunk into memory before truncating
//! (derivation-worker ticket §1).
//!
//! Writing the vector itself needs a raw, parameterized `UPDATE` with an
//! explicit `::vector(384)` cast — SeaORM's `ActiveModel` insert sends
//! `embedding` as a text parameter, which Postgres rejects for a native
//! `vector` column. Same fix `crates/seshat/src/embed_worker.rs` and
//! `crates/seshat/src/entity/pa_embedding.rs` already apply.

use std::sync::Arc;

use chrono::Utc;
use sea_orm::{ColumnTrait, ConnectionTrait, DatabaseBackend, DatabaseConnection, Statement};
use sea_orm::{EntityTrait, QueryFilter, QueryOrder, QuerySelect};

use akh_embed::SharedEmbedder;

use crate::entity::chunk::{self, vec_to_pgvector};
use crate::error::{TranscriptError, TranscriptResult};

/// Chunks fetched and embedded per inner batch. Distinct from
/// [`super::DEFAULT_BATCH_STREAMS`] — embedding batches chunks, not streams.
pub const DEFAULT_BATCH: u64 = 64;

/// Embed every `th_chunks` row with `embedding IS NULL`, looping in batches
/// of `batch_size` until none remain, and return the total number embedded.
///
/// A chunk that fails to embed is **not** silently retried forever by this
/// function alone: chunks are derived rows carrying `derivation_version`
/// like everything else (embedding-pipeline ticket §6), so a chunk that
/// keeps failing here should be investigated the same way a rising
/// `th_drift_counters.degraded_count` would be — this function has no
/// separate poison-chunk marker of its own because chunk failure here means
/// the ONNX runtime itself is broken, which one retry loop cannot paper
/// over, and is intentionally left to surface as a hard error rather than
/// inventing a chunk-level degrade state the embedding-pipeline ticket never
/// specified.
pub async fn embed_pending(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: u64,
) -> TranscriptResult<usize> {
    let mut total = 0usize;
    loop {
        let pending = chunk::Entity::find()
            .filter(chunk::Column::EmbeddedAt.is_null())
            .order_by_asc(chunk::Column::Id)
            .limit(batch_size)
            .all(db)
            .await
            .map_err(TranscriptError::from)?;

        if pending.is_empty() {
            return Ok(total);
        }
        let batch_len = pending.len();

        let texts: Vec<String> = pending.iter().map(|c| c.text.clone()).collect();
        let embedder = Arc::clone(embedder);
        let embeddings = tokio::task::spawn_blocking(move || {
            let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let mut guard = embedder.lock().map_err(|e| TranscriptError::Db {
                source: sea_orm::DbErr::Custom(format!("embedder lock poisoned: {e}")),
            })?;
            guard
                .embed_batch(&text_refs)
                .map_err(|e| TranscriptError::Db {
                    source: sea_orm::DbErr::Custom(format!("embedding failed: {e}")),
                })
        })
        .await
        .map_err(|e| TranscriptError::Db {
            source: sea_orm::DbErr::Custom(format!("embedding spawn_blocking: {e}")),
        })??;

        let now = Utc::now();
        for (chunk, embedding) in pending.iter().zip(embeddings.iter()) {
            let stmt = Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                "UPDATE th_chunks SET embedding = $1::vector(384), embedded_at = $2 WHERE id = $3",
                [
                    vec_to_pgvector(embedding).into(),
                    now.into(),
                    chunk.id.into(),
                ],
            );
            db.execute(stmt).await.map_err(TranscriptError::from)?;
            total += 1;
        }

        // A full batch means more may be waiting; a short one means this
        // was the last.
        if batch_len < batch_size as usize {
            return Ok(total);
        }
    }
}
