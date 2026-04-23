//! Async embed worker that polls for unembedded chunks and generates embeddings.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use sea_orm::{
    ActiveModelTrait, ColumnTrait, ConnectionTrait, DatabaseBackend, DatabaseConnection,
    EntityTrait, PaginatorTrait, QueryFilter, QueryOrder, Set, Statement,
};
use uuid::Uuid;

use crate::config::EmbeddingConfig;
use crate::embedder::SharedEmbedder;
use crate::entity::pa_chunk::{self, Column as ChunkCol, Entity as Chunk};
use crate::entity::pa_embedding::vec_to_pgvector;
use crate::error::SeshatResult;

/// Statistics from an embed worker run.
#[derive(Debug, Default)]
pub struct EmbedStats {
    pub chunks_processed: usize,
    pub embeddings_created: usize,
}

/// Run the embed worker loop. This function runs forever, polling for
/// unembedded chunks at the configured interval.
pub async fn run_embed_worker(
    db: Arc<DatabaseConnection>,
    embedder: SharedEmbedder,
    config: EmbeddingConfig,
) {
    let interval = Duration::from_secs(config.poll_interval_secs);
    let batch_size = config.batch_size;
    let model_name = config.model.clone();

    tracing::info!(
        batch_size,
        poll_secs = config.poll_interval_secs,
        "embed worker started"
    );

    loop {
        match embed_pending_batch(&db, &embedder, batch_size, &model_name).await {
            Ok(stats) => {
                if stats.chunks_processed > 0 {
                    tracing::info!(
                        chunks = stats.chunks_processed,
                        embeddings = stats.embeddings_created,
                        "embed batch complete"
                    );
                    // If we processed a full batch, immediately check for more.
                    if stats.chunks_processed >= batch_size {
                        continue;
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "embed worker error");
            }
        }

        tokio::time::sleep(interval).await;
    }
}

/// Process one batch of unembedded chunks.
pub async fn embed_pending_batch(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
) -> SeshatResult<EmbedStats> {
    // Find unembedded chunks.
    let pending_chunks: Vec<pa_chunk::Model> = Chunk::find()
        .filter(ChunkCol::EmbeddedAt.is_null())
        .order_by_asc(ChunkCol::Id)
        .all(db)
        .await?
        .into_iter()
        .take(batch_size)
        .collect();

    if pending_chunks.is_empty() {
        return Ok(EmbedStats::default());
    }

    let chunk_count = pending_chunks.len();

    // Collect texts for batch embedding.
    let texts: Vec<&str> = pending_chunks.iter().map(|c| c.text.as_str()).collect();

    // Run embedding (blocking ONNX inference in a spawn_blocking task).
    let embedder_clone = Arc::clone(embedder);
    let texts_owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
    let embeddings = tokio::task::spawn_blocking(move || {
        let text_refs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
        let mut guard = embedder_clone
            .lock()
            .map_err(|e| crate::SeshatError::Embedding(format!("embedder lock: {e}")))?;
        guard.embed_batch(&text_refs)
    })
    .await
    .map_err(|e| crate::SeshatError::Embedding(format!("spawn_blocking: {e}")))??;

    let now = Utc::now().fixed_offset();
    let mut embeddings_created = 0;

    // Insert embeddings and mark chunks as embedded.
    //
    // SeaORM's ActiveModel insert sends `embedding` as a text parameter,
    // which PostgreSQL rejects with `column "embedding" is of type vector
    // but expression is of type text` — pgvector's `vector(384)` type
    // needs an explicit `::vector(N)` cast on insert. Parameterized raw
    // SQL with the cast is the simplest fix; the SELECT side in
    // `query.rs` already does the same thing.
    for (chunk, embedding) in pending_chunks.iter().zip(embeddings.iter()) {
        let stmt = Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "INSERT INTO pa_embeddings (id, chunk_id, model_name, embedding, created_at) \
             VALUES ($1, $2, $3, $4::vector(384), $5)",
            [
                Uuid::new_v4().into(),
                chunk.id.into(),
                model_name.to_string().into(),
                vec_to_pgvector(embedding).into(),
                sea_orm::Value::ChronoDateTimeUtc(Some(Box::new(now.to_utc()))),
            ],
        );
        db.execute(stmt).await?;
        embeddings_created += 1;

        // Mark chunk as embedded. ActiveModel is fine here — `embedded_at`
        // is a plain timestamptz with no custom cast needed.
        let mut chunk_active: pa_chunk::ActiveModel = chunk.clone().into();
        chunk_active.embedded_at = Set(Some(now.into()));
        chunk_active.update(db).await?;
    }

    Ok(EmbedStats {
        chunks_processed: chunk_count,
        embeddings_created,
    })
}

/// Count chunks waiting to be embedded.
pub async fn count_pending(db: &DatabaseConnection) -> SeshatResult<u64> {
    let count = Chunk::find()
        .filter(ChunkCol::EmbeddedAt.is_null())
        .count(db)
        .await?;
    Ok(count)
}
