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
///
/// `metrics` (`tracing-otel` builds only) is the WI-326 emission handle;
/// `None` makes the per-tick emission a no-op.
pub async fn run_embed_worker(
    db: Arc<DatabaseConnection>,
    embedder: SharedEmbedder,
    config: EmbeddingConfig,
    #[cfg(feature = "tracing-otel")] metrics: Option<std::sync::Arc<crate::metrics::SeshatMetrics>>,
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
        // One tick of the worker: spans the batch (including the blocking
        // embed call, which re-enters this span's context across the
        // `spawn_blocking` hop).
        let tick_span = tracing::info_span!("seshat.embed_worker.tick");
        let tick_result = {
            let _tick = tick_span.entered();
            embed_pending_batch(
                &db,
                &embedder,
                batch_size,
                &model_name,
                #[cfg(feature = "tracing-otel")]
                metrics.as_deref(),
            )
            .await
        };
        match tick_result {
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
///
/// WI-326: records `seshat_embed_batch_duration_seconds` / batch / chunks
/// counters per invocation, and captures `tracing::Span::current()` before
/// `spawn_blocking`, re-entering it inside the blocking closure so the
/// ONNX-inference span context stays stitched to the caller (worker tick or
/// `seshat_embed_now`) trace.
/// WI-326 trace-stitching and `metrics` handling are described at the call
/// site: the closure captures `tracing::Span::current()` before
/// `spawn_blocking` and re-enters it inside the blocking work; `metrics`
/// (`tracing-otel` builds only) receives one `record_embed_batch` per
/// invocation, on both empty and populated ticks.
pub async fn embed_pending_batch(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
    #[cfg(feature = "tracing-otel")] metrics: Option<&crate::metrics::SeshatMetrics>,
) -> SeshatResult<EmbedStats> {
    #[cfg(feature = "tracing-otel")]
    let batch_start = std::time::Instant::now();

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
        // Empty tick still counts as a batch iteration (0 chunks processed).
        #[cfg(feature = "tracing-otel")]
        if let Some(m) = metrics {
            m.record_embed_batch(batch_start.elapsed().as_secs_f64(), 0);
        }
        return Ok(EmbedStats::default());
    }

    let chunk_count = pending_chunks.len();

    // Collect texts for batch embedding.
    let texts: Vec<&str> = pending_chunks.iter().map(|c| c.text.as_str()).collect();

    // Run embedding (blocking ONNX inference in a spawn_blocking task).
    //
    // WI-326 trace-stitching: capture the *current* span (the worker-tick or
    // tool-call span) and re-enter it inside the blocking closure. A plain
    // `spawn_blocking` runs on a separate blocking thread that is NOT running
    // the async future's span context, so without the explicit capture +
    // re-enter the ONNX call would be detached from the tick trace. The guard
    // is a `EnteredRef`-style borrow (Send-safe here because the blocking
    // closure never `.await`s).
    let embedder_clone = Arc::clone(embedder);
    let texts_owned: Vec<String> = texts.iter().map(|t| t.to_string()).collect();
    let blocking_span = tracing::Span::current();
    let embeddings = tokio::task::spawn_blocking(move || {
        let _entered = blocking_span.entered();
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

    let stats = EmbedStats {
        chunks_processed: chunk_count,
        embeddings_created,
    };

    // WI-326: record the full-tick batch duration + chunks processed on the
    // success leg.
    #[cfg(feature = "tracing-otel")]
    if let Some(m) = metrics {
        m.record_embed_batch(batch_start.elapsed().as_secs_f64(), stats.chunks_processed as u64);
    }

    Ok(stats)
}

/// Count chunks waiting to be embedded.
pub async fn count_pending(db: &DatabaseConnection) -> SeshatResult<u64> {
    let count = Chunk::find()
        .filter(ChunkCol::EmbeddedAt.is_null())
        .count(db)
        .await?;
    Ok(count)
}
