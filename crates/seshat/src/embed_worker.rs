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
        // One tick of the worker: spans the whole batch, including the
        // blocking embed call. Attach the tick span to the *future* via
        // `Instrument::instrument` rather than holding `span.entered()`
        // across the `.await` — `EnteredSpan` is `!Send`, and the future is
        // spawned by `tokio::spawn` at the call site, so a held guard would
        // make `run_embed_worker`'s future non-`Send` and fail to compile.
        let tick_span = tracing::info_span!("seshat.embed_worker.tick");
        let tick_result = tracing::Instrument::instrument(
            embed_pending_batch(
                &db,
                &embedder,
                batch_size,
                &model_name,
                #[cfg(feature = "tracing-otel")]
                metrics.as_deref(),
            ),
            tick_span,
        )
        .await;
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
/// counters per invocation via `record_embed_batch`, on empty, populated and
/// failed ticks (`metrics` is `tracing-otel` builds only).
///
/// Trace-stitching: the caller (worker tick or `seshat_embed_now`) attaches
/// its span to this future via `Instrument::instrument`; the closure below
/// then captures `tracing::Span::current()` before `spawn_blocking` and
/// explicitly re-enters it inside the blocking work so the ONNX-inference
/// span context stays parented to the tick trace.
pub async fn embed_pending_batch(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
    #[cfg(feature = "tracing-otel")] metrics: Option<&crate::metrics::SeshatMetrics>,
) -> SeshatResult<EmbedStats> {
    let embedder = Arc::clone(embedder);
    embed_pending_batch_with(
        db,
        batch_size,
        model_name,
        #[cfg(feature = "tracing-otel")]
        metrics,
        move |texts| {
            let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            let mut guard = embedder
                .lock()
                .map_err(|e| crate::SeshatError::Embedding(format!("embedder lock: {e}")))?;
            guard.embed_batch(&text_refs).map_err(crate::SeshatError::from)
        },
    )
    .await
}

// Keep the database and instrumentation path identical for ONNX and tests;
// only the blocking inference function is injected.
async fn embed_pending_batch_with<F>(
    db: &DatabaseConnection,
    batch_size: usize,
    model_name: &str,
    #[cfg(feature = "tracing-otel")] metrics: Option<&crate::metrics::SeshatMetrics>,
    embed: F,
) -> SeshatResult<EmbedStats>
where
    F: FnOnce(Vec<String>) -> SeshatResult<Vec<Vec<f32>>> + Send + 'static,
{
    #[cfg(feature = "tracing-otel")]
    let batch_start = std::time::Instant::now();
    let result = embed_pending_batch_impl(db, batch_size, model_name, embed).await;
    #[cfg(feature = "tracing-otel")]
    if let Some(metrics) = metrics {
        let chunks = result
            .as_ref()
            .map_or(0, |stats| stats.chunks_processed as u64);
        metrics.record_embed_batch(batch_start.elapsed().as_secs_f64(), chunks);
        // Query the full backlog after every outcome, including failures;
        // batch length cannot represent work still waiting beyond this tick.
        match count_pending(db).await {
            Ok(pending) => metrics.record_pending_embedding(Some(pending)),
            Err(error) => {
                metrics.record_pending_embedding(None);
                tracing::warn!(%error, "could not refresh pending embedding metric");
            }
        }
    }
    result
}

async fn embed_pending_batch_impl<F>(
    db: &DatabaseConnection,
    batch_size: usize,
    model_name: &str,
    embed: F,
) -> SeshatResult<EmbedStats>
where
    F: FnOnce(Vec<String>) -> SeshatResult<Vec<Vec<f32>>> + Send + 'static,
{
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

    let texts = pending_chunks
        .iter()
        .map(|chunk| chunk.text.clone())
        .collect();
    let parent = tracing::Span::current();
    let dispatch = tracing::dispatcher::get_default(Clone::clone);
    let embeddings = tokio::task::spawn_blocking(move || {
        tracing::dispatcher::with_default(&dispatch, || {
            let _entered = parent.entered();
            let _inference = tracing::info_span!("seshat.embed_batch.onnx").entered();
            embed(texts)
        })
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

#[cfg(all(test, feature = "tracing-otel"))]
#[path = "../tests/support/embed_worker.rs"]
mod instrumentation_tests;
