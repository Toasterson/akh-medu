//! Async embed worker that claims unembedded drawer chunks and generates
//! embeddings using the ONNX MiniLM-L6-v2 model.
//!
//! A batch is claimed with `FOR UPDATE SKIP LOCKED`, embedded, inserted, and
//! marked complete inside one database transaction. The row lock prevents two
//! workers from processing the same chunk concurrently; the unique index and
//! `ON CONFLICT` make retries safe if an older worker inserted an embedding but
//! did not update `embedded_at`.

use std::sync::Arc;
use std::time::Duration;

use chrono::Utc;
use sea_orm::{
    ConnectionTrait, DatabaseBackend, DatabaseConnection, DatabaseTransaction, FromQueryResult,
    PaginatorTrait, Statement, TransactionTrait,
};
use uuid::Uuid;

use crate::config::EmbeddingConfig;
use crate::embedder::{EMBEDDING_DIM, InferencePermit, SharedEmbedder};
use crate::entity::heka_drawer_chunks::{self, Column as ChunkCol, Entity as Chunk};
use crate::entity::heka_drawer_embeddings::vec_to_pgvector;
use crate::error::{HekaError, HekaResult};

const CLAIM_PENDING_SQL: &str = "SELECT c.id, c.drawer_id, c.chunk_index, c.text, c.heading, c.word_count, \
            c.embedded_at, c.created_at \
       FROM heka_drawer_chunks AS c \
      WHERE c.embedded_at IS NULL \
      ORDER BY c.id \
      LIMIT $1 \
      FOR UPDATE OF c SKIP LOCKED";

const CLAIM_PENDING_FOR_AKH_SQL: &str = "SELECT c.id, c.drawer_id, c.chunk_index, c.text, c.heading, c.word_count, \
            c.embedded_at, c.created_at \
       FROM heka_drawer_chunks AS c \
       JOIN heka_drawers AS d ON d.id = c.drawer_id \
      WHERE c.embedded_at IS NULL AND d.akh_id = $1 \
      ORDER BY c.id \
      LIMIT $2 \
      FOR UPDATE OF c SKIP LOCKED";

const INSERT_EMBEDDING_SQL: &str = "INSERT INTO heka_drawer_embeddings (id, chunk_id, model_name, embedding, created_at) \
     VALUES ($1, $2, $3, $4::vector(384), $5) \
     ON CONFLICT (chunk_id) DO NOTHING";

const MARK_EMBEDDED_SQL: &str = "UPDATE heka_drawer_chunks SET embedded_at = $1 \
     WHERE id = $2 AND embedded_at IS NULL";

const MAX_EMBED_BATCH_SIZE: usize = 256;
const MAX_EMBED_BATCH_TEXT_BYTES: usize = 8 * 1024 * 1024;
const EMBED_BATCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Statistics from an embed worker run.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
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
    let interval = poll_interval(config.poll_interval_secs);
    let batch_size = config.batch_size;
    let model_name = config.model.clone();

    tracing::info!(
        batch_size,
        configured_poll_secs = config.poll_interval_secs,
        effective_poll_secs = interval.as_secs(),
        "heka embed worker started"
    );

    loop {
        match embed_pending_batch(&db, &embedder, batch_size, &model_name).await {
            Ok(stats) => {
                if stats.chunks_processed > 0 {
                    tracing::info!(
                        chunks = stats.chunks_processed,
                        embeddings = stats.embeddings_created,
                        "heka embed batch complete"
                    );
                    if stats.chunks_processed >= batch_size {
                        continue;
                    }
                }
            }
            Err(error) => {
                tracing::error!(%error, "heka embed worker error");
            }
        }

        tokio::time::sleep(interval).await;
    }
}

/// Process one batch of unembedded chunks from the full store.
pub async fn embed_pending_batch(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
) -> HekaResult<EmbedStats> {
    embed_pending_batch_scoped(db, embedder, batch_size, model_name, None).await
}

/// Process one batch scoped to an akh. Used by evaluation so an unrelated
/// backlog cannot delay the benchmark corpus.
pub async fn embed_pending_batch_for_akh(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
    akh_id: Uuid,
) -> HekaResult<EmbedStats> {
    embed_pending_batch_scoped(db, embedder, batch_size, model_name, Some(akh_id)).await
}

async fn embed_pending_batch_scoped(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
    akh_id: Option<Uuid>,
) -> HekaResult<EmbedStats> {
    validate_batch_size(batch_size)?;
    validate_model_name(model_name)?;

    // Acquire the single-flight lease before opening a transaction or claiming
    // rows. If a previous timed-out blocking task is still running, retries
    // backpressure here without holding database locks or spawning more work.
    let inference_permit = embedder.acquire_inference(EMBED_BATCH_TIMEOUT).await?;
    let transaction = db.begin().await?;
    let result = async {
        let pending_chunks = claim_pending(&transaction, batch_size, akh_id).await?;
        if pending_chunks.is_empty() {
            return Ok(EmbedStats::default());
        }
        process_claimed_batch(
            &transaction,
            embedder,
            inference_permit,
            model_name,
            pending_chunks,
        )
        .await
    }
    .await;

    finish_embedding_transaction(transaction, result).await
}

#[cfg(test)]
pub(crate) async fn embed_pending_batch_for_akh_with_claim_probe(
    db: &DatabaseConnection,
    embedder: &SharedEmbedder,
    batch_size: usize,
    model_name: &str,
    akh_id: Uuid,
    claim_barrier: Arc<tokio::sync::Barrier>,
    claimed_ids: tokio::sync::mpsc::Sender<Vec<Uuid>>,
) -> HekaResult<EmbedStats> {
    validate_batch_size(batch_size)?;
    validate_model_name(model_name)?;

    let inference_permit = embedder.acquire_inference(EMBED_BATCH_TIMEOUT).await?;
    let transaction = db.begin().await?;
    let result = async {
        let pending_chunks = claim_pending(&transaction, batch_size, Some(akh_id)).await?;
        if pending_chunks.is_empty() {
            return Err(HekaError::Embedding(
                "embedding claim-overlap probe received an empty claim".to_string(),
            ));
        }
        tokio::time::timeout(
            Duration::from_secs(5),
            claimed_ids.send(pending_chunks.iter().map(|chunk| chunk.id).collect()),
        )
        .await
        .map_err(|_| {
            HekaError::Embedding(
                "timed out reporting an embedding claim to the overlap observer".to_string(),
            )
        })?
        .map_err(|_| {
            HekaError::Embedding(
                "embedding claim-overlap observer closed before receiving a claim".to_string(),
            )
        })?;
        tokio::time::timeout(Duration::from_secs(10), claim_barrier.wait())
            .await
            .map_err(|_| {
                HekaError::Embedding(
                    "timed out proving overlapping disjoint embedding claims".to_string(),
                )
            })?;
        process_claimed_batch(
            &transaction,
            embedder,
            inference_permit,
            model_name,
            pending_chunks,
        )
        .await
    }
    .await;

    finish_embedding_transaction(transaction, result).await
}

async fn finish_embedding_transaction(
    transaction: DatabaseTransaction,
    result: HekaResult<EmbedStats>,
) -> HekaResult<EmbedStats> {
    match result {
        Ok(stats) => {
            transaction.commit().await?;
            Ok(stats)
        }
        Err(error) => {
            if let Err(rollback_error) = transaction.rollback().await {
                return Err(HekaError::Embedding(format!(
                    "{error}; additionally failed to roll back embedding claim: {rollback_error}"
                )));
            }
            Err(error)
        }
    }
}

async fn process_claimed_batch(
    transaction: &DatabaseTransaction,
    embedder: &SharedEmbedder,
    inference_permit: InferencePermit,
    model_name: &str,
    pending_chunks: Vec<heka_drawer_chunks::Model>,
) -> HekaResult<EmbedStats> {
    let chunk_count = pending_chunks.len();
    validate_text_budget(pending_chunks.iter().map(|chunk| chunk.text.as_str()))?;
    let texts_owned: Vec<String> = pending_chunks
        .iter()
        .map(|chunk| chunk.text.clone())
        .collect();
    let embeddings = embedder
        .infer_with_permit(inference_permit, EMBED_BATCH_TIMEOUT, move |model| {
            let text_refs: Vec<&str> = texts_owned.iter().map(String::as_str).collect();
            model.embed_batch(&text_refs)
        })
        .await?;

    if embeddings.len() != pending_chunks.len() {
        return Err(HekaError::Embedding(format!(
            "embedding model returned {} vectors for {} chunks",
            embeddings.len(),
            pending_chunks.len()
        )));
    }
    if let Some(invalid) = embeddings.iter().find(|embedding| {
        embedding.len() != EMBEDDING_DIM || embedding.iter().any(|value| !value.is_finite())
    }) {
        return Err(HekaError::Embedding(format!(
            "embedding model returned an invalid {}-value vector; expected {EMBEDDING_DIM} finite values",
            invalid.len()
        )));
    }

    let now = Utc::now();
    let mut embeddings_created = 0usize;

    for (chunk, embedding) in pending_chunks.iter().zip(&embeddings) {
        let inserted = transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                INSERT_EMBEDDING_SQL,
                [
                    Uuid::new_v4().into(),
                    chunk.id.into(),
                    model_name.to_string().into(),
                    vec_to_pgvector(embedding).into(),
                    now.into(),
                ],
            ))
            .await?;
        embeddings_created += inserted.rows_affected() as usize;

        transaction
            .execute(Statement::from_sql_and_values(
                DatabaseBackend::Postgres,
                MARK_EMBEDDED_SQL,
                [now.into(), chunk.id.into()],
            ))
            .await?;
    }

    Ok(EmbedStats {
        chunks_processed: chunk_count,
        embeddings_created,
    })
}

fn validate_batch_size(batch_size: usize) -> HekaResult<()> {
    if batch_size == 0 {
        return Err(HekaError::Config(
            "embedding batch_size must be greater than zero".to_string(),
        ));
    }
    if batch_size > MAX_EMBED_BATCH_SIZE {
        return Err(HekaError::Config(format!(
            "embedding batch_size {batch_size} exceeds the safety limit {MAX_EMBED_BATCH_SIZE}"
        )));
    }
    Ok(())
}

fn poll_interval(configured_secs: u64) -> Duration {
    const MIN_POLL_INTERVAL_SECS: u64 = 1;
    if configured_secs < MIN_POLL_INTERVAL_SECS {
        tracing::warn!(
            configured_poll_secs = configured_secs,
            effective_poll_secs = MIN_POLL_INTERVAL_SECS,
            "clamping embedding poll interval to prevent a busy loop"
        );
    }
    Duration::from_secs(configured_secs.max(MIN_POLL_INTERVAL_SECS))
}

fn validate_model_name(model_name: &str) -> HekaResult<()> {
    if model_name.trim().is_empty() || model_name.len() > 256 {
        return Err(HekaError::Config(
            "embedding model name must contain 1-256 bytes".to_string(),
        ));
    }
    Ok(())
}

fn validate_text_budget<'a>(texts: impl Iterator<Item = &'a str>) -> HekaResult<()> {
    let mut total = 0usize;
    for text in texts {
        total = total.checked_add(text.len()).ok_or_else(|| {
            HekaError::Embedding("embedding batch text size overflow".to_string())
        })?;
        if total > MAX_EMBED_BATCH_TEXT_BYTES {
            return Err(HekaError::Embedding(format!(
                "embedding batch contains {total} text bytes, exceeding the {MAX_EMBED_BATCH_TEXT_BYTES} byte limit"
            )));
        }
    }
    Ok(())
}

async fn claim_pending(
    transaction: &DatabaseTransaction,
    batch_size: usize,
    akh_id: Option<Uuid>,
) -> HekaResult<Vec<heka_drawer_chunks::Model>> {
    let limit = i64::try_from(batch_size).unwrap_or(i64::MAX);
    let statement = if let Some(akh_id) = akh_id {
        Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            CLAIM_PENDING_FOR_AKH_SQL,
            [akh_id.into(), limit.into()],
        )
    } else {
        Statement::from_sql_and_values(DatabaseBackend::Postgres, CLAIM_PENDING_SQL, [limit.into()])
    };

    Ok(heka_drawer_chunks::Model::find_by_statement(statement)
        .all(transaction)
        .await?)
}

/// Count chunks waiting to be embedded.
pub async fn count_pending(db: &DatabaseConnection) -> HekaResult<u64> {
    use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};

    Ok(Chunk::find()
        .filter(ChunkCol::EmbeddedAt.is_null())
        .count(db)
        .await?)
}

#[derive(Debug, FromQueryResult)]
struct PendingCount {
    count: i64,
}

/// Count pending chunks belonging to one akh.
pub async fn count_pending_for_akh(db: &DatabaseConnection, akh_id: Uuid) -> HekaResult<u64> {
    let statement = Statement::from_sql_and_values(
        DatabaseBackend::Postgres,
        "SELECT COUNT(*)::bigint AS count \
           FROM heka_drawer_chunks AS c \
           JOIN heka_drawers AS d ON d.id = c.drawer_id \
          WHERE c.embedded_at IS NULL AND d.akh_id = $1",
        [akh_id.into()],
    );
    let row = PendingCount::find_by_statement(statement)
        .one(db)
        .await?
        .ok_or_else(|| {
            HekaError::Database(sea_orm::DbErr::RecordNotFound(
                "pending chunk count returned no row".to_string(),
            ))
        })?;
    Ok(row.count.max(0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_claim_contract_is_bounded_ordered_and_skip_locked() {
        assert!(CLAIM_PENDING_SQL.contains("FOR UPDATE OF c SKIP LOCKED"));
        assert!(CLAIM_PENDING_FOR_AKH_SQL.contains("FOR UPDATE OF c SKIP LOCKED"));
        assert!(CLAIM_PENDING_SQL.contains("ORDER BY c.id"));
        assert!(CLAIM_PENDING_FOR_AKH_SQL.contains("ORDER BY c.id"));
        assert!(CLAIM_PENDING_SQL.contains("LIMIT $1"));
        assert!(CLAIM_PENDING_FOR_AKH_SQL.contains("LIMIT $2"));
        assert!(CLAIM_PENDING_FOR_AKH_SQL.contains("d.akh_id = $1"));
    }

    #[test]
    fn retry_contract_matches_the_database_uniqueness_constraint() {
        const MIGRATION_SQL: &str =
            include_str!("../migrations/20260604_000003_embedding_idempotency.sql");

        assert!(INSERT_EMBEDDING_SQL.contains("ON CONFLICT (chunk_id) DO NOTHING"));
        assert!(MARK_EMBEDDED_SQL.contains("embedded_at IS NULL"));
        assert!(MIGRATION_SQL.contains(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_heka_embeddings_chunk\n    ON heka_drawer_embeddings(chunk_id)"
        ));
    }

    #[test]
    fn resource_limits_reject_unbounded_batches_before_inference() {
        assert!(validate_batch_size(0).is_err());
        assert!(validate_batch_size(MAX_EMBED_BATCH_SIZE).is_ok());
        assert!(validate_batch_size(MAX_EMBED_BATCH_SIZE + 1).is_err());
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name(&"m".repeat(257)).is_err());

        let oversized = "x".repeat(MAX_EMBED_BATCH_TEXT_BYTES + 1);
        assert!(validate_text_budget([oversized.as_str()].into_iter()).is_err());
        assert!(validate_text_budget(["small", "batch"].into_iter()).is_ok());
    }

    #[test]
    fn zero_poll_interval_is_clamped_away_from_a_busy_loop() {
        assert_eq!(poll_interval(0), Duration::from_secs(1));
        assert_eq!(poll_interval(7), Duration::from_secs(7));
    }
}
