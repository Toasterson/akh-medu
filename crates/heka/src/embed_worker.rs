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
use crate::embedder::SharedEmbedder;
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
    let interval = Duration::from_secs(config.poll_interval_secs);
    let batch_size = config.batch_size;
    let model_name = config.model.clone();

    tracing::info!(
        batch_size,
        poll_secs = config.poll_interval_secs,
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
    if batch_size == 0 {
        return Err(HekaError::Config(
            "embedding batch_size must be greater than zero".to_string(),
        ));
    }

    let transaction = db.begin().await?;
    let pending_chunks = claim_pending(&transaction, batch_size, akh_id).await?;
    if pending_chunks.is_empty() {
        transaction.commit().await?;
        return Ok(EmbedStats::default());
    }

    let chunk_count = pending_chunks.len();
    let texts_owned: Vec<String> = pending_chunks
        .iter()
        .map(|chunk| chunk.text.clone())
        .collect();
    let embedder = Arc::clone(embedder);
    let embeddings = tokio::task::spawn_blocking(move || {
        let text_refs: Vec<&str> = texts_owned.iter().map(String::as_str).collect();
        let mut guard = embedder
            .lock()
            .map_err(|error| HekaError::Embedding(format!("embedder lock: {error}")))?;
        guard.embed_batch(&text_refs)
    })
    .await
    .map_err(|error| HekaError::Embedding(format!("spawn_blocking: {error}")))??;

    if embeddings.len() != pending_chunks.len() {
        return Err(HekaError::Embedding(format!(
            "embedding model returned {} vectors for {} chunks",
            embeddings.len(),
            pending_chunks.len()
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

    transaction.commit().await?;

    Ok(EmbedStats {
        chunks_processed: chunk_count,
        embeddings_created,
    })
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
    fn claim_insert_and_marker_share_one_transaction_before_commit() {
        const SOURCE: &str = include_str!("embed_worker.rs");
        let body = SOURCE
            .split_once("async fn embed_pending_batch_scoped")
            .unwrap()
            .1
            .split_once("async fn claim_pending")
            .unwrap()
            .0;

        let begin = body.find("let transaction = db.begin().await?;").unwrap();
        let claim = body
            .find("claim_pending(&transaction, batch_size, akh_id)")
            .unwrap();
        let insert = body.find("INSERT_EMBEDDING_SQL").unwrap();
        let marker = body.find("MARK_EMBEDDED_SQL").unwrap();
        let commit = body.rfind("transaction.commit().await?;").unwrap();

        assert!(begin < claim);
        assert!(claim < insert);
        assert!(insert < marker);
        assert!(marker < commit);
        assert!(!body[claim..commit].contains("db.execute("));
    }
}
