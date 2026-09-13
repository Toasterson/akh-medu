//! pgvector similarity search queries.
//!
//! Vector operations (cosine distance `<=>`) are not expressible through
//! SeaORM's query builder, so this module uses parameterized SQL via
//! [`sea_orm::ConnectionTrait`]. All other CRUD uses SeaORM entities.

use sea_orm::{DatabaseBackend, DatabaseConnection, FromQueryResult, PaginatorTrait, Statement};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entity::pa_embedding::vec_to_pgvector;
use crate::error::SeshatResult;

/// A chunk retrieved by vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize, FromQueryResult, schemars::JsonSchema)]
pub struct RetrievedChunk {
    pub chunk_id: Uuid,
    pub document_id: Uuid,
    pub document_title: String,
    pub text: String,
    pub chunk_index: i32,
    pub similarity: f32,
    pub heading: Option<String>,
    pub document_format: String,
    pub document_tags: serde_json::Value,
    pub document_metadata: serde_json::Value,
}

/// Optional exact metadata filters for vector search.
///
/// These fields match the top-level metadata stamped by Anima's transcript
/// producer. Values are bound as SQL parameters; they are never interpolated
/// into the query.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchFilters {
    pub work_item_id: Option<String>,
    pub executor: Option<String>,
    pub outcome: Option<String>,
}

/// Search for chunks similar to the given embedding vector.
///
/// Returns up to `top_k` chunks with cosine similarity above `min_similarity`.
pub async fn search_similar(
    db: &DatabaseConnection,
    query_embedding: &[f32],
    top_k: usize,
    min_similarity: f32,
    #[cfg(feature = "tracing-otel")] metrics: Option<&crate::metrics::SeshatMetrics>,
) -> SeshatResult<Vec<RetrievedChunk>> {
    search_similar_filtered(
        db,
        query_embedding,
        top_k,
        min_similarity,
        &SearchFilters::default(),
        #[cfg(feature = "tracing-otel")]
        metrics,
    )
    .await
}

/// Search for similar chunks, optionally constrained by Anima session
/// metadata. Document metadata and tags are returned with every hit so callers
/// retain provenance without a second lookup.
pub async fn search_similar_filtered(
    db: &DatabaseConnection,
    query_embedding: &[f32],
    top_k: usize,
    min_similarity: f32,
    filters: &SearchFilters,
    #[cfg(feature = "tracing-otel")] metrics: Option<&crate::metrics::SeshatMetrics>,
) -> SeshatResult<Vec<RetrievedChunk>> {
    #[cfg(feature = "tracing-otel")]
    let span = tracing::info_span!("seshat.pgvector_search", top_k, min_similarity);
    #[cfg(feature = "tracing-otel")]
    let start = std::time::Instant::now();
    let embedding_str = vec_to_pgvector(query_embedding);
    let statement = build_search_statement(&embedding_str, top_k, min_similarity, filters);
    let query = RetrievedChunk::find_by_statement(statement);
    let query_fut = query.all(db);
    #[cfg(feature = "tracing-otel")]
    let results = tracing::Instrument::instrument(query_fut, span).await;
    #[cfg(not(feature = "tracing-otel"))]
    let results = query_fut.await;

    // Record latency before propagating a database error.
    #[cfg(feature = "tracing-otel")]
    if let Some(metrics) = metrics {
        metrics.record_pgvector_search(start.elapsed().as_secs_f64());
    }
    let results = results?;

    Ok(results)
}

fn build_search_statement(
    embedding_str: &str,
    top_k: usize,
    min_similarity: f32,
    filters: &SearchFilters,
) -> Statement {
    // pgvector's `<=>` operator returns cosine *distance* (0 = identical, 2 = opposite).
    // We convert to similarity: `1 - distance`.
    let mut sql = String::from(
        "SELECT \
            c.id AS chunk_id, \
            c.document_id, \
            d.title AS document_title, \
            c.text, \
            c.chunk_index, \
            (1.0 - (e.embedding <=> $1::vector))::real AS similarity, \
            c.heading, \
            d.format AS document_format, \
            d.tags AS document_tags, \
            d.metadata AS document_metadata \
         FROM pa_chunks c \
         JOIN pa_embeddings e ON e.chunk_id = c.id \
         JOIN pa_documents d ON d.id = c.document_id \
         WHERE (1.0 - (e.embedding <=> $1::vector)) > $2",
    );
    let mut values: Vec<sea_orm::Value> =
        vec![embedding_str.to_string().into(), min_similarity.into()];

    if let Some(value) = nonempty(filters.work_item_id.as_deref()) {
        values.push(value.to_string().into());
        sql.push_str(&format!(
            " AND d.metadata ->> 'work_item_id' = ${}",
            values.len()
        ));
    }
    if let Some(value) = nonempty(filters.executor.as_deref()) {
        values.push(value.to_string().into());
        sql.push_str(&format!(
            " AND LOWER(d.metadata ->> 'executor') = LOWER(${})",
            values.len()
        ));
    }
    if let Some(value) = nonempty(filters.outcome.as_deref()) {
        values.push(value.to_string().into());
        sql.push_str(&format!(
            " AND LOWER(d.metadata ->> 'outcome') = LOWER(${})",
            values.len()
        ));
    }
    values.push((top_k.min(i64::MAX as usize) as i64).into());
    sql.push_str(&format!(
        " ORDER BY e.embedding <=> $1::vector LIMIT ${}",
        values.len()
    ));

    Statement::from_sql_and_values(DatabaseBackend::Postgres, sql, values)
}

fn nonempty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use sea_orm::MockDatabase;

    const COMPLETED_SESSION: &str = include_str!("../tests/fixtures/completed_agent_session.jsonl");

    #[test]
    fn transcript_search_filters_are_parameterized_and_composable() {
        let filters = SearchFilters {
            work_item_id: Some("01a03511-6678-7e93-89d7-64fdb0033c44".to_string()),
            executor: Some("Codex'); DROP TABLE pa_documents; --".to_string()),
            outcome: Some("completed".to_string()),
        };
        let statement = build_search_statement("[0.1,0.2]", 7, 0.42, &filters);

        assert!(statement.sql.contains("d.metadata ->> 'work_item_id' = $3"));
        assert!(
            statement
                .sql
                .contains("LOWER(d.metadata ->> 'executor') = LOWER($4)")
        );
        assert!(
            statement
                .sql
                .contains("LOWER(d.metadata ->> 'outcome') = LOWER($5)")
        );
        assert!(statement.sql.ends_with("LIMIT $6"));
        assert!(!statement.sql.contains("DROP TABLE"));
        assert_eq!(statement.values.expect("bound values").0.len(), 6);
    }

    #[test]
    fn blank_search_filters_do_not_constrain_generic_search() {
        let filters = SearchFilters {
            work_item_id: Some("  ".to_string()),
            executor: None,
            outcome: Some(String::new()),
        };
        let statement = build_search_statement("[0.1,0.2]", 10, 0.3, &filters);

        assert!(!statement.sql.contains("d.metadata ->>"));
        assert!(statement.sql.ends_with("LIMIT $3"));
        assert_eq!(statement.values.expect("bound values").0.len(), 3);
    }

    #[tokio::test]
    async fn filtered_search_returns_transcript_content_with_anima_provenance() {
        let tags = serde_json::json!(["WI-341", "agent-session"]);
        let metadata = serde_json::json!({
            "work_item_id": "01a03511-6678-7e93-89d7-64fdb0033c44",
            "executor": "codex",
            "outcome": "completed",
            "producer_extension": { "retry": 2 }
        });
        let transcript_chunks = crate::transcript::chunk_agent_session_transcript(
            COMPLETED_SESSION,
            crate::ingest::DEFAULT_TARGET_CHUNK_CHARS,
            &["WI-341".to_string(), "agent-session".to_string()],
            &metadata,
        )
        .expect("representative completed transcript should produce searchable chunks");
        let document_id = Uuid::new_v4();
        let chunk_id = Uuid::new_v4();
        let row = BTreeMap::from([
            ("chunk_id", chunk_id.into()),
            ("document_id", document_id.into()),
            ("document_title", "WI-341 completed session".into()),
            ("text", transcript_chunks[0].text.clone().into()),
            ("chunk_index", 0i32.into()),
            ("similarity", 0.91f32.into()),
            ("heading", transcript_chunks[0].heading.clone().into()),
            (
                "document_format",
                crate::transcript::AGENT_SESSION_TRANSCRIPT_FORMAT.into(),
            ),
            ("document_tags", tags.clone().into()),
            ("document_metadata", metadata.clone().into()),
        ]);
        let db = MockDatabase::new(DatabaseBackend::Postgres)
            .append_query_results([[row]])
            .into_connection();
        let filters = SearchFilters {
            work_item_id: Some("01a03511-6678-7e93-89d7-64fdb0033c44".to_string()),
            executor: Some("CODEX".to_string()),
            outcome: Some("Completed".to_string()),
        };

        let hits = search_similar_filtered(
            &db,
            &[0.1, 0.2],
            5,
            0.3,
            &filters,
            #[cfg(feature = "tracing-otel")]
            None,
        )
        .await
        .expect("filtered transcript search should succeed");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].document_id, document_id);
        assert_eq!(hits[0].chunk_id, chunk_id);
        assert_eq!(
            hits[0].document_format,
            crate::transcript::AGENT_SESSION_TRANSCRIPT_FORMAT
        );
        assert_eq!(hits[0].document_tags, tags);
        assert_eq!(hits[0].document_metadata, metadata);
        assert!(hits[0].text.contains("Work item: 01a03511"));
        assert!(
            hits[0]
                .text
                .contains("Implemented metadata-preserving transcript extraction.")
        );

        let transaction_log = db.into_transaction_log();
        let statement = &transaction_log[0].statements()[0];
        assert!(statement.sql.contains("d.metadata ->> 'work_item_id' = $3"));
        assert!(statement.sql.contains("LOWER(d.metadata ->> 'executor')"));
        assert!(statement.sql.contains("LOWER(d.metadata ->> 'outcome')"));
    }
}

/// Count total documents in the archive.
pub async fn count_documents(db: &DatabaseConnection) -> SeshatResult<u64> {
    use crate::entity::pa_document::Entity as Document;
    use sea_orm::EntityTrait;

    let count = Document::find().count(db).await?;
    Ok(count)
}

/// Count total chunks in the archive.
pub async fn count_chunks(db: &DatabaseConnection) -> SeshatResult<u64> {
    use crate::entity::pa_chunk::Entity as Chunk;
    use sea_orm::EntityTrait;

    let count = Chunk::find().count(db).await?;
    Ok(count)
}

/// Count chunks that have been embedded.
pub async fn count_embedded_chunks(db: &DatabaseConnection) -> SeshatResult<u64> {
    use crate::entity::pa_chunk::{Column, Entity as Chunk};
    use sea_orm::{ColumnTrait, EntityTrait, QueryFilter};

    let count = Chunk::find()
        .filter(Column::EmbeddedAt.is_not_null())
        .count(db)
        .await?;
    Ok(count)
}

/// Count cached microtheories.
pub async fn count_cached_microtheories(db: &DatabaseConnection) -> SeshatResult<u64> {
    use crate::entity::pa_microtheory_cache::Entity as Cache;
    use sea_orm::EntityTrait;

    let count = Cache::find().count(db).await?;
    Ok(count)
}

/// Retrieve surrounding chunks for context assembly.
///
/// Given a chunk hit, returns ±`window` surrounding chunks from the same document.
pub async fn surrounding_chunks(
    db: &DatabaseConnection,
    document_id: Uuid,
    chunk_index: i32,
    window: i32,
) -> SeshatResult<Vec<crate::entity::pa_chunk::Model>> {
    use crate::entity::pa_chunk::{Column, Entity as Chunk};
    use sea_orm::{ColumnTrait, EntityTrait, QueryFilter, QueryOrder};

    let lo = chunk_index.saturating_sub(window);
    let hi = chunk_index.saturating_add(window);

    let chunks = Chunk::find()
        .filter(Column::DocumentId.eq(document_id))
        .filter(Column::ChunkIndex.gte(lo))
        .filter(Column::ChunkIndex.lte(hi))
        .order_by_asc(Column::ChunkIndex)
        .all(db)
        .await?;

    Ok(chunks)
}
