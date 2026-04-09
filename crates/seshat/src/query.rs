//! pgvector similarity search queries.
//!
//! Vector operations (cosine distance `<=>`) are not expressible through
//! SeaORM's query builder, so this module uses parameterized SQL via
//! [`sea_orm::ConnectionTrait`]. All other CRUD uses SeaORM entities.

use sea_orm::{DatabaseConnection, FromQueryResult, PaginatorTrait, Statement};
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
}

/// Search for chunks similar to the given embedding vector.
///
/// Returns up to `top_k` chunks with cosine similarity above `min_similarity`.
pub async fn search_similar(
    db: &DatabaseConnection,
    query_embedding: &[f32],
    top_k: usize,
    min_similarity: f32,
) -> SeshatResult<Vec<RetrievedChunk>> {
    let embedding_str = vec_to_pgvector(query_embedding);

    // pgvector's `<=>` operator returns cosine *distance* (0 = identical, 2 = opposite).
    // We convert to similarity: `1 - distance`.
    let sql = format!(
        "SELECT \
            c.id AS chunk_id, \
            c.document_id, \
            d.title AS document_title, \
            c.text, \
            c.chunk_index, \
            (1.0 - (e.embedding <=> '{embedding_str}'::vector))::real AS similarity, \
            c.heading \
         FROM pa_chunks c \
         JOIN pa_embeddings e ON e.chunk_id = c.id \
         JOIN pa_documents d ON d.id = c.document_id \
         WHERE (1.0 - (e.embedding <=> '{embedding_str}'::vector)) > {min_similarity} \
         ORDER BY e.embedding <=> '{embedding_str}'::vector \
         LIMIT {top_k}"
    );

    let results =
        RetrievedChunk::find_by_statement(Statement::from_string(sea_orm::DatabaseBackend::Postgres, sql))
            .all(db)
            .await?;

    Ok(results)
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
