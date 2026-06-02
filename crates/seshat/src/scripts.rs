//! REST API handlers for ELIZA KDL script management.
//!
//! Exposes three routes:
//! - `GET  /scripts`        — list all ELIZA scripts (name, revision, saved_at)
//! - `GET  /scripts/{name}` — load a script by name (content, revision, saved_at)
//! - `POST /scripts/{name}` — save/overwrite a script by name → revision ID
//!
//! Scripts are stored as `pa_documents` rows with the slug
//! `eliza/scripts/{name}` and the `eliza-script` tag.
//! Full content is preserved verbatim in `metadata.content` so callers can
//! reconstruct the exact KDL text without re-joining chunks.

use axum::{
    Json,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};
use chrono::{DateTime, Utc};
use sea_orm::{
    ColumnTrait, DatabaseConnection, EntityTrait, QueryFilter, QueryOrder, TransactionTrait,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use crate::entity::{pa_chunk, pa_document, pa_embedding};
use crate::error::SeshatResult;
use crate::ingest::{IngestRequest, ingest_document};

const ELIZA_TAG: &str = "eliza-script";
const ELIZA_SLUG_PREFIX: &str = "eliza/scripts/";
/// Use a very large chunk budget so the entire KDL script stays in one chunk.
const SCRIPT_CHUNK_CHARS: usize = 100_000;

/// Shared state for script handlers — just the DB pool.
pub type ScriptsState = Arc<DatabaseConnection>;

// ── Response / request types ────────────────────────────────────────────────

/// Brief metadata about a stored script.
#[derive(Debug, Serialize, Deserialize)]
pub struct ScriptInfo {
    pub name: String,
    /// Document UUID; acts as a monotonically-increasing revision identifier.
    pub revision: String,
    pub saved_at: DateTime<Utc>,
}

/// Full script content plus metadata.
#[derive(Debug, Serialize, Deserialize)]
pub struct ScriptContent {
    pub name: String,
    pub content: String,
    pub revision: String,
    pub saved_at: DateTime<Utc>,
}

/// Save-script request body.
#[derive(Debug, Deserialize)]
pub struct SaveRequest {
    pub content: String,
}

/// Save-script response.
#[derive(Debug, Serialize)]
pub struct SaveResult {
    pub revision: String,
    pub saved_at: DateTime<Utc>,
}

// ── Axum handlers ───────────────────────────────────────────────────────────

pub async fn list_scripts(State(db): State<ScriptsState>) -> Response {
    match list_scripts_inner(&db).await {
        Ok(scripts) => (StatusCode::OK, Json(scripts)).into_response(),
        Err(e) => {
            tracing::error!("list_scripts: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("error listing scripts: {e}"),
            )
                .into_response()
        }
    }
}

pub async fn get_script(State(db): State<ScriptsState>, Path(name): Path<String>) -> Response {
    match get_script_inner(&db, &name).await {
        Ok(Some(script)) => (StatusCode::OK, Json(script)).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            format!("script '{name}' not found"),
        )
            .into_response(),
        Err(e) => {
            tracing::error!("get_script({name}): {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("error loading script: {e}"),
            )
                .into_response()
        }
    }
}

pub async fn save_script(
    State(db): State<ScriptsState>,
    Path(name): Path<String>,
    Json(req): Json<SaveRequest>,
) -> Response {
    if req.content.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "content must not be empty").into_response();
    }
    match save_script_inner(&db, &name, &req.content).await {
        Ok(result) => (StatusCode::OK, Json(result)).into_response(),
        Err(e) => {
            tracing::error!("save_script({name}): {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("error saving script: {e}"),
            )
                .into_response()
        }
    }
}

// ── Core logic ──────────────────────────────────────────────────────────────

async fn list_scripts_inner(db: &DatabaseConnection) -> SeshatResult<Vec<ScriptInfo>> {
    use pa_document::{Column, Entity as Document};

    let docs = Document::find()
        .filter(Column::Slug.like(format!("{ELIZA_SLUG_PREFIX}%")))
        .order_by_desc(Column::IngestedAt)
        .all(db)
        .await?;

    Ok(docs
        .into_iter()
        .map(|d| {
            let name = d.slug[ELIZA_SLUG_PREFIX.len()..].to_string();
            ScriptInfo {
                name,
                revision: d.id.to_string(),
                saved_at: d.ingested_at,
            }
        })
        .collect())
}

async fn get_script_inner(
    db: &DatabaseConnection,
    name: &str,
) -> SeshatResult<Option<ScriptContent>> {
    use pa_document::{Column, Entity as Document};

    let slug = format!("{ELIZA_SLUG_PREFIX}{name}");

    let Some(doc) = Document::find()
        .filter(Column::Slug.eq(&slug))
        .one(db)
        .await?
    else {
        return Ok(None);
    };

    // Prefer the verbatim content stored in metadata (exact whitespace preserved).
    // Fall back to concatenating chunks for documents ingested without it.
    let content = if let Some(body) = doc.metadata.get("content").and_then(|v| v.as_str()) {
        body.to_string()
    } else {
        assemble_chunks(db, doc.id).await?
    };

    Ok(Some(ScriptContent {
        name: name.to_string(),
        content,
        revision: doc.id.to_string(),
        saved_at: doc.ingested_at,
    }))
}

async fn save_script_inner(
    db: &DatabaseConnection,
    name: &str,
    content: &str,
) -> SeshatResult<SaveResult> {
    use pa_document::{Column, Entity as Document};

    let slug = format!("{ELIZA_SLUG_PREFIX}{name}");

    // Upsert: delete any existing document with this slug before re-ingesting.
    if let Some(existing) = Document::find()
        .filter(Column::Slug.eq(&slug))
        .one(db)
        .await?
    {
        delete_document_and_chunks(db, existing.id).await?;
    }

    let req = IngestRequest {
        title: format!("ELIZA Script: {name}"),
        slug: Some(slug),
        format: "text".to_string(),
        body: content.to_string(),
        source_path: None,
        tags: vec![ELIZA_TAG.to_string()],
        // Store the verbatim content in metadata so get_script_inner can
        // retrieve it without chunk re-assembly (preserves exact whitespace).
        metadata: serde_json::json!({ "script_name": name, "content": content }),
        ingested_by: Some("eliza-script-store".to_string()),
        target_chunk_chars: SCRIPT_CHUNK_CHARS,
    };

    let summary = ingest_document(db, req).await?;

    Ok(SaveResult {
        revision: summary.document_id.to_string(),
        saved_at: Utc::now(),
    })
}

// ── DB helpers ──────────────────────────────────────────────────────────────

/// Concatenate all chunks for a document in chunk-index order.
async fn assemble_chunks(db: &DatabaseConnection, doc_id: Uuid) -> SeshatResult<String> {
    use pa_chunk::{Column, Entity as Chunk};

    let chunks = Chunk::find()
        .filter(Column::DocumentId.eq(doc_id))
        .order_by_asc(Column::ChunkIndex)
        .all(db)
        .await?;

    Ok(chunks.into_iter().map(|c| c.text).collect::<Vec<_>>().join(""))
}

/// Delete a document together with its chunks and embeddings, all in one transaction.
async fn delete_document_and_chunks(db: &DatabaseConnection, doc_id: Uuid) -> SeshatResult<()> {
    use pa_chunk::{Column as ChunkCol, Entity as Chunk};
    use pa_document::{Column as DocCol, Entity as Document};
    use pa_embedding::{Column as EmbCol, Entity as Embedding};

    let txn = db.begin().await?;

    // Collect chunk IDs so we can remove their embeddings first.
    let chunk_ids: Vec<Uuid> = Chunk::find()
        .filter(ChunkCol::DocumentId.eq(doc_id))
        .all(&txn)
        .await?
        .into_iter()
        .map(|c| c.id)
        .collect();

    if !chunk_ids.is_empty() {
        Embedding::delete_many()
            .filter(EmbCol::ChunkId.is_in(chunk_ids))
            .exec(&txn)
            .await?;
    }

    Chunk::delete_many()
        .filter(ChunkCol::DocumentId.eq(doc_id))
        .exec(&txn)
        .await?;

    Document::delete_many()
        .filter(DocCol::Id.eq(doc_id))
        .exec(&txn)
        .await?;

    txn.commit().await?;
    Ok(())
}
