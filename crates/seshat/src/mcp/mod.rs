//! MCP server for Seshat's Archive.
//!
//! Exposes corpus search, wish, status, and document management tools
//! via the Model Context Protocol over streamable HTTP.

use std::sync::Arc;

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::schemars;
use rmcp::{ErrorData as McpError, tool, tool_router};
use schemars::JsonSchema;
use sea_orm::DatabaseConnection;
use serde::{Deserialize, Serialize};

use crate::embedder::SharedEmbedder;

/// Shared state for MCP tool handlers.
pub struct SeshatMcpState {
    pub db: Arc<DatabaseConnection>,
    pub embedder: Option<SharedEmbedder>,
}

/// MCP server exposing Seshat's Archive tools.
#[derive(Clone)]
pub struct SeshatMcpServer {
    state: Arc<SeshatMcpState>,
    tool_router: ToolRouter<Self>,
}

// ── Tool parameter types ────────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
struct SearchParams {
    /// Natural-language query to search for in the corpus.
    query: String,
    /// Maximum number of results to return.
    #[serde(default = "default_top_k")]
    top_k: usize,
    /// Minimum cosine similarity threshold (0.0–1.0).
    #[serde(default = "default_min_similarity")]
    min_similarity: f32,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct WishParams {
    /// Topic or question to wish for knowledge about.
    query: String,
    /// If true, skip cache and re-synthesize.
    #[serde(default)]
    force_refresh: bool,
    /// If true, run LLM synthesis in this service.
    /// If false, return raw context chunks for client-side synthesis.
    #[serde(default)]
    synthesize_here: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct DocumentsParams {
    /// Maximum documents to return.
    #[serde(default = "default_doc_limit")]
    limit: usize,
    /// Offset for pagination.
    #[serde(default)]
    offset: usize,
}

fn default_top_k() -> usize {
    10
}
fn default_min_similarity() -> f32 {
    0.3
}
fn default_doc_limit() -> usize {
    50
}

// ── Tool response types ─────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ArchiveStatus {
    pub total_documents: u64,
    pub total_chunks: u64,
    pub embedded_chunks: u64,
    pub pending_embedding: u64,
    pub cached_microtheories: u64,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct DocumentSummary {
    pub id: String,
    pub title: String,
    pub slug: String,
    pub format: String,
    pub chunk_count: i32,
    pub ingested_at: String,
}

// ── MCP Server implementation ───────────────────────────────────────────

#[tool_router]
impl SeshatMcpServer {
    pub fn new(state: Arc<SeshatMcpState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        name = "seshat_search",
        description = "Search the corpus for chunks similar to a query using vector similarity. Returns ranked text chunks with source document info and similarity scores."
    )]
    async fn seshat_search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let embedder = self
            .state
            .embedder
            .as_ref()
            .ok_or_else(|| McpError::internal_error("embedding not configured", None))?;

        // Embed the query.
        let query_embedding = {
            let embedder_clone = Arc::clone(embedder);
            let query = params.query.clone();
            tokio::task::spawn_blocking(move || {
                let mut guard = embedder_clone
                    .lock()
                    .map_err(|e| format!("embedder lock: {e}"))?;
                guard
                    .embed_text(&query)
                    .map_err(|e| format!("embedding: {e}"))
            })
            .await
            .map_err(|e| McpError::internal_error(format!("task: {e}"), None))?
            .map_err(|e| McpError::internal_error(e, None))?
        };

        // Search pgvector.
        let results = crate::query::search_similar(
            &self.state.db,
            &query_embedding,
            params.top_k,
            params.min_similarity,
        )
        .await
        .map_err(|e| McpError::internal_error(format!("search: {e}"), None))?;

        let json = serde_json::to_string_pretty(&results)
            .map_err(|e| McpError::internal_error(format!("serialize: {e}"), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "seshat_wish",
        description = "Wish for knowledge about a topic. Performs RAG retrieval from the corpus and optionally synthesizes a microtheory (structured triples). Returns either synthesized triples (if synthesize_here=true) or raw context chunks for client-side synthesis."
    )]
    async fn seshat_wish(
        &self,
        Parameters(params): Parameters<WishParams>,
    ) -> Result<CallToolResult, McpError> {
        // Check cache first (unless force_refresh).
        if !params.force_refresh {
            if let Some(cached) = self.check_cache(&params.query).await? {
                return Ok(CallToolResult::success(vec![Content::text(cached)]));
            }
        }

        // Embed query and search.
        let embedder = self
            .state
            .embedder
            .as_ref()
            .ok_or_else(|| McpError::internal_error("embedding not configured", None))?;

        let query_embedding = {
            let embedder_clone = Arc::clone(embedder);
            let query = params.query.clone();
            tokio::task::spawn_blocking(move || {
                let mut guard = embedder_clone
                    .lock()
                    .map_err(|e| format!("embedder lock: {e}"))?;
                guard
                    .embed_text(&query)
                    .map_err(|e| format!("embedding: {e}"))
            })
            .await
            .map_err(|e| McpError::internal_error(format!("task: {e}"), None))?
            .map_err(|e| McpError::internal_error(e, None))?
        };

        let chunks = crate::query::search_similar(&self.state.db, &query_embedding, 10, 0.3)
            .await
            .map_err(|e| McpError::internal_error(format!("search: {e}"), None))?;

        if chunks.is_empty() {
            return Err(McpError::internal_error(
                format!("no relevant chunks found for query: {}", params.query),
                None,
            ));
        }

        if params.synthesize_here {
            // TODO (37d): Run LLM synthesis here using Candle.
            // For now, return the context chunks as-is.
            let result = serde_json::json!({
                "status": "synthesis_not_yet_implemented",
                "message": "LLM synthesis in service requires Phase 37d. Returning raw chunks.",
                "context_chunks": chunks,
                "source_documents": chunks.iter().map(|c| &c.document_title).collect::<Vec<_>>(),
                "cache_hit": false,
            });
            let json = serde_json::to_string_pretty(&result)
                .map_err(|e| McpError::internal_error(format!("serialize: {e}"), None))?;
            Ok(CallToolResult::success(vec![Content::text(json)]))
        } else {
            // Return raw chunks for client-side synthesis.
            let result = serde_json::json!({
                "context_chunks": chunks,
                "source_documents": chunks.iter().map(|c| &c.document_title).collect::<Vec<_>>(),
                "cache_hit": false,
                "confidence": 0.0,
            });
            let json = serde_json::to_string_pretty(&result)
                .map_err(|e| McpError::internal_error(format!("serialize: {e}"), None))?;
            Ok(CallToolResult::success(vec![Content::text(json)]))
        }
    }

    #[tool(
        name = "seshat_status",
        description = "Get Seshat archive status: document count, chunk count, embedding progress, and cache statistics."
    )]
    async fn seshat_status(&self) -> Result<CallToolResult, McpError> {
        let total_documents = crate::query::count_documents(&self.state.db)
            .await
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;
        let total_chunks = crate::query::count_chunks(&self.state.db)
            .await
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;
        let embedded_chunks = crate::query::count_embedded_chunks(&self.state.db)
            .await
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;
        let cached_microtheories = crate::query::count_cached_microtheories(&self.state.db)
            .await
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;

        let status = ArchiveStatus {
            total_documents,
            total_chunks,
            embedded_chunks,
            pending_embedding: total_chunks.saturating_sub(embedded_chunks),
            cached_microtheories,
        };

        let json = serde_json::to_string_pretty(&status)
            .map_err(|e| McpError::internal_error(format!("serialize: {e}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "seshat_documents",
        description = "List documents in Seshat's Archive with title, format, chunk count, and ingestion timestamp."
    )]
    async fn seshat_documents(
        &self,
        Parameters(params): Parameters<DocumentsParams>,
    ) -> Result<CallToolResult, McpError> {
        use crate::entity::pa_document::Entity as Document;
        use sea_orm::{EntityTrait, QueryOrder, QuerySelect};

        let docs = Document::find()
            .order_by_desc(crate::entity::pa_document::Column::IngestedAt)
            .offset(params.offset as u64)
            .limit(params.limit as u64)
            .all(self.state.db.as_ref())
            .await
            .map_err(|e| McpError::internal_error(format!("query: {e}"), None))?;

        let summaries: Vec<DocumentSummary> = docs
            .into_iter()
            .map(|d| DocumentSummary {
                id: d.id.to_string(),
                title: d.title,
                slug: d.slug,
                format: d.format,
                chunk_count: d.chunk_count,
                ingested_at: d.ingested_at.to_string(),
            })
            .collect();

        let json = serde_json::to_string_pretty(&summaries)
            .map_err(|e| McpError::internal_error(format!("serialize: {e}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "seshat_embed_now",
        description = "Force the embedding worker to process all pending chunks immediately. Returns the count of newly embedded chunks."
    )]
    async fn seshat_embed_now(&self) -> Result<CallToolResult, McpError> {
        let embedder = self
            .state
            .embedder
            .as_ref()
            .ok_or_else(|| McpError::internal_error("embedding not configured", None))?;

        let stats = crate::embed_worker::embed_pending_batch(
            &self.state.db,
            embedder,
            256, // larger batch for manual trigger
            "all-MiniLM-L6-v2",
        )
        .await
        .map_err(|e| McpError::internal_error(format!("embed: {e}"), None))?;

        let result = serde_json::json!({
            "chunks_processed": stats.chunks_processed,
            "embeddings_created": stats.embeddings_created,
        });
        let json = serde_json::to_string_pretty(&result)
            .map_err(|e| McpError::internal_error(format!("serialize: {e}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

// ── Helper methods ──────────────────────────────────────────────────────

impl SeshatMcpServer {
    /// Check the microtheory cache for a previous synthesis of this query.
    async fn check_cache(&self, query: &str) -> Result<Option<String>, McpError> {
        use crate::entity::pa_microtheory_cache::{Column, Entity as Cache};
        use sea_orm::{ActiveModelTrait, ColumnTrait, EntityTrait, QueryFilter, Set};
        use sha2::{Digest, Sha256};

        let normalized = query.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
        let hash = format!("{:x}", Sha256::digest(normalized.as_bytes()));

        let cached = Cache::find()
            .filter(Column::QueryHash.eq(&hash))
            .one(self.state.db.as_ref())
            .await
            .map_err(|e| McpError::internal_error(format!("cache query: {e}"), None))?;

        if let Some(entry) = cached {
            // Update access stats.
            let mut active: crate::entity::pa_microtheory_cache::ActiveModel = entry.clone().into();
            active.access_count = Set(entry.access_count + 1);
            active.last_accessed_at = Set(chrono::Utc::now().fixed_offset().into());
            let _ = active.update(self.state.db.as_ref()).await;

            let result = serde_json::json!({
                "triples": entry.triples_json,
                "source_documents": [],
                "cache_hit": true,
                "confidence": entry.confidence,
                "compartment_id": entry.compartment_id,
            });
            let json = serde_json::to_string_pretty(&result)
                .map_err(|e| McpError::internal_error(format!("serialize: {e}"), None))?;
            Ok(Some(json))
        } else {
            Ok(None)
        }
    }
}

// ── ServerHandler impl ──────────────────────────────────────────────────

#[rmcp::tool_handler]
impl rmcp::handler::server::ServerHandler for SeshatMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_03_26,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "seshat".to_string(),
                title: Some("Seshat's Archive".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: Some(
                    "Shared corpus library with RAG + pgvector for akh-medu instances".to_string(),
                ),
                icons: None,
                website_url: Some("https://akh-medu.dev".to_string()),
            },
            instructions: Some(
                "Seshat's Archive provides corpus search and knowledge synthesis. \
                 Use `seshat_search` for raw vector similarity search. \
                 Use `seshat_wish` to request a microtheory on a topic (RAG + LLM synthesis). \
                 Use `seshat_status` to check archive statistics. \
                 Use `seshat_documents` to list ingested documents. \
                 Use `seshat_embed_now` to force embedding of pending chunks."
                    .to_string(),
            ),
        }
    }
}
