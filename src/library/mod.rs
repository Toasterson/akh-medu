//! Shared content library for ingesting books, websites, and documents.
//!
//! Documents are parsed into structural elements (chapters, sections, paragraphs),
//! stored as KG symbols with well-known `doc:*` predicates, enriched with NLP
//! extraction, and embedded via VSA for semantic search.
//!
//! Each document gets its own compartment (`library:{slug}`) that can be
//! mounted by any workspace.

pub mod catalog;
pub mod chunker;
pub mod error;
pub mod inbox;
pub mod ingest;
pub mod model;
pub mod parser;
pub mod predicates;

pub use catalog::LibraryCatalog;
pub use error::{LibraryError, LibraryResult};
pub use ingest::{IngestConfig, IngestResult, ingest_document, ingest_file, ingest_url};
pub use model::{ContentFormat, DocumentRecord, DocumentSource};
pub use predicates::LibraryPredicates;

// ---------------------------------------------------------------------------
// Wire types for HTTP boundary (client ↔ server serialization)
// ---------------------------------------------------------------------------

use serde::{Deserialize, Serialize};

/// Request body for `POST /library` (add a document).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryAddRequest {
    pub source: String,
    pub title: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub format: Option<String>,
}

/// Response body for `POST /library` (add a document).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibraryAddResponse {
    pub id: String,
    pub title: String,
    pub format: String,
    pub chunk_count: usize,
    pub triple_count: usize,
}

/// Request body for `POST /library/search`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibrarySearchRequest {
    pub query: String,
    #[serde(default = "default_search_top_k")]
    pub top_k: usize,
}

fn default_search_top_k() -> usize {
    5
}

/// Single result from a library semantic search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LibrarySearchResult {
    pub rank: usize,
    pub symbol_label: String,
    pub similarity: f32,
}
