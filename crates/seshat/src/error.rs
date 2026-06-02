//! Seshat error types with miette diagnostics.

use miette::Diagnostic;
use thiserror::Error;

/// Result alias for Seshat operations.
pub type SeshatResult<T> = Result<T, SeshatError>;

/// Errors that can occur in the Seshat service.
#[derive(Debug, Error, Diagnostic)]
pub enum SeshatError {
    /// Database connection or query failure.
    #[error("database error: {0}")]
    #[diagnostic(
        code(seshat::database),
        help("Check that PostgreSQL is running and the connection URL is correct in seshat.toml")
    )]
    Database(#[from] sea_orm::DbErr),

    /// Migration failure.
    #[error("migration error: {0}")]
    #[diagnostic(
        code(seshat::migration),
        help("Ensure the database user has CREATE TABLE privileges and pgvector extension is available")
    )]
    Migration(String),

    /// Configuration parsing error.
    #[error("config error: {0}")]
    #[diagnostic(
        code(seshat::config),
        help("Check seshat.toml syntax and required fields: [database] url, [server] bind")
    )]
    Config(String),

    /// Document not found.
    #[error("document not found: {slug}")]
    #[diagnostic(
        code(seshat::not_found),
        help("Use `seshat_documents` to list available documents")
    )]
    DocumentNotFound { slug: String },

    /// Embedding model error.
    #[error("embedding error: {0}")]
    #[diagnostic(
        code(seshat::embedding),
        help("Check that the ONNX model exists at the configured path or can be downloaded")
    )]
    Embedding(String),

    /// LLM synthesis error.
    #[error("synthesis error: {0}")]
    #[diagnostic(
        code(seshat::synthesis),
        help("Check that the GGUF model path is correct in [synthesis] config")
    )]
    Synthesis(String),

    /// MCP server error.
    #[error("MCP server error: {0}")]
    #[diagnostic(code(seshat::mcp))]
    Mcp(String),

    /// Serialization error.
    #[error("serialization error: {0}")]
    #[diagnostic(code(seshat::serde))]
    Serde(#[from] serde_json::Error),

    /// I/O error.
    #[error("I/O error: {0}")]
    #[diagnostic(code(seshat::io))]
    Io(#[from] std::io::Error),

    /// Query returned no results.
    #[error("no chunks found matching query: {query}")]
    #[diagnostic(
        code(seshat::empty_search),
        help("The corpus may not contain relevant content, or embeddings may still be processing. Check `seshat_status`.")
    )]
    EmptySearch { query: String },

    /// Cache miss (not an error per se, but useful for control flow).
    #[error("no cached microtheory for query hash: {hash}")]
    #[diagnostic(code(seshat::cache_miss))]
    CacheMiss { hash: String },
}
