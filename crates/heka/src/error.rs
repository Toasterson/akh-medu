//! Heka error types with miette diagnostics.
//!
//! Mirrors `crates/seshat/src/error.rs` — every variant carries a diagnostic
//! code and a help string so `hekad` and library consumers get actionable
//! error output via `miette`.

use miette::Diagnostic;
use thiserror::Error;

/// Result alias for Heka operations.
pub type HekaResult<T> = Result<T, HekaError>;

/// Errors that can occur in the Heka service.
#[derive(Debug, Error, Diagnostic)]
pub enum HekaError {
    /// Database connection or query failure.
    #[error("database error: {0}")]
    #[diagnostic(
        code(heka::database),
        help("Check that PostgreSQL is running and the connection URL is correct in heka.toml")
    )]
    Database(#[from] sea_orm::DbErr),

    /// Migration failure.
    #[error("migration error: {0}")]
    #[diagnostic(
        code(heka::migration),
        help(
            "Ensure the database user has CREATE TABLE privileges and the pgvector extension is available"
        )
    )]
    Migration(String),

    /// Configuration parsing error.
    #[error("config error: {0}")]
    #[diagnostic(
        code(heka::config),
        help("Check heka.toml syntax and required fields: [database] url, [server] bind")
    )]
    Config(String),

    /// Akh not found.
    #[error("akh not found: {akh_id}")]
    #[diagnostic(
        code(heka::akh::not_found),
        help("Register the akh first via the `heka_akhs` table")
    )]
    AkhNotFound { akh_id: String },

    /// Drawer not found.
    #[error("drawer not found: {drawer_id}")]
    #[diagnostic(
        code(heka::drawer::not_found),
        help("Use the `heka_drawers` table to look up the drawer UUID")
    )]
    DrawerNotFound { drawer_id: String },

    /// Embedding model error.
    #[error("embedding error: {0}")]
    #[diagnostic(
        code(heka::embedding),
        help("Check that the ONNX model exists at the configured path or can be downloaded")
    )]
    Embedding(String),

    /// Serialization error.
    #[error("serialization error: {0}")]
    #[diagnostic(code(heka::serde))]
    Serde(#[from] serde_json::Error),

    /// I/O error.
    #[error("I/O error: {0}")]
    #[diagnostic(code(heka::io))]
    Io(#[from] std::io::Error),
}
