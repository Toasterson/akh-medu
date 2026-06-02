//! ELIZA-specific error types with rich miette diagnostics.
//!
//! Combines the VSA pattern-matching engine errors (PR #173) and the Seshat
//! script-store / hot-reloader errors (PR #179) into a single subsystem enum.

use miette::Diagnostic;
use thiserror::Error;

#[derive(Debug, Error, Diagnostic)]
pub enum ElizaError {
    // ── VSA pattern-matching engine (PR #173) ─────────────────────────────────
    #[error("failed to parse ELIZA script: {reason}")]
    #[diagnostic(
        code(akh::eliza::parse),
        help(
            "Check that the TOML syntax is valid and all required fields are present. \
             Each [[rules]] section must have 'keywords' and 'responses' lists."
        )
    )]
    ScriptParse { reason: String },

    #[error("rule at index {index} has no keywords")]
    #[diagnostic(
        code(akh::eliza::no_keywords),
        help("Each [[rules]] section must have at least one entry in the 'keywords' list.")
    )]
    NoKeywords { index: usize },

    #[error("rule at index {index} has no responses")]
    #[diagnostic(
        code(akh::eliza::no_responses),
        help("Each [[rules]] section must have at least one entry in the 'responses' list.")
    )]
    NoResponses { index: usize },

    #[error("VSA operation failed: {message}")]
    #[diagnostic(
        code(akh::eliza::vsa),
        help(
            "An internal vector operation failed. \
             This is usually caused by a dimension mismatch in the keyword encoder."
        )
    )]
    Vsa { message: String },

    // ── KG-grounded response templates (PR #174) ──────────────────────────────
    #[error("template parse error: {message}")]
    #[diagnostic(
        code(akh::eliza::template_parse),
        help(
            "Check the template syntax. Use {{N}} for regex capture groups (1-indexed) \
             and {{fact_N:type:entity}} for KG facts. \
             Types: 'triple', 'vsa' (optionally ':k' for top-k), 'competence'."
        )
    )]
    TemplateParse { message: String },

    // ── Seshat script store + hot-reloader (PR #179) ──────────────────────────
    #[error("Seshat HTTP error: {0}")]
    #[diagnostic(
        code(eliza::seshat_http),
        help(
            "Check that the Seshat service is reachable at the configured URL and \
             the /scripts endpoint is available. Start seshd if it is not running."
        )
    )]
    SeshatHttp(String),

    #[error("Seshat returned HTTP {status}: {body}")]
    #[diagnostic(
        code(eliza::seshat_status),
        help(
            "Seshat returned a non-2xx response. \
             Check the Seshat server logs for details. \
             Ensure the script name is URL-safe (alphanumeric + hyphens/underscores)."
        )
    )]
    SeshatStatus { status: u16, body: String },

    #[error("script not found: '{name}'")]
    #[diagnostic(
        code(eliza::not_found),
        help(
            "Use `SeshatScriptStore::list()` to see available scripts, \
             or save one first with `SeshatScriptStore::save(name, content)`."
        )
    )]
    NotFound { name: String },

    #[error("deserialization error from Seshat response: {0}")]
    #[diagnostic(
        code(eliza::serde),
        help(
            "Failed to deserialize the response from Seshat. \
             Check that the running Seshat version matches what the client expects."
        )
    )]
    Serde(String),

    #[error("I/O error for script '{name}': {source}")]
    #[diagnostic(
        code(eliza::io),
        help("A filesystem operation failed while accessing local script files.")
    )]
    Io {
        name: String,
        #[source]
        source: std::io::Error,
    },

    #[error("cache lock poisoned")]
    #[diagnostic(
        code(eliza::cache_poison),
        help(
            "The internal script cache lock was poisoned due to a panic in another thread. \
             Restart the process to recover."
        )
    )]
    CachePoisoned,
}

impl From<crate::error::VsaError> for ElizaError {
    fn from(e: crate::error::VsaError) -> Self {
        ElizaError::Vsa { message: e.to_string() }
    }
}

/// Result alias for ELIZA operations.
pub type ElizaResult<T> = Result<T, ElizaError>;
