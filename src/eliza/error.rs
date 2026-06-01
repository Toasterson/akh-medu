//! Error types for the ELIZA script subsystem.

use miette::Diagnostic;
use thiserror::Error;

/// Errors produced by the ELIZA script store and hot-reloader.
#[derive(Debug, Error, Diagnostic)]
pub enum ElizaError {
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

/// Result alias for ELIZA operations.
pub type ElizaResult<T> = Result<T, ElizaError>;
