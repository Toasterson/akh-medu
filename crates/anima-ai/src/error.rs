//! Error types for the anima-ai routing layer.

use miette::Diagnostic;
use thiserror::Error;

/// Errors originating from the Gemma generation backend.
#[derive(Debug, Error, Diagnostic)]
pub enum GemmaError {
    #[error("Gemma backend unavailable")]
    #[diagnostic(
        code(anima_ai::gemma::unavailable),
        help("Load a Gemma GGUF model or configure a remote generation backend.")
    )]
    Unavailable,

    #[error("semantic cache miss — no cached response for this prompt")]
    #[diagnostic(
        code(anima_ai::gemma::cache_miss),
        help("No cached response found; retry with GenerationMode::Full.")
    )]
    CacheMiss,

    #[error("Gemma inference failed: {message}")]
    #[diagnostic(
        code(anima_ai::gemma::inference_failed),
        help("Check the Gemma backend configuration and model integrity.")
    )]
    InferenceFailed { message: String },
}

/// Errors originating from the remote LLM fallback.
#[derive(Debug, Error, Diagnostic)]
pub enum RemoteError {
    #[error("remote LLM request failed: {message}")]
    #[diagnostic(
        code(anima_ai::remote::request_failed),
        help("Check network connectivity and API credentials for the remote endpoint.")
    )]
    RequestFailed { message: String },

    #[error("remote LLM backend unreachable")]
    #[diagnostic(
        code(anima_ai::remote::unavailable),
        help("Verify the remote endpoint URL and API key are correctly configured.")
    )]
    Unavailable,
}

/// Top-level error type for the anima-ai orchestration layer.
#[derive(Debug, Error, Diagnostic)]
pub enum AnimaError {
    #[error(transparent)]
    #[diagnostic(transparent)]
    Gemma(#[from] GemmaError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Remote(#[from] RemoteError),

    #[error("all NL engines exhausted for input: \"{input_preview}\"")]
    #[diagnostic(
        code(anima_ai::all_engines_exhausted),
        help(
            "Configure at least one generation backend (Gemma model path, remote API key) \
             or ensure the knowledge graph contains relevant data for ELIZA to answer from."
        )
    )]
    AllEnginesExhausted { input_preview: String },
}

/// Convenience result alias for anima-ai operations.
pub type AnimaResult<T> = std::result::Result<T, AnimaError>;

/// Result alias for Gemma backend operations.
pub type GemmaResult<T> = std::result::Result<T, GemmaError>;
