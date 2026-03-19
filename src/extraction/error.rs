//! Extraction error types.

use miette::Diagnostic;
use thiserror::Error;

/// Errors from the triple extraction subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum ExtractionError {
    #[error("LLM backend unavailable: {reason}")]
    #[diagnostic(
        code(akh::extraction::llm_unavailable),
        help("Ensure the nlu-llm feature is enabled and model files are present, \
              or configure an external API in [extraction.external_api].")
    )]
    LlmUnavailable { reason: String },

    #[error("failed to parse LLM output as triples: {reason}")]
    #[diagnostic(
        code(akh::extraction::parse_failed),
        help("The LLM produced output that could not be parsed as a JSON array of triples. \
              Raw output (truncated): {raw_output}")
    )]
    ParseFailed { raw_output: String, reason: String },

    #[error("external API request failed: {url} (status {status})")]
    #[diagnostic(
        code(akh::extraction::api_failed),
        help("Check the API base URL, API key (AKH_EXTRACTION_API_KEY), and model name \
              in [extraction.external_api]. Response: {body}")
    )]
    ApiRequestFailed {
        url: String,
        status: u16,
        body: String,
    },

    #[error("external API request timed out after {timeout_secs}s: {url}")]
    #[diagnostic(
        code(akh::extraction::api_timeout),
        help("Increase timeout_secs in [extraction.external_api] or switch to local LLM.")
    )]
    ApiTimeout { url: String, timeout_secs: u64 },

    #[error("{0}")]
    #[diagnostic(
        code(akh::extraction::engine),
        help("An engine-level error occurred during triple extraction.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for ExtractionError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Convenience alias.
pub type ExtractionResult<T> = std::result::Result<T, ExtractionError>;
