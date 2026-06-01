//! ELIZA-specific error types with rich miette diagnostics.

use miette::Diagnostic;
use thiserror::Error;

#[derive(Debug, Error, Diagnostic)]
pub enum ElizaError {
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
}

impl From<crate::error::VsaError> for ElizaError {
    fn from(e: crate::error::VsaError) -> Self {
        ElizaError::Vsa { message: e.to_string() }
    }
}

pub type ElizaResult<T> = Result<T, ElizaError>;
