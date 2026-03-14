//! NLU pipeline error types with miette diagnostics.

use miette::Diagnostic;
use thiserror::Error;

/// Result type for NLU operations.
pub type NluResult<T> = Result<T, NluError>;

/// Errors from the NLU pipeline.
#[derive(Debug, Error, Diagnostic)]
pub enum NluError {
    /// No tier could parse the input into a structured representation.
    #[error("NLU parse failed: no tier could handle \"{input}\"")]
    #[diagnostic(
        code(akh::nlu::parse_failed),
        help("The input could not be parsed by any NLU tier. It will be treated as freeform text.")
    )]
    ParseFailed { input: String },

    /// A requested NLU tier is not available (e.g., ML tier without the feature gate).
    #[error("NLU tier {tier} is not available")]
    #[diagnostic(
        code(akh::nlu::tier_unavailable),
        help(
            "Enable the corresponding feature gate: \
             tier 2 requires `nlu-ml`, tier 3 requires `nlu-llm`."
        )
    )]
    TierUnavailable { tier: u8 },

    /// The VSA parse ranker encountered an error.
    #[error("NLU ranker error: {reason}")]
    #[diagnostic(
        code(akh::nlu::ranker_error),
        help("The parse ranker's exemplar memory may be corrupted. Try resetting it.")
    )]
    RankerError { reason: String },

    /// A required model file was not found on disk.
    #[error("NLU model not found: {path}")]
    #[diagnostic(
        code(akh::nlu::model_not_found),
        help("Download with `akh setup models`")
    )]
    ModelNotFound { path: std::path::PathBuf },

    /// A model file exists but could not be loaded.
    #[error("NLU model load failed: {reason}")]
    #[diagnostic(
        code(akh::nlu::model_load),
        help("Check model file integrity")
    )]
    ModelLoadFailed { reason: String },

    /// The LLM could not produce valid structured output.
    #[error("LLM generation failed: {reason}")]
    #[diagnostic(
        code(akh::nlu::llm_generation),
        help("The LLM could not produce valid output")
    )]
    LlmGenerationFailed { reason: String },

    /// GBNF grammar initialization failed (malformed grammar file).
    #[error("GBNF grammar initialization failed: {reason}")]
    #[diagnostic(
        code(akh::nlu::grammar_init),
        help("The GBNF grammar file may be malformed. Check abstree.gbnf syntax.")
    )]
    GrammarInitFailed { reason: String },

    /// Wrapped grammar error.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Grammar(#[from] crate::grammar::error::GrammarError),

    /// Wrapped VSA error.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Vsa(#[from] crate::error::VsaError),
}
