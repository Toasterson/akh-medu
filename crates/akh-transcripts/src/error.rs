//! Error type for `akh-transcripts`, following the project's miette + thiserror
//! convention: every variant carries a diagnostic `code` and actionable `help`.

use miette::Diagnostic;
use thiserror::Error;

pub type TranscriptResult<T> = Result<T, TranscriptError>;

#[derive(Debug, Error, Diagnostic)]
pub enum TranscriptError {
    // Filled in by the schema/pool and adapter increments. Keep variants
    // specific: one per failure a caller can act on differently.
    #[error("placeholder — replace during the first increment")]
    #[diagnostic(code(akh_transcripts::placeholder))]
    Placeholder,
}
