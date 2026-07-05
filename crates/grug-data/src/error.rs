//! Rich diagnostic error types for `grug-data`.
//!
//! Every fallible operation in this crate returns [`GrugResult<T>`] so callers
//! (the eval harness, the training loop, ad-hoc scripts) get a miette
//! diagnostic with a `help()` string instead of a bare error message.

use miette::Diagnostic;
use thiserror::Error;

/// Errors from example generation, (de)serialization, and eval scoring.
#[derive(Debug, Error, Diagnostic)]
pub enum GrugError {
    #[error("tool catalog is empty; cannot generate examples")]
    #[diagnostic(
        code(grug_data::empty_catalog),
        help(
            "`Generator` needs at least one training tool and one holdout tool. \
             Use `ToolCatalog::default()` for the built-in dozen-tool catalog, \
             or add schemas via `ToolCatalog::push`."
        )
    )]
    EmptyCatalog,

    #[error("holdout split leaves no tools for training: requested {holdout} of {total}")]
    #[diagnostic(
        code(grug_data::degenerate_split),
        help(
            "The whole-tool holdout split must reserve strictly fewer tools than \
             the catalog contains, and leave at least one tool for training. \
             Lower `holdout_count` or grow the catalog."
        )
    )]
    DegenerateSplit { holdout: usize, total: usize },

    #[error("failed to serialize example to JSONL: {message}")]
    #[diagnostic(
        code(grug_data::serialize_failed),
        help("This indicates a bug in `Example`'s data shape — please report it.")
    )]
    SerializeFailed { message: String },

    #[error("failed to parse JSONL line into an example: {message}")]
    #[diagnostic(
        code(grug_data::deserialize_failed),
        help(
            "The line is not valid JSON, or does not match the `Example` schema. \
             Check that the file was produced by `Example::to_jsonl`."
        )
    )]
    DeserializeFailed { message: String },
}

/// Convenience alias for `grug-data` operation results.
pub type GrugResult<T> = std::result::Result<T, GrugError>;
