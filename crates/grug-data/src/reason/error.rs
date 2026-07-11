//! Errors specific to the Phase 51 P3-R procedural reasoning corpus.

use miette::Diagnostic;
use thiserror::Error;

/// Errors from [`super::ReasonDataset`] construction and example generation.
/// Chained transparently into [`crate::error::GrugError`] via `#[from]`, so
/// callers can use `?` against `crate::error::GrugResult` uniformly.
#[derive(Debug, Error, Diagnostic)]
pub enum ReasonError {
    #[error("ReasonConfig.train_depths is empty; need at least one training depth")]
    #[diagnostic(
        code(grug_data::reason::empty_train_depths),
        help(
            "Set `ReasonConfig.train_depths` to a non-empty `Vec`, e.g. `vec![2, 3, 4]` (the default)."
        )
    )]
    EmptyTrainDepths,

    #[error("depth 0 is not a valid reasoning depth (need >= 1 hop)")]
    #[diagnostic(
        code(grug_data::reason::zero_depth),
        help(
            "Every depth in `ReasonConfig.train_depths`, and any `depths` slice \
             passed to `ReasonDataset::axis_a_examples`, must be >= 1 -- a \
             depth-0 query has no relation to chain."
        )
    )]
    ZeroDepth,

    #[error("ReasonDataset::axis_a_examples was called with an empty `depths` slice")]
    #[diagnostic(
        code(grug_data::reason::empty_axis_a_depths),
        help(
            "Pass at least one depth, e.g. `&[5, 6]` -- the Phase 51 P3-R \
             length-generalization axis (depths never seen in `train_depths`)."
        )
    )]
    EmptyAxisADepths,

    #[error("entities_per_type ({got}) is too small; need at least {min}")]
    #[diagnostic(
        code(grug_data::reason::entities_per_type_too_small),
        help(
            "Raise `ReasonConfig.entities_per_type` -- it must be large enough \
             to hold both a non-empty TRAIN and non-empty HOLDOUT subject pool \
             per entity type, given `holdout_subject_frac`."
        )
    )]
    EntitiesPerTypeTooSmall { got: usize, min: usize },
}
