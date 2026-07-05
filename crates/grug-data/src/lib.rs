//! `grug-data` — Phase 51 (Grug-Brain BabyLLM) data generator + eval scorer.
//!
//! Pure Rust, no machine-learning dependency. Two jobs:
//!
//! 1. **Generate** deterministic, seeded JSON tool-calling training examples
//!    with owned ground truth (see [`Generator`], [`ToolCatalog`],
//!    [`Dataset`]).
//! 2. **Score** a model's raw output string against an example's ground
//!    truth, gated level-by-level, plus an anti-collapse diversity alarm
//!    (see [`eval::score_output`], [`eval::EvalReport`],
//!    [`eval::DiversityReport`]).
//!
//! # Quick start
//!
//! ```
//! use grug_data::{Dataset, eval};
//!
//! let dataset = Dataset::new(42).expect("built-in catalog splits cleanly");
//! let train = dataset.train_examples(100).unwrap();
//! let holdout = dataset.holdout_examples(20).unwrap();
//!
//! // Hand a model `train[i].user_request` + `train[i].tools`, get back a
//! // raw string, score it:
//! let raw = train[0].expected_call_as_json_string();
//! let outcome = eval::score_output(&raw, &train[0]);
//! assert!(outcome.arg_correct());
//! ```

pub mod catalog;
pub mod dataset;
pub mod error;
pub mod eval;
pub mod generator;
pub mod schema;

pub use catalog::ToolCatalog;
pub use dataset::Dataset;
pub use error::{GrugError, GrugResult};
pub use eval::{ArgMismatch, DiversityReport, EvalOutcome, EvalReport, SchemaViolation};
pub use generator::Generator;
pub use schema::{Example, ParamSpec, ParamType, ToolCall, ToolSchema};
