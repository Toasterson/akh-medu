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
mod catalog_tools;
pub mod constrain;
pub mod dataset;
pub mod error;
pub mod eval;
pub mod format;
pub mod ground;
pub mod repair;
pub mod generator;
pub mod organ;
pub mod phrasing;
pub mod procedural;
pub mod retriever;
pub mod schema;

pub use catalog::ToolCatalog;
pub use constrain::{GrammarState, ToolCallGrammar};
pub use dataset::{AxisReport, Dataset, EvalAxis};
pub use error::{GrugError, GrugResult};
pub use eval::{
    score_abstention_fmt, AbstentionOutcome, AbstentionReport, ArgMismatch, DiversityReport,
    EvalOutcome, EvalReport, SchemaViolation,
};
pub use format::{parse_call, parse_output, render_call, render_tools, ParsedOutput, TargetFormat};
pub use ground::{ground_values, GroundStats};
pub use organ::{respond_to_gap, GapRecoveryReport, GapResponse};
pub use repair::{repair_call, RepairStats};
pub use retriever::{retrieval_recall_at_k, semantic_recall_at_k, Retriever, SemanticRetriever};
pub use generator::{DistractorConfig, Generator, PhrasingMode};
pub use phrasing::{PhrasingBank, PhrasingRegistry, PhrasingTemplate, TemplatePart};
pub use procedural::{build_catalog as build_procedural_catalog, DEFAULT_TEMPLATES_PER_TOOL};
pub use schema::{Example, ParamSpec, ParamType, ToolCall, ToolSchema};
