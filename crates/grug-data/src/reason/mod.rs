//! Phase 51 P3-R -- procedural multi-hop reasoning corpus with High-Grug v1
//! traces (ADR 092) and its replay-based scorer.
//!
//! Mirrors the shape of the tool-calling corpus (`crate::schema` /
//! `crate::dataset` / `crate::eval`) but for a different skill: chaining
//! facts over a synthetic knowledge graph through an *imposed* token-stream
//! workspace, rather than picking a tool call. The scorer
//! ([`replay_trace`]) is a total, never-panicking replayer -- it checks every
//! emitted statement (`recall`/`derive`/`check`/`call answer`) against the
//! example's own prompt facts, exactly as ADR 092 decision D6 specifies.
//!
//! See `docs/ai/decisions/092-high-grug-v1-reasoning-register.md` (the locked
//! trace-format design) and
//! `docs/ai/plans/2026-07-11-phase51-p3r-reasoning-corpus.md` (the corpus /
//! training spec) for the design this module implements.
//!
//! # Quick start
//!
//! ```
//! use grug_data::reason::{ReasonConfig, ReasonDataset, TraceArm, replay_trace};
//!
//! let dataset = ReasonDataset::new(ReasonConfig::default()).unwrap();
//! let train = dataset.train_examples(4, TraceArm::Trace).unwrap();
//!
//! // A gold trace replays perfectly against its own prompt.
//! let report = replay_trace(&train[0], &train[0].target);
//! assert_eq!(report.executability, 1.0);
//! assert!(report.answer_correct);
//! ```

mod corpus;
mod error;
mod handoff;
mod kg;
mod quality;
pub mod register;
mod replay;
pub mod rule;
mod score;

pub use corpus::{ReasonConfig, ReasonDataset, ReasonExample, TraceArm};
pub use error::ReasonError;
pub use handoff::{HandoffOutcome, handoff};
pub use kg::{EntityId, EntityType, Fact, RelationKind, SyntheticKg, query_name, random_path};
pub use quality::{
    TraceQualityReport, TraceQualitySummary, score_trace_quality, summarize_trace_quality,
};
pub use replay::{ReplayReport, replay_trace};
pub use score::{
    AbstentionStats, DepthStats, ReasonDiversityReport, ReasonEvalSummary, score_reason_outputs,
};

/// The bare abstention value literal used inside a `call answer | value gap`
/// trace line and a `step N | check ... | gap` result segment.
///
/// Distinct from (but philosophically the same signal as)
/// `crate::format::GRUG_GAP_MARKER`, which marks abstention at the
/// tool-calling *boundary* (a whole-output marker) rather than as a value
/// embedded inside a reasoning trace line -- ADR 092 D1 explicitly reuses the
/// `gap` mechanism rather than inventing a second one.
pub(crate) const GAP_VALUE: &str = "gap";
