//! Phase 41g benchmark harness: prompt → plan → execute + Pattern B comparator.
//!
//! Drives the full NLU → ThinkingLoop (stub) → Pattern C dispatch → verbalize pipeline
//! across curated code-shaped prompts and sweeps over step budgets.  Pattern B runs
//! the same prompts through an OpenRouter thinking-mode call and scores each against
//! identical acceptance predicates.

pub mod dataset;
pub mod metrics;
pub mod pattern_b;
pub mod pipeline;

pub use dataset::{Dataset, DatasetError, load_dataset};
pub use metrics::{AgreementRecord, MetricsSummary, emit_agreement_csv, emit_csv, emit_plot_script};
pub use pattern_b::{PatternBClient, PatternBConfig, PatternBResult};
pub use pipeline::{BenchResult, HaltReason, PipelineConfig, run_pipeline};

use serde::{Deserialize, Serialize};

/// Step budgets swept by the harness (matches Phase 41f sweep points).
pub const STEP_BUDGETS: &[u64] = &[100, 300, 1_000, 3_000, 10_000];

/// Default dataset path relative to the workspace root.
pub const DATASET_PATH: &str = "data/bench/prompt_to_plan/dataset.toml";

/// Output directory for CSV and plot artefacts.
pub const OUTPUT_DIR: &str = "target/bench-results/prompt-to-plan";

// ---------------------------------------------------------------------------
// Core benchmark types
// ---------------------------------------------------------------------------

/// Prompt category, kept as a closed enum so acceptance-predicate generators
/// can branch on it without stringly-typed comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PromptCategory {
    CodeAdd,
    CodeEdit,
    Refactor,
    DocWrite,
    Debug,
}

impl std::fmt::Display for PromptCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::CodeAdd => "code-add",
            Self::CodeEdit => "code-edit",
            Self::Refactor => "refactor",
            Self::DocWrite => "doc-write",
            Self::Debug => "debug",
        };
        f.write_str(s)
    }
}

/// Ground-truth acceptance predicate for a prompt.
///
/// Evaluated against the text output produced by the pipeline run.
/// Compound predicates are joined with logical AND: all must pass for success.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AcceptancePredicate {
    /// Pipeline output must contain the given literal substring.
    OutputContains { pattern: String },
    /// The generated plan must have at least this many executed steps.
    PlanHasMinSteps { min_steps: usize },
    /// The halt reason must match. `reason` is the kebab-case variant name.
    HaltReasonIs { reason: String },
}

impl AcceptancePredicate {
    /// Evaluate the predicate given the pipeline output text and step count.
    pub fn evaluate(&self, output: &str, steps_taken: usize, halt_reason: &HaltReason) -> bool {
        match self {
            Self::OutputContains { pattern } => output.contains(pattern.as_str()),
            Self::PlanHasMinSteps { min_steps } => steps_taken >= *min_steps,
            Self::HaltReasonIs { reason } => halt_reason.as_str() == reason.as_str(),
        }
    }
}

/// A single entry in the benchmark dataset.
#[derive(Debug, Clone, Deserialize)]
pub struct BenchPrompt {
    pub id: String,
    pub category: PromptCategory,
    pub prompt: String,
    pub acceptance: Vec<AcceptancePredicate>,
}

impl BenchPrompt {
    /// Evaluate all acceptance predicates; returns true iff all pass.
    pub fn accepts(&self, output: &str, steps_taken: usize, halt_reason: &HaltReason) -> bool {
        self.acceptance
            .iter()
            .all(|pred| pred.evaluate(output, steps_taken, halt_reason))
    }
}
