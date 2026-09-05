//! Evaluation harness for long-term memory benchmarks.
//!
//! LongMemEval uses the Phase 42c drawer, embedding, and pgvector pipeline.
//! LoCoMo remains a baseline harness for a later integration phase.
//!
//! # Datasets
//!
//! - **LongMemEval**: multi-session conversational-memory evidence retrieval.
//!   Phase 42c implements `heka eval longmemeval --mode raw`; unsupported modes
//!   are refused rather than mislabeled.
//! - **LoCoMo**: long conversational memory benchmark. Run with
//!   `heka eval locomo --top-k auto`.
//!
//! # Report format
//!
//! The reporter emits both JSON (machine-readable) and Markdown (human-readable)
//! summaries capturing hardware, model versions, dataset hashes, exact retrieval
//! parameters, and p50/p99 latency.

pub mod locomo;
pub mod longmemeval;
pub mod reporter;

use clap::Subcommand;

/// Evaluation subcommands.
#[derive(Debug, Clone, Subcommand)]
pub enum EvalCommand {
    /// Run the LongMemEval benchmark suite.
    Longmemeval {
        /// Retrieval mode. Phase 42c supports only `raw` vector retrieval;
        /// `hybrid` is rejected until a symbolic reranker exists.
        #[arg(long, default_value = "raw")]
        mode: String,

        /// Number of non-abstention benchmark questions to evaluate.
        #[arg(long, default_value_t = 10)]
        sessions: usize,

        /// Path to the dataset directory (defaults to `eval_data/longmemeval`).
        #[arg(long)]
        data_dir: Option<String>,

        /// Output directory for reports (defaults to `eval_reports/`).
        #[arg(long)]
        output: Option<String>,
    },

    /// Run the LoCoMo benchmark suite.
    Locomo {
        /// Top-K retrieval limit. `auto` selects `min(5, conv_len/4)` per session
        /// to prevent the MemPalace `top_k=50`-on-30-items exploit by construction.
        #[arg(long, default_value = "auto")]
        top_k: String,

        /// Explicit top-K override. Emits a loud warning in the report since it
        /// bypasses the auto-cap safety guard.
        #[arg(long)]
        top_k_explicit: Option<usize>,

        /// Number of sessions to evaluate (default: 10 for smoke test).
        #[arg(long, default_value_t = 10)]
        sessions: usize,

        /// Path to the dataset directory (defaults to `eval_data/locomo`).
        #[arg(long)]
        data_dir: Option<String>,

        /// Output directory for reports (defaults to `eval_reports/`).
        #[arg(long)]
        output: Option<String>,
    },
}

/// Result of a single retrieval query.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RetrievalResult {
    /// Whether this benchmark's binary retrieval target was found in top-K.
    pub hit: bool,
    /// Number of items retrieved.
    pub retrieved_count: usize,
    /// Latency in seconds for this single retrieval.
    pub latency_secs: f64,
}

/// Aggregated metrics for a single evaluation run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EvalMetrics {
    /// Total number of questions evaluated.
    pub total_questions: usize,
    /// Benchmark-specific retrieval score with an explicit semantic label.
    pub scores: EvalScores,
    /// p50 latency in seconds.
    pub p50_latency_secs: f64,
    /// p99 latency in seconds.
    pub p99_latency_secs: f64,
    /// Exact retrieval parameters used.
    pub retrieval_params: RetrievalParams,
}

/// Explicit metric semantics. Evidence-session retrieval is not answer
/// accuracy and is kept distinct from a generic binary retrieval hit rate.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvalScores {
    BinaryRetrieval {
        hits: usize,
        recall_at_k: f64,
    },
    EvidenceSessionRetrieval {
        retrieved_any_at_5: usize,
        retrieved_all_at_5: usize,
        #[serde(rename = "recall_any@5")]
        recall_any_at_5: f64,
        #[serde(rename = "recall_all@5")]
        recall_all_at_5: f64,
    },
}

/// Exact retrieval parameters recorded for reproducibility.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RetrievalParams {
    /// Mode: "raw" or "hybrid" (LongMemEval) or "locomo" (LoCoMo).
    pub mode: String,
    /// Top-K value used. "auto" means the harness computed it per session.
    pub top_k: String,
    /// If top_k_explicit was set, the explicit value.
    pub top_k_explicit: Option<usize>,
    /// If auto top-k was used, the computed cap per session.
    pub auto_top_k_cap: Option<usize>,
}

/// Run an evaluation command.
pub async fn run(cmd: &EvalCommand) -> miette::Result<()> {
    match cmd {
        EvalCommand::Longmemeval {
            mode,
            sessions,
            data_dir,
            output,
        } => longmemeval::run(mode, *sessions, data_dir.as_deref(), output.as_deref()).await,
        EvalCommand::Locomo {
            top_k,
            top_k_explicit,
            sessions,
            data_dir,
            output,
        } => {
            locomo::run(
                top_k,
                *top_k_explicit,
                *sessions,
                data_dir.as_deref(),
                output.as_deref(),
            )
            .await
        }
    }
}
