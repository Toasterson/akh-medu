//! Rich diagnostic error types for `harness-core`.
//!
//! Every fallible operation in this crate returns [`HarnessResult<T>`] so the
//! `bench-tool` CLI (and, later, the W3/W5/W6 per-harness runners) surface a
//! miette diagnostic with a `help()` string instead of a bare error message.

use std::path::PathBuf;

use miette::Diagnostic;
use thiserror::Error;

/// Errors from schema validation, run-file storage, and site generation.
#[derive(Debug, Error, Diagnostic)]
pub enum HarnessError {
    #[error(
        "`{field}` must be a filesystem-safe slug (lowercase ascii alphanumerics, `-`, `.`, `_` only), got {value:?}"
    )]
    #[diagnostic(
        code(harness_core::invalid_slug),
        help(
            "Slugs become directory-name components under `benchmarks/results/`. \
             Use lowercase letters, digits, `-`, `.`, and `_` only — e.g. \
             `frontiercode`, `deepseek-v3.2`, `smf_001`."
        )
    )]
    InvalidSlug { field: &'static str, value: String },

    #[error("`{field}` must be a lowercase hex string of exactly {expected_len} characters, got {value:?}")]
    #[diagnostic(
        code(harness_core::invalid_hex),
        help(
            "This field stores a content hash (blake3, 64 hex chars) and is compared \
             byte-for-byte across runs. Recompute it with `hash::hash_task_set_dir` \
             (or the matching helper) rather than hand-typing it."
        )
    )]
    InvalidHex {
        field: &'static str,
        value: String,
        expected_len: usize,
    },

    #[error("`{field}` must be a lowercase hex string between 7 and 40 characters (a git SHA), got {value:?}")]
    #[diagnostic(
        code(harness_core::invalid_git_sha),
        help(
            "Pass the output of `git rev-parse HEAD` (or `--short`), or use \
             `GitSha::from_repo` to read it directly from the harness checkout."
        )
    )]
    InvalidGitSha { field: &'static str, value: String },

    #[error("score {value} for task {task_id:?} is outside the valid range 0.0..=1.0")]
    #[diagnostic(
        code(harness_core::score_out_of_range),
        help(
            "Normalize partial credit into 0.0 (full failure) ..= 1.0 (full pass) \
             before constructing a `TaskResult`. A binary pass/fail task should use \
             0.0 or 1.0 exactly."
        )
    )]
    ScoreOutOfRange { task_id: String, value: f64 },

    #[error("run directory {path} is missing `run.jsonl`")]
    #[diagnostic(
        code(harness_core::missing_run_file),
        help(
            "Every run directory under `benchmarks/results/` must contain a \
             `run.jsonl` written by `store::write_run`. If this directory is a \
             work-in-progress, move it out of `benchmarks/results/` until it's ready."
        )
    )]
    MissingRunFile { path: PathBuf },

    #[error("run.jsonl at {path} has no header line")]
    #[diagnostic(
        code(harness_core::missing_header),
        help(
            "The first line of `run.jsonl` must be a `{{\"kind\":\"header\",...}}` \
             record. Regenerate the file with `store::write_run` rather than \
             hand-editing it."
        )
    )]
    MissingHeader { path: PathBuf },

    #[error("run.jsonl at {path} has more than one header line")]
    #[diagnostic(
        code(harness_core::duplicate_header),
        help(
            "Only one header line is allowed per run file; this usually means two \
             runs were concatenated by mistake. Split them into separate result \
             directories."
        )
    )]
    DuplicateHeader { path: PathBuf },

    #[error("failed to parse line {line_no} of {path}: {message}")]
    #[diagnostic(
        code(harness_core::parse_line),
        help(
            "Each line of `run.jsonl` must be one JSON object matching `RunLine`. \
             Check for hand-edits, truncated writes, or a schema-version mismatch."
        )
    )]
    ParseLine {
        path: PathBuf,
        line_no: usize,
        message: String,
    },

    #[error("io error at {path}")]
    #[diagnostic(
        code(harness_core::io),
        help("Check that the path exists, and that permissions allow the operation.")
    )]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to serialize {what} to JSON")]
    #[diagnostic(
        code(harness_core::serialize),
        help("This indicates a bug in the schema types — please report it.")
    )]
    Serialize {
        what: &'static str,
        #[source]
        source: serde_json::Error,
    },

    #[error("`git rev-parse HEAD` in {repo_dir} exited with {status}: {stderr}")]
    #[diagnostic(
        code(harness_core::git_rev_parse_failed),
        help(
            "Confirm `{repo_dir:?}` is inside a git worktree with at least one commit. \
             If git itself is unavailable, construct a `GitSha` directly from a \
             known-good SHA instead of calling `GitSha::from_repo`."
        )
    )]
    GitRevParseFailed {
        repo_dir: PathBuf,
        status: String,
        stderr: String,
    },
}

/// Convenience alias for `harness-core` operation results.
pub type HarnessResult<T> = std::result::Result<T, HarnessError>;
