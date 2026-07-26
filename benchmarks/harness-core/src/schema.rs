//! `RunRecord`: the self-describing schema for one benchmark run.
//!
//! This is the ADR 086 §4 "results discipline" contract, expressed in types
//! rather than a comment. Every field listed in that section as required
//! provenance is a *required* (non-`Option`) field here, or is folded into an
//! enum variant that makes the absent case structurally distinct from the
//! present case (e.g. `GradingMode::Autonomous` vs. `GradingMode::Advisory`,
//! `HostTier::Host` vs. `HostTier::Vm`) rather than a `None` that could be
//! "forgot to fill this in" or "genuinely doesn't apply".
//!
//! ADR 086 §4 provenance mapping:
//!
//! | ADR 086 §4 field                         | Type here                              |
//! |-------------------------------------------|-----------------------------------------|
//! | harness git SHA                            | [`RunHeader::harness_git_sha`] : [`GitSha`] |
//! | task-set content hash                      | [`RunHeader::task_set_hash`] : [`TaskSetHash`] |
//! | model tag + backing (cloud tag / GGUF hash)| [`RunHeader::model`] : [`ModelIdentity`] |
//! | judge model + rotation seed                | [`RunHeader::grading`] : [`GradingMode::Advisory`] |
//! | VM image identity (registry/recipe hash)   | [`RunHeader::host_tier`] : [`HostTier::Vm`] |
//! | host tier                                  | [`RunHeader::host_tier`] : [`HostTier`] |
//! | per-task rows (id, score, pass/fail, wall time, tokens/requests) | [`TaskResult`] |
//! | aggregate scores                           | [`RunSummary`] (derived via [`RunRecord::summary`]) |

use std::fmt;
use std::path::Path;
use std::process::Command;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{HarnessError, HarnessResult};

/// Current on-disk schema version. Bump when `RunLine`'s shape changes in a
/// way that breaks older readers; old result directories keep their
/// originally-written version so `store::read_run` can branch on it if a
/// future version needs migration.
pub const SCHEMA_VERSION: u32 = 1;

fn is_slug_char(c: char) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.' || c == '_'
}

fn validate_slug(field: &'static str, value: &str) -> HarnessResult<()> {
    if !value.is_empty() && value.chars().all(is_slug_char) {
        Ok(())
    } else {
        Err(HarnessError::InvalidSlug {
            field,
            value: value.to_string(),
        })
    }
}

fn validate_hex(field: &'static str, value: &str, expected_len: usize) -> HarnessResult<()> {
    let ok = value.len() == expected_len
        && value
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase());
    if ok {
        Ok(())
    } else {
        Err(HarnessError::InvalidHex {
            field,
            value: value.to_string(),
            expected_len,
        })
    }
}

// ---------------------------------------------------------------------
// Newtypes
// ---------------------------------------------------------------------

/// A blake3 digest, hex-encoded (64 lowercase hex chars). The raw building
/// block for the three distinct hash roles below — kept as its own type so
/// [`TaskSetHash`], [`GgufFileHash`], and [`VmRecipeHash`] can each wrap it
/// without being interchangeable with one another (a task-set hash and a
/// GGUF-file hash are never comparable, even though both happen to be
/// blake3 digests).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Blake3Hex(String);

impl Blake3Hex {
    pub fn new(value: impl Into<String>) -> HarnessResult<Self> {
        let value = value.into();
        validate_hex("blake3_hex", &value, 64)?;
        Ok(Self(value))
    }

    pub fn from_hash(hash: blake3::Hash) -> Self {
        Self(hash.to_hex().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// First 12 hex chars, for compact leaderboard/site display.
    pub fn short(&self) -> &str {
        &self.0[..12]
    }
}

impl fmt::Display for Blake3Hex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// The task-set content hash. Freezes on publish per ADR 086 §4: once a run
/// referencing a given hash is committed, task-set edits must produce a new
/// hash (a "v2" task series), not silently mutate history under the same id.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskSetHash(Blake3Hex);

impl TaskSetHash {
    pub fn from_blake3(hash: Blake3Hex) -> Self {
        Self(hash)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn short(&self) -> &str {
        self.0.short()
    }
}

impl fmt::Display for TaskSetHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.0, f)
    }
}

/// Content hash of a local GGUF weights file, proving *which* weights a
/// self-trained (or quantized) model run actually used.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GgufFileHash(Blake3Hex);

impl GgufFileHash {
    pub fn from_blake3(hash: Blake3Hex) -> Self {
        Self(hash)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn short(&self) -> &str {
        self.0.short()
    }
}

/// Content hash of a VM image recipe (or a registry digest), identifying
/// exactly which OpenIndiana/OmniOS image a VM-tier run executed against.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct VmRecipeHash(Blake3Hex);

impl VmRecipeHash {
    pub fn from_blake3(hash: Blake3Hex) -> Self {
        Self(hash)
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    pub fn short(&self) -> &str {
        self.0.short()
    }
}

/// A git commit SHA (7..=40 lowercase hex chars — short or full).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GitSha(String);

impl GitSha {
    pub fn new(value: impl Into<String>) -> HarnessResult<Self> {
        let value = value.into();
        let ok = (7..=40).contains(&value.len())
            && value
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase());
        if ok {
            Ok(Self(value))
        } else {
            Err(HarnessError::InvalidGitSha {
                field: "harness_git_sha",
                value,
            })
        }
    }

    /// Shell out to `git rev-parse HEAD` in `repo_dir`. Convenience for
    /// runners that want to stamp the harness SHA automatically rather than
    /// threading it through from CI/CLI.
    pub fn from_repo(repo_dir: &Path) -> HarnessResult<Self> {
        let output = Command::new("git")
            .arg("-C")
            .arg(repo_dir)
            .arg("rev-parse")
            .arg("HEAD")
            .output()
            .map_err(|source| HarnessError::Io {
                path: repo_dir.to_path_buf(),
                source,
            })?;
        if !output.status.success() {
            return Err(HarnessError::GitRevParseFailed {
                repo_dir: repo_dir.to_path_buf(),
                status: output.status.to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            });
        }
        let sha = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Self::new(sha)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn short(&self) -> &str {
        &self.0[..self.0.len().min(12)]
    }
}

impl fmt::Display for GitSha {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Filesystem-safe harness identifier (e.g. `frontiercode`, `illumos-e2e`,
/// `locomo`, `fleet-orchestration`). Becomes a path component under
/// `benchmarks/results/`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HarnessId(String);

impl HarnessId {
    pub fn new(value: impl Into<String>) -> HarnessResult<Self> {
        let value = value.into();
        validate_slug("harness", &value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HarnessId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Filesystem-safe model tag (e.g. `deepseek-v3.2`, `glm-5.2`, `qwen3-max`,
/// or a self-trained tag like `grug-v1`). Becomes a path component under
/// `benchmarks/results/` and the leaderboard's grouping key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelTag(String);

impl ModelTag {
    pub fn new(value: impl Into<String>) -> HarnessResult<Self> {
        let value = value.into();
        validate_slug("model_tag", &value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Slug identifying one task within a task set (e.g. `smf-001`,
/// `swe-bench-django-16255`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TaskId(String);

impl TaskId {
    pub fn new(value: impl Into<String>) -> HarnessResult<Self> {
        let value = value.into();
        validate_slug("task_id", &value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------
// Model identity + backing
// ---------------------------------------------------------------------

/// How a model was served for this run. The two variants are mutually
/// exclusive by construction: a cloud-served model has a provider tag and no
/// weights file to hash; a self-hosted GGUF model has a content hash and no
/// provider tag. There is no `Option<hash>` for callers to leave `None` by
/// accident.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "backing", rename_all = "snake_case")]
pub enum ModelBacking {
    /// Rented frontier model over a cloud API (e.g. ollama-cloud), identified
    /// by the provider's own tag string (may differ from `ModelIdentity::tag`,
    /// which is the filesystem-safe slug used for grouping).
    Cloud { provider_tag: String },
    /// Locally-run GGUF weights (Candle backend), identified by content hash
    /// so "which weights, exactly" survives file renames.
    LocalGguf {
        file_name: String,
        file_hash: GgufFileHash,
    },
}

/// The model under test: a stable grouping tag plus exactly how it was
/// served.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub tag: ModelTag,
    pub backing: ModelBacking,
}

// ---------------------------------------------------------------------
// Grading mode (judge model + rotation seed)
// ---------------------------------------------------------------------

/// Rubric-grading judge identity, required only in [`GradingMode::Advisory`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JudgeInfo {
    /// Which model graded this run's transcripts. Per ADR 086 §2, this must
    /// never equal the model under test ("no model grades its own
    /// transcript") — `store`/CLI callers are expected to enforce that
    /// invariant before constructing a `RunRecord`; it is documented, not
    /// re-checked here, since the rotation policy lives with the caller's
    /// model roster, not with this schema.
    pub judge_model: ModelTag,
    /// Seed controlling judge rotation across a benchmark cycle, so a given
    /// cycle's rotation assignment is reproducible.
    pub rotation_seed: u64,
}

/// Autonomous (deterministic-only, `verify.sh` grades, no LLM ever sees the
/// outcome) vs. advisory (engineer-simulator + rotation LLM judge) grading,
/// per ADR 086 §2. Modeled as an enum rather than `Option<JudgeInfo>` so
/// "this run used no judge" and "forgot to record the judge" cannot be
/// confused.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum GradingMode {
    Autonomous,
    Advisory(JudgeInfo),
}

// ---------------------------------------------------------------------
// Host tier (+ VM image identity)
// ---------------------------------------------------------------------

/// Identity of the VM image a `Vm`-tier run executed against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VmImageIdentity {
    /// Package registry the image was pulled from (e.g.
    /// `code.aopc.cloud/CloudNebulaProject`).
    pub registry: String,
    pub recipe_hash: VmRecipeHash,
}

/// Which tier of the two-tier feature matrix (ADR 086 §1) executed this run.
/// `Vm` carries the image identity inline — there is no way to construct a
/// `Vm` tier without one, matching the ADR's requirement that VM-tier runs
/// stamp image identity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "tier", rename_all = "snake_case")]
pub enum HostTier {
    /// Linux dev-box tier: full default features + GPU/candle + nlu-ml/ORT.
    Host,
    /// `vmctl`-spun OpenIndiana/OmniOS guest tier.
    Vm(VmImageIdentity),
}

// ---------------------------------------------------------------------
// Per-task rows
// ---------------------------------------------------------------------

/// Outcome of one task. `Aborted` is a distinct variant (not folded into
/// `Fail`) so budget-cap aborts (ADR 086 §5: "abort counts as failure") stay
/// attributable to a cause in the record, while [`TaskOutcome::is_pass`]
/// still gives callers the simple pass/fail view for aggregation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum TaskOutcome {
    Pass,
    Fail,
    /// Stopped early by a per-run budget cap (tokens/requests/wall-clock).
    Aborted { reason: String },
}

impl TaskOutcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, TaskOutcome::Pass)
    }
}

/// Resource consumption for one task. Both fields are `Option` because
/// different harnesses meter different things: a pure local-CPU harness may
/// report neither, a cloud-API harness typically reports both, and a
/// token-metered-only backend reports just `tokens_used`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct ResourceUsage {
    pub tokens_used: Option<u64>,
    pub requests_used: Option<u64>,
}

impl ResourceUsage {
    pub const fn none() -> Self {
        Self {
            tokens_used: None,
            requests_used: None,
        }
    }
}

/// Wall-clock time for one task, in seconds. A newtype (rather than a bare
/// `f64`) so a caller cannot pass milliseconds by accident where seconds are
/// expected.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WallTimeSecs(pub f64);

impl WallTimeSecs {
    pub fn from_duration(d: std::time::Duration) -> Self {
        Self(d.as_secs_f64())
    }
}

/// One task's result row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub outcome: TaskOutcome,
    /// Normalized score in `0.0..=1.0` (0 = full failure, 1 = full pass).
    /// Binary pass/fail tasks use exactly 0.0 or 1.0; rubric-graded tasks may
    /// use partial credit.
    pub score: f64,
    pub wall_time: WallTimeSecs,
    pub usage: ResourceUsage,
    pub notes: Option<String>,
}

impl TaskResult {
    pub fn new(
        task_id: TaskId,
        outcome: TaskOutcome,
        score: f64,
        wall_time: WallTimeSecs,
        usage: ResourceUsage,
        notes: Option<String>,
    ) -> HarnessResult<Self> {
        if !(0.0..=1.0).contains(&score) || score.is_nan() {
            return Err(HarnessError::ScoreOutOfRange {
                task_id: task_id.as_str().to_string(),
                value: score,
            });
        }
        Ok(Self {
            task_id,
            outcome,
            score,
            wall_time,
            usage,
            notes,
        })
    }
}

// ---------------------------------------------------------------------
// Aggregate
// ---------------------------------------------------------------------

/// Aggregate scores for a run, derived (never hand-constructed as ground
/// truth) from its `tasks` via [`RunSummary::compute`] / [`RunRecord::summary`].
/// A copy is written as the trailing line of `run.jsonl` for the convenience
/// of readers that only want the aggregate, but [`crate::store::read_run`]
/// treats `tasks` as the source of truth and does not trust the stored copy.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct RunSummary {
    pub total_tasks: usize,
    pub passed: usize,
    pub failed: usize,
    pub aborted: usize,
    pub pass_rate: f64,
    pub mean_score: f64,
    pub total_wall_time_secs: f64,
    pub total_tokens: Option<u64>,
    pub total_requests: Option<u64>,
}

impl RunSummary {
    pub fn compute(tasks: &[TaskResult]) -> Self {
        let total_tasks = tasks.len();
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut aborted = 0usize;
        let mut score_sum = 0.0f64;
        let mut wall_time_sum = 0.0f64;
        let mut total_tokens: Option<u64> = None;
        let mut total_requests: Option<u64> = None;

        for t in tasks {
            match &t.outcome {
                TaskOutcome::Pass => passed += 1,
                TaskOutcome::Fail => failed += 1,
                TaskOutcome::Aborted { .. } => aborted += 1,
            }
            score_sum += t.score;
            wall_time_sum += t.wall_time.0;
            if let Some(tok) = t.usage.tokens_used {
                total_tokens = Some(total_tokens.unwrap_or(0) + tok);
            }
            if let Some(req) = t.usage.requests_used {
                total_requests = Some(total_requests.unwrap_or(0) + req);
            }
        }

        let pass_rate = if total_tasks == 0 {
            0.0
        } else {
            passed as f64 / total_tasks as f64
        };
        let mean_score = if total_tasks == 0 {
            0.0
        } else {
            score_sum / total_tasks as f64
        };

        Self {
            total_tasks,
            passed,
            failed,
            aborted,
            pass_rate,
            mean_score,
            total_wall_time_secs: wall_time_sum,
            total_tokens,
            total_requests,
        }
    }
}

// ---------------------------------------------------------------------
// Header + line-oriented file format + combined in-memory record
// ---------------------------------------------------------------------

/// Every ADR 086 §4 provenance field, everything in `run.jsonl` *except* the
/// per-task rows. Written once as the first line of `run.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunHeader {
    pub schema_version: u32,
    pub timestamp: DateTime<Utc>,
    pub harness: HarnessId,
    pub harness_git_sha: GitSha,
    pub task_set_hash: TaskSetHash,
    pub model: ModelIdentity,
    pub grading: GradingMode,
    pub host_tier: HostTier,
}

/// One line of `run.jsonl`. Each variant is a complete, independently
/// parseable JSON object (tagged by `"kind"`), so the format is
/// "self-describing" in the sense ADR 086 §4 asks for: a line does not need
/// surrounding context to be understood.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RunLine {
    Header(RunHeader),
    Task(TaskResult),
    Summary(RunSummary),
}

/// The in-memory combined view of a run: header provenance plus task rows.
/// This is what callers build up and pass to [`crate::store::write_run`];
/// [`crate::store::read_run`] reconstructs the same shape from `run.jsonl`.
#[derive(Debug, Clone, PartialEq)]
pub struct RunRecord {
    pub header: RunHeader,
    pub tasks: Vec<TaskResult>,
}

impl RunRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        timestamp: DateTime<Utc>,
        harness: HarnessId,
        harness_git_sha: GitSha,
        task_set_hash: TaskSetHash,
        model: ModelIdentity,
        grading: GradingMode,
        host_tier: HostTier,
        tasks: Vec<TaskResult>,
    ) -> Self {
        Self {
            header: RunHeader {
                schema_version: SCHEMA_VERSION,
                timestamp,
                harness,
                harness_git_sha,
                task_set_hash,
                model,
                grading,
                host_tier,
            },
            tasks,
        }
    }

    /// Aggregate scores, recomputed from `tasks` every call so it can never
    /// drift from the per-task rows.
    pub fn summary(&self) -> RunSummary {
        RunSummary::compute(&self.tasks)
    }

    /// The `benchmarks/results/<date>-<harness>-<model>/` directory-name
    /// component for this run (date in UTC, `YYYY-MM-DD`).
    pub fn result_dir_name(&self) -> String {
        format!(
            "{}-{}-{}",
            self.header.timestamp.format("%Y-%m-%d"),
            self.header.harness.as_str(),
            self.header.model.tag.as_str(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_task(id: &str, pass: bool, score: f64) -> TaskResult {
        TaskResult::new(
            TaskId::new(id).unwrap(),
            if pass {
                TaskOutcome::Pass
            } else {
                TaskOutcome::Fail
            },
            score,
            WallTimeSecs(1.5),
            ResourceUsage {
                tokens_used: Some(100),
                requests_used: Some(1),
            },
            None,
        )
        .unwrap()
    }

    #[test]
    fn blake3_hex_round_trip_and_validation() {
        let hash = blake3::hash(b"hello world");
        let h = Blake3Hex::from_hash(hash);
        assert_eq!(h.as_str().len(), 64);
        assert_eq!(h.short().len(), 12);

        let reparsed = Blake3Hex::new(h.as_str().to_string()).unwrap();
        assert_eq!(reparsed, h);

        assert!(Blake3Hex::new("too-short").is_err());
        assert!(Blake3Hex::new("A".repeat(64)).is_err()); // uppercase rejected
    }

    #[test]
    fn git_sha_accepts_short_and_full_hex() {
        assert!(GitSha::new("774181c").is_ok());
        assert!(GitSha::new("a".repeat(40)).is_ok());
        assert!(GitSha::new("a".repeat(6)).is_err()); // too short
        assert!(GitSha::new("a".repeat(41)).is_err()); // too long
        assert!(GitSha::new("ZZZZZZZ").is_err()); // not hex
    }

    #[test]
    fn slug_validation_rejects_unsafe_chars() {
        assert!(HarnessId::new("frontiercode").is_ok());
        assert!(HarnessId::new("illumos-e2e").is_ok());
        assert!(HarnessId::new("").is_err());
        assert!(HarnessId::new("has spaces").is_err());
        assert!(HarnessId::new("Has/Slash").is_err());
        assert!(ModelTag::new("deepseek-v3.2").is_ok());
        assert!(TaskId::new("smf_001").is_ok());
    }

    #[test]
    fn task_result_rejects_out_of_range_score() {
        let ok = TaskResult::new(
            TaskId::new("t1").unwrap(),
            TaskOutcome::Pass,
            1.0,
            WallTimeSecs(0.1),
            ResourceUsage::none(),
            None,
        );
        assert!(ok.is_ok());

        let too_high = TaskResult::new(
            TaskId::new("t1").unwrap(),
            TaskOutcome::Pass,
            1.5,
            WallTimeSecs(0.1),
            ResourceUsage::none(),
            None,
        );
        assert!(too_high.is_err());

        let nan = TaskResult::new(
            TaskId::new("t1").unwrap(),
            TaskOutcome::Fail,
            f64::NAN,
            WallTimeSecs(0.1),
            ResourceUsage::none(),
            None,
        );
        assert!(nan.is_err());
    }

    #[test]
    fn run_summary_aggregates_pass_rate_and_usage() {
        let tasks = vec![
            sample_task("t1", true, 1.0),
            sample_task("t2", false, 0.0),
            sample_task("t3", true, 0.8),
        ];
        let summary = RunSummary::compute(&tasks);
        assert_eq!(summary.total_tasks, 3);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.aborted, 0);
        assert!((summary.pass_rate - (2.0 / 3.0)).abs() < 1e-9);
        assert!((summary.mean_score - (1.0 + 0.0 + 0.8) / 3.0).abs() < 1e-9);
        assert_eq!(summary.total_tokens, Some(300));
        assert_eq!(summary.total_requests, Some(3));
    }

    #[test]
    fn run_summary_empty_task_list_is_zeroed_not_nan() {
        let summary = RunSummary::compute(&[]);
        assert_eq!(summary.total_tasks, 0);
        assert_eq!(summary.pass_rate, 0.0);
        assert_eq!(summary.mean_score, 0.0);
    }

    #[test]
    fn run_record_result_dir_name_is_date_harness_model() {
        let record = RunRecord::new(
            "2026-07-05T00:00:00Z".parse().unwrap(),
            HarnessId::new("demo").unwrap(),
            GitSha::new("774181cabcd1234000000000000000000000000").unwrap(),
            TaskSetHash::from_blake3(Blake3Hex::from_hash(blake3::hash(b"tasks"))),
            ModelIdentity {
                tag: ModelTag::new("deepseek").unwrap(),
                backing: ModelBacking::Cloud {
                    provider_tag: "deepseek-v3.2-cloud".to_string(),
                },
            },
            GradingMode::Autonomous,
            HostTier::Host,
            vec![sample_task("t1", true, 1.0)],
        );
        assert_eq!(record.result_dir_name(), "2026-07-05-demo-deepseek");
        assert_eq!(record.summary().total_tasks, 1);
    }
}
