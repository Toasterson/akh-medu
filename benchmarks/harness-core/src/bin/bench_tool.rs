//! `bench-tool`: the CLI seam for Phase 50 W7.
//!
//! Three subcommands:
//! - `record-synthetic` — write one deterministic synthetic [`RunRecord`]
//!   under `benchmarks/results/`. This is the "make the results-writing path
//!   real and callable" demonstration the W7 scope calls for: until the W3
//!   (illumos E2E), W5 (public legs), and W6 (fleet orchestration) runners
//!   exist, `just bench <harness> <model>` calls this so the
//!   schema→leaderboard→site pipeline stays end-to-end runnable.
//! - `leaderboard` — regenerate `benchmarks/LEADERBOARD.md` from every
//!   committed run.
//! - `site` — regenerate the self-contained HTML site.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use harness_core::hash::{TaskFile, hash_task_set, hash_task_set_dir};
use harness_core::schema::{
    GitSha, GradingMode, HarnessId, HostTier, JudgeInfo, ModelBacking, ModelIdentity, ModelTag, ResourceUsage,
    RunRecord, TaskId, TaskOutcome, TaskResult, TaskSetHash, WallTimeSecs,
};
use harness_core::{leaderboard, site, store};
use miette::{Context, IntoDiagnostic, Result, miette};
use serde::Deserialize;

#[derive(Parser)]
#[command(
    name = "bench-tool",
    about = "Phase 50 (W7) benchmark results: record, leaderboard, and site generation"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write one deterministic synthetic run (no real harness execution).
    /// Demonstrates the results-writing path end-to-end; real W3/W5/W6
    /// runners call `harness_core::store::write_run` directly instead of
    /// this subcommand.
    RecordSynthetic {
        /// Harness slug, e.g. `demo`, `frontiercode`, `illumos-e2e`.
        #[arg(long)]
        harness: String,
        /// Model tag slug, e.g. `deepseek`, `glm-5.2`, `qwen`.
        #[arg(long)]
        model: String,
        /// Where `benchmarks/results/` lives.
        #[arg(long, default_value = "benchmarks/results")]
        results_dir: PathBuf,
        /// Deterministic PRNG seed (same seed + task count = same scores).
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Number of synthetic tasks to fabricate.
        #[arg(long, default_value_t = 5)]
        tasks: usize,
    },
    /// Record ONE real run from a harness-produced JSON task-row file
    /// (see [`RawTaskRow`] for the shape). This is the seam real W3/W5/W6
    /// runners (including the Python harnesses under `benchmarks/`) use to
    /// turn their own results format into a schema-validated, provenance-
    /// stamped [`RunRecord`] without each one re-implementing the ADR 086 §4
    /// contract by hand.
    RecordRun {
        /// Harness slug, e.g. `frontiercode`, `locomo`, `longmemeval`.
        #[arg(long)]
        harness: String,
        /// Model tag slug (filesystem-safe grouping key), e.g. `deepseek`,
        /// `glm-5.2`, `qwen`.
        #[arg(long)]
        model: String,
        /// The model's own provider tag as sent to the cloud API, e.g.
        /// `deepseek-v4-pro`, `glm-5.2`, `qwen3.5:397b`.
        #[arg(long)]
        provider_tag: String,
        /// Path to a JSON array of [`RawTaskRow`] — the harness's per-task
        /// results, already run.
        #[arg(long)]
        tasks_json: PathBuf,
        /// Directory whose full recursive content is hashed to produce the
        /// task-set hash (e.g. the harness's `tasks/` dir, or a dataset
        /// subset directory prepared for this run).
        #[arg(long)]
        task_set_dir: PathBuf,
        /// Where `benchmarks/results/` lives.
        #[arg(long, default_value = "benchmarks/results")]
        results_dir: PathBuf,
        /// Grading mode: `autonomous` or `advisory`.
        #[arg(long, default_value = "autonomous")]
        grading: String,
        /// Judge model tag slug — required when `--grading advisory`.
        #[arg(long)]
        judge_model: Option<String>,
        /// Judge rotation seed — required when `--grading advisory`.
        #[arg(long, default_value_t = 0)]
        rotation_seed: u64,
        /// Host tier: only `host` is supported by this subcommand (VM-tier
        /// runs carry image identity and should call
        /// `harness_core::store::write_run` directly instead).
        #[arg(long, default_value = "host")]
        host_tier: String,
        /// Repo checkout to stamp `harness_git_sha` from (defaults to the
        /// current directory).
        #[arg(long)]
        repo_dir: Option<PathBuf>,
    },
    /// Regenerate `benchmarks/LEADERBOARD.md` from every committed run.
    Leaderboard {
        #[arg(long, default_value = "benchmarks/results")]
        results_dir: PathBuf,
        #[arg(long, default_value = "benchmarks/LEADERBOARD.md")]
        out: PathBuf,
    },
    /// Regenerate the self-contained HTML site.
    Site {
        #[arg(long, default_value = "benchmarks/results")]
        results_dir: PathBuf,
        #[arg(long, default_value = "benchmarks/site/index.html")]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::RecordSynthetic {
            harness,
            model,
            results_dir,
            seed,
            tasks,
        } => record_synthetic(&harness, &model, &results_dir, seed, tasks),
        Cmd::RecordRun {
            harness,
            model,
            provider_tag,
            tasks_json,
            task_set_dir,
            results_dir,
            grading,
            judge_model,
            rotation_seed,
            host_tier,
            repo_dir,
        } => record_run(
            &harness,
            &model,
            &provider_tag,
            &tasks_json,
            &task_set_dir,
            &results_dir,
            &grading,
            judge_model.as_deref(),
            rotation_seed,
            &host_tier,
            repo_dir.as_deref(),
        ),
        Cmd::Leaderboard { results_dir, out } => run_leaderboard(&results_dir, &out),
        Cmd::Site { results_dir, out } => run_site(&results_dir, &out),
    }
}

/// A tiny deterministic PRNG built on blake3 (already a dependency) rather
/// than pulling in `rand` for a handful of fabricated demo numbers: hash
/// `(seed, index)` and take the low 53 bits as a `[0.0, 1.0)` float.
fn synthetic_unit(seed: u64, index: usize, salt: &str) -> f64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&seed.to_le_bytes());
    hasher.update(&(index as u64).to_le_bytes());
    hasher.update(salt.as_bytes());
    let hash = hasher.finalize();
    let bytes: [u8; 8] = hash.as_bytes()[0..8].try_into().expect("8 bytes");
    let v = u64::from_le_bytes(bytes) >> 11; // 53 significant bits
    (v as f64) / ((1u64 << 53) as f64)
}

fn record_synthetic(harness: &str, model: &str, results_dir: &PathBuf, seed: u64, n_tasks: usize) -> Result<()> {
    let harness_id = HarnessId::new(harness).into_diagnostic().wrap_err("invalid --harness slug")?;
    let model_tag = ModelTag::new(model).into_diagnostic().wrap_err("invalid --model slug")?;

    let task_files: Vec<TaskFile> = (0..n_tasks.max(1))
        .map(|i| TaskFile {
            relative_path: format!("synthetic-task-{i:03}/prompt.md"),
            content: format!("synthetic task {i} for harness {harness}").into_bytes(),
        })
        .collect();
    let task_set_hash: TaskSetHash = hash_task_set(&task_files);

    let harness_git_sha = GitSha::from_repo(&std::env::current_dir().into_diagnostic()?)
        .unwrap_or_else(|_| GitSha::new("0".repeat(40)).expect("all-zero SHA is valid hex"));

    let mut tasks = Vec::with_capacity(n_tasks.max(1));
    for i in 0..n_tasks.max(1) {
        let score = synthetic_unit(seed, i, "score");
        let passed = score >= 0.5;
        let wall_time = 1.0 + synthetic_unit(seed, i, "wall_time") * 9.0;
        let tokens = 200 + (synthetic_unit(seed, i, "tokens") * 4000.0) as u64;
        let task = TaskResult::new(
            TaskId::new(format!("synthetic-{i:03}")).into_diagnostic()?,
            if passed {
                TaskOutcome::Pass
            } else {
                TaskOutcome::Fail
            },
            score,
            WallTimeSecs(wall_time),
            ResourceUsage {
                tokens_used: Some(tokens),
                requests_used: Some(1),
            },
            Some("synthetic demo row — not a real benchmark result".to_string()),
        )
        .into_diagnostic()
        .wrap_err("constructing synthetic TaskResult")?;
        tasks.push(task);
    }

    let record = RunRecord::new(
        chrono::Utc::now(),
        harness_id,
        harness_git_sha,
        task_set_hash,
        ModelIdentity {
            tag: model_tag,
            backing: ModelBacking::Cloud {
                provider_tag: format!("{model}-synthetic-demo"),
            },
        },
        GradingMode::Autonomous,
        HostTier::Host,
        tasks,
    );

    let dir = store::write_run(results_dir, &record)
        .into_diagnostic()
        .wrap_err("writing synthetic run")?;
    println!("wrote synthetic run to {}", dir.display());
    Ok(())
}

/// One task row as produced by an external harness (Rust or Python), read
/// from the `--tasks-json` file as a JSON array of these. Deliberately
/// looser/simpler than [`harness_core::schema::TaskResult`]: `outcome` is a
/// plain string (`"pass"` / `"fail"` / `"aborted"`) rather than a tagged enum,
/// so a Python harness can emit it with `json.dumps` and no knowledge of
/// serde's internal representation.
#[derive(Debug, Deserialize)]
struct RawTaskRow {
    task_id: String,
    /// `"pass"`, `"fail"`, or `"aborted"`.
    outcome: String,
    /// Required when `outcome == "aborted"` (ADR 086 §5: a budget-cap abort
    /// counts as failure, but the reason is recorded).
    #[serde(default)]
    abort_reason: Option<String>,
    score: f64,
    wall_time_secs: f64,
    #[serde(default)]
    tokens_used: Option<u64>,
    #[serde(default)]
    requests_used: Option<u64>,
    #[serde(default)]
    notes: Option<String>,
}

fn record_run(
    harness: &str,
    model: &str,
    provider_tag: &str,
    tasks_json: &PathBuf,
    task_set_dir: &PathBuf,
    results_dir: &PathBuf,
    grading: &str,
    judge_model: Option<&str>,
    rotation_seed: u64,
    host_tier: &str,
    repo_dir: Option<&std::path::Path>,
) -> Result<()> {
    let harness_id = HarnessId::new(harness).into_diagnostic().wrap_err("invalid --harness slug")?;
    let model_tag = ModelTag::new(model).into_diagnostic().wrap_err("invalid --model slug")?;

    let raw_json = std::fs::read_to_string(tasks_json)
        .into_diagnostic()
        .wrap_err_with(|| format!("reading {}", tasks_json.display()))?;
    let raw_rows: Vec<RawTaskRow> = serde_json::from_str(&raw_json)
        .into_diagnostic()
        .wrap_err_with(|| format!("parsing {} as a JSON array of task rows", tasks_json.display()))?;
    if raw_rows.is_empty() {
        return Err(miette!("{} contains zero task rows — nothing to record", tasks_json.display()));
    }

    let mut tasks = Vec::with_capacity(raw_rows.len());
    for row in raw_rows {
        let outcome = match row.outcome.as_str() {
            "pass" => TaskOutcome::Pass,
            "fail" => TaskOutcome::Fail,
            "aborted" => TaskOutcome::Aborted {
                reason: row.abort_reason.clone().unwrap_or_else(|| "unspecified".to_string()),
            },
            other => {
                return Err(miette!(
                    "task {:?} has unknown outcome {:?} (expected pass/fail/aborted)",
                    row.task_id,
                    other
                ));
            }
        };
        let task = TaskResult::new(
            TaskId::new(row.task_id.clone()).into_diagnostic()?,
            outcome,
            row.score,
            WallTimeSecs(row.wall_time_secs),
            ResourceUsage {
                tokens_used: row.tokens_used,
                requests_used: row.requests_used,
            },
            row.notes,
        )
        .into_diagnostic()
        .wrap_err_with(|| format!("constructing TaskResult for task {:?}", row.task_id))?;
        tasks.push(task);
    }

    let task_set_hash: TaskSetHash = hash_task_set_dir(task_set_dir)
        .into_diagnostic()
        .wrap_err_with(|| format!("hashing task-set dir {}", task_set_dir.display()))?;

    let repo_dir = repo_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().expect("current dir"));
    let harness_git_sha = GitSha::from_repo(&repo_dir)
        .into_diagnostic()
        .wrap_err_with(|| format!("reading HEAD sha from {}", repo_dir.display()))?;

    let grading_mode = match grading {
        "autonomous" => GradingMode::Autonomous,
        "advisory" => {
            let judge_model = judge_model
                .ok_or_else(|| miette!("--grading advisory requires --judge-model"))?;
            let judge_tag = ModelTag::new(judge_model)
                .into_diagnostic()
                .wrap_err("invalid --judge-model slug")?;
            GradingMode::Advisory(JudgeInfo {
                judge_model: judge_tag,
                rotation_seed,
            })
        }
        other => return Err(miette!("unknown --grading {:?} (expected autonomous|advisory)", other)),
    };

    let host_tier = match host_tier {
        "host" => HostTier::Host,
        other => {
            return Err(miette!(
                "unsupported --host-tier {:?} for record-run (only \"host\" is supported here; \
                 VM-tier runs should call harness_core::store::write_run directly so they can \
                 supply VmImageIdentity)",
                other
            ));
        }
    };

    let record = RunRecord::new(
        chrono::Utc::now(),
        harness_id,
        harness_git_sha,
        task_set_hash,
        ModelIdentity {
            tag: model_tag,
            backing: ModelBacking::Cloud {
                provider_tag: provider_tag.to_string(),
            },
        },
        grading_mode,
        host_tier,
        tasks,
    );

    let dir = store::write_run(results_dir, &record)
        .into_diagnostic()
        .wrap_err("writing run")?;
    let summary = record.summary();
    println!(
        "wrote run to {} ({}/{} passed, mean score {:.3})",
        dir.display(),
        summary.passed,
        summary.total_tasks,
        summary.mean_score
    );
    Ok(())
}

fn run_leaderboard(results_dir: &PathBuf, out: &PathBuf) -> Result<()> {
    let runs = store::scan_results(results_dir)
        .into_diagnostic()
        .wrap_err_with(|| format!("scanning {}", results_dir.display()))?;
    let latest = store::latest_per_pair(&runs);
    let md = leaderboard::render_leaderboard_markdown(&latest);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).into_diagnostic()?;
    }
    std::fs::write(out, &md)
        .into_diagnostic()
        .wrap_err_with(|| format!("writing {}", out.display()))?;
    println!("wrote leaderboard ({} pair(s)) to {}", latest.len(), out.display());
    Ok(())
}

fn run_site(results_dir: &PathBuf, out: &PathBuf) -> Result<()> {
    let runs = store::scan_results(results_dir)
        .into_diagnostic()
        .wrap_err_with(|| format!("scanning {}", results_dir.display()))?;
    let latest = store::latest_per_pair(&runs);
    let html = site::render_html_site(&latest, &runs);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).into_diagnostic()?;
    }
    std::fs::write(out, &html)
        .into_diagnostic()
        .wrap_err_with(|| format!("writing {}", out.display()))?;
    println!(
        "wrote site ({} pair(s), {} total run(s)) to {}",
        latest.len(),
        runs.len(),
        out.display()
    );
    Ok(())
}
