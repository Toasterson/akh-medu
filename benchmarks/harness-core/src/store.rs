//! Storage layout: `benchmarks/results/<date>-<harness>-<model>/{run.jsonl,summary.md}`.
//!
//! `run.jsonl` is the machine-readable source of truth (one [`RunLine`] JSON
//! object per line: one `header`, N `task` rows, one trailing `summary`).
//! `summary.md` is a human-readable rendering of the same data, regenerated
//! from `run.jsonl` — never hand-edited.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{HarnessError, HarnessResult};
use crate::schema::{HarnessId, ModelTag, RunHeader, RunLine, RunRecord, TaskResult};

/// `(directory, error)` pairs collected by [`scan_results_lenient`] for runs
/// that failed to parse.
type ScanFailures = Vec<(PathBuf, HarnessError)>;

const RUN_FILE: &str = "run.jsonl";
const SUMMARY_FILE: &str = "summary.md";

fn to_json_line(what: &'static str, line: &RunLine) -> HarnessResult<String> {
    serde_json::to_string(line).map_err(|source| HarnessError::Serialize { what, source })
}

/// Write `record` under `results_root`, creating
/// `<date>-<harness>-<model>/run.jsonl` and `.../summary.md`. Returns the
/// directory written to. Overwrites an existing directory of the same name
/// (re-running the same harness/model on the same day replaces that day's
/// result — callers that want to preserve history should not re-run same-day,
/// matching the "manual cadence" model in ADR 086 §5).
pub fn write_run(results_root: &Path, record: &RunRecord) -> HarnessResult<PathBuf> {
    let dir = results_root.join(record.result_dir_name());
    fs::create_dir_all(&dir).map_err(|source| HarnessError::Io {
        path: dir.clone(),
        source,
    })?;

    let run_path = dir.join(RUN_FILE);
    let mut body = String::new();
    body.push_str(&to_json_line("RunLine::Header", &RunLine::Header(record.header.clone()))?);
    body.push('\n');
    for task in &record.tasks {
        body.push_str(&to_json_line("RunLine::Task", &RunLine::Task(task.clone()))?);
        body.push('\n');
    }
    body.push_str(&to_json_line(
        "RunLine::Summary",
        &RunLine::Summary(record.summary()),
    )?);
    body.push('\n');
    fs::write(&run_path, body).map_err(|source| HarnessError::Io {
        path: run_path.clone(),
        source,
    })?;

    let summary_path = dir.join(SUMMARY_FILE);
    fs::write(&summary_path, render_summary_markdown(record)).map_err(|source| HarnessError::Io {
        path: summary_path,
        source,
    })?;

    Ok(dir)
}

/// Parse `<dir>/run.jsonl` back into a [`RunRecord`]. The stored trailing
/// `summary` line is read (to fail loudly on a corrupt file) but discarded —
/// `RunRecord::summary()` recomputes it from `tasks`, which is always the
/// source of truth.
pub fn read_run(dir: &Path) -> HarnessResult<RunRecord> {
    let run_path = dir.join(RUN_FILE);
    if !run_path.is_file() {
        return Err(HarnessError::MissingRunFile { path: dir.to_path_buf() });
    }
    let content = fs::read_to_string(&run_path).map_err(|source| HarnessError::Io {
        path: run_path.clone(),
        source,
    })?;

    let mut header: Option<RunHeader> = None;
    let mut tasks: Vec<TaskResult> = Vec::new();

    for (idx, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed: RunLine = serde_json::from_str(line).map_err(|e| HarnessError::ParseLine {
            path: run_path.clone(),
            line_no: idx + 1,
            message: e.to_string(),
        })?;
        match parsed {
            RunLine::Header(h) => {
                if header.is_some() {
                    return Err(HarnessError::DuplicateHeader { path: run_path.clone() });
                }
                header = Some(h);
            }
            RunLine::Task(t) => tasks.push(t),
            RunLine::Summary(_) => {
                // Informational only; recomputed via RunRecord::summary().
            }
        }
    }

    let header = header.ok_or_else(|| HarnessError::MissingHeader { path: run_path.clone() })?;
    Ok(RunRecord { header, tasks })
}

/// Scan every immediate subdirectory of `results_root` for a `run.jsonl` and
/// parse it. Strict: the first unreadable/corrupt run directory aborts the
/// whole scan (fail loudly rather than silently produce a partial
/// leaderboard). Use [`scan_results_lenient`] when a caller would rather
/// skip bad directories and report them separately (e.g. an interactive
/// `bench-tool` run that wants to warn but still render what it can).
pub fn scan_results(results_root: &Path) -> HarnessResult<Vec<RunRecord>> {
    let mut out = Vec::new();
    for dir in result_subdirs(results_root)? {
        out.push(read_run(&dir)?);
    }
    Ok(out)
}

/// Like [`scan_results`], but never aborts: returns the runs that parsed
/// successfully alongside `(directory, error)` pairs for the ones that
/// didn't.
pub fn scan_results_lenient(results_root: &Path) -> HarnessResult<(Vec<RunRecord>, ScanFailures)> {
    let mut ok = Vec::new();
    let mut errs = Vec::new();
    for dir in result_subdirs(results_root)? {
        match read_run(&dir) {
            Ok(record) => ok.push(record),
            Err(e) => errs.push((dir, e)),
        }
    }
    Ok((ok, errs))
}

fn result_subdirs(results_root: &Path) -> HarnessResult<Vec<PathBuf>> {
    if !results_root.is_dir() {
        return Ok(Vec::new());
    }
    let mut dirs = Vec::new();
    for entry in fs::read_dir(results_root).map_err(|source| HarnessError::Io {
        path: results_root.to_path_buf(),
        source,
    })? {
        let entry = entry.map_err(|source| HarnessError::Io {
            path: results_root.to_path_buf(),
            source,
        })?;
        if entry.path().is_dir() {
            dirs.push(entry.path());
        }
    }
    dirs.sort();
    Ok(dirs)
}

/// For each `(harness, model tag)` pair, keep only the most recent run by
/// timestamp — the definition of "the leaderboard" per ADR 086 §4.
pub fn latest_per_pair(runs: &[RunRecord]) -> BTreeMap<(HarnessId, ModelTag), RunRecord> {
    let mut latest: BTreeMap<(HarnessId, ModelTag), RunRecord> = BTreeMap::new();
    for run in runs {
        let key = (run.header.harness.clone(), run.header.model.tag.clone());
        match latest.get(&key) {
            Some(existing) if existing.header.timestamp >= run.header.timestamp => {}
            _ => {
                latest.insert(key, run.clone());
            }
        }
    }
    latest
}

/// Render the human-readable `summary.md` for a single run.
pub fn render_summary_markdown(record: &RunRecord) -> String {
    let summary = record.summary();
    let mut out = String::new();

    let _ = writeln!(
        out,
        "# {} × {} — {}",
        record.header.harness,
        record.header.model.tag,
        record.header.timestamp.format("%Y-%m-%d %H:%M UTC")
    );
    out.push('\n');

    let _ = writeln!(out, "- **Harness git SHA**: `{}`", record.header.harness_git_sha);
    let _ = writeln!(out, "- **Task-set hash**: `{}`", record.header.task_set_hash);
    match &record.header.model.backing {
        crate::schema::ModelBacking::Cloud { provider_tag } => {
            let _ = writeln!(out, "- **Model backing**: cloud (`{provider_tag}`)");
        }
        crate::schema::ModelBacking::LocalGguf { file_name, file_hash } => {
            let _ = writeln!(out, "- **Model backing**: local GGUF `{file_name}` (`{file_hash}`)", file_hash = file_hash.as_str());
        }
    }
    match &record.header.grading {
        crate::schema::GradingMode::Autonomous => {
            let _ = writeln!(out, "- **Grading**: autonomous (deterministic `verify.sh` only, no LLM judge)");
        }
        crate::schema::GradingMode::Advisory(judge) => {
            let _ = writeln!(
                out,
                "- **Grading**: advisory — judge `{}`, rotation seed `{}`",
                judge.judge_model, judge.rotation_seed
            );
        }
    }
    match &record.header.host_tier {
        crate::schema::HostTier::Host => {
            let _ = writeln!(out, "- **Host tier**: host");
        }
        crate::schema::HostTier::Vm(image) => {
            let _ = writeln!(
                out,
                "- **Host tier**: VM (`{}`, recipe `{}`)",
                image.registry,
                image.recipe_hash.as_str()
            );
        }
    }
    out.push('\n');

    let _ = write!(
        out,
        "**{}/{} tasks passed ({:.1}%), mean score {:.3}, {:.1}s total wall time**",
        summary.passed,
        summary.total_tasks,
        summary.pass_rate * 100.0,
        summary.mean_score,
        summary.total_wall_time_secs,
    );
    if let Some(tok) = summary.total_tokens {
        let _ = write!(out, ", {tok} tokens");
    }
    if let Some(req) = summary.total_requests {
        let _ = write!(out, ", {req} requests");
    }
    out.push_str("\n\n");

    out.push_str("| Task | Outcome | Score | Wall time (s) | Tokens | Requests | Notes |\n");
    out.push_str("|---|---|---|---|---|---|---|\n");
    for t in &record.tasks {
        let outcome = match &t.outcome {
            crate::schema::TaskOutcome::Pass => "pass".to_string(),
            crate::schema::TaskOutcome::Fail => "fail".to_string(),
            crate::schema::TaskOutcome::Aborted { reason } => format!("aborted ({reason})"),
        };
        let _ = writeln!(
            out,
            "| {} | {} | {:.3} | {:.1} | {} | {} | {} |",
            t.task_id,
            outcome,
            t.score,
            t.wall_time.0,
            t.usage.tokens_used.map(|v| v.to_string()).unwrap_or_default(),
            t.usage.requests_used.map(|v| v.to_string()).unwrap_or_default(),
            t.notes.as_deref().unwrap_or(""),
        );
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{
        GradingMode, HostTier, ModelBacking, ModelIdentity, ResourceUsage, TaskId, TaskOutcome, TaskSetHash,
        WallTimeSecs,
    };
    use crate::schema::{Blake3Hex, GitSha};

    fn sample_record(harness: &str, model: &str, day: &str) -> RunRecord {
        RunRecord::new(
            format!("{day}T00:00:00Z").parse().unwrap(),
            HarnessId::new(harness).unwrap(),
            GitSha::new("a".repeat(40)).unwrap(),
            TaskSetHash::from_blake3(Blake3Hex::from_hash(blake3::hash(b"tasks"))),
            ModelIdentity {
                tag: ModelTag::new(model).unwrap(),
                backing: ModelBacking::Cloud {
                    provider_tag: format!("{model}-cloud"),
                },
            },
            GradingMode::Autonomous,
            HostTier::Host,
            vec![
                TaskResult::new(
                    TaskId::new("t1").unwrap(),
                    TaskOutcome::Pass,
                    1.0,
                    WallTimeSecs(2.0),
                    ResourceUsage {
                        tokens_used: Some(50),
                        requests_used: Some(1),
                    },
                    None,
                )
                .unwrap(),
                TaskResult::new(
                    TaskId::new("t2").unwrap(),
                    TaskOutcome::Fail,
                    0.0,
                    WallTimeSecs(1.0),
                    ResourceUsage::none(),
                    Some("timed out".to_string()),
                )
                .unwrap(),
            ],
        )
    }

    #[test]
    fn write_then_read_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let record = sample_record("demo", "deepseek", "2026-07-05");
        let written_dir = write_run(dir.path(), &record).unwrap();
        assert_eq!(written_dir.file_name().unwrap(), "2026-07-05-demo-deepseek");
        assert!(written_dir.join("run.jsonl").is_file());
        assert!(written_dir.join("summary.md").is_file());

        let read_back = read_run(&written_dir).unwrap();
        assert_eq!(read_back.header, record.header);
        assert_eq!(read_back.tasks, record.tasks);
        assert_eq!(read_back.summary().total_tasks, 2);
    }

    #[test]
    fn missing_run_file_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let err = read_run(dir.path()).unwrap_err();
        assert!(matches!(err, HarnessError::MissingRunFile { .. }));
    }

    #[test]
    fn scan_results_finds_all_written_runs() {
        let dir = tempfile::tempdir().unwrap();
        write_run(dir.path(), &sample_record("demo", "deepseek", "2026-07-01")).unwrap();
        write_run(dir.path(), &sample_record("demo", "glm", "2026-07-02")).unwrap();
        let runs = scan_results(dir.path()).unwrap();
        assert_eq!(runs.len(), 2);
    }

    #[test]
    fn latest_per_pair_keeps_only_the_most_recent_run() {
        let older = sample_record("demo", "deepseek", "2026-07-01");
        let newer = sample_record("demo", "deepseek", "2026-07-05");
        let other_model = sample_record("demo", "glm", "2026-07-03");
        let latest = latest_per_pair(&[older.clone(), newer.clone(), other_model.clone()]);

        assert_eq!(latest.len(), 2);
        let key = (HarnessId::new("demo").unwrap(), ModelTag::new("deepseek").unwrap());
        assert_eq!(latest.get(&key).unwrap().header.timestamp, newer.header.timestamp);
    }

    #[test]
    fn scan_results_on_missing_root_is_empty_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("does-not-exist");
        let runs = scan_results(&missing).unwrap();
        assert!(runs.is_empty());
    }
}
