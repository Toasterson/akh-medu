//! Compile feedback tool: run `cargo check` with JSON output, parse diagnostics,
//! and return structured error information for the iterative refinement loop.
//!
//! Implements a CEGIS-like (Counter-Example Guided Inductive Synthesis) loop:
//! each compiler error is a counterexample that guides the next generation attempt.
//!
//! ## Supervision and progress-based liveness
//!
//! As of this implementation, `compile_feedback` uses **progress-based process
//! supervision** instead of an elapsed-time timeout. The process is killed only
//! if it goes silent (no stdout/stderr output) for the configured duration
//! (default 600s, tunable via `AKH_COMPILE_FEEDBACK_TIMEOUT_SECS` or the
//! `timeout` parameter). A build that continuously emits output is never killed,
//! however long it runs.
//!
//! This replaces the historic elapsed-time timeout (which would kill a cold
//! build at 600s wall time even if it was healthy and emitting diagnostics).
//! The old timeout confused three distinct deadlines and made slow-but-healthy
//! builds indistinguishable from hung ones. The new supervision judges liveness
//! by **progress** (output received), not by wall time.
//!
//! ### Other timeouts still in play
//!
//! Two other timeouts exist outside this tool's control:
//!
//! 1. **`acp_adapter.py --timeout`** — the *client-side* ACP adapter's
//!    transport deadline (default 180s). This is an RPC-level timeout on the
//!    Python→Rust ACP session; when it fires, the session ends and the operator
//!    sees it in logs, not as a tool result.
//! 2. **ACP turn idle watchdog** — [`DEFAULT_TURN_IDLE_TIMEOUT_SECS`] (300s) in
//!    `src/acp/mod.rs`, which aborts turns that emit no events (tool calls,
//!    answers, audit records) for N seconds. A healthy turn streams events; a
//!    completely silent LLM loop trips this. The incremental stdout drain in
//!    this tool keeps the watchdog fed (each read is visible as progress), so
//!    a slow build does not trigger it.
//!
//! The new supervision makes (1) the tool's own concern (output-based liveness,
//! reported as `STUCK:` when it fires), while (2) and (3) remain transport and
//! turn-level guards that should never fire on a healthy, progress-making build.
//!
//! [`DEFAULT_TURN_IDLE_TIMEOUT_SECS`]: crate::acp::DEFAULT_TURN_IDLE_TIMEOUT_SECS

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::backend::{ProgressTracker, SupervisedExecution, SupervisorError};
use crate::engine::Engine;

// ---------------------------------------------------------------------------
// Diagnostic types
// ---------------------------------------------------------------------------

/// A single parsed compiler diagnostic.
#[derive(Debug, Clone)]
pub struct CompilerDiagnostic {
    /// Error level: "error", "warning", "note", "help".
    pub level: String,
    /// The error message.
    pub message: String,
    /// Error code if available (e.g., "E0308").
    pub code: Option<String>,
    /// File path where the error occurred.
    pub file: Option<String>,
    /// Line number.
    pub line: Option<u32>,
    /// Column number.
    pub column: Option<u32>,
    /// Suggested fix from the compiler, if any.
    pub suggestion: Option<String>,
}

impl CompilerDiagnostic {
    /// Whether this is an error (vs warning/note).
    pub fn is_error(&self) -> bool {
        self.level == "error"
    }

    /// Format as a human-readable string.
    pub fn display(&self) -> String {
        let location = match (&self.file, self.line, self.column) {
            (Some(f), Some(l), Some(c)) => format!("{f}:{l}:{c}"),
            (Some(f), Some(l), None) => format!("{f}:{l}"),
            (Some(f), None, None) => f.clone(),
            _ => "unknown".to_string(),
        };

        let code_str = self.code.as_deref().unwrap_or("");
        let suggestion_str = self
            .suggestion
            .as_ref()
            .map(|s| format!("\n  suggestion: {s}"))
            .unwrap_or_default();

        format!(
            "{level}[{code}] at {loc}: {msg}{sug}",
            level = self.level,
            code = code_str,
            loc = location,
            msg = self.message,
            sug = suggestion_str,
        )
    }
}

/// Summary of a compilation attempt.
#[derive(Debug, Clone)]
pub struct CompilationResult {
    /// Whether compilation succeeded.
    pub success: bool,
    /// Parsed diagnostics.
    pub diagnostics: Vec<CompilerDiagnostic>,
    /// Number of errors.
    pub error_count: usize,
    /// Number of warnings.
    pub warning_count: usize,
    /// Raw stderr output (truncated).
    pub raw_output: String,
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/// Run `cargo check` and parse compiler diagnostics for the refinement loop.
pub struct CompileFeedbackTool;

/// Default timeout for `cargo check` (seconds).
///
/// Measured cold-build cost on the akh-medu workspace (2026-07-27,
/// `CARGO_TARGET_DIR` shared across worktrees, sccache → S3):
///   * first-ever check (sccache cold, target cold): ~90s wall
///   * second check after `touch src/lib.rs` (sccache hit, target rebuild):
///     ~14s wall
///   * no-op check (target fresh): <1s
///
/// 600s gives ~7× margin over the worst observed cold build, which is enough
/// for a fresh container that has to download and compile the dependency graph
/// before reaching the akh-medu crates themselves. It is also deliberately
/// *below* the ACP turn idle watchdog's 300s default? No — see the module docs:
/// the watchdog is fed by incremental stdout draining + the `ToolCallStarted`
/// event, so the two budgets are independent. The caller can still request a
/// larger budget via the `timeout` parameter (e.g. 900–1200s for a known-cold
/// container) or a smaller one for a fast `cargo check` in a tight loop.
const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// Maximum output size to capture.
const MAX_OUTPUT_SIZE: usize = 64 * 1024;

/// Maximum pre-existing warnings kept in the feedback context. The total is
/// always reported; warnings above the budget are deliberately not presented
/// as work for the implementer.
const PREEXISTING_WARNING_BUDGET: usize = 50;

/// Maximum individual baseline diagnostics rendered in one response.
const MAX_BASELINE_DIAGNOSTICS_SHOWN: usize = 10;

/// Bound state retained by a long-lived daemon across sessions.
const MAX_BASELINE_CACHE_ENTRIES: usize = 64;

// ---------------------------------------------------------------------------
// Work-item-aware scope and immutable-base baseline
// ---------------------------------------------------------------------------

/// Discovery failures fail open to a real workspace check. They never turn
/// into a successful tool result that merely asks the Akh to check manually.
#[derive(Debug, thiserror::Error)]
enum ScopeDiscoveryError {
    #[error("command failed to spawn: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("command exited with non-zero status: {0}")]
    NonZeroExit(String),
    #[error("command output was not valid UTF-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("invalid cargo metadata: {0}")]
    Metadata(#[from] serde_json::Error),
    #[error("working directory is outside repository root")]
    OutsideRepository,
}

#[derive(Debug, Clone)]
struct BaseRevision {
    display_ref: String,
    sha: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CompileScope {
    Packages(Vec<String>),
    Workspace,
    NoRustChanges,
}

#[derive(Debug, Clone)]
struct ScopePlan {
    repo_root: PathBuf,
    relative_workdir: PathBuf,
    base: BaseRevision,
    changed_files: Vec<String>,
    scope: CompileScope,
}

#[derive(Debug, Clone)]
struct WorkspacePackage {
    name: String,
    root: PathBuf,
}

#[derive(Debug, Clone)]
struct WorkspaceIndex {
    root: PathBuf,
    packages: Vec<WorkspacePackage>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct DiagnosticKey {
    level: String,
    message: String,
    code: Option<String>,
    file: Option<String>,
}

impl From<&CompilerDiagnostic> for DiagnosticKey {
    fn from(diagnostic: &CompilerDiagnostic) -> Self {
        Self {
            level: diagnostic.level.clone(),
            message: diagnostic.message.clone(),
            code: diagnostic.code.clone(),
            // Line numbers are intentionally excluded. An edit above an
            // existing diagnostic moves the line without making it task work.
            file: diagnostic
                .file
                .as_deref()
                .map(|file| file.replace('\\', "/")),
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BaselineKey {
    repo_root: PathBuf,
    base_sha: String,
    relative_workdir: PathBuf,
    command: String,
    scope: String,
}

#[derive(Debug, Clone)]
struct BaselineSnapshot {
    diagnostics: HashMap<DiagnosticKey, usize>,
    error_count: usize,
    warning_count: usize,
    exit_code: i32,
}

#[derive(Default)]
struct BaselineCache {
    entries: HashMap<BaselineKey, BaselineSnapshot>,
    order: VecDeque<BaselineKey>,
}

impl BaselineCache {
    fn get(&self, key: &BaselineKey) -> Option<BaselineSnapshot> {
        self.entries.get(key).cloned()
    }

    fn insert(&mut self, key: BaselineKey, value: BaselineSnapshot) {
        if !self.entries.contains_key(&key) {
            self.order.push_back(key.clone());
        }
        self.entries.insert(key, value);
        while self.order.len() > MAX_BASELINE_CACHE_ENTRIES {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }
}

static BASELINE_CACHE: OnceLock<Mutex<BaselineCache>> = OnceLock::new();
static BASELINE_WORKTREE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn baseline_cache() -> &'static Mutex<BaselineCache> {
    BASELINE_CACHE.get_or_init(|| Mutex::new(BaselineCache::default()))
}

fn run_command_text(mut command: Command) -> Result<String, ScopeDiscoveryError> {
    let output = command.output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(ScopeDiscoveryError::NonZeroExit(if stderr.is_empty() {
            format!("exit status {}", output.status)
        } else {
            stderr
        }));
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn git_text(workdir: &Path, args: &[&str]) -> Result<String, ScopeDiscoveryError> {
    let mut command = Command::new("git");
    command.args(args).current_dir(workdir);
    run_command_text(command)
}

fn git_paths(workdir: &Path, args: &[&str]) -> Result<Vec<String>, ScopeDiscoveryError> {
    let output = git_text(workdir, args)?;
    Ok(output
        .split('\0')
        .filter(|path| !path.is_empty())
        .map(str::to_owned)
        .collect())
}

fn git_ref_exists(workdir: &Path, reference: &str) -> bool {
    Command::new("git")
        .args([
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("{reference}^{{commit}}"),
        ])
        .current_dir(workdir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn remote_base_candidates(workdir: &Path) -> Vec<String> {
    let Ok(refs) = git_text(
        workdir,
        &[
            "for-each-ref",
            "--format=%(refname:short)",
            "refs/remotes/origin",
        ],
    ) else {
        return Vec::new();
    };
    let mut scored = Vec::new();
    for reference in refs
        .lines()
        .map(str::trim)
        .filter(|reference| !reference.is_empty() && *reference != "origin/HEAD")
    {
        let is_ancestor = Command::new("git")
            .args(["merge-base", "--is-ancestor", reference, "HEAD"])
            .current_dir(workdir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false);
        if !is_ancestor {
            continue;
        }
        let count = git_text(
            workdir,
            &["rev-list", "--count", &format!("{reference}..HEAD")],
        )
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0);
        scored.push((count, reference.to_string()));
    }
    // In an Anima continuation clone both origin/<work-item> and origin/<base>
    // may exist. The base is the older ancestor (more commits behind HEAD),
    // whereas the prior work-item head is the nearer one.
    scored.sort_by(|left, right| right.cmp(left));
    scored.into_iter().map(|(_, reference)| reference).collect()
}

/// Resolve the immutable revision the worktree was cut from. Anima clones the
/// configured base and creates an `agent/*` branch; `origin/HEAD` is the normal
/// path. The explicit/environment override covers non-default project lanes.
fn resolve_base_revision(
    workdir: &Path,
    explicit: Option<&str>,
) -> Result<BaseRevision, ScopeDiscoveryError> {
    let mut candidates = Vec::new();
    let configured = explicit
        .filter(|value| !value.trim().is_empty())
        .map(str::trim)
        .map(str::to_owned)
        .or_else(|| {
            std::env::var("AKH_COMPILE_FEEDBACK_BASE_REF")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .map(|value| value.trim().to_owned())
        });
    if let Some(reference) = configured {
        // A caller-supplied base is a correctness contract, not a hint. Do not
        // silently compare against a different trunk when it is misspelled.
        candidates.push(reference);
    } else {
        if let Ok(symbolic) = git_text(
            workdir,
            &[
                "symbolic-ref",
                "--quiet",
                "--short",
                "refs/remotes/origin/HEAD",
            ],
        ) {
            candidates.push(symbolic.trim().to_string());
        }
        for conventional in [
            "origin/main",
            "origin/master",
            "origin/develop",
            "main",
            "master",
            "develop",
        ] {
            candidates.push(conventional.to_string());
        }
        candidates.extend(remote_base_candidates(workdir));
    }

    let mut seen = HashSet::new();
    let Some(display_ref) = candidates
        .into_iter()
        .find(|candidate| seen.insert(candidate.clone()) && git_ref_exists(workdir, candidate))
    else {
        return Err(ScopeDiscoveryError::NonZeroExit(
            "could not resolve a base ref (set base_ref or AKH_COMPILE_FEEDBACK_BASE_REF)".into(),
        ));
    };
    let sha = git_text(workdir, &["merge-base", "HEAD", &display_ref])?
        .trim()
        .to_string();
    if sha.is_empty() {
        return Err(ScopeDiscoveryError::NonZeroExit(format!(
            "no merge base between HEAD and {display_ref}"
        )));
    }
    Ok(BaseRevision { display_ref, sha })
}

/// Union committed branch changes, staged edits, unstaged edits, and untracked
/// paths. Looking only at `git diff HEAD` loses prior Anima turns after the
/// runner commits them.
fn changed_files_since_base(
    workdir: &Path,
    base_sha: &str,
) -> Result<Vec<String>, ScopeDiscoveryError> {
    let committed_range = format!("{base_sha}..HEAD");
    let commands: [&[&str]; 4] = [
        &["diff", "--name-only", "-z", &committed_range],
        &["diff", "--name-only", "-z"],
        &["diff", "--cached", "--name-only", "-z"],
        &["ls-files", "--others", "--exclude-standard", "-z"],
    ];
    let mut files = BTreeSet::new();
    for args in commands {
        for path in git_paths(workdir, args)? {
            files.insert(path);
        }
    }
    Ok(files.into_iter().collect())
}

fn load_workspace_index(workdir: &Path) -> Result<WorkspaceIndex, ScopeDiscoveryError> {
    let mut command = Command::new("cargo");
    command
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(workdir);
    let metadata: serde_json::Value = serde_json::from_str(&run_command_text(command)?)?;
    let root = metadata
        .get("workspace_root")
        .and_then(serde_json::Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| {
            ScopeDiscoveryError::NonZeroExit("cargo metadata omitted workspace_root".into())
        })?;
    let mut packages = metadata
        .get("packages")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|package| {
            let name = package.get("name")?.as_str()?.to_string();
            let manifest = PathBuf::from(package.get("manifest_path")?.as_str()?);
            Some(WorkspacePackage {
                name,
                root: manifest.parent()?.to_path_buf(),
            })
        })
        .collect::<Vec<_>>();
    // Deepest package root wins for nested workspace members.
    packages.sort_by_key(|package| std::cmp::Reverse(package.root.components().count()));
    Ok(WorkspaceIndex { root, packages })
}

fn is_workspace_wide_path(path: &Path) -> bool {
    matches!(
        path.to_string_lossy().as_ref(),
        "Cargo.toml" | "Cargo.lock" | "rust-toolchain" | "rust-toolchain.toml"
    ) || path.starts_with(".cargo")
}

fn package_for_path(index: &WorkspaceIndex, absolute_path: &Path) -> Option<String> {
    let package = index
        .packages
        .iter()
        .find(|package| absolute_path.starts_with(&package.root))?;

    // A root package's directory is also the workspace root. Do not make a
    // docs-only or CI-only edit look like a root-crate change.
    if package.root == index.root {
        let relative = absolute_path.strip_prefix(&package.root).ok()?;
        let first = relative.components().next()?.as_os_str().to_string_lossy();
        if !matches!(
            first.as_ref(),
            "src" | "tests" | "examples" | "benches" | "build.rs"
        ) {
            return None;
        }
    }
    Some(package.name.clone())
}

fn scope_for_changes(index: &WorkspaceIndex, repo_root: &Path, files: &[String]) -> CompileScope {
    if files.is_empty() {
        return CompileScope::NoRustChanges;
    }

    let mut packages = BTreeSet::new();
    let mut unmapped_rust_path = false;
    for file in files {
        let repo_relative = Path::new(file);
        let absolute = repo_root.join(repo_relative);
        let workspace_relative = absolute.strip_prefix(&index.root).ok();
        if workspace_relative.is_some_and(is_workspace_wide_path) {
            return CompileScope::Workspace;
        }
        match package_for_path(index, &absolute) {
            Some(package) => {
                packages.insert(package);
            }
            None if repo_relative
                .extension()
                .is_some_and(|extension| extension == "rs")
                || repo_relative
                    .file_name()
                    .is_some_and(|name| name == "Cargo.toml") =>
            {
                // A deleted/new package may not appear in current metadata.
                // An unmapped Rust path therefore expands rather than skips.
                unmapped_rust_path = true;
            }
            None => {}
        }
    }

    if unmapped_rust_path {
        CompileScope::Workspace
    } else if packages.is_empty() {
        CompileScope::NoRustChanges
    } else {
        CompileScope::Packages(packages.into_iter().collect())
    }
}

fn discover_scope(
    workdir: &Path,
    explicit_base: Option<&str>,
) -> Result<ScopePlan, ScopeDiscoveryError> {
    let repo_root = PathBuf::from(git_text(workdir, &["rev-parse", "--show-toplevel"])?.trim());
    let canonical_workdir = fs::canonicalize(workdir)?;
    let canonical_root = fs::canonicalize(&repo_root)?;
    let relative_workdir = canonical_workdir
        .strip_prefix(&canonical_root)
        .map_err(|_| ScopeDiscoveryError::OutsideRepository)?
        .to_path_buf();
    let base = resolve_base_revision(&canonical_workdir, explicit_base)?;
    let changed_files = changed_files_since_base(&canonical_workdir, &base.sha)?;
    let index = load_workspace_index(&canonical_workdir)?;
    let scope = scope_for_changes(&index, &canonical_root, &changed_files);
    Ok(ScopePlan {
        repo_root: canonical_root,
        relative_workdir,
        base,
        changed_files,
        scope,
    })
}

fn scope_label(scope: &CompileScope) -> String {
    match scope {
        CompileScope::Packages(packages) => packages.join(","),
        CompileScope::Workspace => "--workspace".into(),
        CompileScope::NoRustChanges => "no-rust-changes".into(),
    }
}

fn append_scope_args(command: &mut Command, scope: &CompileScope) {
    match scope {
        CompileScope::Packages(packages) => {
            for package in packages {
                command.arg("-p").arg(package);
            }
        }
        CompileScope::Workspace => {
            command.arg("--workspace");
        }
        CompileScope::NoRustChanges => {}
    }
}

fn cargo_command(command: &str, workdir: &Path, scope: &CompileScope) -> Result<Command, String> {
    let mut cargo = Command::new("cargo");
    match command {
        "check" => {
            cargo.arg("check").arg("--message-format=json");
        }
        "clippy" => {
            cargo.arg("clippy").arg("--message-format=json");
        }
        "test" => {
            cargo
                .arg("test")
                .arg("--no-run")
                .arg("--message-format=json");
        }
        other => {
            return Err(format!(
                "Unsupported command: '{other}'. Use 'check', 'clippy', or 'test'."
            ));
        }
    }
    append_scope_args(&mut cargo, scope);
    cargo
        .current_dir(workdir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    Ok(cargo)
}

struct CargoRun {
    exit_code: i32,
    stdout: String,
    stderr: String,
    elapsed_secs: u64,
}

fn run_cargo_supervised(
    command_name: &str,
    command: Command,
    timeout_secs: u64,
) -> Result<CargoRun, String> {
    let start = Instant::now();
    let mut exec = match SupervisedExecution::spawn(command) {
        Ok(execution) => execution,
        Err(SupervisorError::Spawn { source }) => {
            return Err(format!("Failed to spawn `cargo {command_name}`: {source}"));
        }
        Err(error) => {
            return Err(format!(
                "Supervision error spawning `cargo {command_name}`: {error}"
            ));
        }
    };
    let mut tracker = ProgressTracker::with_timeout(Duration::from_secs(timeout_secs));
    let drain = exec.take_drain();
    let (status, output) =
        match exec.wait_with_drain_capped(&mut tracker, drain, MAX_OUTPUT_SIZE * 4) {
            Ok(result) => result,
            Err(SupervisorError::Killed {
                reason,
                partial_stdout,
                partial_stderr,
                elapsed,
            }) => {
                return Err(format_stuck_result(
                    command_name,
                    &reason,
                    elapsed.as_secs(),
                    &String::from_utf8_lossy(&partial_stdout),
                    &String::from_utf8_lossy(&partial_stderr),
                ));
            }
            Err(SupervisorError::Wait { source }) => {
                return Err(format!("Failed to wait: {source}"));
            }
            Err(error) => return Err(format!("Supervision error: {error}")),
        };
    Ok(CargoRun {
        exit_code: status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        elapsed_secs: start.elapsed().as_secs(),
    })
}

/// A detached checkout of the worktree's immutable base. The holder path is
/// uniquely created next to the active worktree, and Drop removes both Git's
/// administrative entry and the known temporary directory.
struct DetachedBaselineWorktree {
    repo_root: PathBuf,
    holder: PathBuf,
    checkout: PathBuf,
}

impl DetachedBaselineWorktree {
    fn create(repo_root: &Path, sha: &str) -> Result<Self, ScopeDiscoveryError> {
        let sequence = BASELINE_WORKTREE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let parent = repo_root.parent().unwrap_or(repo_root);
        let holder = parent.join(format!(
            ".akh-compile-baseline-{}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&holder)?;
        let checkout = holder.join("source");
        let checkout_text = checkout.to_string_lossy().into_owned();
        if let Err(error) = git_text(
            repo_root,
            &[
                "worktree",
                "add",
                "--detach",
                "--quiet",
                &checkout_text,
                sha,
            ],
        ) {
            let _ = fs::remove_dir_all(&holder);
            return Err(error);
        }
        Ok(Self {
            repo_root: repo_root.to_path_buf(),
            holder,
            checkout,
        })
    }
}

impl Drop for DetachedBaselineWorktree {
    fn drop(&mut self) {
        let checkout = self.checkout.to_string_lossy().into_owned();
        let _ = Command::new("git")
            .args(["worktree", "remove", "--force", &checkout])
            .current_dir(&self.repo_root)
            .status();
        let _ = fs::remove_dir_all(&self.holder);
    }
}

fn diagnostic_counts(diagnostics: &[CompilerDiagnostic]) -> HashMap<DiagnosticKey, usize> {
    let mut counts = HashMap::new();
    for diagnostic in diagnostics {
        *counts.entry(DiagnosticKey::from(diagnostic)).or_insert(0) += 1;
    }
    counts
}

/// Multiset comparison preserves duplicate diagnostics while tolerating line
/// shifts. A set would incorrectly hide a second occurrence introduced by the
/// task; a line-sensitive key would incorrectly blame every shifted warning.
fn classify_against_baseline(
    diagnostics: Vec<CompilerDiagnostic>,
    baseline: Option<&BaselineSnapshot>,
) -> (Vec<CompilerDiagnostic>, Vec<CompilerDiagnostic>) {
    let Some(baseline) = baseline else {
        return (diagnostics, Vec::new());
    };
    let mut remaining = baseline.diagnostics.clone();
    let mut task = Vec::new();
    let mut preexisting = Vec::new();
    for diagnostic in diagnostics {
        let key = DiagnosticKey::from(&diagnostic);
        match remaining.get_mut(&key) {
            Some(count) if *count > 0 => {
                *count -= 1;
                preexisting.push(diagnostic);
            }
            _ => task.push(diagnostic),
        }
    }
    (task, preexisting)
}

fn capture_baseline(
    plan: &ScopePlan,
    command_name: &str,
    timeout_secs: u64,
) -> Result<(BaselineSnapshot, bool), String> {
    let key = BaselineKey {
        repo_root: plan.repo_root.clone(),
        base_sha: plan.base.sha.clone(),
        relative_workdir: plan.relative_workdir.clone(),
        command: command_name.to_string(),
        scope: scope_label(&plan.scope),
    };
    if let Some(cached) = baseline_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
    {
        return Ok((cached, true));
    }

    let worktree = DetachedBaselineWorktree::create(&plan.repo_root, &plan.base.sha)
        .map_err(|error| error.to_string())?;
    let baseline_cwd = worktree.checkout.join(&plan.relative_workdir);
    let mut command = cargo_command(command_name, &baseline_cwd, &plan.scope)?;
    // Baseline and edited sources MUST NOT share Cargo fingerprints. With a
    // shared target and same-second mtimes Cargo can consider the edited tree
    // fresh against artifacts produced from the detached base, yielding a
    // false green. Keep the baseline target inside its disposable holder;
    // compiler-level caches such as sccache still provide safe reuse.
    command.env("CARGO_TARGET_DIR", worktree.holder.join("target"));
    let run = run_cargo_supervised(command_name, command, timeout_secs)?;
    let diagnostics = parse_json_diagnostics(&run.stdout);
    if run.exit_code != 0 && !diagnostics.iter().any(CompilerDiagnostic::is_error) {
        return Err(format!(
            "base cargo {command_name} exited {} without a compiler diagnostic: {}",
            run.exit_code,
            truncate_text(&run.stderr, 2_000)
        ));
    }
    let snapshot = BaselineSnapshot {
        diagnostics: diagnostic_counts(&diagnostics),
        error_count: diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_error())
            .count(),
        warning_count: diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.level == "warning")
            .count(),
        exit_code: run.exit_code,
    };
    baseline_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(key, snapshot.clone());
    Ok((snapshot, false))
}

fn truncate_text(text: &str, limit: usize) -> String {
    if text.len() <= limit {
        return text.to_string();
    }
    let mut boundary = limit;
    while boundary > 0 && !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    format!("{}... [truncated]", &text[..boundary])
}

// ---------------------------------------------------------------------------
// Resolve timeout
// ---------------------------------------------------------------------------

/// Resolve the per-call timeout: explicit `timeout` parameter wins, then
/// `AKH_COMPILE_FEEDBACK_TIMEOUT_SECS`, then the built-in default. `0` (or any
/// parse failure that yields zero) falls back to the default so a malformed
/// caller argument cannot disable the deadline entirely.
fn resolve_timeout(input: &ToolInput) -> u64 {
    if let Some(secs) = input.get_u64("timeout") {
        if secs > 0 {
            return secs;
        }
    }
    if let Ok(s) = std::env::var("AKH_COMPILE_FEEDBACK_TIMEOUT_SECS") {
        if let Ok(secs) = s.trim().parse::<u64>() {
            if secs > 0 {
                return secs;
            }
        }
    }
    DEFAULT_TIMEOUT_SECS
}

impl Tool for CompileFeedbackTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "compile_feedback".into(),
            description: "Run `cargo check` and return parsed compiler diagnostics. \
                          Supports JSON output mode for structured error analysis. \
                          By default, scopes to packages changed since the Git worktree base, \
                          compares against that immutable base's diagnostics, and reports only \
                          task-introduced issues as work. Pass `workspace=true` for the workspace."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "path".into(),
                    description: "Working directory for cargo (default: current directory).".into(),
                    required: false,
                },
                ToolParam {
                    name: "command".into(),
                    description: "Cargo command: 'check' (default), 'clippy', or 'test'.".into(),
                    required: false,
                },
                ToolParam {
                    name: "timeout".into(),
                    description: concat!(
                        "Timeout in seconds (default: 600; override process-wide with ",
                        "AKH_COMPILE_FEEDBACK_TIMEOUT_SECS). A cold akh-medu build can ",
                        "take 90s+; request 900–1200s for a known-cold container, or a ",
                        "smaller value for a tight edit→check loop on a warm target."
                    )
                    .into(),
                    required: false,
                },
                ToolParam {
                    name: "workspace".into(),
                    description: "Check the entire workspace instead of changed packages. \
                                  Default: false (blast-radius mode)."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "base_ref".into(),
                    description: "Optional Git base ref override. Defaults to origin/HEAD, then \
                                  conventional trunk refs; AKH_COMPILE_FEEDBACK_BASE_REF is the \
                                  process-wide override."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let path = input.get("path").unwrap_or(".");
        let command = input.get("command").unwrap_or("check");
        let timeout_secs = resolve_timeout(&input);
        let workspace_mode = input.get_bool("workspace").unwrap_or(false);
        let explicit_base = input.get("base_ref");
        if !matches!(command, "check" | "clippy" | "test") {
            return Ok(ToolOutput::err(format!(
                "Unsupported command: '{command}'. Use 'check', 'clippy', or 'test'."
            )));
        }

        // Discover both the committed and uncommitted work-item delta. On any
        // failure, expand to an actual workspace build and carry the reason in
        // the result; never return a false green/manual instruction.
        let (plan, scope, discovery_notice) = match discover_scope(Path::new(path), explicit_base) {
            Ok(mut plan) => {
                if workspace_mode {
                    plan.scope = CompileScope::Workspace;
                }
                let scope = plan.scope.clone();
                (Some(plan), scope, None)
            }
            Err(error) => (
                None,
                CompileScope::Workspace,
                Some(format!(
                    "scope discovery failed ({error}); expanded to a real --workspace check"
                )),
            ),
        };

        if scope == CompileScope::NoRustChanges {
            let plan = plan.expect("NoRustChanges is only produced by successful discovery");
            return Ok(ToolOutput::ok(format!(
                "BLAST RADIUS: no Rust package changes among {} changed path(s) since {}@{}; \
                 no compile invocation is needed. Pass workspace=true to validate everything.",
                plan.changed_files.len(),
                plan.base.display_ref,
                &plan.base.sha[..12.min(plan.base.sha.len())],
            )));
        }

        // Materialise the immutable worktree base once per command/scope and
        // cache its diagnostic multiset. This is session-start semantics even
        // when the Akh calls the tool after several committed turns.
        let (baseline, baseline_cached, baseline_error) = match plan
            .as_ref()
            .map(|plan| capture_baseline(plan, command, timeout_secs))
        {
            Some(Ok((snapshot, cached))) => (Some(snapshot), cached, None),
            Some(Err(error)) => (None, false, Some(error)),
            None => (None, false, Some("base revision unavailable".into())),
        };

        let cargo = match cargo_command(command, Path::new(path), &scope) {
            Ok(command) => command,
            Err(error) => return Ok(ToolOutput::err(error)),
        };
        let run = match run_cargo_supervised(command, cargo, timeout_secs) {
            Ok(run) => run,
            Err(error) => return Ok(ToolOutput::err(error)),
        };
        let all_diagnostics = parse_json_diagnostics(&run.stdout);
        let (task_diagnostics, preexisting) =
            classify_against_baseline(all_diagnostics, baseline.as_ref());

        let task_error_count = task_diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.is_error())
            .count();
        let task_warning_count = task_diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.level == "warning")
            .count();
        let preexisting_error_count = preexisting
            .iter()
            .filter(|diagnostic| diagnostic.is_error())
            .count();
        let preexisting_warning_count = preexisting
            .iter()
            .filter(|diagnostic| diagnostic.level == "warning")
            .count();

        // A non-zero build made entirely of diagnostics already present on a
        // non-zero base is context, not a new task. Any unclassified failure
        // still fails closed (build scripts and linker failures may emit no
        // compiler diagnostic at all).
        let blocked_only_by_baseline = run.exit_code != 0
            && task_error_count == 0
            && preexisting_error_count > 0
            && baseline
                .as_ref()
                .is_some_and(|snapshot| snapshot.exit_code != 0);
        let unexplained_failure = run.exit_code != 0 && !blocked_only_by_baseline;
        let raw_output = truncate_text(&run.stderr, MAX_OUTPUT_SIZE);
        let compilation = CompilationResult {
            success: task_error_count == 0 && !unexplained_failure,
            diagnostics: task_diagnostics.clone(),
            error_count: task_error_count,
            warning_count: task_warning_count,
            raw_output: raw_output.clone(),
        };

        // Format output with work-item scope and baseline context.
        let mut result = String::new();
        match &scope {
            CompileScope::Packages(packages) => result.push_str(&format!(
                "BLAST RADIUS: {} changed package(s): [{}]\n",
                packages.len(),
                packages.join(", ")
            )),
            CompileScope::Workspace => result.push_str("MODE: full workspace (--workspace)\n"),
            CompileScope::NoRustChanges => unreachable!("handled before cargo invocation"),
        }
        if let Some(plan) = &plan {
            result.push_str(&format!(
                "WORKTREE DELTA: {} path(s) since {}@{} (committed + staged + unstaged + untracked)\n",
                plan.changed_files.len(),
                plan.base.display_ref,
                &plan.base.sha[..12.min(plan.base.sha.len())]
            ));
        }
        if let Some(notice) = &discovery_notice {
            result.push_str(&format!("SCOPE FALLBACK: {notice}\n"));
        }
        match (&baseline, &baseline_error) {
            (Some(snapshot), _) => result.push_str(&format!(
                "BASELINE: {}{} (exit {}, {} error(s), {} warning(s))\n",
                if baseline_cached { "cached " } else { "captured " },
                plan.as_ref()
                    .map(|plan| format!(
                        "{}@{}",
                        plan.base.display_ref,
                        &plan.base.sha[..12.min(plan.base.sha.len())]
                    ))
                    .unwrap_or_else(|| "worktree base".into()),
                snapshot.exit_code,
                snapshot.error_count,
                snapshot.warning_count,
            )),
            (None, Some(error)) => result.push_str(&format!(
                "BASELINE UNAVAILABLE: {error}; current diagnostics are conservatively treated as task work.\n"
            )),
            (None, None) => {}
        }
        result.push_str(&format!(
            "cargo {command}: exit code {} ({} task error(s), {} task warning(s), {}s)\n",
            run.exit_code, task_error_count, task_warning_count, run.elapsed_secs
        ));

        if preexisting_error_count > 0 || preexisting_warning_count > 0 {
            result.push_str(&format!(
                "BASELINE CONTEXT: {} pre-existing error(s), {} pre-existing warning(s); not assigned as task work.\n",
                preexisting_error_count, preexisting_warning_count
            ));
            if preexisting_warning_count > PREEXISTING_WARNING_BUDGET {
                result.push_str(&format!(
                    "LINT BUDGET: {} baseline warning(s) exceed the {}-warning context budget; {} suppressed.\n",
                    preexisting_warning_count,
                    PREEXISTING_WARNING_BUDGET,
                    preexisting_warning_count - PREEXISTING_WARNING_BUDGET
                ));
            }
        }

        if compilation.success {
            result
                .push_str("\n✓ No task-introduced compile errors in the selected blast radius.\n");
            if blocked_only_by_baseline {
                result.push_str(
                    "The command remains non-zero only because the same errors exist on the immutable base. \
                     Treat them as lane context, not cleanup assigned to this work item.\n",
                );
            }
            if task_warning_count > 0 {
                result.push_str("\nTask warnings:\n");
                for diagnostic in task_diagnostics
                    .iter()
                    .filter(|diagnostic| diagnostic.level == "warning")
                {
                    result.push_str(&format!("  {}\n", diagnostic.display()));
                }
            }
            if preexisting_error_count > 0 {
                result.push_str("\nRepresentative baseline errors (context only):\n");
                for diagnostic in preexisting
                    .iter()
                    .filter(|diagnostic| diagnostic.is_error())
                    .take(MAX_BASELINE_DIAGNOSTICS_SHOWN)
                {
                    result.push_str(&format!("  {}\n", diagnostic.display()));
                }
            }
            Ok(ToolOutput::ok(truncate_text(&result, MAX_OUTPUT_SIZE)))
        } else {
            if task_error_count > 0 {
                result.push_str("\nTask-introduced errors:\n");
                for diagnostic in task_diagnostics
                    .iter()
                    .filter(|diagnostic| diagnostic.is_error())
                {
                    result.push_str(&format!("  {}\n", diagnostic.display()));
                }
            } else {
                result.push_str(
                    "\nCargo failed without an error attributable to the baseline. \
                     This is a task blocker (for example a build-script/linker failure).\n",
                );
                if !raw_output.trim().is_empty() {
                    result.push_str(&format!("\nstderr:\n{}\n", raw_output.trim()));
                }
            }

            if preexisting_error_count > 0 {
                result.push_str("\nRepresentative baseline errors (context only):\n");
                for diagnostic in preexisting
                    .iter()
                    .filter(|diagnostic| diagnostic.is_error())
                    .take(MAX_BASELINE_DIAGNOSTICS_SHOWN)
                {
                    result.push_str(&format!("  {}\n", diagnostic.display()));
                }
            }

            let suggestions: Vec<_> = task_diagnostics
                .iter()
                .filter_map(|diagnostic| diagnostic.suggestion.as_ref())
                .collect();
            if !suggestions.is_empty() {
                result.push_str("\nSuggested fixes:\n");
                for suggestion in suggestions {
                    result.push_str(&format!("  - {suggestion}\n"));
                }
            }
            Ok(ToolOutput::err(truncate_text(&result, MAX_OUTPUT_SIZE)))
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "compile_feedback".into(),
            description: "Runs cargo check/clippy/test and parses compiler diagnostics. \
                          Defaults to packages changed since the Git worktree base and compares \
                          diagnostics with that immutable baseline; workspace mode is opt-in."
                .into(),
            parameters: vec![
                ToolParamSchema::optional("path", "Working directory for cargo."),
                ToolParamSchema::optional(
                    "command",
                    "Cargo command: 'check', 'clippy', or 'test'.",
                ),
                ToolParamSchema::optional(
                    "timeout",
                    concat!(
                        "Timeout in seconds (default 600; cold builds can take 90s+, ",
                        "so request 900–1200s for a known-cold container, or a smaller ",
                        "value for a warm-target edit→check loop). Override the ",
                        "process-wide default with AKH_COMPILE_FEEDBACK_TIMEOUT_SECS."
                    ),
                ),
                ToolParamSchema::optional(
                    "workspace",
                    "Check the entire workspace instead of changed packages. \
                     Default: false (blast-radius mode).",
                ),
                ToolParamSchema::optional(
                    "base_ref",
                    "Optional Git base ref override. Defaults to origin/HEAD then \
                     conventional trunk refs; AKH_COMPILE_FEEDBACK_BASE_REF is the \
                     process-wide override.",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::ProcessExec]),
                description:
                    "Runs Cargo and Git; creates build artifacts and a temporary detached \
                              baseline worktree, but does not modify source files."
                        .into(),
                shadow_triggers: vec!["compile".into(), "build".into(), "check".into()],
            },
            source: ToolSource::Native,
        }
    }
}

// ---------------------------------------------------------------------------
// JSON diagnostic parsing
// ---------------------------------------------------------------------------

/// Format a stuck process result distinguishable from both compile errors
/// and traditional timeouts.
///
/// A stuck process was killed by the progress-based supervisor (no output for
/// N minutes), not by an elapsed-time timeout. The distinction matters: stuck
/// usually means a true hang, while the old timeout could fire on slow-but-
/// healthy builds. The agent should know which one it hit.
fn format_stuck_result(
    command: &str,
    reason: &str,
    elapsed: u64,
    partial_stdout: &str,
    partial_stderr: &str,
) -> String {
    let partial_diagnostics = parse_json_diagnostics(partial_stdout);
    let partial_errors = partial_diagnostics.iter().filter(|d| d.is_error()).count();

    let mut result = String::new();
    result.push_str(&format!(
        "STUCK: cargo {command} was killed after {elapsed}s: {reason}. \
         This is distinct from both a compile error and a simple timeout — \
         the process appeared stuck (no output for an extended period), so \
         the supervisor terminated it.\n"
    ));

    if partial_errors > 0 {
        result.push_str(&format!(
            "\nPartial diagnostics captured before termination \
             ({partial_errors} error(s) already emitted — these ARE real \
             compile errors you should fix before retrying):\n"
        ));
        for d in partial_diagnostics.iter().filter(|d| d.is_error()) {
            result.push_str(&format!("  {}\n", d.display()));
        }
    } else {
        result.push_str(
            "\nNo compiler diagnostics were emitted before termination. \
             This usually means cargo hung while compiling dependencies or \
             that a build script is stuck.\n",
        );
    }

    if !partial_stderr.is_empty() {
        let trimmed = if partial_stderr.len() > MAX_OUTPUT_SIZE {
            &partial_stderr[..MAX_OUTPUT_SIZE]
        } else {
            partial_stderr
        };
        result.push_str(&format!(
            "\nPartial stderr ({} bytes):\n{trimmed}\n",
            partial_stderr.len()
        ));
    }

    result.push_str(
        "\nA stuck process typically indicates a hang (e.g., a build script \
         waiting for input, a deadlock, or a network fetch that never returns). \
         The progress timeout can be adjusted via AKH_TOOL_PROGRESS_TIMEOUT_SECS \
         if you know a legitimate operation needs longer without output. \
         If the build is repeatedly stuck, investigate what happens after the \
         last diagnostic shown above — that's where the hang occurred.",
    );
    result
}

/// Parse cargo's JSON diagnostic messages from stdout.
///
/// Cargo outputs one JSON object per line when `--message-format=json` is used.
/// We extract `compiler-message` entries which contain diagnostic information.
fn parse_json_diagnostics(json_output: &str) -> Vec<CompilerDiagnostic> {
    let mut diagnostics = Vec::new();

    for line in json_output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Parse the JSON line
        let parsed: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Only process compiler-message reason
        if parsed.get("reason").and_then(|r| r.as_str()) != Some("compiler-message") {
            continue;
        }

        let Some(message) = parsed.get("message") else {
            continue;
        };

        let level = message
            .get("level")
            .and_then(|l| l.as_str())
            .unwrap_or("unknown")
            .to_string();

        let msg = message
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("")
            .to_string();

        let code = message
            .get("code")
            .and_then(|c| c.get("code"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string());

        // Extract primary span location
        let (file, line_num, column) = message
            .get("spans")
            .and_then(|spans| spans.as_array())
            .and_then(|spans| {
                spans
                    .iter()
                    .find(|s| s.get("is_primary") == Some(&serde_json::Value::Bool(true)))
            })
            .map(|span| {
                let file = span
                    .get("file_name")
                    .and_then(|f| f.as_str())
                    .map(|s| s.to_string());
                let line = span
                    .get("line_start")
                    .and_then(|l| l.as_u64())
                    .map(|l| l as u32);
                let col = span
                    .get("column_start")
                    .and_then(|c| c.as_u64())
                    .map(|c| c as u32);
                (file, line, col)
            })
            .unwrap_or((None, None, None));

        // Extract suggested replacement if available
        let suggestion = message
            .get("spans")
            .and_then(|spans| spans.as_array())
            .and_then(|spans| {
                spans.iter().find_map(|span| {
                    span.get("suggested_replacement")
                        .and_then(|s| s.as_str())
                        .filter(|s| !s.is_empty())
                        .map(|s| s.to_string())
                })
            });

        // Skip "aborting due to" summary messages
        if msg.starts_with("aborting due to") {
            continue;
        }

        diagnostics.push(CompilerDiagnostic {
            level,
            message: msg,
            code,
            file,
            line: line_num,
            column,
            suggestion,
        });
    }

    diagnostics
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_git(workdir: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(workdir)
            .output()
            .expect("spawn git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn init_test_repo(root: &Path, source: &str, package: &str) -> String {
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            format!("[package]\nname = \"{package}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n"),
        )
        .unwrap();
        fs::write(root.join("src/lib.rs"), source).unwrap();
        test_git(root, &["init", "-b", "main"]);
        test_git(root, &["config", "user.name", "Compile Feedback Test"]);
        test_git(
            root,
            &["config", "user.email", "compile-feedback@example.invalid"],
        );
        test_git(root, &["config", "commit.gpgsign", "false"]);
        test_git(root, &["add", "-A"]);
        test_git(root, &["commit", "-m", "baseline"]);
        test_git(root, &["rev-parse", "HEAD"])
    }

    fn warning_at(file: &str, line: u32) -> CompilerDiagnostic {
        CompilerDiagnostic {
            level: "warning".into(),
            message: "unused variable: `ambient`".into(),
            code: Some("unused_variables".into()),
            file: Some(file.into()),
            line: Some(line),
            column: Some(9),
            suggestion: None,
        }
    }

    #[test]
    fn parse_empty_output() {
        let diagnostics = parse_json_diagnostics("");
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn parse_non_json_output() {
        let diagnostics = parse_json_diagnostics("this is not json\nanother line\n");
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn parse_compiler_message() {
        let json = r#"{"reason":"compiler-message","package_id":"test","manifest_path":"","target":{},"message":{"rendered":"","children":[],"code":{"code":"E0308","explanation":null},"level":"error","message":"mismatched types","spans":[{"byte_end":100,"byte_start":90,"column_end":20,"column_start":10,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":5,"line_start":5,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;

        let diagnostics = parse_json_diagnostics(json);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, "error");
        assert_eq!(diagnostics[0].message, "mismatched types");
        assert_eq!(diagnostics[0].code.as_deref(), Some("E0308"));
        assert_eq!(diagnostics[0].file.as_deref(), Some("src/lib.rs"));
        assert_eq!(diagnostics[0].line, Some(5));
        assert_eq!(diagnostics[0].column, Some(10));
    }

    #[test]
    fn parse_warning() {
        let json = r#"{"reason":"compiler-message","package_id":"test","manifest_path":"","target":{},"message":{"rendered":"","children":[],"code":null,"level":"warning","message":"unused variable","spans":[{"byte_end":50,"byte_start":40,"column_end":10,"column_start":5,"expansion":null,"file_name":"src/main.rs","is_primary":true,"label":null,"line_end":3,"line_start":3,"suggested_replacement":"_x","suggestion_applicability":"MachineApplicable","text":[]}]}}"#;

        let diagnostics = parse_json_diagnostics(json);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, "warning");
        assert_eq!(diagnostics[0].message, "unused variable");
        assert_eq!(diagnostics[0].suggestion.as_deref(), Some("_x"));
    }

    #[test]
    fn parse_non_compiler_message_ignored() {
        let json = r#"{"reason":"build-script-executed","package_id":"test"}"#;
        let diagnostics = parse_json_diagnostics(json);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn diagnostic_display() {
        let d = CompilerDiagnostic {
            level: "error".into(),
            message: "mismatched types".into(),
            code: Some("E0308".into()),
            file: Some("src/lib.rs".into()),
            line: Some(5),
            column: Some(10),
            suggestion: None,
        };
        let display = d.display();
        assert!(display.contains("error"));
        assert!(display.contains("E0308"));
        assert!(display.contains("src/lib.rs:5:10"));
        assert!(display.contains("mismatched types"));
    }

    #[test]
    fn diagnostic_with_suggestion() {
        let d = CompilerDiagnostic {
            level: "warning".into(),
            message: "unused variable".into(),
            code: None,
            file: None,
            line: None,
            column: None,
            suggestion: Some("_x".into()),
        };
        let display = d.display();
        assert!(display.contains("suggestion: _x"));
    }

    #[test]
    fn compilation_result_success() {
        let result = CompilationResult {
            success: true,
            diagnostics: vec![],
            error_count: 0,
            warning_count: 0,
            raw_output: String::new(),
        };
        assert!(result.success);
        assert_eq!(result.error_count, 0);
    }

    #[test]
    fn changed_files_include_committed_staged_unstaged_and_untracked_paths() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let base = init_test_repo(root, "pub fn initial() {}\n", "scope-fixture");

        fs::write(root.join("src/committed.rs"), "pub fn committed() {}\n").unwrap();
        test_git(root, &["add", "src/committed.rs"]);
        test_git(root, &["commit", "-m", "committed task edit"]);

        fs::write(root.join("src/staged.rs"), "pub fn staged() {}\n").unwrap();
        test_git(root, &["add", "src/staged.rs"]);
        fs::write(root.join("src/lib.rs"), "pub fn unstaged() {}\n").unwrap();
        fs::write(root.join("src/untracked.rs"), "pub fn untracked() {}\n").unwrap();

        let files = changed_files_since_base(root, &base).unwrap();
        assert_eq!(
            files,
            vec![
                "src/committed.rs",
                "src/lib.rs",
                "src/staged.rs",
                "src/untracked.rs",
            ]
        );
    }

    #[test]
    fn scope_uses_deepest_package_and_expands_workspace_inputs() {
        let index = WorkspaceIndex {
            root: PathBuf::from("/repo"),
            packages: vec![
                WorkspacePackage {
                    name: "nested".into(),
                    root: PathBuf::from("/repo/crates/nested"),
                },
                WorkspacePackage {
                    name: "root".into(),
                    root: PathBuf::from("/repo"),
                },
            ],
        };

        assert_eq!(
            scope_for_changes(
                &index,
                Path::new("/repo"),
                &["src/lib.rs".into(), "crates/nested/src/lib.rs".into()]
            ),
            CompileScope::Packages(vec!["nested".into(), "root".into()])
        );
        assert_eq!(
            scope_for_changes(&index, Path::new("/repo"), &["docs/design.md".into()]),
            CompileScope::NoRustChanges
        );
        assert_eq!(
            scope_for_changes(&index, Path::new("/repo"), &["Cargo.lock".into()]),
            CompileScope::Workspace
        );
        assert_eq!(
            scope_for_changes(&index, Path::new("/repo"), &["new-crate/src/lib.rs".into()]),
            CompileScope::Workspace
        );

        let nested_workspace = WorkspaceIndex {
            root: PathBuf::from("/repo/rust"),
            packages: vec![WorkspacePackage {
                name: "member".into(),
                root: PathBuf::from("/repo/rust/crates/member"),
            }],
        };
        assert_eq!(
            scope_for_changes(
                &nested_workspace,
                Path::new("/repo"),
                &["rust/crates/member/src/lib.rs".into()]
            ),
            CompileScope::Packages(vec!["member".into()])
        );
        assert_eq!(
            scope_for_changes(
                &nested_workspace,
                Path::new("/repo"),
                &["rust/Cargo.lock".into()]
            ),
            CompileScope::Workspace
        );
    }

    #[test]
    fn baseline_comparison_tolerates_line_shift_but_consumes_multiplicity() {
        let baseline_diagnostic = warning_at("src/lib.rs", 3);
        let baseline = BaselineSnapshot {
            diagnostics: diagnostic_counts(std::slice::from_ref(&baseline_diagnostic)),
            error_count: 0,
            warning_count: 1,
            exit_code: 0,
        };
        let shifted = warning_at("src/lib.rs", 30);
        let duplicate = warning_at("src/lib.rs", 40);

        let (task, preexisting) =
            classify_against_baseline(vec![shifted, duplicate], Some(&baseline));
        assert_eq!(
            preexisting.len(),
            1,
            "shifted baseline warning stays context"
        );
        assert_eq!(task.len(), 1, "a second copy remains task-introduced");
    }

    #[test]
    fn explicit_invalid_base_does_not_silently_fall_back_to_main() {
        let temp = tempfile::tempdir().unwrap();
        init_test_repo(temp.path(), "pub fn initial() {}\n", "base-ref-fixture");
        let error = resolve_base_revision(temp.path(), Some("missing-ref")).unwrap_err();
        assert!(error.to_string().contains("could not resolve a base ref"));
    }

    #[test]
    fn generic_remote_fallback_prefers_base_over_prior_work_item_head() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let base = init_test_repo(root, "pub fn initial() {}\n", "remote-base-fixture");
        test_git(
            root,
            &["update-ref", "refs/remotes/origin/custom-base", &base],
        );

        fs::write(root.join("src/lib.rs"), "pub fn round_one() {}\n").unwrap();
        test_git(root, &["add", "src/lib.rs"]);
        test_git(root, &["commit", "-m", "round one"]);
        let prior_head = test_git(root, &["rev-parse", "HEAD"]);
        test_git(
            root,
            &["update-ref", "refs/remotes/origin/work-item", &prior_head],
        );
        fs::write(root.join("src/lib.rs"), "pub fn round_two() {}\n").unwrap();
        test_git(root, &["add", "src/lib.rs"]);
        test_git(root, &["commit", "-m", "round two"]);

        let candidates = remote_base_candidates(root);
        assert_eq!(
            candidates.first().map(String::as_str),
            Some("origin/custom-base")
        );
        assert_eq!(
            candidates.get(1).map(String::as_str),
            Some("origin/work-item")
        );
    }

    #[test]
    fn detached_base_captures_ambient_warning_and_current_error_is_task_work() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        init_test_repo(
            root,
            "pub fn ambient() { let ambient = 1; }\n",
            "baseline-fixture",
        );
        fs::write(
            root.join("src/lib.rs"),
            "pub fn ambient() { let ambient = 1; }\npub fn broken() -> u32 { \"not a number\" }\n",
        )
        .unwrap();

        let plan = discover_scope(root, Some("main")).unwrap();
        assert_eq!(
            plan.scope,
            CompileScope::Packages(vec!["baseline-fixture".into()])
        );
        let (baseline, cached) = capture_baseline(&plan, "check", 120).unwrap();
        assert!(!cached);
        assert_eq!(baseline.error_count, 0);
        assert!(baseline.warning_count >= 1);

        let current = run_cargo_supervised(
            "check",
            cargo_command("check", root, &plan.scope).unwrap(),
            120,
        )
        .unwrap();
        assert_ne!(current.exit_code, 0);
        let diagnostics = parse_json_diagnostics(&current.stdout);
        let (task, preexisting) = classify_against_baseline(diagnostics, Some(&baseline));
        assert!(task.iter().any(CompilerDiagnostic::is_error));
        assert!(
            preexisting
                .iter()
                .any(|diagnostic| diagnostic.level == "warning")
        );
    }

    #[test]
    fn truncation_keeps_utf8_boundary() {
        let text = "ééé";
        let truncated = truncate_text(text, 3);
        assert!(truncated.starts_with('é'));
        assert!(truncated.ends_with("... [truncated]"));
    }

    /// A stuck process must return a result distinguishable from both compile
    /// errors and simple timeouts. The acceptance test: a stuck result contains
    /// the `STUCK:` prefix, names the elapsed time and reason, and includes any
    /// partial output — none of which a real compile error carries.
    #[test]
    fn stuck_result_is_distinguishable_from_compile_error() {
        let result = format_stuck_result(
            "check",
            "no output for 300 seconds (appears stuck)",
            540,
            "",
            "",
        );

        // 1. Must start with STUCK: so the model can branch on it.
        assert!(
            result.starts_with("STUCK:"),
            "stuck result must start with STUCK: so the model can \
             distinguish it from a compile error; got: {}",
            &result[..80.min(result.len())]
        );
        // 2. Must report the elapsed time.
        assert!(
            result.contains("540s"),
            "stuck result must report elapsed time; got: {result}"
        );
        // 3. Must NOT use the compile-error phrasing (no "exit code" line).
        assert!(
            !result.contains("exit code"),
            "stuck result must not include an exit code line (that's a \
             compile-error shape); got: {result}"
        );
        // 4. With no partial output, it must say so explicitly.
        assert!(
            result.contains("No compiler diagnostics were emitted"),
            "stuck with no partial output must say so; got: {result}"
        );
    }

    /// A stuck process that fires *after* the compiler has already emitted errors
    /// must surface those errors as partial diagnostics.
    #[test]
    fn stuck_result_includes_partial_diagnostics() {
        let partial = r#"{"reason":"compiler-message","package_id":"test","manifest_path":"","target":{},"message":{"rendered":"","children":[],"code":{"code":"E0308","explanation":null},"level":"error","message":"mismatched types","spans":[{"byte_end":100,"byte_start":90,"column_end":20,"column_start":10,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":5,"line_start":5,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;
        let result = format_stuck_result("check", "no output for 300 seconds", 540, partial, "");

        assert!(
            result.starts_with("STUCK:"),
            "must still be STUCK: {result}"
        );
        assert!(
            result.contains("Partial diagnostics captured"),
            "must surface partial diagnostics; got: {result}"
        );
        assert!(
            result.contains("mismatched types"),
            "must include the actual error message; got: {result}"
        );
    }

    #[test]
    fn resolve_timeout_uses_parameter_when_positive() {
        let mut params = serde_json::Map::new();
        params.insert("timeout".into(), serde_json::Value::from(120u64));
        let input = ToolInput { params };
        assert_eq!(resolve_timeout(&input), 120);
    }

    #[test]
    fn resolve_timeout_falls_back_to_default_when_zero() {
        let mut params = serde_json::Map::new();
        params.insert("timeout".into(), serde_json::Value::from(0u64));
        let input = ToolInput { params };
        // 0 is not a valid budget — must fall back to the default, not disable
        // the deadline entirely.
        assert_eq!(resolve_timeout(&input), DEFAULT_TIMEOUT_SECS);
    }

    #[test]
    fn resolve_timeout_falls_back_to_default_when_absent() {
        let input = ToolInput::default();
        assert_eq!(resolve_timeout(&input), DEFAULT_TIMEOUT_SECS);
    }
}
