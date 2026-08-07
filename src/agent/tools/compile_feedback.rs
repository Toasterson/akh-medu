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

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::backend::{ProgressTracker, SupervisedExecution, SupervisorError};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
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

/// Maximum pre-existing warnings to report before applying the lint budget.
const MAX_PREEXISTING_WARNINGS: usize = 50;

// ---------------------------------------------------------------------------
// Blast radius detection
// ---------------------------------------------------------------------------

/// Error type for git operations.
#[derive(Debug, thiserror::Error)]
pub enum GitError {
    #[error("git command failed to spawn: {0}")]
    Spawn(#[from] std::io::Error),
    #[error("git command exited with non-zero status: {0}")]
    NonZeroExit(String),
    #[error("git output was not valid UTF-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

/// Get list of changed files from git diff in the worktree.
/// Returns Result so callers can fail open to --workspace on git errors.
fn get_changed_files(workdir: &str) -> Result<Vec<String>, GitError> {
    let output = Command::new("git")
        .args(["diff", "--name-only", "HEAD"])
        .current_dir(workdir)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(GitError::NonZeroExit(stderr));
    }

    let files = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|l| !l.is_empty())
        .map(|s| s.to_string())
        .collect();
    
    Ok(files)
}

/// Find the Cargo.toml nearest to a file path and extract the package name.
fn file_to_crate(workdir: &str, file_path: &str) -> Option<String> {
    let mut current = PathBuf::from(workdir);
    current.push(file_path);
    
    // Walk up from the file's directory to find Cargo.toml
    let mut dir = current.parent()?.to_path_buf();
    
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Some(name) = extract_package_name(&cargo_toml) {
                return Some(name);
            }
        }
        
        // Can't go higher than workdir
        if !dir.pop() {
            break;
        }
        if dir < PathBuf::from(workdir) {
            break;
        }
    }
    
    None
}

/// Extract package name from a Cargo.toml file.
fn extract_package_name(cargo_toml: &Path) -> Option<String> {
    let content = fs::read_to_string(cargo_toml).ok()?;
    
    // Simple parsing: look for `name = "..."` under [package]
    let mut in_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "[package]" {
            in_package = true;
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_package = false;
            continue;
        }
        if in_package && trimmed.starts_with("name") {
            if let Some(eq_pos) = trimmed.find('=') {
                let value = trimmed[eq_pos + 1..].trim();
                // Remove quotes
                let name = value.trim_matches('"').trim_matches('\'');
                if !name.is_empty() {
                    return Some(name.to_string());
                }
            }
        }
    }
    
    // For workspace root without [package], return None
    None
}

/// Map file paths to crate names.
fn files_to_crates(workdir: &str, files: &[String]) -> Vec<String> {
    let mut crates = HashSet::new();
    for file in files {
        if let Some(crate_name) = file_to_crate(workdir, file) {
            crates.insert(crate_name);
        }
    }
    crates.into_iter().collect()
}

/// Load baseline diagnostics from a file (JSON format).
/// Key includes file and line to avoid false dedup across locations.
fn load_baseline(baseline_path: &str) -> HashSet<(Option<String>, Option<u32>, String, String, Option<String>)> {
    let content = match fs::read_to_string(baseline_path) {
        Ok(c) => c,
        Err(_) => return HashSet::new(),
    };
    
    let mut baseline = HashSet::new();
    for line in content.lines() {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
            if parsed.get("reason").and_then(|r| r.as_str()) == Some("compiler-message") {
                if let Some(message) = parsed.get("message") {
                    let level = message.get("level").and_then(|l| l.as_str()).unwrap_or("").to_string();
                    let msg = message.get("message").and_then(|m| m.as_str()).unwrap_or("").to_string();
                    let code = message
                        .get("code")
                        .and_then(|c| c.get("code"))
                        .and_then(|c| c.as_str())
                        .map(|s| s.to_string());
                    
                    // Extract file and line from spans
                    let (file, line) = message
                        .get("spans")
                        .and_then(|s| s.as_array())
                        .and_then(|arr| arr.first())
                        .map(|span| {
                            (
                                span.get("file_name").and_then(|f| f.as_str()).map(|s| s.to_string()),
                                span.get("line_start").and_then(|l| l.as_u64()).map(|l| l as u32),
                            )
                        })
                        .unwrap_or((None, None));
                    
                    // Skip non-warnings/errors for baseline
                    if level == "warning" || level == "error" {
                        baseline.insert((file, line, level, msg, code));
                    }
                }
            }
        }
    }
    
    baseline
}

/// Filter diagnostics: separate new issues from pre-existing baseline.
/// Uses file+line+level+code as dedup key to avoid false matches across locations.
/// Uses file_to_crate for blast-radius matching instead of hardcoded path prefixes.
fn filter_by_baseline(
    workdir: &str,
    diagnostics: Vec<CompilerDiagnostic>,
    baseline: &HashSet<(Option<String>, Option<u32>, String, String, Option<String>)>,
    blast_radius_crates: Option<&HashSet<String>>,
) -> (Vec<CompilerDiagnostic>, Vec<CompilerDiagnostic>, usize) {
    let mut new_diagnostics = Vec::new();
    let mut preexisting = Vec::new();
    let mut preexisting_outside_radius = 0;
    
    for diag in diagnostics {
        let key = (diag.file.clone(), diag.line, diag.level.clone(), diag.message.clone(), diag.code.clone());
        
        if baseline.contains(&key) {
            // Pre-existing
            if let Some(ref crates) = blast_radius_crates {
                // Use file_to_crate to resolve the file's owning package
                let file_crate = diag.file.as_ref().and_then(|f| file_to_crate(workdir, f));
                let is_inside = file_crate.as_ref().map(|c| crates.contains(c)).unwrap_or(false);
                if !is_inside {
                    preexisting_outside_radius += 1;
                    continue; // Don't report pre-existing outside blast radius
                }
            }
            preexisting.push(diag);
        } else {
            new_diagnostics.push(diag);
        }
    }
    
    (new_diagnostics, preexisting, preexisting_outside_radius)
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
                          By default, scopes to crates affected by the current worktree diff \
                          (blast-radius mode); pass `workspace=true` to check the entire workspace."
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
                    description: "Check entire workspace instead of scoping to affected crates. \
                                  Default: false (blast-radius mode).".into(),
                    required: false,
                },
                ToolParam {
                    name: "baseline_path".into(),
                    description: "Optional path to a baseline diagnostics file. If provided, \
                                  pre-existing warnings outside the blast radius are reported \
                                  as context rather than as work.".into(),
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
        let baseline_path = input.get("baseline_path");

        // Determine blast radius: which crates to check
        let blast_radius_crates: Option<HashSet<String>> = if workspace_mode {
            None // None means check all crates (--workspace)
        } else {
            // Blast-radius mode: detect from git diff
            let changed_files = match get_changed_files(path) {
                Ok(files) => files,
                Err(e) => {
                    // Fail open to workspace mode on git errors
                    // This prevents silently skipping compile check when git fails
                    return Ok(ToolOutput::ok(format!(
                        "Git diff failed ({e}); failing open to workspace check. \
                         Run `cargo check --workspace` manually to verify build health."
                    )));
                }
            };
            let crates = files_to_crates(path, &changed_files);
            
            // If no changed files, check nothing (skip)
            if crates.is_empty() && changed_files.is_empty() {
                return Ok(ToolOutput::ok(
                    "No changed files detected in worktree; skipping compile check. \
                     Pass workspace=true to check the entire workspace."
                ));
            }
            
            // If we have changed files but couldn't map them to crates, 
            // fall back to checking the root package
            if crates.is_empty() {
                Some(HashSet::from(["akh-medu".to_string()]))
            } else {
                Some(crates.into_iter().collect())
            }
        };

        // Load baseline if provided
        let baseline = baseline_path
            .map(|p| load_baseline(p))
            .unwrap_or_default();

        // Build the cargo command with blast-radius scoping
        let mut cmd = Command::new("cargo");
        match command {
            "check" => {
                cmd.arg("check").arg("--message-format=json");
            }
            "clippy" => {
                cmd.arg("clippy").arg("--message-format=json");
            }
            "test" => {
                cmd.arg("test").arg("--no-run").arg("--message-format=json");
            }
            other => {
                return Ok(ToolOutput::err(format!(
                    "Unsupported command: '{other}'. Use 'check', 'clippy', or 'test'."
                )));
            }
        }

        // Add crate scoping if not in workspace mode
        if let Some(ref crates) = blast_radius_crates {
            for c in crates {
                cmd.arg("-p").arg(c);
            }
        } else {
            // Workspace mode: check all crates
            cmd.arg("--workspace");
        }

        cmd.current_dir(path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Spawn process under supervision
        let start = Instant::now();
        let mut exec = match SupervisedExecution::spawn(cmd) {
            Ok(e) => e,
            Err(SupervisorError::Spawn { source }) => {
                return Ok(ToolOutput::err(format!(
                    "Failed to spawn `cargo {command}`: {source}"
                )));
            }
            Err(e) => {
                return Ok(ToolOutput::err(format!(
                    "Supervision error spawning `cargo {command}`: {e}"
                )));
            }
        };

        // Progress tracker: no output for timeout_secs = stuck
        let mut tracker = ProgressTracker::with_timeout(Duration::from_secs(timeout_secs));

        // Take the drain threads
        let drain = exec.take_drain();

        // Wait with drain-based supervision
        let (status, output) =
            match exec.wait_with_drain_capped(&mut tracker, drain, MAX_OUTPUT_SIZE * 4) {
                Ok((s, o)) => (s, o),
                Err(SupervisorError::Killed {
                    reason,
                    partial_stdout,
                    partial_stderr,
                    elapsed,
                }) => {
                    let stdout_str = String::from_utf8_lossy(&partial_stdout);
                    let stderr_str = String::from_utf8_lossy(&partial_stderr);
                    let result = format_stuck_result(
                        command,
                        &reason,
                        elapsed.as_secs(),
                        &stdout_str,
                        &stderr_str,
                    );
                    return Ok(ToolOutput::err(result));
                }
                Err(SupervisorError::Wait { source }) => {
                    return Ok(ToolOutput::err(format!("Failed to wait: {source}")));
                }
                Err(e) => {
                    return Ok(ToolOutput::err(format!("Supervision error: {e}")));
                }
            };

        let stdout_str = String::from_utf8_lossy(&output.stdout);
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        let exit_code = status.code().unwrap_or(-1);

        // Parse JSON diagnostics from stdout
        let all_diagnostics = parse_json_diagnostics(&stdout_str);

        // Filter by baseline and blast radius
        let (new_diagnostics, preexisting, preexisting_outside_radius) = filter_by_baseline(
            path,
            all_diagnostics,
            &baseline,
            blast_radius_crates.as_ref(),
        );

        let new_error_count = new_diagnostics.iter().filter(|d| d.is_error()).count();
        let new_warning_count = new_diagnostics.iter().filter(|d| d.level == "warning").count();
        let preexisting_error_count = preexisting.iter().filter(|d| d.is_error()).count();
        let preexisting_warning_count = preexisting.iter().filter(|d| d.level == "warning").count();

        let raw_output = if stderr_str.len() > MAX_OUTPUT_SIZE {
            format!(
                "{}... [truncated at {} bytes]",
                &stderr_str[..MAX_OUTPUT_SIZE],
                stderr_str.len()
            )
        } else {
            stderr_str.to_string()
        };

        let compilation = CompilationResult {
            success: exit_code == 0 && new_error_count == 0,
            diagnostics: new_diagnostics.clone(),
            error_count: new_error_count,
            warning_count: new_warning_count,
            raw_output: raw_output.clone(),
        };

        // Format output with blast-radius context
        let elapsed = start.elapsed().as_secs();
        let mut result = String::new();
        
        // Report blast radius
        if let Some(ref crates) = blast_radius_crates {
            let crate_list: Vec<_> = crates.iter().collect();
            result.push_str(&format!("BLAST RADIUS: Checking crates [{:?}]\n", crate_list));
        } else {
            result.push_str("MODE: Full workspace check (--workspace)\n");
        }

        // Report pre-existing warnings outside blast radius (lint budget)
        if preexisting_outside_radius > 0 {
            let reported = preexisting_outside_radius.min(MAX_PREEXISTING_WARNINGS);
            result.push_str(&format!(
                "PRE-EXISTING: {} warning(s) outside blast radius (showing {} of {})\n",
                preexisting_outside_radius, reported, preexisting_outside_radius
            ));
            if preexisting_outside_radius > MAX_PREEXISTING_WARNINGS {
                result.push_str(&format!(
                    "  (lint budget exceeded; {} warning(s) suppressed)\n",
                    preexisting_outside_radius - MAX_PREEXISTING_WARNINGS
                ));
            }
        }

        // Main summary: only count NEW issues as "work"
        result.push_str(&format!(
            "cargo {command}: exit code {exit_code} ({} NEW error(s), {} NEW warning(s), {elapsed}s)\n",
            new_error_count, new_warning_count
        ));

        if preexisting_error_count > 0 || preexisting_warning_count > 0 {
            result.push_str(&format!(
                "  (plus {} pre-existing error(s), {} pre-existing warning(s) in blast radius)\n",
                preexisting_error_count, preexisting_warning_count
            ));
        }

        if compilation.success && new_error_count == 0 {
            result.push_str("\n✓ No NEW errors in blast radius.\n");
            if new_warning_count > 0 {
                result.push_str("\nNEW Warnings (your change may have introduced these):\n");
                for d in new_diagnostics.iter().filter(|d| d.level == "warning") {
                    result.push_str(&format!("  {}\n", d.display()));
                }
            }
            if preexisting_outside_radius > 0 {
                result.push_str("\nNote: Pre-existing warnings outside blast radius are not shown. \
                                 Focus on fixing issues within your blast radius first.\n");
            }
            Ok(ToolOutput::ok(result))
        } else {
            result.push_str("\nNEW Errors (your change broke this):\n");
            for d in new_diagnostics.iter().filter(|d| d.is_error()) {
                result.push_str(&format!("  {}\n", d.display()));
            }

            // Include pre-existing errors in blast radius as context
            if preexisting_error_count > 0 {
                result.push_str("\nPre-existing errors in blast radius (not your fault, but blocking):\n");
                for d in preexisting.iter().filter(|d| d.is_error()).take(10) {
                    result.push_str(&format!("  {}\n", d.display()));
                }
                if preexisting_error_count > 10 {
                    result.push_str(&format!("  ... and {} more\n", preexisting_error_count - 10));
                }
            }

            // Include suggestions if available
            let suggestions: Vec<_> = new_diagnostics
                .iter()
                .filter_map(|d| d.suggestion.as_ref())
                .collect();
            if !suggestions.is_empty() {
                result.push_str("\nSuggested fixes:\n");
                for s in &suggestions {
                    result.push_str(&format!("  - {s}\n"));
                }
            }

            // Truncate if too long
            if result.len() > MAX_OUTPUT_SIZE {
                result.truncate(MAX_OUTPUT_SIZE);
                result.push_str("\n... [truncated]");
            }

            Ok(ToolOutput::err(result))
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "compile_feedback".into(),
            description: "Runs cargo check/clippy/test and parses compiler diagnostics. \
                          By default, scopes to crates affected by the current worktree diff \
                          (blast-radius mode); pass workspace=true to check the entire workspace."
                .into(),
            parameters: vec![
                ToolParamSchema::optional(
                    "path",
                    "Working directory for cargo.",
                ),
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
                    "Check entire workspace instead of scoping to affected crates. \
                     Default: false (blast-radius mode).",
                ),
                ToolParamSchema::optional(
                    "baseline_path",
                    "Optional path to a baseline diagnostics file. If provided, \
                     pre-existing warnings outside the blast radius are reported \
                     as context rather than as work.",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::ProcessExec]),
                description: "Runs cargo check — process execution with no side effects.".into(),
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
            .and_then(|spans| spans.iter().find(|s| s.get("is_primary") == Some(&serde_json::Value::Bool(true))))
            .map(|span| {
                let file = span.get("file_name").and_then(|f| f.as_str()).map(|s| s.to_string());
                let line = span.get("line_start").and_then(|l| l.as_u64()).map(|l| l as u32);
                let col = span.get("column_start").and_then(|c| c.as_u64()).map(|c| c as u32);
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

    /// A stuck process must return a result distinguishable from both compile
    /// errors and simple timeouts. The acceptance test: a stuck result contains
    /// the `STUCK:` prefix, names the elapsed time and reason, and includes any
    /// partial output — none of which a real compile error carries.
    #[test]
    fn stuck_result_is_distinguishable_from_compile_error() {
        let result = format_stuck_result("check", "no output for 300 seconds (appears stuck)", 540, "", "");

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

        assert!(result.starts_with("STUCK:"), "must still be STUCK: {result}");
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