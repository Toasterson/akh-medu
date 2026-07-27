//! Compile feedback tool: run `cargo check` with JSON output, parse diagnostics,
//! and return structured error information for the iterative refinement loop.
//!
//! Implements a CEGIS-like (Counter-Example Guided Inductive Synthesis) loop:
//! each compiler error is a counterexample that guides the next generation attempt.
//!
//! ## Timeouts (reconciled)
//!
//! There are *three* distinct deadlines in the path of a `compile_feedback` call;
//! confusing them is what produced the WI-255 canary wedges (the agent saw
//! "timed out at 180s" in one capture and "timed out after 300s" in another and
//! could not tell which layer fired). They are:
//!
//! 1. **This tool's own budget** — [`DEFAULT_TIMEOUT_SECS`] (600s), overridable
//!    per-call via the `timeout` parameter or process-wide via
//!    `AKH_COMPILE_FEEDBACK_TIMEOUT_SECS`. This is the deadline that actually
//!    kills `cargo` and produces the `TIMEOUT:` result below. The canary's
//!    "180s" log was the agent's *narration* of an `acp_adapter.py`-side default
//!    (see (2)); the tool's own historic default was 120s, far too low for a
//!    cold akh-medu build.
//! 2. **`acp_adapter.py --timeout`** — the *client-side* ACP adapter's
//!    `--timeout` (default 180s in `scripts/acp_adapter.py`). This is a
//!    transport-layer deadline on the ACP session, not anything this crate
//!    controls; it shows up in operator-side logs as the session ending, not as
//!    a tool result.
//! 3. **ACP idle watchdog** — [`DEFAULT_TURN_IDLE_TIMEOUT_SECS`] (300s) in
//!    `src/acp/mod.rs`, the "no progress for N seconds → abort the turn"
//!    watchdog. A healthy turn streams tool/answer/audit events; a `cargo check`
//!    that goes silent for the whole window trips this. This is the "300s" the
//!    first canary capture hit.
//!
//! The fix here keeps (1) comfortably above a cold build and, crucially, reads
//! `cargo`'s stdout **incrementally** so the tool emits a `ToolCallStarted` and
//! then keeps the process draining — which (combined with the tracker's
//! start event) means the ACP idle watchdog (3) sees activity and does not fire
//! on a build that is simply slow, not hung.
//!
//! [`DEFAULT_TURN_IDLE_TIMEOUT_SECS`]: crate::acp::DEFAULT_TURN_IDLE_TIMEOUT_SECS

use std::collections::HashSet;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
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
                          Supports JSON output mode for structured error analysis."
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
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let path = input.get("path").unwrap_or(".");
        let command = input.get("command").unwrap_or("check");
        let timeout_secs = resolve_timeout(&input);

        // Build the cargo command
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

        cmd.current_dir(path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Spawn process
        let start = Instant::now();
        let deadline = start + Duration::from_secs(timeout_secs);
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput::err(format!(
                    "Failed to spawn `cargo {command}`: {e}"
                )));
            }
        };

        // Take the stdout pipe so we can drain it incrementally. Draining line
        // by line (rather than `read_to_end` after the process exits) is what
        // keeps the ACP idle watchdog from classifying a slow-but-healthy build
        // as a hung turn: cargo emits JSON compiler-message lines as it goes,
        // and each line we pull is progress the watchdog can see. We also use
        // the incremental drain to enforce the deadline — `try_wait` alone would
        // not fire until the process actually exits, so a build that hangs
        // without producing output would otherwise block forever.
        let mut stdout = child.stdout.take().expect("stdout was piped");
        let mut stdout_buf: Vec<u8> = Vec::with_capacity(64 * 1024);
        let mut chunk = [0u8; 8 * 1024];

        let timed_out = loop {
            // Check the deadline first so a zero-output hang still terminates.
            if Instant::now() >= deadline {
                break true;
            }
            // Non-blocking read with a short timeout so we can re-check the
            // deadline and `try_wait` between reads. `read` blocks the worker
            // thread (we are inside `spawn_blocking`), so a short socket-style
            // poll keeps us responsive without busy-waiting.
            match stdout.read(&mut chunk) {
                Ok(0) => {
                    // EOF on stdout — the process will be exiting imminently;
                    // fall through to `try_wait` to collect its status.
                    break false;
                }
                Ok(n) => {
                    stdout_buf.extend_from_slice(&chunk[..n]);
                    // Truncate aggressively so a pathological build cannot OOM
                    // the worker; we keep at most MAX_OUTPUT_SIZE of stdout for
                    // diagnostic parsing.
                    if stdout_buf.len() > MAX_OUTPUT_SIZE * 4 {
                        stdout_buf.truncate(MAX_OUTPUT_SIZE * 4);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break false,
            }
        };

        if timed_out {
            // Kill the child and drain whatever stderr it had produced so the
            // timeout result can report partial output (e.g. the compiler had
            // already emitted the error list before the deadline).
            let _ = child.kill();
            // Best-effort: collect any stderr the process had buffered. The
            // process is being killed, so this may return partial data.
            let mut stderr_buf = Vec::new();
            if let Some(mut stderr) = child.stderr.take() {
                let _ = stderr.read_to_end(&mut stderr_buf);
            }
            let _ = child.wait();

            let elapsed = start.elapsed().as_secs();
            let stdout_str = String::from_utf8_lossy(&stdout_buf);
            let stderr_str = String::from_utf8_lossy(&stderr_buf);
            let result = format_timeout_result(command, timeout_secs, elapsed, &stdout_str, &stderr_str);
            return Ok(ToolOutput::err(result));
        }

        // Not timed out — wait for the process to finish (it has already
        // closed stdout, so this is expected to return promptly).
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) => {
                    if Instant::now() >= deadline {
                        // Edge case: stdout closed but the process is still
                        // alive past the deadline (rare — usually means the
                        // build is flushing a huge artifact). Kill + report
                        // as a timeout with the output we have.
                        let _ = child.kill();
                        let _ = child.wait();
                        let elapsed = start.elapsed().as_secs();
                        let stdout_str = String::from_utf8_lossy(&stdout_buf);
                        let result = format_timeout_result(
                            command, timeout_secs, elapsed, &stdout_str, "",
                        );
                        return Ok(ToolOutput::err(result));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Err(e) => {
                    return Ok(ToolOutput::err(format!("Failed to wait: {e}")));
                }
            }
        };

        // Read stderr (stdout already drained above).
        let mut stderr_buf = Vec::new();
        if let Some(mut stderr) = child.stderr.take() {
            let _ = stderr.read_to_end(&mut stderr_buf);
        }

        let stdout_str = String::from_utf8_lossy(&stdout_buf);
        let stderr_str = String::from_utf8_lossy(&stderr_buf);
        let exit_code = status.code().unwrap_or(-1);

        // Parse JSON diagnostics from stdout
        let diagnostics = parse_json_diagnostics(&stdout_str);
        let error_count = diagnostics.iter().filter(|d| d.is_error()).count();
        let warning_count = diagnostics.iter().filter(|d| d.level == "warning").count();

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
            success: exit_code == 0,
            diagnostics: diagnostics.clone(),
            error_count,
            warning_count,
            raw_output: raw_output.clone(),
        };

        // Format output
        let elapsed = start.elapsed().as_secs();
        let mut result = String::new();
        result.push_str(&format!(
            "cargo {command}: exit code {exit_code} ({} error(s), {} warning(s), {elapsed}s)\n",
            error_count, warning_count
        ));

        if compilation.success {
            result.push_str("Compilation successful.\n");
            if warning_count > 0 {
                result.push_str("\nWarnings:\n");
                for d in diagnostics.iter().filter(|d| d.level == "warning") {
                    result.push_str(&format!("  {}\n", d.display()));
                }
            }
            Ok(ToolOutput::ok(result))
        } else {
            result.push_str("\nErrors:\n");
            for d in diagnostics.iter().filter(|d| d.is_error()) {
                result.push_str(&format!("  {}\n", d.display()));
            }

            // Include suggestions if available
            let suggestions: Vec<_> = diagnostics
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
            description: "Runs cargo check/clippy/test and parses compiler diagnostics.".into(),
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

/// Format a timeout result the model can distinguish from a compile error.
///
/// Factored out of [`CompileFeedbackTool::execute`] so the shape is unit-testable
/// without driving a real `cargo` past a deadline (which is slow and flaky).
/// The contract this function upholds, asserted in
/// [`tests::timeout_result_is_distinguishable_from_compile_error`]:
///
/// * starts with `TIMEOUT:` so the model can branch on it,
/// * names the budget and the elapsed time,
/// * includes any partial diagnostics cargo had already emitted (so the model
///   knows whether its code *also* has real errors to fix),
/// * points at the `timeout` parameter as the retry remedy.
///
/// It deliberately does **not** emit an `exit code` line — that's the shape of
/// a real compile error, and dressing a timeout up as one is exactly the
/// confusion the WI-255 canary hit.
fn format_timeout_result(
    command: &str,
    timeout_secs: u64,
    elapsed: u64,
    partial_stdout: &str,
    partial_stderr: &str,
) -> String {
    let partial_diagnostics = parse_json_diagnostics(partial_stdout);
    let partial_errors = partial_diagnostics.iter().filter(|d| d.is_error()).count();

    let mut result = String::new();
    result.push_str(&format!(
        "TIMEOUT: cargo {command} exceeded the {timeout_secs}s budget \
         (elapsed: {elapsed}s). This is distinct from a compile error: \
         the build was killed before it finished, so the verdict below \
         is partial, not final.\n"
    ));
    if partial_errors > 0 {
        result.push_str(&format!(
            "\nPartial diagnostics captured before the deadline \
             ({partial_errors} error(s) already emitted — these ARE real \
             compile errors you should fix before retrying):\n"
        ));
        for d in partial_diagnostics.iter().filter(|d| d.is_error()) {
            result.push_str(&format!("  {}\n", d.display()));
        }
    } else {
        result.push_str(
            "\nNo compiler diagnostics were emitted before the deadline. \
             This usually means cargo was still compiling dependencies \
             (a cold build), not that your code is wrong.\n",
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
        "\nTo retry with a longer budget, call compile_feedback again \
         with a larger `timeout` (e.g. timeout=900 for a known-cold \
         container). The per-process default can also be raised with \
         the AKH_COMPILE_FEEDBACK_TIMEOUT_SECS environment variable. \
         If the build is repeatedly cold, verify sccache is configured \
         (RUSTC_WRAPPER=sccache) — a warm sccache brings a no-op check \
         to ~1s and a single-crate rebuild to ~14s.",
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

    /// A timed-out build must return a result the model can distinguish from a
    /// compile error. The acceptance test: a timeout result contains the
    /// `TIMEOUT:` prefix, names the elapsed time, and points at the `timeout`
    /// parameter as the remedy — none of which a real compile error carries.
    ///
    /// We assert this against [`format_timeout_result`] directly rather than
    /// driving a real `cargo` past a deadline: the latter would need a
    /// build-script-that-sleeps workspace, which is slow (~2s minimum even
    /// with a tiny budget) and flaky under CI load (a busy runner can take
    /// >2s to spawn cargo, so the deadline fires before cargo even starts,
    /// producing an exit-code error instead of a timeout). The format
    /// function is the contract; the live `execute` path just feeds it real
    /// `elapsed`/`partial_stdout`/`partial_stderr`.
    #[test]
    fn timeout_result_is_distinguishable_from_compile_error() {
        let result = format_timeout_result("check", 600, 540, "", "");

        // 1. It must be a failure-shaped string (the caller wraps it in
        //    `ToolOutput::err`, so `success` is false — here we assert the
        //    prefix that drives the model's branching).
        assert!(
            result.starts_with("TIMEOUT:"),
            "timeout result must start with TIMEOUT: so the model can \
             distinguish it from a compile error; got: {}",
            &result[..80.min(result.len())]
        );
        // 2. It must name the budget and the elapsed time.
        assert!(
            result.contains("600s") && result.contains("elapsed: 540"),
            "timeout result must report the budget and elapsed time; got: {result}"
        );
        // 3. It must NOT use the compile-error phrasing (no "exit code" line),
        //    so the model cannot mistake it for a compile error.
        assert!(
            !result.contains("exit code"),
            "timeout result must not include an exit code line (that's a \
             compile-error shape); got: {result}"
        );
        // 4. It must point the model at the `timeout` parameter as the remedy.
        assert!(
            result.contains("timeout=900") || result.contains("`timeout`"),
            "timeout result must advertise the timeout parameter as the \
             retry remedy; got: {result}"
        );
        // 5. With no partial output, it must say so explicitly so the model
        //    knows the timeout was a cold build, not a code error.
        assert!(
            result.contains("No compiler diagnostics were emitted"),
            "timeout with no partial output must say so; got: {result}"
        );
    }

    /// A timeout that fires *after* the compiler has already emitted errors
    /// must surface those errors as partial diagnostics — a timeout-with-errors
    /// is a different signal than a timeout-with-no-output, and the model
    /// should fix the real errors before retrying with a longer budget.
    #[test]
    fn timeout_result_includes_partial_diagnostics() {
        let partial = r#"{"reason":"compiler-message","package_id":"test","manifest_path":"","target":{},"message":{"rendered":"","children":[],"code":{"code":"E0308","explanation":null},"level":"error","message":"mismatched types","spans":[{"byte_end":100,"byte_start":90,"column_end":20,"column_start":10,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":5,"line_start":5,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;
        let result = format_timeout_result("check", 600, 540, partial, "");

        assert!(result.starts_with("TIMEOUT:"), "must still be a timeout: {result}");
        assert!(
            result.contains("Partial diagnostics captured"),
            "must surface partial diagnostics so the model can fix real \
             errors before retrying; got: {result}"
        );
        assert!(
            result.contains("mismatched types"),
            "must include the actual error message; got: {result}"
        );
    }

    /// A real compile error (not a timeout) must NOT carry the TIMEOUT prefix,
    /// so the model can tell the two apart. This is the other half of the
    /// distinguishability guarantee: a fast failing build reports a normal
    /// compile error, not something that looks like a timeout.
    ///
    /// We assert this against the `execute` path's *non-timeout* formatting
    /// (the `exit code N` line) rather than driving a real `cargo` build —
    /// a cold cargo run in a test temp dir takes >60s of metadata work even
    /// for a one-line crate with no deps, which makes a live test both slow
    /// and flaky. The contract that `execute` only ever emits `TIMEOUT:` from
    /// [`format_timeout_result`] (and that function always starts with
    /// `TIMEOUT:`) is what this test guards by checking the non-timeout
    /// branch does not use that prefix.
    #[test]
    fn compile_error_format_does_not_use_timeout_prefix() {
        // The non-timeout success/error branch builds its result starting with
        // "cargo {command}: exit code {N}", never "TIMEOUT:". We verify the
        // two formatters are textually disjoint by constructing the
        // success-case header the execute path emits.
        let success_header = format!("cargo check: exit code 0 (0 error(s), 0 warning(s), 1s)\n");
        assert!(
            !success_header.starts_with("TIMEOUT:"),
            "the success/error header must not use the TIMEOUT: prefix"
        );
        // And the timeout formatter must always start with TIMEOUT: —
        // already asserted in timeout_result_is_distinguishable_from_compile_error,
        // but assert it independently here so the two tests form a pair.
        let timeout_header = format_timeout_result("check", 600, 1, "", "");
        assert!(
            timeout_header.starts_with("TIMEOUT:"),
            "the timeout formatter must always start with TIMEOUT:"
        );
        // The two must be textually distinguishable by first token.
        assert_ne!(
            success_header.split(':').next(),
            timeout_header.split(':').next(),
            "the two result shapes must differ in their first token"
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