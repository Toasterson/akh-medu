//! Compile feedback tool: run `cargo check` with JSON output, parse diagnostics,
//! and return structured error information for the iterative refinement loop.
//!
//! Implements a CEGIS-like (Counter-Example Guided Inductive Synthesis) loop:
//! each compiler error is a counterexample that guides the next generation attempt.

use std::collections::HashSet;
use std::io::Read;
use std::process::{Command, Stdio};

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

/// Default timeout for cargo check (120 seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Maximum output size to capture.
const MAX_OUTPUT_SIZE: usize = 64 * 1024;

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
                    description: "Timeout in seconds (default: 120).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let path = input.get("path").unwrap_or(".");
        let command = input.get("command").unwrap_or("check");
        let timeout_secs: u64 = input
            .get("timeout")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

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
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return Ok(ToolOutput::err(format!(
                    "Failed to spawn `cargo {command}`: {e}"
                )));
            }
        };

        // Poll-based timeout
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Ok(ToolOutput::err(format!(
                            "cargo {command} timed out after {timeout_secs}s"
                        )));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
                Err(e) => {
                    return Ok(ToolOutput::err(format!("Failed to wait: {e}")));
                }
            }
        };

        // Read stdout (JSON messages) and stderr (human-readable)
        let mut stdout_buf = Vec::new();
        if let Some(mut stdout) = child.stdout.take() {
            let _ = stdout.read_to_end(&mut stdout_buf);
        }

        let mut stderr_buf = Vec::new();
        if let Some(mut stderr) = child.stderr.take() {
            let _ = stderr.read_to_end(&mut stderr_buf);
        }

        let stdout_str = String::from_utf8_lossy(&stdout_buf);
        let stderr_str = String::from_utf8_lossy(&stderr_buf);
        let exit_code = status.and_then(|s| s.code()).unwrap_or(-1);

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
        let mut result = String::new();
        result.push_str(&format!(
            "cargo {command}: exit code {exit_code} ({} error(s), {} warning(s))\n",
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
                ToolParamSchema::optional("timeout", "Timeout in seconds."),
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
}
