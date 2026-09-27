//! Shell execution tool: run commands with timeout and output limits.
//!
//! Executes a command through the session's [`Backend`] (ADR 082) — never by
//! spawning processes itself. The default backend is the in-process one
//! ([`InProcessBackend`]), which runs `/bin/sh -c` in the akhomed process
//! exactly as this tool historically did; 48b/48c swap in privilege-dropped or
//! remote executors via [`ShellExecTool::with_backend`] without changing the
//! model-facing surface. Enforces a timeout and maximum output size to prevent
//! runaway processes.

use std::sync::Arc;
use std::time::Duration;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolEffect, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::backend::{Backend, ExecError, ExecOutcome, ExecRequest, ExecResult, InProcessBackend};
use crate::engine::Engine;
use std::collections::HashSet;

/// Maximum combined stdout+stderr size (64 KB).
const MAX_OUTPUT_SIZE: usize = 64 * 1024;

/// Default timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Execute shell commands with timeout and output limits through a
/// [`Backend`].
pub struct ShellExecTool {
    /// Explicitly injected backend; `None` routes through a per-call default
    /// [`InProcessBackend`] (the historic in-process behavior).
    backend: Option<Arc<dyn Backend>>,
}

/// Source-compat unit-style value: pre-Phase-48 call sites construct the tool
/// as the bare expression `ShellExecTool`, which a field-carrying struct can
/// no longer satisfy. This same-named const keeps those call sites compiling
/// unchanged; it executes through the default in-process backend.
#[allow(non_upper_case_globals)]
pub const ShellExecTool: ShellExecTool = ShellExecTool { backend: None };

impl ShellExecTool {
    /// The tool bound to an explicit backend — the seam where 48b (local
    /// privilege drop) and 48c (russh remote) plug in.
    pub fn with_backend(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend: Some(backend),
        }
    }

    /// The backend commands execute through: the injected one, or a fresh
    /// exec-only in-process backend (file verbs denied, matching the historic
    /// tool that had no file surface).
    fn backend(&self) -> Arc<dyn Backend> {
        match &self.backend {
            Some(backend) => Arc::clone(backend),
            None => Arc::new(InProcessBackend::exec_only()),
        }
    }
}

impl Tool for ShellExecTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "shell_exec".into(),
            description:
                "Execute a shell command with timeout (default 30s) and output limit (64 KB)."
                    .into(),
            parameters: vec![
                ToolParam {
                    name: "command".into(),
                    description: "Shell command to execute.".into(),
                    required: true,
                },
                ToolParam {
                    name: "timeout".into(),
                    description: "Timeout in seconds (default: 30).".into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let command_str = input.require("command", "shell_exec")?;
        let timeout_secs: u64 = input
            .get("timeout")
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_TIMEOUT_SECS);

        let mut request = ExecRequest::shell(command_str, Duration::from_secs(timeout_secs));
        // ACP session-cwd threading: run in the client's project directory
        // when the current turn has one (thread-local, set by
        // `AkhAcpAgent::run_turn` after validating it against this tool's
        // own `backend_roots()` — see that function). Absent a session cwd
        // (no ACP caller, or one that never sent a real `cwd`), this is
        // `None` and the request is byte-identical to before session-cwd
        // threading existed: the backend falls back to its own configured
        // workdir (`ExecRequest.cwd.or(config.workdir)`).
        if let Some(cwd) = crate::fs::session_cwd() {
            request = request.with_cwd(cwd);
        }
        Ok(render_exec_result(
            command_str,
            self.backend().exec(request),
        ))
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "shell_exec".into(),
            description: "Executes arbitrary shell commands — full system access.".into(),
            parameters: vec![
                ToolParamSchema::required("command", "Shell command to execute."),
                ToolParamSchema::optional("timeout", "Timeout in seconds (default: 30)."),
            ],
            danger: DangerInfo {
                effect: ToolEffect::Mutating,
                level: DangerLevel::Critical,
                capabilities: HashSet::from([Capability::ProcessExec]),
                description: "Executes arbitrary shell commands — full system access.".into(),
                shadow_triggers: vec![
                    "exec".into(),
                    "shell".into(),
                    "command".into(),
                    "rm".into(),
                    "sudo".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }

    fn locus(&self) -> Option<crate::backend::ExecLocus> {
        Some(self.backend().locus())
    }

    fn backend_roots(&self) -> Option<Vec<std::path::PathBuf>> {
        Some(self.backend().read_roots().to_vec())
    }
}

/// The longest prefix of `text` that is at most `max` bytes and ends on a
/// UTF-8 character boundary. Slicing at a raw byte offset panics when a
/// multi-byte character straddles it (lossy-decoded output often has them).
fn truncate_at_char_boundary(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Render a [`Backend::exec`] result the way `shell_exec` reports it to the
/// model: `Command:` + `Exit code:` + stdout/stderr (each truncated at 64 KB),
/// with every [`ExecError`] mapped to an error output that keeps the
/// diagnostic's guidance (for execd: whether a retry is safe).
///
/// Shared with `sandbox_exec` so both tools report identically.
pub(crate) fn render_exec_result(command_str: &str, result: ExecResult<ExecOutcome>) -> ToolOutput {
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(ExecError::Spawn { source }) => {
            return ToolOutput::err(format!("Failed to spawn shell: {source}"));
        }
        Err(ExecError::Timeout { timeout, command }) => {
            return ToolOutput::err(format!(
                "Command timed out after {}s: {command}",
                timeout.as_secs()
            ));
        }
        Err(ExecError::Stuck {
            command,
            reason,
            partial_stdout,
            partial_stderr,
            elapsed_secs,
        }) => {
            let stdout_str = String::from_utf8_lossy(&partial_stdout);
            let stderr_str = String::from_utf8_lossy(&partial_stderr);
            let mut msg =
                format!("Command stuck after {elapsed_secs}s: {reason} (command: {command})\n");
            if !stdout_str.is_empty() {
                let trimmed = if stdout_str.len() > MAX_OUTPUT_SIZE {
                    truncate_at_char_boundary(&stdout_str, MAX_OUTPUT_SIZE)
                } else {
                    &stdout_str
                };
                msg.push_str(&format!(
                    "\nPartial stdout ({} bytes):\n{trimmed}\n",
                    stdout_str.len()
                ));
            }
            if !stderr_str.is_empty() {
                let trimmed = if stderr_str.len() > MAX_OUTPUT_SIZE {
                    truncate_at_char_boundary(&stderr_str, MAX_OUTPUT_SIZE)
                } else {
                    &stderr_str
                };
                msg.push_str(&format!(
                    "\nPartial stderr ({} bytes):\n{trimmed}\n",
                    stderr_str.len()
                ));
            }
            msg.push_str(
                "\nThe process produced no output for an extended period and was \
                 terminated. Increase the timeout or investigate whether the command \
                 is waiting for input or stuck on a network operation.",
            );
            return ToolOutput::err(msg);
        }
        Err(ExecError::Wait { source }) => {
            return ToolOutput::err(format!("Failed to wait on command: {source}"));
        }
        // Only the russh `RemoteBackend` (feature `remote-executor`)
        // produces this; the rich `RemoteError` diagnostic already carries
        // the host/auth/timeout detail, so surface it verbatim.
        #[cfg(feature = "remote-executor")]
        Err(remote @ ExecError::Remote(_)) => {
            return ToolOutput::err(format!("Remote execution failed: {remote}"));
        }
        // Only `ExecdBackend` (feature `execd-executor`) produces this.
        // Include the diagnostic help: for an unknown outcome or a lapsed
        // lease it tells the model not to retry blindly.
        #[cfg(feature = "execd-executor")]
        Err(ExecError::Execd(execd)) => {
            use miette::Diagnostic as _;
            let help = execd
                .help()
                .map(|help| format!("\n{help}"))
                .unwrap_or_default();
            return ToolOutput::err(format!("Sandboxed execution failed: {execd}{help}"));
        }
    };

    let stdout_str = String::from_utf8_lossy(&outcome.stdout);
    let stderr_str = String::from_utf8_lossy(&outcome.stderr);

    let mut output = String::new();
    if !stdout_str.is_empty() {
        if stdout_str.len() > MAX_OUTPUT_SIZE {
            output.push_str(&format!(
                "stdout ({} bytes, truncated):\n{}...\n",
                stdout_str.len(),
                truncate_at_char_boundary(&stdout_str, MAX_OUTPUT_SIZE)
            ));
        } else {
            output.push_str(&format!("stdout:\n{stdout_str}\n"));
        }
    }
    if !stderr_str.is_empty() {
        if stderr_str.len() > MAX_OUTPUT_SIZE {
            output.push_str(&format!(
                "stderr ({} bytes, truncated):\n{}...\n",
                stderr_str.len(),
                truncate_at_char_boundary(&stderr_str, MAX_OUTPUT_SIZE)
            ));
        } else {
            output.push_str(&format!("stderr:\n{stderr_str}\n"));
        }
    }

    let exit_code = outcome.exit_code;
    let result = format!("Command: {command_str}\nExit code: {exit_code}\n{output}");

    if outcome.success {
        ToolOutput::ok(result)
    } else {
        ToolOutput::err(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_never_splits_a_multibyte_character() {
        // 'é' is two bytes; put one across the 64 KiB boundary.
        let text = format!("{}é tail", "a".repeat(MAX_OUTPUT_SIZE - 1));
        let head = truncate_at_char_boundary(&text, MAX_OUTPUT_SIZE);
        assert_eq!(head.len(), MAX_OUTPUT_SIZE - 1);
        assert_eq!(truncate_at_char_boundary("short", MAX_OUTPUT_SIZE), "short");
        let outcome = ExecOutcome {
            exit_code: 0,
            success: true,
            stdout: text.clone().into_bytes(),
            stderr: Vec::new(),
        };
        let rendered = render_exec_result("x", Ok(outcome));
        assert!(rendered.result.contains("truncated"));
    }
    #[test]
    fn backend_roots_reflects_the_injected_backends_read_roots() {
        let dir = tempfile::tempdir().unwrap();
        let fs: Arc<dyn crate::fs::Filesystem> = Arc::new(crate::fs::LocalFs::sandbox(dir.path()));
        let backend: Arc<dyn Backend> = Arc::new(InProcessBackend::new(fs));
        let tool = ShellExecTool::with_backend(backend);

        let roots = tool.backend_roots().expect("a wired backend reports roots");
        assert_eq!(roots, vec![dir.path().canonicalize().unwrap()]);
    }

    #[test]
    fn backend_roots_is_empty_for_the_default_unwired_backend() {
        // `ShellExecTool` (the const) has no injected backend; its ad-hoc
        // exec-only `InProcessBackend` wraps `LocalFs::denied()`, which has
        // no read roots at all — the ACP layer must treat that as "no
        // containment boundary" and never thread a session cwd through it.
        let tool = ShellExecTool;
        assert_eq!(tool.backend_roots(), Some(Vec::new()));
    }
}
