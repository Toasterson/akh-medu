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
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::backend::{Backend, ExecError, ExecRequest, InProcessBackend};
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

        let request = ExecRequest::shell(command_str, Duration::from_secs(timeout_secs));
        let outcome = match self.backend().exec(request) {
            Ok(outcome) => outcome,
            Err(ExecError::Spawn { source }) => {
                return Ok(ToolOutput::err(format!("Failed to spawn shell: {source}")));
            }
            Err(ExecError::Timeout { timeout, command }) => {
                return Ok(ToolOutput::err(format!(
                    "Command timed out after {}s: {command}",
                    timeout.as_secs()
                )));
            }
            Err(ExecError::Stuck { command, reason, partial_stdout, partial_stderr, elapsed_secs }) => {
                let stdout_str = String::from_utf8_lossy(&partial_stdout);
                let stderr_str = String::from_utf8_lossy(&partial_stderr);
                let mut msg = format!(
                    "Command stuck after {elapsed_secs}s: {reason} (command: {command})\n"
                );
                if !stdout_str.is_empty() {
                    let trimmed = if stdout_str.len() > MAX_OUTPUT_SIZE {
                        &stdout_str[..MAX_OUTPUT_SIZE]
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
                        &stderr_str[..MAX_OUTPUT_SIZE]
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
                     is waiting for input or stuck on a network operation."
                );
                return Ok(ToolOutput::err(msg));
            }
            Err(ExecError::Wait { source }) => {
                return Ok(ToolOutput::err(format!(
                    "Failed to wait on command: {source}"
                )));
            }
            // Only the russh `RemoteBackend` (feature `remote-executor`)
            // produces this; the rich `RemoteError` diagnostic already carries
            // the host/auth/timeout detail, so surface it verbatim.
            #[cfg(feature = "remote-executor")]
            Err(remote @ ExecError::Remote(_)) => {
                return Ok(ToolOutput::err(format!("Remote execution failed: {remote}")));
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
                    &stdout_str[..MAX_OUTPUT_SIZE]
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
                    &stderr_str[..MAX_OUTPUT_SIZE]
                ));
            } else {
                output.push_str(&format!("stderr:\n{stderr_str}\n"));
            }
        }

        let exit_code = outcome.exit_code;
        let result = format!("Command: {command_str}\nExit code: {exit_code}\n{output}");

        if outcome.success {
            Ok(ToolOutput::ok(result))
        } else {
            Ok(ToolOutput::err(result))
        }
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
}
