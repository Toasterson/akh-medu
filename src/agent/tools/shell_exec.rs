//! Shell execution tool: run commands with timeout and output limits.
//!
//! Executes a command via the system shell (`/bin/sh -c` on Unix).
//! Enforces a timeout and maximum output size to prevent runaway processes.

use std::process::Command;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;
use std::collections::HashSet;

/// Maximum combined stdout+stderr size (64 KB).
const MAX_OUTPUT_SIZE: usize = 64 * 1024;

/// Default timeout in seconds.
const DEFAULT_TIMEOUT_SECS: u64 = 30;

/// Execute shell commands with timeout and output limits.
pub struct ShellExecTool;

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

        // Spawn the child process.
        let mut child = match Command::new("/bin/sh")
            .arg("-c")
            .arg(command_str)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(child) => child,
            Err(e) => {
                return Ok(ToolOutput::err(format!("Failed to spawn shell: {e}")));
            }
        };

        // Poll for completion with a timeout loop.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        // Timeout — kill the process.
                        let _ = child.kill();
                        let _ = child.wait();
                        return Ok(ToolOutput::err(format!(
                            "Command timed out after {timeout_secs}s: {command_str}"
                        )));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Err(e) => {
                    return Ok(ToolOutput::err(format!("Failed to wait on command: {e}")));
                }
            }
        };

        let status = match status {
            Some(s) => s,
            None => {
                return Ok(ToolOutput::err("Command ended without status."));
            }
        };

        // Read stdout and stderr.
        let stdout = child
            .stdout
            .take()
            .and_then(|mut s| {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut s, &mut buf).ok()?;
                Some(buf)
            })
            .unwrap_or_default();

        let stderr = child
            .stderr
            .take()
            .and_then(|mut s| {
                let mut buf = Vec::new();
                std::io::Read::read_to_end(&mut s, &mut buf).ok()?;
                Some(buf)
            })
            .unwrap_or_default();

        let stdout_str = String::from_utf8_lossy(&stdout);
        let stderr_str = String::from_utf8_lossy(&stderr);

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

        let exit_code = status.code().unwrap_or(-1);
        let result = format!("Command: {command_str}\nExit code: {exit_code}\n{output}");

        if status.success() {
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
}
