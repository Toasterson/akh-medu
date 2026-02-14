//! CLI tool system: external binary tools with danger manifests.
//!
//! CLI tools execute binaries directly (no shell) with per-binary danger
//! metadata, argument schema mapping, and timeout enforcement.

use std::collections::HashMap;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::engine::Engine;

use super::error::{AgentError, AgentResult};
use super::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use super::tool_manifest::{DangerInfo, ToolManifest, ToolParamSchema, ToolSource};

/// Manifest for a CLI tool: describes the binary, its arguments, danger, and limits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliToolManifest {
    /// Unique tool name.
    pub name: String,
    /// What this tool does.
    pub description: String,
    /// Path to the executable binary.
    pub binary: String,
    /// Argument schema: how ToolInput params map to CLI arguments.
    pub arguments: Vec<CliArgSchema>,
    /// Danger metadata for this binary.
    pub danger: DangerInfo,
    /// Execution timeout in seconds (default: 30).
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Maximum combined stdout+stderr size in bytes (default: 64 KB).
    #[serde(default = "default_max_output")]
    pub max_output_bytes: usize,
    /// Environment variables to set for the process.
    #[serde(default)]
    pub env: HashMap<String, String>,
}

fn default_timeout() -> u64 {
    30
}

fn default_max_output() -> usize {
    64 * 1024
}

/// Schema for a single CLI argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliArgSchema {
    /// Maps to a key in `ToolInput.params`.
    pub name: String,
    /// How this argument is passed on the command line.
    pub arg_type: CliArgType,
    /// The flag string (e.g., "--output", "-o"). Required for Flag/FlagBool types.
    #[serde(default)]
    pub flag: Option<String>,
    /// Human-readable description.
    pub description: String,
    /// Whether this argument must be provided.
    pub required: bool,
}

/// How a CLI argument is passed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CliArgType {
    /// Positional argument: value is appended directly.
    Positional,
    /// Flag with value: `--flag value`.
    Flag,
    /// Boolean flag: `--flag` if truthy, omitted otherwise.
    FlagBool,
}

/// A tool that executes an external binary directly (no shell).
pub struct CliTool {
    manifest: CliToolManifest,
    skill_id: Option<String>,
}

impl CliTool {
    /// Create a new CLI tool from its manifest.
    pub fn new(manifest: CliToolManifest, skill_id: Option<String>) -> Self {
        Self { manifest, skill_id }
    }

    /// Build the argument list from ToolInput params using the argument schema.
    fn build_args(&self, input: &ToolInput) -> AgentResult<Vec<String>> {
        let mut args = Vec::new();
        for schema in &self.manifest.arguments {
            let value = input.get(&schema.name);

            match (&schema.arg_type, value) {
                (CliArgType::Positional, Some(v)) => {
                    args.push(v.to_string());
                }
                (CliArgType::Flag, Some(v)) => {
                    if let Some(ref flag) = schema.flag {
                        args.push(flag.clone());
                        args.push(v.to_string());
                    }
                }
                (CliArgType::FlagBool, Some(v)) => {
                    let truthy = matches!(v, "true" | "1" | "yes");
                    if truthy {
                        if let Some(ref flag) = schema.flag {
                            args.push(flag.clone());
                        }
                    }
                }
                (_, None) if schema.required => {
                    return Err(AgentError::CliToolError {
                        tool_name: self.manifest.name.clone(),
                        message: format!("missing required argument: {}", schema.name),
                    });
                }
                _ => {}
            }
        }
        Ok(args)
    }
}

impl Tool for CliTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: self.manifest.name.clone(),
            description: self.manifest.description.clone(),
            parameters: self
                .manifest
                .arguments
                .iter()
                .map(|a| ToolParam {
                    name: a.name.clone(),
                    description: a.description.clone(),
                    required: a.required,
                })
                .collect(),
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let args = self.build_args(&input)?;

        // Spawn the child process directly (no shell).
        let mut cmd = Command::new(&self.manifest.binary);
        cmd.args(&args)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Set environment variables.
        for (key, value) in &self.manifest.env {
            cmd.env(key, value);
        }

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => {
                return Ok(ToolOutput::err(format!(
                    "Failed to spawn {}: {e}",
                    self.manifest.binary
                )));
            }
        };

        // Poll for completion with timeout.
        let deadline =
            std::time::Instant::now() + std::time::Duration::from_secs(self.manifest.timeout_secs);

        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Ok(ToolOutput::err(format!(
                            "Command timed out after {}s: {} {}",
                            self.manifest.timeout_secs,
                            self.manifest.binary,
                            args.join(" ")
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
            None => return Ok(ToolOutput::err("Command ended without status.")),
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

        let max = self.manifest.max_output_bytes;
        let mut output = String::new();

        if !stdout_str.is_empty() {
            if stdout_str.len() > max {
                output.push_str(&format!(
                    "stdout ({} bytes, truncated):\n{}...\n",
                    stdout_str.len(),
                    &stdout_str[..max]
                ));
            } else {
                output.push_str(&format!("stdout:\n{stdout_str}\n"));
            }
        }
        if !stderr_str.is_empty() {
            if stderr_str.len() > max {
                output.push_str(&format!(
                    "stderr ({} bytes, truncated):\n{}...\n",
                    stderr_str.len(),
                    &stderr_str[..max]
                ));
            } else {
                output.push_str(&format!("stderr:\n{stderr_str}\n"));
            }
        }

        let exit_code = status.code().unwrap_or(-1);
        let result = format!(
            "CLI: {} {}\nExit code: {exit_code}\n{output}",
            self.manifest.binary,
            args.join(" ")
        );

        if status.success() {
            Ok(ToolOutput::ok(result))
        } else {
            Ok(ToolOutput::err(result))
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: self.manifest.name.clone(),
            description: self.manifest.description.clone(),
            parameters: self
                .manifest
                .arguments
                .iter()
                .map(|a| ToolParamSchema {
                    name: a.name.clone(),
                    description: a.description.clone(),
                    required: a.required,
                    param_type: "string".into(),
                })
                .collect(),
            danger: self.manifest.danger.clone(),
            source: ToolSource::Cli {
                binary_path: self.manifest.binary.clone(),
                skill_id: self.skill_id.clone(),
            },
        }
    }
}
