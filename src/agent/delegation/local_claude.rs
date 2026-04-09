//! Local Claude Code subprocess worker.
//!
//! Spawns `claude --print` as a child process with a generated `.mcp.json`
//! that points back to the running akhomed server, creating a bidirectional
//! MCP bridge: the subprocess can query and assert into the knowledge graph.

use std::path::{Path, PathBuf};
use std::time::Instant;

use super::error::{DelegationError, DelegationResult};
use super::types::{DelegationId, DelegationOutcome, DelegationOutput, DelegationTask, LocalDelegationConfig};

/// Worker that delegates tasks to a local `claude` CLI subprocess.
pub struct LocalClaudeWorker {
    mcp_config_path: PathBuf,
    timeout_seconds: u64,
}

impl LocalClaudeWorker {
    /// Create a new worker, generating the MCP config file.
    pub fn new(config: &LocalDelegationConfig, akhomed_url: String) -> Self {
        let mcp_config_path = Self::generate_mcp_config(&akhomed_url)
            .unwrap_or_else(|_| std::env::temp_dir().join("akh-delegate.mcp.json"));
        Self {
            mcp_config_path,
            timeout_seconds: config.timeout_seconds,
        }
    }

    /// Generate a temporary `.mcp.json` that points the subprocess back to akhomed.
    fn generate_mcp_config(akhomed_url: &str) -> DelegationResult<PathBuf> {
        let config = serde_json::json!({
            "mcpServers": {
                "akh-medu": {
                    "type": "url",
                    "url": format!("{}/mcp", akhomed_url)
                }
            }
        });
        let path = std::env::temp_dir().join("akh-delegate.mcp.json");
        std::fs::write(&path, serde_json::to_string_pretty(&config).map_err(|e| {
            DelegationError::McpConfig {
                message: e.to_string(),
            }
        })?).map_err(|e| DelegationError::McpConfig {
            message: e.to_string(),
        })?;
        Ok(path)
    }

    /// Execute a delegation task as a Claude subprocess.
    pub async fn execute(
        &self,
        task: &DelegationTask,
        model: &str,
        allowed_tools: &[String],
        working_dir: Option<&Path>,
    ) -> DelegationResult<DelegationOutcome> {
        let prompt = Self::build_prompt(task);
        let id = DelegationId(0); // Caller assigns real ID

        // Verify claude is available
        let claude_path = which_claude()?;

        let mut cmd = tokio::process::Command::new(&claude_path);
        cmd.args(["--print", "-p", &prompt, "--model", model]);

        // Point to MCP config for KG access
        if let Some(path_str) = self.mcp_config_path.to_str() {
            cmd.args(["--mcp-config", path_str]);
        }

        cmd.args(["--output-format", "text"]);

        for tool in allowed_tools {
            cmd.args(["--allowedTools", tool]);
        }
        // Always allow MCP tools so the subprocess can query the KG.
        cmd.args(["--allowedTools", "mcp__akh-medu__*"]);

        if let Some(dir) = working_dir {
            cmd.current_dir(dir);
        }

        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let start = Instant::now();
        let timeout = std::time::Duration::from_secs(self.timeout_seconds);

        let child = cmd.spawn().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                DelegationError::ClaudeNotFound
            } else {
                DelegationError::Io(e)
            }
        })?;

        match tokio::time::timeout(timeout, child.wait_with_output()).await {
            Ok(Ok(output)) => {
                let wall_time = start.elapsed();
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let structured = Self::parse_structured_output(&stdout);

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let exit_code = output.status.code().unwrap_or(-1);
                    return Err(DelegationError::WorkerFailed { exit_code, stderr });
                }

                Ok(DelegationOutcome {
                    id,
                    stdout,
                    structured,
                    wall_time,
                    tokens_consumed: None,
                    cost_usd: None,
                    exit_code: output.status.code(),
                    goal_id: task.goal_id,
                    error: None,
                })
            }
            Ok(Err(e)) => Err(DelegationError::Io(e)),
            Err(_) => Err(DelegationError::Timeout {
                timeout_secs: self.timeout_seconds,
                delegation_id: 0,
            }),
        }
    }

    /// Build a task prompt that instructs the worker to use MCP tools
    /// and produce structured output.
    fn build_prompt(task: &DelegationTask) -> String {
        let mut prompt = String::with_capacity(1024);

        prompt.push_str(
            "You are a worker agent delegated a task by akh-medu, \
             a neuro-symbolic AI engine. You have access to its knowledge \
             graph via MCP tools (akh-medu server).\n\n",
        );

        prompt.push_str("## Task\n\n");
        prompt.push_str(&task.description);
        prompt.push_str("\n\n");

        // Include KG context snapshot if available.
        if !task.context_triples.is_empty() {
            prompt.push_str("## Relevant Knowledge (from KG)\n\n");
            for triple in &task.context_triples {
                prompt.push_str(&format!(
                    "- {} --{}-> {}\n",
                    triple.subject, triple.predicate, triple.object
                ));
            }
            prompt.push('\n');
        }

        prompt.push_str("## Instructions\n\n");
        prompt.push_str("1. Use the akh-medu MCP tools to query relevant knowledge.\n");
        prompt.push_str("2. Perform the task.\n");
        prompt.push_str(
            "3. At the end of your response, include a structured result block:\n\n",
        );
        prompt.push_str("```akh-result\n");
        prompt.push_str("{\n");
        prompt.push_str("  \"status\": \"success\" | \"partial\" | \"failed\",\n");
        prompt.push_str("  \"summary\": \"brief description of what was done\",\n");
        prompt.push_str("  \"triples\": [\n");
        prompt.push_str(
            "    {\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\"}\n",
        );
        prompt.push_str("  ],\n");
        prompt.push_str("  \"files_modified\": [\"path/to/file\"],\n");
        prompt.push_str("  \"follow_up\": \"optional: what should happen next\"\n");
        prompt.push_str("}\n");
        prompt.push_str("```\n");

        prompt
    }

    /// Parse structured output from the `akh-result` fenced code block.
    fn parse_structured_output(stdout: &str) -> Option<DelegationOutput> {
        let marker = "```akh-result\n";
        let start = stdout.find(marker)?;
        let after_marker = start + marker.len();
        let rest = stdout.get(after_marker..)?;
        let end = rest.find("```")?;
        let json_str = rest.get(..end)?;
        serde_json::from_str(json_str).ok()
    }
}

/// Locate the `claude` binary on PATH.
fn which_claude() -> DelegationResult<PathBuf> {
    // Check common locations, then fall back to PATH lookup.
    let candidates = [
        "/usr/local/bin/claude",
        "/usr/bin/claude",
    ];
    for path in &candidates {
        let p = PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
    }
    // Try which(1)
    match std::process::Command::new("which")
        .arg("claude")
        .output()
    {
        Ok(output) if output.status.success() => {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Ok(PathBuf::from(path));
            }
        }
        _ => {}
    }
    Err(DelegationError::ClaudeNotFound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_structured_output_valid() {
        let stdout = r#"Some preamble text...

```akh-result
{
  "status": "success",
  "summary": "Created 3 test files",
  "triples": [
    {"subject": "test_module", "predicate": "has_tests", "object": "true"}
  ],
  "files_modified": ["src/tests/foo.rs"]
}
```

Some trailing text.
"#;
        let output = LocalClaudeWorker::parse_structured_output(stdout).unwrap();
        assert_eq!(output.status, "success");
        assert_eq!(output.triples.len(), 1);
        assert_eq!(output.files_modified.len(), 1);
    }

    #[test]
    fn parse_structured_output_missing() {
        let stdout = "Just regular output, no structured block.";
        assert!(LocalClaudeWorker::parse_structured_output(stdout).is_none());
    }

    #[test]
    fn build_prompt_includes_context() {
        use crate::skills::LabelTriple;
        let task = DelegationTask {
            description: "Write unit tests".into(),
            context_triples: vec![LabelTriple {
                subject: "module_a".into(),
                predicate: "has_function".into(),
                object: "calculate".into(),
                confidence: 1.0,
            }],
            required_capabilities: vec![],
            goal_id: 42,
            priority: 0.8,
        };
        let prompt = LocalClaudeWorker::build_prompt(&task);
        assert!(prompt.contains("Write unit tests"));
        assert!(prompt.contains("module_a --has_function-> calculate"));
        assert!(prompt.contains("akh-result"));
    }
}
