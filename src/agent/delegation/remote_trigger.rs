//! Remote trigger worker (Phase 36c).
//!
//! Creates one-shot remote agents via the claude.ai API. The remote worker
//! runs on Anthropic infrastructure and calls back to akhomed via its public
//! URL as an MCP server.
//!
//! **Current status**: The API endpoint for remote triggers is not yet
//! publicly documented. This implementation provides the worker structure,
//! prompt building, and polling scaffold. The `delegate_once` method will
//! return `DelegationError::RemoteTriggerUnavailable` until the API is
//! available and credentials are configured.

use std::time::{Duration, Instant};

use super::error::{DelegationError, DelegationResult};
use super::types::{DelegationId, DelegationOutcome, DelegationOutput, DelegationTask};

/// Worker that delegates tasks via the claude.ai remote trigger API.
pub struct RemoteTriggerWorker {
    akhomed_public_url: String,
    model: String,
    max_cost_cents: u64,
}

impl RemoteTriggerWorker {
    pub fn new(
        akhomed_public_url: impl Into<String>,
        model: impl Into<String>,
        max_cost_cents: u64,
    ) -> Self {
        Self {
            akhomed_public_url: akhomed_public_url.into(),
            model: model.into(),
            max_cost_cents,
        }
    }

    /// Delegate a task as a one-shot remote trigger.
    ///
    /// Builds the task prompt and submits it to the claude.ai trigger API.
    /// Polls for completion (or awaits a webhook from akhomed).
    pub async fn delegate_once(
        &self,
        task: &DelegationTask,
        id: DelegationId,
    ) -> DelegationResult<DelegationOutcome> {
        // Remote trigger API not yet publicly available.
        // When available, the implementation will:
        //   1. POST to https://api.claude.ai/v1/code/triggers with:
        //      { name, prompt, mcp_servers: [{type:"url", url: akhomed_url/mcp}] }
        //   2. Trigger a run via POST .../run
        //   3. Poll GET .../runs/{run_id} until status == "completed"
        //   4. Parse the run output for akh-result block
        //
        // For now, return an unavailable error so callers can fall back to local.
        let _ = (task, id, &self.akhomed_public_url, &self.model, self.max_cost_cents);
        Err(DelegationError::RemoteTriggerUnavailable)
    }

    /// Create a recurring remote trigger (cron-scheduled).
    ///
    /// Returns the trigger ID that can be used to cancel or inspect it later.
    pub async fn delegate_recurring(
        &self,
        task: &DelegationTask,
        cron: &str,
    ) -> DelegationResult<String> {
        let _ = (task, cron, &self.akhomed_public_url, &self.model);
        Err(DelegationError::RemoteTriggerUnavailable)
    }

    /// Build the task prompt for a remote worker.
    pub fn build_prompt(&self, task: &DelegationTask) -> String {
        let mut prompt = String::with_capacity(1024);
        prompt.push_str(
            "You are a remote worker agent delegated a task by akh-medu, \
             a neuro-symbolic AI engine. You have access to its knowledge \
             graph via MCP tools (akh-medu server at the configured endpoint).\n\n",
        );
        prompt.push_str("## Task\n\n");
        prompt.push_str(&task.description);
        prompt.push_str("\n\n");

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

        push_output_instructions(&mut prompt);
        prompt
    }
}

/// Polling helper — calls a status endpoint until completion or timeout.
#[allow(dead_code)]
async fn poll_completion(
    _trigger_id: &str,
    timeout: Duration,
) -> DelegationResult<String> {
    let _deadline = Instant::now() + timeout;
    // Polling implementation placeholder.
    // Real implementation: GET https://api.claude.ai/v1/code/triggers/{id}/runs/{run_id}
    // and return the output text when status == "completed".
    Err(DelegationError::RemoteTriggerUnavailable)
}

/// Parse the structured `akh-result` block from worker output.
pub fn parse_structured_output(stdout: &str) -> Option<DelegationOutput> {
    let marker = "```akh-result\n";
    let start = stdout.find(marker)?;
    let after_marker = start + marker.len();
    let rest = stdout.get(after_marker..)?;
    let end = rest.find("```")?;
    let json_str = rest.get(..end)?;
    serde_json::from_str(json_str).ok()
}

fn push_output_instructions(prompt: &mut String) {
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
    prompt.push_str("  \"triples\": [{\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\"}],\n");
    prompt.push_str("  \"files_modified\": [\"path/to/file\"],\n");
    prompt.push_str("  \"follow_up\": \"optional: what should happen next\"\n");
    prompt.push_str("}\n");
    prompt.push_str("```\n");
}
