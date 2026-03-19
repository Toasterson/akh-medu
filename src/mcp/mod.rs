//! MCP (Model Context Protocol) server integration for akhomed.
//!
//! Exposes akh-medu's knowledge engine as MCP tools so that AI assistants
//! (e.g. Claude Code) can query, mutate, and manage the knowledge graph
//! via the standard MCP protocol over HTTP.
//!
//! The [`AkhMcpServer`] struct implements rmcp's `ServerHandler` trait and
//! is mounted at `/mcp` on the akhomed axum router via `StreamableHttpService`.
//!
//! All workspace-scoped tools accept an optional `workspace` parameter that
//! defaults to `"default"`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::schemars;
use rmcp::{ErrorData as McpError, tool, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::sync::RwLock;

use crate::agent::{Agent, AgentConfig};
use crate::engine::{Engine, EngineConfig};
use crate::message::AkhMessage;
use crate::paths::AkhPaths;
use crate::vsa::Dimension;

// ── Shared state ─────────────────────────────────────────────────────────

fn default_workspace() -> String {
    "default".into()
}

/// Shared state needed by MCP tool handlers.
///
/// Holds Arc-wrapped workspace and agent maps shared with `ServerState`,
/// enabling lazy-loaded access to any workspace without pre-warming.
pub struct McpState {
    pub paths: AkhPaths,
    pub workspaces: Arc<RwLock<HashMap<String, Arc<Engine>>>>,
    pub agents: Arc<RwLock<HashMap<String, Arc<Mutex<Agent>>>>>,
    /// Cached NLU pipelines per workspace — avoids reloading 1GB+ LLM on every chat call.
    pub nlu_pipelines: RwLock<HashMap<String, Arc<Mutex<crate::nlu::NluPipeline>>>>,
}

impl McpState {
    pub fn new(
        paths: AkhPaths,
        workspaces: Arc<RwLock<HashMap<String, Arc<Engine>>>>,
        agents: Arc<RwLock<HashMap<String, Arc<Mutex<Agent>>>>>,
    ) -> Self {
        Self {
            paths,
            workspaces,
            agents,
            nlu_pipelines: RwLock::new(HashMap::new()),
        }
    }

    /// Get or lazily create a shared NLU pipeline for the given workspace.
    pub async fn get_nlu_pipeline(
        &self,
        name: &str,
        engine: &Engine,
    ) -> Arc<Mutex<crate::nlu::NluPipeline>> {
        {
            let pipelines = self.nlu_pipelines.read().await;
            if let Some(pipeline) = pipelines.get(name) {
                return Arc::clone(pipeline);
            }
        }

        let data_dir = engine.config().data_dir.clone();
        let ranker_bytes = engine
            .store()
            .get_meta(b"nlu_ranker_state")
            .ok()
            .flatten();

        let pipeline = tokio::task::spawn_blocking(move || {
            let data_dir_ref = data_dir.as_deref();
            ranker_bytes
                .and_then(|bytes| crate::nlu::parse_ranker::ParseRanker::from_bytes(&bytes))
                .map(|ranker| {
                    crate::nlu::NluPipeline::with_ranker_and_models(ranker, data_dir_ref)
                })
                .unwrap_or_else(|| crate::nlu::NluPipeline::new_with_models(data_dir_ref))
        })
        .await
        .unwrap_or_else(|_| crate::nlu::NluPipeline::new());

        tracing::info!(workspace = %name, "shared NLU pipeline created (MCP)");
        let shared = Arc::new(Mutex::new(pipeline));
        let mut pipelines = self.nlu_pipelines.write().await;
        let entry = pipelines
            .entry(name.to_string())
            .or_insert(Arc::clone(&shared));
        Arc::clone(entry)
    }

    /// Get or lazily open an engine for the given workspace.
    pub async fn get_engine(&self, name: &str) -> Result<Arc<Engine>, McpError> {
        // Fast path: already loaded.
        {
            let map = self.workspaces.read().await;
            if let Some(engine) = map.get(name) {
                return Ok(Arc::clone(engine));
            }
        }

        // Slow path: open workspace from disk.
        let ws_paths = self.paths.workspace(name);
        if !ws_paths.root.exists() {
            return Err(McpError::invalid_params(
                format!("workspace \"{name}\" not found"),
                None,
            ));
        }

        let config = EngineConfig {
            dimension: Dimension::DEFAULT,
            data_dir: Some(ws_paths.kg_dir.clone()),
            compartments_dir: Some(ws_paths.compartments_dir.clone()),
            ..Default::default()
        };

        let engine = Engine::new(config).map_err(|e| {
            McpError::internal_error(
                format!("failed to open workspace \"{name}\": {e}"),
                None,
            )
        })?;

        let engine = Arc::new(engine);
        let mut map = self.workspaces.write().await;
        map.insert(name.to_string(), Arc::clone(&engine));
        Ok(engine)
    }

    /// Get or lazily create a shared Agent for the given workspace.
    pub async fn get_agent(&self, name: &str) -> Result<Arc<Mutex<Agent>>, McpError> {
        // Fast path: already cached.
        {
            let agents = self.agents.read().await;
            if let Some(agent) = agents.get(name) {
                return Ok(Arc::clone(agent));
            }
        }

        // Slow path: create agent.
        let engine = self.get_engine(name).await?;
        let agent = tokio::task::spawn_blocking({
            let engine = Arc::clone(&engine);
            move || {
                let config = AgentConfig::default();
                if Agent::has_persisted_session(&engine) {
                    Agent::resume(engine, config)
                } else {
                    Agent::new(engine, config)
                }
            }
        })
        .await
        .map_err(|e| McpError::internal_error(format!("agent task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(format!("failed to create agent: {e}"), None))?;

        let shared = Arc::new(Mutex::new(agent));
        let mut agents = self.agents.write().await;
        // Another request may have raced us — keep the first one.
        let entry = agents
            .entry(name.to_string())
            .or_insert(Arc::clone(&shared));
        Ok(Arc::clone(entry))
    }
}

// ── Tool parameter structs ──────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WorkspaceParam {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AskParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Natural-language question to investigate")]
    pub question: String,
    #[schemars(description = "Maximum OODA cycles to run (default: 10)")]
    pub max_cycles: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SparqlParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "SPARQL SELECT query string")]
    pub query: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Search term (entity name or partial match)")]
    pub term: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AssertTripleParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Subject entity (e.g. 'Sun')")]
    pub subject: String,
    #[schemars(description = "Predicate relation (e.g. 'is-a')")]
    pub predicate: String,
    #[schemars(description = "Object entity (e.g. 'Star')")]
    pub object: String,
    #[schemars(description = "Confidence 0.0-1.0 (default: 0.9)")]
    pub confidence: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestTextParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Text to extract knowledge from")]
    pub text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestUrlParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "URL to fetch and ingest")]
    pub url: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompartmentIdParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Compartment ID")]
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunAgentParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Comma-separated goals for the agent to pursue")]
    pub goals: String,
    #[schemars(description = "Maximum OODA cycles (default: 10)")]
    pub max_cycles: Option<u32>,
}

// ── New tool parameter structs ──────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CreateWorkspaceParams {
    #[schemars(description = "Name for the new workspace")]
    pub name: String,
    #[schemars(description = "Optional role to assign (e.g. 'astronomy researcher')")]
    pub role: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DeleteWorkspaceParams {
    #[schemars(description = "Name of the workspace to delete")]
    pub name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ApplySeedParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Seed pack ID to apply (e.g. 'foundation', 'reasoning')")]
    pub pack_name: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AwakenParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(
        description = "Purpose statement (e.g. 'I want to learn about stellar evolution'). Required for fresh bootstrap."
    )]
    pub statement: Option<String>,
    #[schemars(description = "Resume a previously started bootstrap session")]
    #[serde(default)]
    pub resume: bool,
    #[schemars(description = "Only return the current bootstrap status")]
    #[serde(default)]
    pub status: bool,
    #[schemars(description = "Parse and plan without executing learning cycles")]
    #[serde(default)]
    pub plan_only: bool,
    #[schemars(description = "Maximum learning cycles (default: 3)")]
    pub max_cycles: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AwakenParseParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Purpose statement to parse (preview only — nothing is committed)")]
    pub statement: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ChatParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(
        description = "User message to process through the full NLU pipeline (dialogue acts, facts, queries, goals)"
    )]
    pub message: String,
}

// ── Batch Operations ────────────────────────────────────────────────────

/// A single triple in a batch assertion.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TripleEntry {
    #[schemars(description = "Subject entity")]
    pub s: String,
    #[schemars(description = "Predicate relation")]
    pub p: String,
    #[schemars(description = "Object entity")]
    pub o: String,
    #[schemars(description = "Confidence 0.0-1.0 (default: 0.9)")]
    pub c: Option<f64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AssertBatchParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(
        description = "Array of triples to assert: [{s, p, o, c?}, ...]. Example: [{\"s\":\"Sun\",\"p\":\"is-a\",\"o\":\"Star\"}]"
    )]
    pub triples: Vec<TripleEntry>,
}

// ── Phase 1: Bootstrap Stage Tools ──────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ResolveIdentityParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Name of the figure to resolve (e.g. 'Ptah', 'Gandalf', 'Turing')")]
    pub name: String,
    #[schemars(
        description = "Optional entity type hint: 'deity', 'fictional_character', 'historical_figure'"
    )]
    pub entity_type: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RitualOfAwakeningParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Character name (must match a previously resolved identity)")]
    pub name: String,
    #[schemars(description = "Culture of origin: 'egyptian', 'greek', 'norse', 'latin', 'fictional'")]
    pub culture: String,
    #[schemars(description = "Character personality traits (e.g. ['creative', 'precise', 'wise'])")]
    pub traits: Vec<String>,
    #[schemars(description = "Jungian archetypes (e.g. ['creator', 'ruler'])")]
    pub archetypes: Vec<String>,
    #[schemars(description = "Primary domain (e.g. 'architecture')")]
    pub domain: String,
    #[schemars(
        description = "Target competence level: 'novice', 'advanced_beginner', 'competent', 'proficient', 'expert'"
    )]
    pub competence_level: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ExpandDomainParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Seed concepts to expand from (e.g. ['architecture', 'creation', 'craftsmanship'])")]
    pub seed_concepts: Vec<String>,
    #[schemars(description = "Domain name for the expansion")]
    pub domain: String,
    #[schemars(description = "VSA similarity threshold for candidate acceptance (default: 0.6)")]
    pub similarity_threshold: Option<f32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AnalyzePrerequisitesParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Domain to analyze prerequisites for")]
    pub domain: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AssessCompetenceParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Domain to assess")]
    pub domain: String,
    #[schemars(
        description = "Target Dreyfus level: 'novice', 'advanced_beginner', 'competent', 'proficient', 'expert' (default: 'competent')"
    )]
    pub target_level: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RemoveTripleParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Subject entity label")]
    pub subject: String,
    #[schemars(description = "Predicate relation label")]
    pub predicate: String,
    #[schemars(description = "Object entity label")]
    pub object: String,
}

// ── Phase 2: Agent Introspection Tools ──────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AgentRunCycleParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AgentRecallParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Query terms to search episodic memory for")]
    pub query: Vec<String>,
    #[schemars(description = "Maximum number of results (default: 5)")]
    pub top_k: Option<usize>,
}

// ── Phase 3: Knowledge Introspection Tools ──────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct TriplesOfParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Entity label to get triples for")]
    pub entity: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProvenanceOfParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Entity label to trace provenance for")]
    pub entity: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct InferAnalogyParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "First term (A in 'A is to B as C is to ?')")]
    pub a: String,
    #[schemars(description = "Second term (B)")]
    pub b: String,
    #[schemars(description = "Third term (C)")]
    pub c: String,
    #[schemars(description = "Number of results to return (default: 5)")]
    pub top_k: Option<usize>,
}

// ── Event Calculus Parameter Structs (Phase 15b) ────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecordEventParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Name of the event (e.g. 'switch-on', 'deploy-v2')")]
    pub name: String,
    #[schemars(description = "Timestamp (seconds since UNIX epoch). Defaults to now if omitted.")]
    pub timestamp: Option<u64>,
    #[schemars(
        description = "Fluent labels that this event initiates (starts being true)"
    )]
    pub initiates: Option<Vec<String>>,
    #[schemars(
        description = "Fluent labels that this event terminates (stops being true)"
    )]
    pub terminates: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct HoldsAtParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Fluent label to check (e.g. 'light-on', 'server-running')")]
    pub fluent: String,
    #[schemars(description = "Time to check (seconds since UNIX epoch)")]
    pub time: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ProjectStateParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Start of interval (seconds since UNIX epoch)")]
    pub from_time: u64,
    #[schemars(description = "End of interval (seconds since UNIX epoch)")]
    pub to_time: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SimulateActionsParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(
        description = "Ordered list of action names to simulate (must have causal schemas registered)"
    )]
    pub actions: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct WhatChangedSinceParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Return changes after this timestamp (seconds since UNIX epoch)")]
    pub since: u64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct FluentHistoryParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "Fluent label to get history for")]
    pub fluent: String,
}

// ── Counterfactual Parameter Structs (Phase 15c) ────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CounterfactualParams {
    #[schemars(description = "Target workspace (default: \"default\")")]
    #[serde(default = "default_workspace")]
    pub workspace: String,
    #[schemars(description = "The action that was actually taken (tool name)")]
    pub actual_action: String,
    #[schemars(description = "The hypothetical alternative action (tool name)")]
    pub hypothetical_action: String,
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Format a `Vec<AkhMessage>` into human-readable text for MCP tool output.
fn format_messages(msgs: &[AkhMessage]) -> String {
    let mut lines = Vec::new();
    for msg in msgs {
        match msg {
            AkhMessage::Fact {
                text,
                confidence,
                provenance,
            } => {
                let mut line = text.clone();
                if let Some(c) = confidence {
                    line.push_str(&format!(" (confidence: {c:.2})"));
                }
                if let Some(p) = provenance {
                    line.push_str(&format!(" [source: {p}]"));
                }
                lines.push(line);
            }
            AkhMessage::Reasoning { step, expression } => {
                let mut line = format!("Reasoning: {step}");
                if let Some(expr) = expression {
                    line.push_str(&format!(" → {expr}"));
                }
                lines.push(line);
            }
            AkhMessage::Gap {
                entity,
                description,
            } => {
                lines.push(format!("Knowledge gap: {entity} — {description}"));
            }
            AkhMessage::ToolResult {
                tool,
                success,
                output,
            } => {
                let status = if *success { "ok" } else { "failed" };
                lines.push(format!("[{tool} {status}] {output}"));
            }
            AkhMessage::Narrative { text, .. } => {
                lines.push(text.clone());
            }
            AkhMessage::System { text } => {
                lines.push(text.clone());
            }
            AkhMessage::Error {
                code,
                message,
                help,
            } => {
                let mut line = format!("Error [{code}]: {message}");
                if let Some(h) = help {
                    line.push_str(&format!("\nHelp: {h}"));
                }
                lines.push(line);
            }
            AkhMessage::GoalProgress {
                goal,
                status,
                detail,
            } => {
                let mut line = format!("Goal \"{goal}\": {status}");
                if let Some(d) = detail {
                    line.push_str(&format!(" — {d}"));
                }
                lines.push(line);
            }
            AkhMessage::Prompt { question } => {
                lines.push(format!("? {question}"));
            }
            AkhMessage::AuditLog { .. } => {}
        }
    }
    lines.join("\n")
}

// ── MCP Server ──────────────────────────────────────────────────────────

/// MCP server exposing akh-medu tools to AI assistants.
///
/// Supports all workspaces via the shared engine/agent maps.
/// Each tool accepts an optional `workspace` parameter (default: `"default"`).
#[derive(Clone)]
pub struct AkhMcpServer {
    state: Arc<McpState>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl AkhMcpServer {
    pub fn new(state: Arc<McpState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    // ── Knowledge Query ─────────────────────────────────────────────

    #[tool(
        name = "ask",
        description = "Ask the knowledge agent a natural-language question. The agent runs OODA cycles to find the answer and returns a narrative summary of findings."
    )]
    async fn ask(
        &self,
        Parameters(params): Parameters<AskParams>,
    ) -> Result<CallToolResult, McpError> {
        let agent = self.state.get_agent(&params.workspace).await?;
        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let mut agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let cycles = params.max_cycles.unwrap_or(10) as usize;
            agent.set_max_cycles(cycles);
            agent.clear_goals();
            agent
                .add_goal(&params.question, 128, "Agent-determined completion")
                .map_err(|e| format!("{e}"))?;
            let _ = agent.run_until_complete();
            let summary = agent.synthesize_findings(&params.question);
            let _ = agent.persist_session();
            Ok(summary.overview)
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "sparql_query",
        description = "Run a SPARQL query against the knowledge graph. Returns matching bindings as JSON rows."
    )]
    async fn sparql_query(
        &self,
        Parameters(params): Parameters<SparqlParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let rows = engine
            .sparql_query(&params.query)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;
        let json = serde_json::to_string_pretty(&rows)
            .map_err(|e| McpError::internal_error(format!("json: {e}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    #[tool(
        name = "search",
        description = "Search for concepts matching a term. Returns symbol IDs, labels, and kinds."
    )]
    async fn search(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let term = &params.term;

        let results = if let Ok(sym_id) = engine.resolve_symbol(term) {
            let label = engine.resolve_label(sym_id);
            let mut lines = vec![format!("Exact match: {} (id={})", label, sym_id.get())];
            if let Ok(similar) = engine.search_similar_to(sym_id, 5) {
                for sr in similar {
                    let sl = engine.resolve_label(sr.symbol_id);
                    lines.push(format!(
                        "  similar: {} (id={}, similarity={:.3})",
                        sl,
                        sr.symbol_id.get(),
                        sr.similarity
                    ));
                }
            }
            lines.join("\n")
        } else {
            let term_lower = term.to_lowercase();
            let matches: Vec<String> = engine
                .all_symbols()
                .into_iter()
                .filter(|m| m.label.to_lowercase().contains(&term_lower))
                .take(20)
                .map(|m| format!("{} (id={}, kind={:?})", m.label, m.id.get(), m.kind))
                .collect();
            if matches.is_empty() {
                format!("No symbols matching \"{term}\"")
            } else {
                format!("Found {} matches:\n{}", matches.len(), matches.join("\n"))
            }
        };
        Ok(CallToolResult::success(vec![Content::text(results)]))
    }

    // ── Knowledge Mutation ──────────────────────────────────────────

    #[tool(
        name = "assert_triple",
        description = "Assert a new fact as a subject-predicate-object triple. Creates entities/relations if they don't exist."
    )]
    async fn assert_triple(
        &self,
        Parameters(params): Parameters<AssertTripleParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let conf = params.confidence.unwrap_or(0.9) as f32;

        let s = engine
            .resolve_or_create_entity(&params.subject)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;
        let p = engine
            .resolve_or_create_relation(&params.predicate)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;
        let o = engine
            .resolve_or_create_entity(&params.object)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;

        let triple = crate::graph::Triple::new(s, p, o).with_confidence(conf);
        engine
            .add_triple(&triple)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;

        let _ = engine.persist();
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Asserted: {} {} {} (confidence={:.2})",
            params.subject, params.predicate, params.object, conf
        ))]))
    }

    #[tool(
        name = "assert_batch",
        description = "Assert multiple triples at once. Far more efficient than individual assert_triple calls. Accepts an array of {s, p, o, c?} objects. Use this when Claude generates structured triples from text — it's the primary knowledge population path."
    )]
    async fn assert_batch(
        &self,
        Parameters(params): Parameters<AssertBatchParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let triples = params.triples;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let mut asserted = 0usize;
            let mut errors = Vec::new();

            for entry in &triples {
                let conf = entry.c.unwrap_or(0.9) as f32;
                let res = (|| -> Result<(), String> {
                    let s = engine
                        .resolve_or_create_entity(&entry.s)
                        .map_err(|e| format!("{e}"))?;
                    let p = engine
                        .resolve_or_create_relation(&entry.p)
                        .map_err(|e| format!("{e}"))?;
                    let o = engine
                        .resolve_or_create_entity(&entry.o)
                        .map_err(|e| format!("{e}"))?;
                    let triple = crate::graph::Triple::new(s, p, o).with_confidence(conf);
                    engine.add_triple(&triple).map_err(|e| format!("{e}"))?;
                    Ok(())
                })();

                match res {
                    Ok(()) => asserted += 1,
                    Err(e) => errors.push(format!("{} {} {}: {}", entry.s, entry.p, entry.o, e)),
                }
            }

            let _ = engine.persist();

            let mut msg = format!("Asserted {asserted}/{} triples", triples.len());
            if !errors.is_empty() {
                msg.push_str(&format!("\nErrors ({}):", errors.len()));
                for e in errors.iter().take(5) {
                    msg.push_str(&format!("\n  - {e}"));
                }
            }
            Ok(msg)
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "ingest_text",
        description = "Ingest natural-language text into the knowledge graph. Extracts entities and relations from sentences."
    )]
    async fn ingest_text(
        &self,
        Parameters(params): Parameters<IngestTextParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            use crate::agent::tool::{Tool, ToolInput};
            use crate::agent::tools::TextIngestTool;
            let input = ToolInput::new()
                .with_param("text", &params.text)
                .with_param("max_sentences", "100");
            let tool = TextIngestTool;
            let output = tool.execute(&engine, input).map_err(|e| format!("{e}"))?;
            let _ = engine.persist();
            Ok(output.result)
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "ingest_url",
        description = "Ingest content from a URL into the knowledge graph. Fetches the page and extracts entities and relations."
    )]
    async fn ingest_url(
        &self,
        Parameters(params): Parameters<IngestUrlParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            use crate::agent::tool::{Tool, ToolInput};
            use crate::agent::tools::ContentIngestTool;
            let input = ToolInput::new().with_param("source", &params.url);
            let tool = ContentIngestTool;
            let output = tool.execute(&engine, input).map_err(|e| format!("{e}"))?;
            let _ = engine.persist();
            Ok(output.result)
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    // ── Compartment Management ──────────────────────────────────────

    #[tool(
        name = "list_compartments",
        description = "List all knowledge compartments with their state (Dormant/Loaded/Active), kind, and triple count."
    )]
    async fn list_compartments(
        &self,
        Parameters(params): Parameters<WorkspaceParam>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let mgr = engine.compartments().ok_or_else(|| {
            McpError::internal_error("compartment manager not available".to_string(), None)
        })?;
        let all = mgr.all_compartments();
        if all.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No compartments discovered. Run discover_compartments first or check compartments directory.",
            )]));
        }
        let lines: Vec<String> = all
            .into_iter()
            .map(|(m, st, tc)| {
                format!(
                    "- {} (id={}, kind={:?}, state={}, triples={}): {}",
                    m.name, m.id, m.kind, st, tc, m.description
                )
            })
            .collect();
        Ok(CallToolResult::success(vec![Content::text(
            lines.join("\n"),
        )]))
    }

    #[tool(
        name = "discover_compartments",
        description = "Scan the compartments directory for new compartment manifests. Returns how many were newly discovered."
    )]
    async fn discover_compartments(
        &self,
        Parameters(params): Parameters<WorkspaceParam>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let mgr = engine.compartments().ok_or_else(|| {
            McpError::internal_error("compartment manager not available".to_string(), None)
        })?;
        let count = mgr
            .discover()
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;
        let total = mgr.all_compartments().len();
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Discovered {count} new compartments ({total} total)"
        ))]))
    }

    #[tool(
        name = "load_compartment",
        description = "Load a compartment's triples into the knowledge graph. Knowledge becomes queryable via SPARQL and search."
    )]
    async fn load_compartment(
        &self,
        Parameters(params): Parameters<CompartmentIdParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let mgr = engine.compartments().ok_or_else(|| {
            McpError::internal_error("compartment manager not available".to_string(), None)
        })?;
        mgr.load(&params.id, &engine)
            .map_err(|e| McpError::invalid_params(format!("{e}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Compartment \"{}\" loaded into knowledge graph",
            params.id
        ))]))
    }

    #[tool(
        name = "unload_compartment",
        description = "Unload a compartment, removing its triples from the knowledge graph."
    )]
    async fn unload_compartment(
        &self,
        Parameters(params): Parameters<CompartmentIdParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let mgr = engine.compartments().ok_or_else(|| {
            McpError::internal_error("compartment manager not available".to_string(), None)
        })?;
        mgr.unload(&params.id, &engine)
            .map_err(|e| McpError::invalid_params(format!("{e}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Compartment \"{}\" unloaded",
            params.id
        ))]))
    }

    #[tool(
        name = "activate_compartment",
        description = "Activate a loaded compartment so it influences the agent's reasoning (OODA loop). Must be loaded first."
    )]
    async fn activate_compartment(
        &self,
        Parameters(params): Parameters<CompartmentIdParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let mgr = engine.compartments().ok_or_else(|| {
            McpError::internal_error("compartment manager not available".to_string(), None)
        })?;
        mgr.activate(&params.id)
            .map_err(|e| McpError::invalid_params(format!("{e}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Compartment \"{}\" activated — now influencing agent reasoning",
            params.id
        ))]))
    }

    #[tool(
        name = "deactivate_compartment",
        description = "Deactivate a compartment. Triples stay loaded but stop influencing the agent's reasoning."
    )]
    async fn deactivate_compartment(
        &self,
        Parameters(params): Parameters<CompartmentIdParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let mgr = engine.compartments().ok_or_else(|| {
            McpError::internal_error("compartment manager not available".to_string(), None)
        })?;
        mgr.deactivate(&params.id)
            .map_err(|e| McpError::invalid_params(format!("{e}"), None))?;
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Compartment \"{}\" deactivated — triples still loaded but inactive",
            params.id
        ))]))
    }

    // ── Agent ───────────────────────────────────────────────────────

    #[tool(
        name = "run_agent",
        description = "Run the agent with specific goals. Returns findings after OODA cycles complete."
    )]
    async fn run_agent(
        &self,
        Parameters(params): Parameters<RunAgentParams>,
    ) -> Result<CallToolResult, McpError> {
        let agent = self.state.get_agent(&params.workspace).await?;
        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let mut agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let cycles = params.max_cycles.unwrap_or(10) as usize;
            agent.set_max_cycles(cycles);
            agent.clear_goals();
            let goal_list: Vec<&str> = params.goals.split(',').map(|s| s.trim()).collect();
            for g in &goal_list {
                agent
                    .add_goal(g, 128, "Agent-determined completion")
                    .map_err(|e| format!("{e}"))?;
            }
            let _ = agent.run_until_complete();
            let summary = agent.synthesize_findings(&params.goals);
            let _ = agent.persist_session();

            let mut output = summary.overview;
            if !summary.gaps.is_empty() {
                output.push_str("\n\nKnowledge gaps:\n");
                for gap in &summary.gaps {
                    output.push_str(&format!("- {gap}\n"));
                }
            }
            Ok(output)
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    // ── Workspace Status ────────────────────────────────────────────

    #[tool(
        name = "status",
        description = "Get workspace status: symbol count, triple count, loaded compartments, and engine info."
    )]
    async fn status(
        &self,
        Parameters(params): Parameters<WorkspaceParam>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let info = engine.info();
        let mut lines = vec![
            format!("Workspace: {}", params.workspace),
            format!("Symbols: {}", info.symbol_count),
            format!("Triples: {}", info.triple_count),
            format!("Provenance records: {}", info.provenance_count),
            format!("VSA dimension: {}", info.dimension),
        ];

        if let Some(mgr) = engine.compartments() {
            let active: Vec<String> = mgr
                .active_compartments()
                .into_iter()
                .map(|m| m.id)
                .collect();
            let all = mgr.all_compartments();
            let loaded = all
                .iter()
                .filter(|(_, st, _)| st.to_string() != "Dormant")
                .count();
            lines.push(format!(
                "Compartments: {} total, {} loaded, {} active",
                all.len(),
                loaded,
                active.len()
            ));
            if !active.is_empty() {
                lines.push(format!("Active: {}", active.join(", ")));
            }
        }

        Ok(CallToolResult::success(vec![Content::text(
            lines.join("\n"),
        )]))
    }

    // ── Workspace Management ────────────────────────────────────────

    #[tool(
        name = "list_workspaces",
        description = "List all available workspace names."
    )]
    async fn list_workspaces(&self) -> Result<CallToolResult, McpError> {
        let names = self.state.paths.list_workspaces();
        if names.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No workspaces found. Use create_workspace to create one.",
            )]));
        }
        let lines: Vec<String> = names.iter().map(|n| format!("- {n}")).collect();
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Workspaces ({}):\n{}",
            names.len(),
            lines.join("\n")
        ))]))
    }

    #[tool(
        name = "create_workspace",
        description = "Create a new workspace. Optionally assign a role and apply the foundation seed pack."
    )]
    async fn create_workspace(
        &self,
        Parameters(params): Parameters<CreateWorkspaceParams>,
    ) -> Result<CallToolResult, McpError> {
        let paths = self.state.paths.clone();
        let name = params.name.clone();

        let manager = crate::workspace::WorkspaceManager::new(paths);
        let config = crate::workspace::WorkspaceConfig {
            name: name.clone(),
            ..Default::default()
        };
        manager
            .create(config)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;

        let mut extra = String::new();

        // Assign role if provided.
        if let Some(ref role) = params.role {
            let engine = self.state.get_engine(&name).await?;
            engine
                .assign_role(role)
                .map_err(|e| McpError::internal_error(format!("{e}"), None))?;
            engine
                .persist()
                .map_err(|e| McpError::internal_error(format!("{e}"), None))?;
            extra.push_str(&format!(", role=\"{role}\""));
        }

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Workspace \"{name}\" created{extra}"
        ))]))
    }

    #[tool(
        name = "delete_workspace",
        description = "Delete a workspace and all its data. This action is irreversible."
    )]
    async fn delete_workspace(
        &self,
        Parameters(params): Parameters<DeleteWorkspaceParams>,
    ) -> Result<CallToolResult, McpError> {
        let name = &params.name;

        // Remove from loaded maps.
        {
            let mut map = self.state.workspaces.write().await;
            map.remove(name);
        }
        {
            let mut agents = self.state.agents.write().await;
            agents.remove(name);
        }

        let manager = crate::workspace::WorkspaceManager::new(self.state.paths.clone());
        manager
            .delete(name)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Workspace \"{name}\" deleted"
        ))]))
    }

    #[tool(
        name = "apply_seed",
        description = "Apply a seed pack to a workspace, populating it with foundational knowledge triples."
    )]
    async fn apply_seed(
        &self,
        Parameters(params): Parameters<ApplySeedParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let registry = crate::seeds::SeedRegistry::bundled();
        let report = registry
            .apply(&params.pack_name, &engine)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;

        let status = if report.already_applied {
            "already applied"
        } else {
            "applied"
        };
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Seed pack \"{}\": {} ({} triples, {} skipped)",
            params.pack_name, status, report.triples_applied, report.triples_skipped
        ))]))
    }

    #[tool(
        name = "list_seeds",
        description = "List all available seed packs with their descriptions."
    )]
    async fn list_seeds(&self) -> Result<CallToolResult, McpError> {
        let registry = crate::seeds::SeedRegistry::bundled();
        let packs = registry.list();
        if packs.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No seed packs available.",
            )]));
        }
        let lines: Vec<String> = packs
            .iter()
            .map(|p| {
                format!(
                    "- {} (v{}): {} [{} triples]",
                    p.id,
                    p.version,
                    p.description,
                    p.triples.len()
                )
            })
            .collect();
        Ok(CallToolResult::success(vec![Content::text(format!(
            "Seed packs ({}):\n{}",
            packs.len(),
            lines.join("\n")
        ))]))
    }

    // ── Bootstrap / Awaken ──────────────────────────────────────────

    #[tool(
        name = "awaken",
        description = "Run the full bootstrap orchestrator: parse purpose, resolve identity, expand domain, ingest resources, and assess competence. Use status=true for progress check, resume=true to continue a previous session."
    )]
    async fn awaken(
        &self,
        Parameters(params): Parameters<AwakenParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            use crate::bootstrap::OrchestratorConfig;

            // Status-only request.
            if params.status {
                let session = crate::bootstrap::BootstrapOrchestrator::status(&engine)
                    .map_err(|e| format!("{e}"))?;
                return Ok(format!(
                    "Bootstrap status:\n\
                     Stage: {:?}\n\
                     Learning cycle: {}\n\
                     Purpose: {}\n\
                     Name: {}\n\
                     Last assessment: {}",
                    session.current_stage,
                    session.learning_cycle,
                    session.raw_purpose,
                    session.chosen_name.as_deref().unwrap_or("(none)"),
                    session
                        .last_assessment
                        .as_ref()
                        .map(|a| format!("{} (score: {:.1})", a.overall_dreyfus, a.overall_score))
                        .unwrap_or_else(|| "(none)".to_string()),
                ));
            }

            let config = OrchestratorConfig {
                max_learning_cycles: params.max_cycles.unwrap_or(3),
                plan_only: params.plan_only,
                ..Default::default()
            };

            let mut orchestrator = if params.resume {
                crate::bootstrap::BootstrapOrchestrator::resume(&engine, config)
                    .map_err(|e| format!("{e}"))?
            } else {
                let stmt = params
                    .statement
                    .as_deref()
                    .ok_or("statement required for fresh bootstrap")?;
                crate::bootstrap::BootstrapOrchestrator::new(stmt, config)
                    .map_err(|e| format!("{e}"))?
            };

            let (result, checkpoints) =
                orchestrator.run(&engine).map_err(|e| format!("{e}"))?;

            let mut lines = vec![
                format!("Domain: {}", result.intent.purpose.domain),
                format!("Target level: {}", result.intent.purpose.competence_level),
            ];
            if let Some(ref name) = result.chosen_name {
                lines.push(format!("Identity: {name}"));
            }
            lines.push(format!("Learning cycles: {}", result.learning_cycles));
            lines.push(format!("Target reached: {}", result.target_reached));
            lines.push(format!(
                "Stages completed: {}",
                result
                    .stages_completed
                    .iter()
                    .map(|s| format!("{s:?}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));

            if let Some(ref report) = result.final_report {
                lines.push(format!(
                    "Final assessment: {} (score: {:.1})",
                    report.overall_dreyfus, report.overall_score
                ));
                lines.push(format!("Recommendation: {}", report.recommendation));
            }

            if !checkpoints.is_empty() {
                lines.push(format!("\nCheckpoints ({})", checkpoints.len()));
                for cp in &checkpoints {
                    lines.push(format!("  - {cp:?}"));
                }
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "awaken_parse",
        description = "Parse a purpose statement without committing anything. Returns domain, competence level, seed concepts, and identity reference if detected. Useful for previewing what 'awaken' will do."
    )]
    async fn awaken_parse(
        &self,
        Parameters(params): Parameters<AwakenParseParams>,
    ) -> Result<CallToolResult, McpError> {
        let statement = params.statement;
        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let intent = crate::bootstrap::purpose::parse_purpose(&statement)
                .map_err(|e| format!("{e}"))?;

            let mut lines = vec![
                format!("Domain: {}", intent.purpose.domain),
                format!("Competence level: {}", intent.purpose.competence_level),
                format!("Description: {}", intent.purpose.description),
            ];

            if !intent.purpose.seed_concepts.is_empty() {
                lines.push(format!(
                    "Seed concepts: {}",
                    intent.purpose.seed_concepts.join(", ")
                ));
            }

            if let Some(ref identity) = intent.identity {
                lines.push(format!(
                    "Identity reference: {} ({}, from: \"{}\")",
                    identity.name, identity.entity_type, identity.source_phrase
                ));
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    // ── Phase 1: Bootstrap Stage Tools ────────────────────────────

    #[tool(
        name = "resolve_identity",
        description = "Resolve a cultural/historical/fictional figure to structured CharacterKnowledge (traits, archetypes, culture, domains). Use this to preview identity data before ritual_of_awakening, or to fill gaps when the monolithic awaken tool fails identity resolution."
    )]
    async fn resolve_identity(
        &self,
        Parameters(params): Parameters<ResolveIdentityParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let name = params.name;
        let entity_type_hint = params.entity_type;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            use crate::bootstrap::identity::resolve_identity;
            use crate::bootstrap::purpose::{EntityType, IdentityRef};

            let entity_type = entity_type_hint
                .as_deref()
                .and_then(|t| match t.to_lowercase().as_str() {
                    "deity" => Some(EntityType::Deity),
                    "fictional_character" | "fictional" => Some(EntityType::FictionalCharacter),
                    "historical_figure" | "historical" => Some(EntityType::HistoricalFigure),
                    _ => None,
                })
                .unwrap_or(EntityType::Deity);

            let identity_ref = IdentityRef {
                name: name.clone(),
                entity_type,
                source_phrase: name.clone(),
            };

            let knowledge = resolve_identity(&identity_ref, &engine)
                .map_err(|e| format!("{e}"))?;

            let json = serde_json::to_string_pretty(&knowledge)
                .map_err(|e| format!("json: {e}"))?;
            Ok(json)
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "ritual_of_awakening",
        description = "Perform the Ritual of Awakening: construct a Psyche with Jungian archetypes, OCEAN personality, shadow patterns, and a culture-specific self-name. Requires CharacterKnowledge data (from resolve_identity or manually provided)."
    )]
    async fn ritual_of_awakening(
        &self,
        Parameters(params): Parameters<RitualOfAwakeningParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            use crate::bootstrap::identity::{
                ritual_of_awakening, CharacterKnowledge, CultureOrigin,
            };
            use crate::bootstrap::purpose::{DreyfusLevel, EntityType, PurposeModel};

            let culture = CultureOrigin::from_label(&params.culture)
                .unwrap_or(CultureOrigin::Unknown);

            let entity_type = match culture {
                CultureOrigin::Egyptian | CultureOrigin::Greek | CultureOrigin::Norse => {
                    EntityType::Deity
                }
                CultureOrigin::Fictional => EntityType::FictionalCharacter,
                _ => EntityType::HistoricalFigure,
            };

            let character = CharacterKnowledge {
                name: params.name.clone(),
                entity_type,
                culture,
                description: format!("Identity for {}", params.name),
                domains: vec![params.domain.clone()],
                traits: params.traits,
                archetypes: params.archetypes,
            };

            let competence_level = params
                .competence_level
                .as_deref()
                .and_then(|l| match l.to_lowercase().as_str() {
                    "novice" => Some(DreyfusLevel::Novice),
                    "advanced_beginner" => Some(DreyfusLevel::AdvancedBeginner),
                    "competent" => Some(DreyfusLevel::Competent),
                    "proficient" => Some(DreyfusLevel::Proficient),
                    "expert" => Some(DreyfusLevel::Expert),
                    _ => None,
                })
                .unwrap_or(DreyfusLevel::Competent);

            let purpose = PurposeModel {
                domain: params.domain,
                description: format!("Awakening as {}", params.name),
                seed_concepts: character.domains.clone(),
                competence_level,
            };

            let ritual_result =
                ritual_of_awakening(&character, &purpose, &engine).map_err(|e| format!("{e}"))?;

            let _ = engine.persist();

            let shadow_names: Vec<&str> = ritual_result
                .psyche
                .shadow
                .veto_patterns
                .iter()
                .chain(ritual_result.psyche.shadow.bias_patterns.iter())
                .map(|p| p.name.as_str())
                .collect();

            let lines = vec![
                format!("Chosen name: {}", ritual_result.chosen_name),
                format!("Persona: {}", ritual_result.psyche.persona.name),
                format!("Traits: {}", ritual_result.psyche.persona.traits.join(", ")),
                format!(
                    "Archetypes: healer={:.1} sage={:.1} guardian={:.1} explorer={:.1}",
                    ritual_result.psyche.archetypes.healer,
                    ritual_result.psyche.archetypes.sage,
                    ritual_result.psyche.archetypes.guardian,
                    ritual_result.psyche.archetypes.explorer,
                ),
                format!("Shadow: {}", shadow_names.join(", ")),
                format!(
                    "Self-integration: {:.2}",
                    ritual_result.psyche.self_integration.individuation_level
                ),
                format!("Provenance: {} records", ritual_result.provenance_ids.len()),
            ];

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "expand_domain",
        description = "Expand seed concepts into a skeleton ontology by querying external knowledge sources (Wikidata, Wikipedia, ConceptNet) and filtering by VSA similarity. Persists concepts and relations to the KG."
    )]
    async fn expand_domain(
        &self,
        Parameters(params): Parameters<ExpandDomainParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            use crate::bootstrap::expand::{DomainExpander, ExpansionConfig};
            use crate::bootstrap::purpose::{DreyfusLevel, PurposeModel};

            let purpose = PurposeModel {
                domain: params.domain.clone(),
                description: format!("Domain expansion for {}", params.domain),
                seed_concepts: params.seed_concepts,
                competence_level: DreyfusLevel::Competent,
            };

            let config = ExpansionConfig {
                similarity_threshold: params.similarity_threshold.unwrap_or(0.6),
                ..Default::default()
            };

            let mut expander =
                DomainExpander::new(&engine, config).map_err(|e| format!("{e}"))?;
            let result = expander
                .expand(&purpose, &engine)
                .map_err(|e| format!("{e}"))?;

            let _ = engine.persist();

            let mut lines = vec![
                format!("Concepts added: {}", result.concept_count),
                format!("Relations added: {}", result.relation_count),
                format!("Candidates rejected: {}", result.rejected_count),
                format!("API calls: {}", result.api_calls),
            ];

            if !result.accepted_labels.is_empty() {
                lines.push(format!(
                    "Accepted: {}",
                    result.accepted_labels.join(", ")
                ));
            }
            if !result.boundary_rejects.is_empty() {
                lines.push(format!(
                    "Boundary rejects: {}",
                    result.boundary_rejects.join(", ")
                ));
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "analyze_prerequisites",
        description = "Discover prerequisite relationships between domain concepts, classify by Vygotsky ZPD zones (Known/Proximal/Beyond), and generate a curriculum ordering. Requires prior domain expansion."
    )]
    async fn analyze_prerequisites(
        &self,
        Parameters(params): Parameters<AnalyzePrerequisitesParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            use crate::bootstrap::expand::ExpansionResult;
            use crate::bootstrap::prerequisite::{PrerequisiteAnalyzer, PrerequisiteConfig};

            // Find domain concepts: entities that participate in expand:* triples
            let expand_preds: Vec<(&str, Option<crate::symbol::SymbolId>)> = vec![
                ("expand:expanded_from", engine.resolve_symbol("expand:expanded_from").ok()),
                ("expand:subclass_of", engine.resolve_symbol("expand:subclass_of").ok()),
                ("expand:instance_of", engine.resolve_symbol("expand:instance_of").ok()),
                ("expand:part_of", engine.resolve_symbol("expand:part_of").ok()),
                ("expand:has_prerequisite", engine.resolve_symbol("expand:has_prerequisite").ok()),
                ("expand:related_to", engine.resolve_symbol("expand:related_to").ok()),
            ];

            let mut domain_ids = std::collections::HashSet::new();
            for (_name, pred_opt) in &expand_preds {
                if let Some(pred) = pred_opt {
                    let pairs = engine.knowledge_graph().triples_for_predicate(*pred);
                    for (subj, obj) in &pairs {
                        domain_ids.insert(*subj);
                        domain_ids.insert(*obj);
                    }
                }
            }

            // Filter out prototype/microtheory meta-symbols
            let accepted_labels: Vec<String> = domain_ids
                .iter()
                .map(|id| engine.resolve_label(*id))
                .filter(|label| !label.starts_with("expand:"))
                .collect();

            if accepted_labels.is_empty() {
                return Ok("No domain expansion concepts found. Run expand_domain first.".to_string());
            }

            let expansion_result = ExpansionResult {
                concept_count: accepted_labels.len(),
                relation_count: 0,
                rejected_count: 0,
                api_calls: 0,
                domain_prototype_id: None,
                microtheory_id: None,
                provenance_ids: vec![],
                accepted_labels,
                boundary_rejects: vec![],
            };

            // Lower min_edge_confidence for standalone use — the orchestrator
            // combines ConceptNet + structural + VSA edges so the default 0.3
            // works there, but standalone structural edges are 0.6 * 0.3 = 0.18.
            let config = PrerequisiteConfig {
                min_edge_confidence: 0.15,
                ..Default::default()
            };
            let analyzer =
                PrerequisiteAnalyzer::new(&engine, config).map_err(|e| format!("{e}"))?;
            let result = analyzer
                .analyze(&expansion_result, &engine)
                .map_err(|e| format!("{e}"))?;

            let _ = engine.persist();

            let mut lines = vec![
                format!("Concepts analyzed: {}", result.concepts_analyzed),
                format!("Prerequisite edges: {}", result.edge_count),
                format!("Cycles broken: {}", result.cycles_broken),
                format!("Curriculum entries: {}", result.curriculum.len()),
            ];

            if !result.curriculum.is_empty() {
                lines.push("Curriculum (top 20):".to_string());
                for entry in result.curriculum.iter().take(20) {
                    lines.push(format!(
                        "  tier={} zone={:?}: {}",
                        entry.tier, entry.zone, entry.label
                    ));
                }
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "assess_competence",
        description = "Evaluate how well the workspace knows its domain. Runs gap analysis, schema discovery, Bloom's taxonomy evaluation, and produces a Dreyfus-level assessment report."
    )]
    async fn assess_competence(
        &self,
        Parameters(params): Parameters<AssessCompetenceParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            use crate::bootstrap::competence::{CompetenceAssessor, CompetenceConfig};
            use crate::bootstrap::expand::ExpansionResult;
            use crate::bootstrap::prerequisite::{PrerequisiteAnalyzer, PrerequisiteConfig};
            use crate::bootstrap::purpose::{DreyfusLevel, PurposeModel};

            let target_level = params
                .target_level
                .as_deref()
                .and_then(|l| match l.to_lowercase().as_str() {
                    "novice" => Some(DreyfusLevel::Novice),
                    "advanced_beginner" => Some(DreyfusLevel::AdvancedBeginner),
                    "competent" => Some(DreyfusLevel::Competent),
                    "proficient" => Some(DreyfusLevel::Proficient),
                    "expert" => Some(DreyfusLevel::Expert),
                    _ => None,
                })
                .unwrap_or(DreyfusLevel::Competent);

            let purpose = PurposeModel {
                domain: params.domain.clone(),
                description: format!("Assessment for {}", params.domain),
                seed_concepts: vec![],
                competence_level: target_level,
            };

            // Find domain concepts: entities that participate in expand:* triples
            let expand_preds: Vec<Option<crate::symbol::SymbolId>> = vec![
                engine.resolve_symbol("expand:expanded_from").ok(),
                engine.resolve_symbol("expand:subclass_of").ok(),
                engine.resolve_symbol("expand:instance_of").ok(),
                engine.resolve_symbol("expand:part_of").ok(),
                engine.resolve_symbol("expand:has_prerequisite").ok(),
                engine.resolve_symbol("expand:related_to").ok(),
            ];

            let mut domain_ids = std::collections::HashSet::new();
            for pred_opt in &expand_preds {
                if let Some(pred) = pred_opt {
                    let pairs = engine.knowledge_graph().triples_for_predicate(*pred);
                    for (subj, obj) in &pairs {
                        domain_ids.insert(*subj);
                        domain_ids.insert(*obj);
                    }
                }
            }

            let accepted_labels: Vec<String> = domain_ids
                .iter()
                .map(|id| engine.resolve_label(*id))
                .filter(|label| !label.starts_with("expand:"))
                .collect();

            if accepted_labels.is_empty() {
                return Err("No domain expansion concepts found. Run expand_domain first.".to_string());
            }

            let expansion_result = ExpansionResult {
                concept_count: accepted_labels.len(),
                relation_count: 0,
                rejected_count: 0,
                api_calls: 0,
                domain_prototype_id: None,
                microtheory_id: None,
                provenance_ids: vec![],
                accepted_labels,
                boundary_rejects: vec![],
            };

            let prereq_config = PrerequisiteConfig {
                min_edge_confidence: 0.15,
                ..Default::default()
            };
            let prereq_analyzer =
                PrerequisiteAnalyzer::new(&engine, prereq_config).map_err(|e| format!("{e}"))?;
            let prereq_result = prereq_analyzer
                .analyze(&expansion_result, &engine)
                .map_err(|e| format!("{e}"))?;

            let comp_config = CompetenceConfig::default();
            let assessor =
                CompetenceAssessor::new(&engine, comp_config).map_err(|e| format!("{e}"))?;
            let report = assessor
                .assess(&prereq_result, &purpose, &engine)
                .map_err(|e| format!("{e}"))?;

            let _ = engine.persist();

            let mut lines = vec![
                format!("Overall Dreyfus level: {}", report.overall_dreyfus),
                format!("Overall score: {:.2}", report.overall_score),
                format!("Recommendation: {}", report.recommendation),
            ];

            if !report.knowledge_areas.is_empty() {
                lines.push("Knowledge areas:".to_string());
                for ka in &report.knowledge_areas {
                    lines.push(format!(
                        "  {} — {} ({:.2}), triples={}, gaps={}",
                        ka.name, ka.dreyfus_level, ka.score, ka.triple_count, ka.gap_count
                    ));
                }
            }

            if !report.remaining_gaps.is_empty() {
                lines.push("Remaining gaps:".to_string());
                for gap in &report.remaining_gaps {
                    lines.push(format!("  - {gap}"));
                }
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "remove_triple",
        description = "Remove a specific triple from the knowledge graph and SPARQL store. Triggers TMS retraction cascade if enabled."
    )]
    async fn remove_triple(
        &self,
        Parameters(params): Parameters<RemoveTripleParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;

        let s = engine
            .resolve_symbol(&params.subject)
            .map_err(|e| McpError::invalid_params(format!("subject: {e}"), None))?;
        let p = engine
            .resolve_symbol(&params.predicate)
            .map_err(|e| McpError::invalid_params(format!("predicate: {e}"), None))?;
        let o = engine
            .resolve_symbol(&params.object)
            .map_err(|e| McpError::invalid_params(format!("object: {e}"), None))?;

        let retraction = engine
            .remove_triple(s, p, o)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;

        let _ = engine.persist();

        let msg = if retraction.retracted.is_empty() {
            format!(
                "No triple found: {} {} {}",
                params.subject, params.predicate, params.object
            )
        } else {
            format!(
                "Removed: {} {} {} (cascade: {} retracted, {} re-evaluated, depth={})",
                params.subject,
                params.predicate,
                params.object,
                retraction.retracted.len(),
                retraction.re_evaluated.len(),
                retraction.cascade_depth
            )
        };

        Ok(CallToolResult::success(vec![Content::text(msg)]))
    }

    // ── Phase 2: Agent Introspection Tools ─────────────────────────

    #[tool(
        name = "agent_run_cycle",
        description = "Run a single OODA cycle (Observe → Orient → Decide → Act) instead of the full agent loop. Returns detailed cycle results. Agent must have active goals."
    )]
    async fn agent_run_cycle(
        &self,
        Parameters(params): Parameters<AgentRunCycleParams>,
    ) -> Result<CallToolResult, McpError> {
        let agent = self.state.get_agent(&params.workspace).await?;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let mut agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let cycle = agent.run_cycle().map_err(|e| format!("{e}"))?;
            let _ = agent.persist_session();

            let lines = vec![
                format!("Cycle #{}", cycle.cycle_number),
                format!("Observation: {:?}", cycle.observation),
                format!("Orientation: {:?}", cycle.orientation),
                format!("Decision: {:?}", cycle.decision),
                format!("Action result: {:?}", cycle.action_result),
            ];
            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "agent_goals",
        description = "List the agent's current goals with their status, priority, and success criteria."
    )]
    async fn agent_goals(
        &self,
        Parameters(params): Parameters<WorkspaceParam>,
    ) -> Result<CallToolResult, McpError> {
        let agent = self.state.get_agent(&params.workspace).await?;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let goals = agent.goals();

            if goals.is_empty() {
                return Ok("No goals.".to_string());
            }

            let lines: Vec<String> = goals
                .iter()
                .map(|g| {
                    format!(
                        "- [{}] \"{}\" (priority={}, criteria=\"{}\")",
                        format!("{:?}", g.status),
                        g.description,
                        g.priority,
                        g.success_criteria
                    )
                })
                .collect();

            Ok(format!("Goals ({}):\n{}", goals.len(), lines.join("\n")))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "agent_recall",
        description = "Search the agent's episodic memory by query terms. Returns matching episodes with summaries and learnings."
    )]
    async fn agent_recall(
        &self,
        Parameters(params): Parameters<AgentRecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let agent = self.state.get_agent(&params.workspace).await?;
        let query_terms = params.query;
        let top_k = params.top_k.unwrap_or(5);

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;

            // Resolve query terms to symbol IDs
            let query_ids: Vec<crate::symbol::SymbolId> = query_terms
                .iter()
                .filter_map(|term| engine.resolve_symbol(term).ok())
                .collect();

            if query_ids.is_empty() {
                return Ok("No query terms could be resolved to known symbols.".to_string());
            }

            let episodes = agent.recall(&query_ids, top_k).map_err(|e| format!("{e}"))?;

            if episodes.is_empty() {
                return Ok("No matching episodes found.".to_string());
            }

            let lines: Vec<String> = episodes
                .iter()
                .map(|ep| {
                    let learnings: Vec<String> = ep
                        .learnings
                        .iter()
                        .map(|l| engine.resolve_label(*l))
                        .collect();
                    format!(
                        "- {} (learnings: {})",
                        ep.summary,
                        if learnings.is_empty() {
                            "none".to_string()
                        } else {
                            learnings.join(", ")
                        }
                    )
                })
                .collect();

            Ok(format!(
                "Episodes ({}):\n{}",
                episodes.len(),
                lines.join("\n")
            ))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "agent_psyche",
        description = "Read the agent's current psyche state: persona, archetypes, OCEAN profile, shadow patterns, and self-integration."
    )]
    async fn agent_psyche(
        &self,
        Parameters(params): Parameters<WorkspaceParam>,
    ) -> Result<CallToolResult, McpError> {
        let agent = self.state.get_agent(&params.workspace).await?;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;

            match agent.psyche() {
                Some(psyche) => {
                    let mut lines = vec![
                        format!("Persona: {}", psyche.persona.name),
                        format!("Traits: {}", psyche.persona.traits.join(", ")),
                        format!(
                            "Archetypes: healer={:.1} sage={:.1} guardian={:.1} explorer={:.1}",
                            psyche.archetypes.healer,
                            psyche.archetypes.sage,
                            psyche.archetypes.guardian,
                            psyche.archetypes.explorer,
                        ),
                    ];

                    let shadow_names: Vec<&str> = psyche
                        .shadow
                        .veto_patterns
                        .iter()
                        .chain(psyche.shadow.bias_patterns.iter())
                        .map(|p| p.name.as_str())
                        .collect();
                    if !shadow_names.is_empty() {
                        lines.push(format!("Shadow patterns: {}", shadow_names.join(", ")));
                    }

                    lines.push(format!(
                        "Self-integration: {:.2} (dominant: {})",
                        psyche.self_integration.individuation_level,
                        psyche.self_integration.dominant_archetype,
                    ));

                    Ok(lines.join("\n"))
                }
                None => Ok("No psyche awakened. Run ritual_of_awakening first.".to_string()),
            }
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    // ── Phase 3: Knowledge Introspection Tools ─────────────────────

    #[tool(
        name = "triples_of",
        description = "Get all triples involving a symbol (both outgoing and incoming). Returns subject-predicate-object with confidence."
    )]
    async fn triples_of(
        &self,
        Parameters(params): Parameters<TriplesOfParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;

        let sym_id = engine
            .resolve_symbol(&params.entity)
            .map_err(|e| McpError::invalid_params(format!("entity: {e}"), None))?;

        let outgoing = engine.triples_from(sym_id);
        let incoming = engine.triples_to(sym_id);

        let mut lines = Vec::new();

        if !outgoing.is_empty() {
            lines.push(format!("Outgoing ({}):", outgoing.len()));
            for t in &outgoing {
                lines.push(format!(
                    "  {} → {} → {} (conf={:.2})",
                    engine.resolve_label(t.subject),
                    engine.resolve_label(t.predicate),
                    engine.resolve_label(t.object),
                    t.confidence
                ));
            }
        }

        if !incoming.is_empty() {
            lines.push(format!("Incoming ({}):", incoming.len()));
            for t in &incoming {
                lines.push(format!(
                    "  {} → {} → {} (conf={:.2})",
                    engine.resolve_label(t.subject),
                    engine.resolve_label(t.predicate),
                    engine.resolve_label(t.object),
                    t.confidence
                ));
            }
        }

        if lines.is_empty() {
            lines.push(format!("No triples involving \"{}\"", params.entity));
        }

        Ok(CallToolResult::success(vec![Content::text(
            lines.join("\n"),
        )]))
    }

    #[tool(
        name = "provenance_of",
        description = "Trace the derivation chain for a symbol: how it was derived, from what sources, at what confidence."
    )]
    async fn provenance_of(
        &self,
        Parameters(params): Parameters<ProvenanceOfParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;

        let sym_id = engine
            .resolve_symbol(&params.entity)
            .map_err(|e| McpError::invalid_params(format!("entity: {e}"), None))?;

        let records = engine
            .provenance_of(sym_id)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;

        if records.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "No provenance records for \"{}\"",
                params.entity
            ))]));
        }

        let lines: Vec<String> = records
            .iter()
            .map(|r| {
                let sources: Vec<String> =
                    r.sources.iter().map(|s| format!("{}", s.get())).collect();
                format!(
                    "- id={} kind={:?} confidence={:.2} depth={} sources=[{}] ts={}",
                    r.id.map(|id| id.get()).unwrap_or(0),
                    r.kind,
                    r.confidence,
                    r.depth,
                    sources.join(", "),
                    r.timestamp
                )
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Provenance for \"{}\" ({} records):\n{}",
            params.entity,
            records.len(),
            lines.join("\n")
        ))]))
    }

    #[tool(
        name = "export_symbols",
        description = "Export the full symbol table with IDs, labels, kinds, and creation timestamps."
    )]
    async fn export_symbols(
        &self,
        Parameters(params): Parameters<WorkspaceParam>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let symbols = engine.export_symbol_table();

        if symbols.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No symbols in workspace.",
            )]));
        }

        let json = serde_json::to_string_pretty(&symbols)
            .map_err(|e| McpError::internal_error(format!("json: {e}"), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} symbols:\n{}",
            symbols.len(),
            json
        ))]))
    }

    #[tool(
        name = "infer_analogy",
        description = "VSA analogy inference: 'A is to B as C is to ?' Returns ranked candidate symbols with similarity scores."
    )]
    async fn infer_analogy(
        &self,
        Parameters(params): Parameters<InferAnalogyParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;

        let a = engine
            .resolve_symbol(&params.a)
            .map_err(|e| McpError::invalid_params(format!("a: {e}"), None))?;
        let b = engine
            .resolve_symbol(&params.b)
            .map_err(|e| McpError::invalid_params(format!("b: {e}"), None))?;
        let c = engine
            .resolve_symbol(&params.c)
            .map_err(|e| McpError::invalid_params(format!("c: {e}"), None))?;

        let top_k = params.top_k.unwrap_or(5);

        let results = engine
            .infer_analogy(a, b, c, top_k)
            .map_err(|e| McpError::internal_error(format!("{e}"), None))?;

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No analogy candidates found.",
            )]));
        }

        let lines: Vec<String> = results
            .iter()
            .map(|(sym_id, score)| {
                format!(
                    "  {} (similarity={:.3})",
                    engine.resolve_label(*sym_id),
                    score
                )
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(format!(
            "{} is to {} as {} is to:\n{}",
            params.a,
            params.b,
            params.c,
            lines.join("\n")
        ))]))
    }

    // ── VSA Grounding ───────────────────────────────────────────────

    #[tool(
        name = "ground_symbols",
        description = "Re-encode all symbol hypervectors using their relational neighborhoods (holographic reduced representations). Improves analogy and similarity quality by grounding vectors in KG structure rather than just labels."
    )]
    async fn ground_symbols(
        &self,
        Parameters(params): Parameters<WorkspaceParam>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let (grounded, total_rels) = engine
                .ground_all_symbols()
                .map_err(|e| format!("{e}"))?;
            let _ = engine.persist();
            Ok(format!(
                "Grounded {grounded} symbols using {total_rels} relations"
            ))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    // ── Event Calculus (Phase 15b) ─────────────────────────────────

    #[tool(
        name = "record_event",
        description = "Record a temporal event that initiates and/or terminates fluents (time-varying properties). Events are the fundamental building blocks of the event calculus — they represent things that happen at a point in time and change what is true."
    )]
    async fn record_event(
        &self,
        Parameters(params): Parameters<RecordEventParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let agent = self.state.get_agent(&params.workspace).await?;
        let event_name = params.name.clone();
        let timestamp = params.timestamp.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        });
        let initiates_labels = params.initiates.unwrap_or_default();
        let terminates_labels = params.terminates.unwrap_or_default();

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let event_id = engine
                .resolve_or_create_entity(&format!("ec:event:{event_name}"))
                .map_err(|e| format!("{e}"))?;

            let mut initiates = Vec::new();
            for label in &initiates_labels {
                let id = engine
                    .resolve_or_create_entity(&format!("ec:fluent:{label}"))
                    .map_err(|e| format!("{e}"))?;
                initiates.push(id);
            }

            let mut terminates = Vec::new();
            for label in &terminates_labels {
                let id = engine
                    .resolve_or_create_entity(&format!("ec:fluent:{label}"))
                    .map_err(|e| format!("{e}"))?;
                terminates.push(id);
            }

            let event = crate::agent::Event {
                symbol_id: event_id,
                name: event_name.clone(),
                timestamp,
                initiates: initiates.clone(),
                terminates: terminates.clone(),
            };

            let mut agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            agent
                .ec_engine_mut()
                .record_event(event, &engine)
                .map_err(|e| format!("{e}"))?;

            let _ = engine.persist();
            Ok(format!(
                "Recorded event \"{event_name}\" at t={timestamp}: initiates {} fluent(s), terminates {} fluent(s)",
                initiates.len(),
                terminates.len()
            ))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "holds_at",
        description = "Check whether a fluent (time-varying property) holds at a specific time. Evaluates the core Event Calculus axiom: a fluent holds if it was initiated by some event and not clipped (terminated) since."
    )]
    async fn holds_at(
        &self,
        Parameters(params): Parameters<HoldsAtParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let agent = self.state.get_agent(&params.workspace).await?;
        let fluent_label = params.fluent.clone();
        let time = params.time;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let fluent_id = engine
                .resolve_or_create_entity(&format!("ec:fluent:{fluent_label}"))
                .map_err(|e| format!("{e}"))?;

            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let holds = agent.ec_engine().holds_at(fluent_id, time);

            Ok(format!(
                "Fluent \"{fluent_label}\" {} at t={time}",
                if holds { "HOLDS" } else { "does NOT hold" }
            ))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "project_state",
        description = "Project temporal state: what fluents hold at a future time, what was terminated, and what events occurred in the interval. Useful for understanding the temporal evolution of the world."
    )]
    async fn project_state(
        &self,
        Parameters(params): Parameters<ProjectStateParams>,
    ) -> Result<CallToolResult, McpError> {
        let agent = self.state.get_agent(&params.workspace).await?;
        let from_time = params.from_time;
        let to_time = params.to_time;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let proj = agent.ec_engine().project_state(from_time, to_time);

            let mut lines = Vec::new();
            lines.push(format!(
                "State projection [{from_time} → {to_time}]:"
            ));
            lines.push(format!("  Holding fluents: {}", proj.holding.len()));
            for f in &proj.holding {
                lines.push(format!("    ✓ {}", f.label));
            }
            lines.push(format!("  Terminated: {}", proj.terminated.len()));
            for (f, _ev) in &proj.terminated {
                lines.push(format!("    ✗ {}", f.label));
            }
            lines.push(format!("  Events in interval: {}", proj.events.len()));
            for e in &proj.events {
                lines.push(format!("    @ t={}: {}", e.timestamp, e.name));
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "simulate_actions",
        description = "Simulate a hypothetical sequence of actions using the causal model. Predicts how fluents change at each step, with confidence decaying geometrically. Requires causal action schemas to be registered."
    )]
    async fn simulate_actions(
        &self,
        Parameters(params): Parameters<SimulateActionsParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let agent = self.state.get_agent(&params.workspace).await?;
        let action_names = params.actions.clone();

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let mut action_ids = Vec::new();
            for name in &action_names {
                let id = engine
                    .resolve_or_create_entity(&format!("tool:{name}"))
                    .map_err(|e| format!("{e}"))?;
                action_ids.push(id);
            }

            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let sim = agent
                .ec_engine()
                .simulate_actions(&action_ids, agent.causal_manager(), &engine)
                .map_err(|e| format!("{e}"))?;

            let mut lines = Vec::new();
            lines.push(format!(
                "Simulation of {} actions (confidence: {:.2}):",
                sim.action_sequence.len(),
                sim.confidence
            ));
            for (i, state) in sim.state_trajectory.iter().enumerate() {
                let holding: Vec<&str> = state
                    .iter()
                    .filter(|f| f.current_value)
                    .map(|f| f.label.as_str())
                    .collect();
                lines.push(format!(
                    "  Step {}: {} fluents holding [{}]",
                    i + 1,
                    holding.len(),
                    holding.join(", ")
                ));
            }
            lines.push(format!("  Final state: {} fluents holding", sim.final_state.len()));
            for f in &sim.final_state {
                lines.push(format!("    ✓ {}", f.label));
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "what_changed_since",
        description = "Temporal diff: what fluents were initiated or terminated since a given timestamp? Returns changes sorted by time."
    )]
    async fn what_changed_since(
        &self,
        Parameters(params): Parameters<WhatChangedSinceParams>,
    ) -> Result<CallToolResult, McpError> {
        let agent = self.state.get_agent(&params.workspace).await?;
        let since = params.since;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let changes = agent.ec_engine().what_changed_since(since);

            if changes.is_empty() {
                return Ok(format!("No fluent changes since t={since}"));
            }

            let mut lines = Vec::new();
            lines.push(format!("Changes since t={since}: ({} total)", changes.len()));
            for (fluent, event, initiated) in &changes {
                let status = if *initiated { "initiated" } else { "terminated" };
                lines.push(format!(
                    "  @ t={}: \"{}\" {} by \"{}\"",
                    event.timestamp, fluent.label, status, event.name
                ));
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "fluent_history",
        description = "Get the full history of a fluent: every initiation and termination event with timestamps."
    )]
    async fn fluent_history(
        &self,
        Parameters(params): Parameters<FluentHistoryParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let agent = self.state.get_agent(&params.workspace).await?;
        let fluent_label = params.fluent.clone();

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let fluent_id = engine
                .resolve_or_create_entity(&format!("ec:fluent:{fluent_label}"))
                .map_err(|e| format!("{e}"))?;

            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let history = agent
                .ec_engine()
                .fluent_history(fluent_id)
                .map_err(|e| format!("{e}"))?;

            if history.is_empty() {
                return Ok(format!("No history for fluent \"{fluent_label}\""));
            }

            let mut lines = Vec::new();
            lines.push(format!(
                "History of \"{}\" ({} entries):",
                fluent_label,
                history.len()
            ));
            for entry in &history {
                let action = if entry.initiated { "INITIATED" } else { "TERMINATED" };
                let event_label = agent
                    .ec_engine()
                    .get_event(entry.by_event)
                    .map(|e| e.name.as_str())
                    .unwrap_or("unknown");
                lines.push(format!(
                    "  @ t={}: {} by \"{}\"",
                    entry.timestamp, action, event_label
                ));
            }

            // Current status.
            if let Some(f) = agent.ec_engine().get_fluent(fluent_id) {
                lines.push(format!(
                    "  Current: {}",
                    if f.current_value { "HOLDING" } else { "NOT holding" }
                ));
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    // ── Counterfactual Reasoning (Phase 15c) ────────────────────────

    #[tool(
        name = "counterfactual",
        description = "Pearl Level 3 counterfactual reasoning: 'What would have happened if I had done action Y instead of action X?' Compares predicted outcomes and identifies divergent fluents. Both actions must have registered causal schemas."
    )]
    async fn counterfactual(
        &self,
        Parameters(params): Parameters<CounterfactualParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let agent = self.state.get_agent(&params.workspace).await?;
        let actual = params.actual_action.clone();
        let hypothetical = params.hypothetical_action.clone();

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let actual_id = engine
                .resolve_or_create_entity(&format!("tool:{actual}"))
                .map_err(|e| format!("{e}"))?;
            let hypo_id = engine
                .resolve_or_create_entity(&format!("tool:{hypothetical}"))
                .map_err(|e| format!("{e}"))?;

            let query = crate::agent::CounterfactualQuery {
                actual_action: actual_id,
                hypothetical_action: hypo_id,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };

            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let result = crate::agent::counterfactual_query(
                &query,
                agent.causal_manager(),
                &engine,
            )
            .map_err(|e| format!("{e}"))?;

            let mut lines = Vec::new();
            lines.push(format!(
                "Counterfactual: \"{actual}\" vs \"{hypothetical}\" (confidence: {:.2})",
                result.confidence
            ));
            lines.push(format!(
                "  Actual: {} assertions, {} retractions",
                result.actual_outcome.assertions.len(),
                result.actual_outcome.retractions.len()
            ));
            lines.push(format!(
                "  Hypothetical: {} assertions, {} retractions",
                result.hypothetical_outcome.assertions.len(),
                result.hypothetical_outcome.retractions.len()
            ));
            lines.push(format!(
                "  Divergent fluents: {}",
                result.divergent_fluents.len()
            ));
            for (sym, actual_holds, hypo_holds) in &result.divergent_fluents {
                let label = engine.resolve_label(*sym);
                lines.push(format!(
                    "    {label}: actual={actual_holds}, hypothetical={hypo_holds}"
                ));
            }
            match result.hypothetical_better {
                Some(true) => lines.push("  Verdict: hypothetical would have been BETTER".into()),
                Some(false) => lines.push("  Verdict: actual was BETTER".into()),
                None => lines.push("  Verdict: TIE (same net effect)".into()),
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    #[tool(
        name = "prediction_accuracy",
        description = "Show the causal model's prediction accuracy: overall accuracy, EMA, per-action breakdown, and suggestions for schemas that need refinement."
    )]
    async fn prediction_accuracy(
        &self,
        Parameters(params): Parameters<WorkspaceParam>,
    ) -> Result<CallToolResult, McpError> {
        let agent = self.state.get_agent(&params.workspace).await?;
        let engine = self.state.get_engine(&params.workspace).await?;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let tracker = agent.prediction_tracker();

            let mut lines = Vec::new();
            lines.push("Prediction Tracker:".into());
            lines.push(format!("  Predictions made: {}", tracker.predictions_made));
            lines.push(format!("  Correct: {}", tracker.predictions_correct));
            lines.push(format!("  Incorrect: {}", tracker.predictions_incorrect));
            lines.push(format!("  Pending: {}", tracker.pending_count()));
            lines.push(format!(
                "  Overall accuracy: {:.1}%",
                tracker.prediction_accuracy() * 100.0
            ));
            lines.push(format!("  EMA accuracy: {:.1}%", tracker.accuracy_ema * 100.0));

            if !tracker.per_action_accuracy.is_empty() {
                lines.push("  Per-action:".into());
                for (&action_key, &(correct, total)) in &tracker.per_action_accuracy {
                    let label = crate::symbol::SymbolId::new(action_key)
                        .map(|id| engine.resolve_label(id))
                        .unwrap_or_else(|| format!("#{action_key}"));
                    let acc = if total > 0 {
                        correct as f32 / total as f32 * 100.0
                    } else {
                        0.0
                    };
                    lines.push(format!(
                        "    {label}: {correct}/{total} ({acc:.0}%)"
                    ));
                }
            }

            let suggestions = tracker.refinement_suggestions(0.6, 3);
            if !suggestions.is_empty() {
                lines.push("  Refinement needed:".into());
                for (action_key, acc) in &suggestions {
                    let label = crate::symbol::SymbolId::new(*action_key)
                        .map(|id| engine.resolve_label(id))
                        .unwrap_or_else(|| format!("#{action_key}"));
                    lines.push(format!(
                        "    {label}: accuracy {:.0}% — update schema",
                        acc * 100.0
                    ));
                }
            }

            Ok(lines.join("\n"))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }

    // ── Chat ────────────────────────────────────────────────────────

    #[tool(
        name = "chat",
        description = "Send a message through the full NLU pipeline — handles dialogue acts, fact assertions, queries, and goal escalation, just like typing in the TUI."
    )]
    async fn chat(
        &self,
        Parameters(params): Parameters<ChatParams>,
    ) -> Result<CallToolResult, McpError> {
        let engine = self.state.get_engine(&params.workspace).await?;
        let agent = self.state.get_agent(&params.workspace).await?;
        let nlu_cached = self.state.get_nlu_pipeline(&params.workspace, &engine).await;
        let message = params.message;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            // Clone the shared pipeline — models are Arc-shared (cheap),
            // ranker is cloned by value for per-session learning.
            let session_pipeline = nlu_cached.lock().unwrap().clone();

            let mut chat_processor =
                crate::chat::ChatProcessor::new(&engine, session_pipeline);

            let mut agent = agent.lock().map_err(|e| format!("agent lock: {e}"))?;
            let responses = chat_processor.process_input(&message, &mut agent, &engine);

            // Persist NLU ranker state and engine.
            chat_processor.persist_nlu_state(&engine);
            let _ = engine.persist();
            let _ = agent.persist_session();

            let text = format_messages(&responses);
            if text.is_empty() {
                Ok("(no response)".to_string())
            } else {
                Ok(text)
            }
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e, None))?;

        Ok(CallToolResult::success(vec![Content::text(result)]))
    }
}

// ── ServerHandler impl ──────────────────────────────────────────────────

#[rmcp::tool_handler]
impl rmcp::handler::server::ServerHandler for AkhMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2025_03_26,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "akh-medu".to_string(),
                title: Some("akh-medu Knowledge Engine".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                description: Some(
                    "Neuro-symbolic AI knowledge engine with VSA, knowledge graphs, and symbolic reasoning".to_string(),
                ),
                icons: None,
                website_url: Some("https://akh-medu.dev".to_string()),
            },
            instructions: Some(
                "akh-medu is a neuro-symbolic AI knowledge engine. All workspace-scoped tools \
                 accept an optional `workspace` parameter (default: \"default\"). Start with \
                 `list_workspaces` to see available workspaces, or `create_workspace` to make one. \
                 Use `status` to see the current state. Use `chat` for natural-language interaction \
                 through the full NLU pipeline, or `ask` for autonomous investigation. Use \
                 `awaken` to bootstrap a workspace with domain knowledge.\n\n\
                 For fine-grained orchestration, use the granular bootstrap tools: \
                 `resolve_identity` → `ritual_of_awakening` → `expand_domain` → \
                 `analyze_prerequisites` → `assess_competence`. These let you control each stage \
                 independently, handle errors per-stage, and fill gaps (e.g. provide identity data \
                 when resolution fails).\n\n\
                 Agent introspection: `agent_run_cycle` (single OODA step), `agent_goals`, \
                 `agent_recall` (episodic memory), `agent_psyche`.\n\n\
                 Knowledge introspection: `triples_of`, `provenance_of`, `export_symbols`, \
                 `infer_analogy`, `remove_triple`.\n\n\
                 Event calculus (temporal reasoning): `record_event` (create events that initiate/terminate \
                 fluents), `holds_at` (does fluent hold at time?), `project_state` (state at future time), \
                 `simulate_actions` (predict action sequence outcomes), `what_changed_since` (temporal diff), \
                 `fluent_history` (full lifecycle of a fluent).\n\n\
                 Counterfactual reasoning: `counterfactual` (Pearl Level 3: what if action Y instead \
                 of X?), `prediction_accuracy` (causal model accuracy stats and refinement suggestions)."
                    .to_string(),
            ),
        }
    }
}
