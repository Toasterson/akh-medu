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
        }
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
        let message = params.message;

        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            // Build NLU pipeline with persisted ranker state.
            let data_dir = engine.config().data_dir.as_deref();
            let nlu_pipeline = engine
                .store()
                .get_meta(b"nlu_ranker_state")
                .ok()
                .flatten()
                .and_then(|bytes| {
                    crate::nlu::parse_ranker::ParseRanker::from_bytes(&bytes)
                })
                .map(|ranker| {
                    crate::nlu::NluPipeline::with_ranker_and_models(ranker, data_dir)
                })
                .unwrap_or_else(|| crate::nlu::NluPipeline::new_with_models(data_dir));

            let mut chat_processor =
                crate::chat::ChatProcessor::new(&engine, nlu_pipeline);

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
                 `awaken` to bootstrap a workspace with domain knowledge."
                    .to_string(),
            ),
        }
    }
}
