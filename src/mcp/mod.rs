//! MCP (Model Context Protocol) server integration for akhomed.
//!
//! Exposes akh-medu's knowledge engine as MCP tools so that AI assistants
//! (e.g. Claude Code) can query, mutate, and manage the knowledge graph
//! via the standard MCP protocol over HTTP.
//!
//! The [`AkhMcpServer`] struct implements rmcp's `ServerHandler` trait and
//! is mounted at `/mcp` on the akhomed axum router via `StreamableHttpService`.

use std::sync::{Arc, Mutex};

use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::schemars;
use rmcp::{ErrorData as McpError, tool, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::agent::Agent;
use crate::engine::Engine;

/// Shared state needed by MCP tool handlers — mirrors what axum handlers use.
pub struct McpState {
    /// Engine instance for the target workspace.
    pub engine: Arc<Engine>,
    /// Shared agent — same one the daemon and HTTP handlers use.
    pub agent: Arc<Mutex<Agent>>,
}

// ── Tool parameter structs ──────────────────────────────────────────────

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AskParams {
    #[schemars(description = "Natural-language question to investigate")]
    pub question: String,
    #[schemars(description = "Maximum OODA cycles to run (default: 10)")]
    pub max_cycles: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SparqlParams {
    #[schemars(description = "SPARQL SELECT query string")]
    pub query: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchParams {
    #[schemars(description = "Search term (entity name or partial match)")]
    pub term: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AssertTripleParams {
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
    #[schemars(description = "Text to extract knowledge from")]
    pub text: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct IngestUrlParams {
    #[schemars(description = "URL to fetch and ingest")]
    pub url: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CompartmentIdParams {
    #[schemars(description = "Compartment ID")]
    pub id: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RunAgentParams {
    #[schemars(description = "Comma-separated goals for the agent to pursue")]
    pub goals: String,
    #[schemars(description = "Maximum OODA cycles (default: 10)")]
    pub max_cycles: Option<u32>,
}

// ── MCP Server ──────────────────────────────────────────────────────────

/// MCP server exposing akh-medu tools to AI assistants.
///
/// Each instance targets a single workspace. Tool calls access the engine
/// and agent directly (no HTTP roundtrip — in-process).
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
        let state = Arc::clone(&self.state);
        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let mut agent = state.agent.lock().map_err(|e| format!("agent lock: {e}"))?;
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
        let engine = Arc::clone(&self.state.engine);
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
        let engine = Arc::clone(&self.state.engine);
        let term = &params.term;

        // First try exact symbol resolution, then fall back to substring search.
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
            // Substring search across all symbols.
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
        let engine = Arc::clone(&self.state.engine);
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
        let engine = Arc::clone(&self.state.engine);
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
        let engine = Arc::clone(&self.state.engine);
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
    async fn list_compartments(&self) -> Result<CallToolResult, McpError> {
        let engine = Arc::clone(&self.state.engine);
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
    async fn discover_compartments(&self) -> Result<CallToolResult, McpError> {
        let engine = Arc::clone(&self.state.engine);
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
        let engine = Arc::clone(&self.state.engine);
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
        let engine = Arc::clone(&self.state.engine);
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
        let engine = Arc::clone(&self.state.engine);
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
        let engine = Arc::clone(&self.state.engine);
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
        let state = Arc::clone(&self.state);
        let result = tokio::task::spawn_blocking(move || -> Result<String, String> {
            let mut agent = state.agent.lock().map_err(|e| format!("agent lock: {e}"))?;
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

    // ── Workspace ───────────────────────────────────────────────────

    #[tool(
        name = "status",
        description = "Get workspace status: symbol count, triple count, loaded compartments, and engine info."
    )]
    async fn status(&self) -> Result<CallToolResult, McpError> {
        let engine = Arc::clone(&self.state.engine);
        let info = engine.info();
        let mut lines = vec![
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
                "akh-medu is a neuro-symbolic AI knowledge engine. Use the tools to query, \
                 mutate, and manage knowledge in the graph. Start with `status` to see the \
                 current state, then use `list_compartments` to see available knowledge domains. \
                 Load and activate compartments to scope queries. Use `ask` for natural-language \
                 questions or `sparql_query` for precise graph queries."
                    .to_string(),
            ),
        }
    }
}
