//! Heka MCP server.
//!
//! Exposes Heka memory tools via the Model Context Protocol:
//! - `heka_remember` — store a memory drawer
//! - `heka_recall`   — retrieve drawers by subject
//! - `heka_contradict` — ACH-driven contradiction analysis (Phase 42i)

pub mod tools;

use std::sync::Arc;

use akh_medu::engine::Engine;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::tool::ToolCallContext;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::service::RequestContext;
use rmcp::{ErrorData as McpError, RoleServer, tool, tool_router};
use rmcp::schemars::JsonSchema;
use serde::Deserialize;

use crate::drawer::DrawerStore;
use tools::contradict::{HekaContradictParams, heka_contradict_handler};

// ═══════════════════════════════════════════════════════════════════════
// Shared state
// ═══════════════════════════════════════════════════════════════════════

/// Shared state for all Heka MCP tool handlers.
pub struct HekaMcpState {
    /// Drawer store (redb-backed).
    pub store: Arc<DrawerStore>,
    /// Optional akh-medu engine for KG evidence lookup.
    pub engine: Option<Arc<Engine>>,
}

// ═══════════════════════════════════════════════════════════════════════
// Parameter types
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Deserialize, JsonSchema)]
struct RememberParams {
    #[schemars(description = "Short subject label (e.g. \"temperature\", \"meeting notes\")")]
    pub subject: String,

    #[schemars(description = "Full claim text to remember (e.g. \"temperature is 37 degrees\")")]
    pub claim: String,

    #[schemars(description = "Source of this claim (e.g. \"alice\", \"thermometer-A\")")]
    pub source: Option<String>,

    #[schemars(description = "Confidence [0.0-1.0] in this claim (default: 0.8)")]
    pub confidence: Option<f32>,

    #[schemars(description = "Unix timestamp when this was observed (default: now)")]
    pub timestamp: Option<u64>,

    #[schemars(description = "Tool that captured this (e.g. \"browser\", \"manual\")")]
    pub tool: Option<String>,

    #[schemars(description = "Freeform topic tags")]
    #[serde(default)]
    pub topic_ids: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
struct RecallParams {
    #[schemars(description = "Subject keyword to search for (case-insensitive)")]
    pub subject: String,

    #[schemars(description = "Maximum drawers to return (default: 20)")]
    pub limit: Option<usize>,
}

// ═══════════════════════════════════════════════════════════════════════
// MCP server
// ═══════════════════════════════════════════════════════════════════════

/// MCP server exposing Heka memory tools.
#[derive(Clone)]
pub struct HekaMcpServer {
    state: Arc<HekaMcpState>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl HekaMcpServer {
    pub fn new(state: Arc<HekaMcpState>) -> Self {
        Self {
            state,
            tool_router: Self::tool_router(),
        }
    }

    // ── heka_remember ─────────────────────────────────────────────────

    #[tool(
        name = "heka_remember",
        description = "Store a memory drawer. Returns the assigned drawer ID."
    )]
    async fn heka_remember(
        &self,
        Parameters(params): Parameters<RememberParams>,
    ) -> Result<CallToolResult, McpError> {
        let store = Arc::clone(&self.state.store);
        let confidence = params.confidence.unwrap_or(0.8);

        let id = tokio::task::spawn_blocking(move || {
            store.insert(
                &params.subject,
                &params.claim,
                params.source,
                confidence,
                params.timestamp,
                params.tool,
                params.topic_ids,
            )
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(format!(
            "Drawer {id} stored."
        ))]))
    }

    // ── heka_recall ───────────────────────────────────────────────────

    #[tool(
        name = "heka_recall",
        description = "Retrieve drawers matching a subject keyword. Returns IDs, claims, timestamps, and confidence scores."
    )]
    async fn heka_recall(
        &self,
        Parameters(params): Parameters<RecallParams>,
    ) -> Result<CallToolResult, McpError> {
        let store = Arc::clone(&self.state.store);
        let limit = params.limit.unwrap_or(20);
        let subject = params.subject.clone();

        let records = tokio::task::spawn_blocking(move || {
            store.search_by_subject(&params.subject)
        })
        .await
        .map_err(|e| McpError::internal_error(format!("task panicked: {e}"), None))?
        .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        if records.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "No drawers found for subject \"{subject}\".",
            ))]));
        }

        let mut lines = vec![format!(
            "Found {} drawer(s) for \"{}\":",
            records.len().min(limit),
            subject
        )];
        for r in records.iter().take(limit) {
            let source = r.source.as_deref().unwrap_or("unknown");
            lines.push(format!(
                "  [{}] \"{claim}\"  conf={conf:.2}  ts={ts}  source={source}",
                r.id,
                claim = r.claim,
                conf = r.confidence,
                ts = r.timestamp,
            ));
        }

        Ok(CallToolResult::success(vec![Content::text(
            lines.join("\n"),
        )]))
    }

    // ── heka_contradict ───────────────────────────────────────────────

    #[tool(
        name = "heka_contradict",
        description = "Analyze a contradiction between two memory drawers using ACH + Dempster-Shafer evidence theory (Phase 18 + Phase 17). \
        Returns per-side confidence intervals [Bel, Pl], inconsistency scores, a ranked verdict, and the deciding-evidence chain \
        with citations to specific drawer IDs and KG triples. \
        Demo: ingest two contradictory drawers then call heka_contradict(drawer_a=ID1, drawer_b=ID2)."
    )]
    async fn heka_contradict(
        &self,
        Parameters(params): Parameters<HekaContradictParams>,
    ) -> Result<CallToolResult, McpError> {
        heka_contradict_handler(
            Arc::clone(&self.state.store),
            self.state.engine.clone(),
            params,
        )
        .await
    }
}

impl rmcp::ServerHandler for HekaMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::LATEST,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "heka".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                title: None,
                description: None,
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Heka memory daemon. Use heka_remember to store claims, \
                heka_recall to retrieve them, and heka_contradict to \
                analyze conflicts between two drawers via ACH + Dempster-Shafer."
                    .into(),
            ),
        }
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        Ok(ListToolsResult {
            tools: self.tool_router.list_all(),
            meta: None,
            next_cursor: None,
        })
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let tcc = ToolCallContext::new(self, request, context);
        self.tool_router.call(tcc).await
    }

    fn get_tool(&self, name: &str) -> Option<Tool> {
        self.tool_router.get(name).cloned()
    }
}
