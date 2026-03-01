//! Client abstraction for talking to akh-medu engines.
//!
//! `AkhClient` wraps either a local `Arc<Engine>` or an HTTP connection to
//! an `akhomed` server instance. The CLI resolves which variant to use at
//! startup via [`discover_server`].

use std::sync::Arc;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use std::path::Path;

#[cfg(not(feature = "client-only"))]
use crate::agent::trigger::TriggerStore;
use crate::agent::trigger::Trigger;
use crate::engine::{Engine, EngineInfo};
use crate::export::{ProvenanceExport, SymbolExport, TripleExport};
use crate::graph::Triple;
use crate::graph::analytics;
use crate::graph::traverse::{TraversalConfig, TraversalResult};
use crate::infer::{InferenceQuery, InferenceResult};
#[cfg(not(feature = "client-only"))]
use crate::library::{ContentFormat, IngestConfig, LibraryCatalog};
use crate::library::{
    DocumentRecord, LibraryAddRequest, LibraryAddResponse,
    LibrarySearchRequest, LibrarySearchResult,
};
use crate::paths::AkhPaths;
use crate::skills::{SkillActivation, SkillInfo, SkillInstallPayload};
use crate::symbol::{SymbolId, SymbolKind, SymbolMeta};
use crate::vsa::item_memory::SearchResult;

// ---------------------------------------------------------------------------
// Server discovery
// ---------------------------------------------------------------------------

/// Information about a running akhomed instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    pub pid: u32,
    pub port: u16,
    pub bind: String,
}

impl ServerInfo {
    /// Base URL for HTTP requests.
    pub fn base_url(&self) -> String {
        let host = if self.bind == "0.0.0.0" {
            "127.0.0.1"
        } else {
            &self.bind
        };
        format!("http://{host}:{}", self.port)
    }
}

/// Discover a running akhomed server via its PID file.
///
/// Returns `Some(ServerInfo)` when:
/// 1. The PID file exists and parses correctly
/// 2. The process is still alive (`kill(pid, 0)` succeeds)
/// 3. The server responds to `GET /health`
pub fn discover_server(paths: &AkhPaths) -> Option<ServerInfo> {
    let pid_path = paths.pid_file();
    let contents = match std::fs::read_to_string(&pid_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!(path = %pid_path.display(), error = %e, "PID file not readable");
            return None;
        }
    };
    let info: ServerInfo = match serde_json::from_str(&contents) {
        Ok(i) => i,
        Err(e) => {
            tracing::warn!(path = %pid_path.display(), error = %e, "PID file corrupt");
            return None;
        }
    };

    // Check process is alive.
    if !process_alive(info.pid) {
        tracing::debug!(pid = info.pid, "stale PID file — process not alive, removing");
        let _ = std::fs::remove_file(&pid_path);
        return None;
    }

    // Health-check the server.
    let url = format!("{}/health", info.base_url());
    match ureq::get(&url).timeout(std::time::Duration::from_secs(2)).call() {
        Ok(resp) if resp.status() == 200 => Some(info),
        Ok(resp) => {
            tracing::warn!(
                pid = info.pid, url = %url, status = resp.status(),
                "akhomed process alive but health check returned non-200"
            );
            None
        }
        Err(e) => {
            tracing::warn!(
                pid = info.pid, url = %url, error = %e,
                "akhomed process alive but health check failed"
            );
            None
        }
    }
}

/// Write a PID file for the current akhomed process.
pub fn write_pid_file(paths: &AkhPaths, port: u16, bind: &str) -> std::io::Result<()> {
    let info = ServerInfo {
        pid: std::process::id(),
        port,
        bind: bind.to_string(),
    };
    let json = serde_json::to_string_pretty(&info).expect("ServerInfo is always serializable");
    std::fs::write(paths.pid_file(), json)
}

/// Remove the PID file on shutdown.
pub fn remove_pid_file(paths: &AkhPaths) {
    let _ = std::fs::remove_file(paths.pid_file());
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    // SAFETY: kill with signal 0 doesn't actually send a signal;
    // it only checks whether the process exists.
    unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
}

#[cfg(not(unix))]
fn process_alive(_pid: u32) -> bool {
    // On non-unix, fall back to trusting the PID file.
    true
}

// ---------------------------------------------------------------------------
// Client error
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Diagnostic)]
pub enum ClientError {
    #[error("remote request failed: {message}")]
    #[diagnostic(code(akh::client::request), help("Is akhomed running?"))]
    Request { message: String },

    #[error("unexpected response from server: {message}")]
    #[diagnostic(code(akh::client::response), help("Server version mismatch?"))]
    Response { message: String },

    #[error(transparent)]
    #[diagnostic(transparent)]
    Engine(#[from] crate::error::AkhError),

    #[error("no akhomed server found — client-only mode requires a running server")]
    #[diagnostic(
        code(akh::client::no_server),
        help("{hint}")
    )]
    NoServer { hint: String },
}

pub type ClientResult<T> = Result<T, ClientError>;

// ---------------------------------------------------------------------------
// Daemon status (shared between client & server)
// ---------------------------------------------------------------------------

/// Status of a workspace daemon (background agent task manager).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub running: bool,
    pub total_cycles: usize,
    pub started_at: u64,
    pub trigger_count: usize,
    /// Unix timestamp of last successful session persist.
    #[serde(default)]
    pub last_persist_at: Option<u64>,
    /// Unix timestamp of last continuous learning run.
    #[serde(default)]
    pub last_learning_at: Option<u64>,
    /// Unix timestamp of last sleep/consolidation cycle.
    #[serde(default)]
    pub last_sleep_at: Option<u64>,
    /// Unix timestamp of last goal generation run.
    #[serde(default)]
    pub last_goal_gen_at: Option<u64>,
    /// Number of currently active goals.
    #[serde(default)]
    pub active_goals: usize,
    /// Total symbols in knowledge graph.
    #[serde(default)]
    pub kg_symbols: usize,
    /// Total triples in knowledge graph.
    #[serde(default)]
    pub kg_triples: usize,
}

// ---------------------------------------------------------------------------
// AkhClient
// ---------------------------------------------------------------------------

/// Either a local engine or a remote HTTP connection to akhomed.
pub enum AkhClient {
    /// Direct local engine access.
    #[cfg(not(feature = "client-only"))]
    Local(Arc<Engine>),
    /// HTTP client to a running akhomed server.
    Remote {
        base_url: String,
        workspace: String,
        http: ureq::Agent,
    },
}

/// Require a running akhomed server (client-only mode).
///
/// Returns `AkhClient::Remote` if a server is discovered, or
/// `ClientError::NoServer` if no server is available.
pub fn require_server(workspace: &str, paths: Option<&AkhPaths>) -> ClientResult<AkhClient> {
    if let Some(paths) = paths {
        if let Some(server) = discover_server(paths) {
            return Ok(AkhClient::remote(&server, workspace));
        }
        let pid_path = paths.pid_file();
        let hint = if pid_path.exists() {
            format!(
                "PID file exists at {} but the health check failed.\n\
                 Check akhomed logs: journalctl --user -u akh-medu.akhomed  (Linux)\n\
                 Or: log show --predicate 'process==\"akhomed\"' --last 5m  (macOS)\n\
                 Or build without --features client-only to use a local engine.",
                pid_path.display()
            )
        } else {
            format!(
                "No PID file at {}.\n\
                 Start the server: akhomed  (or: cargo run --features server --bin akhomed)\n\
                 Or build without --features client-only to use a local engine.",
                pid_path.display()
            )
        };
        return Err(ClientError::NoServer { hint });
    }
    Err(ClientError::NoServer {
        hint: "Could not resolve XDG paths. Set $HOME or $XDG_RUNTIME_DIR.".to_string(),
    })
}

impl AkhClient {
    /// Connect to a discovered server.
    pub fn remote(info: &ServerInfo, workspace: &str) -> Self {
        AkhClient::Remote {
            base_url: info.base_url(),
            workspace: workspace.to_string(),
            http: ureq::Agent::new(),
        }
    }

    /// Wrap a local engine.
    #[cfg(not(feature = "client-only"))]
    pub fn local(engine: Arc<Engine>) -> Self {
        AkhClient::Local(engine)
    }

    /// Returns true if this is a remote client.
    pub fn is_remote(&self) -> bool {
        matches!(self, AkhClient::Remote { .. })
    }

    /// Get reference to local engine, if available.
    #[cfg(not(feature = "client-only"))]
    pub fn engine(&self) -> Option<&Arc<Engine>> {
        match self {
            AkhClient::Local(e) => Some(e),
            AkhClient::Remote { .. } => None,
        }
    }

    /// Get reference to local engine, if available.
    #[cfg(feature = "client-only")]
    pub fn engine(&self) -> Option<&Arc<Engine>> {
        None
    }

    // -- helpers for remote calls --

    /// Extract the (base_url, workspace, http) triple from `Remote`.
    /// Panics on `Local` — callers are gated so this never happens at runtime.
    fn remote_parts(&self) -> (&str, &str, &ureq::Agent) {
        match self {
            AkhClient::Remote {
                base_url,
                workspace,
                http,
            } => (base_url, workspace, http),
            #[cfg(not(feature = "client-only"))]
            _ => unreachable!("remote_parts called on local client"),
        }
    }

    #[allow(dead_code)] // Reserved for WebSocket-based TUI streaming
    fn ws_url(&self, path: &str) -> String {
        let (base_url, workspace, _) = self.remote_parts();
        format!("{base_url}/workspaces/{workspace}{path}")
    }

    fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> ClientResult<T> {
        let (base_url, workspace, http) = self.remote_parts();
        let url = format!("{base_url}/workspaces/{workspace}{path}");
        let resp = http
            .get(&url)
            .call()
            .map_err(|e| ClientError::Request {
                message: e.to_string(),
            })?;
        resp.into_json().map_err(|e| ClientError::Response {
            message: format!("failed to parse JSON: {e}"),
        })
    }

    fn post_json<B: Serialize, T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> ClientResult<T> {
        let (base_url, workspace, http) = self.remote_parts();
        let url = format!("{base_url}/workspaces/{workspace}{path}");
        let resp = http
            .post(&url)
            .send_json(body)
            .map_err(|e| ClientError::Request {
                message: e.to_string(),
            })?;
        resp.into_json().map_err(|e| ClientError::Response {
            message: format!("failed to parse JSON: {e}"),
        })
    }

    fn delete_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> ClientResult<T> {
        let (base_url, workspace, http) = self.remote_parts();
        let url = format!("{base_url}/workspaces/{workspace}{path}");
        let resp = http
            .delete(&url)
            .call()
            .map_err(|e| ClientError::Request {
                message: e.to_string(),
            })?;
        resp.into_json().map_err(|e| ClientError::Response {
            message: format!("failed to parse JSON: {e}"),
        })
    }

    // -----------------------------------------------------------------------
    // Symbol operations
    // -----------------------------------------------------------------------

    pub fn resolve_symbol(&self, name_or_id: &str) -> ClientResult<SymbolId> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.resolve_symbol(name_or_id)?);
        }
        #[derive(Deserialize)]
        struct Resp {
            id: u64,
        }
        let resp: Resp = self.get_json(&format!("/symbols/{name_or_id}"))?;
        SymbolId::new(resp.id).ok_or_else(|| ClientError::Response {
            message: format!("server returned invalid symbol id: {}", resp.id),
        })
    }

    pub fn create_symbol(
        &self,
        kind: SymbolKind,
        label: &str,
    ) -> ClientResult<SymbolMeta> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.create_symbol(kind, label)?);
        }
        #[derive(Serialize)]
        struct Req<'a> {
            kind: &'a str,
            label: &'a str,
        }
        let kind_str = match kind {
            SymbolKind::Entity => "entity",
            SymbolKind::Relation => "relation",
            _ => "entity",
        };
        self.post_json("/symbols", &Req { kind: kind_str, label })
    }

    pub fn lookup_symbol(&self, label: &str) -> ClientResult<SymbolId> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.lookup_symbol(label)?);
        }
        self.resolve_symbol(label)
    }

    pub fn resolve_label(&self, id: SymbolId) -> ClientResult<String> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.resolve_label(id));
        }
        #[derive(Deserialize)]
        struct Resp {
            label: String,
        }
        let resp: Resp = self.get_json(&format!("/symbols/{}", id.get()))?;
        Ok(resp.label)
    }

    pub fn all_symbols(&self) -> ClientResult<Vec<SymbolMeta>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.all_symbols());
        }
        self.get_json("/symbols")
    }

    // -----------------------------------------------------------------------
    // Triple operations
    // -----------------------------------------------------------------------

    pub fn add_triple(&self, triple: &Triple) -> ClientResult<()> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.add_triple(triple)?);
        }
        #[derive(Serialize)]
        struct Req {
            subject: u64,
            predicate: u64,
            object: u64,
            confidence: f32,
        }
        let _: serde_json::Value = self.post_json(
            "/triples",
            &Req {
                subject: triple.subject.get(),
                predicate: triple.predicate.get(),
                object: triple.object.get(),
                confidence: triple.confidence,
            },
        )?;
        Ok(())
    }

    pub fn triples_from(&self, symbol: SymbolId) -> ClientResult<Vec<Triple>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.triples_from(symbol));
        }
        self.get_json(&format!("/triples/from/{}", symbol.get()))
    }

    pub fn triples_to(&self, symbol: SymbolId) -> ClientResult<Vec<Triple>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.triples_to(symbol));
        }
        self.get_json(&format!("/triples/to/{}", symbol.get()))
    }

    pub fn all_triples(&self) -> ClientResult<Vec<Triple>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.all_triples());
        }
        self.get_json("/triples")
    }

    pub fn ingest_label_triples(
        &self,
        triples: &[(String, String, String, f32)],
    ) -> ClientResult<(usize, usize)> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.ingest_label_triples(triples)?);
        }
        #[derive(Serialize)]
        struct Req<'a> {
            triples: &'a [(String, String, String, f32)],
        }
        #[derive(Deserialize)]
        struct Resp {
            symbols_created: usize,
            triples_ingested: usize,
        }
        let resp: Resp = self.post_json("/ingest", &Req { triples })?;
        Ok((resp.symbols_created, resp.triples_ingested))
    }

    // -----------------------------------------------------------------------
    // Query & reasoning
    // -----------------------------------------------------------------------

    pub fn sparql_query(
        &self,
        sparql: &str,
    ) -> ClientResult<Vec<Vec<(String, String)>>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.sparql_query(sparql)?);
        }
        #[derive(Serialize)]
        struct Req<'a> {
            query: &'a str,
        }
        self.post_json("/sparql", &Req { query: sparql })
    }

    pub fn infer(&self, query: &InferenceQuery) -> ClientResult<InferenceResult> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.infer(query)?);
        }
        self.post_json("/infer", query)
    }

    pub fn simplify_expression(&self, expr: &str) -> ClientResult<String> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.simplify_expression(expr)?);
        }
        #[derive(Serialize)]
        struct Req<'a> {
            expr: &'a str,
        }
        #[derive(Deserialize)]
        struct Resp {
            result: String,
        }
        let resp: Resp = self.post_json("/reason", &Req { expr })?;
        Ok(resp.result)
    }

    pub fn search_similar_to(
        &self,
        symbol: SymbolId,
        top_k: usize,
    ) -> ClientResult<Vec<SearchResult>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.search_similar_to(symbol, top_k)?);
        }
        #[derive(Serialize)]
        struct Req {
            symbol: u64,
            top_k: usize,
        }
        self.post_json(
            "/search",
            &Req {
                symbol: symbol.get(),
                top_k,
            },
        )
    }

    pub fn infer_analogy(
        &self,
        a: SymbolId,
        b: SymbolId,
        c: SymbolId,
        top_k: usize,
    ) -> ClientResult<Vec<(SymbolId, f32)>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.infer_analogy(a, b, c, top_k)?);
        }
        #[derive(Serialize)]
        struct Req {
            a: u64,
            b: u64,
            c: u64,
            top_k: usize,
        }
        #[derive(Deserialize)]
        struct Item {
            symbol: u64,
            score: f32,
        }
        let items: Vec<Item> = self.post_json(
            "/analogy",
            &Req {
                a: a.get(),
                b: b.get(),
                c: c.get(),
                top_k,
            },
        )?;
        Ok(items
            .into_iter()
            .filter_map(|i| Some((SymbolId::new(i.symbol)?, i.score)))
            .collect())
    }

    pub fn recover_filler(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        top_k: usize,
    ) -> ClientResult<Vec<(SymbolId, f32)>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.recover_filler(subject, predicate, top_k)?);
        }
        #[derive(Serialize)]
        struct Req {
            subject: u64,
            predicate: u64,
            top_k: usize,
        }
        #[derive(Deserialize)]
        struct Item {
            symbol: u64,
            score: f32,
        }
        let items: Vec<Item> = self.post_json(
            "/filler",
            &Req {
                subject: subject.get(),
                predicate: predicate.get(),
                top_k,
            },
        )?;
        Ok(items
            .into_iter()
            .filter_map(|i| Some((SymbolId::new(i.symbol)?, i.score)))
            .collect())
    }

    // -----------------------------------------------------------------------
    // Graph traversal & analytics
    // -----------------------------------------------------------------------

    pub fn traverse(
        &self,
        seeds: &[SymbolId],
        config: TraversalConfig,
    ) -> ClientResult<TraversalResult> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.traverse(seeds, config)?);
        }
        #[derive(Serialize)]
        struct Req<'a> {
            seeds: Vec<u64>,
            #[serde(flatten)]
            config: &'a TraversalConfig,
        }
        let req = Req {
            seeds: seeds.iter().map(|s| s.get()).collect(),
            config: &config,
        };
        self.post_json("/traverse", &req)
    }

    pub fn degree_centrality(&self) -> ClientResult<Vec<analytics::DegreeCentrality>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.degree_centrality());
        }
        self.get_json("/analytics/centrality")
    }

    pub fn pagerank(
        &self,
        damping: f64,
        iterations: usize,
    ) -> ClientResult<Vec<analytics::PageRankScore>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.pagerank(damping, iterations)?);
        }
        self.get_json(&format!(
            "/analytics/pagerank?damping={damping}&iterations={iterations}"
        ))
    }

    pub fn strongly_connected_components(
        &self,
    ) -> ClientResult<Vec<analytics::ConnectedComponent>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.strongly_connected_components()?);
        }
        self.get_json("/analytics/components")
    }

    pub fn shortest_path(
        &self,
        from: SymbolId,
        to: SymbolId,
    ) -> ClientResult<Option<Vec<SymbolId>>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.shortest_path(from, to)?);
        }
        #[derive(Serialize)]
        struct Req {
            from: u64,
            to: u64,
        }
        self.post_json(
            "/analytics/shortest-path",
            &Req {
                from: from.get(),
                to: to.get(),
            },
        )
    }

    // -----------------------------------------------------------------------
    // Export
    // -----------------------------------------------------------------------

    pub fn export_symbols(&self) -> ClientResult<Vec<SymbolExport>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.export_symbol_table());
        }
        self.get_json("/export/symbols")
    }

    pub fn export_triples(&self) -> ClientResult<Vec<TripleExport>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.export_triples());
        }
        self.get_json("/export/triples")
    }

    pub fn export_provenance(
        &self,
        symbol: SymbolId,
    ) -> ClientResult<Vec<ProvenanceExport>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.export_provenance_chain(symbol)?);
        }
        self.get_json(&format!("/export/provenance/{}", symbol.get()))
    }

    // -----------------------------------------------------------------------
    // Skills
    // -----------------------------------------------------------------------

    pub fn list_skills(&self) -> ClientResult<Vec<SkillInfo>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.list_skills());
        }
        self.get_json("/skills")
    }

    pub fn load_skill(&self, name: &str) -> ClientResult<SkillActivation> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.load_skill(name)?);
        }
        self.post_json(&format!("/skills/{name}/load"), &serde_json::json!({}))
    }

    pub fn unload_skill(&self, name: &str) -> ClientResult<()> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.unload_skill(name)?);
        }
        let _: serde_json::Value =
            self.post_json(&format!("/skills/{name}/unload"), &serde_json::json!({}))?;
        Ok(())
    }

    pub fn skill_info(&self, name: &str) -> ClientResult<SkillInfo> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.skill_info(name)?);
        }
        self.get_json(&format!("/skills/{name}"))
    }

    pub fn install_skill(&self, payload: &SkillInstallPayload) -> ClientResult<SkillActivation> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.install_skill(payload)?);
        }
        self.post_json("/skills/install", payload)
    }

    // -----------------------------------------------------------------------
    // Engine info
    // -----------------------------------------------------------------------

    pub fn info(&self) -> ClientResult<EngineInfo> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.info());
        }
        self.get_json("/info")
    }

    // -----------------------------------------------------------------------
    // Workspace role
    // -----------------------------------------------------------------------

    /// Assign a role to the workspace (write-once).
    pub fn assign_role(&self, role: &str) -> ClientResult<()> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(e) = self {
            return Ok(e.assign_role(role)?);
        }
        #[derive(Serialize)]
        struct Req<'a> {
            role: &'a str,
        }
        let _: serde_json::Value =
            self.post_json("/assign-role", &Req { role })?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Library
    // -----------------------------------------------------------------------

    /// List all documents in the library.
    pub fn library_list(&self, library_dir: &Path) -> ClientResult<Vec<DocumentRecord>> {
        #[cfg(not(feature = "client-only"))]
        if matches!(self, AkhClient::Local(_)) {
            let catalog = LibraryCatalog::open(library_dir).map_err(|e| {
                ClientError::Engine(crate::error::AkhError::Library(e))
            })?;
            return Ok(catalog.list().to_vec());
        }
        let _ = library_dir;
        self.get_json("/library")
    }

    /// Add a document to the library. Returns ingestion summary.
    pub fn library_add(
        &self,
        library_dir: &Path,
        req: &LibraryAddRequest,
    ) -> ClientResult<LibraryAddResponse> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(engine) = self {
            let mut catalog = LibraryCatalog::open(library_dir).map_err(|e| {
                ClientError::Engine(crate::error::AkhError::Library(e))
            })?;

            let fmt = req.format.as_deref().and_then(|f| match f {
                "html" => Some(ContentFormat::Html),
                "pdf" => Some(ContentFormat::Pdf),
                "epub" => Some(ContentFormat::Epub),
                "text" | "txt" => Some(ContentFormat::PlainText),
                _ => None,
            });

            let ingest_config = IngestConfig {
                title: req.title.clone(),
                tags: req.tags.clone(),
                format: fmt,
                ..Default::default()
            };

            let result = if req.source.starts_with("http://")
                || req.source.starts_with("https://")
            {
                crate::library::ingest_url(engine, &mut catalog, &req.source, ingest_config, None)
                    .map_err(|e| ClientError::Engine(crate::error::AkhError::Library(e)))?
            } else {
                let path = std::path::PathBuf::from(&req.source);
                crate::library::ingest_file(engine, &mut catalog, &path, ingest_config, None)
                    .map_err(|e| ClientError::Engine(crate::error::AkhError::Library(e)))?
            };

            // Persist symbols, registry, and allocator state to durable store.
            engine.persist().map_err(ClientError::Engine)?;

            return Ok(LibraryAddResponse {
                id: result.record.id,
                title: result.record.title,
                format: result.record.format.to_string(),
                chunk_count: result.chunk_count,
                triple_count: result.triple_count,
                concept_count: result.concept_count,
            });
        }
        let _ = library_dir;
        self.post_json("/library", req)
    }

    /// Remove a document from the library by ID.
    pub fn library_remove(&self, library_dir: &Path, id: &str) -> ClientResult<DocumentRecord> {
        #[cfg(not(feature = "client-only"))]
        if matches!(self, AkhClient::Local(_)) {
            let mut catalog = LibraryCatalog::open(library_dir).map_err(|e| {
                ClientError::Engine(crate::error::AkhError::Library(e))
            })?;
            return catalog.remove(id).map_err(|e| {
                ClientError::Engine(crate::error::AkhError::Library(e))
            });
        }
        let _ = library_dir;
        self.delete_json(&format!("/library/{id}"))
    }

    /// Get info for a single document by ID.
    pub fn library_info(&self, library_dir: &Path, id: &str) -> ClientResult<DocumentRecord> {
        #[cfg(not(feature = "client-only"))]
        if matches!(self, AkhClient::Local(_)) {
            let catalog = LibraryCatalog::open(library_dir).map_err(|e| {
                ClientError::Engine(crate::error::AkhError::Library(e))
            })?;
            return catalog
                .get(id)
                .cloned()
                .ok_or_else(|| ClientError::Response {
                    message: format!("document not found: \"{id}\""),
                });
        }
        let _ = library_dir;
        self.get_json(&format!("/library/{id}"))
    }

    /// Search library content by text similarity.
    pub fn library_search(
        &self,
        query: &str,
        top_k: usize,
    ) -> ClientResult<Vec<LibrarySearchResult>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(engine) = self {
            use crate::vsa::encode::encode_label;

            let query_vec = encode_label(engine.ops(), query).map_err(|e| {
                ClientError::Engine(e.into())
            })?;
            let results = engine
                .item_memory()
                .search(&query_vec, top_k)
                .map_err(|e| ClientError::Engine(e.into()))?;

            return Ok(results
                .into_iter()
                .enumerate()
                .map(|(rank, sr)| {
                    let label = engine.resolve_label(sr.symbol_id);
                    LibrarySearchResult {
                        rank: rank + 1,
                        symbol_label: label,
                        similarity: sr.similarity,
                    }
                })
                .collect());
        }
        let req = LibrarySearchRequest {
            query: query.to_string(),
            top_k,
        };
        self.post_json("/library/search", &req)
    }

    // -----------------------------------------------------------------------
    // Awaken status
    // -----------------------------------------------------------------------

    /// Query the awaken (psyche/identity) status.
    pub fn awaken_status(&self) -> ClientResult<serde_json::Value> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(engine) = self {
            let psyche = engine.compartments().and_then(|m| m.psyche());

            let agent_config = crate::agent::AgentConfig::default();
            let agent = crate::agent::Agent::new(Arc::clone(engine), agent_config)
                .map_err(|e| ClientError::Engine(crate::error::AkhError::Agent(e)))?;

            let active_goals = agent
                .goals()
                .iter()
                .filter(|g| matches!(g.status, crate::agent::GoalStatus::Active))
                .count();

            return Ok(serde_json::json!({
                "awakened": psyche.is_some(),
                "psyche": psyche,
                "active_goals": active_goals,
                "total_goals": agent.goals().len(),
                "cycle_count": agent.cycle_count(),
            }));
        }
        self.get_json("/awaken/status")
    }

    // -----------------------------------------------------------------------
    // Daemon (server-side only — Local returns an error)
    // -----------------------------------------------------------------------

    /// Start a workspace daemon. Requires akhomed.
    pub fn start_daemon(
        &self,
        config: Option<serde_json::Value>,
    ) -> ClientResult<DaemonStatus> {
        #[cfg(not(feature = "client-only"))]
        if matches!(self, AkhClient::Local(_)) {
            return Err(ClientError::Request {
                message: "daemon requires akhomed — start the server first".into(),
            });
        }
        let body = config.unwrap_or(serde_json::json!({}));
        self.post_json("/daemon/start", &body)
    }

    /// Stop a workspace daemon. Requires akhomed.
    pub fn stop_daemon(&self) -> ClientResult<()> {
        #[cfg(not(feature = "client-only"))]
        if matches!(self, AkhClient::Local(_)) {
            return Err(ClientError::Request {
                message: "daemon requires akhomed — start the server first".into(),
            });
        }
        let _: serde_json::Value =
            self.post_json("/daemon/stop", &serde_json::json!({}))?;
        Ok(())
    }

    /// Get daemon status. Requires akhomed.
    pub fn daemon_status(&self) -> ClientResult<DaemonStatus> {
        #[cfg(not(feature = "client-only"))]
        if matches!(self, AkhClient::Local(_)) {
            return Err(ClientError::Request {
                message: "daemon requires akhomed — start the server first".into(),
            });
        }
        self.get_json("/daemon")
    }

    // -----------------------------------------------------------------------
    // Triggers
    // -----------------------------------------------------------------------

    /// List all registered triggers.
    pub fn list_triggers(&self) -> ClientResult<Vec<Trigger>> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(engine) = self {
            let store = TriggerStore::new(engine);
            return Ok(store.list());
        }
        self.get_json("/triggers")
    }

    /// Add a trigger.
    pub fn add_trigger(&self, trigger: &Trigger) -> ClientResult<Trigger> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(engine) = self {
            let store = TriggerStore::new(engine);
            return store.add(trigger.clone()).map_err(|e| {
                ClientError::Engine(crate::error::AkhError::Agent(e))
            });
        }
        self.post_json("/triggers", trigger)
    }

    /// Remove a trigger by ID.
    pub fn remove_trigger(&self, id: &str) -> ClientResult<()> {
        #[cfg(not(feature = "client-only"))]
        if let AkhClient::Local(engine) = self {
            let store = TriggerStore::new(engine);
            return store.remove(id).map_err(|e| {
                ClientError::Engine(crate::error::AkhError::Agent(e))
            });
        }
        let _: serde_json::Value =
            self.delete_json(&format!("/triggers/{id}"))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Seeds (remote-only — local uses SeedRegistry directly)
    // -----------------------------------------------------------------------

    /// List available seed packs.
    pub fn seed_list(&self) -> ClientResult<Vec<crate::api_types::SeedPackInfo>> {
        self.get_json_root("/seeds")
    }

    /// Get seed application status for the workspace.
    pub fn seed_status(&self) -> ClientResult<crate::api_types::SeedStatusResponse> {
        self.get_json("/seeds/status")
    }

    /// Apply a seed pack.
    pub fn seed_apply(&self, pack: &str) -> ClientResult<serde_json::Value> {
        self.post_json(&format!("/seed/{pack}"), &serde_json::json!({}))
    }

    // -----------------------------------------------------------------------
    // Render (remote-only — local uses glyph module directly)
    // -----------------------------------------------------------------------

    /// Render knowledge in hieroglyphic notation.
    pub fn render(
        &self,
        req: &crate::api_types::RenderRequest,
    ) -> ClientResult<crate::api_types::RenderResponse> {
        self.post_json("/render", req)
    }

    // -----------------------------------------------------------------------
    // Agent run/resume (remote-only — local creates Agent directly)
    // -----------------------------------------------------------------------

    /// Run agent with goals.
    pub fn agent_run(
        &self,
        req: &crate::api_types::AgentRunRequest,
    ) -> ClientResult<crate::api_types::AgentRunResponse> {
        self.post_json("/agent/run", req)
    }

    /// Resume a persisted agent session.
    pub fn agent_resume(
        &self,
        req: &crate::api_types::AgentResumeRequest,
    ) -> ClientResult<crate::api_types::AgentResumeResponse> {
        self.post_json("/agent/resume", req)
    }

    // -----------------------------------------------------------------------
    // PIM (remote-only — local creates Agent directly)
    // -----------------------------------------------------------------------

    /// Get inbox tasks.
    pub fn pim_inbox(&self) -> ClientResult<crate::api_types::PimTaskList> {
        self.get_json("/pim/inbox")
    }

    /// Get next actions.
    pub fn pim_next(
        &self,
        req: &crate::api_types::PimNextRequest,
    ) -> ClientResult<crate::api_types::PimTaskList> {
        self.post_json("/pim/next", req)
    }

    /// Run GTD weekly review.
    pub fn pim_review(&self) -> ClientResult<crate::api_types::PimReviewResponse> {
        self.get_json("/pim/review")
    }

    /// Get project tasks.
    pub fn pim_project(&self, name: &str) -> ClientResult<crate::api_types::PimProjectResponse> {
        self.get_json(&format!("/pim/project/{name}"))
    }

    /// Add PIM metadata to a goal.
    pub fn pim_add(
        &self,
        req: &crate::api_types::PimAddRequest,
    ) -> ClientResult<crate::api_types::PimAddResponse> {
        self.post_json("/pim/add", req)
    }

    /// Transition a task's GTD state.
    pub fn pim_transition(
        &self,
        req: &crate::api_types::PimTransitionRequest,
    ) -> ClientResult<serde_json::Value> {
        self.post_json("/pim/transition", req)
    }

    /// Get Eisenhower matrix.
    pub fn pim_matrix(&self) -> ClientResult<crate::api_types::PimMatrixResponse> {
        self.get_json("/pim/matrix")
    }

    /// Get dependency graph.
    pub fn pim_deps(&self) -> ClientResult<crate::api_types::PimDepsResponse> {
        self.get_json("/pim/deps")
    }

    /// Get overdue tasks.
    pub fn pim_overdue(&self) -> ClientResult<crate::api_types::PimTaskList> {
        self.get_json("/pim/overdue")
    }

    // -----------------------------------------------------------------------
    // Causal (remote-only — local creates Agent directly)
    // -----------------------------------------------------------------------

    /// List causal schemas.
    pub fn causal_schemas(&self) -> ClientResult<Vec<crate::api_types::CausalSchemaSummary>> {
        self.get_json("/causal/schemas")
    }

    /// Get a specific causal schema.
    pub fn causal_schema(
        &self,
        name: &str,
    ) -> ClientResult<crate::api_types::CausalSchemaDetail> {
        self.get_json(&format!("/causal/schemas/{name}"))
    }

    /// Predict effects of an action.
    pub fn causal_predict(
        &self,
        req: &crate::api_types::CausalPredictRequest,
    ) -> ClientResult<crate::api_types::CausalPredictResponse> {
        self.post_json("/causal/predict", req)
    }

    /// List applicable actions.
    pub fn causal_applicable(&self) -> ClientResult<Vec<crate::api_types::CausalSchemaSummary>> {
        self.get_json("/causal/applicable")
    }

    /// Bootstrap schemas from tool registry.
    pub fn causal_bootstrap(&self) -> ClientResult<crate::api_types::CausalBootstrapResponse> {
        self.post_json("/causal/bootstrap", &serde_json::json!({}))
    }

    // -----------------------------------------------------------------------
    // Preference (remote-only — local creates Agent directly)
    // -----------------------------------------------------------------------

    /// Get preference profile status.
    pub fn pref_status(&self) -> ClientResult<crate::api_types::PrefStatusResponse> {
        self.get_json("/pref/status")
    }

    /// Train preference with feedback.
    pub fn pref_train(
        &self,
        req: &crate::api_types::PrefTrainRequest,
    ) -> ClientResult<crate::api_types::PrefTrainResponse> {
        self.post_json("/pref/train", req)
    }

    /// Set proactivity level.
    pub fn pref_level(
        &self,
        req: &crate::api_types::PrefLevelRequest,
    ) -> ClientResult<serde_json::Value> {
        self.post_json("/pref/level", req)
    }

    /// Get top interests.
    pub fn pref_interests(
        &self,
        count: usize,
    ) -> ClientResult<Vec<crate::api_types::PrefInterest>> {
        self.get_json(&format!("/pref/interests?count={count}"))
    }

    /// Run JITIR and get suggestions.
    pub fn pref_suggest(&self) -> ClientResult<serde_json::Value> {
        self.get_json("/pref/suggest")
    }

    // -----------------------------------------------------------------------
    // Calendar (remote-only — local creates Agent directly)
    // -----------------------------------------------------------------------

    /// Get today's events.
    pub fn cal_today(&self) -> ClientResult<crate::api_types::CalEventList> {
        self.get_json("/cal/today")
    }

    /// Get this week's events.
    pub fn cal_week(&self) -> ClientResult<crate::api_types::CalEventList> {
        self.get_json("/cal/week")
    }

    /// Detect scheduling conflicts.
    pub fn cal_conflicts(&self) -> ClientResult<Vec<crate::api_types::CalConflict>> {
        self.get_json("/cal/conflicts")
    }

    /// Add a calendar event.
    pub fn cal_add(
        &self,
        req: &crate::api_types::CalAddRequest,
    ) -> ClientResult<crate::api_types::CalAddResponse> {
        self.post_json("/cal/add", req)
    }

    /// Import events from iCalendar data.
    pub fn cal_import(
        &self,
        req: &crate::api_types::CalImportRequest,
    ) -> ClientResult<crate::api_types::CalImportResponse> {
        self.post_json("/cal/import", req)
    }

    /// Sync a CalDAV calendar. Requires akhomed with calendar feature.
    pub fn cal_sync(
        &self,
        req: &crate::api_types::CalSyncRequest,
    ) -> ClientResult<crate::api_types::CalImportResponse> {
        self.post_json("/cal/sync", req)
    }

    // -----------------------------------------------------------------------
    // Ingest (CSV / Text — remote-only)
    // -----------------------------------------------------------------------

    /// Ingest CSV content (SPO or entity format). Requires akhomed.
    pub fn ingest_csv(
        &self,
        req: &crate::api_types::CsvIngestRequest,
    ) -> ClientResult<crate::api_types::IngestResponse> {
        self.post_json("/ingest/csv", req)
    }

    /// Ingest natural language text (regex triple extraction). Requires akhomed.
    pub fn ingest_text(
        &self,
        req: &crate::api_types::TextIngestRequest,
    ) -> ClientResult<crate::api_types::IngestResponse> {
        self.post_json("/ingest/text", req)
    }

    // -----------------------------------------------------------------------
    // Library scan (remote-only)
    // -----------------------------------------------------------------------

    /// Scan the library inbox for new files and ingest them. Requires akhomed.
    pub fn library_scan(
        &self,
        req: &crate::api_types::LibraryScanRequest,
    ) -> ClientResult<crate::api_types::LibraryScanResponse> {
        self.post_json("/library/scan", req)
    }

    // -----------------------------------------------------------------------
    // Awaken (remote-only — local uses bootstrap module directly)
    // -----------------------------------------------------------------------

    /// Parse a purpose statement.
    pub fn awaken_parse(
        &self,
        req: &crate::api_types::AwakenParseRequest,
    ) -> ClientResult<crate::api_types::AwakenParseResponse> {
        self.post_json("/awaken/parse", req)
    }

    /// Resolve an identity reference.
    pub fn awaken_resolve(
        &self,
        req: &crate::api_types::AwakenResolveRequest,
    ) -> ClientResult<crate::api_types::AwakenResolveResponse> {
        self.post_json("/awaken/resolve", req)
    }

    /// Expand domain from seed concepts.
    pub fn awaken_expand(
        &self,
        req: &crate::api_types::AwakenExpandRequest,
    ) -> ClientResult<crate::api_types::AwakenExpandResponse> {
        self.post_json("/awaken/expand", req)
    }

    /// Discover prerequisites.
    pub fn awaken_prerequisite(
        &self,
        req: &crate::api_types::AwakenPrerequisiteRequest,
    ) -> ClientResult<crate::api_types::AwakenPrerequisiteResponse> {
        self.post_json("/awaken/prerequisite", req)
    }

    /// Discover learning resources.
    pub fn awaken_resources(
        &self,
        req: &crate::api_types::AwakenResourcesRequest,
    ) -> ClientResult<crate::api_types::AwakenResourcesResponse> {
        self.post_json("/awaken/resources", req)
    }

    /// Ingest discovered resources.
    pub fn awaken_ingest(
        &self,
        req: &crate::api_types::AwakenIngestRequest,
    ) -> ClientResult<crate::api_types::AwakenIngestResponse> {
        self.post_json("/awaken/ingest", req)
    }

    /// Assess competence.
    pub fn awaken_assess(
        &self,
        req: &crate::api_types::AwakenAssessRequest,
    ) -> ClientResult<crate::api_types::AwakenAssessResponse> {
        self.post_json("/awaken/assess", req)
    }

    /// Run full bootstrap pipeline.
    pub fn awaken_bootstrap(
        &self,
        req: &crate::api_types::AwakenBootstrapRequest,
    ) -> ClientResult<crate::api_types::AwakenBootstrapResponse> {
        self.post_json("/awaken/bootstrap", req)
    }

    // -----------------------------------------------------------------------
    // Workspace management (remote-only for client-only)
    // -----------------------------------------------------------------------

    /// List all workspaces.
    pub fn workspace_list(&self) -> ClientResult<Vec<String>> {
        #[derive(Deserialize)]
        struct Resp {
            workspaces: Vec<String>,
        }
        let resp: Resp = self.get_json_root("/workspaces")?;
        Ok(resp.workspaces)
    }

    /// Create a workspace.
    pub fn workspace_create(
        &self,
        name: &str,
        role: Option<&str>,
    ) -> ClientResult<crate::api_types::WorkspaceCreateResponse> {
        let req = crate::api_types::WorkspaceCreateRequest {
            role: role.map(|s| s.to_string()),
        };
        self.post_json_root(&format!("/workspaces/{name}"), &req)
    }

    /// Delete a workspace.
    pub fn workspace_delete(
        &self,
        name: &str,
    ) -> ClientResult<crate::api_types::WorkspaceDeleteResponse> {
        self.delete_json_root(&format!("/workspaces/{name}"))
    }

    // -----------------------------------------------------------------------
    // Root-level HTTP helpers (not workspace-scoped)
    // -----------------------------------------------------------------------

    fn get_json_root<T: serde::de::DeserializeOwned>(&self, path: &str) -> ClientResult<T> {
        let (base_url, _, http) = self.remote_parts();
        let url = format!("{base_url}{path}");
        let resp = http
            .get(&url)
            .call()
            .map_err(|e| ClientError::Request {
                message: e.to_string(),
            })?;
        resp.into_json().map_err(|e| ClientError::Response {
            message: format!("failed to parse JSON: {e}"),
        })
    }

    fn post_json_root<B: Serialize, T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> ClientResult<T> {
        let (base_url, _, http) = self.remote_parts();
        let url = format!("{base_url}{path}");
        let resp = http
            .post(&url)
            .send_json(body)
            .map_err(|e| ClientError::Request {
                message: e.to_string(),
            })?;
        resp.into_json().map_err(|e| ClientError::Response {
            message: format!("failed to parse JSON: {e}"),
        })
    }

    fn delete_json_root<T: serde::de::DeserializeOwned>(&self, path: &str) -> ClientResult<T> {
        let (base_url, _, http) = self.remote_parts();
        let url = format!("{base_url}{path}");
        let resp = http
            .delete(&url)
            .call()
            .map_err(|e| ClientError::Request {
                message: e.to_string(),
            })?;
        resp.into_json().map_err(|e| ClientError::Response {
            message: format!("failed to parse JSON: {e}"),
        })
    }
}
