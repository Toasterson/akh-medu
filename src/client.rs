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

use crate::agent::trigger::{Trigger, TriggerStore};
use crate::engine::{Engine, EngineInfo};
use crate::export::{ProvenanceExport, SymbolExport, TripleExport};
use crate::graph::Triple;
use crate::graph::analytics;
use crate::graph::traverse::{TraversalConfig, TraversalResult};
use crate::infer::{InferenceQuery, InferenceResult};
use crate::library::{
    ContentFormat, DocumentRecord, IngestConfig, LibraryAddRequest, LibraryAddResponse,
    LibraryCatalog, LibrarySearchRequest, LibrarySearchResult,
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
    let contents = std::fs::read_to_string(&pid_path).ok()?;
    let info: ServerInfo = serde_json::from_str(&contents).ok()?;

    // Check process is alive.
    if !process_alive(info.pid) {
        // Stale PID file — clean up.
        let _ = std::fs::remove_file(&pid_path);
        return None;
    }

    // Health-check the server.
    let url = format!("{}/health", info.base_url());
    match ureq::get(&url).timeout(std::time::Duration::from_secs(2)).call() {
        Ok(resp) if resp.status() == 200 => Some(info),
        _ => None,
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
}

// ---------------------------------------------------------------------------
// AkhClient
// ---------------------------------------------------------------------------

/// Either a local engine or a remote HTTP connection to akhomed.
pub enum AkhClient {
    /// Direct local engine access.
    Local(Arc<Engine>),
    /// HTTP client to a running akhomed server.
    Remote {
        base_url: String,
        workspace: String,
        http: ureq::Agent,
    },
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
    pub fn local(engine: Arc<Engine>) -> Self {
        AkhClient::Local(engine)
    }

    /// Returns true if this is a remote client.
    pub fn is_remote(&self) -> bool {
        matches!(self, AkhClient::Remote { .. })
    }

    /// Get reference to local engine, if available.
    pub fn engine(&self) -> Option<&Arc<Engine>> {
        match self {
            AkhClient::Local(e) => Some(e),
            AkhClient::Remote { .. } => None,
        }
    }

    // -- helpers for remote calls --

    fn ws_url(&self, path: &str) -> String {
        match self {
            AkhClient::Remote {
                base_url,
                workspace,
                ..
            } => format!("{base_url}/workspaces/{workspace}{path}"),
            _ => unreachable!("ws_url called on local client"),
        }
    }

    fn get_json<T: serde::de::DeserializeOwned>(&self, path: &str) -> ClientResult<T> {
        let AkhClient::Remote {
            base_url,
            workspace,
            http,
        } = self
        else {
            unreachable!("get_json called on local client");
        };
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
        let AkhClient::Remote {
            base_url,
            workspace,
            http,
        } = self
        else {
            unreachable!("post_json called on local client");
        };
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
        let AkhClient::Remote {
            base_url,
            workspace,
            http,
        } = self
        else {
            unreachable!("delete_json called on local client");
        };
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
        match self {
            AkhClient::Local(e) => Ok(e.resolve_symbol(name_or_id)?),
            AkhClient::Remote { .. } => {
                #[derive(Deserialize)]
                struct Resp {
                    id: u64,
                }
                let resp: Resp = self.get_json(&format!("/symbols/{name_or_id}"))?;
                SymbolId::new(resp.id).ok_or_else(|| ClientError::Response {
                    message: format!("server returned invalid symbol id: {}", resp.id),
                })
            }
        }
    }

    pub fn create_symbol(
        &self,
        kind: SymbolKind,
        label: &str,
    ) -> ClientResult<SymbolMeta> {
        match self {
            AkhClient::Local(e) => Ok(e.create_symbol(kind, label)?),
            AkhClient::Remote { .. } => {
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
        }
    }

    pub fn lookup_symbol(&self, label: &str) -> ClientResult<SymbolId> {
        match self {
            AkhClient::Local(e) => Ok(e.lookup_symbol(label)?),
            AkhClient::Remote { .. } => self.resolve_symbol(label),
        }
    }

    pub fn resolve_label(&self, id: SymbolId) -> ClientResult<String> {
        match self {
            AkhClient::Local(e) => Ok(e.resolve_label(id)),
            AkhClient::Remote { .. } => {
                #[derive(Deserialize)]
                struct Resp {
                    label: String,
                }
                let resp: Resp = self.get_json(&format!("/symbols/{}", id.get()))?;
                Ok(resp.label)
            }
        }
    }

    pub fn all_symbols(&self) -> ClientResult<Vec<SymbolMeta>> {
        match self {
            AkhClient::Local(e) => Ok(e.all_symbols()),
            AkhClient::Remote { .. } => self.get_json("/symbols"),
        }
    }

    // -----------------------------------------------------------------------
    // Triple operations
    // -----------------------------------------------------------------------

    pub fn add_triple(&self, triple: &Triple) -> ClientResult<()> {
        match self {
            AkhClient::Local(e) => Ok(e.add_triple(triple)?),
            AkhClient::Remote { .. } => {
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
        }
    }

    pub fn triples_from(&self, symbol: SymbolId) -> ClientResult<Vec<Triple>> {
        match self {
            AkhClient::Local(e) => Ok(e.triples_from(symbol)),
            AkhClient::Remote { .. } => {
                self.get_json(&format!("/triples/from/{}", symbol.get()))
            }
        }
    }

    pub fn triples_to(&self, symbol: SymbolId) -> ClientResult<Vec<Triple>> {
        match self {
            AkhClient::Local(e) => Ok(e.triples_to(symbol)),
            AkhClient::Remote { .. } => {
                self.get_json(&format!("/triples/to/{}", symbol.get()))
            }
        }
    }

    pub fn all_triples(&self) -> ClientResult<Vec<Triple>> {
        match self {
            AkhClient::Local(e) => Ok(e.all_triples()),
            AkhClient::Remote { .. } => self.get_json("/triples"),
        }
    }

    pub fn ingest_label_triples(
        &self,
        triples: &[(String, String, String, f32)],
    ) -> ClientResult<(usize, usize)> {
        match self {
            AkhClient::Local(e) => Ok(e.ingest_label_triples(triples)?),
            AkhClient::Remote { .. } => {
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
        }
    }

    // -----------------------------------------------------------------------
    // Query & reasoning
    // -----------------------------------------------------------------------

    pub fn sparql_query(
        &self,
        sparql: &str,
    ) -> ClientResult<Vec<Vec<(String, String)>>> {
        match self {
            AkhClient::Local(e) => Ok(e.sparql_query(sparql)?),
            AkhClient::Remote { .. } => {
                #[derive(Serialize)]
                struct Req<'a> {
                    query: &'a str,
                }
                self.post_json("/sparql", &Req { query: sparql })
            }
        }
    }

    pub fn infer(&self, query: &InferenceQuery) -> ClientResult<InferenceResult> {
        match self {
            AkhClient::Local(e) => Ok(e.infer(query)?),
            AkhClient::Remote { .. } => self.post_json("/infer", query),
        }
    }

    pub fn simplify_expression(&self, expr: &str) -> ClientResult<String> {
        match self {
            AkhClient::Local(e) => Ok(e.simplify_expression(expr)?),
            AkhClient::Remote { .. } => {
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
        }
    }

    pub fn search_similar_to(
        &self,
        symbol: SymbolId,
        top_k: usize,
    ) -> ClientResult<Vec<SearchResult>> {
        match self {
            AkhClient::Local(e) => Ok(e.search_similar_to(symbol, top_k)?),
            AkhClient::Remote { .. } => {
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
        }
    }

    pub fn infer_analogy(
        &self,
        a: SymbolId,
        b: SymbolId,
        c: SymbolId,
        top_k: usize,
    ) -> ClientResult<Vec<(SymbolId, f32)>> {
        match self {
            AkhClient::Local(e) => Ok(e.infer_analogy(a, b, c, top_k)?),
            AkhClient::Remote { .. } => {
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
        }
    }

    pub fn recover_filler(
        &self,
        subject: SymbolId,
        predicate: SymbolId,
        top_k: usize,
    ) -> ClientResult<Vec<(SymbolId, f32)>> {
        match self {
            AkhClient::Local(e) => Ok(e.recover_filler(subject, predicate, top_k)?),
            AkhClient::Remote { .. } => {
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
        }
    }

    // -----------------------------------------------------------------------
    // Graph traversal & analytics
    // -----------------------------------------------------------------------

    pub fn traverse(
        &self,
        seeds: &[SymbolId],
        config: TraversalConfig,
    ) -> ClientResult<TraversalResult> {
        match self {
            AkhClient::Local(e) => Ok(e.traverse(seeds, config)?),
            AkhClient::Remote { .. } => {
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
        }
    }

    pub fn degree_centrality(&self) -> ClientResult<Vec<analytics::DegreeCentrality>> {
        match self {
            AkhClient::Local(e) => Ok(e.degree_centrality()),
            AkhClient::Remote { .. } => self.get_json("/analytics/centrality"),
        }
    }

    pub fn pagerank(
        &self,
        damping: f64,
        iterations: usize,
    ) -> ClientResult<Vec<analytics::PageRankScore>> {
        match self {
            AkhClient::Local(e) => Ok(e.pagerank(damping, iterations)?),
            AkhClient::Remote { .. } => {
                self.get_json(&format!(
                    "/analytics/pagerank?damping={damping}&iterations={iterations}"
                ))
            }
        }
    }

    pub fn strongly_connected_components(
        &self,
    ) -> ClientResult<Vec<analytics::ConnectedComponent>> {
        match self {
            AkhClient::Local(e) => Ok(e.strongly_connected_components()?),
            AkhClient::Remote { .. } => self.get_json("/analytics/components"),
        }
    }

    pub fn shortest_path(
        &self,
        from: SymbolId,
        to: SymbolId,
    ) -> ClientResult<Option<Vec<SymbolId>>> {
        match self {
            AkhClient::Local(e) => Ok(e.shortest_path(from, to)?),
            AkhClient::Remote { .. } => {
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
        }
    }

    // -----------------------------------------------------------------------
    // Export
    // -----------------------------------------------------------------------

    pub fn export_symbols(&self) -> ClientResult<Vec<SymbolExport>> {
        match self {
            AkhClient::Local(e) => Ok(e.export_symbol_table()),
            AkhClient::Remote { .. } => self.get_json("/export/symbols"),
        }
    }

    pub fn export_triples(&self) -> ClientResult<Vec<TripleExport>> {
        match self {
            AkhClient::Local(e) => Ok(e.export_triples()),
            AkhClient::Remote { .. } => self.get_json("/export/triples"),
        }
    }

    pub fn export_provenance(
        &self,
        symbol: SymbolId,
    ) -> ClientResult<Vec<ProvenanceExport>> {
        match self {
            AkhClient::Local(e) => Ok(e.export_provenance_chain(symbol)?),
            AkhClient::Remote { .. } => {
                self.get_json(&format!("/export/provenance/{}", symbol.get()))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Skills
    // -----------------------------------------------------------------------

    pub fn list_skills(&self) -> ClientResult<Vec<SkillInfo>> {
        match self {
            AkhClient::Local(e) => Ok(e.list_skills()),
            AkhClient::Remote { .. } => self.get_json("/skills"),
        }
    }

    pub fn load_skill(&self, name: &str) -> ClientResult<SkillActivation> {
        match self {
            AkhClient::Local(e) => Ok(e.load_skill(name)?),
            AkhClient::Remote { .. } => {
                self.post_json(&format!("/skills/{name}/load"), &serde_json::json!({}))
            }
        }
    }

    pub fn unload_skill(&self, name: &str) -> ClientResult<()> {
        match self {
            AkhClient::Local(e) => Ok(e.unload_skill(name)?),
            AkhClient::Remote { .. } => {
                let _: serde_json::Value =
                    self.post_json(&format!("/skills/{name}/unload"), &serde_json::json!({}))?;
                Ok(())
            }
        }
    }

    pub fn skill_info(&self, name: &str) -> ClientResult<SkillInfo> {
        match self {
            AkhClient::Local(e) => Ok(e.skill_info(name)?),
            AkhClient::Remote { .. } => self.get_json(&format!("/skills/{name}")),
        }
    }

    pub fn install_skill(&self, payload: &SkillInstallPayload) -> ClientResult<SkillActivation> {
        match self {
            AkhClient::Local(e) => Ok(e.install_skill(payload)?),
            AkhClient::Remote { .. } => self.post_json("/skills/install", payload),
        }
    }

    // -----------------------------------------------------------------------
    // Engine info
    // -----------------------------------------------------------------------

    pub fn info(&self) -> ClientResult<EngineInfo> {
        match self {
            AkhClient::Local(e) => Ok(e.info()),
            AkhClient::Remote { .. } => self.get_json("/info"),
        }
    }

    // -----------------------------------------------------------------------
    // Library
    // -----------------------------------------------------------------------

    /// List all documents in the library.
    pub fn library_list(&self, library_dir: &Path) -> ClientResult<Vec<DocumentRecord>> {
        match self {
            AkhClient::Local(_) => {
                let catalog = LibraryCatalog::open(library_dir).map_err(|e| {
                    ClientError::Engine(crate::error::AkhError::Library(e))
                })?;
                Ok(catalog.list().to_vec())
            }
            AkhClient::Remote { .. } => self.get_json("/library"),
        }
    }

    /// Add a document to the library. Returns ingestion summary.
    pub fn library_add(
        &self,
        library_dir: &Path,
        req: &LibraryAddRequest,
    ) -> ClientResult<LibraryAddResponse> {
        match self {
            AkhClient::Local(engine) => {
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
                    crate::library::ingest_url(engine, &mut catalog, &req.source, ingest_config)
                        .map_err(|e| ClientError::Engine(crate::error::AkhError::Library(e)))?
                } else {
                    let path = std::path::PathBuf::from(&req.source);
                    crate::library::ingest_file(engine, &mut catalog, &path, ingest_config)
                        .map_err(|e| ClientError::Engine(crate::error::AkhError::Library(e)))?
                };

                Ok(LibraryAddResponse {
                    id: result.record.id,
                    title: result.record.title,
                    format: result.record.format.to_string(),
                    chunk_count: result.chunk_count,
                    triple_count: result.triple_count,
                })
            }
            AkhClient::Remote { .. } => self.post_json("/library", req),
        }
    }

    /// Remove a document from the library by ID.
    pub fn library_remove(&self, library_dir: &Path, id: &str) -> ClientResult<DocumentRecord> {
        match self {
            AkhClient::Local(_) => {
                let mut catalog = LibraryCatalog::open(library_dir).map_err(|e| {
                    ClientError::Engine(crate::error::AkhError::Library(e))
                })?;
                catalog.remove(id).map_err(|e| {
                    ClientError::Engine(crate::error::AkhError::Library(e))
                })
            }
            AkhClient::Remote { .. } => self.delete_json(&format!("/library/{id}")),
        }
    }

    /// Get info for a single document by ID.
    pub fn library_info(&self, library_dir: &Path, id: &str) -> ClientResult<DocumentRecord> {
        match self {
            AkhClient::Local(_) => {
                let catalog = LibraryCatalog::open(library_dir).map_err(|e| {
                    ClientError::Engine(crate::error::AkhError::Library(e))
                })?;
                catalog
                    .get(id)
                    .cloned()
                    .ok_or_else(|| ClientError::Response {
                        message: format!("document not found: \"{id}\""),
                    })
            }
            AkhClient::Remote { .. } => self.get_json(&format!("/library/{id}")),
        }
    }

    /// Search library content by text similarity.
    pub fn library_search(
        &self,
        query: &str,
        top_k: usize,
    ) -> ClientResult<Vec<LibrarySearchResult>> {
        match self {
            AkhClient::Local(engine) => {
                use crate::vsa::encode::encode_label;

                let query_vec = encode_label(engine.ops(), query).map_err(|e| {
                    ClientError::Engine(e.into())
                })?;
                let results = engine
                    .item_memory()
                    .search(&query_vec, top_k)
                    .map_err(|e| ClientError::Engine(e.into()))?;

                Ok(results
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
                    .collect())
            }
            AkhClient::Remote { .. } => {
                let req = LibrarySearchRequest {
                    query: query.to_string(),
                    top_k,
                };
                self.post_json("/library/search", &req)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Daemon (server-side only — Local returns an error)
    // -----------------------------------------------------------------------

    /// Start a workspace daemon. Requires akhomed.
    pub fn start_daemon(
        &self,
        config: Option<serde_json::Value>,
    ) -> ClientResult<DaemonStatus> {
        match self {
            AkhClient::Local(_) => Err(ClientError::Request {
                message: "daemon requires akhomed — start the server first".into(),
            }),
            AkhClient::Remote { .. } => {
                let body = config.unwrap_or(serde_json::json!({}));
                self.post_json("/daemon/start", &body)
            }
        }
    }

    /// Stop a workspace daemon. Requires akhomed.
    pub fn stop_daemon(&self) -> ClientResult<()> {
        match self {
            AkhClient::Local(_) => Err(ClientError::Request {
                message: "daemon requires akhomed — start the server first".into(),
            }),
            AkhClient::Remote { .. } => {
                let _: serde_json::Value =
                    self.post_json("/daemon/stop", &serde_json::json!({}))?;
                Ok(())
            }
        }
    }

    /// Get daemon status. Requires akhomed.
    pub fn daemon_status(&self) -> ClientResult<DaemonStatus> {
        match self {
            AkhClient::Local(_) => Err(ClientError::Request {
                message: "daemon requires akhomed — start the server first".into(),
            }),
            AkhClient::Remote { .. } => self.get_json("/daemon"),
        }
    }

    // -----------------------------------------------------------------------
    // Triggers
    // -----------------------------------------------------------------------

    /// List all registered triggers.
    pub fn list_triggers(&self) -> ClientResult<Vec<Trigger>> {
        match self {
            AkhClient::Local(engine) => {
                let store = TriggerStore::new(engine);
                Ok(store.list())
            }
            AkhClient::Remote { .. } => self.get_json("/triggers"),
        }
    }

    /// Add a trigger.
    pub fn add_trigger(&self, trigger: &Trigger) -> ClientResult<Trigger> {
        match self {
            AkhClient::Local(engine) => {
                let store = TriggerStore::new(engine);
                store.add(trigger.clone()).map_err(|e| {
                    ClientError::Engine(crate::error::AkhError::Agent(e))
                })
            }
            AkhClient::Remote { .. } => self.post_json("/triggers", trigger),
        }
    }

    /// Remove a trigger by ID.
    pub fn remove_trigger(&self, id: &str) -> ClientResult<()> {
        match self {
            AkhClient::Local(engine) => {
                let store = TriggerStore::new(engine);
                store.remove(id).map_err(|e| {
                    ClientError::Engine(crate::error::AkhError::Agent(e))
                })
            }
            AkhClient::Remote { .. } => {
                let _: serde_json::Value =
                    self.delete_json(&format!("/triggers/{id}"))?;
                Ok(())
            }
        }
    }
}
