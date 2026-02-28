//! akhomed — the akh-medu daemon.
//!
//! Single authority over engine instances; the `akh` CLI connects here.
//! Hosts N engine workspaces with REST and WebSocket APIs.
//!
//! Build and run: `cargo run --features server --bin akhomed`
//!
//! See `docs/ai/decisions/023-client-only-mode.md` for the full endpoint inventory.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use akh_medu::agent::trigger::{Trigger, TriggerStore};
use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::client::DaemonStatus;
use akh_medu::config::AkhomedConfig;
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::grammar::concrete::ParseContext;
use akh_medu::grammar::entity_resolution::{EquivalenceStats, LearnedEquivalence};
use akh_medu::grammar::preprocess::{PreProcessRequest, PreProcessResponse, preprocess_batch};
use akh_medu::graph::Triple;
use akh_medu::graph::traverse::TraversalConfig;
use akh_medu::infer::InferenceQuery;
use akh_medu::library::{
    ContentFormat, DocumentRecord, IngestConfig, LibraryAddRequest, LibraryAddResponse,
    LibraryCatalog, LibrarySearchRequest, LibrarySearchResult,
};
use akh_medu::message::AkhMessage;
use akh_medu::paths::AkhPaths;
use akh_medu::seeds::SeedRegistry;
use akh_medu::symbol::{SymbolId, SymbolKind};
use akh_medu::vsa::Dimension;
use akh_medu::workspace::WorkspaceManager;

// ── Server state ──────────────────────────────────────────────────────────

struct ServerState {
    paths: AkhPaths,
    config: RwLock<AkhomedConfig>,
    workspaces: RwLock<HashMap<String, Arc<Engine>>>,
    daemons: RwLock<HashMap<String, WorkspaceDaemon>>,
    /// Broadcast channel for audit entries — WS clients subscribe to this.
    audit_broadcast: tokio::sync::broadcast::Sender<akh_medu::audit::AuditEntry>,
}

/// Background daemon state for a workspace.
struct WorkspaceDaemon {
    handle: tokio::task::JoinHandle<()>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    status: Arc<tokio::sync::Mutex<DaemonStatus>>,
}

impl ServerState {
    fn new(paths: AkhPaths, config: AkhomedConfig) -> Self {
        let (audit_tx, _) = tokio::sync::broadcast::channel(256);
        Self {
            paths,
            config: RwLock::new(config),
            workspaces: RwLock::new(HashMap::new()),
            daemons: RwLock::new(HashMap::new()),
            audit_broadcast: audit_tx,
        }
    }

    /// Get or lazily open an engine for the given workspace.
    async fn get_engine(&self, name: &str) -> Result<Arc<Engine>, (StatusCode, String)> {
        // Check if already loaded.
        {
            let map = self.workspaces.read().await;
            if let Some(engine) = map.get(name) {
                return Ok(Arc::clone(engine));
            }
        }

        // Try to load it.
        let ws_paths = self.paths.workspace(name);
        if !ws_paths.root.exists() {
            return Err((
                StatusCode::NOT_FOUND,
                format!("workspace \"{name}\" not found"),
            ));
        }

        let config = EngineConfig {
            dimension: Dimension::DEFAULT,
            data_dir: Some(ws_paths.kg_dir.clone()),
            ..Default::default()
        };

        let engine = Engine::new(config).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to open workspace \"{name}\": {e}"),
            )
        })?;

        // Wire audit broadcast sender so WS clients receive live entries.
        if let Some(ledger) = engine.audit_ledger() {
            ledger.set_broadcast(self.audit_broadcast.clone());
        }

        let engine = Arc::new(engine);
        let mut map = self.workspaces.write().await;
        map.insert(name.to_string(), Arc::clone(&engine));
        Ok(engine)
    }
}

/// Create or resume an Agent for the given engine.
///
/// Tries to resume a persisted session; falls back to a fresh agent.
fn create_agent(engine: &Arc<Engine>) -> Result<Agent, String> {
    let config = AgentConfig::default();
    if Agent::has_persisted_session(engine) {
        Agent::resume(Arc::clone(engine), config)
    } else {
        Agent::new(Arc::clone(engine), config)
    }
    .map_err(|e| format!("failed to create agent: {e}"))
}

// ── Response types ────────────────────────────────────────────────────────

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    workspaces_loaded: usize,
}

#[derive(Serialize)]
struct WorkspaceInfo {
    name: String,
    symbols: usize,
    triples: usize,
}

#[derive(Serialize)]
struct WorkspaceListResponse {
    workspaces: Vec<String>,
}

#[derive(Serialize)]
struct WorkspaceCreatedResponse {
    name: String,
    created: bool,
}

#[derive(Serialize)]
struct SeedAppliedResponse {
    pack: String,
    triples_applied: usize,
    already_applied: bool,
}

#[derive(Deserialize)]
struct WsInput {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    text: String,
}

// ── Handlers ──────────────────────────────────────────────────────────────

async fn health(State(state): State<Arc<ServerState>>) -> Json<HealthResponse> {
    let map = state.workspaces.read().await;
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        workspaces_loaded: map.len(),
    })
}

// ── Config handlers ──────────────────────────────────────────────────────

async fn get_config_handler(
    State(state): State<Arc<ServerState>>,
) -> Json<AkhomedConfig> {
    let config = state.config.read().await;
    Json(config.clone())
}

async fn put_config_handler(
    State(state): State<Arc<ServerState>>,
    Json(new_config): Json<AkhomedConfig>,
) -> Result<Json<AkhomedConfig>, (StatusCode, String)> {
    // Persist to disk.
    let config_path = state.paths.global_config_file();
    new_config
        .save(&config_path)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;

    // Update in-memory state.
    let mut config = state.config.write().await;
    *config = new_config.clone();

    tracing::info!(path = %config_path.display(), "config updated");
    Ok(Json(new_config))
}

// ── Workspace handlers ──────────────────────────────────────────────────

async fn list_workspaces(State(state): State<Arc<ServerState>>) -> Json<WorkspaceListResponse> {
    let names = state.paths.list_workspaces();
    Json(WorkspaceListResponse { workspaces: names })
}

#[derive(Deserialize)]
struct CreateWorkspaceReq {
    role: Option<String>,
}

async fn create_workspace(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
    body: Option<Json<CreateWorkspaceReq>>,
) -> Result<Json<WorkspaceCreatedResponse>, (StatusCode, String)> {
    let manager = WorkspaceManager::new(state.paths.clone());
    let config = akh_medu::workspace::WorkspaceConfig {
        name: name.clone(),
        ..Default::default()
    };
    match manager.create(config) {
        Ok(_) => {
            // Assign role if provided in the request body.
            if let Some(Json(req)) = body
                && let Some(ref role) = req.role {
                    let engine = state.get_engine(&name).await?;
                    engine.assign_role(role).map_err(|e| {
                        (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))
                    })?;
                    engine.persist().map_err(|e| {
                        (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))
                    })?;
                }
            Ok(Json(WorkspaceCreatedResponse {
                name,
                created: true,
            }))
        }
        Err(e) => Err((StatusCode::BAD_REQUEST, format!("{e}"))),
    }
}

#[derive(Deserialize)]
struct AssignRoleReq {
    role: String,
}

async fn assign_role_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<AssignRoleReq>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    engine
        .assign_role(&req.role)
        .map_err(|e| (StatusCode::CONFLICT, format!("{e}")))?;
    engine
        .persist()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;
    Ok(Json(
        serde_json::json!({ "role": req.role, "assigned": true }),
    ))
}

async fn delete_workspace(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Remove from loaded engines.
    {
        let mut map = state.workspaces.write().await;
        map.remove(&name);
    }

    let manager = WorkspaceManager::new(state.paths.clone());
    match manager.delete(&name) {
        Ok(_) => Ok(Json(serde_json::json!({ "deleted": name }))),
        Err(e) => Err((StatusCode::BAD_REQUEST, format!("{e}"))),
    }
}

async fn workspace_status(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> Result<Json<WorkspaceInfo>, (StatusCode, String)> {
    let engine = state.get_engine(&name).await?;
    Ok(Json(WorkspaceInfo {
        name,
        symbols: engine.all_symbols().len(),
        triples: engine.all_triples().len(),
    }))
}

async fn apply_seed(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, pack_name)): Path<(String, String)>,
) -> Result<Json<SeedAppliedResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let registry = SeedRegistry::bundled();
    match registry.apply(&pack_name, &engine) {
        Ok(report) => Ok(Json(SeedAppliedResponse {
            pack: pack_name,
            triples_applied: report.triples_applied,
            already_applied: report.already_applied,
        })),
        Err(e) => Err((StatusCode::BAD_REQUEST, format!("{e}"))),
    }
}

async fn workspace_preprocess(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
    Json(request): Json<PreProcessRequest>,
) -> Result<Json<PreProcessResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&name).await?;
    let ctx = ParseContext::with_engine(engine.registry(), engine.ops(), engine.item_memory());

    let start = Instant::now();
    let results = preprocess_batch(&request.chunks, &ctx);
    let elapsed = start.elapsed().as_millis() as u64;

    Ok(Json(PreProcessResponse {
        results,
        processing_time_ms: elapsed,
    }))
}

async fn workspace_equivalences(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> Result<Json<Vec<LearnedEquivalence>>, (StatusCode, String)> {
    let engine = state.get_engine(&name).await?;
    Ok(Json(engine.export_equivalences()))
}

async fn workspace_equivalences_stats(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> Result<Json<EquivalenceStats>, (StatusCode, String)> {
    let engine = state.get_engine(&name).await?;
    Ok(Json(engine.equivalence_stats()))
}

// ── Symbol handlers ──────────────────────────────────────────────────────

async fn list_symbols(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> Result<Json<Vec<akh_medu::symbol::SymbolMeta>>, (StatusCode, String)> {
    let engine = state.get_engine(&name).await?;
    Ok(Json(engine.all_symbols()))
}

async fn resolve_symbol(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, sym)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    match engine.resolve_symbol(&sym) {
        Ok(id) => {
            let label = engine.resolve_label(id);
            Ok(Json(serde_json::json!({ "id": id.get(), "label": label })))
        }
        Err(e) => Err((StatusCode::NOT_FOUND, format!("{e}"))),
    }
}

#[derive(Deserialize)]
struct CreateSymbolReq {
    kind: String,
    label: String,
}

async fn create_symbol(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<CreateSymbolReq>,
) -> Result<Json<akh_medu::symbol::SymbolMeta>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let kind = match req.kind.as_str() {
        "entity" => SymbolKind::Entity,
        "relation" => SymbolKind::Relation,
        other => return Err((StatusCode::BAD_REQUEST, format!("unknown kind: {other}"))),
    };
    engine
        .create_symbol(kind, &req.label)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

// ── Triple handlers ─────────────────────────────────────────────────────

async fn list_triples(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> Result<Json<Vec<Triple>>, (StatusCode, String)> {
    let engine = state.get_engine(&name).await?;
    Ok(Json(engine.all_triples()))
}

async fn triples_from(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, sym_id)): Path<(String, u64)>,
) -> Result<Json<Vec<Triple>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let id = SymbolId::new(sym_id)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid symbol id".to_string()))?;
    Ok(Json(engine.triples_from(id)))
}

async fn triples_to(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, sym_id)): Path<(String, u64)>,
) -> Result<Json<Vec<Triple>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let id = SymbolId::new(sym_id)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid symbol id".to_string()))?;
    Ok(Json(engine.triples_to(id)))
}

#[derive(Deserialize)]
struct AddTripleReq {
    subject: u64,
    predicate: u64,
    object: u64,
    #[serde(default = "default_confidence")]
    confidence: f32,
}

fn default_confidence() -> f32 {
    1.0
}

async fn add_triple(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<AddTripleReq>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let s = SymbolId::new(req.subject)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid subject id".to_string()))?;
    let p = SymbolId::new(req.predicate)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid predicate id".to_string()))?;
    let o = SymbolId::new(req.object)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid object id".to_string()))?;
    let mut triple = Triple::new(s, p, o);
    triple.confidence = req.confidence;
    engine
        .add_triple(&triple)
        .map(|_| Json(serde_json::json!({"ok": true})))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

#[derive(Deserialize)]
struct IngestReq {
    triples: Vec<(String, String, String, f32)>,
}

async fn ingest_triples(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<IngestReq>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    match engine.ingest_label_triples(&req.triples) {
        Ok((syms, trips)) => {
            let _ = engine.persist();
            Ok(Json(serde_json::json!({
                "symbols_created": syms,
                "triples_ingested": trips,
            })))
        }
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))),
    }
}

// ── Query & reasoning handlers ──────────────────────────────────────────

#[derive(Deserialize)]
struct SparqlReq {
    query: String,
}

async fn sparql_query(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<SparqlReq>,
) -> Result<Json<Vec<Vec<(String, String)>>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    engine
        .sparql_query(&req.query)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

async fn infer_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(query): Json<InferenceQuery>,
) -> Result<Json<akh_medu::infer::InferenceResult>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    engine
        .infer(&query)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

#[derive(Deserialize)]
struct ReasonReq {
    expr: String,
}

async fn reason_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<ReasonReq>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    match engine.simplify_expression(&req.expr) {
        Ok(result) => Ok(Json(serde_json::json!({ "result": result }))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))),
    }
}

#[derive(Deserialize)]
struct SearchReq {
    symbol: u64,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_top_k() -> usize {
    5
}

async fn search_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<SearchReq>,
) -> Result<Json<Vec<akh_medu::vsa::item_memory::SearchResult>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let id = SymbolId::new(req.symbol)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid symbol id".to_string()))?;
    engine
        .search_similar_to(id, req.top_k)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

#[derive(Deserialize)]
struct AnalogyReq {
    a: u64,
    b: u64,
    c: u64,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

async fn analogy_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<AnalogyReq>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let a = SymbolId::new(req.a)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid a".to_string()))?;
    let b = SymbolId::new(req.b)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid b".to_string()))?;
    let c = SymbolId::new(req.c)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid c".to_string()))?;
    match engine.infer_analogy(a, b, c, req.top_k) {
        Ok(results) => Ok(Json(
            results
                .into_iter()
                .map(|(sym, score)| serde_json::json!({"symbol": sym.get(), "score": score}))
                .collect(),
        )),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))),
    }
}

#[derive(Deserialize)]
struct FillerReq {
    subject: u64,
    predicate: u64,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

async fn filler_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<FillerReq>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let s = SymbolId::new(req.subject)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid subject".to_string()))?;
    let p = SymbolId::new(req.predicate)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid predicate".to_string()))?;
    match engine.recover_filler(s, p, req.top_k) {
        Ok(results) => Ok(Json(
            results
                .into_iter()
                .map(|(sym, score)| serde_json::json!({"symbol": sym.get(), "score": score}))
                .collect(),
        )),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))),
    }
}

// ── Traversal & analytics handlers ──────────────────────────────────────

#[derive(Deserialize)]
struct TraverseReq {
    seeds: Vec<u64>,
    #[serde(default = "default_max_depth")]
    max_depth: usize,
    #[serde(default)]
    predicate_filter: std::collections::HashSet<u64>,
    #[serde(default)]
    min_confidence: f32,
    #[serde(default = "default_max_results")]
    max_results: usize,
}

fn default_max_depth() -> usize {
    3
}
fn default_max_results() -> usize {
    1000
}

async fn traverse_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<TraverseReq>,
) -> Result<Json<akh_medu::graph::traverse::TraversalResult>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let seeds: Vec<SymbolId> = req
        .seeds
        .iter()
        .filter_map(|&id| SymbolId::new(id))
        .collect();
    let pred_filter: std::collections::HashSet<SymbolId> = req
        .predicate_filter
        .iter()
        .filter_map(|&id| SymbolId::new(id))
        .collect();
    let config = TraversalConfig {
        max_depth: req.max_depth,
        predicate_filter: pred_filter,
        min_confidence: req.min_confidence,
        max_results: req.max_results,
    };
    engine
        .traverse(&seeds, config)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

async fn degree_centrality_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<akh_medu::graph::analytics::DegreeCentrality>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    Ok(Json(engine.degree_centrality()))
}

async fn pagerank_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<Vec<akh_medu::graph::analytics::PageRankScore>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let damping: f64 = params
        .get("damping")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.85);
    let iterations: usize = params
        .get("iterations")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    engine
        .pagerank(damping, iterations)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

async fn components_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<akh_medu::graph::analytics::ConnectedComponent>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    engine
        .strongly_connected_components()
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

#[derive(Deserialize)]
struct ShortestPathReq {
    from: u64,
    to: u64,
}

async fn shortest_path_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<ShortestPathReq>,
) -> Result<Json<Option<Vec<u64>>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let from = SymbolId::new(req.from)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid from".to_string()))?;
    let to = SymbolId::new(req.to)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid to".to_string()))?;
    match engine.shortest_path(from, to) {
        Ok(path) => Ok(Json(path.map(|p| p.into_iter().map(|s| s.get()).collect()))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))),
    }
}

// ── Export handlers ─────────────────────────────────────────────────────

async fn export_symbols(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<akh_medu::export::SymbolExport>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    Ok(Json(engine.export_symbol_table()))
}

async fn export_triples(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<akh_medu::export::TripleExport>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    Ok(Json(engine.export_triples()))
}

async fn export_provenance(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, sym)): Path<(String, String)>,
) -> Result<Json<Vec<akh_medu::export::ProvenanceExport>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let id = engine
        .resolve_symbol(&sym)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("{e}")))?;
    engine
        .export_provenance_chain(id)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

// ── Skills handlers ─────────────────────────────────────────────────────

async fn list_skills_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<akh_medu::skills::SkillInfo>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    Ok(Json(engine.list_skills()))
}

async fn load_skill_handler(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, skill_name)): Path<(String, String)>,
) -> Result<Json<akh_medu::skills::SkillActivation>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    engine
        .load_skill(&skill_name)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

async fn unload_skill_handler(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, skill_name)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    engine
        .unload_skill(&skill_name)
        .map(|_| Json(serde_json::json!({"ok": true})))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

async fn skill_info_handler(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, skill_name)): Path<(String, String)>,
) -> Result<Json<akh_medu::skills::SkillInfo>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    engine
        .skill_info(&skill_name)
        .map(Json)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("{e}")))
}

async fn install_skill_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(payload): Json<akh_medu::skills::SkillInstallPayload>,
) -> Result<Json<akh_medu::skills::SkillActivation>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    engine
        .install_skill(&payload)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

// ── Library handlers ─────────────────────────────────────────────────────

async fn library_list_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<DocumentRecord>>, (StatusCode, String)> {
    let _engine = state.get_engine(&ws_name).await?;
    let library_dir = state.paths.library_dir();
    let catalog = LibraryCatalog::open(&library_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;
    Ok(Json(catalog.list().to_vec()))
}

async fn library_add_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<LibraryAddRequest>,
) -> Result<Json<LibraryAddResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let library_dir = state.paths.library_dir();
    let mut catalog = LibraryCatalog::open(&library_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;

    let fmt = req.format.as_deref().and_then(|f| match f {
        "html" => Some(ContentFormat::Html),
        "pdf" => Some(ContentFormat::Pdf),
        "epub" => Some(ContentFormat::Epub),
        "text" | "txt" => Some(ContentFormat::PlainText),
        _ => None,
    });

    let ingest_config = IngestConfig {
        title: req.title,
        tags: req.tags,
        format: fmt,
        ..Default::default()
    };

    // Initialise NLU pipeline from this workspace's persisted ranker state.
    let data_dir = engine.config().data_dir.as_deref();
    let mut nlu_pipeline = engine
        .store()
        .get_meta(b"nlu_ranker_state")
        .ok()
        .flatten()
        .and_then(|bytes| akh_medu::nlu::parse_ranker::ParseRanker::from_bytes(&bytes))
        .map(|ranker| akh_medu::nlu::NluPipeline::with_ranker_and_models(ranker, data_dir))
        .unwrap_or_else(|| akh_medu::nlu::NluPipeline::new_with_models(data_dir));

    let result = if req.source.starts_with("http://") || req.source.starts_with("https://") {
        akh_medu::library::ingest_url(
            &engine,
            &mut catalog,
            &req.source,
            ingest_config,
            Some(&mut nlu_pipeline),
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    } else {
        let path = std::path::PathBuf::from(&req.source);
        akh_medu::library::ingest_file(
            &engine,
            &mut catalog,
            &path,
            ingest_config,
            Some(&mut nlu_pipeline),
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    };

    // Persist NLU ranker state (parse successes train the ranker).
    let ranker_bytes = nlu_pipeline.ranker().to_bytes();
    let _ = engine.store().put_meta(b"nlu_ranker_state", &ranker_bytes);

    // Persist engine state so ingested symbols survive restarts.
    engine
        .persist()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("persist: {e}")))?;

    Ok(Json(LibraryAddResponse {
        id: result.record.id,
        title: result.record.title,
        format: result.record.format.to_string(),
        chunk_count: result.chunk_count,
        triple_count: result.triple_count,
        concept_count: result.concept_count,
    }))
}

async fn library_search_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<LibrarySearchRequest>,
) -> Result<Json<Vec<LibrarySearchResult>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;

    let query_vec = akh_medu::vsa::encode::encode_label(engine.ops(), &req.query)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("encode: {e}")))?;

    let results = engine
        .item_memory()
        .search(&query_vec, req.top_k)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("search: {e}")))?;

    let out: Vec<LibrarySearchResult> = results
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
        .collect();

    Ok(Json(out))
}

async fn library_info_handler(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, doc_id)): Path<(String, String)>,
) -> Result<Json<DocumentRecord>, (StatusCode, String)> {
    let _engine = state.get_engine(&ws_name).await?;
    let library_dir = state.paths.library_dir();
    let catalog = LibraryCatalog::open(&library_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;
    let doc = catalog
        .get(&doc_id)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("document not found: \"{doc_id}\"")))?;
    Ok(Json(doc.clone()))
}

async fn library_remove_handler(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, doc_id)): Path<(String, String)>,
) -> Result<Json<DocumentRecord>, (StatusCode, String)> {
    let _engine = state.get_engine(&ws_name).await?;
    let library_dir = state.paths.library_dir();
    let mut catalog = LibraryCatalog::open(&library_dir)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;
    let removed = catalog
        .remove(&doc_id)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("{e}")))?;
    Ok(Json(removed))
}

// ── Daemon handlers ─────────────────────────────────────────────────────

#[derive(Deserialize)]
struct StartDaemonReq {
    #[serde(default = "default_daemon_max_cycles")]
    max_cycles: usize,
}

fn default_daemon_max_cycles() -> usize {
    0
}

async fn start_daemon_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    body: Option<Json<StartDaemonReq>>,
) -> Result<Json<DaemonStatus>, (StatusCode, String)> {
    // Check if already running.
    {
        let daemons = state.daemons.read().await;
        if let Some(d) = daemons.get(&ws_name)
            && !d.handle.is_finished() {
                let st = d.status.lock().await;
                return Ok(Json(st.clone()));
            }
    }

    let engine = state.get_engine(&ws_name).await?;
    let max_cycles = body.map(|b| b.max_cycles).unwrap_or(0);

    // Build DaemonConfig from global config with request-level override.
    let mut daemon_config = state.config.read().await.daemon.to_daemon_config();
    if max_cycles > 0 {
        daemon_config.max_cycles = max_cycles;
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let trigger_count = TriggerStore::new(&engine).list().len();

    let status = Arc::new(tokio::sync::Mutex::new(DaemonStatus {
        running: true,
        total_cycles: 0,
        started_at: now,
        trigger_count,
    }));

    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    let daemon_status = Arc::clone(&status);
    let daemon_engine = Arc::clone(&engine);
    let daemon_ws_name = ws_name.clone();

    let handle = tokio::task::spawn(async move {
        run_daemon_task(
            daemon_engine,
            daemon_status,
            shutdown_rx,
            daemon_config,
            daemon_ws_name,
        )
        .await;
    });

    let daemon = WorkspaceDaemon {
        handle,
        shutdown_tx,
        status: Arc::clone(&status),
    };

    let st = status.lock().await.clone();
    state.daemons.write().await.insert(ws_name, daemon);
    Ok(Json(st))
}

async fn stop_daemon_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let mut daemons = state.daemons.write().await;
    if let Some(d) = daemons.remove(&ws_name) {
        let _ = d.shutdown_tx.send(true);
        // Give the task a moment to finish.
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), d.handle).await;
        Ok(Json(serde_json::json!({"stopped": true})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("no daemon running for workspace \"{ws_name}\""),
        ))
    }
}

async fn daemon_status_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<DaemonStatus>, (StatusCode, String)> {
    let daemons = state.daemons.read().await;
    if let Some(d) = daemons.get(&ws_name) {
        let st = d.status.lock().await;
        Ok(Json(st.clone()))
    } else {
        Ok(Json(DaemonStatus {
            running: false,
            total_cycles: 0,
            started_at: 0,
            trigger_count: 0,
        }))
    }
}

/// Background daemon task: delegates to [`AgentDaemon`] for the full set of
/// background learning tasks (goal generation, OODA cycles, equivalence
/// learning, reflection, consolidation, schema discovery, rule inference,
/// gap analysis, continuous learning, sleep cycles, and trigger evaluation).
async fn run_daemon_task(
    engine: Arc<Engine>,
    status: Arc<tokio::sync::Mutex<DaemonStatus>>,
    shutdown_rx: tokio::sync::watch::Receiver<bool>,
    daemon_config: akh_medu::agent::DaemonConfig,
    ws_name: String,
) {
    use akh_medu::agent::AgentDaemon;

    // Create agent (heavy sync work).
    let agent_config = AgentConfig::default();
    let agent_result = tokio::task::spawn_blocking({
        let engine = Arc::clone(&engine);
        move || {
            if Agent::has_persisted_session(&engine) {
                Agent::resume(engine, agent_config)
            } else {
                Agent::new(engine, agent_config)
            }
        }
    })
    .await;

    let agent = match agent_result {
        Ok(Ok(a)) => a,
        Ok(Err(e)) => {
            tracing::error!(error = %e, ws = %ws_name, "daemon: failed to create agent");
            status.lock().await.running = false;
            return;
        }
        Err(e) => {
            tracing::error!(error = %e, ws = %ws_name, "daemon: agent task panicked");
            status.lock().await.running = false;
            return;
        }
    };

    let mut daemon = AgentDaemon::new(agent, daemon_config)
        .with_shutdown(shutdown_rx)
        .with_status(Arc::clone(&status));

    if let Err(e) = daemon.run().await {
        tracing::error!(error = %e, ws = %ws_name, "daemon task failed");
    }

    // AgentDaemon::run() already sets status.running = false and persists,
    // but belt-and-suspenders in case of early return.
    status.lock().await.running = false;
    tracing::info!(ws = %ws_name, "daemon stopped");
}

// ── Trigger handlers ─────────────────────────────────────────────────────

async fn list_triggers_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<Trigger>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let store = TriggerStore::new(&engine);
    Ok(Json(store.list()))
}

async fn add_trigger_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(trigger): Json<Trigger>,
) -> Result<Json<Trigger>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let store = TriggerStore::new(&engine);
    store
        .add(trigger)
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

async fn remove_trigger_handler(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, trigger_id)): Path<(String, String)>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let store = TriggerStore::new(&engine);
    store
        .remove(&trigger_id)
        .map(|_| Json(serde_json::json!({"removed": trigger_id})))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))
}

// ── Engine info handler ─────────────────────────────────────────────────

async fn engine_info(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::engine::EngineInfo>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    Ok(Json(engine.info()))
}

// ── WebSocket handler ─────────────────────────────────────────────────────

async fn ws_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    let engine_result = state.get_engine(&ws_name).await;
    let audit_rx = state.audit_broadcast.subscribe();
    ws.on_upgrade(move |socket| async move {
        match engine_result {
            Ok(engine) => handle_ws_session(socket, engine, ws_name, audit_rx).await,
            Err((_, msg)) => {
                let err = AkhMessage::error("ws", msg);
                let _ = send_message(&err, &mut None::<&mut WebSocket>).await;
            }
        }
    })
}

async fn handle_ws_session(
    mut socket: WebSocket,
    engine: Arc<Engine>,
    ws_name: String,
    mut audit_rx: tokio::sync::broadcast::Receiver<akh_medu::audit::AuditEntry>,
) {
    // Create an agent for this session.
    let agent_config = AgentConfig::default();
    let mut agent = match Agent::new(Arc::clone(&engine), agent_config) {
        Ok(a) => a,
        Err(e) => {
            let err = AkhMessage::error("init", format!("failed to create agent: {e}"));
            let _ = send_akh_message(&mut socket, &err).await;
            return;
        }
    };

    // Initialise the unified ChatProcessor for this session.
    let data_dir = engine.config().data_dir.as_deref();
    let nlu_pipeline = engine
        .store()
        .get_meta(b"nlu_ranker_state")
        .ok()
        .flatten()
        .and_then(|bytes| akh_medu::nlu::parse_ranker::ParseRanker::from_bytes(&bytes))
        .map(|ranker| akh_medu::nlu::NluPipeline::with_ranker_and_models(ranker, data_dir))
        .unwrap_or_else(|| akh_medu::nlu::NluPipeline::new_with_models(data_dir));
    let mut chat_processor = akh_medu::chat::ChatProcessor::new(&engine, nlu_pipeline);

    let welcome = AkhMessage::system(format!(
        "Connected to workspace \"{ws_name}\". {} symbols, {} triples.",
        engine.all_symbols().len(),
        engine.all_triples().len(),
    ));
    if send_akh_message(&mut socket, &welcome).await.is_err() {
        return;
    }

    loop {
        tokio::select! {
            ws_msg = socket.recv() => {
                match ws_msg {
                    Some(Ok(Message::Text(text))) => {
                        let input: WsInput = match serde_json::from_str(&text) {
                            Ok(i) => i,
                            Err(e) => {
                                let err = AkhMessage::error("parse", format!("invalid JSON: {e}"));
                                if send_akh_message(&mut socket, &err).await.is_err() {
                                    break;
                                }
                                continue;
                            }
                        };

                        let responses = match input.msg_type.as_str() {
                            "input" => {
                                chat_processor.process_input(&input.text, &mut agent, &engine)
                            }
                            "command" => {
                                process_ws_command(&input.text, &agent, &engine)
                            }
                            _ => {
                                vec![AkhMessage::error(
                                    "protocol",
                                    format!("unknown message type: \"{}\"", input.msg_type),
                                )]
                            }
                        };
                        for msg in &responses {
                            if send_akh_message(&mut socket, msg).await.is_err() {
                                break;
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => break,
                    _ => {}
                }
            }
            audit_entry = audit_rx.recv() => {
                if let Ok(entry) = audit_entry {
                    let msg = entry.to_message();
                    if send_akh_message(&mut socket, &msg).await.is_err() {
                        break;
                    }
                }
            }
        }
    }

    // Persist ChatProcessor NLU state and agent session on disconnect.
    chat_processor.persist_nlu_state(&engine);
    let _ = agent.persist_session();
}

/// Handle WS protocol commands (status, goals) — these are WS-specific
/// and not part of the ChatProcessor's concern.
fn process_ws_command(cmd: &str, agent: &Agent, engine: &Engine) -> Vec<AkhMessage> {
    let mut msgs = Vec::new();
    let trimmed = cmd.trim();
    match trimmed {
        "status" => {
            msgs.push(AkhMessage::system(format!(
                "Cycles: {}, WM: {}, Symbols: {}, Triples: {}",
                agent.cycle_count(),
                agent.working_memory().len(),
                engine.all_symbols().len(),
                engine.all_triples().len(),
            )));
        }
        "goals" => {
            let goals = agent.goals();
            if goals.is_empty() {
                msgs.push(AkhMessage::system("No active goals.".to_string()));
            } else {
                for g in goals {
                    msgs.push(AkhMessage::goal_progress(
                        &g.description,
                        format!("{}", g.status),
                    ));
                }
            }
        }
        _ => {
            msgs.push(AkhMessage::system(format!("Unknown command: \"{trimmed}\"")));
        }
    }
    msgs
}

// ── Audit handlers ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct AuditListQuery {
    #[serde(default)]
    offset: u64,
    #[serde(default = "default_audit_limit")]
    limit: usize,
    kind: Option<u8>,
}

fn default_audit_limit() -> usize {
    50
}

async fn audit_list_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    axum::extract::Query(query): axum::extract::Query<AuditListQuery>,
) -> Result<Json<akh_medu::audit::AuditPage>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let ledger = engine
        .audit_ledger()
        .ok_or_else(|| (StatusCode::NOT_FOUND, "audit ledger not available".into()))?;

    let page = if let Some(kind_tag) = query.kind {
        ledger.list_by_kind(kind_tag, query.offset, query.limit)
    } else {
        ledger.list_page(query.offset, query.limit)
    }
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;

    Ok(Json(page))
}

async fn audit_get_handler(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, id)): Path<(String, u64)>,
) -> Result<Json<akh_medu::audit::AuditEntry>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let ledger = engine
        .audit_ledger()
        .ok_or_else(|| (StatusCode::NOT_FOUND, "audit ledger not available".into()))?;

    let audit_id = akh_medu::audit::AuditId::new(id)
        .ok_or_else(|| (StatusCode::BAD_REQUEST, "invalid audit id".into()))?;

    let entry = ledger
        .get(audit_id)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("{e}")))?;

    Ok(Json(entry))
}

// ── Awaken status handler ─────────────────────────────────────────────

#[derive(Serialize)]
struct AwakenStatusResponse {
    awakened: bool,
    psyche: Option<akh_medu::compartment::psyche::Psyche>,
    active_goals: usize,
    total_goals: usize,
    cycle_count: u64,
}

async fn awaken_status_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<AwakenStatusResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let psyche = engine.compartments().and_then(|m| m.psyche());

    let agent = Agent::new(Arc::clone(&engine), AgentConfig::default())
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;

    let active_goals = agent
        .goals()
        .iter()
        .filter(|g| matches!(g.status, akh_medu::agent::GoalStatus::Active))
        .count();

    Ok(Json(AwakenStatusResponse {
        awakened: psyche.as_ref().is_some_and(|p| p.is_awakened()),
        psyche,
        active_goals,
        total_goals: agent.goals().len(),
        cycle_count: agent.cycle_count(),
    }))
}

// ── Goals handler ──────────────────────────────────────────────────────

async fn goals_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;

    // Create a temporary agent to restore persisted goals from the KG.
    let agent_config = AgentConfig::default();
    let agent = Agent::new(Arc::clone(&engine), agent_config)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;

    let goals: Vec<serde_json::Value> = agent
        .goals()
        .iter()
        .map(|g| {
            serde_json::json!({
                "description": g.description,
                "status": format!("{}", g.status),
                "priority": g.priority,
            })
        })
        .collect();

    Ok(Json(goals))
}

async fn send_akh_message(socket: &mut WebSocket, msg: &AkhMessage) -> Result<(), axum::Error> {
    let json = serde_json::to_string(msg).unwrap_or_default();
    socket.send(Message::Text(json.into())).await
}

/// Helper for error sending when we may not have a socket.
async fn send_message(_msg: &AkhMessage, _socket: &mut Option<&mut WebSocket>) -> Result<(), ()> {
    // No-op when socket is None.
    Ok(())
}

// ── Seed list/status handlers ────────────────────────────────────────────

async fn list_seeds_handler(
    State(state): State<Arc<ServerState>>,
) -> Json<Vec<akh_medu::api_types::SeedPackInfo>> {
    let seeds_dir = state.paths.seeds_dir();
    let registry = SeedRegistry::discover(&seeds_dir);
    let packs: Vec<akh_medu::api_types::SeedPackInfo> = registry
        .list()
        .into_iter()
        .map(|p| akh_medu::api_types::SeedPackInfo {
            id: p.id.clone(),
            version: p.version.clone(),
            description: p.description.clone(),
            source: format!("{:?}", p.source),
            triple_count: p.triples.len(),
        })
        .collect();
    Json(packs)
}

async fn seed_status_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::SeedStatusResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let seeds_dir = state.paths.seeds_dir();
    let registry = SeedRegistry::discover(&seeds_dir);
    let seeds: Vec<akh_medu::api_types::SeedStatusEntry> = registry
        .list()
        .into_iter()
        .map(|p| akh_medu::api_types::SeedStatusEntry {
            applied: akh_medu::seeds::is_seed_applied_public(&engine, &p.id),
            id: p.id.clone(),
        })
        .collect();
    Ok(Json(akh_medu::api_types::SeedStatusResponse {
        workspace: ws_name,
        seeds,
    }))
}

// ── Render handler ──────────────────────────────────────────────────────

async fn render_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::RenderRequest>,
) -> Result<Json<akh_medu::api_types::RenderResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;

    let render_config = akh_medu::glyph::RenderConfig {
        color: false, // No ANSI in HTTP responses.
        notation: akh_medu::glyph::NotationConfig {
            use_pua: false,
            show_confidence: true,
            show_provenance: false,
            show_sigils: true,
            compact: false,
        },
        ..Default::default()
    };

    let output = if req.legend {
        akh_medu::glyph::render::render_legend(&render_config)
    } else if let Some(ref entity) = req.entity {
        let sym_id = engine
            .resolve_symbol(entity)
            .map_err(|e| (StatusCode::NOT_FOUND, format!("{e}")))?;
        let result = engine
            .extract_subgraph(&[sym_id], req.depth)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?;
        akh_medu::glyph::render::render_to_terminal(&engine, &result.triples, &render_config)
    } else if req.all {
        let triples = engine.all_triples();
        akh_medu::glyph::render::render_to_terminal(&engine, &triples, &render_config)
    } else {
        String::new()
    };

    Ok(Json(akh_medu::api_types::RenderResponse { output }))
}

// ── Agent run/resume handlers ───────────────────────────────────────────

async fn agent_run_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AgentRunRequest>,
) -> Result<Json<akh_medu::api_types::AgentRunResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<akh_medu::api_types::AgentRunResponse, String> {
        let agent_config = AgentConfig {
            max_cycles: req.max_cycles,
            ..Default::default()
        };
        let mut agent = if req.fresh {
            Agent::new(Arc::clone(&engine), agent_config)
        } else if Agent::has_persisted_session(&engine) {
            Agent::resume(Arc::clone(&engine), agent_config)
        } else {
            Agent::new(Arc::clone(&engine), agent_config)
        }
        .map_err(|e| format!("{e}"))?;

        if req.fresh {
            agent.clear_goals();
        }

        for goal_str in &req.goals {
            agent
                .add_goal(goal_str, 128, "Agent-determined completion")
                .map_err(|e| format!("{e}"))?;
        }

        let _ = agent.run_until_complete();

        let goals: Vec<akh_medu::api_types::GoalSummary> = agent
            .goals()
            .iter()
            .map(|g| akh_medu::api_types::GoalSummary {
                symbol_id: g.symbol_id.get(),
                label: engine.resolve_label(g.symbol_id),
                status: format!("{}", g.status),
                description: g.description.clone(),
            })
            .collect();

        let goals_joined = req.goals.join(", ");
        let summary = agent.synthesize_findings(&goals_joined);
        let _ = agent.persist_session();

        Ok(akh_medu::api_types::AgentRunResponse {
            cycles_completed: agent.cycle_count() as usize,
            goals,
            overview: summary.overview,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("task panicked: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(result))
}

async fn agent_resume_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AgentResumeRequest>,
) -> Result<Json<akh_medu::api_types::AgentResumeResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<akh_medu::api_types::AgentResumeResponse, String> {
        let agent_config = AgentConfig {
            max_cycles: req.max_cycles,
            ..Default::default()
        };
        let mut agent = if Agent::has_persisted_session(&engine) {
            Agent::resume(Arc::clone(&engine), agent_config)
        } else {
            return Err("no persisted session to resume".into());
        }
        .map_err(|e| format!("{e}"))?;

        let _ = agent.run_until_complete();

        let goals: Vec<akh_medu::api_types::GoalSummary> = agent
            .goals()
            .iter()
            .map(|g| akh_medu::api_types::GoalSummary {
                symbol_id: g.symbol_id.get(),
                label: engine.resolve_label(g.symbol_id),
                status: format!("{}", g.status),
                description: g.description.clone(),
            })
            .collect();

        let _ = agent.persist_session();

        Ok(akh_medu::api_types::AgentResumeResponse {
            cycles_completed: agent.cycle_count() as usize,
            goals,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("task panicked: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(result))
}

// ── PIM handlers ────────────────────────────────────────────────────────

async fn pim_inbox_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::PimTaskList>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let ids = agent
            .pim_manager()
            .tasks_by_gtd_state(akh_medu::agent::GtdState::Inbox);
        let tasks = ids
            .iter()
            .map(|&id| akh_medu::api_types::pim_task_item(&engine, &agent, id))
            .collect();
        Ok(akh_medu::api_types::PimTaskList { tasks })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn pim_next_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::PimNextRequest>,
) -> Result<Json<akh_medu::api_types::PimTaskList>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let ctx = req
            .context
            .as_deref()
            .map(|s| akh_medu::agent::PimContext(s.to_string()));
        let nrg = req
            .energy
            .as_deref()
            .and_then(akh_medu::api_types::parse_energy_level);
        let ids = agent
            .pim_manager()
            .available_tasks(ctx.as_ref(), nrg, agent.goals());
        let tasks = ids
            .iter()
            .map(|&id| akh_medu::api_types::pim_task_item(&engine, &agent, id))
            .collect();
        Ok(akh_medu::api_types::PimTaskList { tasks })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn pim_review_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::PimReviewResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let review = akh_medu::agent::pim::gtd_weekly_review(
            agent.pim_manager(),
            agent.goals(),
            agent.projects(),
            now,
        );
        let overdue = review
            .overdue
            .iter()
            .map(|&id| akh_medu::api_types::pim_task_item(&engine, &agent, id))
            .collect();
        let stale_inbox = review
            .stale_inbox
            .iter()
            .map(|&id| akh_medu::api_types::pim_task_item(&engine, &agent, id))
            .collect();
        let stalled_projects = review
            .stalled_projects
            .iter()
            .map(|&id| akh_medu::api_types::pim_task_item(&engine, &agent, id))
            .collect();
        Ok(akh_medu::api_types::PimReviewResponse {
            summary: review.summary,
            overdue,
            stale_inbox,
            stalled_projects,
            adjustment_count: review.adjustments.len(),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn pim_project_handler(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, project_name)): Path<(String, String)>,
) -> Result<Json<akh_medu::api_types::PimProjectResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let project = agent
            .projects()
            .iter()
            .find(|p| p.name == project_name)
            .ok_or_else(|| format!("project \"{project_name}\" not found"))?;
        let goals = project
            .goals
            .iter()
            .map(|&gid| akh_medu::api_types::pim_task_item(&engine, &agent, gid))
            .collect();
        Ok(akh_medu::api_types::PimProjectResponse {
            name: project.name.clone(),
            status: format!("{}", project.status),
            goals,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::NOT_FOUND, e))?;
    Ok(Json(result))
}

async fn pim_add_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::PimAddRequest>,
) -> Result<Json<akh_medu::api_types::PimAddResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let mut agent = create_agent(&engine)?;
        let goal_sym = SymbolId::new(req.goal)
            .ok_or_else(|| "invalid goal symbol id".to_string())?;
        let gtd = akh_medu::api_types::parse_gtd_state(&req.gtd)
            .ok_or_else(|| format!("invalid GTD state: {}", req.gtd))?;
        agent
            .pim_manager_mut()
            .add_task(&engine, goal_sym, gtd, req.urgency, req.importance)
            .map_err(|e| format!("{e}"))?;

        if let Some(ref para_str) = req.para {
            if let Some(para) = akh_medu::agent::ParaCategory::from_label(para_str) {
                let _ = agent.pim_manager_mut().set_para(&engine, goal_sym, para);
            }
        }
        if let Some(ref ctxs) = req.contexts {
            for c in ctxs {
                let _ = agent
                    .pim_manager_mut()
                    .add_context(&engine, goal_sym, akh_medu::agent::PimContext(c.clone()));
            }
        }
        if let Some(ref recur_str) = req.recur {
            if let Ok(r) = akh_medu::agent::Recurrence::parse(recur_str) {
                let _ = agent.pim_manager_mut().set_recurrence(&engine, goal_sym, r);
            }
        }
        if let Some(dl) = req.deadline {
            if let Some(m) = agent.pim_manager_mut().get_metadata_mut(goal_sym.get()) {
                m.deadline = Some(dl);
            }
        }

        let quadrant = akh_medu::agent::EisenhowerQuadrant::classify(req.urgency, req.importance);
        let _ = agent.persist_session();

        Ok(akh_medu::api_types::PimAddResponse {
            goal: req.goal,
            gtd_state: format!("{gtd}"),
            quadrant: quadrant.to_string(),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(result))
}

async fn pim_transition_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::PimTransitionRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let mut agent = create_agent(&engine)?;
        let goal_sym = SymbolId::new(req.goal)
            .ok_or_else(|| "invalid goal symbol id".to_string())?;
        let new_state = akh_medu::api_types::parse_gtd_state(&req.to)
            .ok_or_else(|| format!("invalid GTD state: {}", req.to))?;
        agent
            .pim_manager_mut()
            .transition_gtd(&engine, goal_sym, new_state)
            .map_err(|e| format!("{e}"))?;
        let _ = agent.persist_session();
        Ok(serde_json::json!({"ok": true, "goal": req.goal, "new_state": req.to}))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(result))
}

async fn pim_matrix_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::PimMatrixResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let make_items = |quad: akh_medu::agent::EisenhowerQuadrant| -> Vec<akh_medu::api_types::PimTaskItem> {
            agent
                .pim_manager()
                .tasks_by_quadrant(quad)
                .iter()
                .map(|&id| akh_medu::api_types::pim_task_item(&engine, &agent, id))
                .collect()
        };
        Ok(akh_medu::api_types::PimMatrixResponse {
            do_tasks: make_items(akh_medu::agent::EisenhowerQuadrant::Do),
            schedule_tasks: make_items(akh_medu::agent::EisenhowerQuadrant::Schedule),
            delegate_tasks: make_items(akh_medu::agent::EisenhowerQuadrant::Delegate),
            eliminate_tasks: make_items(akh_medu::agent::EisenhowerQuadrant::Eliminate),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn pim_deps_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::PimDepsResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let order = agent
            .pim_manager()
            .topological_order()
            .map_err(|e| format!("{e}"))?;
        let items = order
            .iter()
            .map(|&id| akh_medu::api_types::pim_task_item(&engine, &agent, id))
            .collect();
        Ok(akh_medu::api_types::PimDepsResponse { order: items })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn pim_overdue_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::PimTaskList>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let ids = agent.pim_manager().overdue_tasks(now);
        let tasks = ids
            .iter()
            .map(|&id| {
                let mut item = akh_medu::api_types::pim_task_item(&engine, &agent, id);
                if let Some(meta) = agent.pim_manager().get_metadata(id.get()) {
                    if let Some(due) = meta.next_due {
                        item.overdue_days = Some(now.saturating_sub(due) / 86400);
                    }
                }
                item
            })
            .collect();
        Ok(akh_medu::api_types::PimTaskList { tasks })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

// ── Causal handlers ─────────────────────────────────────────────────────

async fn causal_schemas_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<akh_medu::api_types::CausalSchemaSummary>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let schemas = agent.causal_manager().list_schemas();
        Ok(schemas
            .iter()
            .map(|s| akh_medu::api_types::CausalSchemaSummary {
                name: s.name.clone(),
                precondition_count: s.preconditions.len(),
                effect_count: s.effects.len(),
                success_rate: s.success_rate as f32,
                execution_count: s.execution_count as usize,
            })
            .collect())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn causal_schema_handler(
    State(state): State<Arc<ServerState>>,
    Path((ws_name, schema_name)): Path<(String, String)>,
) -> Result<Json<akh_medu::api_types::CausalSchemaDetail>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let s = agent
            .causal_manager()
            .get_schema(&schema_name)
            .ok_or_else(|| format!("schema \"{schema_name}\" not found"))?;
        Ok(akh_medu::api_types::CausalSchemaDetail {
            name: s.name.clone(),
            action_id: s.action_id.get(),
            precondition_count: s.preconditions.len(),
            effect_count: s.effects.len(),
            success_rate: s.success_rate as f32,
            execution_count: s.execution_count as usize,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::NOT_FOUND, e))?;
    Ok(Json(result))
}

async fn causal_predict_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::CausalPredictRequest>,
) -> Result<Json<akh_medu::api_types::CausalPredictResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let transition = agent
            .causal_manager()
            .predict_effects(&req.name, &engine)
            .map_err(|e| format!("{e}"))?;
        let label = |id: SymbolId| engine.resolve_label(id);
        Ok(akh_medu::api_types::CausalPredictResponse {
            assertions: transition
                .assertions
                .iter()
                .map(|&(s, p, o)| akh_medu::api_types::TransitionTriple {
                    subject: label(s),
                    predicate: label(p),
                    object: label(o),
                })
                .collect(),
            retractions: transition
                .retractions
                .iter()
                .map(|&(s, p, o)| akh_medu::api_types::TransitionTriple {
                    subject: label(s),
                    predicate: label(p),
                    object: label(o),
                })
                .collect(),
            confidence_changes: transition
                .confidence_changes
                .iter()
                .map(|&(s, p, o, d)| akh_medu::api_types::TransitionConfidenceChange {
                    subject: label(s),
                    predicate: label(p),
                    object: label(o),
                    delta: d,
                })
                .collect(),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn causal_applicable_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<akh_medu::api_types::CausalSchemaSummary>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let schemas = agent.causal_manager().applicable_actions(&engine);
        Ok(schemas
            .iter()
            .map(|s| akh_medu::api_types::CausalSchemaSummary {
                name: s.name.clone(),
                precondition_count: s.preconditions.len(),
                effect_count: s.effects.len(),
                success_rate: s.success_rate as f32,
                execution_count: s.execution_count as usize,
            })
            .collect())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn causal_bootstrap_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::CausalBootstrapResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let mut agent = create_agent(&engine)?;
        let tool_names: Vec<String> = agent.list_tools().iter().map(|t| t.name.clone()).collect();
        let tools_scanned = tool_names.len();
        let count = agent
            .causal_manager_mut()
            .bootstrap_schemas_from_tools(&tool_names, &engine)
            .map_err(|e| format!("{e}"))?;
        let _ = agent.persist_session();
        Ok(akh_medu::api_types::CausalBootstrapResponse {
            schemas_created: count,
            tools_scanned,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

// ── Pref handlers ───────────────────────────────────────────────────────

async fn pref_status_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::PrefStatusResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let pref = agent.preference_manager();
        Ok(akh_medu::api_types::PrefStatusResponse {
            interaction_count: pref.profile.interaction_count as usize,
            proactivity_level: format!("{}", pref.profile.proactivity_level),
            decay_rate: pref.profile.decay_rate as f32,
            suggestions_offered: pref.profile.suggestions_offered,
            suggestions_accepted: pref.profile.suggestions_accepted,
            acceptance_rate: pref.suggestion_acceptance_rate() as f32,
            prototype_active: pref.profile.interest_prototype.is_some(),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn pref_train_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::PrefTrainRequest>,
) -> Result<Json<akh_medu::api_types::PrefTrainResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let mut agent = create_agent(&engine)?;
        let sym = SymbolId::new(req.entity)
            .ok_or_else(|| "invalid entity symbol id".to_string())?;
        let signal = akh_medu::agent::FeedbackSignal::ExplicitPreference {
            topic: sym,
            weight: req.weight,
        };
        agent
            .preference_manager_mut()
            .record_feedback(&signal, &engine)
            .map_err(|e| format!("{e}"))?;
        let label = engine.resolve_label(sym);
        let total = agent.preference_manager().profile.interaction_count as usize;
        let _ = agent.persist_session();
        Ok(akh_medu::api_types::PrefTrainResponse {
            entity_label: label,
            weight: req.weight,
            total_interactions: total,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(result))
}

async fn pref_level_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::PrefLevelRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let mut agent = create_agent(&engine)?;
        let level = akh_medu::api_types::parse_proactivity_level(&req.level)
            .ok_or_else(|| format!("invalid proactivity level: {}", req.level))?;
        agent.preference_manager_mut().set_proactivity_level(level);
        let _ = agent.persist_session();
        Ok(serde_json::json!({"ok": true, "level": req.level}))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(result))
}

async fn pref_interests_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<Vec<akh_medu::api_types::PrefInterest>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let count: usize = params
        .get("count")
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let interests = agent.preference_manager().top_interests(&engine, count);
        Ok(interests
            .into_iter()
            .map(|(label, sim)| akh_medu::api_types::PrefInterest {
                label,
                similarity: sim,
            })
            .collect())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn pref_suggest_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let jitir = agent
            .preference_manager()
            .jitir_query(agent.working_memory(), agent.goals(), &engine)
            .map_err(|e| format!("{e}"))?;
        Ok(serde_json::json!({
            "context_summary": jitir.context_summary,
            "direct_matches": jitir.direct_matches.iter().map(|s| {
                serde_json::json!({
                    "label": s.label,
                    "relevance": s.relevance,
                })
            }).collect::<Vec<_>>(),
            "serendipity_matches": jitir.serendipity_matches.iter().map(|s| {
                serde_json::json!({
                    "label": s.label,
                    "relevance": s.relevance,
                    "reasoning": s.reasoning,
                })
            }).collect::<Vec<_>>(),
        }))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

// ── Calendar handlers ───────────────────────────────────────────────────

async fn cal_today_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::CalEventList>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let events: Vec<akh_medu::api_types::CalEventSummary> = agent
            .calendar_manager()
            .today_events(now)
            .iter()
            .map(|e| akh_medu::api_types::CalEventSummary {
                symbol_id: e.symbol_id.get(),
                summary: e.summary.clone(),
                duration_minutes: e.duration_secs() / 60,
                location: e.location.clone(),
            })
            .collect();
        Ok(akh_medu::api_types::CalEventList { events })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn cal_week_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<akh_medu::api_types::CalEventList>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let events: Vec<akh_medu::api_types::CalEventSummary> = agent
            .calendar_manager()
            .week_events(now)
            .iter()
            .map(|e| akh_medu::api_types::CalEventSummary {
                symbol_id: e.symbol_id.get(),
                summary: e.summary.clone(),
                duration_minutes: e.duration_secs() / 60,
                location: e.location.clone(),
            })
            .collect();
        Ok(akh_medu::api_types::CalEventList { events })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn cal_conflicts_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
) -> Result<Json<Vec<akh_medu::api_types::CalConflict>>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let agent = create_agent(&engine)?;
        let conflicts = agent.calendar_manager().detect_conflicts();
        Ok(conflicts
            .iter()
            .map(|&(a, b)| akh_medu::api_types::CalConflict {
                event_a: engine.resolve_label(a),
                event_b: engine.resolve_label(b),
            })
            .collect())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn cal_add_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::CalAddRequest>,
) -> Result<Json<akh_medu::api_types::CalAddResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let mut agent = create_agent(&engine)?;
        let sym = agent
            .calendar_manager_mut()
            .add_event(
                &engine,
                &req.summary,
                req.start,
                req.end,
                req.location.as_deref(),
                None,
                None,
                None,
            )
            .map_err(|e| format!("{e}"))?;
        let duration_minutes = req.end.saturating_sub(req.start) / 60;
        let _ = agent.persist_session();
        Ok(akh_medu::api_types::CalAddResponse {
            symbol_id: sym.get(),
            summary: req.summary,
            duration_minutes,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(result))
}

#[cfg(feature = "calendar")]
async fn cal_import_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::CalImportRequest>,
) -> Result<Json<akh_medu::api_types::CalImportResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let mut agent = create_agent(&engine)?;
        let imported = akh_medu::agent::calendar::import_ical(
            agent.calendar_manager_mut(),
            &engine,
            &req.ical_data,
        )
        .map_err(|e| format!("{e}"))?;
        let _ = agent.persist_session();
        Ok(akh_medu::api_types::CalImportResponse {
            imported_count: imported.len(),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(result))
}

#[cfg(not(feature = "calendar"))]
async fn cal_import_handler(
    _state: State<Arc<ServerState>>,
    _ws_name: Path<String>,
    _req: Json<akh_medu::api_types::CalImportRequest>,
) -> Result<Json<akh_medu::api_types::CalImportResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "calendar import requires --features calendar".into(),
    ))
}

// ── Awaken handlers ─────────────────────────────────────────────────────

async fn awaken_parse_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AwakenParseRequest>,
) -> Result<Json<akh_medu::api_types::AwakenParseResponse>, (StatusCode, String)> {
    let _engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let intent = akh_medu::bootstrap::purpose::parse_purpose(&req.statement)
            .map_err(|e| format!("{e}"))?;
        Ok(akh_medu::api_types::AwakenParseResponse {
            domain: intent.purpose.domain.clone(),
            competence_level: intent.purpose.competence_level.to_string(),
            seed_concepts: intent.purpose.seed_concepts.clone(),
            identity_name: intent.identity.as_ref().map(|i| i.name.clone()),
            identity_type: intent.identity.as_ref().map(|i| i.entity_type.to_string()),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::BAD_REQUEST, e))?;
    Ok(Json(result))
}

async fn awaken_resolve_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AwakenResolveRequest>,
) -> Result<Json<akh_medu::api_types::AwakenResolveResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let identity_ref = akh_medu::bootstrap::IdentityRef {
            name: req.name.clone(),
            entity_type: akh_medu::bootstrap::purpose::classify_entity_type(&req.name),
            source_phrase: format!("resolve {}", req.name),
        };
        let knowledge = akh_medu::bootstrap::identity::resolve_identity(&identity_ref, &engine)
            .map_err(|e| format!("{e}"))?;

        // Build a minimal purpose for the ritual.
        let purpose = akh_medu::bootstrap::PurposeModel {
            domain: knowledge.domains.first().cloned().unwrap_or_default(),
            competence_level: akh_medu::bootstrap::DreyfusLevel::Novice,
            seed_concepts: knowledge.domains.clone(),
            description: knowledge.description.clone(),
        };
        let ritual = akh_medu::bootstrap::identity::ritual_of_awakening(
            &knowledge, &purpose, &engine,
        )
        .ok();

        Ok(akh_medu::api_types::AwakenResolveResponse {
            name: knowledge.name,
            entity_type: knowledge.entity_type.to_string(),
            culture: knowledge.culture.to_string(),
            description: knowledge.description,
            domains: knowledge.domains,
            traits: knowledge.traits,
            archetypes: knowledge.archetypes,
            chosen_name: ritual.as_ref().map(|r| r.chosen_name.clone()),
            persona: ritual
                .as_ref()
                .map(|r| format!("{}", r.psyche.persona.name)),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn awaken_expand_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AwakenExpandRequest>,
) -> Result<Json<akh_medu::api_types::AwakenExpandResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let purpose = build_purpose_model(req.seeds.as_deref(), req.purpose.as_deref())?;
        let config = akh_medu::bootstrap::ExpansionConfig {
            similarity_threshold: req.threshold,
            max_concepts: req.max_concepts,
            use_conceptnet: !req.no_conceptnet,
            ..Default::default()
        };
        let mut expander =
            akh_medu::bootstrap::DomainExpander::new(&engine, config).map_err(|e| format!("{e}"))?;
        let result = expander
            .expand(&purpose, &engine)
            .map_err(|e| format!("{e}"))?;
        Ok(akh_medu::api_types::AwakenExpandResponse {
            concept_count: result.concept_count,
            relation_count: result.relation_count,
            rejected_count: result.rejected_count,
            api_calls: result.api_calls,
            accepted_labels: result.accepted_labels,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn awaken_prerequisite_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AwakenPrerequisiteRequest>,
) -> Result<Json<akh_medu::api_types::AwakenPrerequisiteResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let purpose = build_purpose_model(req.seeds.as_deref(), req.purpose.as_deref())?;

        // Expand first.
        let exp_config = akh_medu::bootstrap::ExpansionConfig::default();
        let mut expander =
            akh_medu::bootstrap::DomainExpander::new(&engine, exp_config).map_err(|e| format!("{e}"))?;
        let expansion = expander.expand(&purpose, &engine).map_err(|e| format!("{e}"))?;

        // Then prerequisites.
        let prereq_config = akh_medu::bootstrap::PrerequisiteConfig {
            known_min_triples: req.known_threshold,
            proximal_similarity_low: req.zpd_low,
            proximal_similarity_high: req.zpd_high,
            ..Default::default()
        };
        let analyzer =
            akh_medu::bootstrap::PrerequisiteAnalyzer::new(&engine, prereq_config).map_err(|e| format!("{e}"))?;
        let result = analyzer
            .analyze(&expansion, &engine)
            .unwrap_or_else(|_| {
                akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                    &expansion, &engine,
                )
            });

        let curriculum: Vec<akh_medu::api_types::CurriculumEntry> = result
            .curriculum
            .iter()
            .map(|e| akh_medu::api_types::CurriculumEntry {
                tier: e.tier as usize,
                zone: format!("{}", e.zone),
                label: e.label.clone(),
                prereq_coverage: e.prereq_coverage,
                similarity_to_known: e.similarity_to_known,
            })
            .collect();

        Ok(akh_medu::api_types::AwakenPrerequisiteResponse {
            concepts_analyzed: result.concepts_analyzed,
            edge_count: result.edge_count,
            cycles_broken: result.cycles_broken,
            max_tier: result.max_tier as usize,
            zone_distribution: result
                .zone_distribution
                .iter()
                .map(|(zone, &count)| (format!("{zone}"), count))
                .collect(),
            curriculum,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn awaken_resources_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AwakenResourcesRequest>,
) -> Result<Json<akh_medu::api_types::AwakenResourcesResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let purpose = build_purpose_model(req.seeds.as_deref(), req.purpose.as_deref())?;

        // Expand + prerequisite analysis (required for resource discovery).
        let exp_config = akh_medu::bootstrap::ExpansionConfig::default();
        let mut expander =
            akh_medu::bootstrap::DomainExpander::new(&engine, exp_config).map_err(|e| format!("{e}"))?;
        let expansion = expander.expand(&purpose, &engine).map_err(|e| format!("{e}"))?;

        let prereq_config = akh_medu::bootstrap::PrerequisiteConfig::default();
        let analyzer =
            akh_medu::bootstrap::PrerequisiteAnalyzer::new(&engine, prereq_config).map_err(|e| format!("{e}"))?;
        let prereq_result = analyzer
            .analyze(&expansion, &engine)
            .unwrap_or_else(|_| {
                akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                    &expansion, &engine,
                )
            });

        // Resource discovery.
        let res_config = akh_medu::bootstrap::ResourceDiscoveryConfig {
            min_quality: req.min_quality,
            max_api_calls: req.max_api_calls,
            use_semantic_scholar: !req.no_semantic_scholar,
            use_openalex: !req.no_openalex,
            use_open_library: !req.no_open_library,
            ..Default::default()
        };
        let mut discoverer =
            akh_medu::bootstrap::ResourceDiscoverer::new(&engine, res_config).map_err(|e| format!("{e}"))?;
        let result = discoverer
            .discover(
                &prereq_result,
                &expansion,
                &purpose.seed_concepts,
                &engine,
            )
            .map_err(|e| format!("{e}"))?;

        Ok(akh_medu::api_types::AwakenResourcesResponse {
            resources_discovered: result.resources.len(),
            api_calls_used: result.api_calls_made,
            concepts_covered: result.concepts_searched,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn awaken_ingest_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AwakenIngestRequest>,
) -> Result<Json<akh_medu::api_types::AwakenIngestResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let purpose = build_purpose_model(req.seeds.as_deref(), req.purpose.as_deref())?;

        // Full pipeline: expand → prereq → resources → ingest.
        let exp_config = akh_medu::bootstrap::ExpansionConfig::default();
        let mut expander =
            akh_medu::bootstrap::DomainExpander::new(&engine, exp_config).map_err(|e| format!("{e}"))?;
        let expansion = expander.expand(&purpose, &engine).map_err(|e| format!("{e}"))?;

        let prereq_config = akh_medu::bootstrap::PrerequisiteConfig::default();
        let analyzer =
            akh_medu::bootstrap::PrerequisiteAnalyzer::new(&engine, prereq_config).map_err(|e| format!("{e}"))?;
        let prereq_result = analyzer
            .analyze(&expansion, &engine)
            .unwrap_or_else(|_| {
                akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                    &expansion, &engine,
                )
            });

        let res_config = akh_medu::bootstrap::ResourceDiscoveryConfig::default();
        let mut discoverer =
            akh_medu::bootstrap::ResourceDiscoverer::new(&engine, res_config).map_err(|e| format!("{e}"))?;
        let resource_result = discoverer
            .discover(
                &prereq_result,
                &expansion,
                &purpose.seed_concepts,
                &engine,
            )
            .map_err(|e| format!("{e}"))?;

        let ingest_config = akh_medu::bootstrap::IngestionConfig {
            max_cycles: req.max_cycles,
            saturation_threshold: req.saturation,
            cross_validation_boost: req.xval_boost,
            try_url_ingestion: !req.no_url,
            catalog_dir: req.catalog_dir.map(std::path::PathBuf::from),
            ..Default::default()
        };
        let mut ingestor =
            akh_medu::bootstrap::CurriculumIngestor::new(&engine, ingest_config).map_err(|e| format!("{e}"))?;
        let result = ingestor
            .ingest(&prereq_result, &resource_result, &engine)
            .map_err(|e| format!("{e}"))?;

        Ok(akh_medu::api_types::AwakenIngestResponse {
            triples_added: result.total_triples,
            concepts_covered: result.concepts_ingested,
            cycles_used: result.cycles,
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn awaken_assess_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AwakenAssessRequest>,
) -> Result<Json<akh_medu::api_types::AwakenAssessResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let purpose = build_purpose_model(req.seeds.as_deref(), req.purpose.as_deref())?;

        // Expand + prereq (needed for assessment).
        let exp_config = akh_medu::bootstrap::ExpansionConfig::default();
        let mut expander =
            akh_medu::bootstrap::DomainExpander::new(&engine, exp_config).map_err(|e| format!("{e}"))?;
        let expansion = expander.expand(&purpose, &engine).map_err(|e| format!("{e}"))?;

        let prereq_config = akh_medu::bootstrap::PrerequisiteConfig::default();
        let analyzer =
            akh_medu::bootstrap::PrerequisiteAnalyzer::new(&engine, prereq_config).map_err(|e| format!("{e}"))?;
        let prereq_result = analyzer
            .analyze(&expansion, &engine)
            .unwrap_or_else(|_| {
                akh_medu::bootstrap::resources::synthetic_curriculum_from_expansion(
                    &expansion, &engine,
                )
            });

        let assess_config = akh_medu::bootstrap::CompetenceConfig {
            min_triples_per_concept: req.min_triples,
            bloom_max_depth: req.bloom_depth,
            ..Default::default()
        };
        let assessor =
            akh_medu::bootstrap::CompetenceAssessor::new(&engine, assess_config).map_err(|e| format!("{e}"))?;
        let report = assessor
            .assess(&prereq_result, &purpose, &engine)
            .map_err(|e| format!("{e}"))?;

        Ok(akh_medu::api_types::AwakenAssessResponse {
            overall_dreyfus: format!("{}", report.overall_dreyfus),
            overall_score: report.overall_score as f32,
            recommendation: report.recommendation.to_string(),
            knowledge_areas: report
                .knowledge_areas
                .iter()
                .map(|ka| akh_medu::api_types::KnowledgeAreaSummary {
                    name: ka.name.clone(),
                    dreyfus_level: format!("{}", ka.dreyfus_level),
                    score: ka.score as f32,
                })
                .collect(),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

async fn awaken_bootstrap_handler(
    State(state): State<Arc<ServerState>>,
    Path(ws_name): Path<String>,
    Json(req): Json<akh_medu::api_types::AwakenBootstrapRequest>,
) -> Result<Json<akh_medu::api_types::AwakenBootstrapResponse>, (StatusCode, String)> {
    let engine = state.get_engine(&ws_name).await?;
    let result = tokio::task::spawn_blocking(move || -> Result<_, String> {
        // Status-only request.
        if req.status {
            let session = akh_medu::bootstrap::BootstrapOrchestrator::status(&engine)
                .map_err(|e| format!("{e}"))?;
            return Ok(akh_medu::api_types::AwakenBootstrapResponse {
                domain: session.raw_purpose.clone(),
                target_level: String::new(),
                chosen_name: session.chosen_name,
                learning_cycles: session.learning_cycle,
                target_reached: false,
                final_dreyfus: session
                    .last_assessment
                    .as_ref()
                    .map(|a| format!("{}", a.overall_dreyfus)),
                final_score: session
                    .last_assessment
                    .as_ref()
                    .map(|a| a.overall_score as f32),
                recommendation: None,
            });
        }

        let config = akh_medu::bootstrap::OrchestratorConfig {
            max_learning_cycles: req.max_cycles,
            plan_only: req.plan_only,
            ..Default::default()
        };

        let mut orchestrator = if req.resume {
            akh_medu::bootstrap::BootstrapOrchestrator::resume(&engine, config)
                .map_err(|e| format!("{e}"))?
        } else {
            let stmt = req
                .statement
                .as_deref()
                .ok_or_else(|| "statement required for fresh bootstrap".to_string())?;
            akh_medu::bootstrap::BootstrapOrchestrator::new(stmt, config)
                .map_err(|e| format!("{e}"))?
        };

        let (result, _checkpoints) = orchestrator.run(&engine).map_err(|e| format!("{e}"))?;

        Ok(akh_medu::api_types::AwakenBootstrapResponse {
            domain: result.intent.purpose.domain.clone(),
            target_level: result.intent.purpose.competence_level.to_string(),
            chosen_name: result.chosen_name,
            learning_cycles: result.learning_cycles,
            target_reached: result.target_reached,
            final_dreyfus: result
                .final_report
                .as_ref()
                .map(|r| format!("{}", r.overall_dreyfus)),
            final_score: result
                .final_report
                .as_ref()
                .map(|r| r.overall_score as f32),
            recommendation: result
                .final_report
                .as_ref()
                .map(|r| r.recommendation.to_string()),
        })
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
    Ok(Json(result))
}

/// Build a PurposeModel from optional seeds/purpose fields.
fn build_purpose_model(
    seeds: Option<&[String]>,
    purpose: Option<&str>,
) -> Result<akh_medu::bootstrap::PurposeModel, String> {
    if let Some(purpose_str) = purpose {
        let intent = akh_medu::bootstrap::purpose::parse_purpose(purpose_str)
            .map_err(|e| format!("{e}"))?;
        Ok(intent.purpose)
    } else if let Some(seed_list) = seeds {
        Ok(akh_medu::bootstrap::PurposeModel {
            domain: seed_list.first().cloned().unwrap_or_default(),
            competence_level: akh_medu::bootstrap::DreyfusLevel::Novice,
            seed_concepts: seed_list.to_vec(),
            description: String::new(),
        })
    } else {
        Err("either 'seeds' or 'purpose' must be provided".into())
    }
}

// ── Main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                tracing_subscriber::EnvFilter::new("info,egg=warn,hnsw_rs=warn")
            }),
        )
        .init();

    let bind = std::env::var("AKH_SERVER_BIND").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("AKH_SERVER_PORT").unwrap_or_else(|_| "8200".to_string());
    let addr = format!("{bind}:{port}");

    let paths = AkhPaths::resolve().unwrap_or_else(|e| {
        tracing::error!("failed to resolve XDG paths: {e}");
        std::process::exit(1);
    });
    if let Err(e) = paths.ensure_dirs() {
        tracing::error!("failed to create XDG directories: {e}");
        std::process::exit(1);
    }

    let port_num: u16 = port.parse().expect("AKH_SERVER_PORT must be a valid u16");

    // Load global config from $XDG_CONFIG_HOME/akh-medu/config.toml.
    // Creates a default config file on first boot.
    let config_path = paths.global_config_file();
    let config = match AkhomedConfig::load_or_create(&config_path) {
        Ok(c) => {
            tracing::info!(path = %config_path.display(), "loaded config");
            c
        }
        Err(e) => {
            tracing::warn!(error = %e, "failed to load config, using defaults");
            AkhomedConfig::default()
        }
    };

    // AKH_AUTO_START env var overrides config if set.
    let auto_start: Vec<String> = match std::env::var("AKH_AUTO_START") {
        Ok(val) if val.is_empty() => Vec::new(),
        Ok(val) => val.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        Err(_) => config.daemon.auto_start.clone(),
    };

    let state = Arc::new(ServerState::new(paths.clone(), config));

    tracing::info!("akhomed initialized");

    // Write PID file so `akh` CLI can discover this server.
    if let Err(e) = akh_medu::client::write_pid_file(&paths, port_num, &bind) {
        tracing::warn!("failed to write PID file: {e}");
    }

    // Auto-start workspace daemons from config (or AKH_AUTO_START override).
    if !auto_start.is_empty() {
        let daemon_config = state.config.read().await.daemon.to_daemon_config();
        for ws_name in &auto_start {
            tracing::info!(workspace = %ws_name, "auto-starting daemon");
            match state.get_engine(ws_name).await {
                Ok(engine) => {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    let trigger_count = TriggerStore::new(&engine).list().len();
                    let status = Arc::new(tokio::sync::Mutex::new(DaemonStatus {
                        running: true,
                        total_cycles: 0,
                        started_at: now,
                        trigger_count,
                    }));
                    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
                    let daemon_status = Arc::clone(&status);
                    let daemon_engine = Arc::clone(&engine);
                    let daemon_ws = ws_name.clone();
                    let dc = daemon_config.clone();
                    let handle = tokio::task::spawn(async move {
                        run_daemon_task(
                            daemon_engine,
                            daemon_status,
                            shutdown_rx,
                            dc,
                            daemon_ws,
                        )
                        .await;
                    });
                    state.daemons.write().await.insert(
                        ws_name.to_string(),
                        WorkspaceDaemon {
                            handle,
                            shutdown_tx,
                            status,
                        },
                    );
                    tracing::info!(workspace = %ws_name, "daemon auto-started");
                }
                Err((_status_code, msg)) => {
                    tracing::warn!(
                        workspace = %ws_name,
                        error = %msg,
                        "failed to auto-start daemon (workspace may not exist yet)",
                    );
                }
            }
        }
    }

    let app = Router::new()
        // Health.
        .route("/health", get(health))
        // Global config.
        .route("/config", get(get_config_handler).put(put_config_handler))
        // Workspace management.
        .route("/workspaces", get(list_workspaces))
        .route("/workspaces/{name}", post(create_workspace))
        .route("/workspaces/{name}", delete(delete_workspace))
        .route("/workspaces/{name}/status", get(workspace_status))
        .route(
            "/workspaces/{ws_name}/assign-role",
            post(assign_role_handler),
        )
        // Symbols.
        .route(
            "/workspaces/{ws_name}/symbols",
            get(list_symbols).post(create_symbol),
        )
        .route(
            "/workspaces/{ws_name}/symbols/{sym}",
            get(resolve_symbol),
        )
        // Triples.
        .route(
            "/workspaces/{ws_name}/triples",
            get(list_triples).post(add_triple),
        )
        .route(
            "/workspaces/{ws_name}/triples/from/{sym_id}",
            get(triples_from),
        )
        .route(
            "/workspaces/{ws_name}/triples/to/{sym_id}",
            get(triples_to),
        )
        .route("/workspaces/{ws_name}/ingest", post(ingest_triples))
        // Query & reasoning.
        .route("/workspaces/{ws_name}/sparql", post(sparql_query))
        .route("/workspaces/{ws_name}/infer", post(infer_handler))
        .route("/workspaces/{ws_name}/reason", post(reason_handler))
        .route("/workspaces/{ws_name}/search", post(search_handler))
        .route("/workspaces/{ws_name}/analogy", post(analogy_handler))
        .route("/workspaces/{ws_name}/filler", post(filler_handler))
        // Traversal & analytics.
        .route("/workspaces/{ws_name}/traverse", post(traverse_handler))
        .route(
            "/workspaces/{ws_name}/analytics/centrality",
            get(degree_centrality_handler),
        )
        .route(
            "/workspaces/{ws_name}/analytics/pagerank",
            get(pagerank_handler),
        )
        .route(
            "/workspaces/{ws_name}/analytics/components",
            get(components_handler),
        )
        .route(
            "/workspaces/{ws_name}/analytics/shortest-path",
            post(shortest_path_handler),
        )
        // Export.
        .route(
            "/workspaces/{ws_name}/export/symbols",
            get(export_symbols),
        )
        .route(
            "/workspaces/{ws_name}/export/triples",
            get(export_triples),
        )
        .route(
            "/workspaces/{ws_name}/export/provenance/{sym}",
            get(export_provenance),
        )
        // Library.
        .route(
            "/workspaces/{ws_name}/library",
            get(library_list_handler).post(library_add_handler),
        )
        // Static /search before wildcard /{doc_id}.
        .route(
            "/workspaces/{ws_name}/library/search",
            post(library_search_handler),
        )
        .route(
            "/workspaces/{ws_name}/library/{doc_id}",
            get(library_info_handler).delete(library_remove_handler),
        )
        // Daemon.
        .route(
            "/workspaces/{ws_name}/daemon/start",
            post(start_daemon_handler),
        )
        .route(
            "/workspaces/{ws_name}/daemon/stop",
            post(stop_daemon_handler),
        )
        .route(
            "/workspaces/{ws_name}/daemon",
            get(daemon_status_handler),
        )
        // Triggers.
        .route(
            "/workspaces/{ws_name}/triggers",
            get(list_triggers_handler).post(add_trigger_handler),
        )
        .route(
            "/workspaces/{ws_name}/triggers/{trigger_id}",
            delete(remove_trigger_handler),
        )
        // Skills.
        .route(
            "/workspaces/{ws_name}/skills",
            get(list_skills_handler),
        )
        // Static /install must come before wildcard /{skill_name}.
        .route(
            "/workspaces/{ws_name}/skills/install",
            post(install_skill_handler),
        )
        .route(
            "/workspaces/{ws_name}/skills/{skill_name}/load",
            post(load_skill_handler),
        )
        .route(
            "/workspaces/{ws_name}/skills/{skill_name}/unload",
            post(unload_skill_handler),
        )
        .route(
            "/workspaces/{ws_name}/skills/{skill_name}",
            get(skill_info_handler),
        )
        // Audit.
        .route(
            "/workspaces/{ws_name}/audit",
            get(audit_list_handler),
        )
        .route(
            "/workspaces/{ws_name}/audit/{id}",
            get(audit_get_handler),
        )
        // Goals.
        .route(
            "/workspaces/{ws_name}/goals",
            get(goals_handler),
        )
        // Awaken status.
        .route(
            "/workspaces/{ws_name}/awaken/status",
            get(awaken_status_handler),
        )
        // Engine info.
        .route("/workspaces/{ws_name}/info", get(engine_info))
        // Seed packs.
        .route("/workspaces/{ws_name}/seed/{pack_name}", post(apply_seed))
        // Preprocessing.
        .route("/workspaces/{name}/preprocess", post(workspace_preprocess))
        .route(
            "/workspaces/{name}/equivalences",
            get(workspace_equivalences),
        )
        .route(
            "/workspaces/{name}/equivalences/stats",
            get(workspace_equivalences_stats),
        )
        // Seed list (root-level, not workspace-scoped).
        .route("/seeds", get(list_seeds_handler))
        // Seed status (workspace-scoped).
        .route(
            "/workspaces/{ws_name}/seeds/status",
            get(seed_status_handler),
        )
        // Render.
        .route(
            "/workspaces/{ws_name}/render",
            post(render_handler),
        )
        // Agent run/resume.
        .route(
            "/workspaces/{ws_name}/agent/run",
            post(agent_run_handler),
        )
        .route(
            "/workspaces/{ws_name}/agent/resume",
            post(agent_resume_handler),
        )
        // PIM.
        .route(
            "/workspaces/{ws_name}/pim/inbox",
            get(pim_inbox_handler),
        )
        .route(
            "/workspaces/{ws_name}/pim/next",
            post(pim_next_handler),
        )
        .route(
            "/workspaces/{ws_name}/pim/review",
            get(pim_review_handler),
        )
        .route(
            "/workspaces/{ws_name}/pim/project/{project_name}",
            get(pim_project_handler),
        )
        .route(
            "/workspaces/{ws_name}/pim/add",
            post(pim_add_handler),
        )
        .route(
            "/workspaces/{ws_name}/pim/transition",
            post(pim_transition_handler),
        )
        .route(
            "/workspaces/{ws_name}/pim/matrix",
            get(pim_matrix_handler),
        )
        .route(
            "/workspaces/{ws_name}/pim/deps",
            get(pim_deps_handler),
        )
        .route(
            "/workspaces/{ws_name}/pim/overdue",
            get(pim_overdue_handler),
        )
        // Causal.
        .route(
            "/workspaces/{ws_name}/causal/schemas",
            get(causal_schemas_handler),
        )
        .route(
            "/workspaces/{ws_name}/causal/schemas/{schema_name}",
            get(causal_schema_handler),
        )
        .route(
            "/workspaces/{ws_name}/causal/predict",
            post(causal_predict_handler),
        )
        .route(
            "/workspaces/{ws_name}/causal/applicable",
            get(causal_applicable_handler),
        )
        .route(
            "/workspaces/{ws_name}/causal/bootstrap",
            post(causal_bootstrap_handler),
        )
        // Pref.
        .route(
            "/workspaces/{ws_name}/pref/status",
            get(pref_status_handler),
        )
        .route(
            "/workspaces/{ws_name}/pref/train",
            post(pref_train_handler),
        )
        .route(
            "/workspaces/{ws_name}/pref/level",
            put(pref_level_handler),
        )
        .route(
            "/workspaces/{ws_name}/pref/interests",
            get(pref_interests_handler),
        )
        .route(
            "/workspaces/{ws_name}/pref/suggest",
            get(pref_suggest_handler),
        )
        // Calendar.
        .route(
            "/workspaces/{ws_name}/cal/today",
            get(cal_today_handler),
        )
        .route(
            "/workspaces/{ws_name}/cal/week",
            get(cal_week_handler),
        )
        .route(
            "/workspaces/{ws_name}/cal/conflicts",
            get(cal_conflicts_handler),
        )
        .route(
            "/workspaces/{ws_name}/cal/add",
            post(cal_add_handler),
        )
        .route(
            "/workspaces/{ws_name}/cal/import",
            post(cal_import_handler),
        )
        // Awaken (extends existing /awaken/status).
        .route(
            "/workspaces/{ws_name}/awaken/parse",
            post(awaken_parse_handler),
        )
        .route(
            "/workspaces/{ws_name}/awaken/resolve",
            post(awaken_resolve_handler),
        )
        .route(
            "/workspaces/{ws_name}/awaken/expand",
            post(awaken_expand_handler),
        )
        .route(
            "/workspaces/{ws_name}/awaken/prerequisite",
            post(awaken_prerequisite_handler),
        )
        .route(
            "/workspaces/{ws_name}/awaken/resources",
            post(awaken_resources_handler),
        )
        .route(
            "/workspaces/{ws_name}/awaken/ingest",
            post(awaken_ingest_handler),
        )
        .route(
            "/workspaces/{ws_name}/awaken/assess",
            post(awaken_assess_handler),
        )
        .route(
            "/workspaces/{ws_name}/awaken/bootstrap",
            post(awaken_bootstrap_handler),
        )
        // WebSocket.
        .route("/ws/{ws_name}", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(Arc::clone(&state));

    tracing::info!("akhomed listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind");

    // Serve with graceful shutdown on SIGTERM/SIGINT.
    let paths_for_shutdown = paths.clone();
    let state_for_shutdown = state;
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let ctrl_c = tokio::signal::ctrl_c();
            #[cfg(unix)]
            {
                let mut sigterm =
                    tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                        .expect("failed to register SIGTERM handler");
                tokio::select! {
                    _ = ctrl_c => {},
                    _ = sigterm.recv() => {},
                }
            }
            #[cfg(not(unix))]
            {
                ctrl_c.await.ok();
            }
            tracing::info!("akhomed shutting down — draining workspace daemons");

            // Signal all workspace daemons to stop, then join with timeout.
            let mut daemons = state_for_shutdown.daemons.write().await;
            for (name, daemon) in daemons.drain() {
                tracing::info!(workspace = %name, "stopping workspace daemon");
                let _ = daemon.shutdown_tx.send(true);
                match tokio::time::timeout(Duration::from_secs(5), daemon.handle).await {
                    Ok(Ok(())) => tracing::info!(workspace = %name, "daemon stopped cleanly"),
                    Ok(Err(e)) => tracing::warn!(workspace = %name, error = %e, "daemon task panicked"),
                    Err(_) => tracing::warn!(workspace = %name, "daemon did not stop within 5s, abandoning"),
                }
            }
            drop(daemons);

            akh_medu::client::remove_pid_file(&paths_for_shutdown);
        })
        .await
        .expect("server error");

    // Belt-and-suspenders: clean up PID file on normal exit too.
    akh_medu::client::remove_pid_file(&paths);
}
