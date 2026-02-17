//! akhomed — the akh-medu daemon.
//!
//! Single authority over engine instances; the `akh` CLI connects here.
//! Hosts N engine workspaces with REST and WebSocket APIs:
//!
//! **Workspace management:**
//! - `GET  /workspaces` — list workspaces
//! - `POST /workspaces/{name}` — create workspace
//! - `DELETE /workspaces/{name}` — delete workspace
//! - `GET  /workspaces/{name}/status` — engine stats
//!
//! **Seed packs:**
//! - `POST /workspaces/{name}/seed/{pack}` — apply seed pack
//!
//! **Preprocessing (per-workspace):**
//! - `POST /workspaces/{name}/preprocess` — preprocess text chunks
//! - `GET  /workspaces/{name}/equivalences` — list equivalences
//!
//! **WebSocket (TUI connection):**
//! - `GET  /ws/{workspace}` — WebSocket upgrade for TUI streaming
//!
//! **Health:**
//! - `GET  /health` — server status
//!
//! Build and run: `cargo run --features server --bin akhomed`

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, State, WebSocketUpgrade};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;

use akh_medu::agent::trigger::{self as trigger_mod, Trigger, TriggerStore};
use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::client::DaemonStatus;
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
    workspaces: RwLock<HashMap<String, Arc<Engine>>>,
    daemons: RwLock<HashMap<String, WorkspaceDaemon>>,
}

/// Background daemon state for a workspace.
struct WorkspaceDaemon {
    handle: tokio::task::JoinHandle<()>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    status: Arc<tokio::sync::Mutex<DaemonStatus>>,
}

impl ServerState {
    fn new(paths: AkhPaths) -> Self {
        Self {
            paths,
            workspaces: RwLock::new(HashMap::new()),
            daemons: RwLock::new(HashMap::new()),
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

        let engine = Arc::new(engine);
        let mut map = self.workspaces.write().await;
        map.insert(name.to_string(), Arc::clone(&engine));
        Ok(engine)
    }
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
            if let Some(Json(req)) = body {
                if let Some(ref role) = req.role {
                    let engine = state.get_engine(&name).await?;
                    engine.assign_role(role).map_err(|e| {
                        (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))
                    })?;
                    engine.persist().map_err(|e| {
                        (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}"))
                    })?;
                }
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

    let result = if req.source.starts_with("http://") || req.source.starts_with("https://") {
        akh_medu::library::ingest_url(&engine, &mut catalog, &req.source, ingest_config)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    } else {
        let path = std::path::PathBuf::from(&req.source);
        akh_medu::library::ingest_file(&engine, &mut catalog, &path, ingest_config)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e}")))?
    };

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
        if let Some(d) = daemons.get(&ws_name) {
            if !d.handle.is_finished() {
                let st = d.status.lock().await;
                return Ok(Json(st.clone()));
            }
        }
    }

    let engine = state.get_engine(&ws_name).await?;
    let max_cycles = body.map(|b| b.max_cycles).unwrap_or(0);

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
        run_daemon_task(daemon_engine, daemon_status, shutdown_rx, max_cycles, daemon_ws_name)
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

/// Background daemon task: runs agent cycles, evaluates triggers.
async fn run_daemon_task(
    engine: Arc<Engine>,
    status: Arc<tokio::sync::Mutex<DaemonStatus>>,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
    max_cycles: usize,
    ws_name: String,
) {
    use tokio::time::{interval, Duration};

    let mut cycle_tick = interval(Duration::from_secs(30));
    let mut trigger_tick = interval(Duration::from_secs(15));
    let mut persist_tick = interval(Duration::from_secs(60));

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

    let agent = Arc::new(tokio::sync::Mutex::new(agent));

    loop {
        tokio::select! {
            _ = cycle_tick.tick() => {
                let cycle_agent = Arc::clone(&agent);
                let cycle_engine = Arc::clone(&engine);
                let cycle_status = Arc::clone(&status);
                let _ = tokio::task::spawn_blocking(move || {
                    let mut agent = cycle_agent.blocking_lock();
                    if akh_medu::agent::goal::active_goals(agent.goals()).is_empty() {
                        return;
                    }
                    if let Ok(_result) = agent.run_cycle() {
                        let mut st = cycle_status.blocking_lock();
                        st.total_cycles += 1;
                    }
                    let _ = cycle_engine; // keep alive
                }).await;

                if max_cycles > 0 {
                    let st = status.lock().await;
                    if st.total_cycles >= max_cycles {
                        tracing::info!(ws = %ws_name, "daemon: max cycles reached");
                        break;
                    }
                }
            }
            _ = trigger_tick.tick() => {
                let trig_agent = Arc::clone(&agent);
                let trig_engine = Arc::clone(&engine);
                let trig_status = Arc::clone(&status);
                let _ = tokio::task::spawn_blocking(move || {
                    let store = TriggerStore::new(&trig_engine);
                    let triggers = store.list();
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();

                    let mut agent = trig_agent.blocking_lock();
                    for trigger in &triggers {
                        if trigger_mod::should_fire(trigger, &agent, now) {
                            match trigger_mod::execute_trigger(trigger, &mut agent) {
                                Ok(msg) => tracing::info!("{msg}"),
                                Err(e) => tracing::warn!(error = %e, "trigger execution failed"),
                            }
                            store.update_last_fired(&trigger.id, now);
                        }
                    }

                    // Update trigger count in status.
                    let mut st = trig_status.blocking_lock();
                    st.trigger_count = triggers.len();
                }).await;
            }
            _ = persist_tick.tick() => {
                let persist_agent = Arc::clone(&agent);
                let _ = tokio::task::spawn_blocking(move || {
                    let agent = persist_agent.blocking_lock();
                    let _ = agent.persist_session();
                }).await;
            }
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    tracing::info!(ws = %ws_name, "daemon: shutdown signal received");
                    break;
                }
            }
        }
    }

    // Final persist.
    let agent = Arc::clone(&agent);
    let _ = tokio::task::spawn_blocking(move || {
        let agent = agent.blocking_lock();
        let _ = agent.persist_session();
    })
    .await;

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
    ws.on_upgrade(move |socket| async move {
        match engine_result {
            Ok(engine) => handle_ws_session(socket, engine, ws_name).await,
            Err((_, msg)) => {
                let err = AkhMessage::error("ws", msg);
                let _ = send_message(&err, &mut None::<&mut WebSocket>).await;
            }
        }
    })
}

async fn handle_ws_session(mut socket: WebSocket, engine: Arc<Engine>, ws_name: String) {
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

    let welcome = AkhMessage::system(format!(
        "Connected to workspace \"{ws_name}\". {} symbols, {} triples.",
        engine.all_symbols().len(),
        engine.all_triples().len(),
    ));
    if send_akh_message(&mut socket, &welcome).await.is_err() {
        return;
    }

    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            Message::Text(text) => {
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

                let responses = process_ws_input(&input, &mut agent, &engine);
                for msg in &responses {
                    if send_akh_message(&mut socket, msg).await.is_err() {
                        return;
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    // Persist session on disconnect.
    let _ = agent.persist_session();
}

fn process_ws_input(input: &WsInput, agent: &mut Agent, engine: &Engine) -> Vec<AkhMessage> {
    let mut msgs = Vec::new();

    match input.msg_type.as_str() {
        "input" => {
            let text = input.text.trim();
            if text.is_empty() {
                return msgs;
            }

            let intent = akh_medu::agent::classify_intent(text);
            match intent {
                akh_medu::agent::UserIntent::Query { subject, original_input, question_word, capability_signal } => {
                    let grammar_name = engine
                        .compartments()
                        .and_then(|mgr| mgr.psyche())
                        .map(|p| p.persona.grammar_preference.clone())
                        .unwrap_or_else(|| "narrative".to_string());

                    // Try discourse-aware response first.
                    let discourse_prose = akh_medu::grammar::discourse::resolve_discourse(
                        &subject,
                        question_word,
                        &original_input,
                        engine,
                        capability_signal,
                    )
                    .ok()
                    .and_then(|ctx| {
                        let from = engine.triples_from(ctx.subject_id);
                        let to = engine.triples_to(ctx.subject_id);
                        let mut all = from;
                        all.extend(to);
                        akh_medu::grammar::discourse::build_discourse_response(
                            &all, &ctx, engine,
                        )
                    })
                    .and_then(|tree| {
                        let registry = akh_medu::grammar::GrammarRegistry::new();
                        registry.linearize(&grammar_name, &tree).ok()
                    })
                    .filter(|s| !s.trim().is_empty());

                    if let Some(prose) = discourse_prose {
                        msgs.push(AkhMessage::narrative(&prose, &grammar_name));
                    } else {
                        // Fallback: existing synthesis path.
                        match engine.resolve_symbol(&subject) {
                            Ok(sym_id) => {
                                let from_triples = engine.triples_from(sym_id);
                                let to_triples = engine.triples_to(sym_id);
                                if from_triples.is_empty() && to_triples.is_empty() {
                                    msgs.push(AkhMessage::system(format!(
                                        "No facts found for \"{subject}\"."
                                    )));
                                } else {
                                    let mut all_triples = from_triples;
                                    all_triples.extend(to_triples);
                                    let summary =
                                        akh_medu::agent::synthesize::synthesize_from_triples(
                                            &subject,
                                            &all_triples,
                                            engine,
                                            &grammar_name,
                                        );
                                    if !summary.overview.is_empty() {
                                        msgs.push(AkhMessage::narrative(
                                            &summary.overview,
                                            &grammar_name,
                                        ));
                                    }
                                    for section in &summary.sections {
                                        msgs.push(AkhMessage::narrative(
                                            format!("## {}\n{}", section.heading, section.prose),
                                            &grammar_name,
                                        ));
                                    }
                                    for gap in &summary.gaps {
                                        msgs.push(AkhMessage::gap("(unknown)", gap));
                                    }
                                }
                            }
                            Err(_) => {
                                msgs.push(AkhMessage::system(format!(
                                    "Symbol \"{subject}\" not found."
                                )));
                            }
                        }
                    }
                }
                akh_medu::agent::UserIntent::Assert { text } => {
                    use akh_medu::agent::tool::Tool;
                    let tool_input = akh_medu::agent::ToolInput::new().with_param("text", &text);
                    match akh_medu::agent::tools::TextIngestTool.execute(engine, tool_input) {
                        Ok(output) => {
                            msgs.push(AkhMessage::tool_result(
                                "text_ingest",
                                output.success,
                                &output.result,
                            ));
                        }
                        Err(e) => {
                            msgs.push(AkhMessage::error("ingest", e.to_string()));
                        }
                    }
                }
                akh_medu::agent::UserIntent::SetGoal { description } => {
                    match agent.add_goal(&description, 128, "User-directed goal") {
                        Ok(id) => {
                            msgs.push(AkhMessage::system(format!(
                                "Goal added: \"{description}\" (id: {})",
                                id.get()
                            )));
                            // Run a few cycles.
                            for _ in 0..5 {
                                match agent.run_cycle() {
                                    Ok(result) => {
                                        msgs.push(AkhMessage::tool_result(
                                            &result.decision.chosen_tool,
                                            result.action_result.tool_output.success,
                                            &result.action_result.tool_output.result,
                                        ));
                                    }
                                    Err(_) => break,
                                }
                                let active = akh_medu::agent::goal::active_goals(agent.goals());
                                if active.is_empty() {
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            msgs.push(AkhMessage::error("goal", e.to_string()));
                        }
                    }
                }
                akh_medu::agent::UserIntent::RunAgent { cycles } => {
                    let n = cycles.unwrap_or(1);
                    for _ in 0..n {
                        match agent.run_cycle() {
                            Ok(result) => {
                                msgs.push(AkhMessage::tool_result(
                                    &result.decision.chosen_tool,
                                    result.action_result.tool_output.success,
                                    &result.action_result.tool_output.result,
                                ));
                            }
                            Err(e) => {
                                msgs.push(AkhMessage::error("cycle", e.to_string()));
                                break;
                            }
                        }
                    }
                }
                akh_medu::agent::UserIntent::ShowStatus => {
                    let goals = agent.goals();
                    for g in goals {
                        msgs.push(AkhMessage::goal_progress(
                            &g.description,
                            format!("{}", g.status),
                        ));
                    }
                    msgs.push(AkhMessage::system(format!(
                        "Cycles: {}, WM: {}, Triples: {}",
                        agent.cycle_count(),
                        agent.working_memory().len(),
                        engine.all_triples().len(),
                    )));
                }
                akh_medu::agent::UserIntent::Help => {
                    msgs.push(AkhMessage::system(
                        "Send JSON: {\"type\":\"input\",\"text\":\"...\"} or {\"type\":\"command\",\"cmd\":\"...\"}"
                            .to_string(),
                    ));
                }
                _ => {
                    msgs.push(AkhMessage::system("Unrecognized input.".to_string()));
                }
            }
        }
        "command" => {
            let cmd = input.text.trim();
            match cmd {
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
                    msgs.push(AkhMessage::system(format!("Unknown command: \"{cmd}\"")));
                }
            }
        }
        _ => {
            msgs.push(AkhMessage::error(
                "protocol",
                format!("unknown message type: \"{}\"", input.msg_type),
            ));
        }
    }

    msgs
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

    let state = Arc::new(ServerState::new(paths.clone()));

    tracing::info!("akhomed initialized");

    // Write PID file so `akh` CLI can discover this server.
    if let Err(e) = akh_medu::client::write_pid_file(&paths, port_num, &bind) {
        tracing::warn!("failed to write PID file: {e}");
    }

    let app = Router::new()
        // Health.
        .route("/health", get(health))
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
        // WebSocket.
        .route("/ws/{ws_name}", get(ws_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    tracing::info!("akhomed listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind");

    // Serve with graceful shutdown on SIGTERM/SIGINT.
    let paths_for_shutdown = paths.clone();
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
            tracing::info!("akhomed shutting down");
            akh_medu::client::remove_pid_file(&paths_for_shutdown);
        })
        .await
        .expect("server error");

    // Belt-and-suspenders: clean up PID file on normal exit too.
    akh_medu::client::remove_pid_file(&paths);
}
