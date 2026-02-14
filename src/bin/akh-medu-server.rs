//! akh-medu multi-workspace server.
//!
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
//! Build and run: `cargo run --features server --bin akh-medu-server`

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

use akh_medu::agent::{Agent, AgentConfig};
use akh_medu::engine::{Engine, EngineConfig};
use akh_medu::grammar::concrete::ParseContext;
use akh_medu::grammar::entity_resolution::{EquivalenceStats, LearnedEquivalence};
use akh_medu::grammar::preprocess::{preprocess_batch, PreProcessRequest, PreProcessResponse};
use akh_medu::message::AkhMessage;
use akh_medu::paths::AkhPaths;
use akh_medu::seeds::SeedRegistry;
use akh_medu::vsa::Dimension;
use akh_medu::workspace::WorkspaceManager;

// ── Server state ──────────────────────────────────────────────────────────

struct ServerState {
    paths: AkhPaths,
    workspaces: RwLock<HashMap<String, Arc<Engine>>>,
}

impl ServerState {
    fn new(paths: AkhPaths) -> Self {
        Self {
            paths,
            workspaces: RwLock::new(HashMap::new()),
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

async fn list_workspaces(
    State(state): State<Arc<ServerState>>,
) -> Json<WorkspaceListResponse> {
    let names = state.paths.list_workspaces();
    Json(WorkspaceListResponse { workspaces: names })
}

async fn create_workspace(
    State(state): State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> Result<Json<WorkspaceCreatedResponse>, (StatusCode, String)> {
    let manager = WorkspaceManager::new(state.paths.clone());
    let config = akh_medu::workspace::WorkspaceConfig {
        name: name.clone(),
        ..Default::default()
    };
    match manager.create(config) {
        Ok(_) => Ok(Json(WorkspaceCreatedResponse {
            name,
            created: true,
        })),
        Err(e) => Err((StatusCode::BAD_REQUEST, format!("{e}"))),
    }
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
    let ctx = ParseContext::with_engine(
        engine.registry(),
        engine.ops(),
        engine.item_memory(),
    );

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

fn process_ws_input(
    input: &WsInput,
    agent: &mut Agent,
    engine: &Engine,
) -> Vec<AkhMessage> {
    let mut msgs = Vec::new();

    match input.msg_type.as_str() {
        "input" => {
            let text = input.text.trim();
            if text.is_empty() {
                return msgs;
            }

            let intent = akh_medu::agent::classify_intent(text);
            match intent {
                akh_medu::agent::UserIntent::Query { subject } => {
                    match engine.resolve_symbol(&subject) {
                        Ok(sym_id) => {
                            let triples = engine.triples_from(sym_id);
                            let to_triples = engine.triples_to(sym_id);
                            if triples.is_empty() && to_triples.is_empty() {
                                msgs.push(AkhMessage::system(format!(
                                    "No facts found for \"{subject}\"."
                                )));
                            } else {
                                for t in &triples {
                                    msgs.push(AkhMessage::fact(format!(
                                        "{} {} {}",
                                        engine.resolve_label(t.subject),
                                        engine.resolve_label(t.predicate),
                                        engine.resolve_label(t.object),
                                    )));
                                }
                                for t in &to_triples {
                                    msgs.push(AkhMessage::fact(format!(
                                        "{} {} {}",
                                        engine.resolve_label(t.subject),
                                        engine.resolve_label(t.predicate),
                                        engine.resolve_label(t.object),
                                    )));
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
                akh_medu::agent::UserIntent::Assert { text } => {
                    use akh_medu::agent::tool::Tool;
                    let tool_input =
                        akh_medu::agent::ToolInput::new().with_param("text", &text);
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
                                let active =
                                    akh_medu::agent::goal::active_goals(agent.goals());
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
                    msgs.push(AkhMessage::system(
                        "Unrecognized input.".to_string(),
                    ));
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
                    msgs.push(AkhMessage::system(format!(
                        "Unknown command: \"{cmd}\""
                    )));
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

async fn send_akh_message(
    socket: &mut WebSocket,
    msg: &AkhMessage,
) -> Result<(), axum::Error> {
    let json = serde_json::to_string(msg).unwrap_or_default();
    socket.send(Message::Text(json.into())).await
}

/// Helper for error sending when we may not have a socket.
async fn send_message(
    _msg: &AkhMessage,
    _socket: &mut Option<&mut WebSocket>,
) -> Result<(), ()> {
    // No-op when socket is None.
    Ok(())
}

// ── Main ──────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    tracing_subscriber::EnvFilter::new("info,egg=warn,hnsw_rs=warn")
                }),
        )
        .init();

    let bind = std::env::var("AKH_SERVER_BIND")
        .unwrap_or_else(|_| "0.0.0.0".to_string());
    let port = std::env::var("AKH_SERVER_PORT")
        .unwrap_or_else(|_| "8200".to_string());
    let addr = format!("{bind}:{port}");

    let paths = AkhPaths::resolve().unwrap_or_else(|e| {
        tracing::error!("failed to resolve XDG paths: {e}");
        std::process::exit(1);
    });
    if let Err(e) = paths.ensure_dirs() {
        tracing::error!("failed to create XDG directories: {e}");
        std::process::exit(1);
    }

    let state = Arc::new(ServerState::new(paths));

    tracing::info!("akh-medu server initialized");

    let app = Router::new()
        // Health.
        .route("/health", get(health))
        // Workspace management.
        .route("/workspaces", get(list_workspaces))
        .route("/workspaces/{name}", post(create_workspace))
        .route("/workspaces/{name}", delete(delete_workspace))
        .route("/workspaces/{name}/status", get(workspace_status))
        // Seed packs.
        .route(
            "/workspaces/{ws_name}/seed/{pack_name}",
            post(apply_seed),
        )
        // Preprocessing.
        .route(
            "/workspaces/{name}/preprocess",
            post(workspace_preprocess),
        )
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

    tracing::info!("akh-medu server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind");
    axum::serve(listener, app).await.expect("server error");
}
