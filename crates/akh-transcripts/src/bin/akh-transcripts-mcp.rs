//! `akh-transcripts-mcp` — read-only MCP server over the `harvest-pg` store.
//!
//! Increment 6 of `docs/ai/plans/2026-09-19-transcript-harvest.md`. Five
//! tools (`list_sessions`, `get_session`, `get_lines`, `grep_transcripts`,
//! `search_transcripts`) mounted at `/mcp` (streamable HTTP, rmcp), plus
//! `/healthz` (process up) and `/readyz` (a `SELECT 1` succeeded within the
//! last 30 s) for the chart's probes. Connects with the `harvest_read`
//! Postgres role — `SELECT` only, this binary never writes and never runs
//! migrations.
//!
//! Env: `AKH_TRANSCRIPTS_DATABASE_URL` (required), `AKH_TRANSCRIPTS_BIND`
//! (default `0.0.0.0:9500`), `AKH_TRANSCRIPTS_BEARER_TOKEN[_FILE]`
//! (required — the server refuses to start unauthenticated),
//! `AKH_TRANSCRIPTS_MODEL_DIR` (optional — `search_transcripts` degrades to
//! a clear error without it; the other four tools are unaffected),
//! `RUST_LOG`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use akh_transcripts::auth::{self, SharedSecretAuth};
use akh_transcripts::mcp::{Readiness, StartupError, TranscriptsMcpServer, TranscriptsMcpState};
use akh_transcripts::pool;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use clap::Parser;
use sea_orm::DatabaseConnection;

/// Read-only MCP server exposing harvested Claude Code / Codex / OpenCode
/// agent session transcripts. Configured entirely by environment variable —
/// see the module docs — this exists mainly so `--help`/`--version` work.
#[derive(Parser, Debug)]
#[command(name = "akh-transcripts-mcp", version, about)]
struct Cli {}

#[derive(Clone)]
struct HealthState {
    db: Arc<DatabaseConnection>,
    readiness: Arc<Readiness>,
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn readyz(State(health): State<HealthState>) -> impl IntoResponse {
    health.readiness.probe(&health.db).await;
    if health.readiness.is_ready() {
        (StatusCode::OK, "ready")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready")
    }
}

fn default_bind() -> String {
    "0.0.0.0:9500".to_string()
}

/// Load the embedder from `AKH_TRANSCRIPTS_MODEL_DIR`. Absence or failure is
/// a **warning, not a startup failure** — the contract requires
/// `search_transcripts` to degrade to a clear per-call error while the other
/// four tools keep working, rather than the whole server refusing to start.
fn load_embedder() -> Option<akh_embed::SharedEmbedder> {
    let Some(dir) = std::env::var_os("AKH_TRANSCRIPTS_MODEL_DIR") else {
        tracing::warn!(target: "akh_transcripts::mcp", 
            "AKH_TRANSCRIPTS_MODEL_DIR is not set — search_transcripts will return a clear \
             error; list_sessions, get_session, get_lines, and grep_transcripts are unaffected"
        );
        return None;
    };
    let dir = PathBuf::from(dir);
    match akh_embed::SentenceEmbedder::load(&dir) {
        Ok(embedder) => {
            tracing::info!(target: "akh_transcripts::mcp", dir = %dir.display(), "embedding model loaded");
            Some(Arc::new(Mutex::new(embedder)))
        }
        Err(error) => {
            tracing::warn!(target: "akh_transcripts::mcp", 
                %error,
                dir = %dir.display(),
                "embedding model failed to load — search_transcripts will return a clear error"
            );
            None
        }
    }
}

#[tokio::main]
async fn main() -> miette::Result<()> {
    let _cli = Cli::parse();

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Authentication is mandatory and checked before the database connects
    // or the listener binds, so the server is never briefly reachable
    // unauthenticated during startup (mirrors crates/seshat/src/main.rs).
    let bearer = SharedSecretAuth::from_env().map_err(StartupError::BearerTokenMissing)?;

    let database_url = std::env::var("AKH_TRANSCRIPTS_DATABASE_URL")
        .map_err(|_| StartupError::DatabaseUrlMissing)?;
    let db = pool::connect(&database_url).await?;
    tracing::info!(target: "akh_transcripts::mcp", "connected to harvest-pg (harvest_read role)");
    let db = Arc::new(db);

    let embedder = load_embedder();

    let readiness = Arc::new(Readiness::new());
    let state = Arc::new(TranscriptsMcpState {
        db: Arc::clone(&db),
        embedder,
        readiness: Arc::clone(&readiness),
    });
    let mcp_server = TranscriptsMcpServer::new(state);

    use rmcp::transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    };
    let session_manager = Arc::new(LocalSessionManager::default());
    let mcp_config = StreamableHttpServerConfig::default();
    let mcp_service =
        StreamableHttpService::new(move || Ok(mcp_server.clone()), session_manager, mcp_config);

    // `/healthz` and `/readyz` stay outside the bearer layer — a Kubernetes
    // probe never carries a bearer token.
    let health_state = HealthState {
        db: Arc::clone(&db),
        readiness: Arc::clone(&readiness),
    };
    let health_router = axum::Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .with_state(health_state);

    let mcp_router = axum::Router::new().nest_service("/mcp", mcp_service).layer(
        axum::middleware::from_fn_with_state(bearer, auth::require_bearer),
    );

    let app = health_router.merge(mcp_router);

    let bind = std::env::var("AKH_TRANSCRIPTS_BIND").unwrap_or_else(|_| default_bind());
    let addr: SocketAddr = bind
        .parse()
        .map_err(|source| StartupError::InvalidBindAddress { addr: bind, source })?;

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|source| StartupError::BindFailed { addr, source })?;
    tracing::info!(target: "akh_transcripts::mcp", %addr, "akh-transcripts-mcp listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!(target: "akh_transcripts::mcp", "shutting down");
        })
        .await
        .map_err(|source| StartupError::BindFailed { addr, source })?;

    Ok(())
}
