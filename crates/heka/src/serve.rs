//! HTTP surface for the `hekad serve` subcommand: `/health` plus the
//! tenant-scoped MCP route at `/akh/{akh_id}/mcp`.
//!
//! The router is composed with [`axum::Router::route_service`], not `nest`,
//! because the rmcp 0.17 `StreamableHttpService` matches the MCP transport
//! semantics of the full route path (including trailing-slash requests) and
//! `nest` would strip the prefix before the service — and its session manager
//! — see it. The auth middleware extracts the `{akh_id}` path parameter
//! itself, so it is layered around the service directly rather than around
//! the whole router (axum path extractors only work in route handlers).

use std::future::Future;
use std::sync::Arc;

use axum::{Json, Router, extract::State, middleware, response::IntoResponse, routing::get};
use sea_orm::DatabaseConnection;

use crate::auth::{AkhAuthState, require_akh_bearer};
use crate::mcp::tenant_mcp_service;

/// Health probe body. `db` is `null` when the probe did not reach the
/// database (the handler then still answers 200 — liveness, not readiness,
/// so an orchestrator does not restart the pod during a brief DB blip; the
/// MCP route's fail-closed auth middleware is what reports DB outages to
/// clients, as 503).
#[derive(Debug, serde::Serialize)]
struct HealthBody {
    status: &'static str,
    db: Option<&'static str>,
}

async fn health(State(db): State<Arc<DatabaseConnection>>) -> impl IntoResponse {
    let db_state = match db.ping().await {
        Ok(()) => Some("up"),
        Err(error) => {
            tracing::warn!(%error, "health probe: database ping failed");
            None
        }
    };
    Json(HealthBody {
        status: "ok",
        db: db_state,
    })
}

/// Build the hekad HTTP router: a liveness `/health` probe plus the
/// per-tenant MCP endpoint behind fail-closed bearer authentication
/// ([`require_akh_bearer`]). The validated tenant flows from the middleware
/// into request extensions and from there into every MCP tool call
/// (see [`crate::mcp`]).
pub fn router(db: Arc<DatabaseConnection>) -> Router {
    let auth_state = AkhAuthState::new(Arc::clone(&db));
    let tenant_route = Router::new()
        .route_service("/akh/{akh_id}/mcp", tenant_mcp_service(auth_state.clone()))
        .layer(middleware::from_fn_with_state(auth_state, require_akh_bearer));

    Router::new()
        .route("/health", get(health))
        .merge(tenant_route)
        .with_state(db)
}

/// Bind `bind_addr` and serve [`router`] until `shutdown` resolves; in-flight
/// MCP requests are then drained by `axum::serve`'s graceful shutdown
/// (pending connections finish; the listener stops accepting new ones).
///
/// Note: the rmcp `LocalSessionManager`'s cancellation token is not tied to
/// this signal, so a client that holds a long-lived SSE stream open without
/// sending further requests can extend the drain until its HTTP connection
/// closes. Heka's tool set is request/response-only (no notifications or
/// server-initiated SSE pushes), so well-behaved clients open at most a
/// short-lived SSE get per session and the drain completes promptly in
/// practice; a hard drain timeout can be layered on later if real deployments
/// show stuck sessions.
pub async fn serve(
    listener: tokio::net::TcpListener,
    db: Arc<DatabaseConnection>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    axum::serve(listener, router(db))
        .with_graceful_shutdown(shutdown)
        .await
}
