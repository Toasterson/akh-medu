use super::*;
use crate::executor::Executor;
use crate::{
    config::Principal,
    http::{Api, ApiError},
    types::*,
};
use axum::{
    Extension,
    extract::{Path, State},
};
use serde::Deserialize;
use serde_json::Value;
pub fn executor_router(executor: Executor) -> Router {
    let config = executor.config.clone();
    protect(
        Router::new()
            .route("/v1/services", post(launch_service))
            .route("/v1/services/{id}", get(service_status))
            .route("/v1/services/{id}/stop", post(stop_service))
            .route("/v1/services/{id}/reconcile", post(reconcile_service))
            .route("/v1/services/{id}/restart", post(restart_service))
            .route("/v1/services/{id}/actions/{action}", post(service_action))
            .route("/v1/workspaces", post(attach_workspace))
            .route("/v1/workspaces/{id}/release", post(release_workspace))
            .route("/v1/workspaces/{id}/renew", post(renew_lease))
            .route("/v1/operations", post(operation))
            .route("/v1/repository-publications", post(publish_repository))
            .route("/v1/operations/{id}", get(receipt))
            .route("/v1/operations/{id}/reconcile", post(reconcile_receipt))
            .with_state(executor),
        config,
    )
}
async fn stop_service(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<HelperStatus> {
    http::role(&p, "admin")?;
    Ok(Json(crate::helper::stop(&executor.service_manifest(&id)?)?))
}
async fn reconcile_service(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<Reconciliation>,
) -> Api<HelperStatus> {
    Ok(Json(executor.reconcile_service(&p, &id, &request)?))
}
async fn restart_service(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<ServiceRestart>,
) -> Api<HelperStatus> {
    Ok(Json(executor.restart_service(&p, &id, &request)?))
}
async fn renew_lease(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<RenewLease>,
) -> Api<WorkspaceLease> {
    http::role(&p, "sandbox")?;
    Ok(Json(executor.renew_lease(&id, &request)?))
}
async fn reconcile_receipt(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<Reconciliation>,
) -> Api<OperationReceipt> {
    Ok(Json(executor.reconcile(&p, &id, &request)?))
}
async fn launch_service(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Json(request): Json<crate::types::LaunchService>,
) -> Api<crate::types::HelperStatus> {
    http::role(&p, "sandbox")?;
    Ok(Json(executor.launch_service(request)?))
}
async fn service_status(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<crate::types::HelperStatus> {
    if !p.has_role("session") && !p.has_role("sandbox") {
        http::role(&p, "admin")?;
    }
    Ok(Json(executor.service_status(&id)?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ServiceAction {
    session_id: String,
}
async fn service_action(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path((id, action)): Path<(String, String)>,
    Json(request): Json<ServiceAction>,
) -> Api<Value> {
    http::role(&p, "session")?;
    let manifest = executor.service_manifest(&id)?;
    if manifest.session_id != request.session_id {
        return Err(Fault::new("forbidden", "service belongs to another session").into());
    }
    Ok(Json(crate::helper::action(&manifest, &action).await?))
}
async fn attach_workspace(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Json(request): Json<AttachWorkspace>,
) -> Api<WorkspaceBinding> {
    http::role(&p, "sandbox")?;
    Ok(Json(executor.attach(&request)?))
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Release {
    session_id: String,
    generation: u64,
}
async fn release_workspace(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<Release>,
) -> Api<Value> {
    http::role(&p, "sandbox")?;
    Ok(Json(executor.release(
        &id,
        &request.session_id,
        request.generation,
    )?))
}
async fn operation(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Json(request): Json<WorkspaceOperation>,
) -> Api<OperationReceipt> {
    if request.operation == "git_publish" {
        return Err(Fault::new(
            "forbidden",
            "Git publication is a controller-owned operation",
        )
        .into());
    }
    if !p.has_role("sandbox") {
        http::role(&p, "session")?;
    }
    Ok(Json(
        tokio::task::spawn_blocking(move || executor.operate(request))
            .await
            .map_err(|_| ApiError(Fault::new("executor", "operation worker failed")))??,
    ))
}
async fn publish_repository(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Json(request): Json<WorkspaceOperation>,
) -> Api<OperationReceipt> {
    http::role(&p, "repository_publish")?;
    if request.operation != "git_publish" {
        return Err(Fault::invalid("expected Git publication operation").into());
    }
    Ok(Json(
        tokio::task::spawn_blocking(move || executor.operate(request))
            .await
            .map_err(|_| ApiError(Fault::new("executor", "publication worker failed")))??,
    ))
}
async fn receipt(
    State(executor): State<Executor>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<OperationReceipt> {
    if !p.has_role("sandbox") && !p.has_role("repository_publish") {
        http::role(&p, "session")?;
    }
    Ok(Json(executor.receipt(&id)?))
}
