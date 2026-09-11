use super::*;
use crate::runtime::Runtime;
use crate::{config::Principal, http::Api, types::*};
use axum::{
    Extension,
    extract::{Path, State},
};
pub fn runtime_router(runtime: Runtime) -> Router {
    let config = runtime.config.clone();
    protect(
        Router::new()
            .route("/v1/jobs", post(start_job))
            .route("/v1/jobs/{id}", get(get_job))
            .route("/v1/jobs/{id}/cancel", post(cancel_job))
            .route("/v1/jobs/{id}/reconcile", post(reconcile_job))
            .with_state(runtime),
        config,
    )
}
async fn reconcile_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<Reconciliation>,
) -> Api<RuntimeJob> {
    Ok(Json(runtime.reconcile(&p, &id, &request)?))
}
async fn start_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Json(request): Json<RuntimeJobRequest>,
) -> Api<RuntimeJob> {
    Ok(Json(runtime.submit(&p, request)?))
}
async fn get_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<RuntimeJob> {
    http::role(&p, "session")?;
    Ok(Json(runtime.get(&id)?))
}
async fn cancel_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<RuntimeJob> {
    http::role(&p, "session")?;
    Ok(Json(runtime.cancel(&id)?))
}
