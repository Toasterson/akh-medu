use super::*;
use crate::sandbox::Sandboxes;
use crate::{config::Principal, http::Api, types::*};
use axum::{
    Extension,
    extract::{Path, State},
};
pub fn sandbox_router(service: Sandboxes) -> Router {
    let config = service.config.clone();
    protect(
        Router::new()
            .route("/v1/allocations", post(allocate))
            .route("/v1/allocations/{id}", get(allocation))
            .route("/v1/allocations/{id}/release", post(release_allocation))
            .with_state(service),
        config,
    )
}
async fn allocate(
    State(service): State<Sandboxes>,
    Extension(p): Extension<Principal>,
    Json(request): Json<Allocate>,
) -> Api<Allocation> {
    http::role(&p, "session")?;
    Ok(Json(service.allocate(request).await?))
}
async fn allocation(
    State(service): State<Sandboxes>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<Allocation> {
    http::role(&p, "session")?;
    Ok(Json(service.get(&id)?))
}
async fn release_allocation(
    State(service): State<Sandboxes>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<Allocation> {
    http::role(&p, "session")?;
    Ok(Json(service.release(&id).await?))
}
