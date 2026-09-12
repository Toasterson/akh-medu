use super::*;
use crate::ingress::Events;
use crate::{config::Principal, http::Api, types::*};
use axum::{Extension, extract::State};
pub fn event_router(service: Events) -> Router {
    let config = service.config.clone();
    protect(
        Router::new()
            .route("/v1/events", post(ingest))
            .route("/v1/deliveries", get(deliveries))
            .with_state(service),
        config,
    )
}
async fn ingest(
    State(service): State<Events>,
    Extension(p): Extension<Principal>,
    Json(event): Json<EventEnvelope>,
) -> Api<crate::ingress::IngressReceipt> {
    Ok(Json(service.ingest(&p, event)?))
}
async fn deliveries(
    State(service): State<Events>,
    Extension(p): Extension<Principal>,
) -> Api<Vec<crate::ingress::Delivery>> {
    http::role(&p, "admin")?;
    Ok(Json(service.store.scan("delivery/")?))
}
