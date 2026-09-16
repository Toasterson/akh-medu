use super::*;
use crate::ingress::Events;
use crate::{config::Principal, http::Api, types::*};
use axum::{Extension, extract::{Path, State}};
pub fn event_router(service: Events) -> Router {
    let config = service.config.clone();
    protect(
        Router::new()
            .route("/v1/events", post(ingest))
            .route("/v1/deliveries", get(deliveries))
            .route("/v1/deliveries/{id}", get(delivery))
            .route("/v1/inboxes", get(inboxes))
            .with_state(service),
        config,
    )
}
async fn delivery(
    State(service): State<Events>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<crate::ingress::Delivery> {
    http::role(&p, "admin")?;
    validate_key(&id)?;
    let value = service.store.get(&crate::store::key("delivery", &[&id]))?
        .ok_or_else(|| Fault::missing("event delivery not found"))?;
    Ok(Json(value))
}
async fn inboxes(
    State(service): State<Events>,
    Extension(p): Extension<Principal>,
) -> Api<Vec<serde_json::Value>> {
    http::role(&p, "admin")?;
    Ok(Json(service.config.djehuty_inboxes.iter().map(|source| {
        let status: Option<serde_json::Value> = service.store.get(&crate::store::key("djehuty_inbox_status", &[&source.source]))?;
        Ok(json!({"source":source.source,"recipient_id":source.recipient_id,"enabled":source.enabled,"last_poll":status}))
    }).collect::<crate::types::Result<Vec<_>>>()?))
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
