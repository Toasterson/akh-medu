use super::*;
use crate::ingress::Events;
use crate::{config::Principal, http::Api, types::*};
use axum::{
    Extension,
    extract::{Path, Query, State},
};
pub fn event_router(service: Events) -> Router {
    let config = service.config.clone();
    protect(
        Router::new()
            .route("/v1/events", post(ingest))
            .route("/v1/deliveries", get(deliveries))
            .route("/v1/deliveries/{id}", get(delivery))
            .route("/v1/inboxes", get(inboxes))
            .route("/v1/monitors/{source}", get(monitor))
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
    let value = service
        .store
        .get(&crate::store::key("delivery", &[&id]))?
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
#[derive(serde::Deserialize, Default)]
struct DeliveryQuery {
    #[serde(default)]
    pending: bool,
}
async fn deliveries(
    State(service): State<Events>,
    Extension(p): Extension<Principal>,
    Query(query): Query<DeliveryQuery>,
) -> Api<Vec<crate::ingress::Delivery>> {
    http::role(&p, "admin")?;
    Ok(Json(
        service
            .store
            .scan::<crate::ingress::Delivery>("delivery/")?
            .into_iter()
            .filter(|d| !query.pending || !d.terminal)
            .collect(),
    ))
}
/// `GET /v1/monitors/{source}` (Phase 53g PR 5): the monitor source's latest
/// structural snapshot, its signature, transition count and feed
/// availability. Readable by the `operator` role (any source) and by an
/// `event` principal granted that source.
async fn monitor(
    State(service): State<Events>,
    Extension(p): Extension<Principal>,
    Path(source): Path<String>,
) -> Api<akh_session_types::monitor::MonitorReading> {
    validate_key(&source)?;
    authorize_monitor_read(&p, &source)?;
    if !service.config.monitors.iter().any(|m| m.source == source) {
        return Err(Fault::missing(format!("no monitor source {source} is configured")).into());
    }
    let reading = service
        .store
        .get(&crate::monitor::snapshot_key(&source))?
        .ok_or_else(|| {
            Fault::missing(format!(
                "monitor source {source} has not completed its first observation yet; retry after one interval"
            ))
        })?;
    Ok(Json(reading))
}
fn authorize_monitor_read(principal: &Principal, source: &str) -> crate::types::Result<()> {
    if principal.has_role("operator")
        || (principal.has_role("event") && principal.sources.iter().any(|s| s == source))
    {
        Ok(())
    } else {
        Err(Fault::new(
            "forbidden",
            "reading a monitor requires the operator role, or the event role granted this source",
        ))
    }
}
