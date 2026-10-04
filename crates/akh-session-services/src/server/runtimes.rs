use super::*;
use crate::runtime::{QuarantineQuery, Runtime};
use crate::{config::Principal, http::Api, types::*};
use axum::{
    Extension,
    extract::{Path, Query, State},
};
pub fn runtime_router(runtime: Runtime) -> Router {
    let config = runtime.config.clone();
    protect(
        Router::new()
            .route("/v1/jobs", get(quarantine_inventory).post(start_job))
            .route("/v1/jobs/cancel-unsubmitted", post(cancel_unsubmitted))
            .route("/v1/jobs/{id}", get(get_job))
            .route("/v1/jobs/{id}/cancel", post(cancel_job))
            .route("/v1/jobs/{id}/reconcile", post(reconcile_job))
            .with_state(runtime),
        config,
    )
}
async fn quarantine_inventory(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Query(query): Query<QuarantineQuery>,
) -> Api<RuntimeQuarantinePage> {
    Ok(Json(runtime.quarantine_inventory(&p, &query)?))
}
async fn reconcile_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<Reconciliation>,
) -> Api<RuntimeJob> {
    Ok(Json(wire_job(runtime.reconcile(&p, &id, &request)?)))
}
async fn start_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Json(request): Json<RuntimeJobRequest>,
) -> Api<RuntimeJob> {
    Ok(Json(wire_job(runtime.submit(&p, request)?)))
}
async fn get_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<RuntimeJob> {
    http::role(&p, "session")?;
    Ok(Json(wire_job(runtime.get(&id)?)))
}
async fn cancel_job(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<RuntimeJob> {
    http::role(&p, "session")?;
    Ok(Json(wire_job(runtime.cancel(&id)?)))
}

async fn cancel_unsubmitted(
    State(runtime): State<Runtime>,
    Extension(p): Extension<Principal>,
    Json(request): Json<RuntimeJobRequest>,
) -> Api<RuntimeJob> {
    Ok(Json(wire_job(runtime.cancel_unsubmitted(&p, request)?)))
}

// Older session services reject unknown RuntimeEvent variants. Keep the full
// durable event stream and OTLP output, but expose only their supported events.
fn wire_job(job: RuntimeJob) -> RuntimeJob {
    compatible_job(
        job,
        std::env::var("AKH_RUNTIME_LEGACY_EVENTS").as_deref() == Ok("true"),
    )
}
fn compatible_job(mut job: RuntimeJob, legacy: bool) -> RuntimeJob {
    if legacy {
        legacy_events(&mut job.events);
    }
    job
}

fn legacy_events(events: &mut Vec<RuntimeEvent>) {
    events.retain(|event| !matches!(event, RuntimeEvent::Reasoning { .. }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn quarantine_collection_requires_existing_session_auth_and_rejects_unknown_queries() {
        use axum::{
            body::Body,
            http::{Request, StatusCode},
        };
        use tower::ServiceExt;
        let dir = tempfile::tempdir().unwrap();
        let session_token = "session-fixture-token-with-32-characters";
        let client_token = "client-fixture-token-with-32-characters";
        let session_file = dir.path().join("session-token");
        let client_file = dir.path().join("client-token");
        std::fs::write(&session_file, session_token).unwrap();
        std::fs::write(&client_file, client_token).unwrap();
        let config = Arc::new(
            serde_json::from_value::<Config>(json!({
                "service_id":"runtime-1", "listen":"127.0.0.1:0", "data_dir":dir.path(),
                "principals":[
                    {"id":"sessions","token_file":session_file,"roles":["session"]},
                    {"id":"client","token_file":client_file,"roles":["client"]}
                ]
            }))
            .unwrap(),
        );
        let runtime =
            Runtime::new(config, Store::open(&dir.path().join("state.redb")).unwrap()).unwrap();
        let app = runtime_router(runtime);
        for (path, token, expected) in [
            ("/v1/jobs?status=unknown", None, StatusCode::UNAUTHORIZED),
            (
                "/v1/jobs?status=unknown",
                Some(client_token),
                StatusCode::FORBIDDEN,
            ),
            (
                "/v1/jobs?status=unknown",
                Some(session_token),
                StatusCode::OK,
            ),
            (
                "/v1/jobs?status=running",
                Some(session_token),
                StatusCode::BAD_REQUEST,
            ),
            (
                "/v1/jobs?limit=0",
                Some(session_token),
                StatusCode::BAD_REQUEST,
            ),
            (
                "/v1/jobs?limit=101",
                Some(session_token),
                StatusCode::BAD_REQUEST,
            ),
            (
                "/v1/jobs?limit=-1",
                Some(session_token),
                StatusCode::BAD_REQUEST,
            ),
            (
                "/v1/jobs?runtime=foreign",
                Some(session_token),
                StatusCode::BAD_REQUEST,
            ),
            (
                "/v1/jobs?profile=a&profile=b",
                Some(session_token),
                StatusCode::BAD_REQUEST,
            ),
        ] {
            let mut request = Request::builder().uri(path);
            if let Some(token) = token {
                request = request.header("authorization", format!("Bearer {token}"));
            }
            let response = app
                .clone()
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), expected, "{path}");
        }
    }
    #[test]
    fn legacy_projection_preserves_results_and_supported_event_order() {
        // Exercise serialization against the actual legacy event variants.
        #[allow(dead_code)]
        #[derive(serde::Deserialize)]
        #[serde(tag = "kind", rename_all = "snake_case")]
        enum LegacyEvent {
            Bound {
                provider_session: String,
                model: Option<String>,
            },
            Text {
                text: String,
            },
            ToolStarted {
                id: String,
                name: String,
            },
            ToolCompleted {
                id: String,
                success: bool,
                details: Option<serde_json::Value>,
            },
        }
        let mut events = vec![
            RuntimeEvent::Text {
                text: "answer".into(),
            },
            RuntimeEvent::Reasoning {
                text: "thinking".into(),
            },
            RuntimeEvent::ToolStarted {
                id: "1".into(),
                name: "read".into(),
            },
        ];
        let value = serde_json::to_value(&events).unwrap();
        assert!(serde_json::from_value::<Vec<LegacyEvent>>(value).is_err());
        legacy_events(&mut events);
        assert_eq!(events.len(), 2);
        assert!(matches!(&events[0], RuntimeEvent::Text { text } if text == "answer"));
        assert!(matches!(&events[1], RuntimeEvent::ToolStarted { id, .. } if id == "1"));
        assert!(
            serde_json::from_value::<Vec<LegacyEvent>>(serde_json::to_value(events).unwrap())
                .is_ok()
        );
    }
}
