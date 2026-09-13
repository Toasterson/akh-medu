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
            .route(
                "/v1/executors/{sandbox}/{*path}",
                axum::routing::any(executor_proxy),
            )
            .route("/v1/allocations", post(allocate))
            .route("/v1/allocations/{id}", get(allocation))
            .route("/v1/allocations/{id}/release", post(release_allocation))
            .route("/v1/allocations/{id}/renew", post(renew_allocation))
            .route("/v1/allocations/{id}/reconcile", post(reconcile_allocation))
            .with_state(service),
        config,
    )
}

/// Stable site gateway for executor RPCs. The original session credential and
/// receipt bytes cross unchanged; allocation management gains no execution authority.
async fn executor_proxy(
    State(service): State<Sandboxes>,
    Extension(principal): Extension<Principal>,
    Path((sandbox, path)): Path<(String, String)>,
    request: axum::extract::Request,
) -> std::result::Result<axum::response::Response, http::ApiError> {
    http::role(&principal, "executor_proxy")?;
    if sandbox.is_empty()
        || sandbox.len() > 63
        || !sandbox
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        || !path.starts_with("v1/")
        || path
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == "..")
        || !matches!(
            *request.method(),
            axum::http::Method::GET | axum::http::Method::POST
        )
    {
        return Err(Fault::invalid(
            "executor gateway requires a sandbox DNS label and a versioned RPC",
        )
        .into());
    }
    let allocation = service
        .store
        .scan::<Allocation>("allocation/")?
        .into_iter()
        .find(|a| {
            matches!(a.state.as_str(), "ready" | "retained")
                && a.binding.as_ref().is_some_and(|b| b.sandbox_id == sandbox)
        })
        .ok_or_else(|| Fault::missing("active sandbox binding not found"))?;
    let binding = allocation
        .binding
        .ok_or_else(|| Fault::missing("executor binding not found"))?;
    let endpoint = service.config.endpoint(&binding.executor)?;
    let path = path
        .split('/')
        .map(http::segment)
        .collect::<Vec<_>>()
        .join("/");
    let mut url = endpoint
        .route(&format!("/{path}"))
        .replace("{sandbox}", &sandbox);
    if let Some(query) = request.uri().query() {
        url.push('?');
        url.push_str(query);
    }
    let mut upstream = endpoint
        .http_client()?
        .request(request.method().clone(), url);
    // The downstream executor authenticates and checks the workspace/session itself.
    for name in [
        axum::http::header::AUTHORIZATION,
        axum::http::header::CONTENT_TYPE,
    ] {
        if let Some(value) = request.headers().get(&name) {
            upstream = upstream.header(name, value);
        }
    }
    let body = axum::body::to_bytes(request.into_body(), 2 * 1024 * 1024)
        .await
        .map_err(|_| Fault::invalid("executor RPC exceeds request limit"))?;
    let mut reply = upstream.body(body).send().await.map_err(|_| {
        Fault::new(
            "transport",
            "executor gateway request failed; effects may have occurred",
        )
    })?;
    let mut response = axum::http::Response::builder().status(reply.status());
    if let Some(value) = reply.headers().get(axum::http::header::CONTENT_TYPE) {
        response = response.header(axum::http::header::CONTENT_TYPE, value);
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = reply
        .chunk()
        .await
        .map_err(|_| Fault::new("transport", "incomplete executor receipt"))?
    {
        if bytes.len() + chunk.len() > 16 * 1024 * 1024 {
            return Err(Fault::new("transport", "executor receipt exceeds response limit").into());
        }
        bytes.extend_from_slice(&chunk);
    }
    response
        .body(axum::body::Body::from(bytes))
        .map_err(|_| Fault::invalid("invalid executor response").into())
}
async fn reconcile_allocation(
    State(service): State<Sandboxes>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
    Json(request): Json<AllocationResolution>,
) -> Api<Allocation> {
    Ok(Json(service.reconcile(&p, &id, &request).await?))
}
async fn renew_allocation(
    State(service): State<Sandboxes>,
    Extension(p): Extension<Principal>,
    Path(id): Path<String>,
) -> Api<Allocation> {
    http::role(&p, "session")?;
    Ok(Json(service.renew(&id).await?))
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

#[cfg(test)]
mod gateway_tests {
    use super::*;
    use axum::{
        body::{Body, to_bytes},
        http::{HeaderMap, Request, StatusCode},
    };
    use tower::ServiceExt;
    #[tokio::test]
    async fn gateway_preserves_executor_authority_and_rejects_unknown_bindings() {
        async fn execute(headers: HeaderMap, body: String) -> (StatusCode, String) {
            assert_eq!(
                headers["authorization"],
                "Bearer fixture-session-executor-0123456789"
            );
            assert_eq!(body, "{\"id\":\"stable-operation\"}");
            (
                StatusCode::CREATED,
                "{\"id\":\"stable-operation\",\"status\":\"completed\"}".into(),
            )
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let upstream = tokio::spawn(async move {
            axum::serve(
                listener,
                Router::new().route("/v1/operations", post(execute)),
            )
            .await
            .unwrap()
        });
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("sessions-executor");
        std::fs::write(&token, "fixture-session-executor-0123456789").unwrap();
        let sandbox_token = dir.path().join("sandbox-executor");
        std::fs::write(&sandbox_token, "fixture-sandbox-only-0123456789").unwrap();
        let config:Config=serde_json::from_value(json!({"service_id":"sandbox","listen":"127.0.0.1:0","data_dir":dir.path(),
            "principals":[{"id":"sessions-executor","token_file":token,"roles":["executor_proxy"]},{"id":"sandbox-controller","token_file":sandbox_token,"roles":["session"]}],
            "endpoints":{"executor":{"url":url,"token_file":sandbox_token}}})).unwrap();
        config.validate().unwrap();
        let store = Store::open(&dir.path().join("sandbox.redb")).unwrap();
        let allocation:Allocation=serde_json::from_value(json!({"id":"akh-known","request":{"session_id":"session","principal":"owner","profile":"workspace"},"profile_revision":"1","state":"ready","error":null,"lease_until":now_ms()+300000,
            "binding":{"id":"workspace","sandbox_id":"akh-known","executor":"executor","generation":1,"site":"archibald","root":"/work","platform":"linux","confinement":"fixture"}})).unwrap();
        store
            .transaction(|tx| tx.put(&crate::store::key("allocation", &["session"]), &allocation))
            .unwrap();
        let app = sandbox_router(Sandboxes::new(Arc::new(config), store).unwrap());
        for (path, credential, status) in [
            (
                "/v1/executors/akh-known/v1/operations",
                "fixture-session-executor-0123456789",
                StatusCode::CREATED,
            ),
            (
                "/v1/executors/akh-unknown/v1/operations",
                "fixture-session-executor-0123456789",
                StatusCode::NOT_FOUND,
            ),
            (
                "/v1/executors/akh-known/v1/operations",
                "fixture-sandbox-only-0123456789",
                StatusCode::FORBIDDEN,
            ),
            (
                "/v1/executors/akh-known/v1/operations",
                "invalid",
                StatusCode::UNAUTHORIZED,
            ),
        ] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .header("authorization", format!("Bearer {credential}"))
                        .header("content-type", "application/json")
                        .body(Body::from("{\"id\":\"stable-operation\"}"))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            if status == StatusCode::CREATED {
                assert_eq!(
                    to_bytes(response.into_body(), 1024).await.unwrap(),
                    "{\"id\":\"stable-operation\",\"status\":\"completed\"}"
                );
            }
        }
        upstream.abort();
    }
}
