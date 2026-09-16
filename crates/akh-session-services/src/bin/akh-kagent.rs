//! Stateless released-kagent BYO facade. Durable tasks/sessions stay in akhsessd.
use akh_session_services::{config::Endpoint, http::segment, types::*};
use axum::{
    Json, Router,
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use clap::Parser;
use serde::Deserialize;
use serde_json::Value;
use std::{net::SocketAddr, path::PathBuf, sync::Arc, time::Duration};
#[path = "kagent/projection.rs"]
mod projection;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    listen: SocketAddr,
    sessions: Endpoint,
    akh: String,
    /// Cluster-local BYO Service URL advertised to kagent, without a path.
    public_url: String,
    /// Explicitly acknowledge that ingress NetworkPolicy is this hop's auth boundary.
    /// One facade represents one centrally scoped kagent integration principal.
    trust_kagent_network: bool,
    #[serde(default)]
    projection: Option<projection::Config>,
}
#[derive(Clone)]
struct App {
    config: Arc<Config>,
    client: reqwest::Client,
    projection: Option<Arc<projection::Projection>>,
}
impl App {
    fn new(config: Config) -> Result<Self> {
        config.sessions.validate()?;
        validate_key(&config.akh)?;
        let url = url::Url::parse(&config.public_url)
            .map_err(|_| Fault::invalid("invalid facade public_url"))?;
        if !config.trust_kagent_network
            || !matches!(url.scheme(), "http" | "https")
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
            || url.path() != "/"
        {
            return Err(Fault::invalid(
                "BYO facade requires an explicit trusted kagent network and an origin-only public_url",
            ));
        }
        let client = config.sessions.http_client()?;
        let projection = config
            .projection
            .clone()
            .map(projection::Projection::new)
            .transpose()?
            .map(Arc::new);
        Ok(Self {
            config: Arc::new(config),
            client,
            projection,
        })
    }
    fn route(&self) -> String {
        self.config
            .sessions
            .route(&format!("/a2a/{}", segment(&self.config.akh)))
    }
}
fn failure(status: StatusCode, message: &'static str) -> Response {
    (status, Json(serde_json::json!({"error":message}))).into_response()
}
async fn card(State(app): State<App>) -> Response {
    let token = match app.config.sessions.token() {
        Ok(t) => t,
        Err(_) => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "central credential unavailable",
            );
        }
    };
    let mut response = match app
        .client
        .get(format!("{}/.well-known/agent-card.json", app.route()))
        .bearer_auth(token)
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r,
        _ => {
            return failure(
                StatusCode::BAD_GATEWAY,
                "central agent discovery unavailable",
            );
        }
    };
    let mut bytes = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) if bytes.len() + chunk.len() <= 1024 * 1024 => {
                bytes.extend_from_slice(&chunk)
            }
            Ok(None) => break,
            _ => {
                return failure(
                    StatusCode::BAD_GATEWAY,
                    "invalid central discovery response",
                );
            }
        }
    }
    let mut card: Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(_) => return failure(StatusCode::BAD_GATEWAY, "invalid central agent card"),
    };
    if card["protocolVersion"] != "0.3.0" || card["name"] != app.config.akh {
        return failure(
            StatusCode::BAD_GATEWAY,
            "central agent identity/protocol differs",
        );
    }
    card["url"] = Value::String(app.config.public_url.clone());
    // This hop is trusted cluster networking; the central bearer is never exposed.
    if let Some(object) = card.as_object_mut() {
        object.remove("security");
        object.remove("securitySchemes");
        object.remove("additionalInterfaces");
        object.remove("supportedInterfaces");
    }
    Json(card).into_response()
}
async fn call(
    State(app): State<App>,
    headers: HeaderMap,
    Json(mut request): Json<Value>,
) -> Response {
    if headers.contains_key(header::ORIGIN) {
        return failure(
            StatusCode::FORBIDDEN,
            "browser calls use the authenticated central endpoint",
        );
    }
    if let Some(projection) = &app.projection {
        if headers.get("x-user-id").and_then(|h| h.to_str().ok())
            != Some(&projection.config.user_id)
        {
            return failure(
                StatusCode::FORBIDDEN,
                "this facade belongs to another kagent user",
            );
        }
    }
    if request["jsonrpc"] != "2.0"
        || request.get("id").is_none()
        || !matches!(
            request["method"].as_str(),
            Some(
                "message/send"
                    | "message/stream"
                    | "tasks/get"
                    | "tasks/cancel"
                    | "tasks/resubscribe"
            )
        )
    {
        return failure(StatusCode::BAD_REQUEST, "unsupported A2A request");
    }
    if let Some(projection) = &app.projection {
        if let Err(error) = projection.prepare(&app, &mut request).await {
            tracing::warn!(code=%error.code, reason=%error.message,"kagent context mapping pending");
            return failure(
                StatusCode::BAD_GATEWAY,
                "context mapping unconfirmed; retain original message identity",
            );
        }
    }
    let token = match app.config.sessions.token() {
        Ok(t) => t,
        Err(_) => {
            return failure(
                StatusCode::SERVICE_UNAVAILABLE,
                "central credential unavailable",
            );
        }
    };
    // Do not forward caller Authorization/identity headers. The fixed integration
    // principal is explicit, and no client may switch the target Akh or endpoint.
    // Request IDs/message IDs remain untouched. A lost response is never replayed.
    let response = match app
        .client
        .post(app.route())
        .bearer_auth(token)
        .json(&request)
        .timeout(Duration::from_secs(24 * 60 * 60))
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => {
            return failure(
                StatusCode::BAD_GATEWAY,
                "central response unconfirmed; retain original message identity",
            );
        }
    };
    let status = response.status();
    let content_type = response.headers().get(header::CONTENT_TYPE).cloned();
    let streaming = content_type
        .as_ref()
        .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"));
    if let Some(projection) = &app.projection {
        if !streaming {
            let mut value = match projection::body(response).await {
                Ok(value) => value,
                Err(_) => {
                    return failure(
                        StatusCode::BAD_GATEWAY,
                        "central response unconfirmed; retain original message identity",
                    );
                }
            };
            if let Some(id) = projection::task_id(&value) {
                match projection.publish(&app, id).await {
                    Ok(task) => {
                        value["result"] = task;
                    }
                    Err(error) => {
                        tracing::warn!(code=%error.code, reason=%error.message,"kagent projection pending after central admission");
                        return failure(
                            StatusCode::BAD_GATEWAY,
                            "kagent projection unconfirmed; retry the original message identity",
                        );
                    }
                }
            }
            return (status, [(header::CACHE_CONTROL, "no-store")], Json(value)).into_response();
        }
    }
    let mut response = response;
    let stream: std::pin::Pin<
        Box<dyn futures_util::Stream<Item = std::io::Result<axum::body::Bytes>> + Send>,
    > = Box::pin(async_stream::try_stream! {
        let mut size=0usize;
        let mut frame = Vec::new();
        loop {
            let chunk=tokio::time::timeout(Duration::from_secs(60),response.chunk()).await
                .map_err(|_|std::io::Error::new(std::io::ErrorKind::TimedOut,"central stream stalled"))?
                .map_err(|_|std::io::Error::other("central stream interrupted"))?;
            let Some(chunk)=chunk else {
                if !frame.is_empty() {Err(std::io::Error::other("central SSE frame was truncated"))?;}
                break;
            };
            size+=chunk.len();
            if size>64*1024*1024 { Err(std::io::Error::other("central stream exceeded connection budget; resume the same task"))?; }
            if let Some(projection) = &app.projection {
                frame.extend_from_slice(&chunk);
                while let Some(end) = frame.windows(2).position(|v| v == b"\n\n") {
                    let event: Vec<_> = frame.drain(..end+2).collect();
                    let mut translated=Vec::new();
                    for line in event.split_inclusive(|v| *v == b'\n') {
                        if let Some(data) = line.strip_prefix(b"data:") {
                            let mut value:Value=serde_json::from_slice(data).map_err(|_|std::io::Error::other("invalid central SSE event"))?;
                            if let Some(id)=projection::task_id(&value) {
                                let task=projection.publish(&app,id).await.map_err(|_|std::io::Error::other("kagent projection pending; resume the same task"))?;
                                projection::map_context(&mut value["result"],task["contextId"].as_str().unwrap(),task["metadata"]["akhSessionId"].as_str().unwrap());
                            }
                            translated.extend_from_slice(b"data: ");
                            translated.extend_from_slice(value.to_string().as_bytes());
                            translated.push(b'\n');
                        } else {
                            translated.extend_from_slice(line);
                        }
                    }
                    yield axum::body::Bytes::from(translated);
                }
                if frame.len()>2*1024*1024 { Err(std::io::Error::other("central SSE frame exceeded budget"))?; }
            } else {yield chunk;}
        }
    });
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    if let Some(content_type) = content_type {
        response
            .headers_mut()
            .insert(header::CONTENT_TYPE, content_type);
    }
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
}
fn router(app: App) -> Router {
    Router::new()
        .route("/", post(call))
        .route("/.well-known/agent-card.json", get(card))
        .route("/.well-known/agent.json", get(card))
        .layer(DefaultBodyLimit::max(2 * 1024 * 1024))
        .with_state(app)
}
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config: Config = serde_json::from_slice(&std::fs::read(args.config)?)?;
    let listen = config.listen;
    let app = App::new(config)?;
    tracing_subscriber::fmt().with_env_filter("warn").init();
    let projection_worker = app.projection.as_ref().map(|projection| {
        let projection = projection.clone();
        let app = app.clone();
        tokio::spawn(async move {
            loop {
                if let Err(error) = projection.refresh(&app).await {
                    tracing::warn!(code=%error.code, reason=%error.message,"kagent projection refresh pending");
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        })
    });
    let listener = tokio::net::TcpListener::bind(listen).await?;
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = axum::serve(listener, router(app)).with_graceful_shutdown(async {
        let _ = stopped.await;
    });
    let server = std::future::IntoFuture::into_future(server);
    tokio::pin!(server);
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        result = &mut server => { result?; },
        _ = async {tokio::select! {_=term.recv()=>{},_=tokio::signal::ctrl_c()=>{}}} => {
            let _ = stop.send(());
            if let Some(worker) = &projection_worker {worker.abort();}
            let _ = tokio::time::timeout(Duration::from_secs(10), &mut server).await;
        }
    }
    if let Some(worker) = projection_worker {
        worker.abort();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::to_bytes, http::Request};
    use serde_json::json;
    use tower::ServiceExt;
    #[tokio::test]
    async fn released_byo_paths_preserve_identity_and_only_forward_scoped_credentials() {
        let central = Router::new().route("/a2a/test/.well-known/agent-card.json",get(||async{Json(json!({"name":"test","protocolVersion":"0.3.0","url":"https://central/a2a/test","security":[{"bearer":[]}],"securitySchemes":{"bearer":{"type":"http","scheme":"bearer"}}}))}))
            .route("/a2a/test",post(|headers:HeaderMap,Json(value):Json<Value>|async move {
                assert_eq!(headers[header::AUTHORIZATION],"Bearer central-integration-token-123456");
                assert!(headers.get("x-user-id").is_none());
                assert_eq!(value["params"]["message"]["messageId"],"immutable-occurrence");
                assert_eq!(value["id"],"rpc-request");
                Json(json!({"jsonrpc":"2.0","id":value["id"],"result":{"id":"turn","contextId":"session","kind":"task","status":{"state":"submitted"}}}))
            }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, central).await.unwrap();
        });
        let dir = tempfile::tempdir().unwrap();
        let token = dir.path().join("token");
        std::fs::write(&token, "central-integration-token-123456").unwrap();
        let config = Config {
            listen: "127.0.0.1:0".parse().unwrap(),
            sessions: Endpoint {
                token_env: None,
                url: format!("http://{address}"),
                token_file: token,
                ca_file: None,
                allow_insecure_http: false,
            },
            akh: "test".into(),
            public_url: "http://akh-byo.kagent:8080/".into(),
            trust_kagent_network: true,
            projection: None,
        };
        let app = App::new(config.clone()).unwrap();
        let card = router(app.clone())
            .oneshot(
                Request::get("/.well-known/agent-card.json")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(card.status(), StatusCode::OK);
        let card: Value =
            serde_json::from_slice(&to_bytes(card.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(card["url"], config.public_url);
        assert!(card.get("security").is_none());
        let body=json!({"jsonrpc":"2.0","id":"rpc-request","method":"message/send","params":{"message":{"messageId":"immutable-occurrence"}}}).to_string();
        let result = router(app.clone())
            .oneshot(
                Request::post("/")
                    .header("content-type", "application/json")
                    .header("authorization", "untrusted-caller-header")
                    .header("x-user-id", "not-an-authenticated-identity")
                    .body(Body::from(body.clone()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(result.status(), StatusCode::OK);
        let value: Value =
            serde_json::from_slice(&to_bytes(result.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(value["result"]["contextId"], "session");
        let browser = router(app)
            .oneshot(
                Request::post("/")
                    .header("content-type", "application/json")
                    .header("origin", "https://browser")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(browser.status(), StatusCode::FORBIDDEN);
        let mut unsafe_config = config;
        unsafe_config.trust_kagent_network = false;
        assert!(App::new(unsafe_config).is_err());
        task.abort();
    }
}
