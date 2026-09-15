mod common;
use base64::{Engine, engine::general_purpose::STANDARD};
use common::{Fixture, TOKEN};
use rmcp::{
    RoleClient, ServiceExt,
    model::{CallToolRequestParams, CallToolResult},
    service::RunningService,
    transport::{
        StreamableHttpClientTransport, streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde_json::{Value, json};
use sokar::{
    Sokar,
    config::{Config, PrincipalConfig},
};
use std::{sync::Arc, time::Duration};

type Peer = RunningService<RoleClient, ()>;
struct Server {
    url: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn serve(f: &Fixture, change: impl FnOnce(&mut Config)) -> Server {
    serve_authorized(f, change, Arc::new(sokar::authz::ServiceOnly)).await
}
async fn serve_authorized(
    f: &Fixture,
    change: impl FnOnce(&mut Config),
    authorizer: Arc<dyn sokar::authz::Authorizer>,
) -> Server {
    let mut config = (*f.config).clone();
    change(&mut config);
    config.validate().unwrap();
    let path = config.mcp.path.clone();
    let mut state = Sokar::new(
        Arc::new(config),
        f.state.stores.clone(),
        f.state.index.clone(),
    );
    state.authorizer = authorizer;
    let state = Arc::new(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}{path}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, sokar::http::api(state))
            .await
            .unwrap();
    });
    Server { url, task }
}
async fn connect(server: &Server, token: &str) -> Peer {
    let transport = StreamableHttpClientTransport::from_config(
        StreamableHttpClientTransportConfig::with_uri(server.url.clone()).auth_header(token),
    );
    tokio::time::timeout(Duration::from_secs(10), ().serve(transport))
        .await
        .unwrap()
        .unwrap()
}
async fn call(peer: &Peer, name: &str, args: Value) -> CallToolResult {
    tokio::time::timeout(
        Duration::from_secs(10),
        peer.call_tool(CallToolRequestParams {
            name: name.to_owned().into(),
            arguments: args.as_object().cloned(),
            meta: None,
            task: None,
        }),
    )
    .await
    .unwrap()
    .unwrap()
}
fn data(result: CallToolResult) -> Value {
    assert_ne!(result.is_error, Some(true), "{result:?}");
    serde_json::from_str(&result.content[0].as_text().unwrap().text).unwrap()
}
fn error(result: CallToolResult, code: &str) -> Value {
    assert_eq!(result.is_error, Some(true), "{result:?}");
    let value: Value = serde_json::from_str(&result.content[0].as_text().unwrap().text).unwrap();
    assert_eq!(value["code"], code);
    assert!(!value["help"].as_str().unwrap().is_empty());
    value
}
async fn collection(peer: &Peer) -> Value {
    data(
        call(
            peer,
            "sokar_collection",
            json!({"scope":{"type":"named","name":"mcp-contract"}}),
        )
        .await,
    )
}

#[tokio::test]
async fn streamable_http_all_tools_roundtrip_and_schemas() {
    let f = Fixture::new(true).await;
    let server = serve(&f, |_| {}).await;
    let peer = connect(&server, TOKEN).await;
    let info = peer.peer_info().unwrap();
    assert!(info.instructions.as_ref().unwrap().contains("latest-wins"));
    let tools = peer.list_all_tools().await.unwrap();
    let mut names: Vec<_> = tools.iter().map(|tool| tool.name.as_ref()).collect();
    names.sort();
    assert_eq!(
        names,
        [
            "sokar_collection",
            "sokar_delete",
            "sokar_find",
            "sokar_get",
            "sokar_list",
            "sokar_put_base64",
            "sokar_put_text",
            "sokar_read_base64",
            "sokar_read_text"
        ]
    );
    for tool in &tools {
        assert_eq!(tool.input_schema["type"], "object");
        for field in tool.input_schema["properties"]
            .as_object()
            .unwrap()
            .values()
        {
            assert!(
                field["description"].as_str().is_some_and(|s| !s.is_empty()),
                "{tool:?}"
            );
        }
    }
    let scope_schema = tools.iter().find(|t| t.name == "sokar_collection").unwrap();
    let schema = serde_json::to_string(&scope_schema.input_schema).unwrap();
    for variant in [
        "named",
        "akh_workspace",
        "akh_session",
        "anima_session",
        "anima_work_item",
    ] {
        assert!(schema.contains(variant));
    }
    let text_schema = &tools
        .iter()
        .find(|t| t.name == "sokar_put_text")
        .unwrap()
        .input_schema;
    assert_eq!(
        text_schema["properties"]["media_type"]["default"],
        "text/plain; charset=utf-8"
    );
    assert_eq!(text_schema["properties"]["kind"]["default"], "document");
    let c = collection(&peer).await;
    assert_eq!(c["owner"], "test-service");
    assert_eq!(collection(&peer).await["id"], c["id"]);
    let text = "Sokar: héllo 🌍\n";
    let put = data(call(&peer, "sokar_put_text", json!({"collection_id":c["id"],"path":"report.txt","text":text,"labels":{"purpose":"test"}})).await);
    assert_eq!(put["outcome"], "stored");
    assert_eq!(put["artifact"]["created_by"], "test-service");
    assert_eq!(put["artifact"]["kind"], "document");
    assert_eq!(
        put["artifact"]["blob"]["media_type"],
        "text/plain; charset=utf-8"
    );
    let id = &put["artifact"]["id"];
    let found = data(
        call(
            &peer,
            "sokar_find",
            json!({"collection_id":c["id"],"path":"report.txt"}),
        )
        .await,
    );
    assert_eq!(found, put["artifact"]);
    assert_eq!(
        data(call(&peer, "sokar_read_text", json!({"artifact_id":id})).await),
        text
    );
    let metadata = data(call(&peer, "sokar_get", json!({"artifact_id":id})).await);
    assert_eq!(metadata["artifact"], found);
    assert_eq!(
        metadata["content_url"],
        format!("/v1/artifacts/{}/content", id.as_str().unwrap())
    );
    let bytes = [0, 254, 255];
    let binary = data(call(&peer, "sokar_put_base64", json!({"collection_id":c["id"],"path":"file.bin","base64":STANDARD.encode(bytes),"media_type":"application/octet-stream"})).await);
    assert_eq!(binary["artifact"]["kind"], "other");
    let binary_id = &binary["artifact"]["id"];
    let read = data(call(&peer, "sokar_read_base64", json!({"artifact_id":binary_id})).await);
    assert_eq!(
        read,
        json!({"base64":STANDARD.encode(bytes),"media_type":"application/octet-stream","size":3})
    );
    let non_utf8 = error(
        call(&peer, "sokar_read_text", json!({"artifact_id":binary_id})).await,
        "sokar::not_utf8",
    );
    assert!(
        non_utf8["help"]
            .as_str()
            .unwrap()
            .contains("sokar_read_base64")
    );
    let newer = data(
        call(
            &peer,
            "sokar_put_text",
            json!({"collection_id":c["id"],"path":"report.txt","text":text}),
        )
        .await,
    );
    assert_eq!(newer["outcome"], "deduplicated");
    assert_eq!(newer["artifact"]["previous"], *id);
    assert_eq!(
        data(
            call(
                &peer,
                "sokar_find",
                json!({"collection_id":c["id"],"path":"report.txt"})
            )
            .await
        )["id"],
        newer["artifact"]["id"]
    );
    let page = data(
        call(
            &peer,
            "sokar_list",
            json!({"collection_id":c["id"],"prefix":"report","kind":"document","limit":1}),
        )
        .await,
    );
    assert_eq!(page["artifacts"].as_array().unwrap().len(), 1);
    assert!(page["next_cursor"].is_string());
    let next = data(call(&peer, "sokar_list", json!({"collection_id":c["id"],"prefix":"report","kind":"document","limit":1,"cursor":page["next_cursor"]})).await);
    assert_eq!(next["artifacts"].as_array().unwrap().len(), 1);
    assert_ne!(page["artifacts"][0]["id"], next["artifacts"][0]["id"]);
    assert_eq!(
        data(call(&peer, "sokar_delete", json!({"artifact_id":id})).await),
        json!({"tombstoned":true})
    );
    error(
        call(&peer, "sokar_get", json!({"artifact_id":id})).await,
        "sokar::not_found",
    );
    error(
        call(
            &peer,
            "sokar_find",
            json!({"collection_id":c["id"],"path":"missing"}),
        )
        .await,
        "sokar::not_found",
    );
    peer.cancel().await.unwrap();
}

#[tokio::test]
async fn inline_limits_are_decoded_exact_and_apply_to_reads_and_text() {
    let f = Fixture::new(true).await;
    let server = serve(&f, |config| config.mcp.max_inline_bytes = 3).await;
    let peer = connect(&server, TOKEN).await;
    let c = collection(&peer).await;
    for text in ["four", "🌍"] {
        let e = error(
            call(
                &peer,
                "sokar_put_text",
                json!({"collection_id":c["id"],"path":"too-big","text":text}),
            )
            .await,
            "sokar::inline_limit",
        );
        assert!(e["message"].as_str().unwrap().contains("3 bytes"));
    }
    let e = error(call(&peer, "sokar_put_base64", json!({"collection_id":c["id"],"path":"too-big","base64":STANDARD.encode([1,2,3,4]),"media_type":"application/octet-stream"})).await, "sokar::inline_limit");
    assert!(e["message"].as_str().unwrap().contains("3 bytes"));
    error(call(&peer, "sokar_put_base64", json!({"collection_id":c["id"],"path":"bad","base64":"!!!!","media_type":"application/octet-stream"})).await, "sokar::request");
    let put = data(
        call(
            &peer,
            "sokar_put_base64",
            json!({"collection_id":c["id"],"path":"ok","base64":"YWJj","media_type":"text/plain"}),
        )
        .await,
    );
    let id = &put["artifact"]["id"];
    assert_eq!(
        data(
            call(
                &peer,
                "sokar_read_text",
                json!({"artifact_id":id,"max_bytes":3})
            )
            .await
        ),
        "abc"
    );
    for name in ["sokar_read_text", "sokar_read_base64"] {
        let e = error(
            call(&peer, name, json!({"artifact_id":id,"max_bytes":2})).await,
            "sokar::inline_limit",
        );
        assert!(e["message"].as_str().unwrap().contains("2 bytes"));
    }
    // REST can create larger artifacts; increasing a tool's requested limit
    // cannot bypass the server's configured cap.
    let client = sokar_client::Client::new(server.url.trim_end_matches("/mcp"), TOKEN).unwrap();
    let collection_id = c["id"].as_str().unwrap().parse().unwrap();
    let options = sokar_client::UploadOptions {
        path: "rest.txt".parse().unwrap(),
        media_type: "text/plain".parse().unwrap(),
        kind: Default::default(),
        labels: Default::default(),
        idempotency_key: None,
    };
    let big = client
        .upload(collection_id, &options, bytes::Bytes::from_static(b"four"))
        .await
        .unwrap();
    error(
        call(
            &peer,
            "sokar_read_base64",
            json!({"artifact_id":big.artifact.id,"max_bytes":1000}),
        )
        .await,
        "sokar::inline_limit",
    );
    peer.cancel().await.unwrap();
}

#[tokio::test]
async fn auth_precedes_mcp_custom_path_and_public_url_work() {
    let f = Fixture::new(true).await;
    let server = serve(&f, |config| {
        config.mcp.path = "/agent/tools".parse().unwrap();
        config.server.public_api_url = Some("https://sokar.example/prefix/".parse().unwrap());
    })
    .await;
    let http = reqwest::Client::new();
    for token in [None, Some("invalid")] {
        let mut request = http.post(&server.url).body("not MCP JSON");
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.unwrap();
        assert_eq!(response.status(), 401);
        assert_eq!(
            response.headers()["content-type"],
            "application/problem+json"
        );
        assert_eq!(
            response.json::<Value>().await.unwrap()["code"],
            "sokar::authn"
        );
    }
    let peer = connect(&server, TOKEN).await;
    let c = collection(&peer).await;
    let put = data(
        call(
            &peer,
            "sokar_put_text",
            json!({"collection_id":c["id"],"path":"text","text":""}),
        )
        .await,
    );
    let metadata = data(
        call(
            &peer,
            "sokar_get",
            json!({"artifact_id":put["artifact"]["id"]}),
        )
        .await,
    );
    assert_eq!(
        metadata["content_url"],
        format!(
            "https://sokar.example/prefix/v1/artifacts/{}/content",
            put["artifact"]["id"].as_str().unwrap()
        )
    );
    let no_route = http
        .post(server.url.replace("/agent/tools", "/mcp"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(no_route.status(), 404);
    peer.cancel().await.unwrap();
    let disabled = serve(&f, |config| config.mcp.enabled = false).await;
    assert_eq!(
        http.post(&disabled.url)
            .bearer_auth(TOKEN)
            .send()
            .await
            .unwrap()
            .status(),
        404
    );
}

#[tokio::test]
async fn sessions_are_principal_bound_and_credentials_reload() {
    let f = Fixture::new(true).await;
    const OTHER: &str = "another-service-token-at-least-24";
    const ROTATED: &str = "rotated-service-token-at-least-24";
    let token_path = f.dir.path().join("other-token");
    std::fs::write(&token_path, OTHER).unwrap();
    let server = serve(&f, |config| {
        config.principals.push(PrincipalConfig {
            id: "other-service".into(),
            token_file: Some(token_path),
            token_sha256_file: None,
        })
    })
    .await;
    let other = connect(&server, OTHER).await;
    assert_eq!(collection(&other).await["owner"], "other-service");
    let http = reqwest::Client::new();
    let response = http.post(&server.url).bearer_auth(TOKEN)
        .header("accept", "application/json, text/event-stream")
        .json(&json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1"}}})).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let session = response.headers()["mcp-session-id"]
        .to_str()
        .unwrap()
        .to_owned();
    let _ = response.text().await.unwrap();
    let response = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "application/json, text/event-stream")
        .header("mcp-session-id", &session)
        .json(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 202);
    let list = json!({"jsonrpc":"2.0","id":2,"method":"tools/list"});
    for method in [reqwest::Method::POST, reqwest::Method::GET] {
        let response = http
            .request(method, &server.url)
            .bearer_auth(OTHER)
            .header("accept", "application/json, text/event-stream")
            .header("mcp-session-id", &session)
            .json(&list)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 404);
    }
    // A cross-principal DELETE cannot close the victim's session, regardless
    // of the transport's response for unknown session IDs.
    let _ = http
        .delete(&server.url)
        .bearer_auth(OTHER)
        .header("mcp-session-id", &session)
        .send()
        .await
        .unwrap();
    std::fs::write(f.config.principals[0].token_file.as_ref().unwrap(), ROTATED).unwrap();
    let denied = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("mcp-session-id", &session)
        .json(&list)
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), 401);
    let allowed = http
        .post(&server.url)
        .bearer_auth(ROTATED)
        .header("accept", "application/json, text/event-stream")
        .header("mcp-session-id", &session)
        .json(&list)
        .send()
        .await
        .unwrap();
    assert_eq!(allowed.status(), 200);
    assert!(allowed.text().await.unwrap().contains("sokar_put_text"));
    http.delete(&server.url)
        .bearer_auth(ROTATED)
        .header("mcp-session-id", &session)
        .send()
        .await
        .unwrap();
    other.cancel().await.unwrap();
}

#[tokio::test]
async fn idle_sessions_expire_and_can_initialize_again() {
    let f = Fixture::new(true).await;
    let server = serve(&f, |config| config.mcp.session_idle_secs = 1).await;
    let http = reqwest::Client::new();
    let initialize = json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"expiry-test","version":"1"}}});
    let response = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "application/json, text/event-stream")
        .json(&initialize)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let session = response.headers()["mcp-session-id"]
        .to_str()
        .unwrap()
        .to_owned();
    response.text().await.unwrap();
    let response = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "application/json, text/event-stream")
        .header("mcp-session-id", &session)
        .json(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 202);
    tokio::time::sleep(Duration::from_millis(1500)).await;
    let expired = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "application/json, text/event-stream")
        .header("mcp-session-id", &session)
        .json(&json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}))
        .send()
        .await
        .unwrap();
    assert_eq!(expired.status(), 404);
    assert_eq!(
        expired.text().await.unwrap(),
        "Not Found: Session not found"
    );
    let fresh = connect(&server, TOKEN).await;
    assert_eq!(fresh.list_all_tools().await.unwrap().len(), 9);
    fresh.cancel().await.unwrap();
}

#[tokio::test]
async fn mcp_body_limit_precedes_parsing_and_transport_errors_survive() {
    let f = Fixture::new(true).await;
    let server = serve(&f, |config| {
        config.mcp.path = "/agents/sokar".parse().unwrap();
        config.mcp.max_inline_bytes = 3;
    })
    .await;
    let http = reqwest::Client::new();
    // Not valid JSON, but the Content-Length boundary must reject before rmcp parsing.
    let oversized = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "application/json, text/event-stream")
        .header("content-type", "application/json")
        .body("!".repeat(3 * 4 / 3 + 64 * 1024 + 1))
        .send()
        .await
        .unwrap();
    assert_eq!(oversized.status(), 413);
    // Unknown-length bodies must also be bounded before JSON parsing.
    let chunks = futures::stream::iter(vec![Ok::<_, std::io::Error>(bytes::Bytes::from(
        vec![b'!'; 65_541],
    ))]);
    let chunked = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "application/json, text/event-stream")
        .header("content-type", "application/json")
        .body(reqwest::Body::wrap_stream(chunks))
        .send()
        .await
        .unwrap();
    assert_eq!(chunked.status(), 413);
    let missing_get = http
        .get(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "text/event-stream")
        .send()
        .await
        .unwrap();
    assert_eq!(missing_get.status(), 400);
    assert_eq!(
        missing_get.text().await.unwrap(),
        "Bad Request: Session ID is required"
    );
    let unsupported = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "application/json, text/event-stream")
        .body("wrong content type")
        .send()
        .await
        .unwrap();
    assert_eq!(unsupported.status(), 415);
    assert_eq!(
        unsupported.text().await.unwrap(),
        "Unsupported Media Type: Content-Type must be application/json"
    );
    let missing = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "application/json, text/event-stream")
        .json(&json!({"jsonrpc":"2.0","id":1,"method":"tools/list"}))
        .send()
        .await
        .unwrap();
    assert_eq!(missing.status(), 422);
    assert_eq!(
        missing.text().await.unwrap(),
        "Unexpected message, expect initialize request"
    );
    let unacceptable = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .header("accept", "text/plain")
        .json(&json!({}))
        .send()
        .await
        .unwrap();
    assert_eq!(unacceptable.status(), 406);
    assert!(!unacceptable.text().await.unwrap().contains("urn:sokar"));
    let no_route = http
        .post(server.url.replace("/agents/sokar", "/agents/sokar-other"))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(no_route.status(), 404);
    assert_eq!(
        no_route.headers()["content-type"],
        "application/problem+json"
    );
}

#[tokio::test]
async fn upload_idempotency_is_shared_with_rest() {
    let f = Fixture::new(true).await;
    let server = serve(&f, |_| {}).await;
    let peer = connect(&server, TOKEN).await;
    let c = collection(&peer).await;
    let rest = sokar_client::Client::new(server.url.trim_end_matches("/mcp"), TOKEN).unwrap();
    for (tool, field, value, changed) in [
        ("sokar_put_text", "text", "abc", "def"),
        ("sokar_put_base64", "base64", "YWJj", "ZGVm"),
    ] {
        let mut args = json!({"collection_id":c["id"],"path":tool,"media_type":"text/plain","kind":"document","idempotency_key":tool});
        args[field] = json!(value);
        let first = data(call(&peer, tool, args.clone()).await);
        let retry = data(call(&peer, tool, args.clone()).await);
        assert_eq!(first["artifact"]["id"], retry["artifact"]["id"]);
        let mut options = common::options(tool);
        options.media_type = "text/plain".parse().unwrap();
        options.idempotency_key = Some(tool.into());
        let rest_retry = rest
            .upload(
                c["id"].as_str().unwrap().parse().unwrap(),
                &options,
                bytes::Bytes::from_static(b"abc"),
            )
            .await
            .unwrap();
        assert_eq!(first["artifact"]["id"], rest_retry.artifact.id.to_string());
        args[field] = json!(changed);
        error(call(&peer, tool, args).await, "sokar::idempotency_mismatch");
    }
    peer.cancel().await.unwrap();
}

#[tokio::test]
async fn list_default_and_limit_validation_match_rest() {
    let f = Fixture::new(true).await;
    let server = serve(&f, |_| {}).await;
    let peer = connect(&server, TOKEN).await;
    let c = collection(&peer).await;
    for i in 0..101 {
        data(
            call(
                &peer,
                "sokar_put_text",
                json!({"collection_id":c["id"],"path":format!("item-{i:03}"),"text":"x"}),
            )
            .await,
        );
    }
    let rest = sokar_client::Client::new(server.url.trim_end_matches("/mcp"), TOKEN).unwrap();
    let id = c["id"].as_str().unwrap().parse().unwrap();
    for limit in [None, Some(1), Some(1000)] {
        let mcp = data(
            call(
                &peer,
                "sokar_list",
                json!({"collection_id":c["id"],"limit":limit}),
            )
            .await,
        );
        let page = rest
            .list_artifacts(
                id,
                &sokar_client::ListArtifacts {
                    limit,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(mcp, serde_json::to_value(&page).unwrap());
        if limit.is_none() {
            assert_eq!(mcp["artifacts"].as_array().unwrap().len(), 100);
        }
    }
    for limit in [0, 1001] {
        error(
            call(
                &peer,
                "sokar_list",
                json!({"collection_id":c["id"],"limit":limit}),
            )
            .await,
            "sokar::request",
        );
    }
    peer.cancel().await.unwrap();
}

#[derive(Default)]
struct RecordingAuthorizer {
    calls: std::sync::Mutex<Vec<(String, sokar::authz::Permission)>>,
    deny_resources: std::sync::atomic::AtomicBool,
    deny_endpoint: std::sync::atomic::AtomicBool,
}
impl sokar::authz::Authorizer for RecordingAuthorizer {
    fn authorize(
        &self,
        _: &sokar::authz::Principal,
        resource: &str,
        permission: sokar::authz::Permission,
    ) -> sokar::SokarResult<()> {
        if self.deny_endpoint.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(sokar::SokarError::Authz);
        }
        if resource.starts_with("/v1/") {
            self.calls
                .lock()
                .unwrap()
                .push((resource.into(), permission));
            if self
                .deny_resources
                .load(std::sync::atomic::Ordering::SeqCst)
            {
                return Err(sokar::SokarError::Authz);
            }
        }
        Ok(())
    }
}

#[tokio::test]
async fn resource_permissions_match_rest_and_upload_auth_precedes_content_work() {
    use sokar::authz::Permission;
    let f = Fixture::new(true).await;
    let auth = Arc::new(RecordingAuthorizer::default());
    let server = serve_authorized(&f, |config| config.mcp.max_inline_bytes = 3, auth.clone()).await;
    let peer = connect(&server, TOKEN).await;
    let c = collection(&peer).await;
    let path = "nested/héllo ?#.txt";
    let put = data(
        call(
            &peer,
            "sokar_put_text",
            json!({"collection_id":c["id"],"path":path,"text":"abc"}),
        )
        .await,
    );
    auth.calls.lock().unwrap().clear();
    data(call(&peer, "sokar_list", json!({"collection_id":c["id"]})).await);
    data(
        call(
            &peer,
            "sokar_find",
            json!({"collection_id":c["id"],"path":path}),
        )
        .await,
    );
    data(
        call(
            &peer,
            "sokar_get",
            json!({"artifact_id":put["artifact"]["id"]}),
        )
        .await,
    );
    data(
        call(
            &peer,
            "sokar_read_text",
            json!({"artifact_id":put["artifact"]["id"]}),
        )
        .await,
    );
    let mcp_calls = std::mem::take(&mut *auth.calls.lock().unwrap());
    assert_eq!(mcp_calls.len(), 4);
    assert!(mcp_calls.iter().all(|(_, p)| *p == Permission::Read));
    let http = reqwest::Client::new();
    for (resource, _) in &mcp_calls {
        let response = http
            .get(format!("{}{resource}", server.url.trim_end_matches("/mcp")))
            .bearer_auth(TOKEN)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{resource}");
    }
    assert_eq!(*auth.calls.lock().unwrap(), mcp_calls);
    assert!(mcp_calls[1].0.contains("h%C3%A9llo%20%3F%23.txt"));
    auth.deny_resources
        .store(true, std::sync::atomic::Ordering::SeqCst);
    for text in ["four", "abc"] {
        error(
            call(
                &peer,
                "sokar_put_text",
                json!({"collection_id":c["id"],"path":"denied","text":text}),
            )
            .await,
            "sokar::authz",
        );
    }
    for base64 in ["!!!!", "YWJjZGVm"] {
        error(call(&peer, "sokar_put_base64", json!({"collection_id":c["id"],"path":"denied","media_type":"text/plain","base64":base64})).await, "sokar::authz");
    }
    let denied = http
        .get(format!(
            "{}{}",
            server.url.trim_end_matches("/mcp"),
            mcp_calls[0].0
        ))
        .bearer_auth(TOKEN)
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), 403);
    assert_eq!(denied.headers()["content-type"], "application/problem+json");
    assert_eq!(
        denied.json::<Value>().await.unwrap()["code"],
        "sokar::authz"
    );
    auth.deny_endpoint
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let denied = http
        .post(&server.url)
        .bearer_auth(TOKEN)
        .body("invalid JSON")
        .send()
        .await
        .unwrap();
    assert_eq!(denied.status(), 403);
    assert_eq!(denied.headers()["content-type"], "application/problem+json");
    assert_eq!(
        denied.json::<Value>().await.unwrap()["code"],
        "sokar::authz"
    );
    auth.deny_endpoint
        .store(false, std::sync::atomic::Ordering::SeqCst);
    peer.cancel().await.unwrap();
}
