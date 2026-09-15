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
    let mut config = (*f.config).clone();
    change(&mut config);
    config.validate().unwrap();
    let path = config.mcp.path.clone();
    let state = Arc::new(Sokar::new(
        Arc::new(config),
        f.state.stores.clone(),
        f.state.index.clone(),
    ));
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
        config.mcp.path = "/agent/tools".into();
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
