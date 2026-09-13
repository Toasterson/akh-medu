use akh_session_services::{
    config::{Config, Principal},
    server::{SessionApp, session_router},
    store::Store,
    types::*,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use std::sync::Arc;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{Message, client::IntoClientRequest},
};

async fn read(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
) -> Value {
    let frame = tokio::time::timeout(std::time::Duration::from_secs(5), socket.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    serde_json::from_str(frame.to_text().unwrap()).unwrap()
}
async fn send(
    socket: &mut tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    value: Value,
) {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .unwrap();
}

#[tokio::test]
async fn websocket_preserves_prompt_order_permissions_and_replays_before_load_response() {
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    std::fs::write(&token, "local-test-token-at-least-24-characters").unwrap();
    let config:Config=serde_json::from_value(json!({"service_id":"sessions","listen":"127.0.0.1:0","data_dir":dir.path(),"principals":[{"id":"user","roles":["client"],"akhs":["test"],"token_file":token}],"akhs":{"test":{"runtime":{"runtime":"runtime","profile":"test","revision":"1","kind":"codex","execution_mode":"colocated"},"instructions":"test","workspace_profiles":["local"]}}})).unwrap();
    let app = SessionApp::new(
        Arc::new(config),
        Store::open(&dir.path().join("store.redb")).unwrap(),
    )
    .unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/acp", listener.local_addr().unwrap());
    let router = session_router(app.clone());
    let server = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    assert!(connect_async(&url).await.is_err());
    let mut handshake = url.into_client_request().unwrap();
    handshake.headers_mut().insert(
        "authorization",
        "Bearer local-test-token-at-least-24-characters"
            .parse()
            .unwrap(),
    );
    let (mut socket, _) = connect_async(handshake.clone()).await.unwrap();
    send(
        &mut socket,
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}),
    )
    .await;
    assert_eq!(read(&mut socket).await["result"]["protocolVersion"], 1);
    send(&mut socket,json!({"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/ignored-client-path","mcpServers":[]}})).await;
    let response = read(&mut socket).await;
    let sid = response["result"]["sessionId"].as_str().unwrap().to_owned();
    app.sessions
        .bind_workspace(
            &sid,
            WorkspaceBinding {
                id: "w".into(),
                sandbox_id: "s".into(),
                executor: "exec".into(),
                generation: 1,
                site: "test".into(),
                root: "/remote/workspace".into(),
                platform: "test".into(),
                confinement: "test".into(),
            },
        )
        .unwrap();
    send(&mut socket,json!({"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":sid,"prompt":[{"type":"text","text":"hello"}],"_meta":{"requestId":"durable-prompt"}}})).await;
    let user = app.sessions.config.principals[0].clone();
    loop {
        let frame = read(&mut socket).await;
        if frame["params"]["update"]["sessionUpdate"] == "user_message_chunk" {
            break;
        }
    }
    let request = app.sessions.next_job(&sid).unwrap().unwrap();
    let runtime = Principal {
        id: "runtime".into(),
        token_file: "/unused".into(),
        token_sha256: false,
        roles: vec!["runtime".into()],
        akhs: vec![],
        sources: vec![],
    };
    app.sessions
        .request_input(
            &runtime,
            &ToolCall {
                id: "permission".into(),
                session_id: sid.clone(),
                turn_id: request.turn_id.clone(),
                generation: request.generation,
                name: "write".into(),
                arguments: json!({}),
            },
            "tool_approval",
            json!({"name":"write"}),
        )
        .unwrap();
    let permission = loop {
        let frame = read(&mut socket).await;
        if frame["method"] == "session/request_permission" {
            break frame;
        }
    };
    send(&mut socket,json!({"jsonrpc":"2.0","id":permission["id"],"result":{"outcome":{"outcome":"selected","optionId":"allow_once"}}})).await;
    for _ in 0..100 {
        if app.sessions.pending_inputs(&user, &sid).unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(app.sessions.pending_inputs(&user, &sid).unwrap().is_empty());
    app.sessions
        .apply_job(&RuntimeJob {
            request: request.clone(),
            status: TurnStatus::Completed,
            provider_session: Some("provider".into()),
            output: Some("world".into()),
            error: None,
            events: vec![RuntimeEvent::Text {
                text: "world".into(),
            }],
        })
        .unwrap();
    let mut saw_text = false;
    loop {
        let frame = read(&mut socket).await;
        if frame["params"]["update"]["content"]["text"] == "world" {
            saw_text = true;
        }
        if frame["id"] == 3 {
            assert!(saw_text);
            assert_eq!(frame["result"]["stopReason"], "end_turn");
            break;
        }
    }
    socket.close(None).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let (mut reconnected, _) = connect_async(handshake).await.unwrap();
    send(
        &mut reconnected,
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}),
    )
    .await;
    read(&mut reconnected).await;
    send(&mut reconnected,json!({"jsonrpc":"2.0","id":2,"method":"session/load","params":{"sessionId":sid,"cwd":"/irrelevant","mcpServers":[]}})).await;
    let mut replayed = false;
    loop {
        let frame = read(&mut reconnected).await;
        if frame["params"]["update"]["content"]["text"] == "world" {
            replayed = true;
        }
        if frame["id"] == 2 {
            assert!(replayed);
            break;
        }
    }
    reconnected.close(None).await.unwrap();
    server.abort();
}
