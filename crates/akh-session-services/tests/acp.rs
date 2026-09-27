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
            events: vec![
                RuntimeEvent::Reasoning {
                    text: "checking evidence".into(),
                },
                RuntimeEvent::Text {
                    text: "world".into(),
                },
            ],
        })
        .unwrap();
    let mut saw_text = false;
    let mut saw_reasoning = false;
    loop {
        let frame = read(&mut socket).await;
        if frame["params"]["update"]["content"]["text"] == "world" {
            saw_text = true;
        }
        if frame["params"]["update"]["sessionUpdate"] == "agent_thought_chunk" {
            saw_reasoning = frame["params"]["update"]["content"]["text"] == "checking evidence";
        }
        if frame["id"] == 3 {
            assert!(saw_text && saw_reasoning);
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

/// A realistic answer: five paragraphs (one with CRLF endings, some non-ASCII)
/// and a trailing paragraph without a boundary.
#[cfg(feature = "runtime")]
const ANSWER: &str = "The build failure comes from the feature flags, not from the code itself. \
The `runtime` feature pulls in the provider adapters, and the session daemon only enables it by default.\n\n\
I checked three places: the crate manifest, the workspace manifest and the CI job. \
Only the CI job disables default features, so its test binary never sees the runtime module.\r\n\r\n\
Zwei Optionen: entweder aktiviert der CI-Job das Feature explizit, oder der Test wird hinter \
`#[cfg(feature = \"runtime\")]` gestellt. Beides ist klein; das zweite ändert nichts am Build.\n\n\
- enable the feature in CI\n- gate the test\n- leave both as they are and accept the red job\n\n\
I recommend gating the test and filing a follow-up for the CI matrix — 変更は小さい。";

#[cfg(feature = "runtime")]
const THOUGHT: &str = "Read the manifest first.\n\nThen compare the CI flags.";

/// Split `text` into 3..=8 byte provider deltas (deterministic, char-safe).
#[cfg(feature = "runtime")]
fn provider_deltas(text: &str, seed: u64) -> Vec<&str> {
    let mut state = seed;
    let mut rest = text;
    let mut out = Vec::new();
    while !rest.is_empty() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut cut = (3 + (state >> 33) % 6) as usize;
        cut = cut.min(rest.len());
        while !rest.is_char_boundary(cut) {
            cut += 1;
        }
        out.push(&rest[..cut]);
        rest = &rest[cut..];
    }
    out
}

/// What one ACP client saw for one turn.
#[cfg(feature = "runtime")]
struct Observed {
    messages: Vec<String>,
    thoughts: Vec<String>,
    runtime_events: usize,
}

/// Drive one real turn: an internal runtime (akhrtd's `Runtime`, with the
/// given `[streaming]` section) streams `THOUGHT` and `ANSWER` from an SSE
/// model in 3..=8 byte deltas; akhsessd applies the job and an ACP client
/// records every `agent_message_chunk` / `agent_thought_chunk`.
#[cfg(feature = "runtime")]
async fn stream_turn_through_acp(streaming: Value) -> Observed {
    use akh_session_services::runtime::Runtime;
    let mut body = String::new();
    for delta in provider_deltas(THOUGHT, 7) {
        body.push_str(&format!(
            "data: {}\n\n",
            json!({"choices":[{"index":0,"delta":{"reasoning_content":delta},"finish_reason":null}]})
        ));
    }
    for delta in provider_deltas(ANSWER, 11) {
        body.push_str(&format!(
            "data: {}\n\n",
            json!({"choices":[{"index":0,"delta":{"content":delta},"finish_reason":null}]})
        ));
    }
    body.push_str(&format!(
        "data: {}\n\ndata: [DONE]\n\n",
        json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})
    ));
    let model = axum::Router::new().route(
        "/",
        axum::routing::post(move || {
            let body = body.clone();
            async move { ([("content-type", "text/event-stream")], body) }
        }),
    );
    let model_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let model_url = format!("http://{}/", model_listener.local_addr().unwrap());
    let model_server =
        tokio::spawn(async move { axum::serve(model_listener, model).await.unwrap() });

    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    std::fs::write(&token, "local-test-token-at-least-24-characters").unwrap();
    let binding = json!({"runtime":"runtime-1","profile":"local","revision":"1","kind":"internal","execution_mode":"brokered","model":"fixture"});
    let config: Config = serde_json::from_value(json!({"service_id":"sessions","listen":"127.0.0.1:0","data_dir":dir.path(),"principals":[{"id":"user","roles":["client"],"akhs":["test"],"token_file":token}],"akhs":{"test":{"runtime":binding,"instructions":"test","workspace_profiles":["local"]}}})).unwrap();
    let app = SessionApp::new(
        Arc::new(config),
        Store::open(&dir.path().join("store.redb")).unwrap(),
    )
    .unwrap();
    let runtime_dir = dir.path().join("runtime");
    std::fs::create_dir_all(&runtime_dir).unwrap();
    let runtime_config: Config = serde_json::from_value(json!({
        "service_id":"runtime-1","listen":"127.0.0.1:0","data_dir":runtime_dir,
        "principals":[{"id":"sessions","token_file":"/unused","roles":["session"],"akhs":[]}],
        "runtimes":{"local":{"revision":"1","kind":"internal","working_dir":runtime_dir,"binary":"/unused",
            "execution_mode":"brokered","timeout_secs":5,"isolation":{"directory":runtime_dir.join("profile")},
            "model_endpoint":{"url":model_url,"max_iterations":2,"max_output_tokens":128}}},
        "streaming":streaming
    }))
    .unwrap();
    runtime_config.validate().unwrap();
    let runtime_principal = runtime_config.principals[0].clone();
    let runtime = Runtime::new(
        Arc::new(runtime_config),
        Store::open(&runtime_dir.join("runtime.redb")).unwrap(),
    )
    .unwrap();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}/acp", listener.local_addr().unwrap());
    let router = session_router(app.clone());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let mut handshake = url.into_client_request().unwrap();
    handshake.headers_mut().insert(
        "authorization",
        "Bearer local-test-token-at-least-24-characters"
            .parse()
            .unwrap(),
    );
    let (mut socket, _) = connect_async(handshake).await.unwrap();
    send(
        &mut socket,
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1}}),
    )
    .await;
    read(&mut socket).await;
    send(&mut socket,json!({"jsonrpc":"2.0","id":2,"method":"session/new","params":{"cwd":"/ignored-client-path","mcpServers":[]}})).await;
    let sid = read(&mut socket).await["result"]["sessionId"]
        .as_str()
        .unwrap()
        .to_owned();
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
    send(&mut socket,json!({"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{"sessionId":sid,"prompt":[{"type":"text","text":"why is CI red?"}],"_meta":{"requestId":"stream-prompt"}}})).await;
    loop {
        let frame = read(&mut socket).await;
        if frame["params"]["update"]["sessionUpdate"] == "user_message_chunk" {
            break;
        }
    }
    let request = app.sessions.next_job(&sid).unwrap().unwrap();
    runtime.submit(&runtime_principal, request.clone()).unwrap();
    let job = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let job = runtime.get(&request.id).unwrap();
            if job.status.terminal() {
                break job;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(job.status, TurnStatus::Completed, "{:?}", job.error);
    assert_eq!(job.output.as_deref(), Some(ANSWER));
    app.sessions.apply_job(&job).unwrap();
    let mut observed = Observed {
        messages: vec![],
        thoughts: vec![],
        runtime_events: job.events.len(),
    };
    loop {
        let frame = read(&mut socket).await;
        let update = &frame["params"]["update"];
        let text = update["content"]["text"].as_str().map(str::to_owned);
        match update["sessionUpdate"].as_str() {
            Some("agent_message_chunk") => observed.messages.push(text.unwrap()),
            Some("agent_thought_chunk") => observed.thoughts.push(text.unwrap()),
            _ => {}
        }
        if frame["id"] == 3 {
            assert_eq!(frame["result"]["stopReason"], "end_turn");
            break;
        }
    }
    socket.close(None).await.unwrap();
    server.abort();
    model_server.abort();
    observed
}

#[cfg(feature = "runtime")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn paragraph_streaming_turns_token_deltas_into_paragraph_chunks_over_acp() {
    // Structural paragraphs: blank-line separated, whatever the line ending.
    let paragraphs = ANSWER.replace("\r\n", "\n").split("\n\n").count();
    assert_eq!(paragraphs, 5);
    let text_deltas = provider_deltas(ANSWER, 11).len();
    let thought_deltas = provider_deltas(THOUGHT, 7).len();

    let token = stream_turn_through_acp(json!({"mode":"token"})).await;
    assert_eq!(token.messages.len(), text_deltas);
    assert_eq!(token.thoughts.len(), thought_deltas);
    assert_eq!(token.messages.concat(), ANSWER);

    let paragraph = stream_turn_through_acp(json!({})).await;
    assert_eq!(paragraph.messages.concat(), ANSWER);
    assert_eq!(paragraph.thoughts.concat(), THOUGHT);
    assert_eq!(
        paragraph.messages.len(),
        paragraphs,
        "{:#?}",
        paragraph.messages
    );
    assert_eq!(paragraph.thoughts.len(), 2);
    for chunk in &paragraph.messages[..paragraphs - 1] {
        assert!(
            chunk.ends_with("\n\n") || chunk.ends_with("\r\n\r\n"),
            "{chunk:?}"
        );
    }
    eprintln!(
        "paragraph streaming: {} text + {} reasoning provider deltas ({} bytes) -> {} + {} ACP chunks; \
         runtime events {} -> {} ({:.1}x fewer)",
        text_deltas,
        thought_deltas,
        ANSWER.len() + THOUGHT.len(),
        paragraph.messages.len(),
        paragraph.thoughts.len(),
        token.runtime_events,
        paragraph.runtime_events,
        token.runtime_events as f64 / paragraph.runtime_events as f64,
    );
}
