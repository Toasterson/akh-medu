use akh_session_services::{
    config::{Config, Principal},
    server::{SessionApp, session_router},
    store::Store,
    types::*,
};
use axum::{
    body::{Body, to_bytes},
    http::Request,
};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;
async fn call(app: &SessionApp, method: &str, params: Value) -> Value {
    let response = session_router(app.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/a2a/test")
                .header(
                    "authorization",
                    "Bearer a2a-test-token-at-least-24-characters",
                )
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({"jsonrpc":"2.0","id":1,"method":method,"params":params}).to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap()).unwrap()
}
#[tokio::test]
async fn a2a_retries_questions_context_continuation_and_unknown_outcomes_keep_central_identity() {
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    std::fs::write(&token, "a2a-test-token-at-least-24-characters").unwrap();
    let config:Config=serde_json::from_value(json!({"service_id":"sessions","listen":"127.0.0.1:0","data_dir":dir.path(),"principals":[{"id":"kagent","roles":["client"],"akhs":["test"],"token_file":token}],"akhs":{"test":{"runtime":{"runtime":"runtime","profile":"test","revision":"1","kind":"codex","execution_mode":"colocated"},"instructions":"test","workspace_profiles":["local"]}}})).unwrap();
    let app = SessionApp::new(
        Arc::new(config),
        Store::open(&dir.path().join("state.redb")).unwrap(),
    )
    .unwrap();
    let params = json!({"message":{"kind":"message","role":"user","messageId":"first","parts":[{"kind":"text","text":"hello"}]}});
    let first = call(&app, "message/send", params.clone()).await;
    assert_eq!(first["result"]["status"]["state"], "submitted");
    assert_eq!(first, call(&app, "message/send", params).await);
    let task = first["result"]["id"].as_str().unwrap();
    let session = first["result"]["contextId"].as_str().unwrap();
    app.sessions
        .bind_workspace(
            session,
            WorkspaceBinding {
                id: "w".into(),
                sandbox_id: "s".into(),
                executor: "exec".into(),
                generation: 1,
                site: "test".into(),
                root: "/remote".into(),
                platform: "test".into(),
                confinement: "test".into(),
            },
        )
        .unwrap();
    let request = app.sessions.next_job(session).unwrap().unwrap();
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
                id: "approval".into(),
                session_id: session.into(),
                turn_id: task.into(),
                generation: request.generation,
                name: "write".into(),
                arguments: json!({}),
            },
            "tool_approval",
            json!({"name":"write"}),
        )
        .unwrap();
    let pending = call(&app, "tasks/get", json!({"id":task})).await;
    assert_eq!(pending["result"]["status"]["state"], "input-required");
    let answer = json!({"message":{"role":"user","messageId":"answer","contextId":session,"taskId":task,"parts":[{"kind":"data","data":{"inputId":"approval","response":{"approved":true}}}]}});
    let resumed = call(&app, "message/send", answer.clone()).await;
    assert_eq!(resumed["result"]["id"], task);
    assert_eq!(resumed["result"]["status"]["state"], "working");
    assert_eq!(resumed, call(&app, "message/send", answer).await);
    app.sessions
        .apply_job(&RuntimeJob {
            request,
            status: TurnStatus::Completed,
            provider_session: Some("native-session".into()),
            output: Some("done".into()),
            error: None,
            events: vec![],
        })
        .unwrap();
    assert!(call(&app,"message/send",json!({"message":{"role":"user","messageId":"restart","taskId":task,"parts":[{"kind":"text","text":"again"}]}})).await.get("error").is_some());
    let second=call(&app,"message/send",json!({"message":{"role":"user","messageId":"second","contextId":session,"parts":[{"kind":"text","text":"continue"}]}})).await;
    assert_eq!(second["result"]["contextId"], session);
    assert_ne!(second["result"]["id"], task);
    let request = app.sessions.next_job(session).unwrap().unwrap();
    assert_eq!(request.provider_session.as_deref(), Some("native-session"));
    app.sessions
        .apply_job(&RuntimeJob {
            request,
            status: TurnStatus::Unknown,
            provider_session: Some("native-session".into()),
            output: None,
            error: Some(Fault::new("unknown_outcome", "runtime disappeared")),
            events: vec![],
        })
        .unwrap();
    assert_eq!(
        call(&app, "tasks/get", json!({"id":second["result"]["id"]})).await["result"]["status"]["state"],
        "unknown"
    );
    assert!(app.sessions.next_job(session).unwrap().is_none());
}
