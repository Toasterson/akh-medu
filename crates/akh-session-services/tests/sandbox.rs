use akh_session_services::{config::Config, sandbox::Sandboxes, store::Store, types::*};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Kubernetes {
    object: Option<Value>,
    creates: usize,
    deleted: bool,
    delete_uid: Option<String>,
}
async fn read(State(state): State<Arc<Mutex<Kubernetes>>>) -> (StatusCode, Json<Value>) {
    match &state.lock().unwrap().object {
        Some(object) => (StatusCode::OK, Json(object.clone())),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"code":"not_found","message":"missing"})),
        ),
    }
}
async fn create(
    State(state): State<Arc<Mutex<Kubernetes>>>,
    Json(mut object): Json<Value>,
) -> Json<Value> {
    let mut state = state.lock().unwrap();
    state.creates += 1;
    object["metadata"]["uid"] = json!("original-uid");
    object["metadata"]["generation"] = json!(1);
    object["status"] =
        json!({"conditions":[{"type":"Ready","status":"True","observedGeneration":0}]});
    state.object = Some(object.clone());
    Json(object)
}
async fn delete(
    State(state): State<Arc<Mutex<Kubernetes>>>,
    Json(body): Json<Value>,
) -> (StatusCode, Json<Value>) {
    let mut state = state.lock().unwrap();
    state.delete_uid = body["preconditions"]["uid"].as_str().map(str::to_owned);
    if state.object.as_ref().unwrap()["metadata"]["uid"] != body["preconditions"]["uid"] {
        return (
            StatusCode::CONFLICT,
            Json(json!({"code":"conflict","message":"UID precondition failed"})),
        );
    }
    state.deleted = true;
    (StatusCode::OK, Json(json!({})))
}

#[tokio::test]
async fn kubernetes_retries_one_allocation_and_never_deletes_a_replacement_uid() {
    let state = Arc::new(Mutex::new(Kubernetes::default()));
    let resource = "/apis/agents.x-k8s.io/v1beta1/namespaces/work/sandboxes";
    let app = Router::new()
        .route(resource, post(create))
        .route(&format!("{resource}/{{id}}"), get(read).delete(delete))
        .route(
            "/v1/workspaces",
            post(|Json(request): Json<AttachWorkspace>| async move { Json(request.binding) }),
        )
        .route(
            "/v1/workspaces/{id}/release",
            post(|| async { Json(json!({})) }),
        )
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    std::fs::write(&token, "fixture-sandbox-token-0123456789").unwrap();
    let template = json!({
        "apiVersion":"agents.x-k8s.io/v1beta1", "kind":"Sandbox", "metadata":{},
        "spec":{"podTemplate":{"spec":{"containers":[{"name":"executor","image":"fixture"}]}}}
    });
    let config:Config=serde_json::from_value(json!({
        "service_id":"sandboxes", "listen":"127.0.0.1:0", "data_dir":dir.path(),
        "principals":[{"id":"sessions","token_file":token,"roles":["session"]}],
        "endpoints":{"k8s":{"url":url,"token_file":token},"executor":{"url":url,"token_file":token}},
        "workspaces":{"pod":{"revision":"1","site":"fixture","provisioner":{
            "kind":"kubernetes","api":"k8s","namespace":"work","resource":resource,
            "executor":"executor","template":template
        }}}
    })).unwrap();
    let service = Sandboxes::new(
        Arc::new(config),
        Store::open(&dir.path().join("sandbox.redb")).unwrap(),
    )
    .unwrap();
    let request = Allocate {
        session_id: "session".into(),
        principal: "user".into(),
        profile: "pod".into(),
    };
    // Ready from an older generation is not current provisioning evidence.
    assert_eq!(
        service.allocate(request.clone()).await.unwrap_err().code,
        "busy"
    );
    assert_eq!(state.lock().unwrap().creates, 1);
    state.lock().unwrap().object.as_mut().unwrap()["status"]["conditions"][0]["observedGeneration"] =
        json!(1);
    let first = service.allocate(request.clone()).await.unwrap();
    assert_eq!(first.state, "ready");
    assert_eq!(first.resource_uid.as_deref(), Some("original-uid"));
    assert_eq!(service.allocate(request).await.unwrap().id, first.id);
    assert_eq!(state.lock().unwrap().creates, 1);
    state.lock().unwrap().object.as_mut().unwrap()["metadata"]["uid"] = json!("replacement-uid");
    assert_eq!(
        service.release("session").await.unwrap_err().code,
        "conflict"
    );
    let state = state.lock().unwrap();
    assert_eq!(state.delete_uid.as_deref(), Some("original-uid"));
    assert!(!state.deleted);
    task.abort();
}
