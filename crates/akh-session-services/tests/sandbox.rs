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
    fail_create: bool,
    pending: Option<Value>,
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
) -> (StatusCode, Json<Value>) {
    let mut state = state.lock().unwrap();
    state.creates += 1;
    object["metadata"]["uid"] = json!("original-uid");
    object["metadata"]["generation"] = json!(1);
    object["status"] =
        json!({"conditions":[{"type":"Ready","status":"True","observedGeneration":0}]});
    if state.fail_create {
        state.pending = Some(object);
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({"code":"transport","message":"create response lost"})),
        );
    }
    state.object = Some(object.clone());
    (StatusCode::OK, Json(object))
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
        bootstrap: None,
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
    {
        let state = state.lock().unwrap();
        assert!(
            state.delete_uid.is_none(),
            "a replacement must be rejected before deletion"
        );
        assert!(!state.deleted);
    }
    // Cancellation while the real controller is still provisioning: no binding
    // exists yet, but the owned CRD still needs foreground cleanup.
    state.lock().unwrap().object = None;
    let pending = Allocate {
        bootstrap: None,
        session_id: "pending".into(),
        principal: "user".into(),
        profile: "pod".into(),
    };
    assert_eq!(
        service.allocate(pending.clone()).await.unwrap_err().code,
        "busy"
    );
    assert!(service.get("pending").unwrap().binding.is_none());
    assert_eq!(service.release("pending").await.unwrap_err().code, "busy");
    assert_eq!(service.get("pending").unwrap().state, "releasing");
    assert_eq!(
        state.lock().unwrap().delete_uid.as_deref(),
        Some("original-uid")
    );
    assert_eq!(
        service.allocate(pending).await.unwrap_err().code,
        "conflict"
    );
    state.lock().unwrap().object = None; // Controller completed foreground deletion.
    assert_eq!(service.release("pending").await.unwrap().state, "released");

    // A missing object after an uncertain create is not permission to POST again
    // or to declare cleanup. Later observation of that object permits deletion.
    state.lock().unwrap().fail_create = true;
    let uncertain = Allocate {
        bootstrap: None,
        session_id: "uncertain".into(),
        principal: "user".into(),
        profile: "pod".into(),
    };
    assert_eq!(
        service.allocate(uncertain.clone()).await.unwrap_err().code,
        "transport"
    );
    assert_eq!(service.get("uncertain").unwrap().state, "creating");
    let creates = state.lock().unwrap().creates;
    assert_eq!(
        service.allocate(uncertain).await.unwrap_err().code,
        "unknown_outcome"
    );
    assert_eq!(
        service.release("uncertain").await.unwrap_err().code,
        "unknown_outcome"
    );
    assert_eq!(state.lock().unwrap().creates, creates);
    {
        let mut state = state.lock().unwrap();
        state.object = state.pending.take();
    }
    assert_eq!(service.release("uncertain").await.unwrap_err().code, "busy");
    state.lock().unwrap().object = None;
    assert_eq!(
        service.release("uncertain").await.unwrap().state,
        "released"
    );

    // Failed provisioning is not absence proof. Explicit inspected absence can
    // close cleanup, including after a prior failed operator resolution.
    let absent = Allocate {
        bootstrap: None,
        session_id: "absent".into(),
        principal: "user".into(),
        profile: "pod".into(),
    };
    assert!(service.allocate(absent.clone()).await.is_err());
    let admin = serde_json::from_value(json!({
        "id":"operator", "token_file":token, "roles":["admin"]
    }))
    .unwrap();
    let mut resolution = AllocationResolution {
        resolution: Reconciliation {
            request_id: "absence-inspection".into(),
            expected_request_sha256: akh_session_services::recovery::fingerprint(&absent).unwrap(),
            status: TurnStatus::Failed,
            quiescent: true,
            evidence: vec!["operator://inspection/no-live-provisioner".into()],
            output: None,
            result: None,
        },
        observed_binding: None,
        observed_absent: false,
    };
    assert_eq!(
        service
            .reconcile(&admin, "absent", &resolution)
            .await
            .unwrap()
            .state,
        "failed"
    );
    assert!(service.release("absent").await.is_err());
    resolution.resolution.request_id = "absence-confirmed".into();
    resolution.resolution.status = TurnStatus::Completed;
    assert!(
        service
            .reconcile(&admin, "absent", &resolution)
            .await
            .is_err()
    );
    resolution.observed_absent = true;
    assert_eq!(
        service
            .reconcile(&admin, "absent", &resolution)
            .await
            .unwrap()
            .state,
        "released"
    );
    assert_eq!(
        service
            .reconcile(&admin, "absent", &resolution)
            .await
            .unwrap()
            .state,
        "released"
    );
    resolution.observed_absent = false;
    assert!(
        service
            .reconcile(&admin, "absent", &resolution)
            .await
            .is_err()
    );
    // A ready allocation can outlive its executor. Neither a stopped process
    // nor an operator's absence assertion permits dropping a present resource.
    resolution.resolution.request_id = "lost-executor-inspection".into();
    resolution.resolution.expected_request_sha256 =
        akh_session_services::recovery::fingerprint(&first.request).unwrap();
    resolution.observed_absent = true;
    state.lock().unwrap().object = Some(json!({"metadata":{"uid":"replacement-uid"}}));
    assert!(
        service
            .reconcile(&admin, "session", &resolution)
            .await
            .is_err()
    );
    assert_eq!(service.get("session").unwrap().state, "ready");
    state.lock().unwrap().object = None;
    resolution.resolution.quiescent = false;
    assert!(
        service
            .reconcile(&admin, "session", &resolution)
            .await
            .is_err()
    );
    resolution.resolution.quiescent = true;
    let recovered = service
        .reconcile(&admin, "session", &resolution)
        .await
        .unwrap();
    assert_eq!(recovered.state, "released");
    assert_eq!(recovered.binding, first.binding);
    assert_eq!(
        service
            .reconcile(&admin, "session", &resolution)
            .await
            .unwrap()
            .state,
        "released"
    );
    task.abort();
}

#[tokio::test]
async fn slow_executor_does_not_serialize_another_allocation() {
    use std::time::Duration;
    let seen = Arc::new(tokio::sync::Notify::new());
    let signal = seen.clone();
    let app = Router::new().route(
        "/v1/workspaces",
        post(move |Json(request): Json<AttachWorkspace>| {
            let signal = signal.clone();
            async move {
                if request.session_id == "slow" {
                    signal.notify_one();
                    tokio::time::sleep(Duration::from_secs(10)).await;
                }
                Json(request.binding)
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    let dir = tempfile::tempdir().unwrap();
    let token = dir.path().join("token");
    std::fs::write(&token, "private-allocation-test-token-123456").unwrap();
    let config:Config=serde_json::from_value(json!({"service_id":"sandboxes","listen":"127.0.0.1:0","data_dir":dir.path(),"principals":[{"id":"sessions","token_file":token,"roles":["session"]}],"endpoints":{"executor":{"url":url,"token_file":token}},"workspaces":{"local":{"revision":"1","site":"test","provisioner":{"kind":"existing","executor":"executor","root":"/workspaces"}}}})).unwrap();
    let service = Sandboxes::new(
        Arc::new(config),
        Store::open(&dir.path().join("state.redb")).unwrap(),
    )
    .unwrap();
    let slow = service.clone();
    let pending = tokio::spawn(async move {
        slow.allocate(Allocate {
            bootstrap: None,
            session_id: "slow".into(),
            principal: "user".into(),
            profile: "local".into(),
        })
        .await
    });
    tokio::time::timeout(Duration::from_secs(1), seen.notified())
        .await
        .unwrap();
    let fast = tokio::time::timeout(
        Duration::from_secs(1),
        service.allocate(Allocate {
            bootstrap: None,
            session_id: "fast".into(),
            principal: "user".into(),
            profile: "local".into(),
        }),
    )
    .await;
    pending.abort();
    let _ = pending.await;
    server.abort();
    assert_eq!(
        fast.expect("unrelated allocation waited for the slow executor")
            .unwrap()
            .state,
        "ready"
    );
}
