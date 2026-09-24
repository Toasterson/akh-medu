//! Native Git and executor HTTP, confined to an owned temporary fixture.
use akh_session_services::{
    config::Config, executor::Executor, sandbox::Sandboxes, server::executor_router, store::Store,
    types::*,
};
use bootstrap::{BootstrapRepository, RepositoryCheckout};
use serde_json::json;
use std::{
    path::Path,
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

struct OwnedGit(std::process::Child);
struct OwnedExecutor(Executor);
impl Drop for OwnedExecutor {
    fn drop(&mut self) {
        self.0.cancel_all();
    }
}
impl Drop for OwnedGit {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
fn git(cwd: &Path, args: &[&str]) -> String {
    let output = Command::new("/usr/bin/git")
        .current_dir(cwd)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .args([
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "fixture Git failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().into()
}

#[cfg(feature = "session")]
async fn verify_session_publications(
    app: &akh_session_services::server::SessionApp,
    owner: &akh_session_services::config::Principal,
    session: &str,
    snapshot_request: &str,
    head: &str,
    remote: &Path,
    executor: &Executor,
    lost: &AtomicBool,
    drop_send: &AtomicBool,
) -> publication::PublishSessionRepository {
    use akh_session_services::server::{SessionApp, session_router};
    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use publication::{PublishSessionRepository, SessionPublication};
    use tower::ServiceExt;
    let request = PublishSessionRepository {
        request_id: new_id(),
        snapshot_request_id: snapshot_request.into(),
        branch_ref: "refs/heads/anima/native-session-result".into(),
    };
    let mut other = owner.clone();
    other.id = "different-owner".into();
    assert_eq!(
        app.sessions
            .prepare_publication(&other, session, &request)
            .unwrap_err()
            .code,
        "forbidden"
    );
    other = owner.clone();
    other.roles = vec!["client".into(), "session_bootstrap".into()];
    assert_eq!(
        app.sessions
            .prepare_publication(&other, session, &request)
            .unwrap_err()
            .code,
        "forbidden"
    );
    let queued = app
        .sessions
        .prepare_publication(owner, session, &request)
        .unwrap();
    assert!(queued.dispatched_at.is_none() && queued.receipt.is_none());
    assert!(app.sessions.archive(owner, session).is_err());
    let mut changed = request.clone();
    changed.branch_ref.push_str("-other");
    assert_eq!(
        app.sessions
            .prepare_publication(owner, session, &changed)
            .unwrap_err()
            .code,
        "conflict"
    );
    let bearer = std::fs::read_to_string(&owner.token_file).unwrap();
    let response = session_router(app.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/sessions/{session}/repository/publications"))
                .header("authorization", format!("Bearer {bearer}"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&request).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::BAD_GATEWAY);
    assert!(lost.load(Ordering::SeqCst));
    assert_eq!(git(remote, &["rev-parse", &request.branch_ref]), head);
    let sent = app
        .sessions
        .publication(owner, session, &request.request_id)
        .unwrap();
    assert!(sent.dispatched_at.is_some() && sent.receipt.is_none());
    assert!(app.sessions.archive(owner, session).is_err());
    // The session has no active model turns. A restarted dispatcher still
    // recovers the original receipt without sending another publication.
    let restarted =
        SessionApp::new(app.sessions.config.clone(), app.sessions.store.clone()).unwrap();
    let dispatch = tokio::spawn(restarted.clone().run_dispatcher());
    let mut completed = None;
    for _ in 0..200 {
        let action = restarted
            .sessions
            .publication(owner, session, &request.request_id)
            .unwrap();
        if action.settled() {
            completed = Some(action);
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    dispatch.abort();
    let _ = dispatch.await;
    let completed = completed.expect("dispatcher did not recover native publication");
    assert!(
        completed
            .receipt
            .as_ref()
            .unwrap()
            .result
            .as_ref()
            .unwrap()
            .success
    );
    assert_eq!(
        restarted
            .publish_repository(owner, session, &request)
            .await
            .unwrap(),
        completed
    );
    assert_eq!(
        executor
            .store
            .scan::<OperationReceipt>("receipt/")
            .unwrap()
            .iter()
            .filter(|r| r.request.operation == "git_publish")
            .count(),
        1
    );
    let response = session_router(restarted.clone())
        .oneshot(
            Request::builder()
                .uri(format!(
                    "/v1/sessions/{session}/repository/publications/{}",
                    request.request_id
                ))
                .header("authorization", format!("Bearer {bearer}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let returned: SessionPublication =
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
            .unwrap();
    assert_eq!(returned, completed);

    let lost_request = PublishSessionRepository {
        request_id: new_id(),
        branch_ref: "refs/heads/anima/never-admitted".into(),
        ..request.clone()
    };
    drop_send.store(true, Ordering::SeqCst);
    assert_eq!(
        restarted
            .publish_repository(owner, session, &lost_request)
            .await
            .unwrap_err()
            .code,
        "transport"
    );
    let uncertain = restarted
        .sessions
        .publication(owner, session, &lost_request.request_id)
        .unwrap();
    assert!(uncertain.dispatched_at.is_some() && uncertain.receipt.is_none());
    assert!(!uncertain.settled());
    for _ in 0..2 {
        assert_eq!(
            restarted
                .publish_repository(owner, session, &lost_request)
                .await
                .unwrap(),
            uncertain
        );
    }
    assert_eq!(
        executor.receipt(&uncertain.operation.id).unwrap_err().code,
        "not_found"
    );
    let unsent_request = PublishSessionRepository {
        request_id: new_id(),
        branch_ref: "refs/heads/anima/cancel-before-send".into(),
        ..request.clone()
    };
    restarted
        .sessions
        .prepare_publication(owner, session, &unsent_request)
        .unwrap();
    restarted
        .sessions
        .retire(
            owner,
            session,
            None,
            &RetireSession {
                request_id: new_id(),
                reason: "owned publication fixture complete".into(),
            },
        )
        .unwrap();
    let cancelled = restarted
        .sessions
        .publication(owner, session, &unsent_request.request_id)
        .unwrap();
    assert!(
        cancelled.cancelled_at.is_some()
            && cancelled.dispatched_at.is_none()
            && cancelled.receipt.is_none()
    );
    assert_eq!(
        restarted
            .publish_repository(owner, session, &unsent_request)
            .await
            .unwrap(),
        cancelled
    );
    assert!(restarted.sessions.archive(owner, session).is_err());
    let fresh = PublishSessionRepository {
        request_id: new_id(),
        ..request.clone()
    };
    assert!(
        restarted
            .sessions
            .prepare_publication(owner, session, &fresh)
            .is_err()
    );
    let mut admin = owner.clone();
    admin.roles = vec!["admin".into()];
    let resolution=Reconciliation {request_id:new_id(), expected_request_sha256:akh_session_services::recovery::fingerprint(&uncertain.operation).unwrap(),
        quiescent:true, status:TurnStatus::Cancelled, result:None, output:None,
        evidence:vec!["Owned fault peer discarded this HTTP request before forwarding; sender returned and dispatcher was joined".into()]};
    assert!(
        restarted
            .reconcile_publication(owner, session, &lost_request.request_id, &resolution)
            .await
            .is_err()
    );
    let settled = restarted
        .reconcile_publication(&admin, session, &lost_request.request_id, &resolution)
        .await
        .unwrap();
    assert!(settled.cancelled_at.is_some() && settled.receipt.is_none());
    assert_eq!(
        restarted
            .reconcile_publication(&admin, session, &lost_request.request_id, &resolution)
            .await
            .unwrap(),
        settled
    );
    restarted.sessions.archive(owner, session).unwrap();
    assert_eq!(
        restarted
            .publish_repository(owner, session, &request)
            .await
            .unwrap(),
        completed
    );
    assert_eq!(git(remote, &["rev-parse", &request.branch_ref]), head);
    request
}

#[tokio::test]
async fn native_checkout_recovers_lost_http_and_manager_restart_then_preserves_existing_files() {
    use axum::{
        body::{Body, to_bytes},
        extract::Request,
        http::StatusCode,
        middleware::{self, Next},
        response::Response,
    };
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source");
    std::fs::create_dir(&source).unwrap();
    git(&source, &["init", "--quiet", "--template="]);
    std::fs::write(source.join("accepted.txt"), "exact assigned revision\n").unwrap();
    git(&source, &["add", "accepted.txt"]);
    git(&source, &["commit", "--quiet", "-m", "fixture"]);
    let revision = git(&source, &["rev-parse", "HEAD"]);
    git(
        dir.path(),
        &[
            "clone",
            "--quiet",
            "--bare",
            source.to_str().unwrap(),
            "source.git",
        ],
    );
    let socket = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let git_address = socket.local_addr().unwrap();
    drop(socket);
    let mut git_daemon = OwnedGit(
        Command::new("/usr/bin/git")
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .args([
                "daemon",
                "--export-all",
                "--enable=receive-pack",
                "--reuseaddr",
                "--listen=127.0.0.1",
                &format!("--port={}", git_address.port()),
                &format!("--base-path={}", dir.path().display()),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let mut ready = false;
    for _ in 0..100 {
        assert!(git_daemon.0.try_wait().unwrap().is_none());
        if tokio::net::TcpStream::connect(git_address).await.is_ok() {
            ready = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(ready);
    let repository = format!("git://{git_address}/source.git");
    let token = dir.path().join("token");
    std::fs::write(&token, "owned-bootstrap-test-capability-0001").unwrap();
    let workspaces = dir.path().join("workspaces");
    let executor_config:Config=serde_json::from_value(json!({
        "service_id":"executor","listen":"127.0.0.1:0","data_dir":dir.path().join("executor"),"workspace_root":workspaces,
        "git_snapshot":{"executable":"/usr/bin/git","executable_sha256":akh_session_services::digest(&std::fs::read("/usr/bin/git").unwrap())},
        "git_publication":[{"repository":repository,"branch_prefixes":["refs/heads/anima/"]}],
        "principals":[{"id":"sandbox","roles":["sandbox","repository_publish"],"token_file":token}]
    })).unwrap();
    let executor = Executor::new(
        Arc::new(executor_config),
        Store::open(&dir.path().join("executor.redb")).unwrap(),
    )
    .unwrap();
    let _owned_executor = OwnedExecutor(executor.clone());
    let lost = Arc::new(AtomicBool::new(false));
    let lose = lost.clone();
    let lost_snapshot = Arc::new(AtomicBool::new(false));
    let lose_snapshot = lost_snapshot.clone();
    let lost_publication = Arc::new(AtomicBool::new(false));
    let lose_publication = lost_publication.clone();
    let drop_publication = Arc::new(AtomicBool::new(false));
    let drop_send = drop_publication.clone();
    let router = executor_router(executor.clone()).layer(middleware::from_fn(
        move |request: Request, next: Next| {
            let lose = lose.clone();
            let lose_snapshot = lose_snapshot.clone();
            let lose_publication = lose_publication.clone();
            let drop_send = drop_send.clone();
            async move {
                let publication = request.uri().path() == "/v1/repository-publications";
                if publication && drop_send.swap(false, Ordering::SeqCst) {
                    return Response::builder()
                        .status(StatusCode::BAD_GATEWAY)
                        .body(Body::from("fixture discarded before executor admission"))
                        .unwrap();
                }
                let operation = publication || request.uri().path() == "/v1/operations";
                let response = next.run(request).await;
                if !operation {
                    return response;
                }
                let (parts, body) = response.into_parts();
                let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
                if let Ok(receipt) = serde_json::from_slice::<OperationReceipt>(&bytes) {
                    if receipt.request.operation == "git_publish"
                        && !lose_publication.swap(true, Ordering::SeqCst)
                    {
                        return Response::builder()
                            .status(StatusCode::BAD_GATEWAY)
                            .body(Body::from("lost after publication receipt"))
                            .unwrap();
                    }
                    if receipt.request.operation == "git_snapshot"
                        && !lose_snapshot.swap(true, Ordering::SeqCst)
                    {
                        return Response::builder()
                            .status(StatusCode::BAD_GATEWAY)
                            .body(Body::from("lost after snapshot receipt"))
                            .unwrap();
                    }
                    if receipt.request.arguments["argv"]
                        .as_array()
                        .is_some_and(|a| a.iter().any(|v| v == "init"))
                        && !lose.swap(true, Ordering::SeqCst)
                    {
                        return Response::builder()
                            .status(StatusCode::BAD_GATEWAY)
                            .body(Body::from("lost after executor receipt"))
                            .unwrap();
                    }
                }
                Response::from_parts(parts, Body::from(bytes))
            }
        },
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let http = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let config:Config=serde_json::from_value(json!({
        "service_id":"sandbox","listen":"127.0.0.1:0","data_dir":dir.path().join("sandbox"),
        "principals":[{"id":"sessions","roles":["session"],"token_file":token}],
        "endpoints":{"executor":{"url":endpoint,"token_file":token}},
        "workspaces":{"git":{"revision":"1","site":"fixture","provisioner":{"kind":"existing","executor":"executor","root":workspaces},
            "repository_bootstrap":{"repositories":[repository],"executable":"/usr/bin/git","config_file":null}}}
    })).unwrap();
    config.validate().unwrap();
    let config = Arc::new(config);
    let path = dir.path().join("sandbox.redb");
    let sandbox = Sandboxes::new(config.clone(), Store::open(&path).unwrap()).unwrap();
    let request = Allocate {
        session_id: "assigned".into(),
        principal: "owner".into(),
        profile: "git".into(),
        bootstrap: Some(BootstrapRepository {
            request_id: "checkout-1".into(),
            checkout: RepositoryCheckout {
                repository: repository.clone(),
                revision: revision.clone(),
            },
        }),
    };
    assert!(sandbox.allocate(request.clone()).await.is_err());
    assert!(lost.load(Ordering::SeqCst));
    assert_ne!(sandbox.get("assigned").unwrap().state, "ready");
    drop(sandbox);
    let sandbox = Sandboxes::new(config, Store::open(&path).unwrap()).unwrap();
    // Simulate a restart which retained the invocation but lost its terminal
    // receipt. Even a visibly initialized .git directory is not replay authority.
    let allocation = sandbox.get("assigned").unwrap();
    let init_id = akh_session_services::digest(
        akh_session_services::store::key("bootstrap_operation", &[&allocation.id, "initialize"])
            .as_bytes(),
    );
    let mut completed = None;
    for _ in 0..100 {
        let receipt = executor.receipt(&init_id).unwrap();
        if receipt.status.terminal() {
            completed = Some(receipt);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let completed = completed.expect("native init did not finish");
    assert!(completed.result.as_ref().unwrap().success);
    let mut uncertain = completed.clone();
    uncertain.status = TurnStatus::Unknown;
    uncertain.result = None;
    uncertain.finished_at = None;
    executor
        .store
        .transaction(|tx| {
            tx.put(
                &akh_session_services::store::key("receipt", &[&init_id]),
                &uncertain,
            )
        })
        .unwrap();
    assert_eq!(
        sandbox.allocate(request.clone()).await.unwrap_err().code,
        "unknown_outcome"
    );
    assert_eq!(
        executor
            .store
            .scan::<OperationReceipt>("receipt/")
            .unwrap()
            .len(),
        2
    );
    let mut admin = executor.config.principals[0].clone();
    admin.roles = vec!["admin".into()];
    let resolution = Reconciliation {
        request_id:"inspected-native-init".into(), expected_request_sha256:akh_session_services::recovery::fingerprint(&completed.request).unwrap(),
        status:TurnStatus::Completed, quiescent:true, evidence:vec!["fixture: native init exited successfully; inspected its .git directory and stopped process".into()],
        output:None,result:completed.result.clone(),
    };
    executor.reconcile(&admin, &init_id, &resolution).unwrap();
    let mut ready = None;
    for _ in 0..250 {
        match sandbox.allocate(request.clone()).await {
            Ok(allocation) => {
                ready = Some(allocation);
                break;
            }
            Err(error) => assert_eq!(error.code, "busy", "{error:?}"),
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let ready = ready.expect("checkout did not complete");
    assert_eq!(ready.state, "ready");
    let result = ready.bootstrap_result.as_ref().unwrap();
    assert_eq!(result.verified_revision, revision);
    assert_eq!(result.operation_receipts.len(), 7);
    let root = Path::new(&ready.binding.as_ref().unwrap().root);
    assert_eq!(git(root, &["rev-parse", "HEAD"]), revision);
    assert_eq!(
        std::fs::read_to_string(root.join("accepted.txt")).unwrap(),
        "exact assigned revision\n"
    );
    assert!(git(root, &["status", "--porcelain"]).is_empty());
    assert_eq!(
        sandbox
            .allocate(request.clone())
            .await
            .unwrap()
            .bootstrap_result,
        ready.bootstrap_result
    );
    assert_eq!(
        executor
            .store
            .scan::<OperationReceipt>("receipt/")
            .unwrap()
            .len(),
        7
    );
    // A changed intent cannot reset the existing workspace.
    let mut changed = request.clone();
    changed.bootstrap.as_mut().unwrap().checkout.revision = "b".repeat(40);
    assert!(sandbox.allocate(changed).await.is_err());
    // A fresh session with pre-existing files fails before Git initialization.
    let mut existing = request.clone();
    existing.session_id = "existing-files".into();
    let id = format!(
        "akh-{}",
        &akh_session_services::digest(existing.session_id.as_bytes())[..32]
    );
    let root = workspaces.join(&id);
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("keep.txt"), "preserve me").unwrap();
    assert!(sandbox.allocate(existing).await.is_err());
    assert_eq!(
        std::fs::read_to_string(root.join("keep.txt")).unwrap(),
        "preserve me"
    );
    assert!(!root.join(".git").exists());
    #[cfg(feature = "session")]
    verify_session_snapshots(
        dir.path(),
        &sandbox,
        &executor,
        &endpoint,
        &token,
        &repository,
        &revision,
        &lost_snapshot,
        &lost_publication,
        &drop_publication,
    )
    .await;
    #[cfg(feature = "session")]
    if let Ok(probe) = std::env::var("ANIMA_AKH_COMPLETION_PROBE") {
        verify_native_completion(
            dir.path(),
            &sandbox,
            &endpoint,
            &token,
            &repository,
            &revision,
            &probe,
        )
        .await;
    }
    sandbox.release("assigned").await.unwrap();
    sandbox.release("existing-files").await.unwrap();
    executor.cancel_all();
    http.abort();
}

#[cfg(feature = "session")]
async fn verify_session_snapshots(
    dir: &Path,
    sandbox: &Sandboxes,
    executor: &Executor,
    endpoint: &str,
    token: &Path,
    repository: &str,
    base: &str,
    lost: &AtomicBool,
    lost_publication: &AtomicBool,
    drop_publication: &AtomicBool,
) {
    use akh_session_services::server::{SessionApp, session_router};
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode},
    };
    use repository::{GitSnapshot, SnapshotSessionRepository};
    use tower::ServiceExt;
    let config: Config = serde_json::from_value(json!({
        "service_id":"sessions", "listen":"127.0.0.1:0", "data_dir":dir.join("sessions"),
        "principals":[{"id":"owner","roles":["client","session_bootstrap","repository_publish"],"akhs":["test"],"token_file":token}],
        "endpoints":{"executor":{"url":endpoint,"token_file":token}},
        "akhs":{"test":{"bootstrap_repositories":[repository],"instructions":"test",
            "runtime":{"runtime":"runtime","profile":"test","revision":"1","kind":"codex","execution_mode":"colocated"},"workspace_profiles":["git"]}}
    })).unwrap();
    config.validate().unwrap();
    let config = Arc::new(config);
    let store = Store::open(&dir.join("sessions.redb")).unwrap();
    let app = SessionApp::new(config.clone(), store.clone()).unwrap();
    let owner = config.principals[0].clone();
    let session = app
        .sessions
        .create(
            &owner,
            &NewSession {
                request_id: "snapshot-session".into(),
                akh: "test".into(),
                workspace_profile: "git".into(),
                conversation_key: None,
                grant: None,
            },
        )
        .unwrap();
    let bootstrap = BootstrapRepository {
        request_id: "checkout".into(),
        checkout: RepositoryCheckout {
            repository: repository.into(),
            revision: base.into(),
        },
    };
    app.sessions
        .prepare_repository(&owner, &session.id, &bootstrap)
        .unwrap();
    let allocate = Allocate {
        session_id: session.id.clone(),
        principal: owner.id.clone(),
        profile: "git".into(),
        bootstrap: Some(bootstrap),
    };
    let ready = {
        let mut result = None;
        for _ in 0..250 {
            match sandbox.allocate(allocate.clone()).await {
                Ok(value) => {
                    result = Some(value);
                    break;
                }
                Err(error) => assert_eq!(error.code, "busy", "{error:?}"),
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        result.expect("snapshot fixture checkout did not finish")
    };
    let binding = ready.binding.unwrap();
    let root = Path::new(&binding.root);
    app.sessions
        .bind_prepared_workspace(&session.id, binding.clone(), ready.bootstrap_result)
        .unwrap();
    std::fs::write(root.join("implementation.txt"), "session result\n").unwrap();
    git(root, &["add", "."]);
    git(root, &["commit", "--quiet", "-m", "session result"]);
    let head = git(root, &["rev-parse", "HEAD"]);
    let request = SnapshotSessionRepository {
        request_id: new_id(),
        head_revision: head.clone(),
        expected_patch: None,
    };
    let call = |app: SessionApp, request: SnapshotSessionRepository| {
        let id = session.id.clone();
        let bearer = std::fs::read_to_string(token).unwrap();
        async move {
            let response = session_router(app)
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(format!("/v1/sessions/{id}/repository/snapshots"))
                        .header("authorization", format!("Bearer {bearer}"))
                        .header("content-type", "application/json")
                        .body(Body::from(serde_json::to_vec(&request).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 4 * 1024 * 1024)
                .await
                .unwrap();
            (status, bytes)
        }
    };
    assert_eq!(
        call(app.clone(), request.clone()).await.0,
        StatusCode::BAD_GATEWAY
    );
    assert!(lost.load(Ordering::SeqCst));
    // Restart while only the executor has a terminal receipt. No new execution.
    let app = SessionApp::new(config, store).unwrap();
    let (status, bytes) = call(app.clone(), request.clone()).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    let receipt: OperationReceipt = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(receipt.request.session_id, session.id);
    assert_eq!(receipt.request.workspace_id, binding.id);
    assert_eq!(receipt.request.arguments["base_revision"], base);
    let snapshot: GitSnapshot =
        serde_json::from_value(receipt.result.as_ref().unwrap().data.clone().unwrap()).unwrap();
    assert_eq!(snapshot.request.head_revision, head);
    assert!(snapshot.patch.contains("+session result"));
    let snapshots: Vec<_> = executor
        .store
        .scan::<OperationReceipt>("receipt/")
        .unwrap()
        .into_iter()
        .filter(|r| r.request.operation == "git_snapshot")
        .collect();
    assert_eq!(snapshots, vec![receipt.clone()]);
    let verified_request = SnapshotSessionRepository {
        request_id: new_id(),
        head_revision: head.clone(),
        expected_patch: Some(format!(
            "{}\n",
            git(root, &["format-patch", "--stdout", "-1", &head])
        )),
    };
    let (verified_status, verified_bytes) = call(app.clone(), verified_request.clone()).await;
    assert_eq!(
        verified_status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&verified_bytes)
    );
    let verified_receipt: OperationReceipt = serde_json::from_slice(&verified_bytes).unwrap();
    assert!(verified_receipt.result.as_ref().unwrap().success);
    assert_eq!(
        verified_receipt
            .result
            .as_ref()
            .unwrap()
            .data
            .as_ref()
            .unwrap()["expected_patch_sha256"],
        akh_session_services::digest(verified_request.expected_patch.as_ref().unwrap().as_bytes())
    );
    let mut other = owner.clone();
    other.id = "other".into();
    assert_eq!(
        app.snapshot_repository(&other, &session.id, &request)
            .await
            .unwrap_err()
            .code,
        "forbidden"
    );
    other = owner.clone();
    other.roles = vec!["client".into()];
    assert_eq!(
        app.snapshot_repository(&other, &session.id, &request)
            .await
            .unwrap_err()
            .code,
        "forbidden"
    );
    let mut changed = request.clone();
    changed.head_revision = base.into();
    assert_eq!(call(app.clone(), changed).await.0, StatusCode::CONFLICT);
    // Model/editor changes do not rewrite the durable selected-commit evidence.
    std::fs::write(root.join("implementation.txt"), "later editor changes\n").unwrap();
    assert_eq!(call(app.clone(), request.clone()).await.1, bytes);
    let turn = app
        .sessions
        .submit(
            &owner,
            &session.id,
            &SubmitInput {
                request_id: "probe-turn".into(),
                text: "owned protocol fixture".into(),
                artifacts: vec![],
            },
        )
        .unwrap();
    app.sessions.cancel(&owner, &session.id, &turn.id).unwrap();
    let publication_request = verify_session_publications(
        &app,
        &owner,
        &session.id,
        &verified_request.request_id,
        &head,
        &dir.join("source.git"),
        executor,
        lost_publication,
        drop_publication,
    )
    .await;
    app.sessions.archive(&owner, &session.id).unwrap();
    sandbox.release(&session.id).await.unwrap();
    assert_eq!(call(app.clone(), request.clone()).await.1, bytes);
    let mut fresh = request.clone();
    fresh.request_id = "after-archive".into();
    assert_eq!(call(app.clone(), fresh).await.0, StatusCode::CONFLICT);
    // Lost local cache after archival still recovers the existing remote receipt.
    app.sessions
        .store
        .transaction(|tx| {
            tx.delete(&akh_session_services::store::key(
                "repository_snapshot_receipt",
                &[&session.id, &request.request_id],
            ))
        })
        .unwrap();
    assert_eq!(call(app.clone(), request.clone()).await.1, bytes);
    if let Ok(probe) = std::env::var("ANIMA_AKH_SNAPSHOT_PROBE") {
        assert!(Path::new(&probe).is_absolute());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let expected_publication = app
            .sessions
            .publication(&owner, &session.id, &publication_request.request_id)
            .unwrap();
        let router = session_router(app);
        let server = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        for (request, expected) in [(&request, &receipt), (&verified_request, &verified_receipt)] {
            let input = json!({"config":{"url":endpoint,"token_file":token,"ca_file":null,"automation":"fixture","source":"fixture"},
            "checkout":{"repository":repository,"revision":base},
            "admission":{"session_id":session.id,"turn_id":turn.id},"request_id":request.request_id,"head":head,"expected_patch":request.expected_patch});
            let command = tokio::process::Command::new(&probe);
            let result = akh_session_services::process::output_with_input(
                command,
                Some(serde_json::to_vec(&input).unwrap()),
                Duration::from_secs(20),
                4 * 1024 * 1024,
            )
            .await;
            if result.is_err() {
                server.abort();
            }
            let remote: OperationReceipt =
                serde_json::from_slice(&result.expect("native Anima snapshot client failed"))
                    .unwrap();
            assert_eq!(&remote, expected);
        }
        let input = json!({"config":{"url":endpoint,"token_file":token,"ca_file":null,"automation":"fixture","source":"fixture"},
            "checkout":{"repository":repository,"revision":base},"admission":{"session_id":session.id,"turn_id":turn.id},
            "request_id":verified_request.request_id,"head":head,"expected_patch":verified_request.expected_patch,
            "publication":{"request_id":publication_request.request_id,"branch_ref":publication_request.branch_ref}});
        let result = akh_session_services::process::output_with_input(
            tokio::process::Command::new(&probe),
            Some(serde_json::to_vec(&input).unwrap()),
            Duration::from_secs(20),
            4 * 1024 * 1024,
        )
        .await;
        if result.is_err() {
            server.abort();
        }
        let observed: publication::SessionPublication =
            serde_json::from_slice(&result.expect("native Anima publication client failed"))
                .unwrap();
        assert_eq!(observed, expected_publication);
        server.abort();
        println!(
            "Native Anima snapshot and publication clients verified original archived executor receipts"
        );
    }
}

#[cfg(feature = "session")]
async fn verify_native_completion(
    dir: &Path,
    sandbox: &Sandboxes,
    executor: &str,
    token: &Path,
    repository: &str,
    revision: &str,
    probe: &str,
) {
    use akh_session_services::server::{SessionApp, sandbox_router, session_router};
    use axum::{
        body::Body,
        extract::Request,
        http::StatusCode,
        middleware::{self, Next},
        response::Response,
    };
    assert!(Path::new(probe).is_absolute());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let sandbox_endpoint = format!("http://{}", listener.local_addr().unwrap());
    let router = sandbox_router(sandbox.clone());
    let sandbox_http = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let config:Config=serde_json::from_value(json!({
        "service_id":"completion-sessions","listen":"127.0.0.1:0","data_dir":dir.join("completion-sessions"),
        "principals":[{"id":"owner","roles":["client","session_bootstrap"],"akhs":["test"],"token_file":token}],
        "endpoints":{"executor":{"url":executor,"token_file":token},"sandboxes":{"url":sandbox_endpoint,"token_file":token}},
        "akhs":{"test":{"bootstrap_repositories":[repository],"instructions":"owned protocol fixture",
            "runtime":{"runtime":"runtime","profile":"test","revision":"1","kind":"codex","execution_mode":"colocated"},"workspace_profiles":["git"]}}
    })).unwrap();
    config.validate().unwrap();
    let owner = config.principals[0].clone();
    let app = SessionApp::new(
        Arc::new(config),
        Store::open(&dir.join("completion-sessions.redb")).unwrap(),
    )
    .unwrap();
    let session = app
        .sessions
        .create(
            &owner,
            &NewSession {
                request_id: "completion".into(),
                akh: "test".into(),
                workspace_profile: "git".into(),
                conversation_key: None,
                grant: None,
            },
        )
        .unwrap();
    let bootstrap = BootstrapRepository {
        request_id: "completion-checkout".into(),
        checkout: RepositoryCheckout {
            repository: repository.into(),
            revision: revision.into(),
        },
    };
    app.sessions
        .prepare_repository(&owner, &session.id, &bootstrap)
        .unwrap();
    let allocate = Allocate {
        session_id: session.id.clone(),
        principal: owner.id.clone(),
        profile: "git".into(),
        bootstrap: Some(bootstrap),
    };
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    let allocation = loop {
        match sandbox.allocate(allocate.clone()).await {
            Ok(allocation) => break allocation,
            Err(error) => assert_eq!(error.code, "busy"),
        }
        assert!(tokio::time::Instant::now() < deadline);
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    app.sessions
        .bind_prepared_workspace(
            &session.id,
            allocation.binding.unwrap(),
            allocation.bootstrap_result,
        )
        .unwrap();
    let mut first = None;
    // Inject terminal provider reports through the actual session ledger. This
    // fixture proves native HTTP/archival/executor cleanup, not model inference.
    for name in ["initial", "followup"] {
        let turn = app
            .sessions
            .submit(
                &owner,
                &session.id,
                &SubmitInput {
                    request_id: name.into(),
                    text: name.into(),
                    artifacts: vec![],
                },
            )
            .unwrap();
        if first.is_none() {
            first = Some(turn.id);
        }
        let request = app.sessions.next_job(&session.id).unwrap().unwrap();
        app.sessions
            .apply_job(&RuntimeJob {
                request,
                status: TurnStatus::Completed,
                provider_session: None,
                output: Some(name.into()),
                error: None,
                events: vec![],
            })
            .unwrap();
    }
    let lost = Arc::new(AtomicBool::new(false));
    let lose = lost.clone();
    let router = session_router(app.clone()).layer(middleware::from_fn(
        move |request: Request, next: Next| {
            let lose = lose.clone();
            async move {
                let selected = request.uri().path().ends_with("/archive-selected");
                let response = next.run(request).await;
                if selected && response.status().is_success() && !lose.swap(true, Ordering::SeqCst)
                {
                    return Response::builder()
                        .status(StatusCode::BAD_GATEWAY)
                        .body(Body::from("owned lost archive reply"))
                        .unwrap();
                }
                response
            }
        },
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    let dispatcher = tokio::spawn(app.clone().run_dispatcher());
    let input = json!({"config":{"url":endpoint,"token_file":token,"ca_file":null,"automation":"fixture","source":"fixture"},"admission":{"session_id":session.id,"turn_id":first},"request_id":new_id()});
    let result = akh_session_services::process::output_with_input(
        tokio::process::Command::new(probe),
        Some(serde_json::to_vec(&input).unwrap()),
        Duration::from_secs(30),
        1024 * 1024,
    )
    .await;
    dispatcher.abort();
    server.abort();
    sandbox_http.abort();
    let proof: serde_json::Value =
        serde_json::from_slice(&result.expect("native Anima completion client failed")).unwrap();
    assert!(lost.load(Ordering::SeqCst));
    let archived = app.sessions.get(&owner, &session.id).unwrap();
    assert!(archived.closed_at.is_some());
    assert_eq!(
        proof["request"]["next_sequence"].as_u64().unwrap() + 1,
        archived.next_sequence
    );
    assert_eq!(sandbox.get(&session.id).unwrap().state, "released");
    assert_eq!(
        app.sessions
            .store
            .get::<bool>(&akh_session_services::store::key(
                "workspace_released",
                &[&session.id]
            ))
            .unwrap(),
        Some(true)
    );
    println!(
        "Native Anima checkpoint recovered lost Akh archive reply and verified actual sandbox/executor release"
    );
}
