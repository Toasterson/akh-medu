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
        "principals":[{"id":"sandbox","roles":["sandbox"],"token_file":token}]
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
    let router = executor_router(executor.clone()).layer(middleware::from_fn(
        move |request: Request, next: Next| {
            let lose = lose.clone();
            let lose_snapshot = lose_snapshot.clone();
            async move {
                let operation = request.uri().path() == "/v1/operations";
                let response = next.run(request).await;
                if !operation {
                    return response;
                }
                let (parts, body) = response.into_parts();
                let bytes = to_bytes(body, 1024 * 1024).await.unwrap();
                if let Ok(receipt) = serde_json::from_slice::<OperationReceipt>(&bytes) {
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
    )
    .await;
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
        "principals":[{"id":"owner","roles":["client","session_bootstrap"],"akhs":["test"],"token_file":token}],
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
        request_id: "result-1".into(),
        head_revision: head.clone(),
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
    assert_eq!(call(app, request).await.1, bytes);
}
