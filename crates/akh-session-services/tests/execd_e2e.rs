//! Loopback end-to-end check of akh-medu's `ExecdBackend` (Phase 53c) against
//! the real akhsbxd and akhexecd HTTP routers: allocation through an
//! `existing` provisioner, operations through the akhsbxd executor gateway,
//! receipts attributed to the authenticated principal, release on drop.
use akh_medu::backend::execd::{ExecdBackend, ExecdConfig};
use akh_medu::backend::{Backend, Confinement, ExecError, ExecRequest};
use akh_medu::fs::Filesystem;
use akh_session_services::{
    config::Config, executor::Executor, sandbox::Sandboxes, store::Store, types::*,
};
use serde_json::json;
use std::{path::Path, sync::Arc, time::Duration};

async fn serve(router: axum::Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    url
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn execd_backend_runs_effects_through_real_akhsbxd_and_akhexecd() {
    let dir = tempfile::tempdir().unwrap();
    let akhomed_token = dir.path().join("akhomed.token");
    std::fs::write(&akhomed_token, "fixture-akhomed-janus-token-0123456789").unwrap();
    let sandbox_token = dir.path().join("akhsbxd.token");
    std::fs::write(&sandbox_token, "fixture-akhsbxd-executor-token-0123456789").unwrap();
    let workspaces = dir.path().join("workspaces");

    let executor_config: Config = serde_json::from_value(json!({
        "service_id":"executor","listen":"127.0.0.1:0","data_dir":dir.path().join("executor"),
        "workspace_root":workspaces,"confinement":"unconfined",
        "principals":[
            {"id":"akhsbxd","roles":["sandbox"],"token_file":sandbox_token},
            {"id":"akhomed-janus","roles":["session"],"token_file":akhomed_token}
        ]
    }))
    .unwrap();
    executor_config.validate().unwrap();
    let executor = Executor::new(
        Arc::new(executor_config),
        Store::open(&dir.path().join("executor.redb")).unwrap(),
    )
    .unwrap();
    let executor_url = serve(akh_session_services::server::executor_router(
        executor.clone(),
    ))
    .await;

    let sandbox_config: Config = serde_json::from_value(json!({
        "service_id":"sandbox","listen":"127.0.0.1:0","data_dir":dir.path().join("sandbox"),
        "principals":[
            {"id":"akhomed-janus","roles":["session","executor_proxy"],"token_file":akhomed_token}
        ],
        "endpoints":{"executor":{"url":executor_url,"token_file":sandbox_token}},
        "workspaces":{"mac-local":{"revision":"1","site":"mac-mini",
            "provisioner":{"kind":"existing","executor":"executor","root":workspaces}}}
    }))
    .unwrap();
    sandbox_config.validate().unwrap();
    let sandboxes = Sandboxes::new(
        Arc::new(sandbox_config),
        Store::open(&dir.path().join("sandbox.redb")).unwrap(),
    )
    .unwrap();
    let sandbox_url = serve(akh_session_services::server::sandbox_router(
        sandboxes.clone(),
    ))
    .await;

    let session = tokio::task::spawn_blocking(move || {
        let config = ExecdConfig::new(&sandbox_url, &akhomed_token, "mac-local", "janus", "janus")
            .with_poll_wait(Duration::from_millis(200));
        let backend = ExecdBackend::connect(config).expect("allocate through akhsbxd");
        // The `existing` provisioner runs akhexecd as a host process: never "sandboxed".
        assert_eq!(backend.handshake().confinement, Confinement::Unconfined);
        assert_eq!(backend.binding().confinement, "unconfined");
        let root = backend.read_roots()[0].clone();

        backend
            .write(Path::new("notes/plan.md"), "one")
            .expect("create with parents");
        backend
            .write(&root.join("notes/plan.md"), "two")
            .expect("replace under the current revision");
        assert_eq!(
            backend.read_to_string(Path::new("notes/plan.md")).unwrap(),
            "two"
        );
        assert_eq!(
            backend.read_dir(Path::new("notes")).unwrap(),
            [root.join("notes/plan.md")]
        );
        assert!(backend.exists(Path::new("notes/plan.md")));
        assert!(!backend.exists(Path::new("notes/absent.md")));
        assert_eq!(backend.file_size(Path::new("notes/plan.md")).unwrap(), 3);
        assert!(!backend.is_symlink(Path::new("notes/plan.md")));

        let outcome = backend
            .exec(
                ExecRequest::shell(
                    "printf \"$GREETING\"; cat plan.md >&2; exit 3",
                    Duration::from_secs(20),
                )
                .with_cwd(root.join("notes"))
                .with_env("GREETING", "hi"),
            )
            .expect("exec through the gateway");
        assert_eq!(outcome.exit_code, 3);
        assert!(!outcome.success);
        assert_eq!(outcome.stdout, b"hi");
        assert_eq!(outcome.stderr, b"two");

        let timed_out = backend
            .exec(ExecRequest::shell("sleep 30", Duration::from_secs(1)))
            .unwrap_err();
        assert!(
            matches!(timed_out, ExecError::Timeout { .. }),
            "{timed_out:?}"
        );
        assert!(backend.renew_lease().unwrap() > now_ms());
        let session = backend.session_id().to_owned();
        drop(backend);
        session
    })
    .await
    .unwrap();

    // Every receipt names the authenticated caller, including the read-only stats.
    let receipts = executor.store.scan::<OperationReceipt>("receipt/").unwrap();
    assert!(receipts.iter().any(|r| r.request.operation == "stat"));
    assert!(receipts.iter().any(|r| r.request.operation == "exec_start"));
    for receipt in &receipts {
        assert_eq!(
            receipt.principal.as_deref(),
            Some("akhomed-janus"),
            "{receipt:?}"
        );
        assert_eq!(receipt.request.session_id, session);
    }
    // Dropping the backend released the allocation.
    assert_eq!(sandboxes.get(&session).unwrap().state, "released");
}
