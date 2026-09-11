use akh_session_services::{
    config::Config,
    executor::{AttachWorkspace, Executor},
    store::{Store, key},
    types::*,
};
use serde_json::json;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

fn setup() -> (tempfile::TempDir, Executor) {
    let dir = tempfile::tempdir().unwrap();
    let config:Config=serde_json::from_value(json!({"service_id":"executor","listen":"127.0.0.1:0","data_dir":dir.path().join("state"),"workspace_root":dir.path().join("workspaces"),"principals":[{"id":"sessions","roles":["session"],"token_file":"/unused"}]})).unwrap();
    let executor = Executor::new(
        Arc::new(config),
        Store::open(&dir.path().join("state.redb")).unwrap(),
    )
    .unwrap();
    let request = AttachWorkspace {
        session_id: "session".into(),
        binding: WorkspaceBinding {
            id: "workspace".into(),
            sandbox_id: "sandbox".into(),
            executor: "executor".into(),
            generation: 1,
            site: "local".into(),
            root: dir.path().join("workspaces/one").to_string_lossy().into(),
            platform: "unknown".into(),
            confinement: "unknown".into(),
        },
    };
    let first = executor.attach(&request).unwrap();
    assert_eq!(executor.attach(&request).unwrap(), first);
    (dir, executor)
}
fn request(id: &str, operation: &str, args: serde_json::Value) -> WorkspaceOperation {
    WorkspaceOperation {
        id: id.into(),
        session_id: "session".into(),
        turn_id: "turn".into(),
        workspace_id: "workspace".into(),
        generation: 1,
        operation: operation.into(),
        arguments: args,
    }
}
fn wait(executor: &Executor, id: &str) -> OperationReceipt {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let receipt = executor.receipt(id).unwrap();
        if receipt.status.terminal() {
            return receipt;
        }
        assert!(Instant::now() < deadline, "process did not finish");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn duplicate_process_request_executes_once_at_bound_workspace_and_preserves_failure_status() {
    let (dir, executor) = setup();
    let op = request(
        "start",
        "exec_start",
        json!({"command":"printf 'once\n' >> marker; printf out; printf err >&2; exit 7"}),
    );
    executor.operate(op.clone()).unwrap();
    let terminal = wait(&executor, "start");
    assert_eq!(executor.operate(op.clone()).unwrap(), terminal);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("workspaces/one/marker")).unwrap(),
        "once\n"
    );
    let result = terminal.result.unwrap();
    assert!(!result.success);
    let data = result.data.unwrap();
    assert_eq!(data["exit_code"], 7);
    assert_eq!(data["stdout"], "out");
    assert_eq!(data["stderr"], "err");
    let mut changed = op;
    changed.arguments = json!({"command":"printf twice >> marker"});
    assert!(executor.operate(changed).is_err());
}

#[test]
fn process_io_and_cancellation_use_owned_handles_and_stream_progress() {
    let (_dir, executor) = setup();
    executor
        .operate(request(
            "start",
            "exec_start",
            json!({"command":"printf ready; read value; printf '%s' \"$value\""}),
        ))
        .unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let poll = executor
        .operate(request("poll", "exec_poll", json!({"process_id":"start"})))
        .unwrap();
    assert_eq!(
        poll.result.unwrap().data.unwrap()["progress"]["stdout"],
        "ready"
    );
    executor
        .operate(request(
            "input",
            "exec_stdin",
            json!({"process_id":"start","text":"hello\n"}),
        ))
        .unwrap();
    assert_eq!(
        wait(&executor, "start").result.unwrap().data.unwrap()["stdout"],
        "readyhello"
    );
    executor
        .operate(request(
            "long",
            "exec_start",
            json!({"command":"sleep 30","timeout_ms":60000}),
        ))
        .unwrap();
    executor
        .operate(request(
            "cancel",
            "exec_cancel",
            json!({"process_id":"long"}),
        ))
        .unwrap();
    assert_eq!(wait(&executor, "long").status, TurnStatus::Cancelled);
    let mut foreign = request("foreign", "read", json!({"path":"marker"}));
    foreign.session_id = "other".into();
    assert!(executor.operate(foreign).is_err());
}

#[test]
fn restart_does_not_replay_an_operation_with_an_unknown_outcome() {
    let (dir, executor) = setup();
    let op = request(
        "unknown",
        "write",
        json!({"path":"marker","content":"new","create":true}),
    );
    executor
        .store
        .transaction(|tx| {
            tx.put(
                &key("receipt", &["unknown"]),
                &OperationReceipt {
                    id: "unknown".into(),
                    request: op.clone(),
                    status: TurnStatus::Running,
                    result: None,
                    started_at: now_ms(),
                    finished_at: None,
                },
            )
        })
        .unwrap();
    let config = executor.config.clone();
    drop(executor);
    let executor =
        Executor::new(config, Store::open(&dir.path().join("state.redb")).unwrap()).unwrap();
    assert_eq!(executor.operate(op).unwrap().status, TurnStatus::Unknown);
    assert!(!dir.path().join("workspaces/one/marker").exists());
    assert!(
        executor
            .operate(request(
                "next",
                "write",
                json!({"path":"marker","content":"new","create":true})
            ))
            .is_err()
    );
    let original = executor.receipt("unknown").unwrap();
    let mut admin = executor.config.principals[0].clone();
    let mut resolution = Reconciliation {
        request_id: "resolve-1".into(),
        expected_request_sha256: akh_session_services::recovery::fingerprint(&original.request)
            .unwrap(),
        status: TurnStatus::Failed,
        quiescent: true,
        evidence: vec!["fixture: inspected marker absence and no active process".into()],
        output: None,
        result: Some(ToolResult {
            success: false,
            text: "No file was written".into(),
            data: None,
            receipt: None,
        }),
    };
    assert_eq!(
        executor
            .reconcile(&admin, "unknown", &resolution)
            .unwrap_err()
            .code,
        "forbidden"
    );
    admin.roles = vec!["admin".into()];
    resolution.quiescent = false;
    assert!(executor.reconcile(&admin, "unknown", &resolution).is_err());
    resolution.quiescent = true;
    assert_eq!(
        executor
            .reconcile(&admin, "unknown", &resolution)
            .unwrap()
            .status,
        TurnStatus::Failed
    );
    assert_eq!(
        executor
            .reconcile(&admin, "unknown", &resolution)
            .unwrap()
            .status,
        TurnStatus::Failed
    );
    assert!(!dir.path().join("workspaces/one/marker").exists());
    assert!(
        executor
            .operate(request(
                "after-recovery",
                "write",
                json!({"path":"marker","content":"explicit new operation","create":true})
            ))
            .unwrap()
            .result
            .unwrap()
            .success
    );
    resolution.evidence.push("changed after acceptance".into());
    assert_eq!(
        executor
            .reconcile(&admin, "unknown", &resolution)
            .unwrap_err()
            .code,
        "conflict"
    );
}

#[test]
fn expired_lease_fences_new_effects_and_cannot_be_renewed_implicitly() {
    let (dir, executor) = setup();
    assert!(
        executor
            .renew_lease(
                "workspace",
                &RenewLease {
                    session_id: "other".into(),
                    generation: 1
                }
            )
            .is_err()
    );
    let lease = executor
        .renew_lease(
            "workspace",
            &RenewLease {
                session_id: "session".into(),
                generation: 1,
            },
        )
        .unwrap();
    assert!(lease.expires_at > now_ms());
    executor
        .store
        .transaction(|tx| {
            tx.put(
                &key("lease", &["workspace"]),
                &WorkspaceLease {
                    expires_at: 1,
                    ..lease
                },
            )
        })
        .unwrap();
    executor.expire_leases().unwrap();
    assert_eq!(
        executor
            .operate(request(
                "expired",
                "write",
                json!({"path":"marker","content":"bad","create":true})
            ))
            .unwrap_err()
            .code,
        "expired"
    );
    assert!(!dir.path().join("workspaces/one/marker").exists());
    assert_eq!(
        executor
            .renew_lease(
                "workspace",
                &RenewLease {
                    session_id: "session".into(),
                    generation: 1
                }
            )
            .unwrap_err()
            .code,
        "expired"
    );
}
