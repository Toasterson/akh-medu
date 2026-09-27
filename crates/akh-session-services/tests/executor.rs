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
fn literal_process_arguments_do_not_expand_shell_syntax_and_keep_execution_receipts() {
    let (dir, executor) = setup();
    let literal = "$(touch should-not-exist); `touch also-absent` $PATH * \"quoted value\"";
    let op = request(
        "literal",
        "exec_start",
        json!({"argv":["/usr/bin/printf","%s",literal]}),
    );
    executor.operate(op.clone()).unwrap();
    let receipt = wait(&executor, "literal");
    assert_eq!(executor.operate(op).unwrap(), receipt);
    let result = receipt.result.unwrap();
    assert!(result.success);
    assert_eq!(result.data.unwrap()["stdout"], literal);
    assert!(!dir.path().join("workspaces/one/should-not-exist").exists());
    assert!(!dir.path().join("workspaces/one/also-absent").exists());
    for (index, args) in [
        json!({"argv":["/usr/bin/printf","ok"],"command":"touch should-not-exist"}),
        json!({"argv":[]}),
        json!({"argv":["printf","ok"]}),
        json!({"argv":["/usr/bin/printf",false]}),
        json!({"argv":["/usr/bin/printf","bad\u{0}value"]}),
    ]
    .into_iter()
    .enumerate()
    {
        let receipt = executor
            .operate(request(&format!("invalid-{index}"), "exec_start", args))
            .unwrap();
        assert_eq!(receipt.status, TurnStatus::Failed);
        assert!(!receipt.result.unwrap().success);
    }
    assert!(!dir.path().join("workspaces/one/should-not-exist").exists());
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
fn bounded_poll_waits_for_completion_and_replays_its_observed_receipt() {
    let (_dir, executor) = setup();
    executor
        .operate(request(
            "wait-start",
            "exec_start",
            json!({"command":"sleep 0.2; printf done"}),
        ))
        .unwrap();
    let short = executor
        .operate(request(
            "short-poll",
            "exec_poll",
            json!({"process_id":"wait-start","wait_ms":1}),
        ))
        .unwrap();
    assert_eq!(short.result.unwrap().data.unwrap()["status"], "running");
    let poll = request(
        "wait-poll",
        "exec_poll",
        json!({"process_id":"wait-start","wait_ms":1000}),
    );
    let result = executor.operate(poll.clone()).unwrap();
    let data = result.result.as_ref().unwrap().data.as_ref().unwrap();
    assert_eq!(data["status"], "completed");
    assert_eq!(data["result"]["data"]["stdout"], "done");
    assert_eq!(executor.operate(poll).unwrap(), result);
    for (index, wait_ms) in [json!(-1), json!(10001), json!("1"), json!(1.5)]
        .into_iter()
        .enumerate()
    {
        let invalid = executor
            .operate(request(
                &format!("invalid-wait-{index}"),
                "exec_poll",
                json!({"process_id":"wait-start","wait_ms":wait_ms}),
            ))
            .unwrap();
        assert_eq!(invalid.status, TurnStatus::Failed);
        assert!(!invalid.result.unwrap().success);
    }
}

#[test]
fn waiting_poll_leaves_mutation_admission_available() {
    let (_dir, executor) = setup();
    let executor = Arc::new(executor);
    executor
        .operate(request(
            "interactive",
            "exec_start",
            json!({"command":"read value; printf '%s' \"$value\""}),
        ))
        .unwrap();
    let other = executor.clone();
    let waiter = std::thread::spawn(move || {
        other
            .operate(request(
                "pending-poll",
                "exec_poll",
                json!({"process_id":"interactive","wait_ms":3000}),
            ))
            .unwrap()
    });
    let deadline = Instant::now() + Duration::from_secs(1);
    while executor.receipt("pending-poll").is_err() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(5));
    }
    let before = Instant::now();
    executor
        .operate(request(
            "supply-input",
            "exec_stdin",
            json!({"process_id":"interactive","text":"unblocked\n"}),
        ))
        .unwrap();
    assert!(
        before.elapsed() < Duration::from_secs(1),
        "poll held mutation admission"
    );
    let observed = waiter.join().unwrap().result.unwrap().data.unwrap();
    assert_eq!(observed["status"], "completed");
    assert_eq!(observed["result"]["data"]["stdout"], "unblocked");
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
                    principal: None,
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

#[test]
fn release_before_attach_fences_delayed_delivery_and_survives_restart() {
    let (dir, executor) = setup();
    let request = AttachWorkspace {
        session_id: "late-session".into(),
        binding: WorkspaceBinding {
            id: "late".into(),
            sandbox_id: "sandbox".into(),
            executor: "executor".into(),
            generation: 1,
            site: "test".into(),
            root: dir
                .path()
                .join("workspaces/late")
                .to_string_lossy()
                .into_owned(),
            platform: "unobserved".into(),
            confinement: "unobserved".into(),
        },
    };
    assert!(
        executor.release("late", "late-session", 1).unwrap()["released"]
            .as_bool()
            .unwrap()
    );
    assert!(executor.release("late", "another-session", 1).is_err());
    assert_eq!(executor.attach(&request).unwrap_err().code, "conflict");
    let config = executor.config.clone();
    drop(executor);
    let executor =
        Executor::new(config, Store::open(&dir.path().join("state.redb")).unwrap()).unwrap();
    assert_eq!(executor.attach(&request).unwrap_err().code, "conflict");
    assert!(!dir.path().join("workspaces/late").exists());
}

#[test]
fn stat_is_read_only_and_stays_admitted_behind_the_unknown_outcome_fence() {
    let (dir, executor) = setup();
    std::fs::write(dir.path().join("workspaces/one/present"), b"12345").unwrap();
    let stat = |id: &str, path: &str| {
        let receipt = executor
            .operate(request(id, "stat", json!({"path":path})))
            .unwrap();
        assert_eq!(receipt.status, TurnStatus::Completed);
        serde_json::from_value::<WorkspaceStat>(receipt.result.unwrap().data.unwrap()).unwrap()
    };
    let present = stat("stat-present", "present");
    assert_eq!(present.kind, Some(WorkspaceEntryKind::File));
    assert_eq!(present.size, Some(5));
    assert_eq!(stat("stat-absent", "absent"), WorkspaceStat::MISSING);

    // An unresolved mutation fences every write, but a read-only stat is still admitted.
    let fenced = request(
        "fenced-write",
        "write",
        json!({"path":"x","content":"x","create":true}),
    );
    executor
        .store
        .transaction(|tx| {
            tx.put(
                &key("receipt", &["fenced-write"]),
                &OperationReceipt {
                    id: "fenced-write".into(),
                    request: fenced.clone(),
                    status: TurnStatus::Unknown,
                    result: None,
                    started_at: now_ms(),
                    finished_at: None,
                    principal: None,
                },
            )
        })
        .unwrap();
    // The raw fixture write bypasses the executor's Unknown index; a restart
    // rebuilds it, as it does for stores written before the index existed.
    let config = executor.config.clone();
    drop(executor);
    let executor =
        Executor::new(config, Store::open(&dir.path().join("state.redb")).unwrap()).unwrap();
    let stat = |id: &str, path: &str| {
        let receipt = executor
            .operate(request(id, "stat", json!({"path":path})))
            .unwrap();
        assert_eq!(receipt.status, TurnStatus::Completed);
        serde_json::from_value::<WorkspaceStat>(receipt.result.unwrap().data.unwrap()).unwrap()
    };
    assert_eq!(
        executor
            .operate(request("after-fence", "mkdir", json!({"path":"d"})))
            .unwrap_err()
            .code,
        "conflict"
    );
    assert!(stat("stat-after-fence", "present").exists);
    assert!(!dir.path().join("workspaces/one/d").exists());
}

#[tokio::test]
async fn http_operations_record_the_authenticated_principal_on_the_receipt() {
    use axum::body::{Body, to_bytes};
    use tower::ServiceExt;
    let (dir, _executor) = setup();
    let token = dir.path().join("akhomed.token");
    std::fs::write(&token, "fixture-akhomed-execd-token-0123456789").unwrap();
    let config: Config = serde_json::from_value(json!({
        "service_id":"executor","listen":"127.0.0.1:0","data_dir":dir.path().join("state"),
        "workspace_root":dir.path().join("workspaces"),
        "principals":[{"id":"akhomed-janus","roles":["session"],"token_file":token}]
    }))
    .unwrap();
    // `setup` attached the workspace in this store; reopen it behind the HTTP surface.
    drop(_executor);
    let executor = Executor::new(
        Arc::new(config),
        Store::open(&dir.path().join("state.redb")).unwrap(),
    )
    .unwrap();
    let app = akh_session_services::server::executor_router(executor.clone());
    let response = app
        .oneshot(
            axum::http::Request::builder()
                .method("POST")
                .uri("/v1/operations")
                .header(
                    "authorization",
                    "Bearer fixture-akhomed-execd-token-0123456789",
                )
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::to_vec(&request("attributed", "stat", json!({"path":"."})))
                        .unwrap(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let receipt: OperationReceipt =
        serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap();
    assert_eq!(receipt.principal.as_deref(), Some("akhomed-janus"));
    assert_eq!(receipt.status, TurnStatus::Completed);
    // The durable receipt carries the same attribution.
    assert_eq!(
        executor.receipt("attributed").unwrap().principal.as_deref(),
        Some("akhomed-janus")
    );
}

#[test]
fn a_legacy_unknown_receipt_is_indexed_on_restart_and_fences_mutations() {
    let (dir, executor) = setup();
    let op = request(
        "legacy-unknown",
        "write",
        json!({"path":"marker","content":"new","create":true}),
    );
    // A store written before the index existed: an Unknown receipt, no index.
    executor
        .store
        .transaction(|tx| {
            tx.put(
                &key("receipt", &["legacy-unknown"]),
                &OperationReceipt {
                    id: "legacy-unknown".into(),
                    request: op.clone(),
                    status: TurnStatus::Unknown,
                    result: None,
                    started_at: now_ms(),
                    finished_at: None,
                    principal: None,
                },
            )
        })
        .unwrap();
    let config = executor.config.clone();
    drop(executor);
    let executor =
        Executor::new(config, Store::open(&dir.path().join("state.redb")).unwrap()).unwrap();
    let indexed = executor
        .store
        .scan::<String>(&key("unknown_receipt", &["workspace"]))
        .unwrap();
    assert_eq!(indexed, ["legacy-unknown"]);
    let fenced = executor
        .operate(request(
            "blocked",
            "write",
            json!({"path":"other","content":"x","create":true}),
        ))
        .unwrap_err();
    assert_eq!(fenced.code, "conflict");
    // Reads stay admitted behind the fence.
    assert_eq!(
        executor
            .operate(request("look", "stat", json!({"path":"."})))
            .unwrap()
            .status,
        TurnStatus::Completed
    );
}

#[test]
fn terminal_read_receipts_are_collected_and_mutation_receipts_are_kept() {
    let (_dir, executor) = setup();
    let write = request(
        "kept-write",
        "write",
        json!({"path":"kept","content":"x","create":true}),
    );
    executor.operate(write.clone()).unwrap();
    let stat = request("old-stat", "stat", json!({"path":"kept"}));
    assert_eq!(
        executor.operate(stat.clone()).unwrap().status,
        TurnStatus::Completed
    );
    executor
        .operate(request("old-read", "read", json!({"path":"kept"})))
        .unwrap();
    // Nothing is old enough yet.
    assert_eq!(executor.collect_read_receipts(0).unwrap(), 0);
    assert_eq!(executor.collect_read_receipts(now_ms() + 1).unwrap(), 2);
    assert_eq!(executor.receipt("old-stat").unwrap_err().code, "not_found");
    assert_eq!(executor.receipt("old-read").unwrap_err().code, "not_found");
    // The mutation keeps its receipt, so its replay is still deduplicated.
    let original = executor.receipt("kept-write").unwrap();
    assert_eq!(executor.operate(write).unwrap(), original);
    // Replaying a collected read just reads again.
    assert_eq!(
        executor.operate(stat).unwrap().status,
        TurnStatus::Completed
    );
}
