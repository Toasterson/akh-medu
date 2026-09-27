use akh_runtime::{Host, ProviderConfig, TrainingWorkerError};
use akh_session_types::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    os::unix::fs::PermissionsExt,
    sync::{Mutex, MutexGuard, atomic::AtomicBool},
};

/// Every test here writes an executable fixture and then forks to run it. Two
/// tests doing that concurrently race: a fork in one inherits the other's
/// still-open write descriptor, and the exec then fails with ETXTBSY ("Text
/// file busy"). Serialise the tests instead of hoping the scheduler is kind.
static SERIAL: Mutex<()> = Mutex::new(());
fn serial() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Default)]
struct Witness {
    events: Vec<RuntimeEvent>,
    usage: Vec<Value>,
    calls: usize,
    approvals: usize,
    structured: bool,
}
impl Host for Witness {
    fn usage(&mut self, usage: &Value) {
        self.usage.push(usage.clone());
    }
    fn event(&mut self, event: RuntimeEvent) -> std::result::Result<(), String> {
        self.events.push(event);
        Ok(())
    }
    fn tool(
        &mut self,
        _id: &str,
        name: &str,
        arguments: Value,
    ) -> std::result::Result<ToolResult, String> {
        assert_eq!(name, "echo");
        assert_eq!(arguments["n"], 7);
        self.calls += 1;
        Ok(ToolResult {
            success: true,
            text: "7".into(),
            data: self
                .structured
                .then(|| json!({"process_id":"remote-process","status":"running"})),
            receipt: Some("receipt".into()),
        })
    }
    fn approval(
        &mut self,
        _id: &str,
        _name: &str,
        _args: &Value,
    ) -> std::result::Result<bool, String> {
        self.approvals += 1;
        Ok(true)
    }
    fn question(&mut self, _id: &str, _questions: &Value) -> std::result::Result<Value, String> {
        Ok(json!({"answers":{}}))
    }
}

fn setup(
    kind: RuntimeKind,
    mode: ExecutionMode,
) -> (tempfile::TempDir, ProviderConfig, RuntimeJobRequest) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("provider");
    std::fs::write(&path, include_str!("fixture.py")).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
    let cwd = dir.path().canonicalize().unwrap();
    let config = ProviderConfig {
        isolation: None,
        model_endpoint: None,
        native_candle: None,
        training_worker: None,
        revision: "1".into(),
        kind,
        working_dir: cwd.clone(),
        binary: path.clone(),
        node_binary: Some(path.clone()),
        bridge_script: Some(path),
        environment: Default::default(),
        launcher: vec![],
        timeout_secs: 5,
        execution_mode: Some(mode),
        conformance_file: None,
        native_aliases: Default::default(),
        effort: None,
    };
    let request = RuntimeJobRequest {
        id: "job".into(),
        session_id: "session".into(),
        turn_id: "turn".into(),
        generation: 1,
        binding: RuntimeBinding {
            runtime: "runtime".into(),
            profile: "test".into(),
            revision: "1".into(),
            kind,
            model: Some("test-model".into()),
            execution_mode: mode,
        },
        workspace: WorkspaceBinding {
            id: "ws".into(),
            sandbox_id: "sandbox".into(),
            executor: "executor".into(),
            generation: 1,
            site: "test".into(),
            root: if mode == ExecutionMode::Brokered {
                "/only/on/remote/executor".into()
            } else {
                cwd.to_string_lossy().into()
            },
            platform: "test".into(),
            confinement: "fixture".into(),
        },
        provider_session: None,
        input: TurnInput {
            origin: InputOrigin::Human {
                principal: "user".into(),
            },
            text: "hello".into(),
            data: None,
            artifacts: vec![],
        },
        instructions: "test".into(),
        tools: vec![],
        approval_policy: ApprovalPolicy::AskWrites,
    };
    (dir, config, request)
}

fn hash(path: &std::path::Path) -> String {
    format!("{:x}", Sha256::digest(std::fs::read(path).unwrap()))
}

#[test]
fn training_worker_uses_host_tools_and_returns_only_a_completed_frame() {
    let _serial = serial();
    let (dir, mut config, mut request) = setup(RuntimeKind::Internal, ExecutionMode::Brokered);
    let worker = dir.path().join("training-worker");
    std::fs::write(
        &worker,
        r#"#!/usr/bin/env python3
import json,sys
start=json.loads(sys.stdin.readline())
assert start['type']=='start' and start['config_path'].endswith('worker.json')
print(json.dumps({'type':'progress','phase':'rollout','usage':{'steps':1}}),flush=True)
print(json.dumps({'type':'tool_call','id':'call-1','name':'echo','arguments':{'n':7}}),flush=True)
result=json.loads(sys.stdin.readline())
assert result['type']=='tool_result' and result['id']=='call-1'
print(json.dumps({'type':'completed','output':result['result']['text']}),flush=True)
"#,
    )
    .unwrap();
    std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o700)).unwrap();
    let worker_config = dir.path().join("worker.json");
    std::fs::write(&worker_config, b"{}\n").unwrap();
    config.training_worker = Some(TrainingWorker {
        executable: worker.clone(),
        executable_sha256: hash(&worker),
        script: None,
        script_sha256: None,
        worker_config_path: worker_config.clone(),
        worker_config_sha256: hash(&worker_config),
        model: "test-model".into(),
        max_iterations: 4,
        max_context_tokens: 4096,
        max_output_tokens: 512,
        max_tool_calls: 2,
    });
    config.isolation = Some(RuntimeIsolation {
        directory: dir.path().into(),
        managed_files: Default::default(),
        environment_files: Default::default(),
        login_command: vec![],
        status_command: vec![],
        version_command: vec![],
        validation_artifacts: vec![],
    });
    request.tools.push(ToolSpec {
        name: "echo".into(),
        description: "echo".into(),
        input_schema: json!({"type":"object"}),
        read_only: true,
        destination: ToolDestination::Client,
    });
    let mut host = Witness::default();
    let output =
        akh_runtime::run_training_worker(&config, &request, &mut host, &AtomicBool::new(false))
            .unwrap();
    assert_eq!(output, "7");
    assert_eq!(host.calls, 1);
    assert_eq!(host.usage[0]["training"]["phase"], "rollout");
}

#[test]
fn training_worker_rejects_uncatalogued_tool_before_host_callback() {
    let _serial = serial();
    let (dir, mut config, request) = setup(RuntimeKind::Internal, ExecutionMode::Brokered);
    let worker = dir.path().join("training-worker");
    std::fs::write(&worker, "#!/usr/bin/env python3\nimport json\nprint(json.dumps({'type':'tool_call','id':'x','name':'write','arguments':{}}),flush=True)\n").unwrap();
    std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o700)).unwrap();
    let worker_config = dir.path().join("worker.json");
    std::fs::write(&worker_config, b"{}\n").unwrap();
    config.training_worker = Some(TrainingWorker {
        executable: worker.clone(),
        executable_sha256: hash(&worker),
        script: None,
        script_sha256: None,
        worker_config_path: worker_config.clone(),
        worker_config_sha256: hash(&worker_config),
        model: "test-model".into(),
        max_iterations: 1,
        max_context_tokens: 4096,
        max_output_tokens: 512,
        max_tool_calls: 1,
    });
    config.isolation = Some(RuntimeIsolation {
        directory: dir.path().into(),
        managed_files: Default::default(),
        environment_files: Default::default(),
        login_command: vec![],
        status_command: vec![],
        version_command: vec![],
        validation_artifacts: vec![],
    });
    let mut host = Witness::default();
    let error =
        akh_runtime::run_training_worker(&config, &request, &mut host, &AtomicBool::new(false))
            .unwrap_err();
    assert!(matches!(error, TrainingWorkerError::UnknownOutcome(_)));
    assert_eq!(host.calls, 0);
}

#[test]
fn training_worker_failure_after_a_tool_is_an_unknown_outcome() {
    let _serial = serial();
    let (dir, mut config, mut request) = setup(RuntimeKind::Internal, ExecutionMode::Brokered);
    let worker = dir.path().join("training-worker");
    std::fs::write(
        &worker,
        r#"#!/usr/bin/env python3
import json,sys
json.loads(sys.stdin.readline())
print(json.dumps({'type':'tool_call','id':'call-1','name':'echo','arguments':{'n':7}}),flush=True)
json.loads(sys.stdin.readline())
print(json.dumps({'type':'completed','output':'done','untrusted_prose':'claim'}),flush=True)
"#,
    )
    .unwrap();
    std::fs::set_permissions(&worker, std::fs::Permissions::from_mode(0o700)).unwrap();
    let worker_config = dir.path().join("worker.json");
    std::fs::write(&worker_config, b"{}\n").unwrap();
    config.training_worker = Some(TrainingWorker {
        executable: worker.clone(),
        executable_sha256: hash(&worker),
        script: None,
        script_sha256: None,
        worker_config_path: worker_config.clone(),
        worker_config_sha256: hash(&worker_config),
        model: "test-model".into(),
        max_iterations: 1,
        max_context_tokens: 4096,
        max_output_tokens: 512,
        max_tool_calls: 1,
    });
    config.isolation = Some(RuntimeIsolation {
        directory: dir.path().into(),
        managed_files: Default::default(),
        environment_files: Default::default(),
        login_command: vec![],
        status_command: vec![],
        version_command: vec![],
        validation_artifacts: vec![],
    });
    request.tools.push(ToolSpec {
        name: "echo".into(),
        description: "echo".into(),
        input_schema: json!({"type":"object"}),
        read_only: true,
        destination: ToolDestination::Client,
    });
    let mut host = Witness::default();
    let error =
        akh_runtime::run_training_worker(&config, &request, &mut host, &AtomicBool::new(false))
            .unwrap_err();
    assert!(matches!(error, TrainingWorkerError::UnknownOutcome(_)));
    assert_eq!(host.calls, 1);
}

#[test]
fn brokered_codex_uses_supplied_tools_and_never_approves_native_execution() {
    let _serial = serial();
    let (_dir, config, mut request) = setup(RuntimeKind::Codex, ExecutionMode::Brokered);
    let mut host = Witness::default();
    let cancel = AtomicBool::new(false);
    request.input.text = "approval".into();
    assert_eq!(
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap(),
        "decline"
    );
    assert_eq!(host.approvals, 0);
    request.input.text = "tool".into();
    request.tools.push(ToolSpec {
        name: "echo".into(),
        description: "echo".into(),
        input_schema: json!({"type":"object","properties":{"n":{"type":"integer"}}}),
        read_only: true,
        destination: ToolDestination::Client,
    });
    assert_eq!(
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap(),
        "7"
    );
    assert_eq!(host.calls, 1);
    assert!(
        host.events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::Reasoning { text } if text == "fixture reasoning"))
    );
    assert_eq!(host.usage.last().unwrap()["prompt_tokens"], 17);
    assert_eq!(
        host.usage.last().unwrap()["completion_tokens_details"]["reasoning_tokens"],
        2
    );
    host.structured = true;
    let output = akh_runtime::run(&config, &request, &mut host, &cancel).unwrap();
    let result: Value = serde_json::from_str(&output).unwrap();
    assert_eq!(result["data"]["process_id"], "remote-process");
    assert_eq!(result["receipt"], "receipt");
    assert_eq!(result["text"], "7");
}

#[test]
fn standalone_transport_resumes_exact_provider_session() {
    let _serial = serial();
    for kind in [RuntimeKind::Codex, RuntimeKind::Claude] {
        let (_dir, config, mut request) = setup(kind, ExecutionMode::Colocated);
        let mut host = Witness::default();
        let cancel = AtomicBool::new(false);
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap();
        if kind == RuntimeKind::Claude {
            assert_eq!(host.usage.last().unwrap()["prompt_tokens"], 11);
            assert_eq!(host.usage.last().unwrap()["completion_tokens"], 5);
            assert!(host.events.iter().any(|e| matches!(e, RuntimeEvent::Reasoning { text } if text == "fixture Claude reasoning")));
        }
        request.provider_session = host.events.iter().find_map(|e| {
            if let RuntimeEvent::Bound {
                provider_session, ..
            } = e
            {
                Some(provider_session.clone())
            } else {
                None
            }
        });
        assert!(request.provider_session.is_some());
        request.input.text = "again".into();
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap();
        assert!(host.events.iter().all(|e| {
            if let RuntimeEvent::Bound {
                provider_session, ..
            } = e
            {
                Some(provider_session) == request.provider_session.as_ref()
            } else {
                true
            }
        }));
    }
}

/// An OpenCode profile around the fixture. Brokered OpenCode requires a private
/// isolation directory; the fixture interpreter is found through the profile PATH.
fn opencode(mode: ExecutionMode) -> (tempfile::TempDir, ProviderConfig, RuntimeJobRequest) {
    let (dir, mut config, mut request) = setup(RuntimeKind::OpenCode, mode);
    config.node_binary = None;
    config.bridge_script = None;
    if mode == ExecutionMode::Brokered {
        config.isolation =
            Some(serde_json::from_value(json!({"directory": dir.path().join("profile")})).unwrap());
        config
            .environment
            .insert("PATH".into(), std::env::var("PATH").unwrap());
    }
    request.binding.model = Some("fixture/model".into());
    (dir, config, request)
}

fn echo_tool() -> ToolSpec {
    ToolSpec {
        name: "echo".into(),
        description: "echo".into(),
        input_schema: json!({"type":"object","properties":{"n":{"type":"integer"}}}),
        read_only: true,
        destination: ToolDestination::Client,
    }
}

fn bound_session(host: &Witness) -> Option<String> {
    host.events.iter().find_map(|e| match e {
        RuntimeEvent::Bound {
            provider_session, ..
        } => Some(provider_session.clone()),
        _ => None,
    })
}

#[test]
fn brokered_opencode_serves_akh_tools_over_loopback_mcp_and_never_approves() {
    let _serial = serial();
    let (_dir, config, mut request) = opencode(ExecutionMode::Brokered);
    let mut host = Witness::default();
    let cancel = AtomicBool::new(false);

    request.tools.push(echo_tool());
    request.input.text = "tool".into();
    assert_eq!(
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap(),
        "7"
    );
    assert_eq!(host.calls, 1, "the rogue call must not reach the host");
    assert!(host.events.iter().any(
        |e| matches!(e, RuntimeEvent::ToolStarted { id, name } if id == "call-1" && name == "akh_echo")
    ));
    assert!(host.events.iter().any(
        |e| matches!(e, RuntimeEvent::ToolCompleted { id, success: true, .. } if id == "call-1")
    ));
    assert!(host.events.iter().any(
        |e| matches!(e, RuntimeEvent::Reasoning { text } if text == "fixture OpenCode reasoning")
    ));
    assert!(
        !host
            .events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::Text { text } if text == "LEAK")),
        "updates for other sessions are ignored"
    );
    assert_eq!(host.usage.last().unwrap()["prompt_tokens"], 13);
    assert_eq!(host.usage.last().unwrap()["completion_tokens"], 4);
    assert_eq!(
        host.usage.last().unwrap()["completion_tokens_details"]["reasoning_tokens"],
        1
    );

    request.tools.clear();
    request.input.text = "approval".into();
    assert_eq!(
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap(),
        "reject"
    );
    assert_eq!(host.approvals, 0);
}

#[test]
fn colocated_opencode_routes_native_permissions_to_the_host() {
    let _serial = serial();
    let (_dir, config, mut request) = opencode(ExecutionMode::Colocated);
    let mut host = Witness::default();
    let cancel = AtomicBool::new(false);
    request.input.text = "approval".into();
    assert_eq!(
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap(),
        "once"
    );
    assert_eq!(host.approvals, 1);
    request.approval_policy = ApprovalPolicy::ReadOnly;
    assert_eq!(
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap(),
        "reject"
    );
    assert_eq!(host.approvals, 1, "read-only sessions never ask the host");
}

#[test]
fn opencode_resumes_the_exact_session_without_replaying_history() {
    let _serial = serial();
    let (_dir, config, mut request) = opencode(ExecutionMode::Brokered);
    let mut host = Witness::default();
    let cancel = AtomicBool::new(false);
    assert_eq!(
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap(),
        "hello"
    );
    request.provider_session = bound_session(&host);
    assert!(request.provider_session.is_some());
    request.input.text = "again".into();
    let mut resumed = Witness::default();
    assert_eq!(
        akh_runtime::run(&config, &request, &mut resumed, &cancel).unwrap(),
        "hello | again"
    );
    assert_eq!(bound_session(&resumed), request.provider_session);
    assert!(
        !resumed
            .events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::Text { text } if text.contains("REPLAYED")))
    );
}

#[test]
fn opencode_cancellation_reaches_the_provider_before_teardown() {
    let _serial = serial();
    let (dir, config, mut request) = opencode(ExecutionMode::Brokered);
    let mut host = Witness::default();
    let cancel = AtomicBool::new(false);
    request.input.text = "hang".into();
    let error = std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(std::time::Duration::from_millis(500));
            cancel.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        akh_runtime::run(&config, &request, &mut host, &cancel).unwrap_err()
    });
    assert!(error.contains("cancelled"), "{error}");
    let marker = dir.path().join("cancelled.marker");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !marker.exists() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert_eq!(
        std::fs::read_to_string(marker).ok(),
        bound_session(&host),
        "session/cancel names the bound session"
    );
}

#[test]
fn opencode_failures_carry_diagnostic_codes() {
    let _serial = serial();
    let (_dir, config, mut request) = opencode(ExecutionMode::Brokered);
    let mut host = Witness::default();
    let cancel = AtomicBool::new(false);
    for (prompt, code) in [
        ("auth", "akh::runtime::opencode::auth_required"),
        ("stop", "akh::runtime::opencode::stopped"),
    ] {
        request.input.text = prompt.into();
        let error = akh_runtime::run(&config, &request, &mut host, &cancel).unwrap_err();
        assert!(error.starts_with(code), "{error}");
        assert!(error.contains(" help: "), "{error}");
    }

    // OpenCode ignores MCP registration failures; the bridge does not.
    request.tools.push(echo_tool());
    std::fs::write(config.working_dir.join("skip-mcp"), "").unwrap();
    let error = akh_runtime::run(&config, &request, &mut host, &cancel).unwrap_err();
    assert!(
        error.starts_with("akh::runtime::opencode::mcp_not_connected"),
        "{error}"
    );
    std::fs::remove_file(config.working_dir.join("skip-mcp")).unwrap();
    request.tools.clear();

    request.binding.model = Some("no-provider".into());
    let error = akh_runtime::run(&config, &request, &mut host, &cancel).unwrap_err();
    assert!(
        error.starts_with("akh::runtime::opencode::model_format"),
        "{error}"
    );

    // Brokered OpenCode refuses to launch without a private profile directory
    // or with bridge-owned configuration in its environment.
    let (_dir, mut ambient, mut request) = opencode(ExecutionMode::Brokered);
    ambient.isolation = None;
    let error = akh_runtime::run(&ambient, &request, &mut host, &cancel).unwrap_err();
    assert!(
        error.starts_with("akh::runtime::opencode::isolation_required"),
        "{error}"
    );
    let (_dir, mut layered, _) = opencode(ExecutionMode::Brokered);
    layered
        .environment
        .insert("OPENCODE_CONFIG".into(), "/tmp/extra.json".into());
    request.binding.revision = layered.revision.clone();
    let error = akh_runtime::run(&layered, &request, &mut host, &cancel).unwrap_err();
    assert!(
        error.starts_with("akh::runtime::opencode::brokered_environment"),
        "{error}"
    );
}

/// Real-binary smoke test; not part of the default suite.
///
/// ```text
/// AKH_OPENCODE_BIN=/absolute/path/to/opencode AKH_OPENCODE_MODEL=provider/model \
///   cargo test -p akh-runtime --test providers real_opencode -- --ignored --nocapture
/// ```
///
/// Runs one colocated, read-only turn with the caller's existing OpenCode
/// login (development only; the test reads and copies no credentials). It shows
/// that the installed release accepts the bridge's generated configuration
/// (OpenCode rejects unknown config keys at load), that the `akh` agent can be
/// selected, and that an Akh tool round-trips through the loopback MCP endpoint.
/// It is not conformance evidence: brokered admission still needs a measured
/// conformance file for the exact binary and profile.
#[test]
#[ignore = "needs an installed OpenCode, a logged-in provider and AKH_OPENCODE_BIN/AKH_OPENCODE_MODEL"]
fn real_opencode_round_trips_an_akh_tool() {
    let binary = std::env::var("AKH_OPENCODE_BIN").expect("AKH_OPENCODE_BIN");
    let model = std::env::var("AKH_OPENCODE_MODEL").expect("AKH_OPENCODE_MODEL");
    let (_dir, mut config, mut request) = opencode(ExecutionMode::Colocated);
    config.binary = binary.into();
    config.timeout_secs = 300;
    request.binding.model = Some(model);
    request.approval_policy = ApprovalPolicy::ReadOnly;
    request.tools.push(echo_tool());
    request.input.text = "Call the akh_echo tool exactly once with n set to 7. \
        Then reply with only the text it returned."
        .into();
    let mut host = Witness::default();
    let output = akh_runtime::run(&config, &request, &mut host, &AtomicBool::new(false)).unwrap();
    eprintln!("OpenCode answered: {output}");
    assert_eq!(host.calls, 1);
    assert!(output.contains('7'));
    assert!(bound_session(&host).is_some());
}
