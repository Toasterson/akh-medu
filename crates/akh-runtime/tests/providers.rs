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
        execution_mode: mode,
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
