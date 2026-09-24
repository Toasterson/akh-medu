use super::super::tool::{Tool, ToolOutput, ToolSignature};
use super::*;
use crate::message::VecSink;
use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn runtime(dir: &std::path::Path, kind: RuntimeKind) -> AgentRuntime {
    let binary = dir.join("fake-provider");
    std::fs::write(&binary, include_str!("fixture.py")).unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    AgentRuntime::new(
        kind,
        RuntimeConfig {
            working_dir: dir.canonicalize().unwrap(),
            binary: Some(binary.clone()),
            node_binary: Some(binary.clone()),
            bridge_script: Some(binary),
            timeout_secs: 10,
        },
        Some("test-model".into()),
        None,
        Some("test persona".into()),
    )
    .unwrap()
}

fn turn<'a>(
    runtime: &'a AgentRuntime,
    engine: &'a Engine,
    registry: &'a ToolRegistry,
    sink: &'a VecSink,
    cancel: &'a AtomicBool,
    prompt: &str,
) -> Turn<'a> {
    Turn {
        runtime,
        cwd: runtime.config.working_dir.clone(),
        prompt: prompt.into(),
        instructions: "test instructions".into(),
        tools: super::super::agentic_chat::ollama_tool_catalog(registry),
        session: None,
        sink,
        cancel,
        registry,
        engine,
        trace: vec![],
        controller: None,
        defer_output: false,
    }
}

#[test]
fn codex_resumes_exact_session_and_isolates_other_conversations() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path(), RuntimeKind::Codex);
    let engine = Engine::new(Default::default()).unwrap();
    let registry = ToolRegistry::new();
    let sink = VecSink::new();
    let cancel = AtomicBool::new(false);
    let mut first = turn(&runtime, &engine, &registry, &sink, &cancel, "alpha");
    assert_eq!(bridge::run(&mut first).unwrap(), "alpha");
    let mut second = turn(&runtime, &engine, &registry, &sink, &cancel, "bravo");
    assert_eq!(bridge::run(&mut second).unwrap(), "bravo");
    assert_ne!(
        first.session.as_ref().unwrap().id,
        second.session.unwrap().id
    );
    first.prompt = "follow-up".into();
    assert_eq!(bridge::run(&mut first).unwrap(), "alpha | follow-up");
}

struct Echo;
impl Tool for Echo {
    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "echo".into(),
            description: "Echo typed arguments".into(),
            parameters: vec![],
            source: ToolSource::Native,
            danger: DangerInfo {
                level: DangerLevel::Safe,
                capabilities: Default::default(),
                description: "No side effects".into(),
                shadow_triggers: vec![],
            },
        }
    }
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "echo".into(),
            description: "Echo".into(),
            parameters: vec![],
        }
    }
    fn execute(
        &self,
        _: &Engine,
        input: ToolInput,
    ) -> super::super::error::AgentResult<ToolOutput> {
        assert_eq!(input.params["nested"]["flags"], json!([true]));
        Ok(ToolOutput::ok(input.params["n"].to_string()))
    }
}

#[test]
fn codex_tools_use_typed_akh_registry_and_record_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path(), RuntimeKind::Codex);
    let engine = Engine::new(Default::default()).unwrap();
    let registry = ToolRegistry::new();
    registry.register(Box::new(Echo));
    let sink = VecSink::new();
    let cancel = AtomicBool::new(false);
    let mut t = turn(&runtime, &engine, &registry, &sink, &cancel, "tool");
    assert_eq!(bridge::run(&mut t).unwrap(), "7");
    assert_eq!(t.trace, vec![json!({"tool":"echo","ok":true})]);
}

struct Approver;
impl super::super::permission::PermissionAsker for Approver {
    fn ask(&self, manifest: &ToolManifest, summary: &str) -> bool {
        assert!(manifest.name.contains("requestApproval"));
        assert!(summary.contains("touch marker"));
        true
    }
}

#[test]
fn upstream_approval_needs_client_even_in_bypass_mode_and_plan_denies() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path(), RuntimeKind::Codex);
    let engine = Engine::new(Default::default()).unwrap();
    let registry = ToolRegistry::new();
    let sink = VecSink::new();
    let cancel = AtomicBool::new(false);
    let mut t = turn(&runtime, &engine, &registry, &sink, &cancel, "approval");
    assert_eq!(bridge::run(&mut t).unwrap(), "decline");
    registry.set_permission_asker(Arc::new(Approver));
    assert_eq!(bridge::run(&mut t).unwrap(), "accept");
    registry.set_permission_mode(super::super::permission::PermissionMode::Plan);
    assert_eq!(bridge::run(&mut t).unwrap(), "decline");
}

#[test]
fn terminal_failures_and_failed_resumes_do_not_start_replacement_turns() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path(), RuntimeKind::Codex);
    let engine = Engine::new(Default::default()).unwrap();
    let registry = ToolRegistry::new();
    let sink = VecSink::new();
    let cancel = AtomicBool::new(false);
    for prompt in ["failed", "crash", "malformed"] {
        let mut t = turn(&runtime, &engine, &registry, &sink, &cancel, prompt);
        assert!(bridge::run(&mut t).is_err(), "{prompt}");
    }
    let mut t = turn(
        &runtime,
        &engine,
        &registry,
        &sink,
        &cancel,
        "fail_next_resume",
    );
    bridge::run(&mut t).unwrap();
    let id = t.session.as_ref().unwrap().id.clone();
    t.prompt = "retry".into();
    assert!(bridge::run(&mut t).unwrap_err().contains("resume refused"));
    assert_eq!(t.session.unwrap().id, id);
}

#[test]
fn cancellation_kills_the_provider_process_group() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path(), RuntimeKind::Codex);
    let engine = Engine::new(Default::default()).unwrap();
    let registry = ToolRegistry::new();
    let sink = VecSink::new();
    let cancel = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&cancel);
    let pid_path = dir.path().join("child.pid");
    let watch_path = pid_path.clone();
    let worker = std::thread::spawn(move || {
        let start = Instant::now();
        while !watch_path.exists() && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        signal.store(true, Ordering::Relaxed);
    });
    let mut t = turn(&runtime, &engine, &registry, &sink, &cancel, "hang");
    assert!(bridge::run(&mut t).unwrap_err().contains("cancelled"));
    worker.join().unwrap();
    let pid: i32 = std::fs::read_to_string(pid_path).unwrap().parse().unwrap();
    let start = Instant::now();
    while unsafe { libc::kill(pid, 0) } == 0 && start.elapsed() < Duration::from_secs(3) {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert_ne!(
        unsafe { libc::kill(pid, 0) },
        0,
        "provider descendant survived cancellation"
    );
}

#[test]
fn claude_bridge_resumes_identity_and_surfaces_failed_results() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path(), RuntimeKind::Claude);
    let engine = Engine::new(Default::default()).unwrap();
    let registry = ToolRegistry::new();
    let sink = VecSink::new();
    let cancel = AtomicBool::new(false);
    let mut t = turn(&runtime, &engine, &registry, &sink, &cancel, "hello");
    assert_eq!(bridge::run(&mut t).unwrap(), "hello");
    let id = t.session.as_ref().unwrap().id.clone();
    t.prompt = "again".into();
    assert_eq!(bridge::run(&mut t).unwrap(), "again");
    assert_eq!(t.session.as_ref().unwrap().id, id);
    t.prompt = "failed".into();
    assert!(
        bridge::run(&mut t)
            .unwrap_err()
            .contains("provider failure")
    );
}

#[test]
fn runtime_config_roundtrips_and_rejects_invalid_paths() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path(), RuntimeKind::Codex);
    let mut cfg = crate::akh_config::AkhConfig::default();
    cfg.backend = crate::akh_config::ChatBacking::Codex;
    cfg.agent_runtime = Some(runtime.config.clone());
    let path = dir.path().join("akh.toml");
    std::fs::write(&path, toml::to_string(&cfg).unwrap()).unwrap();
    let (loaded, unknown) = crate::akh_config::AkhConfig::load_checked(&path).unwrap();
    assert!(
        unknown
            .iter()
            .all(|key| !key.path.starts_with("agent_runtime")),
        "{unknown:?}"
    );
    assert_eq!(loaded.agent_runtime, cfg.agent_runtime);
    let mut config = runtime.config.clone();
    config.working_dir = "relative".into();
    assert!(AgentRuntime::new(RuntimeKind::Codex, config, None, None, None).is_err());
}

#[test]
fn runtime_turn_uses_conversation_binding_and_never_falls_back_on_failure() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(dir.path(), RuntimeKind::Codex);
    let engine = Arc::new(Engine::new(Default::default()).unwrap());
    let mut agent = Agent::new(
        Arc::clone(&engine),
        super::super::agent::AgentConfig::default(),
    )
    .unwrap();
    agent.set_agent_runtime(runtime);
    let a = Mutex::new(ConversationState::new("session-a", "narrative"));
    let b = Mutex::new(ConversationState::new("session-b", "narrative"));
    let sink = VecSink::new();
    let cancel = AtomicBool::new(false);
    let run = |prompt, state| {
        agent.run_agentic_turn_with_conversation_state(
            &engine, "thoth", prompt, &sink, &cancel, false, state,
        )
    };
    assert_eq!(run("alpha", &a), Some(AgenticTurnOutcome::Completed));
    assert_eq!(run("bravo", &b), Some(AgenticTurnOutcome::Completed));
    assert_eq!(run("again", &a), Some(AgenticTurnOutcome::Completed));
    assert_eq!(
        a.lock().unwrap().turns.back().unwrap().text,
        "alpha | again"
    );
    assert_eq!(b.lock().unwrap().turns.back().unwrap().text, "bravo");
    assert_eq!(run("failed", &b), Some(AgenticTurnOutcome::Refused));
    assert_eq!(b.lock().unwrap().turns.back().unwrap().text, "failed");
    let controlled = "Finish <akh:control>{\"stages\":[{\"label\":\"fresh evidence\",\"any_of\":[{\"tool\":\"shell_exec\"}]}]}</akh:control>";
    assert_eq!(run(controlled, &a), Some(AgenticTurnOutcome::Refused));
    assert!(!sink.messages().iter().any(
        |m| matches!(m, AkhMessage::Narrative { text, .. } if text.contains("<akh:control>"))
    ));
    assert_eq!(
        agent.run_agentic_turn_with_conversation_state(
            &engine,
            "thoth",
            "Claim finished",
            &sink,
            &cancel,
            true,
            &b
        ),
        Some(AgenticTurnOutcome::Refused)
    );
}

#[test]
fn timeout_ends_a_stalled_provider() {
    let dir = tempfile::tempdir().unwrap();
    let mut runtime = runtime(dir.path(), RuntimeKind::Codex);
    runtime.config.timeout_secs = 1;
    let engine = Engine::new(Default::default()).unwrap();
    let registry = ToolRegistry::new();
    let sink = VecSink::new();
    let cancel = AtomicBool::new(false);
    let mut t = turn(&runtime, &engine, &registry, &sink, &cancel, "hang");
    let start = Instant::now();
    assert!(bridge::run(&mut t).unwrap_err().contains("timed out"));
    assert!(start.elapsed() < Duration::from_secs(3));
}

fn live_runtime(kind: RuntimeKind) {
    let dir = tempfile::tempdir().unwrap();
    let bridge =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("scripts/agent-backends/claude.mjs");
    let runtime = AgentRuntime::new(
        kind,
        RuntimeConfig {
            working_dir: dir.path().canonicalize().unwrap(),
            binary: None,
            node_binary: None,
            bridge_script: Some(bridge),
            timeout_secs: 120,
        },
        None,
        None,
        None,
    )
    .unwrap();
    let engine = Engine::new(Default::default()).unwrap();
    let registry = ToolRegistry::new();
    struct Probe(Arc<std::sync::atomic::AtomicUsize>);
    impl Tool for Probe {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: "runtime_probe".into(),
                description: "Return the runtime integration test marker".into(),
                parameters: vec![],
            }
        }
        fn manifest(&self) -> ToolManifest {
            let mut m = Echo.manifest();
            m.name = "runtime_probe".into();
            m.description = "Return the runtime integration test marker".into();
            m
        }
        fn execute(
            &self,
            _: &Engine,
            _: ToolInput,
        ) -> super::super::error::AgentResult<ToolOutput> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(ToolOutput::ok("AKH_PROVIDER_7391"))
        }
    }
    let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    registry.register(Box::new(Probe(Arc::clone(&calls))));
    let sink = VecSink::new();
    let cancel = AtomicBool::new(false);
    let mut t = turn(
        &runtime,
        &engine,
        &registry,
        &sink,
        &cancel,
        "Call the Akh runtime_probe tool exactly once, then reply with only its returned text. Do not use any other tools.",
    );
    let run = match kind {
        RuntimeKind::Codex => bridge::run,
        RuntimeKind::Claude => bridge::run,
    };
    let answer = run(&mut t).expect("installed provider first turn");
    assert_eq!(
        calls.load(Ordering::Relaxed),
        1,
        "provider must call the Akh tool exactly once"
    );
    assert!(
        answer.contains("AKH_PROVIDER_7391"),
        "provider did not return the marker"
    );
    let id = t.session.as_ref().unwrap().id.clone();
    t.prompt = "Reply with only the marker from my previous message. Do not use any tools.".into();
    let answer = run(&mut t).expect("installed provider resume");
    assert!(
        answer.contains("AKH_PROVIDER_7391"),
        "provider lost session context"
    );
    assert_eq!(t.session.as_ref().unwrap().id, id);
}

#[test]
#[ignore = "requires installed Codex login and makes two small live model calls"]
fn live_installed_codex_session() {
    live_runtime(RuntimeKind::Codex);
}

#[test]
#[ignore = "requires Claude SDK dependencies and authentication; makes two small live model calls"]
fn live_installed_claude_session() {
    live_runtime(RuntimeKind::Claude);
}

#[test]
fn the_in_process_bridge_is_colocated_only_in_development() {
    use crate::deployment::DeploymentMode;
    assert_eq!(
        bridge::execution_mode(DeploymentMode::Development).unwrap(),
        akh_session_types::ExecutionMode::Colocated
    );
    let refused = bridge::execution_mode(DeploymentMode::Deployed).unwrap_err();
    assert!(
        refused.starts_with("akh::config::in_process_forbidden"),
        "{refused}"
    );
}
