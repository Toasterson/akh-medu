//! Real subprocess cwd regressions run outside the large library-test process.
//! Native CI previously failed fork/spawn with ENOMEM before these assertions.
use akh_medu::{
    agent::{
        tool::{Tool, ToolInput},
        tools::ShellExecTool,
    },
    engine::{Engine, EngineConfig},
    vsa::Dimension,
};

fn test_engine() -> Engine {
    Engine::new(EngineConfig {
        dimension: Dimension::TEST,
        ..Default::default()
    })
    .unwrap()
}

/// A session cwd active on this thread (ACP session-cwd threading) must
/// be what `pwd` reports — the live-verification case from the ACP
/// session-cwd work item, exercised here without a whole ACP harness.
#[test]
fn runs_in_the_thread_local_session_cwd_when_one_is_set() {
    let dir = tempfile::tempdir().unwrap();
    let canonical = dir.path().canonicalize().unwrap();
    let tool = ShellExecTool;
    let engine = test_engine();

    let _guard = akh_medu::fs::set_session_cwd(Some(canonical.clone()));
    let input = ToolInput::new().with_param("command", "pwd");
    let result = tool.execute(&engine, input).unwrap();
    assert!(result.success, "{}", result.result);
    assert!(
        result.result.contains(canonical.to_str().unwrap()),
        "pwd must report the active session cwd: {}",
        result.result
    );
}

/// Once the guard drops (turn over), a later call on the same thread must
/// not still see the old session cwd — the "no session cwd" compatibility
/// bar applies immediately after the turn that had one ends.
#[test]
fn ignores_a_session_cwd_once_its_guard_has_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let canonical = dir.path().canonicalize().unwrap();
    let tool = ShellExecTool;
    let engine = test_engine();

    {
        let _guard = akh_medu::fs::set_session_cwd(Some(canonical.clone()));
    }
    let input = ToolInput::new().with_param("command", "pwd");
    let result = tool.execute(&engine, input).unwrap();
    assert!(result.success, "{}", result.result);
    assert!(
        !result.result.contains(canonical.to_str().unwrap()),
        "pwd must NOT report a dropped session cwd: {}",
        result.result
    );
}

/// With no session cwd ever set, behavior is exactly what it was before
/// session-cwd threading existed: `request.cwd` is `None`, so the
/// backend's own default applies (for the exec-only default backend
/// here, that's the process's own cwd).
#[test]
fn no_session_cwd_leaves_request_cwd_unset() {
    let tool = ShellExecTool;
    let engine = test_engine();
    assert_eq!(akh_medu::fs::session_cwd(), None, "sanity: nothing active");

    let input = ToolInput::new().with_param("command", "pwd");
    let result = tool.execute(&engine, input).unwrap();
    assert!(result.success, "{}", result.result);
    let expected = std::env::current_dir().unwrap();
    assert!(
        result.result.contains(expected.to_str().unwrap()),
        "pwd must report the process cwd unchanged: {} (expected under {})",
        result.result,
        expected.display()
    );
}
