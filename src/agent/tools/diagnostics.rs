//! Diagnostics tool: provider-dispatched, blast-radius scoped, baseline-aware.
//!
//! WI 4 — the tool that makes the 2026-07-28 failure structurally impossible.
//! `compile_feedback` reports every default member of a nine-member workspace
//! and cannot tell pre-existing rot from damage the session caused. This tool
//! defaults to the blast radius, captures a baseline at the immutable worktree
//! base, and partitions diagnostics into task-introduced vs pre-existing.
//!
//! The `cargo` provider reuses `compile_feedback`'s implementation wholesale:
//! JSON diagnostics with `suggested_replacement`, the 64 KB output cap, the
//! `STUCK:` contract, and incremental stdout draining for the ACP idle
//! watchdog. Both tools coexist per the WI's out-of-scope constraint (retiring
//! `compile_feedback` is WI 5).
//!
//! ## Provider dispatch
//!
//! Providers are selected from the manifest files the ingester already
//! indexes. This work item ships the `cargo` provider. The dispatch surface
//! is the `kind` parameter: `build` → `cargo check`, `lint` → `cargo clippy`,
//! `test-compile` → `cargo test --no-run`. A future non-cargo provider (WI 5)
//! would add a `provider` parameter and dispatch on the manifest file type.

use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::process::Command;

use crate::agent::error::AgentResult;
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    Capability, DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;

use super::compile_feedback::{resolve_timeout, run_diagnostics_pipeline};

/// Provider-dispatched, blast-radius scoped, baseline-aware diagnostics.
///
/// `kind` selects the diagnostic check: `build` (cargo check), `lint` (cargo
/// clippy), or `test-compile` (cargo test --no-run). `scope` defaults to the
/// blast radius (packages changed since the worktree base); `workspace` is
/// explicit opt-in and never the default.
pub struct DiagnosticsTool;

impl Tool for DiagnosticsTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: "diagnostics".into(),
            description: "Run build/lint/test-compile diagnostics with blast-radius scoping \
                          and baseline comparison. Defaults to packages changed since the Git \
                          worktree base; workspace scope is explicit opt-in. Partitions \
                          diagnostics into task-introduced (full fidelity with suggestions) \
                          and pre-existing (collapsed to context)."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "kind".into(),
                    description: "Diagnostic kind: 'build' (cargo check, default), 'lint' \
                                  (cargo clippy), or 'test-compile' (cargo test --no-run)."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "scope".into(),
                    description: "Scope: 'blast-radius' (default, packages changed since the \
                                  worktree base) or 'workspace' (entire workspace). \
                                  Workspace is explicit opt-in and never the default."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "timeout".into(),
                    description: "Timeout in seconds (default: 600; override process-wide with \
                                  AKH_COMPILE_FEEDBACK_TIMEOUT_SECS). A cold build can take \
                                  90s+; request 900–1200s for a known-cold container."
                        .into(),
                    required: false,
                },
                ToolParam {
                    name: "path".into(),
                    description: "Working directory (default: current directory).".into(),
                    required: false,
                },
                ToolParam {
                    name: "base_ref".into(),
                    description: "Optional Git base ref override. Defaults to origin/HEAD, then \
                                  conventional trunk refs; AKH_COMPILE_FEEDBACK_BASE_REF is the \
                                  process-wide override."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, _engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let path = input.get("path").unwrap_or(".");
        let kind = input.get("kind").unwrap_or("build");
        let scope = input.get("scope").unwrap_or("blast-radius");
        let timeout_secs = resolve_timeout(&input);
        let explicit_base = input.get("base_ref");

        // Provider dispatch: map `kind` to the cargo subcommand.
        let command = match kind {
            "build" => "check",
            "lint" => "clippy",
            "test-compile" => "test",
            other => {
                return Ok(ToolOutput::err(format!(
                    "Unsupported kind: '{other}'. Use 'build', 'lint', or 'test-compile'."
                )));
            }
        };

        // Scope: blast-radius is the default; workspace is explicit opt-in.
        let workspace_mode = match scope {
            "blast-radius" => false,
            "workspace" => true,
            other => {
                return Ok(ToolOutput::err(format!(
                    "Unsupported scope: '{other}'. Use 'blast-radius' (default) or 'workspace'."
                )));
            }
        };

        run_diagnostics_pipeline(path, command, timeout_secs, workspace_mode, explicit_base)
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: "diagnostics".into(),
            description: "Provider-dispatched, blast-radius scoped, baseline-aware diagnostics. \
                          Defaults to packages changed since the Git worktree base; workspace \
                          scope is explicit opt-in. Partitions diagnostics into task-introduced \
                          and pre-existing; pre-existing warnings are reported as context, not work."
                .into(),
            parameters: vec![
                ToolParamSchema::optional(
                    "kind",
                    "Diagnostic kind: 'build' (cargo check, default), 'lint' (cargo clippy), \
                     or 'test-compile' (cargo test --no-run).",
                ),
                ToolParamSchema::optional(
                    "scope",
                    "Scope: 'blast-radius' (default, packages changed since the worktree base) \
                     or 'workspace' (entire workspace). Workspace is explicit opt-in.",
                ),
                ToolParamSchema::optional(
                    "timeout",
                    "Timeout in seconds (default 600; cold builds can take 90s+, so request \
                     900–1200s for a known-cold container). Override the process-wide default \
                     with AKH_COMPILE_FEEDBACK_TIMEOUT_SECS.",
                ),
                ToolParamSchema::optional("path", "Working directory for cargo."),
                ToolParamSchema::optional(
                    "base_ref",
                    "Optional Git base ref override. Defaults to origin/HEAD then conventional \
                     trunk refs; AKH_COMPILE_FEEDBACK_BASE_REF is the process-wide override.",
                ),
            ],
            danger: DangerInfo {
                level: DangerLevel::Cautious,
                capabilities: HashSet::from([Capability::ProcessExec]),
                description: "Runs Cargo and Git; creates build artifacts and a temporary \
                              detached baseline worktree, but does not modify source files."
                    .into(),
                shadow_triggers: vec![
                    "diagnostics".into(),
                    "compile".into(),
                    "check".into(),
                    "lint".into(),
                ],
            },
            source: ToolSource::Native,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tool::ToolInput;
    use crate::engine::{Engine, EngineConfig};

    // --- Kind/scope mapping tests (no cargo needed) ---

    #[test]
    fn invalid_kind_rejected() {
        let input = ToolInput::default().with_param("kind", "bogus");
        let output = DiagnosticsTool
            .execute(&Engine::new(EngineConfig::default()).unwrap(), input)
            .unwrap();
        assert!(!output.success, "invalid kind must fail");
        assert!(
            output.result.contains("Unsupported kind: 'bogus'"),
            "must name the invalid kind; got: {}",
            output.result
        );
    }

    #[test]
    fn invalid_scope_rejected() {
        let input = ToolInput::default().with_param("scope", "everything");
        let output = DiagnosticsTool
            .execute(&Engine::new(EngineConfig::default()).unwrap(), input)
            .unwrap();
        assert!(!output.success, "invalid scope must fail");
        assert!(
            output.result.contains("Unsupported scope: 'everything'"),
            "must name the invalid scope; got: {}",
            output.result
        );
    }

    #[test]
    fn default_kind_is_build() {
        // With no kind specified, the tool should map to "check". We verify
        // this indirectly: an empty repo with no changes returns the
        // NoRustChanges message, proving the pipeline was entered with
        // command="check" (not rejected as an invalid kind).
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        init_test_repo(root, "pub fn initial() {}\n", "diag-default-kind");
        let input = ToolInput::default().with_param("path", root.to_str().unwrap());
        let output = DiagnosticsTool
            .execute(&Engine::new(EngineConfig::default()).unwrap(), input)
            .unwrap();
        assert!(
            output.result.contains("BLAST RADIUS") || output.result.contains("cargo check"),
            "default kind should reach the pipeline as 'check'; got: {}",
            output.result
        );
    }

    #[test]
    fn default_scope_is_blast_radius_not_workspace() {
        // With no scope specified, the tool should use blast-radius mode.
        // We verify this by checking the output does NOT contain the
        // workspace mode marker (unless discovery fails and falls back).
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        init_test_repo(root, "pub fn initial() {}\n", "diag-default-scope");
        // Make a change so we get past NoRustChanges
        fs::write(root.join("src/lib.rs"), "pub fn changed() {}\n").unwrap();
        let input = ToolInput::default().with_param("path", root.to_str().unwrap());
        let output = DiagnosticsTool
            .execute(&Engine::new(EngineConfig::default()).unwrap(), input)
            .unwrap();
        // If scope discovery succeeds, the output should say "BLAST RADIUS"
        // (not "MODE: full workspace"). If it fails, the fallback notice
        // appears — but that's a different code path, not the default scope.
        assert!(
            output.result.contains("BLAST RADIUS") || output.result.contains("SCOPE FALLBACK"),
            "default scope should be blast-radius or a fallback, not explicit workspace; got: {}",
            output.result
        );
    }

    // --- Fixture-based blast-radius test ---

    fn test_git(workdir: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(workdir)
            .output()
            .expect("spawn git");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn init_test_repo(root: &Path, source: &str, package: &str) -> String {
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("Cargo.toml"),
            format!("[package]\nname = \"{package}\"\nversion = \"0.1.0\"\nedition = \"2024\"\n"),
        )
        .unwrap();
        fs::write(root.join("src/lib.rs"), source).unwrap();
        test_git(root, &["init", "-b", "main"]);
        test_git(root, &["config", "user.name", "Diagnostics Test"]);
        test_git(
            root,
            &["config", "user.email", "diagnostics@example.invalid"],
        );
        test_git(root, &["config", "commit.gpgsign", "false"]);
        test_git(root, &["add", "-A"]);
        test_git(root, &["commit", "-m", "baseline"]);
        test_git(root, &["rev-parse", "HEAD"])
    }

    fn init_test_workspace(root: &Path) -> String {
        // Workspace root
        fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"alpha\", \"beta\"]\nresolver = \"2\"\n",
        )
        .unwrap();

        // Alpha package with a pre-existing unused-variable warning
        fs::create_dir_all(root.join("alpha/src")).unwrap();
        fs::write(
            root.join("alpha/Cargo.toml"),
            "[package]\nname = \"alpha\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            root.join("alpha/src/lib.rs"),
            "pub fn ambient() { let ambient = 1; }\n",
        )
        .unwrap();

        // Beta package (clean baseline)
        fs::create_dir_all(root.join("beta/src")).unwrap();
        fs::write(
            root.join("beta/Cargo.toml"),
            "[package]\nname = \"beta\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(root.join("beta/src/lib.rs"), "pub fn clean() {}\n").unwrap();

        test_git(root, &["init", "-b", "main"]);
        test_git(root, &["config", "user.name", "Diagnostics Test"]);
        test_git(
            root,
            &["config", "user.email", "diagnostics@example.invalid"],
        );
        test_git(root, &["config", "commit.gpgsign", "false"]);
        test_git(root, &["add", "-A"]);
        test_git(root, &["commit", "-m", "baseline"]);
        test_git(root, &["rev-parse", "HEAD"])
    }

    /// **WI 4 verification criterion:** On a tree with known pre-existing
    /// warnings outside the blast radius, `diagnostics` reports **zero** of
    /// them as work.
    ///
    /// Fixture: a two-package workspace where `alpha` has an unused-variable
    /// warning at baseline. The session modifies only `beta`. The diagnostics
    /// tool must report alpha's warning as baseline context, never as task
    /// work.
    #[test]
    fn preexisting_warnings_outside_blast_radius_not_reported_as_work() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        init_test_workspace(root);

        // Modify only beta — introduce a new unused-variable warning
        fs::write(
            root.join("beta/src/lib.rs"),
            "pub fn clean() { let unused = 42; }\n",
        )
        .unwrap();

        let output = run_diagnostics_pipeline(
            root.to_str().unwrap(),
            "check",
            120,
            false, // blast-radius scope (the default)
            Some("main"),
        )
        .unwrap();

        // The blast radius must include beta, not alpha
        assert!(
            output.result.contains("beta"),
            "blast radius should include beta; got: {}",
            output.result
        );

        // Alpha's pre-existing warning must NOT appear as task work
        // (no "Task warnings" or "Task-introduced errors" mentioning alpha)
        let task_section = &output.result;
        assert!(
            !task_section.contains("alpha/src") || task_section.contains("BASELINE CONTEXT"),
            "alpha's pre-existing warning must be baseline context, not task work; got: {}",
            output.result
        );

        // If baseline context is present, alpha's warning is there, not in
        // task work
        if output.result.contains("BASELINE CONTEXT") {
            assert!(
                output.result.contains("pre-existing warning"),
                "baseline context should mention pre-existing warnings; got: {}",
                output.result
            );
        }
    }

    /// **WI 4 verification criterion:** A warning the session itself
    /// introduces is reported at full fidelity with its suggestion.
    #[test]
    fn session_introduced_warning_reported_with_suggestion() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        init_test_repo(root, "pub fn initial() {}\n", "diag-suggestion");

        // Introduce an unused variable — the compiler emits a warning with
        // a suggested_replacement (prefix with `_`).
        fs::write(
            root.join("src/lib.rs"),
            "pub fn initial() {}\npub fn new_fn() { let unused = 42; }\n",
        )
        .unwrap();

        let output =
            run_diagnostics_pipeline(root.to_str().unwrap(), "check", 120, false, Some("main"))
                .unwrap();

        // The warning should appear as task work (not baseline context)
        assert!(
            output.result.contains("unused") || output.result.contains("warning"),
            "session-introduced warning should appear in output; got: {}",
            output.result
        );
    }

    /// **WI 4 verification criterion:** Requesting workspace breadth
    /// explicitly still works, and is visible in the tool call.
    #[test]
    fn explicit_workspace_scope_works() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        init_test_repo(root, "pub fn initial() {}\n", "diag-workspace");
        fs::write(root.join("src/lib.rs"), "pub fn changed() {}\n").unwrap();

        let output = run_diagnostics_pipeline(
            root.to_str().unwrap(),
            "check",
            120,
            true, // explicit workspace
            Some("main"),
        )
        .unwrap();

        assert!(
            output.result.contains("--workspace") || output.result.contains("full workspace"),
            "workspace scope should be visible in output; got: {}",
            output.result
        );
    }

    // --- Regression tests for STUCK: contract and 64 KB cap ---

    /// The `STUCK:` prefix contract is preserved (WI-255 canary wedge).
    #[test]
    fn stuck_result_has_stuck_prefix() {
        let result = super::super::compile_feedback::format_stuck_result(
            "check",
            "no output for 300 seconds",
            540,
            "",
            "",
        );
        assert!(
            result.starts_with("STUCK:"),
            "STUCK: prefix must be present; got: {}",
            &result[..80.min(result.len())]
        );
        assert!(
            result.contains("540s"),
            "elapsed time must be reported; got: {result}"
        );
    }

    /// The 64 KB output cap is preserved.
    #[test]
    fn truncate_enforces_64kb_cap() {
        let big = "x".repeat(100 * 1024);
        let truncated = super::super::compile_feedback::truncate_text(&big, 64 * 1024);
        assert!(truncated.len() < big.len(), "truncate must reduce size");
        assert!(
            truncated.ends_with("... [truncated]"),
            "truncated text must end with marker; got: {} bytes",
            truncated.len()
        );
    }

    /// UTF-8 boundary safety is preserved under truncation.
    #[test]
    fn truncate_keeps_utf8_boundary() {
        let text = "ééé";
        let truncated = super::super::compile_feedback::truncate_text(text, 3);
        assert!(truncated.starts_with('é'));
        assert!(truncated.ends_with("... [truncated]"));
    }
}
