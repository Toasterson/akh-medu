//! OpenCode profile validation and the per-turn configuration the bridge owns.
//!
//! OpenCode has no single "native tools off" switch comparable to Codex's
//! feature flags. Brokered turns therefore rely on OpenCode's permission engine
//! in two layers (pinned against OpenCode v1.18, see
//! docs/ai/research/2026-09-25-opencode-runtime-integration.md):
//!
//! 1. `Permission.disabled` removes every tool whose last matching rule is a
//!    `"*"`-pattern `deny` from the catalog the model sees, and
//! 2. every native tool asks the same ruleset before its effect, so a call
//!    that slips through is refused before anything runs.
//!
//! The bridge defines its own primary agent ([`AGENT`]) whose permission rules
//! are appended after every configuration layer, selects it explicitly for
//! each turn, and refuses profile inputs that could reintroduce executable
//! configuration (plugins, MCP servers, custom agents, formatters, LSP).
use akh_session_types::{ApprovalPolicy, ExecutionMode, ProviderConfig};
use miette::Diagnostic;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::time::Duration;
use thiserror::Error;

/// Name of the MCP server the bridge registers. OpenCode exposes its tools to
/// the model as `akh_<tool>`.
pub(super) const MCP_SERVER: &str = "akh";
/// The primary agent the bridge defines and selects for every turn.
pub(super) const AGENT: &str = "akh";
/// Permission pattern covering exactly the bridge's MCP tools.
const MCP_TOOLS: &str = "akh_*";

/// Native permission keys of the pinned OpenCode tool inventory. Brokered
/// turns deny each one explicitly in addition to the `"*"` rule, so a config
/// layer that already defines `"*"` (whose position a deep merge keeps) cannot
/// reorder the catch-all behind a native allow.
const NATIVE_PERMISSIONS: &[&str] = &[
    "read",
    "edit",
    "glob",
    "grep",
    "list",
    "bash",
    "task",
    "skill",
    "lsp",
    "question",
    "webfetch",
    "websearch",
    "codesearch",
    "todowrite",
    "todoread",
    "external_directory",
    "doom_loop",
    "plan_enter",
    "plan_exit",
];
/// Native permissions without workspace side effects (colocated mode only).
const READ_PERMISSIONS: &[&str] = &["read", "glob", "grep", "list", "lsp"];
/// Native permissions that never make sense under an Akh host: questions have
/// no ACP transport here and subagents/plan switches would leave the agent.
const HOST_DENIED: &[&str] = &["question", "task", "plan_enter", "plan_exit"];

/// Environment the bridge sets for every turn. A profile may not set these.
const BRIDGE_OWNED_ENV: &[&str] = &[
    "OPENCODE_CONFIG_CONTENT",
    "OPENCODE_PERMISSION",
    "OPENCODE_SERVER_PASSWORD",
    "OPENCODE_SERVER_USERNAME",
    "OPENCODE_CLIENT",
    "OPENCODE_PURE",
    "OPENCODE_DISABLE_AUTOUPDATE",
    "OPENCODE_DISABLE_PROJECT_CONFIG",
    "OPENCODE_DISABLE_CLAUDE_CODE",
    "OPENCODE_DISABLE_EXTERNAL_SKILLS",
    "OPENCODE_DISABLE_LSP_DOWNLOAD",
    "OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER",
];
/// `OPENCODE_*` variables a brokered profile may still set: model catalog
/// location only. Every other `OPENCODE_*` switch can widen the tool surface
/// (experimental tools, extra config layers) and is refused.
const BROKERED_PROFILE_ENV: &[&str] = &[
    "OPENCODE_DISABLE_MODELS_FETCH",
    "OPENCODE_MODELS_PATH",
    "OPENCODE_MODELS_URL",
];
/// Top-level keys a brokered profile's managed `opencode.json` may contain:
/// model/provider selection and presentation. Everything else (agents, MCP,
/// plugins, permissions, formatters, LSP, commands, instructions, server,
/// experimental) is either bridge-owned or can run code.
const BROKERED_CONFIG_KEYS: &[&str] = &[
    "$schema",
    "model",
    "small_model",
    "provider",
    "disabled_providers",
    "enabled_providers",
    "username",
    "compaction",
    "logLevel",
    "tool_output",
];
/// Managed files a brokered profile may place under the provider's config
/// directory. Directories such as `plugin/`, `tool/`, `agent/` and `command/`
/// hold executable or tool-defining content and are refused.
const BROKERED_MANAGED_FILES: &[&str] = &["opencode/opencode.json", "opencode/AGENTS.md"];
/// The generated configuration travels in one environment variable. Linux
/// bounds a single environment string at 128 KiB; stay well below it.
const MAX_CONFIG_BYTES: usize = 96 * 1024;

/// A runtime profile that the OpenCode bridge refuses to launch.
#[derive(Debug, Error, Diagnostic)]
pub enum OpenCodeProfileError {
    #[error(
        "{name} is set by the OpenCode bridge for every turn and cannot come from the runtime profile"
    )]
    #[diagnostic(
        code(akh::runtime::opencode::bridge_owned_environment),
        help(
            "remove {name} from `environment` / `isolation.environment_files`; the bridge derives it from the execution mode and approval policy"
        )
    )]
    BridgeOwnedEnvironment { name: String },

    #[error("brokered OpenCode profiles may not set {name}")]
    #[diagnostic(
        code(akh::runtime::opencode::brokered_environment),
        help(
            "brokered profiles may only set OPENCODE_DISABLE_MODELS_FETCH, OPENCODE_MODELS_PATH and OPENCODE_MODELS_URL; other OPENCODE_* switches add config layers or experimental tools. Remove {name}, or use execution_mode = \"colocated\" for local development only"
        )
    )]
    BrokeredEnvironment { name: String },

    #[error("brokered OpenCode requires a private `isolation` directory")]
    #[diagnostic(
        code(akh::runtime::opencode::isolation_required),
        help(
            "without isolation OpenCode reads the daemon's ambient ~/.config/opencode (plugins, MCP servers, agents). Add `isolation.directory` so HOME and XDG_CONFIG_HOME are private to this runtime"
        )
    )]
    IsolationRequired,

    #[error("brokered OpenCode profiles may not manage {path}")]
    #[diagnostic(
        code(akh::runtime::opencode::managed_file_forbidden),
        help(
            "only opencode/opencode.json and opencode/AGENTS.md may be managed; plugin, tool, agent and command directories define code or tools that bypass the Akh broker"
        )
    )]
    ManagedFileForbidden { path: String },

    #[error("{path} is not strict JSON: {reason}")]
    #[diagnostic(
        code(akh::runtime::opencode::managed_config_unreadable),
        help(
            "write the managed OpenCode config as plain JSON (no comments, no trailing commas) so it can be audited before launch"
        )
    )]
    ManagedConfigUnreadable { path: String, reason: String },

    #[error("{path} sets `{key}`, which a brokered OpenCode profile may not configure")]
    #[diagnostic(
        code(akh::runtime::opencode::managed_config_key_forbidden),
        help(
            "brokered profiles may set only $schema, model, small_model, provider, disabled_providers, enabled_providers, username, compaction, logLevel and tool_output; agents, permissions, MCP, plugins, formatters and LSP are owned by the bridge"
        )
    )]
    ManagedConfigKeyForbidden { path: String, key: String },

    #[error("{path} installs an npm package for provider `{provider}`")]
    #[diagnostic(
        code(akh::runtime::opencode::provider_package_forbidden),
        help(
            "a provider `npm` entry downloads and loads JavaScript into the runtime; use a built-in provider or an OpenAI-compatible endpoint in brokered mode"
        )
    )]
    ProviderPackageForbidden { path: String, provider: String },

    #[error("`{field}` is not used by the OpenCode runtime")]
    #[diagnostic(
        code(akh::runtime::opencode::unused_field),
        help(
            "OpenCode is a single executable driven over ACP; remove `{field}` so conformance evidence does not pin an unrelated file"
        )
    )]
    UnusedField { field: &'static str },

    #[error("the generated OpenCode configuration is {bytes} bytes; the limit is {limit}")]
    #[diagnostic(
        code(akh::runtime::opencode::instructions_too_large),
        help(
            "the turn's instructions become the Akh agent prompt inside OPENCODE_CONFIG_CONTENT; shorten the instructions or move stable guidance into the managed opencode/AGENTS.md"
        )
    )]
    InstructionsTooLarge { bytes: usize, limit: usize },

    #[error("runtime model `{model}` is not an OpenCode `provider/model` identifier")]
    #[diagnostic(
        code(akh::runtime::opencode::model_format),
        help(
            "bind the session to a model such as `openai/gpt-5` or `anthropic/claude-sonnet-4-5`; OpenCode never falls back to another model for the bridge"
        )
    )]
    ModelFormat { model: String },
}

impl OpenCodeProfileError {
    /// The stable diagnostic code, for callers that report through their own
    /// fault types instead of miette.
    pub fn diagnostic_code(&self) -> String {
        self.code()
            .map_or_else(|| "akh::runtime::opencode".to_owned(), |c| c.to_string())
    }

    /// The remedy, for callers that report through their own fault types.
    pub fn remedy(&self) -> Option<String> {
        self.help().map(|help| help.to_string())
    }
}

/// Which native tools an OpenCode turn may use, derived from the runtime's
/// execution mode and the session's approval policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ToolPolicy {
    /// Every native tool denied and hidden; only Akh MCP tools remain.
    Brokered,
    /// Colocated development, read-only session: native reads only.
    ColocatedReadOnly,
    /// Colocated development: native reads allowed, everything else asks the host.
    ColocatedAsk,
}

impl ToolPolicy {
    pub(super) fn resolve(mode: ExecutionMode, approval: ApprovalPolicy) -> Self {
        match (mode, approval) {
            (ExecutionMode::Brokered, _) => Self::Brokered,
            (ExecutionMode::Colocated, ApprovalPolicy::ReadOnly) => Self::ColocatedReadOnly,
            (ExecutionMode::Colocated, _) => Self::ColocatedAsk,
        }
    }

    /// OpenCode permission rules. OpenCode applies the last matching rule, so
    /// the catch-all must precede every specific key. That holds whether the
    /// map keeps insertion order or sorts (`*` sorts before every tool name),
    /// and no other pair of keys overlaps: native keys never match `akh_*`.
    pub(super) fn permission(self) -> Map<String, Value> {
        let mut rules = Map::new();
        let catch_all = match self {
            Self::Brokered | Self::ColocatedReadOnly => "deny",
            Self::ColocatedAsk => "ask",
        };
        rules.insert("*".into(), json!(catch_all));
        for name in NATIVE_PERMISSIONS {
            let action = match self {
                Self::Brokered => "deny",
                _ if READ_PERMISSIONS.contains(name) => "allow",
                _ if HOST_DENIED.contains(name) => "deny",
                Self::ColocatedReadOnly => "deny",
                Self::ColocatedAsk => "ask",
            };
            rules.insert((*name).into(), json!(action));
        }
        rules.insert(MCP_TOOLS.into(), json!("allow"));
        rules
    }

    fn brokered(self) -> bool {
        self == Self::Brokered
    }
}

/// Refuse a profile the OpenCode bridge cannot run safely. Called before every
/// launch and by the session service at admission and `verify` time.
pub fn validate_profile(profile: &ProviderConfig) -> Result<(), OpenCodeProfileError> {
    if profile.node_binary.is_some() {
        return Err(OpenCodeProfileError::UnusedField {
            field: "node_binary",
        });
    }
    if profile.bridge_script.is_some() {
        return Err(OpenCodeProfileError::UnusedField {
            field: "bridge_script",
        });
    }
    let brokered = profile.execution_mode() == ExecutionMode::Brokered;
    let file_names = profile
        .isolation
        .iter()
        .flat_map(|scope| scope.environment_files.keys());
    for name in profile.environment.keys().chain(file_names) {
        if BRIDGE_OWNED_ENV.contains(&name.as_str()) {
            return Err(OpenCodeProfileError::BridgeOwnedEnvironment { name: name.clone() });
        }
        if brokered
            && name.starts_with("OPENCODE_")
            && !BROKERED_PROFILE_ENV.contains(&name.as_str())
        {
            return Err(OpenCodeProfileError::BrokeredEnvironment { name: name.clone() });
        }
    }
    if !brokered {
        return Ok(());
    }
    let scope = profile
        .isolation
        .as_ref()
        .ok_or(OpenCodeProfileError::IsolationRequired)?;
    for (path, value) in &scope.managed_files {
        let normalized = path.trim_start_matches("./");
        if !normalized.starts_with("opencode/") && normalized != "opencode" {
            continue;
        }
        if !BROKERED_MANAGED_FILES.contains(&normalized) {
            return Err(OpenCodeProfileError::ManagedFileForbidden { path: path.clone() });
        }
        if normalized == "opencode/opencode.json" {
            audit_managed_config(path, &crate::profile::expand(value, profile))?;
        }
    }
    Ok(())
}

fn audit_managed_config(path: &str, text: &str) -> Result<(), OpenCodeProfileError> {
    let unreadable = |reason: String| OpenCodeProfileError::ManagedConfigUnreadable {
        path: path.into(),
        reason,
    };
    let value: Value = serde_json::from_str(text).map_err(|e| unreadable(e.to_string()))?;
    let object = value
        .as_object()
        .ok_or_else(|| unreadable("the top level must be an object".into()))?;
    if let Some(key) = object
        .keys()
        .find(|key| !BROKERED_CONFIG_KEYS.contains(&key.as_str()))
    {
        return Err(OpenCodeProfileError::ManagedConfigKeyForbidden {
            path: path.into(),
            key: key.clone(),
        });
    }
    if let Some(providers) = object.get("provider").and_then(Value::as_object)
        && let Some((provider, _)) = providers
            .iter()
            .find(|(_, entry)| entry.get("npm").is_some())
    {
        return Err(OpenCodeProfileError::ProviderPackageForbidden {
            path: path.into(),
            provider: provider.clone(),
        });
    }
    Ok(())
}

/// A model binding in OpenCode's `provider/model` form.
pub(super) fn model_id(model: &str) -> Result<&str, OpenCodeProfileError> {
    match model.split_once('/') {
        Some((provider, name)) if !provider.trim().is_empty() && !name.trim().is_empty() => {
            Ok(model)
        }
        _ => Err(OpenCodeProfileError::ModelFormat {
            model: model.into(),
        }),
    }
}

/// The per-turn configuration layer (`OPENCODE_CONFIG_CONTENT`), the highest
/// precedence file-like layer OpenCode loads.
pub(super) fn config_content(
    policy: ToolPolicy,
    instructions: &str,
    tool_timeout: Duration,
) -> Result<String, OpenCodeProfileError> {
    let permission = Value::Object(policy.permission());
    let mut config = json!({
        "$schema": "https://opencode.ai/config.json",
        "default_agent": AGENT,
        "autoupdate": false,
        "share": "disabled",
        "permission": permission,
        "experimental": {
            // Akh tools may legitimately run for most of the turn; the MCP
            // client default (60 s) would abandon them mid-effect.
            "mcp_timeout": u64::try_from(tool_timeout.as_millis()).unwrap_or(u64::MAX).max(1),
            // A denied or declined native call is reported to the model, which
            // then uses the Akh tools, instead of ending the turn.
            "continue_loop_on_deny": true,
        },
        "agent": {
            AGENT: {
                "mode": "primary",
                "description": "Akh-Medu session agent",
                "prompt": instructions,
                "permission": permission,
            }
        }
    });
    if policy.brokered() {
        // Nothing may touch files behind the broker's back: no formatter runs,
        // no language servers start, no snapshot repository is written.
        config["snapshot"] = json!(false);
        config["formatter"] = json!(false);
        config["lsp"] = json!(false);
    }
    let text = config.to_string();
    if text.len() > MAX_CONFIG_BYTES {
        return Err(OpenCodeProfileError::InstructionsTooLarge {
            bytes: text.len(),
            limit: MAX_CONFIG_BYTES,
        });
    }
    Ok(text)
}

/// Environment the bridge adds on top of the profile environment.
pub(super) fn bridge_environment(
    policy: ToolPolicy,
    instructions: &str,
    tool_timeout: Duration,
    server_password: &str,
) -> Result<BTreeMap<String, String>, OpenCodeProfileError> {
    let mut env = BTreeMap::from([
        (
            "OPENCODE_CONFIG_CONTENT".to_owned(),
            config_content(policy, instructions, tool_timeout)?,
        ),
        // `opencode acp` also serves its HTTP API on loopback; nobody but this
        // process may drive the session through it.
        (
            "OPENCODE_SERVER_PASSWORD".to_owned(),
            server_password.to_owned(),
        ),
        // An update would replace the measured executable mid-deployment.
        ("OPENCODE_DISABLE_AUTOUPDATE".to_owned(), "1".to_owned()),
    ]);
    if policy.brokered() {
        env.insert(
            "OPENCODE_PERMISSION".into(),
            Value::Object(policy.permission()).to_string(),
        );
        for flag in [
            // No external plugins: they run code outside every permission check.
            "OPENCODE_PURE",
            "OPENCODE_DISABLE_PROJECT_CONFIG",
            "OPENCODE_DISABLE_CLAUDE_CODE",
            "OPENCODE_DISABLE_EXTERNAL_SKILLS",
            "OPENCODE_DISABLE_LSP_DOWNLOAD",
            "OPENCODE_EXPERIMENTAL_DISABLE_FILEWATCHER",
        ] {
            env.insert(flag.into(), "1".into());
        }
    }
    Ok(env)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(mode: ExecutionMode, isolation: Option<Value>) -> ProviderConfig {
        serde_json::from_value(json!({
            "isolation": isolation, "model_endpoint": null, "revision": "1", "kind": "opencode",
            "working_dir": "/w", "binary": "/b", "node_binary": null, "bridge_script": null,
            "conformance_file": null, "effort": null, "execution_mode": mode,
        }))
        .unwrap()
    }

    fn rule<'a>(rules: &'a Map<String, Value>, tool: &str) -> &'a str {
        // Mirror OpenCode's evaluation: the last rule whose key matches wins.
        let matches = |key: &str| match key.strip_suffix('*') {
            Some(prefix) => tool.starts_with(prefix),
            None => key == tool,
        };
        rules
            .iter()
            .rev()
            .find(|(key, _)| matches(key))
            .and_then(|(_, action)| action.as_str())
            .unwrap()
    }

    #[test]
    fn brokered_policy_denies_every_native_tool_and_allows_only_akh_tools() {
        let rules = ToolPolicy::Brokered.permission();
        assert_eq!(rules.keys().next().map(String::as_str), Some("*"));
        // Only the catch-all overlaps other keys; the rest is order-free.
        for (key, _) in rules.iter().skip(1) {
            let other = key.strip_suffix('*').unwrap_or(key);
            assert!(
                rules
                    .keys()
                    .skip(1)
                    .filter(|k| *k != key)
                    .all(|k| !k.starts_with(other)),
                "{key} overlaps another rule"
            );
        }
        for tool in NATIVE_PERMISSIONS
            .iter()
            .chain(&["write", "unknown_future_tool"])
        {
            assert_eq!(rule(&rules, tool), "deny", "{tool}");
        }
        assert_eq!(rule(&rules, "akh_shell_exec"), "allow");
        assert_eq!(rule(&rules, "other_mcp_tool"), "deny");
    }

    #[test]
    fn colocated_policies_follow_the_approval_policy() {
        assert_eq!(
            ToolPolicy::resolve(ExecutionMode::Colocated, ApprovalPolicy::ReadOnly),
            ToolPolicy::ColocatedReadOnly
        );
        assert_eq!(
            ToolPolicy::resolve(ExecutionMode::Brokered, ApprovalPolicy::AllowGranted),
            ToolPolicy::Brokered
        );
        let read_only = ToolPolicy::ColocatedReadOnly.permission();
        assert_eq!(rule(&read_only, "read"), "allow");
        assert_eq!(rule(&read_only, "bash"), "deny");
        assert_eq!(rule(&read_only, "edit"), "deny");
        let ask = ToolPolicy::ColocatedAsk.permission();
        assert_eq!(rule(&ask, "bash"), "ask");
        assert_eq!(rule(&ask, "grep"), "allow");
        assert_eq!(rule(&ask, "question"), "deny");
        assert_eq!(rule(&ask, "task"), "deny");
        assert_eq!(rule(&ask, "akh_read"), "allow");
    }

    #[test]
    fn generated_config_selects_the_akh_agent_and_disables_side_channels() {
        let text =
            config_content(ToolPolicy::Brokered, "be useful", Duration::from_secs(600)).unwrap();
        let config: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(config["default_agent"], AGENT);
        assert_eq!(config["agent"][AGENT]["prompt"], "be useful");
        assert_eq!(config["agent"][AGENT]["mode"], "primary");
        assert_eq!(config["agent"][AGENT]["permission"]["*"], "deny");
        assert_eq!(config["autoupdate"], false);
        assert_eq!(config["experimental"]["mcp_timeout"], 600_000);
        assert_eq!(config["experimental"]["continue_loop_on_deny"], true);
        for key in ["snapshot", "formatter", "lsp"] {
            assert_eq!(config[key], false, "{key}");
        }
        let colocated: Value = serde_json::from_str(
            &config_content(ToolPolicy::ColocatedAsk, "x", Duration::from_secs(1)).unwrap(),
        )
        .unwrap();
        assert!(colocated.get("lsp").is_none());
        let huge = "x".repeat(MAX_CONFIG_BYTES);
        assert!(matches!(
            config_content(ToolPolicy::Brokered, &huge, Duration::from_secs(1)),
            Err(OpenCodeProfileError::InstructionsTooLarge { .. })
        ));
        let env = bridge_environment(ToolPolicy::Brokered, "x", Duration::from_secs(1), "secret")
            .unwrap();
        assert_eq!(env["OPENCODE_PURE"], "1");
        assert_eq!(env["OPENCODE_SERVER_PASSWORD"], "secret");
        assert!(env.contains_key("OPENCODE_PERMISSION"));
        let dev = bridge_environment(
            ToolPolicy::ColocatedAsk,
            "x",
            Duration::from_secs(1),
            "secret",
        )
        .unwrap();
        assert!(!dev.contains_key("OPENCODE_PURE"));
        assert_eq!(dev["OPENCODE_DISABLE_AUTOUPDATE"], "1");
    }

    #[test]
    fn brokered_profiles_refuse_ambient_config_and_executable_inputs() {
        let isolation = |files: Value| Some(json!({"directory": "/p", "managed_files": files}));
        assert!(matches!(
            validate_profile(&profile(ExecutionMode::Brokered, None)),
            Err(OpenCodeProfileError::IsolationRequired)
        ));
        validate_profile(&profile(ExecutionMode::Colocated, None)).unwrap();
        validate_profile(&profile(
            ExecutionMode::Brokered,
            isolation(json!({
                "opencode/opencode.json": r#"{"model":"openai/gpt-5","provider":{"openai":{"options":{}}}}"#,
                "opencode/AGENTS.md": "notes",
                "codex/config.toml": "ignored",
            })),
        ))
        .unwrap();
        for (files, expected) in [
            (
                json!({"opencode/plugin/x.ts": "export default {}"}),
                "managed_file_forbidden",
            ),
            (
                json!({"opencode/opencode.json": r#"{"mcp":{"x":{"type":"local"}}}"#}),
                "managed_config_key_forbidden",
            ),
            (
                json!({"opencode/opencode.json": r#"{"plugin":["x"]}"#}),
                "managed_config_key_forbidden",
            ),
            (
                json!({"opencode/opencode.json": "{\"model\":\"a/b\",}"}),
                "managed_config_unreadable",
            ),
            (
                json!({"opencode/opencode.json": r#"{"provider":{"x":{"npm":"evil"}}}"#}),
                "provider_package_forbidden",
            ),
        ] {
            let error =
                validate_profile(&profile(ExecutionMode::Brokered, isolation(files))).unwrap_err();
            let code = error.code().unwrap().to_string();
            assert!(code.ends_with(expected), "{code} != {expected}");
        }
        let mut env = profile(ExecutionMode::Brokered, isolation(json!({})));
        env.environment
            .insert("OPENCODE_CONFIG".into(), "/x.json".into());
        assert!(matches!(
            validate_profile(&env),
            Err(OpenCodeProfileError::BrokeredEnvironment { .. })
        ));
        env.environment.clear();
        env.environment
            .insert("OPENCODE_DISABLE_MODELS_FETCH".into(), "1".into());
        validate_profile(&env).unwrap();
        let mut owned = profile(ExecutionMode::Colocated, None);
        owned
            .environment
            .insert("OPENCODE_PERMISSION".into(), "{}".into());
        assert!(matches!(
            validate_profile(&owned),
            Err(OpenCodeProfileError::BridgeOwnedEnvironment { .. })
        ));
        let mut node = profile(ExecutionMode::Colocated, None);
        node.node_binary = Some("/usr/bin/node".into());
        assert!(matches!(
            validate_profile(&node),
            Err(OpenCodeProfileError::UnusedField {
                field: "node_binary"
            })
        ));
    }

    #[test]
    fn models_must_name_their_provider() {
        assert_eq!(model_id("openai/gpt-5").unwrap(), "openai/gpt-5");
        assert!(model_id("gpt-5").is_err());
        assert!(model_id("/gpt-5").is_err());
    }
}
