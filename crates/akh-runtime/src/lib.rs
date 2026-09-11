//! Provider processes with host-owned tools and continuation. No cognitive engine dependency.
mod claude;
mod codex;
mod process;

use akh_session_types::{
    ApprovalPolicy, ExecutionMode, RuntimeEvent, RuntimeJobRequest, RuntimeKind, ToolResult,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    pub revision: String,
    pub kind: RuntimeKind,
    pub working_dir: PathBuf,
    pub binary: PathBuf,
    pub node_binary: Option<PathBuf>,
    pub bridge_script: Option<PathBuf>,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    #[serde(default)]
    pub launcher: Vec<String>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub execution_mode: ExecutionMode,
    pub conformance_file: Option<PathBuf>,
    #[serde(default)]
    pub native_aliases: BTreeMap<String, String>,
    pub effort: Option<String>,
}
fn default_timeout() -> u64 {
    600
}

pub trait Host {
    fn event(&mut self, event: RuntimeEvent) -> Result<(), String>;
    fn tool(&mut self, id: &str, name: &str, arguments: Value) -> Result<ToolResult, String>;
    fn approval(&mut self, id: &str, name: &str, arguments: &Value) -> Result<bool, String>;
    fn question(&mut self, id: &str, questions: &Value) -> Result<Value, String>;
}

pub fn run(
    config: &ProviderConfig,
    request: &RuntimeJobRequest,
    host: &mut dyn Host,
    cancel: &AtomicBool,
) -> Result<String, String> {
    if config.kind != request.binding.kind
        || config.revision != request.binding.revision
        || config.execution_mode != request.binding.execution_mode
    {
        return Err("runtime binding does not match the installed profile".into());
    }
    if config.timeout_secs == 0 || !config.working_dir.is_absolute() || !config.working_dir.is_dir()
    {
        return Err(
            "runtime needs a positive timeout and an existing absolute private working directory"
                .into(),
        );
    }
    // A remote workspace path is never resolved on this host in brokered mode.
    let cwd = if config.execution_mode == ExecutionMode::Brokered {
        config.working_dir.clone()
    } else {
        let path = PathBuf::from(&request.workspace.root)
            .canonicalize()
            .map_err(|e| e.to_string())?;
        let allowed = config
            .working_dir
            .canonicalize()
            .map_err(|e| e.to_string())?;
        if !path.starts_with(allowed) {
            return Err("colocated workspace is outside the runtime's configured root".into());
        }
        path
    };
    let tools = request.tools.iter().map(|t|json!({"function":{"name":t.name,"description":t.description,"parameters":t.input_schema}})).collect();
    let mut instructions = request.instructions.clone();
    if config.execution_mode == ExecutionMode::Brokered {
        instructions.push_str("\nWorkspace operations MUST use the provided Akh tools. The working directory of this runtime is private runtime state, not the workspace. Native workspace tools are disabled. Never substitute local execution if an Akh tool fails.");
    }
    let mut turn = Turn {
        config,
        request,
        host,
        cancel,
        cwd,
        tools,
        instructions,
        prompt: request.input.provider_prompt(),
        session: request.provider_session.clone(),
    };
    match config.kind {
        RuntimeKind::Codex => codex::run(&mut turn),
        RuntimeKind::Claude => claude::run(&mut turn),
        RuntimeKind::Internal => {
            Err("internal runtime is served by the configured Akh cognitive endpoint".into())
        }
    }
}

struct Turn<'a> {
    config: &'a ProviderConfig,
    request: &'a RuntimeJobRequest,
    host: &'a mut dyn Host,
    cancel: &'a AtomicBool,
    cwd: PathBuf,
    tools: Vec<Value>,
    instructions: String,
    prompt: String,
    session: Option<String>,
}

impl Turn<'_> {
    fn text(&mut self, text: &str) -> Result<(), String> {
        if !text.is_empty() {
            self.host.event(RuntimeEvent::Text { text: text.into() })?;
        }
        Ok(())
    }
    fn tool(&mut self, id: &str, name: &str, arguments: Value) -> Result<Value, String> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err("runtime cancelled".into());
        }
        if !self.request.tools.iter().any(|t| t.name == name) {
            return Ok(json!(ToolResult::denied(name, None)));
        }
        Ok(json!(self.host.tool(id, name, arguments)?))
    }
    fn approve(&mut self, id: &str, name: &str, arguments: &Value) -> Result<bool, String> {
        if self.config.execution_mode == ExecutionMode::Brokered
            || self.request.approval_policy == ApprovalPolicy::ReadOnly
            || self.cancel.load(Ordering::Relaxed)
        {
            return Ok(false);
        }
        self.host.approval(id, name, arguments)
    }
    fn bind(&mut self, id: &str, model: Option<&str>) -> Result<(), String> {
        if id.is_empty() || self.session.as_ref().is_some_and(|old| old != id) {
            return Err("provider returned a different or empty session ID".into());
        }
        self.host.event(RuntimeEvent::Bound {
            provider_session: id.into(),
            model: model
                .map(str::to_owned)
                .or_else(|| self.request.binding.model.clone()),
        })?;
        self.session = Some(id.into());
        Ok(())
    }
    fn command(&self, executable: &std::path::Path) -> Command {
        let mut command = if let Some(launcher) = self.config.launcher.first() {
            let mut cmd = Command::new(launcher);
            cmd.args(&self.config.launcher[1..]);
            cmd.arg(executable);
            cmd
        } else {
            Command::new(executable)
        };
        command
            .current_dir(&self.cwd)
            .envs(&self.config.environment);
        command
    }
}
