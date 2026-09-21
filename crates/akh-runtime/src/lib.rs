//! Provider processes with host-owned tools and continuation. No cognitive engine dependency.
mod claude;
mod codex;
mod process;
pub mod profile;
mod questions;

use akh_session_types::{
    ApprovalPolicy, AssembledContext, ExecutionMode, RuntimeEvent, RuntimeJobRequest, RuntimeKind,
    ToolResult,
    neural_spans::{SpanCapture, SpanHit, SpanSpace},
};
use serde_json::{Value, json};
use std::{
    path::PathBuf,
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
};

pub use akh_session_types::ProviderConfig;

pub trait Host {
    /// Optional provider usage observation; never participates in tool authority.
    fn usage(&mut self, _usage: &Value) {}
    fn event(&mut self, event: RuntimeEvent) -> Result<(), String>;
    fn tool(&mut self, id: &str, name: &str, arguments: Value) -> Result<ToolResult, String>;
    fn approval(&mut self, id: &str, name: &str, arguments: &Value) -> Result<bool, String>;
    fn question(&mut self, id: &str, questions: &Value) -> Result<Value, String>;
    /// Same-space neural recall for the active turn. `Ok(None)` means the host
    /// cannot serve recall (older daemon, missing capability or policy); the
    /// runtime then continues without recalled context. Hosts never fabricate hits.
    fn recall(
        &mut self,
        _id: &str,
        _space: &SpanSpace,
        _vector: &[u8],
        _limit: usize,
    ) -> Result<Option<Vec<SpanHit>>, String> {
        Ok(None)
    }
    /// Deliver bounded captures of the active turn's accepted input. `Ok(false)`
    /// means the host does not retain them; the turn itself is unaffected.
    fn spans(&mut self, _id: &str, _captures: &[SpanCapture]) -> Result<bool, String> {
        Ok(false)
    }
    /// Ask the cognitive core to assemble this turn's context to budget.
    ///
    /// `Ok(None)` means the host has no core to defer to — no core configured
    /// for this session, or a session daemon that predates the route — and the
    /// runtime then windows the conversation locally, exactly as before. A host
    /// never fabricates an assembly; the returned messages are the core's.
    fn assemble_context(
        &mut self,
        _messages: &[Value],
        _window_tokens: usize,
        _reserved_output_tokens: usize,
    ) -> Result<Option<AssembledContext>, String> {
        Ok(None)
    }
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
    fn command(&self, executable: &std::path::Path) -> Result<Command, String> {
        let mut command = if let Some(launcher) = self.config.launcher.first() {
            let mut cmd = Command::new(launcher);
            cmd.args(&self.config.launcher[1..]);
            cmd.arg(executable);
            cmd
        } else {
            Command::new(executable)
        };
        if self.config.isolation.is_some() {
            command.env_clear();
        }
        command
            .current_dir(&self.cwd)
            .envs(profile::environment(self.config)?);
        Ok(command)
    }
}
