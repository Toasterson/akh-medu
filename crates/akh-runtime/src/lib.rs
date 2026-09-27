//! Provider processes with host-owned tools and continuation. No cognitive engine dependency.
mod claude;
mod codex;
pub mod opencode;
mod process;
pub mod profile;
mod questions;

use akh_session_types::{
    ApprovalPolicy, AssembledContext, ExecutionMode, RuntimeEvent, RuntimeJobRequest, RuntimeKind,
    ToolResult,
    neural_spans::{SpanCapture, SpanHit, SpanSpace},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs::File,
    io::Read,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrainingWorkerError {
    /// No host callback may have taken effect; retry policy remains with akhrtd.
    Boundary(String),
    /// A host tool callback occurred and the worker did not finish durably.
    UnknownOutcome(String),
    /// The daemon explicitly cancelled the managed process.
    Cancelled(String),
}

impl std::fmt::Display for TrainingWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Boundary(message) | Self::UnknownOutcome(message) | Self::Cancelled(message) => {
                f.write_str(message)
            }
        }
    }
}

fn artifact_sha256(path: &std::path::Path) -> Result<String, String> {
    let mut file =
        File::open(path).map_err(|e| format!("cannot open pinned training artifact: {e}"))?;
    let mut hash = Sha256::new();
    let mut bytes = [0u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut bytes)
            .map_err(|e| format!("cannot hash pinned training artifact: {e}"))?;
        if count == 0 {
            break;
        }
        hash.update(&bytes[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Run a pinned provider-owned training worker while retaining all tool
/// authority in the daemon Host. The child receives no workspace credential.
pub fn run_training_worker(
    config: &ProviderConfig,
    request: &RuntimeJobRequest,
    host: &mut dyn Host,
    cancel: &AtomicBool,
) -> Result<String, TrainingWorkerError> {
    let fail = TrainingWorkerError::Boundary;
    if config.kind != RuntimeKind::Internal
        || request.binding.kind != RuntimeKind::Internal
        || config.revision != request.binding.revision
        || config.execution_mode() != ExecutionMode::Brokered
        || request.binding.execution_mode != ExecutionMode::Brokered
        || config.timeout_secs == 0
        || !config.working_dir.is_absolute()
        || !config.working_dir.is_dir()
    {
        return Err(fail(
            "training worker requires a matching brokered internal runtime binding".into(),
        ));
    }
    let worker = config
        .training_worker
        .as_ref()
        .ok_or_else(|| fail("training worker missing".into()))?;
    if config.isolation.is_none() {
        return Err(fail(
            "training worker requires an isolated provider environment".into(),
        ));
    }
    for (path, expected) in [
        (&worker.executable, &worker.executable_sha256),
        (&worker.worker_config_path, &worker.worker_config_sha256),
    ] {
        if !path.is_absolute()
            || !path.is_file()
            || artifact_sha256(path).map_err(&fail)? != *expected
        {
            return Err(fail("pinned training artifact is absent or changed".into()));
        }
    }
    match (&worker.script, &worker.script_sha256) {
        (Some(path), Some(expected))
            if path.is_absolute()
                && path.is_file()
                && artifact_sha256(path).map_err(&fail)? == *expected => {}
        (None, None) => {}
        _ => {
            return Err(fail(
                "training script path and SHA256 pin must match".into(),
            ));
        }
    }
    if request.binding.model.as_deref() != Some(worker.model.as_str()) {
        return Err(fail(
            "training worker model does not match runtime binding".into(),
        ));
    }

    let tools = request.tools.iter().map(|tool| json!({"function":{"name":tool.name,"description":tool.description,"parameters":tool.input_schema}})).collect();
    let mut turn = Turn {
        config,
        request,
        host,
        cancel,
        cwd: config.working_dir.clone(),
        tools,
        instructions: request.instructions.clone(),
        prompt: request.input.provider_prompt(),
        session: request.provider_session.clone(),
    };
    let mut command = turn.command(&worker.executable).map_err(&fail)?;
    if let Some(script) = &worker.script {
        command.arg(script);
    }
    let mut process = process::Process::spawn(command, config.timeout_secs).map_err(&fail)?;
    process
        .send(&json!({"type":"start","request":request,
        "config_path":worker.worker_config_path,"config_sha256":worker.worker_config_sha256}))
        .map_err(TrainingWorkerError::UnknownOutcome)?;

    let mut ids = HashSet::new();
    let mut tool_calls = 0u32;
    let mut progress = 0u32;
    loop {
        let frame = process.recv(cancel).map_err(|message| {
            if cancel.load(Ordering::Relaxed) {
                TrainingWorkerError::Cancelled(message)
            } else {
                TrainingWorkerError::UnknownOutcome(message)
            }
        })?;
        let uncertain = TrainingWorkerError::UnknownOutcome;
        let kind = frame
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| uncertain("training worker frame has no type".into()))?;
        let exact_keys = |allowed: &[&str]| {
            frame.as_object().is_some_and(|object| {
                object.len() == allowed.len()
                    && object.keys().all(|key| allowed.contains(&key.as_str()))
            })
        };
        match kind {
            "progress" => {
                if !exact_keys(&["type", "phase"]) && !exact_keys(&["type", "phase", "usage"]) {
                    return Err(uncertain(
                        "training progress frame has unexpected fields".into(),
                    ));
                }
                progress = progress
                    .checked_add(1)
                    .ok_or_else(|| uncertain("training progress overflow".into()))?;
                if progress > 100_000 {
                    return Err(uncertain("training progress event limit exceeded".into()));
                }
                let phase = frame
                    .get("phase")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty() && v.len() <= 128)
                    .ok_or_else(|| uncertain("training progress phase is invalid".into()))?;
                let usage = frame.get("usage").cloned().unwrap_or_else(|| json!({}));
                if !usage.is_object() {
                    return Err(uncertain(
                        "training progress usage must be an object".into(),
                    ));
                }
                turn.host
                    .usage(&json!({"training":{"phase":phase,"usage":usage}}));
            }
            "tool_call" => {
                if !exact_keys(&["type", "id", "name", "arguments"]) {
                    return Err(uncertain(
                        "training tool frame has unexpected fields".into(),
                    ));
                }
                tool_calls = tool_calls
                    .checked_add(1)
                    .ok_or_else(|| uncertain("training tool count overflow".into()))?;
                if tool_calls > worker.max_tool_calls {
                    return Err(uncertain("training worker tool limit exceeded".into()));
                }
                let id = frame
                    .get("id")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty() && v.len() <= 512)
                    .ok_or_else(|| uncertain("training tool call ID is invalid".into()))?;
                if !ids.insert(id.to_owned()) {
                    return Err(uncertain("duplicate training tool call ID".into()));
                }
                let name = frame
                    .get("name")
                    .and_then(Value::as_str)
                    .filter(|v| !v.is_empty())
                    .ok_or_else(|| uncertain("training tool name is invalid".into()))?;
                if !request.tools.iter().any(|tool| tool.name == name) {
                    return Err(uncertain(
                        "training worker selected a tool outside the request catalog".into(),
                    ));
                }
                let arguments = frame
                    .get("arguments")
                    .cloned()
                    .filter(Value::is_object)
                    .ok_or_else(|| uncertain("training tool arguments must be an object".into()))?;
                let result = turn
                    .tool(id, name, arguments)
                    .map_err(TrainingWorkerError::UnknownOutcome)?;
                process
                    .send(&json!({"type":"tool_result","id":id,"result":result}))
                    .map_err(TrainingWorkerError::UnknownOutcome)?;
            }
            "completed" => {
                if !exact_keys(&["type", "output"]) {
                    return Err(uncertain(
                        "training completion frame has unexpected fields".into(),
                    ));
                }
                let output = frame
                    .get("output")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty() && text.len() <= 4 * 1024 * 1024)
                    .ok_or_else(|| {
                        uncertain("training worker completion output is invalid".into())
                    })?;
                return Ok(output.to_owned());
            }
            "error" => {
                if !exact_keys(&["type", "message"]) {
                    return Err(uncertain(
                        "training error frame has unexpected fields".into(),
                    ));
                }
                let message = frame
                    .get("message")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty() && text.len() <= 4096)
                    .unwrap_or("training worker reported an invalid error");
                return Err(TrainingWorkerError::UnknownOutcome(message.into()));
            }
            _ => {
                return Err(uncertain(
                    "training worker emitted an unsupported frame".into(),
                ));
            }
        }
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
        || config.execution_mode() != request.binding.execution_mode
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
    let cwd = if config.execution_mode() == ExecutionMode::Brokered {
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
    if config.execution_mode() == ExecutionMode::Brokered {
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
        RuntimeKind::OpenCode => opencode::run(&mut turn),
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
        if self.config.execution_mode() == ExecutionMode::Brokered
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
