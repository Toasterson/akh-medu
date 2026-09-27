//! OpenCode (sst/opencode) over the Agent Client Protocol.
//!
//! One `opencode acp` subprocess per turn, NDJSON JSON-RPC 2.0 on stdio. Akh
//! tools reach OpenCode as a per-turn loopback MCP server ([`mcp`]) because
//! ACP has no client-side tool callback; calls are executed on this thread
//! through the host broker. Native tools are governed by the bridge-owned
//! permission policy in [`config`]: denied and hidden in `Brokered` mode,
//! routed to host approval in `Colocated` development mode.
//!
//! Research and pinned protocol details:
//! docs/ai/research/2026-09-25-opencode-runtime-integration.md.
mod config;
mod mcp;

pub use config::{OpenCodeProfileError, validate_profile};
pub use mcp::McpError;

use super::{Turn, process::Process};
use akh_session_types::RuntimeEvent;
use config::{AGENT, MCP_SERVER, ToolPolicy};
use mcp::{LoopbackMcp, McpTool, ToolReply};
use miette::Diagnostic;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use thiserror::Error;

/// ACP protocol version this bridge speaks.
const ACP_PROTOCOL: u64 = 1;
/// ACP's `RequestError.authRequired` code.
const AUTH_REQUIRED: i64 = -32000;
/// How long one wait on provider stdout lasts before brokered tool calls are
/// served again.
const POLL: Duration = Duration::from_millis(20);

/// Why an OpenCode turn failed.
#[derive(Debug, Error, Diagnostic)]
pub enum OpenCodeError {
    #[error(transparent)]
    #[diagnostic(transparent)]
    Profile(#[from] OpenCodeProfileError),

    #[error(transparent)]
    #[diagnostic(transparent)]
    Mcp(#[from] McpError),

    #[error("OpenCode needs provider authentication: {message}")]
    #[diagnostic(
        code(akh::runtime::opencode::auth_required),
        help(
            "run the profile's login command (`akh-runtime-profile --config FILE --profile ID login`, which runs `opencode auth login` inside the private profile), or provide the provider API key through `isolation.environment_files`"
        )
    )]
    AuthRequired { message: String },

    #[error("OpenCode refused {method}: {message} (JSON-RPC {code})")]
    #[diagnostic(
        code(akh::runtime::opencode::request_failed),
        help(
            "session/set_mode fails when the bridge's `akh` agent was not loaded; session/set_config_option fails for a model or effort the provider does not offer (the bridge never falls back); session/load fails for an unknown provider session"
        )
    )]
    Request {
        method: &'static str,
        code: i64,
        message: String,
    },

    #[error("OpenCode protocol violation: {detail}")]
    #[diagnostic(
        code(akh::runtime::opencode::protocol),
        help(
            "this bridge speaks ACP protocol 1 as implemented by OpenCode 1.18; pin a measured OpenCode release and re-run conformance"
        )
    )]
    Protocol { detail: String },

    #[error("OpenCode turn stopped: {reason}")]
    #[diagnostic(
        code(akh::runtime::opencode::stopped),
        help(
            "`cancelled` follows an Akh cancellation; `max_tokens`, `max_turn_requests` and `refusal` come from the provider model and leave the session resumable"
        )
    )]
    Stopped { reason: String },

    #[error("OpenCode opened the session without listing the Akh tools")]
    #[diagnostic(
        code(akh::runtime::opencode::mcp_not_connected),
        help(
            "OpenCode ignores MCP registration failures, so the bridge refuses to prompt without its tools. Check that loopback HTTP is not routed through HTTP(S)_PROXY (add 127.0.0.1 to NO_PROXY in the profile environment) and that the pinned OpenCode advertises mcpCapabilities.http"
        )
    )]
    McpNotConnected,

    /// Transport, timeout and cancellation failures, worded like the other
    /// provider bridges so callers see one vocabulary.
    #[error("{0}")]
    Transport(String),
}

impl OpenCodeError {
    fn protocol(detail: impl Into<String>) -> Self {
        Self::Protocol {
            detail: detail.into(),
        }
    }
}

/// Render a diagnostic for the string-typed provider boundary: code, message
/// and remedy on one line.
fn render(error: &OpenCodeError) -> String {
    if let OpenCodeError::Transport(message) = error {
        return message.clone();
    }
    let mut text = match error.code() {
        Some(code) => format!("{code}: {error}"),
        None => error.to_string(),
    };
    if let Some(help) = error.help() {
        text.push_str(&format!(" help: {help}"));
    }
    text
}

pub(super) fn run(turn: &mut Turn<'_>) -> Result<String, String> {
    run_turn(turn).map_err(|error| render(&error))
}

fn run_turn(turn: &mut Turn<'_>) -> Result<String, OpenCodeError> {
    validate_profile(turn.config)?;
    let model = turn
        .request
        .binding
        .model
        .as_deref()
        .map(config::model_id)
        .transpose()?
        .map(str::to_owned);
    let policy = ToolPolicy::resolve(turn.config.execution_mode(), turn.request.approval_policy);
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(turn.config.timeout_secs))
        .ok_or_else(|| OpenCodeError::Transport("runtime timeout is too large".into()))?;
    let tools = if turn.tools.is_empty() {
        None
    } else {
        Some(LoopbackMcp::start(
            turn.tools.iter().map(McpTool::from_function).collect(),
            deadline,
        )?)
    };
    let environment = config::bridge_environment(
        policy,
        &turn.instructions,
        Duration::from_secs(turn.config.timeout_secs),
        &mcp::random_token()?,
    )?;
    let mut command = turn
        .command(&turn.config.binary)
        .map_err(OpenCodeError::Transport)?;
    // `opencode acp` also serves its HTTP API; pin it to loopback regardless
    // of any `server` settings and keep mDNS off.
    command
        .arg("acp")
        .arg("--cwd")
        .arg(&turn.cwd)
        .args(["--hostname", "127.0.0.1", "--port", "0", "--mdns=false"])
        .envs(environment);
    let process =
        Process::spawn(command, turn.config.timeout_secs).map_err(OpenCodeError::Transport)?;
    Acp {
        turn,
        process,
        tools,
        model,
        next_id: 0,
        pending: None,
        setup: VecDeque::new(),
        session: None,
        prompting: false,
        answer: String::new(),
    }
    .drive()
}

/// Client requests, sent strictly one at a time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Initialize,
    NewSession,
    LoadSession,
    SetMode,
    SetModel,
    SetEffort,
    Prompt,
}

impl Step {
    fn method(self) -> &'static str {
        match self {
            Self::Initialize => "initialize",
            Self::NewSession => "session/new",
            Self::LoadSession => "session/load",
            Self::SetMode => "session/set_mode",
            Self::SetModel | Self::SetEffort => "session/set_config_option",
            Self::Prompt => "session/prompt",
        }
    }
}

/// One decoded line from the provider.
enum Incoming<'m> {
    Response {
        id: u64,
        outcome: Result<&'m Value, (i64, String)>,
    },
    Request {
        id: &'m Value,
        method: &'m str,
        params: &'m Value,
    },
    Notification {
        method: &'m str,
        params: &'m Value,
    },
}

fn classify(message: &Value) -> Result<Incoming<'_>, OpenCodeError> {
    let id = message.get("id").filter(|id| !id.is_null());
    match (message["method"].as_str(), id) {
        (Some(method), Some(id)) => Ok(Incoming::Request {
            id,
            method,
            params: &message["params"],
        }),
        (Some(method), None) => Ok(Incoming::Notification {
            method,
            params: &message["params"],
        }),
        (None, Some(id)) => {
            let id = id
                .as_u64()
                .ok_or_else(|| OpenCodeError::protocol("response to an unknown request id"))?;
            let outcome = match (message.get("result"), message.get("error")) {
                (_, Some(error)) => Err((
                    error["code"].as_i64().unwrap_or_default(),
                    error["message"]
                        .as_str()
                        .unwrap_or("unspecified error")
                        .to_owned(),
                )),
                (Some(result), None) => Ok(result),
                (None, None) => {
                    return Err(OpenCodeError::protocol("response without result or error"));
                }
            };
            Ok(Incoming::Response { id, outcome })
        }
        (None, None) => Err(OpenCodeError::protocol(
            "message is neither request, notification nor response",
        )),
    }
}

struct Acp<'t, 'a> {
    turn: &'t mut Turn<'a>,
    process: Process,
    tools: Option<LoopbackMcp>,
    model: Option<String>,
    next_id: u64,
    pending: Option<(u64, Step)>,
    setup: VecDeque<Step>,
    session: Option<String>,
    /// Session updates before `session/prompt` are history replay from
    /// `session/load`; they are not part of this turn's output.
    prompting: bool,
    answer: String,
}

impl Acp<'_, '_> {
    fn drive(mut self) -> Result<String, OpenCodeError> {
        self.request(Step::Initialize)?;
        loop {
            self.serve_tools()?;
            let frame = match self.process.poll(self.turn.cancel, POLL) {
                Ok(Some(frame)) => frame,
                Ok(None) => continue,
                Err(error) => {
                    self.cancel_remote();
                    return Err(OpenCodeError::Transport(error));
                }
            };
            if let Some(answer) = self.handle(&frame)? {
                return Ok(answer);
            }
        }
    }

    fn send(&mut self, message: Value) -> Result<(), OpenCodeError> {
        self.process
            .send(&message)
            .map_err(OpenCodeError::Transport)
    }

    fn request(&mut self, step: Step) -> Result<(), OpenCodeError> {
        let params = self.params(step);
        self.next_id += 1;
        let id = self.next_id;
        self.pending = Some((id, step));
        if step == Step::Prompt {
            self.prompting = true;
        }
        self.send(json!({"jsonrpc":"2.0","id":id,"method":step.method(),"params":params}))
    }

    fn params(&self, step: Step) -> Value {
        let session = self.session.as_deref().unwrap_or_default();
        let servers: Vec<Value> = self
            .tools
            .iter()
            .map(|tools| tools.acp_server(MCP_SERVER))
            .collect();
        match step {
            Step::Initialize => json!({
                "protocolVersion": ACP_PROTOCOL,
                // No client filesystem or terminal: OpenCode must not be able
                // to route effects through this client either.
                "clientCapabilities": {"fs": {"readTextFile": false, "writeTextFile": false}, "terminal": false},
                "clientInfo": {"name": "akh_medu", "title": "Akh-Medu", "version": env!("CARGO_PKG_VERSION")},
            }),
            Step::NewSession => json!({"cwd": self.turn.cwd, "mcpServers": servers}),
            Step::LoadSession => {
                json!({"sessionId": session, "cwd": self.turn.cwd, "mcpServers": servers})
            }
            Step::SetMode => json!({"sessionId": session, "modeId": AGENT}),
            Step::SetModel => {
                json!({"sessionId": session, "configId": "model", "value": self.model})
            }
            Step::SetEffort => {
                json!({"sessionId": session, "configId": "effort", "value": self.turn.config.effort})
            }
            Step::Prompt => json!({"sessionId": session,
                "prompt": [{"type": "text", "text": self.turn.prompt}]}),
        }
    }

    /// Execute Akh tool calls that arrived over the loopback MCP endpoint.
    fn serve_tools(&mut self) -> Result<(), OpenCodeError> {
        while let Some(call) = self.tools.as_ref().and_then(LoopbackMcp::next_call) {
            match self.turn.tool(&call.id, &call.name, call.arguments) {
                Ok(result) => {
                    let _ = call.reply.send(ToolReply::Result(result));
                }
                Err(error) => {
                    let _ = call.reply.send(ToolReply::Failed(error.clone()));
                    self.cancel_remote();
                    return Err(OpenCodeError::Transport(error));
                }
            }
        }
        Ok(())
    }

    fn cancel_remote(&mut self) {
        if let Some(session) = &self.session {
            let cancel =
                json!({"jsonrpc":"2.0","method":"session/cancel","params":{"sessionId":session}});
            let _ = self.process.send(&cancel);
        }
    }

    fn handle(&mut self, frame: &Value) -> Result<Option<String>, OpenCodeError> {
        match classify(frame)? {
            Incoming::Response { id, outcome } => self.response(id, outcome),
            Incoming::Request { id, method, params } => {
                self.agent_request(id, method, params)?;
                Ok(None)
            }
            Incoming::Notification { method, params } => {
                if method == "session/update" {
                    self.update(params)?;
                }
                Ok(None)
            }
        }
    }

    fn response(
        &mut self,
        id: u64,
        outcome: Result<&Value, (i64, String)>,
    ) -> Result<Option<String>, OpenCodeError> {
        let step = match self.pending {
            Some((pending, step)) if pending == id => step,
            _ => return Err(OpenCodeError::protocol(format!("unexpected response {id}"))),
        };
        self.pending = None;
        let result = outcome.map_err(|(code, message)| {
            if code == AUTH_REQUIRED {
                OpenCodeError::AuthRequired { message }
            } else {
                OpenCodeError::Request {
                    method: step.method(),
                    code,
                    message,
                }
            }
        })?;
        match step {
            Step::Initialize => {
                if result["protocolVersion"].as_u64() != Some(ACP_PROTOCOL) {
                    return Err(OpenCodeError::protocol(format!(
                        "OpenCode negotiated ACP protocol {}",
                        result["protocolVersion"]
                    )));
                }
                if self.turn.session.is_some() {
                    self.session = self.turn.session.clone();
                    self.request(Step::LoadSession)?;
                } else {
                    self.request(Step::NewSession)?;
                }
            }
            Step::NewSession | Step::LoadSession => {
                let id = match step {
                    Step::NewSession => result["sessionId"]
                        .as_str()
                        .ok_or_else(|| OpenCodeError::protocol("session/new returned no session"))?
                        .to_owned(),
                    _ => self.session.clone().unwrap_or_default(),
                };
                self.turn
                    .bind(&id, self.model.as_deref())
                    .map_err(OpenCodeError::protocol)?;
                self.session = Some(id);
                if self.tools.as_ref().is_some_and(|t| !t.catalog_served()) {
                    return Err(OpenCodeError::McpNotConnected);
                }
                // The mode is reasserted on every turn: a loaded session
                // restores whatever agent its history last used.
                self.setup.push_back(Step::SetMode);
                if self.model.is_some() {
                    self.setup.push_back(Step::SetModel);
                }
                if self.turn.config.effort.is_some() {
                    self.setup.push_back(Step::SetEffort);
                }
                self.setup.push_back(Step::Prompt);
                self.advance()?;
            }
            Step::SetMode | Step::SetModel | Step::SetEffort => self.advance()?,
            Step::Prompt => return self.finish(result).map(Some),
        }
        Ok(None)
    }

    fn advance(&mut self) -> Result<(), OpenCodeError> {
        match self.setup.pop_front() {
            Some(step) => self.request(step),
            None => Err(OpenCodeError::protocol("setup ended without a prompt")),
        }
    }

    fn finish(&mut self, result: &Value) -> Result<String, OpenCodeError> {
        if let Some(usage) = result.get("usage").filter(|u| u.is_object()) {
            self.turn.host.usage(&json!({
                "prompt_tokens": usage["inputTokens"],
                "completion_tokens": usage["outputTokens"],
                "completion_tokens_details": {"reasoning_tokens": usage.get("thoughtTokens").cloned().unwrap_or(json!(0))},
            }));
        }
        match result["stopReason"].as_str() {
            Some("end_turn") => Ok(std::mem::take(&mut self.answer)),
            Some(reason) => Err(OpenCodeError::Stopped {
                reason: reason.into(),
            }),
            None => Err(OpenCodeError::protocol(
                "session/prompt returned no stopReason",
            )),
        }
    }

    /// Requests OpenCode sends to its client.
    fn agent_request(
        &mut self,
        id: &Value,
        method: &str,
        params: &Value,
    ) -> Result<(), OpenCodeError> {
        let result = match method {
            "session/request_permission" => {
                let tool = &params["toolCall"];
                let name = tool["title"]
                    .as_str()
                    .or_else(|| tool["kind"].as_str())
                    .unwrap_or("tool");
                // Brokered turns never approve (Turn::approve), so a native
                // call that reaches this point is rejected before its effect.
                let allow = self
                    .turn
                    .approve(&id.to_string(), name, &tool["rawInput"])
                    .map_err(OpenCodeError::Transport)?;
                json!({"outcome": permission_outcome(&params["options"], allow)})
            }
            // fs/*, terminal/* and anything newer: this client grants nothing.
            _ => {
                return self.send(json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,
                    "message":"Akh does not offer this client capability to OpenCode"}}));
            }
        };
        self.send(json!({"jsonrpc":"2.0","id":id,"result":result}))
    }

    fn update(&mut self, params: &Value) -> Result<(), OpenCodeError> {
        if !self.prompting || params["sessionId"].as_str() != self.session.as_deref() {
            return Ok(());
        }
        let update = &params["update"];
        match update["sessionUpdate"].as_str() {
            Some("agent_message_chunk") => {
                if let Some(text) = update["content"]["text"].as_str() {
                    self.turn.text(text).map_err(OpenCodeError::Transport)?;
                    self.answer.push_str(text);
                }
            }
            Some("agent_thought_chunk") => {
                if let Some(text) = update["content"]["text"].as_str() {
                    self.turn
                        .host
                        .event(RuntimeEvent::Reasoning { text: text.into() })
                        .map_err(OpenCodeError::Transport)?;
                }
            }
            Some("tool_call") => self
                .turn
                .host
                .event(RuntimeEvent::ToolStarted {
                    id: update["toolCallId"].as_str().unwrap_or_default().into(),
                    name: update["title"]
                        .as_str()
                        .or_else(|| update["kind"].as_str())
                        .unwrap_or("tool")
                        .into(),
                })
                .map_err(OpenCodeError::Transport)?,
            Some("tool_call_update") => {
                let status = update["status"].as_str();
                if matches!(status, Some("completed" | "failed")) {
                    self.turn
                        .host
                        .event(RuntimeEvent::ToolCompleted {
                            id: update["toolCallId"].as_str().unwrap_or_default().into(),
                            success: status == Some("completed"),
                            details: Some(update.clone()),
                        })
                        .map_err(OpenCodeError::Transport)?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Pick the ACP permission option matching the host's decision. Without a
/// matching option the request is answered as cancelled, which OpenCode
/// treats as a rejection.
fn permission_outcome(options: &Value, allow: bool) -> Value {
    let wanted: &[&str] = if allow {
        &["allow_once"]
    } else {
        &["reject_once", "reject_always"]
    };
    options
        .as_array()
        .into_iter()
        .flatten()
        .find(|option| {
            option["kind"]
                .as_str()
                .is_some_and(|kind| wanted.contains(&kind))
        })
        .and_then(|option| option["optionId"].as_str())
        .map_or_else(
            || json!({"outcome": "cancelled"}),
            |option| json!({"outcome": "selected", "optionId": option}),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_classified_by_shape() {
        assert!(matches!(
            classify(&json!({"jsonrpc":"2.0","id":3,"result":{}})).unwrap(),
            Incoming::Response {
                id: 3,
                outcome: Ok(_)
            }
        ));
        assert!(matches!(
            classify(&json!({"id":4,"error":{"code":-32000,"message":"auth"}})).unwrap(),
            Incoming::Response {
                id: 4,
                outcome: Err((-32000, _))
            }
        ));
        assert!(matches!(
            classify(&json!({"id":"x","method":"session/request_permission","params":{}})).unwrap(),
            Incoming::Request {
                method: "session/request_permission",
                ..
            }
        ));
        assert!(matches!(
            classify(&json!({"method":"session/update","params":{}})).unwrap(),
            Incoming::Notification {
                method: "session/update",
                ..
            }
        ));
        assert!(classify(&json!({"id":"x","result":{}})).is_err());
        assert!(classify(&json!({"id":1})).is_err());
        assert!(classify(&json!({})).is_err());
    }

    #[test]
    fn permission_outcomes_never_widen_the_host_decision() {
        let options = json!([
            {"optionId":"once","kind":"allow_once","name":"Allow once"},
            {"optionId":"always","kind":"allow_always","name":"Always allow"},
            {"optionId":"reject","kind":"reject_once","name":"Reject"},
        ]);
        assert_eq!(
            permission_outcome(&options, true),
            json!({"outcome":"selected","optionId":"once"})
        );
        assert_eq!(
            permission_outcome(&options, false),
            json!({"outcome":"selected","optionId":"reject"})
        );
        let allow_only = json!([{"optionId":"always","kind":"allow_always"}]);
        assert_eq!(
            permission_outcome(&allow_only, false),
            json!({"outcome":"cancelled"})
        );
        assert_eq!(
            permission_outcome(&allow_only, true),
            json!({"outcome":"cancelled"})
        );
    }

    #[test]
    fn diagnostics_render_code_and_remedy_on_the_string_boundary() {
        let text = render(&OpenCodeError::AuthRequired {
            message: "openai".into(),
        });
        assert!(text.starts_with("akh::runtime::opencode::auth_required: "));
        assert!(text.contains("help: run the profile's login command"));
        let profile = render(&OpenCodeError::Profile(
            OpenCodeProfileError::IsolationRequired,
        ));
        assert!(profile.starts_with("akh::runtime::opencode::isolation_required"));
        assert_eq!(
            render(&OpenCodeError::Transport(
                "coding-agent turn cancelled".into()
            )),
            "coding-agent turn cancelled"
        );
    }
}
