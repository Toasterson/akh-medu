use super::{Turn, process::Process};
use akh_session_types::{ApprovalPolicy, ExecutionMode, RuntimeEvent};
use serde_json::json;

pub(super) fn run(turn: &mut Turn<'_>) -> Result<String, String> {
    let mut command = turn.command(
        turn.config
            .node_binary
            .as_deref()
            .unwrap_or(std::path::Path::new("node")),
    )?;
    command.arg(
        turn.config
            .bridge_script
            .as_ref()
            .ok_or("Claude bridge is not configured")?,
    );
    let mut process = Process::spawn(command, turn.config.timeout_secs)?;
    process.send(
        &json!({"type":"start","prompt":turn.prompt,"instructions":turn.instructions,
        "cwd":turn.cwd,"model":turn.request.binding.model,"effort":turn.config.effort,
        "binary":turn.config.binary,"resume":turn.session.as_ref(),
        "plan":turn.request.approval_policy == ApprovalPolicy::ReadOnly,
        "tools":turn.tools,"brokered":turn.config.execution_mode() == ExecutionMode::Brokered,"aliases":turn.config.native_aliases}),
    )?;
    let mut answer = String::new();
    loop {
        let message = match process.recv(turn.cancel) {
            Ok(m) => m,
            Err(e) => {
                let _ = process.send(&json!({"type":"cancel"}));
                return Err(e);
            }
        };
        match message["type"].as_str() {
            Some("session") => {
                let id = message["id"]
                    .as_str()
                    .ok_or("Claude returned no session ID")?;
                if turn.session.as_ref().is_some_and(|s| s != id) {
                    return Err("Claude resumed a different session".into());
                }
                turn.bind(id, message["model"].as_str())?;
            }
            Some("text") => {
                let delta = message["text"].as_str().unwrap_or("");
                turn.text(delta)?;
                answer.push_str(delta);
            }
            Some("reasoning") => turn.host.event(RuntimeEvent::Reasoning { text: message["text"].as_str().unwrap_or("").into() })?,
            Some("usage") => turn.host.usage(&json!({"prompt_tokens":message["usage"]["input_tokens"],"completion_tokens":message["usage"]["output_tokens"]})),
            Some("tool") => {
                let result = turn.tool(
                    &message["id"].to_string(),
                    message["name"].as_str().unwrap_or(""),
                    message["arguments"].clone(),
                )?;
                process.send(&json!({"type":"response","id":message["id"],"result":result}))?;
            }
            Some("approval") => {
                let allow = turn.approve(
                    &message["id"].to_string(),
                    message["name"].as_str().unwrap_or("tool"),
                    &message["arguments"],
                )?;
                process.send(
                    &json!({"type":"response","id":message["id"],"result":{"allow":allow}}),
                )?;
            }
            Some("tool_started") => turn.host.event(RuntimeEvent::ToolStarted {
                id: message["id"].as_str().unwrap_or("").into(),
                name: message["name"].as_str().unwrap_or("tool").into(),
            })?,
            Some("tool_completed") => turn.host.event(RuntimeEvent::ToolCompleted {
                id: message["id"].as_str().unwrap_or("").into(),
                success: message["success"] == true,
                details: message.get("details").cloned(),
            })?,
            Some("question") => {
                let response = crate::questions::ask(
                    turn.host,
                    &message["id"].to_string(),
                    &message["questions"],
                    crate::questions::Provider::Claude,
                )?;
                process.send(&json!({"type":"response","id":message["id"],"result":response}))?;
            }
            Some("done") => {
                if message["success"] != true {
                    return Err(message["error"]
                        .as_str()
                        .unwrap_or("Claude turn failed")
                        .into());
                }
                if turn.session.is_none() {
                    return Err("Claude completed without a session ID".into());
                }
                let final_text = message["text"].as_str().unwrap_or("");
                if answer.is_empty() {
                    turn.text(final_text)?;
                }
                return Ok(if final_text.is_empty() {
                    answer
                } else {
                    final_text.into()
                });
            }
            Some("error") => {
                return Err(message["message"]
                    .as_str()
                    .unwrap_or("Claude bridge failed")
                    .into());
            }
            _ => {}
        }
    }
}
