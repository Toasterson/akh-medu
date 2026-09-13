use super::{Turn, process::Process};
use crate::message::{AkhMessage, ToolKind};
use serde_json::json;
use std::process::Command;

pub(super) fn run(turn: &mut Turn<'_>) -> Result<String, String> {
    let mut command = Command::new(
        turn.runtime
            .config
            .node_binary
            .as_deref()
            .unwrap_or(std::path::Path::new("node")),
    );
    command
        .arg(
            turn.runtime
                .config
                .bridge_script
                .as_ref()
                .ok_or("Claude bridge is not configured")?,
        )
        .current_dir(&turn.cwd);
    let mut process = Process::spawn(command, turn.runtime.config.timeout_secs)?;
    process.send(
        &json!({"type":"start","prompt":turn.prompt,"instructions":turn.instructions,
        "cwd":turn.cwd,"model":turn.runtime.model,"effort":turn.runtime.effort,
        "binary":turn.runtime.config.binary,"resume":turn.session.as_ref().map(|s| &s.id),
        "plan":turn.registry.permission_mode() == crate::agent::permission::PermissionMode::Plan,
        "tools":turn.tools}),
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
                if turn.session.as_ref().is_some_and(|s| s.id != id) {
                    return Err("Claude resumed a different session".into());
                }
                turn.bind(id, message["model"].as_str());
            }
            Some("text") => {
                let delta = message["text"].as_str().unwrap_or("");
                turn.text(delta);
                answer.push_str(delta);
            }
            Some("tool") => {
                let result = turn.tool(
                    message["name"].as_str().unwrap_or(""),
                    message["arguments"].clone(),
                );
                process.send(&json!({"type":"response","id":message["id"],"result":result}))?;
            }
            Some("approval") => {
                let allow = turn.approve(
                    message["name"].as_str().unwrap_or("tool"),
                    &message["arguments"],
                );
                process.send(
                    &json!({"type":"response","id":message["id"],"result":{"allow":allow}}),
                )?;
            }
            Some("tool_started") => turn.sink.emit(&AkhMessage::ToolCallStarted {
                tool_id: message["id"].as_str().unwrap_or("").into(),
                kind: ToolKind::Other,
                title: format!("Claude: {}", message["name"].as_str().unwrap_or("tool")),
                locations: vec![],
            }),
            Some("tool_completed") => turn.sink.emit(&AkhMessage::ToolCallCompleted {
                tool_id: message["id"].as_str().unwrap_or("").into(),
                success: message["success"] == true,
                output: None,
                diff: None,
            }),
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
                    turn.text(final_text);
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
