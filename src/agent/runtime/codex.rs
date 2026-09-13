use super::{Turn, process::Process};
use crate::message::{AkhMessage, ToolKind};
use serde_json::{Value, json};
use std::process::Command;

pub(super) fn run(turn: &mut Turn<'_>) -> Result<String, String> {
    let mut command = Command::new(
        turn.runtime
            .config
            .binary
            .as_deref()
            .unwrap_or(std::path::Path::new("codex")),
    );
    command
        .args(["app-server", "--listen", "stdio://"])
        .current_dir(&turn.cwd);
    let mut process = Process::spawn(command, turn.runtime.config.timeout_secs)?;
    process.send(&json!({"id":1,"method":"initialize","params":{
        "clientInfo":{"name":"akh_medu","title":"Akh-Medu","version":env!("CARGO_PKG_VERSION")},
        "capabilities":{"experimentalApi":true}
    }}))?;
    let mut active_turn: Option<String> = None;
    let mut answer = String::new();
    let mut streamed_items = std::collections::HashSet::new();
    let mut final_answer = None;
    loop {
        let message = match process.recv(turn.cancel) {
            Ok(m) => m,
            Err(e) => {
                if let (Some(session), Some(id)) = (&turn.session, &active_turn) {
                    let _ = process.send(&json!({"id":4,"method":"turn/interrupt","params":{
                        "threadId":session.id,"turnId":id}}));
                }
                return Err(e);
            }
        };
        if message.get("method").is_none() {
            if let Some(error) = message.get("error") {
                return Err(format!(
                    "Codex request failed: {}",
                    error["message"].as_str().unwrap_or("protocol error")
                ));
            }
            match message["id"].as_u64() {
                Some(1) => {
                    process.send(&json!({"method":"initialized","params":{}}))?;
                    let mut params = json!({"cwd":turn.cwd,"approvalPolicy":"on-request",
                        "approvalsReviewer":"user","sandbox":"workspace-write",
                        "developerInstructions":turn.instructions});
                    // SandboxMode wire values are kebab-case, unlike item types.
                    if turn.registry.permission_mode()
                        == crate::agent::permission::PermissionMode::Plan
                    {
                        params["sandbox"] = json!("read-only");
                        params["approvalPolicy"] = json!("never");
                    }
                    if let Some(model) = &turn.runtime.model {
                        params["model"] = json!(model);
                    }
                    let method = if let Some(session) = &turn.session {
                        params["threadId"] = json!(session.id);
                        "thread/resume"
                    } else {
                        params["dynamicTools"] = json!([{"type":"namespace", "name":"akh",
                            "description":"Akh-Medu knowledge, memory and execution tools",
                            "tools":turn.tools.iter().map(|t| json!({
                            "type":"function",
                            "name":t["function"]["name"],"description":t["function"]["description"],
                            "inputSchema":t["function"]["parameters"]
                        })).collect::<Vec<_>>() }]);
                        params["allowProviderModelFallback"] = json!(false);
                        "thread/start"
                    };
                    process.send(&json!({"id":2,"method":method,"params":params}))?;
                }
                Some(2) => {
                    let id = message["result"]["thread"]["id"]
                        .as_str()
                        .ok_or("Codex returned no thread ID")?;
                    if turn.session.as_ref().is_some_and(|s| s.id != id) {
                        return Err("Codex resumed a different thread".into());
                    }
                    turn.bind(id, message["result"]["model"].as_str());
                    let mut params =
                        json!({"threadId":id,"input":[{"type":"text","text":turn.prompt}]});
                    if let Some(effort) = &turn.runtime.effort {
                        params["effort"] = json!(effort);
                    }
                    process.send(&json!({"id":3,"method":"turn/start","params":params}))?;
                }
                Some(3) => {
                    active_turn = Some(
                        message["result"]["turn"]["id"]
                            .as_str()
                            .ok_or("Codex returned no turn ID")?
                            .into(),
                    )
                }
                _ => {}
            }
            continue;
        }
        let method = message["method"].as_str().unwrap_or("");
        let params = &message["params"];
        if let Some(thread) = params["threadId"].as_str()
            && turn.session.as_ref().is_some_and(|s| s.id != thread)
        {
            if let Some(id) = message.get("id") {
                process.send(
                    &json!({"id":id,"error":{"code":-32602,"message":"unexpected thread"}}),
                )?;
            }
            continue;
        }
        if let Some(id) = message.get("id") {
            let result = match method {
                "item/tool/call" => {
                    if params["namespace"].as_str() != Some("akh") {
                        process.send(&json!({"id":id,"error":{"code":-32602,"message":"unexpected tool namespace"}}))?;
                        continue;
                    }
                    let result = turn.tool(
                        params["tool"].as_str().unwrap_or(""),
                        params["arguments"].clone(),
                    );
                    json!({"success":result["success"],"contentItems":[{"type":"inputText","text":result["text"]}]})
                }
                "item/commandExecution/requestApproval" | "item/fileChange/requestApproval" => {
                    json!({"decision":if turn.approve(method, params) { "accept" } else { "decline" }})
                }
                "item/permissions/requestApproval" => json!({"permissions":{},"scope":"turn"}),
                "mcpServer/elicitation/request" => json!({"action":"decline","content":null}),
                // ACP's current bridge supports approvals, not arbitrary forms.
                "tool/requestUserInput" => json!({"answers":{}}),
                _ => {
                    process.send(&json!({"id":id,"error":{"code":-32601,"message":"unsupported runtime request"}}))?;
                    continue;
                }
            };
            process.send(&json!({"id":id,"result":result}))?;
            continue;
        }
        match method {
            "turn/started" => {
                active_turn = params["turn"]["id"].as_str().map(str::to_owned);
            }
            "item/agentMessage/delta" => {
                let delta = params["delta"].as_str().unwrap_or("");
                turn.text(delta);
                answer.push_str(delta);
                streamed_items.insert(params["itemId"].as_str().unwrap_or("").to_owned());
            }
            "item/started" => {
                let item = &params["item"];
                if is_tool(item) {
                    turn.sink.emit(&AkhMessage::ToolCallStarted {
                        tool_id: item["id"].as_str().unwrap_or("").into(),
                        kind: ToolKind::Other,
                        title: format!("Codex: {}", item["type"].as_str().unwrap_or("tool")),
                        locations: vec![],
                    });
                }
            }
            "item/completed" => {
                let item = &params["item"];
                if item["type"] == "agentMessage" {
                    let text = item["text"].as_str().unwrap_or("");
                    if !streamed_items.contains(item["id"].as_str().unwrap_or("")) {
                        turn.text(text);
                        answer.push_str(text);
                    }
                    if item["phase"] == "final_answer" || !text.is_empty() {
                        final_answer = Some(text.to_owned());
                    }
                } else if is_tool(item) {
                    turn.sink.emit(&AkhMessage::ToolCallCompleted {
                        tool_id: item["id"].as_str().unwrap_or("").into(),
                        success: matches!(item["status"].as_str(), Some("completed")),
                        output: item["aggregatedOutput"].as_str().map(str::to_owned),
                        diff: None,
                    });
                }
            }
            "turn/completed" => {
                if active_turn.as_deref() != params["turn"]["id"].as_str() {
                    continue;
                }
                return match params["turn"]["status"].as_str() {
                    Some("completed") => Ok(final_answer.unwrap_or(answer)),
                    Some("interrupted") => Err("Codex turn interrupted".into()),
                    _ => Err(format!(
                        "Codex turn failed: {}",
                        params["turn"]["error"]["message"]
                            .as_str()
                            .unwrap_or("unknown error")
                    )),
                };
            }
            _ => {}
        }
    }
}

fn is_tool(item: &Value) -> bool {
    matches!(
        item["type"].as_str(),
        Some("commandExecution" | "fileChange" | "mcpToolCall" | "dynamicToolCall" | "webSearch")
    )
}
