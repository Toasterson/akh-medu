//! ACP v1 JSON-RPC over WebSocket, with Akh metadata for durable attachment/replay.
//! This transport is deliberately separate from T3's native RPC WebSocket.
use crate::{config::Principal, http, server::SessionApp, types::*};
use axum::{
    Extension,
    extract::{
        State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::HeaderMap,
    response::Response,
};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    time::Duration,
};

pub async fn upgrade(
    State(app): State<SessionApp>,
    Extension(principal): Extension<Principal>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> std::result::Result<Response, http::ApiError> {
    http::role(&principal, "client")?;
    Ok(upgrade
        .max_message_size(2 * 1024 * 1024)
        .max_frame_size(2 * 1024 * 1024)
        .on_upgrade(move |socket| connection(socket, app, principal, headers)))
}
struct Attachment {
    cursor: u64,
    controller: bool,
    prompt: Option<(Value, String)>,
    load_reply: Option<Value>,
}
struct Connection {
    app: SessionApp,
    principal: Principal,
    id: String,
    initialized: bool,
    attachments: BTreeMap<String, Attachment>,
    pending: HashMap<String, (String, PendingInput)>,
}
fn string<'a>(value: &'a Value, name: &str) -> Result<&'a str> {
    value[name]
        .as_str()
        .ok_or_else(|| Fault::invalid(format!("{name} must be a string")))
}
fn response(id: Value, result: Result<Value>) -> Value {
    match result {
        Ok(result) => json!({"jsonrpc":"2.0","id":id,"result":result}),
        Err(error) => {
            json!({"jsonrpc":"2.0","id":id,"error":{"code":match error.code.as_str(){"unsupported"=>-32601,"invalid_request"=>-32602,_=>-32000},"message":error.message,"data":{"code":error.code}}})
        }
    }
}
async fn send(socket: &mut WebSocket, value: Value) -> std::result::Result<(), ()> {
    tokio::time::timeout(
        Duration::from_secs(5),
        socket.send(Message::Text(value.to_string().into())),
    )
    .await
    .map_err(|_| ())?
    .map_err(|_| ())
}

async fn connection(
    mut socket: WebSocket,
    app: SessionApp,
    principal: Principal,
    headers: HeaderMap,
) {
    let mut connection = Connection {
        app,
        principal,
        id: new_id(),
        initialized: false,
        attachments: Default::default(),
        pending: Default::default(),
    };
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        let messages = tokio::select! {
            frame=socket.next()=>match frame{
                Some(Ok(Message::Text(text)))=>match serde_json::from_str::<Value>(&text){
                    Ok(request)=>connection.receive(request),
                    Err(_)=>vec![json!({"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"Invalid JSON"}})],
                },
                Some(Ok(Message::Ping(bytes)))=>{if socket.send(Message::Pong(bytes)).await.is_err(){break;}continue;},
                Some(Ok(Message::Pong(_)))=>continue,
                _=>break,
            },
            _=tick.tick()=>{
                if http::authenticate(&connection.app.sessions.config,&headers).is_err(){break;}
                match connection.updates(){Ok(messages)=>messages,Err(error)=>vec![json!({"jsonrpc":"2.0","method":"_akh/error","params":error})]}
            },
        };
        let mut failed = false;
        for message in messages {
            if send(&mut socket, message).await.is_err() {
                failed = true;
                break;
            }
        }
        if failed {
            break;
        }
    }
    for id in connection.attachments.keys() {
        let _ = connection
            .app
            .sessions
            .detach(&connection.principal, id, &connection.id);
    }
}
impl Connection {
    fn receive(&mut self, request: Value) -> Vec<Value> {
        if request["jsonrpc"] != "2.0" || !request.is_object() {
            return vec![response(
                request["id"].clone(),
                Err(Fault::invalid("expected a JSON-RPC 2.0 object")),
            )];
        }
        if request.get("method").is_none() {
            if let Some(id) = request["id"].as_str() {
                if let Some((session, pending)) = self.pending.remove(id) {
                    if let Err(error) = crate::server::controller(
                        &self.app,
                        &self.principal,
                        &session,
                        Some(&self.id),
                    ) {
                        return vec![json!({"jsonrpc":"2.0","method":"_akh/error","params":error})];
                    }
                    let value = if matches!(
                        pending.kind.as_str(),
                        "native_approval" | "tool_approval"
                    ) {
                        json!({"approved":request["result"]["outcome"]["outcome"]=="selected"&&request["result"]["outcome"]["optionId"]=="allow_once"})
                    } else {
                        request["result"].clone()
                    };
                    if let Err(error) = self.app.sessions.respond(
                        &self.principal,
                        &session,
                        &pending.turn_id,
                        &pending.id,
                        value,
                    ) {
                        return vec![json!({"jsonrpc":"2.0","method":"_akh/error","params":error})];
                    }
                }
            }
            return vec![];
        }
        let id = request.get("id").cloned();
        let method = request["method"].as_str().unwrap_or("");
        let params = &request["params"];
        let result = self.method(method, params, id.as_ref());
        match result {
            Ok(Some(value)) => id
                .map(|id| vec![response(id, Ok(value))])
                .unwrap_or_default(),
            Ok(None) => vec![],
            Err(error) => id
                .map(|id| vec![response(id, Err(error))])
                .unwrap_or_default(),
        }
    }
    fn method(
        &mut self,
        method: &str,
        params: &Value,
        rpc_id: Option<&Value>,
    ) -> Result<Option<Value>> {
        if method == "initialize" {
            if self.initialized {
                return Err(Fault::conflict("connection already initialized"));
            }
            if params["protocolVersion"].as_u64() != Some(1) {
                return Err(Fault::invalid(
                    "this endpoint implements ACP protocol version 1",
                ));
            }
            self.initialized = true;
            return Ok(Some(
                json!({"protocolVersion":1,"agentInfo":{"name":"akh-session-platform","version":env!("CARGO_PKG_VERSION")},"agentCapabilities":{"loadSession":true,"promptCapabilities":{"image":false,"audio":false,"embeddedContext":false},"mcpCapabilities":{"http":false,"sse":false},"sessionCapabilities":{"list":{}}},"authMethods":[],"_meta":{"akh":{"attachmentId":self.id,"transport":"websocket","requestId":"_meta.requestId","replayCursor":"_meta.afterSequence"}}}),
            ));
        }
        if !self.initialized {
            return Err(Fault::conflict("initialize must be the first request"));
        }
        let request_id = || {
            params["_meta"]["requestId"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| {
                    format!(
                        "{}:{}",
                        self.id,
                        rpc_id.map(Value::to_string).unwrap_or_else(new_id)
                    )
                })
        };
        match method {
            "session/new" => {
                if params["mcpServers"]
                    .as_array()
                    .is_some_and(|v| !v.is_empty())
                {
                    return Err(Fault::new(
                        "unsupported",
                        "configure session tools through an authorized Akh profile",
                    ));
                }
                let allowed: Vec<_> = self
                    .app
                    .sessions
                    .config
                    .akhs
                    .keys()
                    .filter(|a| self.principal.can_use_akh(a))
                    .cloned()
                    .collect();
                let akh = params["_meta"]["akh"]
                    .as_str()
                    .map(str::to_owned)
                    .or_else(|| {
                        if allowed.len() == 1 {
                            allowed.first().cloned()
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| Fault::invalid("select an allowed Akh in _meta.akh"))?;
                let profile = self
                    .app
                    .sessions
                    .config
                    .akhs
                    .get(&akh)
                    .ok_or_else(|| Fault::missing("Akh profile not found"))?;
                let workspace = params["_meta"]["workspaceProfile"]
                    .as_str()
                    .map(str::to_owned)
                    .or_else(|| {
                        if profile.workspace_profiles.len() == 1 {
                            profile.workspace_profiles.first().cloned()
                        } else {
                            None
                        }
                    })
                    .ok_or_else(|| Fault::invalid("select _meta.workspaceProfile"))?;
                let session = self.app.sessions.create(
                    &self.principal,
                    &NewSession {
                        request_id: request_id(),
                        akh,
                        workspace_profile: workspace,
                        conversation_key: params["_meta"]["conversationKey"]
                            .as_str()
                            .map(str::to_owned),
                    },
                )?;
                self.app
                    .sessions
                    .attach(&self.principal, &session.id, &self.id, true)?;
                self.attachments.insert(
                    session.id.clone(),
                    Attachment {
                        cursor: 0,
                        controller: true,
                        prompt: None,
                        load_reply: None,
                    },
                );
                Ok(Some(
                    json!({"sessionId":session.id,"_meta":{"workspaceProfile":session.workspace_profile,"workspacePending":session.workspace.is_none()}}),
                ))
            }
            "session/load" => {
                let id = string(params, "sessionId")?;
                if self.attachments.get(id).is_some_and(|a| a.prompt.is_some()) {
                    return Err(Fault::conflict(
                        "session already has a pending prompt on this connection",
                    ));
                }
                let controller = params["_meta"]["observer"] != true;
                self.app
                    .sessions
                    .attach(&self.principal, id, &self.id, controller)?;
                self.attachments.insert(
                    id.into(),
                    Attachment {
                        cursor: params["_meta"]["afterSequence"].as_u64().unwrap_or(0),
                        controller,
                        prompt: None,
                        load_reply: rpc_id.cloned(),
                    },
                );
                Ok(None)
            }
            "session/list" => {
                let sessions = self.app.sessions.list(&self.principal)?;
                Ok(Some(
                    json!({"sessions":sessions.into_iter().map(|s|json!({"sessionId":s.id,"cwd":s.workspace.map(|w|w.root).unwrap_or_else(||"/".into()),"title":s.akh})).collect::<Vec<_>>()}),
                ))
            }
            "session/prompt" => {
                let id = string(params, "sessionId")?;
                let attachment = self
                    .attachments
                    .get(id)
                    .ok_or_else(|| Fault::conflict("load the session before prompting"))?;
                if !attachment.controller
                    || attachment.prompt.is_some()
                    || attachment.load_reply.is_some()
                {
                    return Err(Fault::conflict("observer or prompt already active"));
                }
                crate::server::controller(&self.app, &self.principal, id, Some(&self.id))?;
                let content = params["prompt"]
                    .as_array()
                    .ok_or_else(|| Fault::invalid("prompt must be an array"))?;
                if content
                    .iter()
                    .any(|c| c["type"] != "text" || !c["text"].is_string())
                {
                    return Err(Fault::new(
                        "unsupported",
                        "only text prompt blocks are enabled for this profile",
                    ));
                }
                let text = content
                    .iter()
                    .filter_map(|c| c["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let rpc_id = rpc_id
                    .cloned()
                    .ok_or_else(|| Fault::invalid("session/prompt requires a request ID"))?;
                let turn = self.app.sessions.submit(
                    &self.principal,
                    id,
                    &SubmitInput {
                        request_id: request_id(),
                        text,
                        artifacts: vec![],
                    },
                )?;
                self.attachments.get_mut(id).unwrap().prompt = Some((rpc_id, turn.id));
                Ok(None)
            }
            "session/cancel" => {
                let id = string(params, "sessionId")?;
                crate::server::controller(&self.app, &self.principal, id, Some(&self.id))?;
                let session = self.app.sessions.get(&self.principal, id)?;
                if let Some(turn) = session.turns.iter().find(|t| !t.status.terminal()) {
                    self.app.sessions.cancel(&self.principal, id, &turn.id)?;
                }
                Ok(Some(json!({})))
            }
            "session/close" => {
                let id = string(params, "sessionId")?;
                self.app.sessions.detach(&self.principal, id, &self.id)?;
                self.attachments.remove(id);
                Ok(Some(json!({})))
            }
            "_akh/respond" => {
                let id = string(params, "sessionId")?;
                crate::server::controller(&self.app, &self.principal, id, Some(&self.id))?;
                let pending = self.app.sessions.respond(
                    &self.principal,
                    id,
                    string(params, "turnId")?,
                    string(params, "inputId")?,
                    params["response"].clone(),
                )?;
                Ok(Some(json!(pending)))
            }
            _ => Err(Fault::new(
                "unsupported",
                format!("method {method} is not supported"),
            )),
        }
    }
    fn updates(&mut self) -> Result<Vec<Value>> {
        let mut messages = vec![];
        for (session, attachment) in &mut self.attachments {
            if attachment.controller {
                let state = self.app.sessions.get(&self.principal, session)?;
                if state
                    .controller
                    .as_ref()
                    .is_none_or(|l| l.expires_at < now_ms() + 30_000)
                {
                    self.app
                        .sessions
                        .attach(&self.principal, session, &self.id, true)?;
                }
            }
            let events = self
                .app
                .sessions
                .events(&self.principal, session, attachment.cursor)?;
            let caught_up = events.len() < 1000;
            for event in events {
                let meta =
                    json!({"sequence":event.sequence,"turnId":event.turn_id,"kind":event.kind});
                let update = match event.kind.as_str() {
                    "input_accepted" => Some(
                        json!({"sessionUpdate":"user_message_chunk","content":{"type":"text","text":event.data["text"]},"_meta":{"origin":event.data["origin"]}}),
                    ),
                    "runtime" => match event.data["kind"].as_str() {
                        Some("text") => Some(
                            json!({"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":event.data["text"]}}),
                        ),
                        Some("tool_started") => Some(
                            json!({"sessionUpdate":"tool_call","toolCallId":event.data["id"],"title":event.data["name"],"kind":"other","status":"in_progress"}),
                        ),
                        Some("tool_completed") => Some(
                            json!({"sessionUpdate":"tool_call_update","toolCallId":event.data["id"],"status":if event.data["success"]==true{"completed"}else{"failed"}}),
                        ),
                        _ => None,
                    },
                    _ => None,
                };
                messages.push(if let Some(mut update)=update{update["_meta"]["akh"]=meta;json!({"jsonrpc":"2.0","method":"session/update","params":{"sessionId":session,"update":update}})}else{json!({"jsonrpc":"2.0","method":"_akh/event","params":event})});
                attachment.cursor = event.sequence;
            }
            if caught_up {
                if let Some(id) = attachment.load_reply.take() {
                    messages.push(response(id, Ok(json!({}))));
                }
            }
            if attachment.controller {
                for pending in self.app.sessions.pending_inputs(&self.principal, session)? {
                    let id = format!("akh:{}:{}:{}", session, pending.turn_id, pending.id);
                    if self.pending.contains_key(&id) {
                        continue;
                    }
                    let message = if matches!(
                        pending.kind.as_str(),
                        "native_approval" | "tool_approval"
                    ) {
                        json!({"jsonrpc":"2.0","id":id,"method":"session/request_permission","params":{"sessionId":session,"toolCall":{"toolCallId":pending.id,"title":pending.prompt["name"].as_str().unwrap_or("Approve operation"),"kind":"other","status":"pending","rawInput":pending.prompt},"options":[{"optionId":"allow_once","name":"Allow once","kind":"allow_once"},{"optionId":"reject_once","name":"Reject","kind":"reject_once"}]}})
                    } else {
                        json!({"jsonrpc":"2.0","method":"_akh/input_required","params":pending})
                    };
                    self.pending.insert(id, (session.clone(), pending));
                    messages.push(message);
                }
            }
            if let Some((id, turn_id)) = &attachment.prompt {
                let state = self.app.sessions.get(&self.principal, session)?;
                let turn = state
                    .turns
                    .iter()
                    .find(|t| &t.id == turn_id)
                    .ok_or_else(|| Fault::missing("turn disappeared"))?;
                if turn.status.terminal() || turn.status == TurnStatus::Unknown {
                    let result = match turn.status {
                        TurnStatus::Completed => {
                            Ok(json!({"stopReason":"end_turn","_meta":{"turnId":turn_id}}))
                        }
                        TurnStatus::Cancelled => {
                            Ok(json!({"stopReason":"cancelled","_meta":{"turnId":turn_id}}))
                        }
                        _ => Err(turn
                            .error
                            .clone()
                            .unwrap_or_else(|| Fault::new("turn_failed", "turn did not complete"))),
                    };
                    messages.push(response(id.clone(), result));
                    attachment.prompt = None;
                }
            }
        }
        Ok(messages)
    }
}
