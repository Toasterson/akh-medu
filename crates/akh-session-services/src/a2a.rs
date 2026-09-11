//! A2A 0.3 JSON-RPC facade for released kagent BYO agents.
use crate::{config::Principal, http, server::SessionApp, store::key, types::*};
use axum::{
    Extension, Json,
    extract::{Path, State},
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{convert::Infallible, time::Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Mapping {
    akh: String,
    message: Value,
    session: String,
    turn: String,
}
fn string<'a>(v: &'a Value, k: &str) -> Result<&'a str> {
    v[k].as_str()
        .ok_or_else(|| Fault::invalid(format!("{k} must be a string")))
}
fn rpc(id: &Value, result: Result<Value>) -> Value {
    match result {
        Ok(value) => json!({"jsonrpc":"2.0","id":id,"result":value}),
        Err(error) => {
            json!({"jsonrpc":"2.0","id":id,"error":{"code":match error.code.as_str(){"not_found"=>-32001,"unsupported"=>-32601,"invalid_request"=>-32602,"conflict"=>-32002,_=>-32000},"message":error.message,"data":{"code":error.code}}})
        }
    }
}

pub async fn card(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(akh): Path<String>,
) -> http::Api<Value> {
    if !p.can_use_akh(&akh) {
        return Err(Fault::new("forbidden", "Akh not granted").into());
    }
    let profile = app
        .sessions
        .config
        .akhs
        .get(&akh)
        .ok_or_else(|| Fault::missing("Akh profile not found"))?;
    let base = app
        .sessions
        .config
        .public_url
        .as_ref()
        .ok_or_else(|| Fault::invalid("A2A discovery requires a configured public_url"))?;
    Ok(Json(
        json!({"protocolVersion":"0.3.0","name":akh,"description":"Centrally managed Akh agent sessions","url":format!("{}/a2a/{}",base.trim_end_matches('/'),http::segment(&akh)),"preferredTransport":"JSONRPC","version":env!("CARGO_PKG_VERSION"),"capabilities":{"streaming":true,"pushNotifications":false},"defaultInputModes":["text/plain","application/json"],"defaultOutputModes":["text/plain","application/json"],"securitySchemes":{"bearer":{"type":"http","scheme":"bearer"}},"security":[{"bearer":[]}],"skills":[{"id":akh,"name":akh,"description":"Work in the granted workspace using the configured Akh tool profile","tags":["workspace","agent"],"examples":[]}],"metadata":{"workspaceProfiles":profile.workspace_profiles}}),
    ))
}
pub async fn endpoint(
    State(app): State<SessionApp>,
    Extension(p): Extension<Principal>,
    Path(akh): Path<String>,
    Json(request): Json<Value>,
) -> Response {
    let id = request["id"].clone();
    let result = (|| {
        http::role(&p, "client")?;
        if !p.can_use_akh(&akh) {
            return Err(Fault::new("forbidden", "Akh not granted"));
        }
        if request["jsonrpc"] != "2.0" || request.get("id").is_none() {
            return Err(Fault::invalid("expected a JSON-RPC request with an ID"));
        }
        let method = string(&request, "method")?;
        let params = &request["params"];
        match method {
            "message/send" | "message/stream" => {
                let mapping = admit(&app, &p, &akh, params)?;
                Ok((mapping.session, mapping.turn, method == "message/stream"))
            }
            "tasks/get" | "tasks/cancel" | "tasks/resubscribe" => {
                let (session, turn) = lookup(&app, &p, &akh, string(params, "id")?)?;
                if method == "tasks/cancel" {
                    crate::server::controller(&app, &p, &session.id, None)?;
                    app.sessions.cancel(&p, &session.id, &turn.id)?;
                }
                Ok((session.id, turn.id, method == "tasks/resubscribe"))
            }
            _ => Err(Fault::new("unsupported", "A2A method is not enabled")),
        }
    })();
    let (session, turn, streaming) = match result {
        Ok(value) => value,
        Err(error) => return Json(rpc(&id, Err(error))).into_response(),
    };
    if !streaming {
        if request["method"] == "message/send"
            && request["params"]["configuration"]["blocking"] == true
        {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(25);
            loop {
                match task(&app, &p, &session, &turn) {
                    Ok(task) if interrupted(&task) || tokio::time::Instant::now() >= deadline => {
                        return Json(rpc(&id, Ok(task))).into_response();
                    }
                    Err(error) => return Json(rpc(&id, Err(error))).into_response(),
                    _ => tokio::time::sleep(Duration::from_millis(100)).await,
                }
            }
        }
        return Json(rpc(&id, task(&app, &p, &session, &turn))).into_response();
    }
    let stream = async_stream::stream! {
        let mut previous=Value::Null;let mut cursor=0;let mut artifact_started=false;
        loop{
            let value=match task(&app,&p,&session,&turn){Ok(value)=>value,Err(error)=>{yield Ok::<_,Infallible>(Event::default().data(rpc(&id,Err(error)).to_string()));break;}};
            if previous.is_null(){yield Ok(Event::default().data(rpc(&id,Ok(value.clone())).to_string()));}
            let events=match app.sessions.events(&p,&session,cursor){Ok(events)=>events,Err(error)=>{yield Ok(Event::default().data(rpc(&id,Err(error)).to_string()));break;}};
            let caught_up=events.len()<1000;
            for event in events{
                cursor=event.sequence;
                if event.turn_id.as_deref()==Some(&turn)&&event.kind=="runtime"&&event.data["kind"]=="text"{
                    let update=json!({"kind":"artifact-update","taskId":turn,"contextId":session,"artifact":{"artifactId":format!("{turn}-answer"),"parts":[{"kind":"text","text":event.data["text"]}]},"append":artifact_started,"lastChunk":false});
                    artifact_started=true;yield Ok(Event::default().id(cursor.to_string()).data(rpc(&id,Ok(update)).to_string()));
                }
            }
            let final_event=interrupted(&value)&&caught_up;
            if value["status"]!=previous["status"]||final_event{
                if final_event&&artifact_started{yield Ok(Event::default().data(rpc(&id,Ok(json!({"kind":"artifact-update","taskId":turn,"contextId":session,"artifact":{"artifactId":format!("{turn}-answer"),"parts":[]},"append":true,"lastChunk":true}))).to_string()));}
                yield Ok(Event::default().data(rpc(&id,Ok(json!({"kind":"status-update","taskId":turn,"contextId":session,"status":value["status"],"final":final_event,"metadata":value["metadata"]}))).to_string()));
            }
            if final_event{break;}previous=value;tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
fn interrupted(task: &Value) -> bool {
    matches!(
        task["status"]["state"].as_str(),
        Some("completed" | "canceled" | "failed" | "rejected" | "input-required" | "unknown")
    )
}

fn lookup(
    app: &SessionApp,
    p: &Principal,
    akh: &str,
    id: &str,
) -> Result<(SessionRecord, TurnRecord)> {
    for session in app.sessions.list(p)?.into_iter().filter(|s| s.akh == akh) {
        if let Some(turn) = session.turns.iter().find(|t| t.id == id).cloned() {
            return Ok((session, turn));
        }
    }
    Err(Fault::missing("A2A task not found"))
}
fn admit(app: &SessionApp, p: &Principal, akh: &str, params: &Value) -> Result<Mapping> {
    let _admission = app.admission_lock.lock().unwrap_or_else(|e| e.into_inner());
    if params["configuration"]
        .get("pushNotificationConfig")
        .is_some()
    {
        return Err(Fault::new("unsupported", "push notifications are disabled"));
    }
    let message = &params["message"];
    if message["role"] != "user" {
        return Err(Fault::invalid("only user messages can initiate tasks"));
    }
    let message_id = string(message, "messageId")?;
    validate_key(message_id)?;
    let k = key("a2a_message", &[&p.id, message_id]);
    if let Some(mapping) = app.sessions.store.get::<Mapping>(&k)? {
        if mapping.akh != akh || mapping.message != *message {
            return Err(Fault::conflict(
                "A2A message ID reused with different content",
            ));
        }
        return Ok(mapping);
    }
    let mapping = if let Some(task_id) = message["taskId"].as_str() {
        let (session, turn) = lookup(app, p, akh, task_id)?;
        if turn.status.terminal() {
            return Err(Fault::conflict(
                "terminal tasks cannot be restarted; send a new message in the same context",
            ));
        }
        if message["contextId"]
            .as_str()
            .is_some_and(|id| id != session.id)
        {
            return Err(Fault::conflict("task context mismatch"));
        }
        crate::server::controller(app, p, &session.id, None)?;
        let parts = message["parts"]
            .as_array()
            .ok_or_else(|| Fault::invalid("parts must be an array"))?;
        let response = parts
            .iter()
            .find(|part| part["kind"] == "data")
            .map(|part| &part["data"])
            .ok_or_else(|| {
                Fault::invalid("pending input requires a data part with inputId and response")
            })?;
        app.sessions.respond(
            p,
            &session.id,
            &turn.id,
            string(response, "inputId")?,
            response["response"].clone(),
        )?;
        Mapping {
            akh: akh.into(),
            message: message.clone(),
            session: session.id,
            turn: turn.id,
        }
    } else {
        let profile = app
            .sessions
            .config
            .akhs
            .get(akh)
            .ok_or_else(|| Fault::missing("Akh profile not found"))?;
        let session = if let Some(id) = message["contextId"].as_str() {
            let session = app.sessions.get(p, id)?;
            if session.akh != akh {
                return Err(Fault::conflict("context belongs to another Akh"));
            }
            session
        } else {
            let workspace = params["metadata"]["workspaceProfile"]
                .as_str()
                .map(str::to_owned)
                .or_else(|| {
                    if profile.workspace_profiles.len() == 1 {
                        profile.workspace_profiles.first().cloned()
                    } else {
                        None
                    }
                })
                .ok_or_else(|| Fault::invalid("select metadata.workspaceProfile"))?;
            app.sessions.create(
                p,
                &NewSession {
                    request_id: format!("a2a-{}", crate::workspace::digest(message_id.as_bytes())),
                    akh: akh.into(),
                    workspace_profile: workspace,
                    conversation_key: None,
                },
            )?
        };
        crate::server::controller(app, p, &session.id, None)?;
        let parts = message["parts"]
            .as_array()
            .ok_or_else(|| Fault::invalid("parts must be an array"))?;
        if parts
            .iter()
            .any(|part| part["kind"] != "text" || !part["text"].is_string())
        {
            return Err(Fault::new(
                "unsupported",
                "new task inputs currently require text parts; no remote URI fetch is performed",
            ));
        }
        let text = parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let turn = app.sessions.submit(
            p,
            &session.id,
            &SubmitInput {
                request_id: format!("a2a-{}", crate::workspace::digest(message_id.as_bytes())),
                text,
                artifacts: vec![],
            },
        )?;
        Mapping {
            akh: akh.into(),
            message: message.clone(),
            session: session.id,
            turn: turn.id,
        }
    };
    app.sessions.store.transaction(|tx| tx.put(&k, &mapping))?;
    Ok(mapping)
}
fn task(app: &SessionApp, p: &Principal, session: &str, turn: &str) -> Result<Value> {
    let session = app.sessions.get(p, session)?;
    let turn = session
        .turns
        .iter()
        .find(|t| t.id == turn)
        .ok_or_else(|| Fault::missing("task not found"))?;
    let pending = app
        .sessions
        .pending_inputs(p, &session.id)?
        .into_iter()
        .filter(|input| input.turn_id == turn.id)
        .collect::<Vec<_>>();
    let state = match turn.status {
        TurnStatus::Queued | TurnStatus::Dispatching => "submitted",
        TurnStatus::Running => "working",
        TurnStatus::AwaitingInput => "input-required",
        TurnStatus::Completed => "completed",
        TurnStatus::Failed => "failed",
        TurnStatus::Cancelled => "canceled",
        TurnStatus::Unknown => "unknown",
    };
    let mut status = json!({"state":state});
    if !pending.is_empty() {
        status["message"] = json!({"kind":"message","role":"agent","messageId":format!("{}-input",turn.id),"taskId":turn.id,"contextId":session.id,"parts":[{"kind":"data","data":{"pending":pending,"responseFormat":{"inputId":"pending.id","response":"typed answer, or {approved: boolean} for an approval"}}}]});
    }
    let artifacts=turn.output.as_ref().map(|text|vec![json!({"artifactId":format!("{}-answer",turn.id),"parts":[{"kind":"text","text":text}]})]).unwrap_or_default();
    Ok(
        json!({"kind":"task","id":turn.id,"contextId":session.id,"status":status,"artifacts":artifacts,"history":[{"kind":"message","role":"user","messageId":turn.request_id,"parts":[{"kind":"text","text":turn.input.text}]}],"metadata":{"akh":session.akh,"generation":turn.generation,"error":turn.error,"executionEvidence":"workspace effects are recorded separately as executor receipts"}}),
    )
}
