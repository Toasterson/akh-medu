//! Model/tool loop owned by akhrtd. Cognitive context comes from akhomed;
//! every effective tool call still traverses the canonical session broker.
use crate::{
    config::Endpoint,
    store::{Store, key},
    types::*,
};
use akh_runtime::Host;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const MAX_BYTES: usize = 8 * 1024 * 1024;
#[derive(Serialize, Deserialize)]
struct Conversation {
    id: String,
    session_id: String,
    binding: RuntimeBinding,
    messages: Vec<Value>,
    active_job: Option<String>,
    #[serde(default)]
    retired: bool,
}

pub fn validate(profile: &ProviderConfig, request: &RuntimeJobRequest) -> Result<()> {
    let endpoint = profile
        .model_endpoint
        .as_ref()
        .ok_or_else(|| Fault::invalid("internal runtime needs a model_endpoint"))?;
    if endpoint.token_env.as_ref().is_some_and(|name| {
        name.is_empty()
            || !name
                .bytes()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_')
            || name.as_bytes()[0].is_ascii_digit()
    }) || (endpoint.token_env.is_some() && endpoint.token_file.is_some())
    {
        return Err(Fault::invalid(
            "model authentication requires one valid credential source",
        ));
    }
    Endpoint {
        token_env: None,
        ca_file: None,
        url: endpoint.url.clone(),
        token_file: endpoint
            .token_file
            .clone()
            .unwrap_or_else(|| "/unused".into()),
        allow_insecure_http: endpoint.allow_insecure_http,
    }
    .validate()?;
    if endpoint.reasoning_effort.as_deref().is_some_and(|effort| !matches!(effort, "none" | "low" | "medium" | "high" | "max")) {
        return Err(Fault::invalid("unsupported reasoning effort"));
    }
    if profile.kind != RuntimeKind::Internal
        || profile.execution_mode != ExecutionMode::Brokered
        || profile.timeout_secs == 0
        || endpoint.max_iterations == 0
        || endpoint.max_iterations > 256
        || endpoint.max_output_tokens == 0
        || endpoint.max_output_tokens > 65536
        || request
            .binding
            .model
            .as_ref()
            .is_none_or(|m| m.trim().is_empty())
    {
        return Err(Fault::invalid(
            "internal runtime requires brokered tools, a model, and bounded positive iteration/token/time limits",
        ));
    }
    if !request.input.artifacts.is_empty() {
        return Err(Fault::new(
            "unsupported",
            "this model transport accepts text and event data; use workspace tools for artifacts",
        ));
    }
    Ok(())
}

pub fn run(
    store: &Store,
    profile: &ProviderConfig,
    request: &RuntimeJobRequest,
    host: &mut dyn Host,
    cancel: &AtomicBool,
) -> std::result::Result<String, String> {
    run_loop(store, profile, request, host, cancel).map_err(|error| error.to_string())
}
pub(crate) fn run_loop(
    store: &Store,
    profile: &ProviderConfig,
    request: &RuntimeJobRequest,
    host: &mut dyn Host,
    cancel: &AtomicBool,
) -> Result<String> {
    validate(profile, request)?;
    let endpoint = profile.model_endpoint.as_ref().unwrap();
    let id = request
        .provider_session
        .clone()
        .unwrap_or_else(|| format!("internal-{}", new_id()));
    let k = key("internal", &[&id]);
    let mut conversation: Conversation = store.transaction(|tx| {
        let mut conversation = match tx.get::<Conversation>(&k)? {
            Some(c) if request.provider_session.is_some() => c,
            Some(_) => return Err(Fault::conflict("internal continuation collision")),
            None if request.provider_session.is_some() => {
                return Err(Fault::missing("exact internal continuation is unavailable"));
            }
            None => Conversation {
                id: id.clone(),
                session_id: request.session_id.clone(),
                binding: request.binding.clone(),
                messages: vec![],
                active_job: None,
                retired: false,
            },
        };
        if conversation.binding != request.binding || conversation.session_id != request.session_id
        {
            return Err(Fault::new(
                "forbidden",
                "continuation belongs to another session or runtime profile",
            ));
        }
        if conversation.retired {
            return Err(Fault::new(
                "continuation_retired",
                "operator closed an interrupted continuation; create an explicit new session",
            ));
        }
        if conversation.active_job.is_some() {
            return Err(Fault::new(
                "unknown_outcome",
                "unfinished model continuation requires reconciliation; never replay it",
            ));
        }
        conversation.active_job = Some(request.id.clone());
        conversation
            .messages
            .push(json!({"role":"user","content":request.input.provider_prompt()}));
        tx.put(&k, &conversation)?;
        Ok(conversation)
    })?;
    host.event(RuntimeEvent::Bound {
        provider_session: id,
        model: request.binding.model.clone(),
    })
    .map_err(Fault::invalid)?;
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| Fault::new("transport", "model client could not initialize"))?;
    let handle = tokio::runtime::Handle::current();
    let deadline = Instant::now() + Duration::from_secs(profile.timeout_secs);
    let tools:Vec<Value> = request.tools.iter().map(|tool| json!({"type":"function","function":{"name":tool.name,"description":tool.description,"parameters":tool.input_schema}})).collect();
    for iteration in 0..endpoint.max_iterations {
        check(cancel, deadline)?;
        let mut messages = vec![json!({"role":"system","content":request.instructions})];
        messages.extend(conversation.messages.iter().cloned());
        let mut body = json!({"model":request.binding.model,"messages":messages,"stream":true,"max_tokens":endpoint.max_output_tokens});
        if let Some(effort) = &endpoint.reasoning_effort {
            body["reasoning_effort"] = json!(effort);
        }
        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        if serde_json::to_vec(&body)
            .map_err(|_| Fault::invalid("cannot encode model request"))?
            .len()
            > MAX_BYTES
        {
            return Err(Fault::new(
                "capacity",
                "internal continuation exceeded its context byte budget; no automatic history loss",
            ));
        }
        let mut call = client.post(&endpoint.url).json(&body);
        if let Some(path) = &endpoint.token_file {
            call = call.bearer_auth(crate::config::read_token(path)?);
        } else if let Some(name) = &endpoint.token_env {
            let token = std::env::var(name).map_err(|_| {
                Fault::new(
                    "auth_unavailable",
                    "model credential environment unavailable",
                )
            })?;
            if token.trim().len() < 24 || token.chars().any(char::is_whitespace) {
                return Err(Fault::new(
                    "auth_unavailable",
                    "invalid model credential environment",
                ));
            }
            call = call.bearer_auth(token);
        }
        let (message, finish) = handle.block_on(completion(call, host, cancel, deadline))?;
        let calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if calls.is_empty() && finish != "stop" {
            return Err(Fault::new(
                "model_incomplete",
                "model did not finish normally",
            ));
        }
        if !calls.is_empty() && finish != "tool_calls" {
            return Err(Fault::new(
                "model_incomplete",
                "model tool response was truncated or malformed",
            ));
        }
        let output = message["content"].as_str().unwrap_or("").to_owned();
        conversation.messages.push(message);
        // A returned tool plan is durable before any callback can have effects.
        store.transaction(|tx| tx.put(&k, &conversation))?;
        if calls.is_empty() {
            conversation.active_job = None;
            store.transaction(|tx| tx.put(&k, &conversation))?;
            return Ok(output);
        }
        let mut ids = HashSet::new();
        for (index, call) in calls.iter().enumerate() {
            let raw_id = call["id"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 512)
                .ok_or_else(|| Fault::invalid("model tool call has no valid ID"))?;
            if !ids.insert(raw_id) {
                return Err(Fault::invalid("duplicate model tool call ID"));
            }
            check(cancel, deadline)?;
            let name = call["function"]["name"]
                .as_str()
                .ok_or_else(|| Fault::invalid("tool name missing"))?;
            let arguments: Value = serde_json::from_str(
                call["function"]["arguments"]
                    .as_str()
                    .ok_or_else(|| Fault::invalid("tool arguments must be JSON text"))?,
            )
            .map_err(|_| Fault::invalid("model tool arguments are invalid JSON"))?;
            let id = format!("internal-{iteration}-{index}");
            let result = if request.tools.iter().any(|t| t.name == name) {
                host.tool(&id, name, arguments)
                    .map_err(|e| Fault::new("tool_callback", e))?
            } else {
                ToolResult::denied(name, None)
            };
            conversation.messages.push(json!({"role":"tool","tool_call_id":raw_id,"content":serde_json::to_string(&result).map_err(|_| Fault::invalid("cannot encode tool result"))?}));
            store.transaction(|tx| tx.put(&k, &conversation))?;
        }
    }
    Err(Fault::new(
        "capacity",
        "model iteration limit reached without a final answer",
    ))
}

pub fn reconcile(
    tx: &mut crate::store::Transaction<'_>,
    job: &RuntimeJob,
    completed: bool,
) -> Result<()> {
    let Some(id) = &job.provider_session else {
        return Ok(());
    };
    let k = key("internal", &[id]);
    let Some(mut conversation) = tx.get::<Conversation>(&k)? else {
        return Err(Fault::missing("internal continuation is unavailable"));
    };
    if conversation.session_id != job.request.session_id {
        return Err(Fault::conflict("continuation owner differs"));
    }
    if completed && conversation.active_job.is_some() {
        return Err(Fault::conflict(
            "unfinished internal model continuation cannot be certified complete",
        ));
    }
    if !completed && conversation.active_job.is_some() {
        conversation.retired = true;
        tx.put(&k, &conversation)?;
    }
    Ok(())
}
fn check(cancel: &AtomicBool, deadline: Instant) -> Result<()> {
    if cancel.load(Ordering::SeqCst) {
        return Err(Fault::new("cancelled", "internal runtime cancelled"));
    }
    if Instant::now() >= deadline {
        return Err(Fault::new(
            "timeout",
            "internal runtime exceeded its turn deadline",
        ));
    }
    Ok(())
}

#[derive(Default)]
struct Call {
    id: String,
    name: String,
    arguments: String,
}
#[derive(Default)]
struct Stream {
    text: String,
    calls: BTreeMap<usize, Call>,
    finish: Option<String>,
    done: bool,
}
impl Stream {
    fn line(&mut self, line: &[u8], host: &mut dyn Host) -> Result<()> {
        let line = std::str::from_utf8(line)
            .map_err(|_| Fault::invalid("model stream is not UTF-8"))?
            .trim_end_matches('\r');
        let Some(data) = line.strip_prefix("data:").map(str::trim_start) else {
            return Ok(());
        };
        if data == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        if self.done {
            return Err(Fault::invalid("model sent data after stream completion"));
        }
        let value: Value = serde_json::from_str(data)
            .map_err(|_| Fault::invalid("model stream contains invalid JSON"))?;
        if value.get("error").is_some() {
            return Err(Fault::new("model_error", "model reported an error"));
        }
        let Some(choices) = value["choices"].as_array() else {
            return Err(Fault::invalid("model stream has no choices"));
        };
        if choices.is_empty() {
            return Ok(());
        } // usage-only trailer
        if choices.len() != 1 || choices[0]["index"] != 0 {
            return Err(Fault::invalid("only one model choice is supported"));
        }
        let choice = &choices[0];
        if self.finish.is_some() {
            return Err(Fault::invalid(
                "model sent another choice after its finish reason",
            ));
        }
        if let Some(text) = choice["delta"]["content"].as_str() {
            self.text.push_str(text);
            host.event(RuntimeEvent::Text { text: text.into() })
                .map_err(Fault::invalid)?;
        }
        if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
            for delta in calls {
                let index = delta["index"]
                    .as_u64()
                    .filter(|i| *i < 128)
                    .ok_or_else(|| Fault::invalid("invalid tool delta index"))?
                    as usize;
                let call = self.calls.entry(index).or_default();
                for (target, value) in [
                    (&mut call.id, &delta["id"]),
                    (&mut call.name, &delta["function"]["name"]),
                    (&mut call.arguments, &delta["function"]["arguments"]),
                ] {
                    if let Some(text) = value.as_str() {
                        target.push_str(text);
                    }
                }
            }
        }
        if let Some(reason) = choice["finish_reason"].as_str() {
            self.finish = Some(reason.into());
        }
        Ok(())
    }
}
async fn completion(
    call: reqwest::RequestBuilder,
    host: &mut dyn Host,
    cancel: &AtomicBool,
    deadline: Instant,
) -> Result<(Value, String)> {
    let work = async {
        let mut response = call.send().await.map_err(|_| {
            Fault::new(
                "transport",
                "model request failed; it will not be retried automatically",
            )
        })?;
        if !response.status().is_success() {
            return Err(Fault::new(
                "model_error",
                format!("model returned HTTP {}", response.status().as_u16()),
            ));
        }
        if !response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"))
        {
            return Err(Fault::new(
                "unsupported",
                "model endpoint must support chat-completions SSE",
            ));
        }
        let mut stream = Stream::default();
        let mut buffer = vec![];
        let mut bytes = 0;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| Fault::new("transport", "model stream interrupted"))?
        {
            check(cancel, deadline)?;
            bytes += chunk.len();
            if bytes > MAX_BYTES {
                return Err(Fault::new("capacity", "model output exceeded byte limit"));
            }
            buffer.extend_from_slice(&chunk);
            while let Some(end) = buffer.iter().position(|b| *b == b'\n') {
                // Buffered SSE can remain ready while callbacks block. Check
                // here too: select's interruption future may not get polled.
                check(cancel, deadline)?;
                stream.line(&buffer[..end], host)?;
                buffer.drain(..=end);
            }
            if stream.done {
                break;
            }
        }
        if !buffer.is_empty() {
            check(cancel, deadline)?;
            stream.line(&buffer, host)?;
        }
        let finish = stream.finish.ok_or_else(|| {
            Fault::new(
                "model_incomplete",
                "model stream ended without finish reason",
            )
        })?;
        if !stream.done {
            return Err(Fault::new(
                "model_incomplete",
                "model stream ended without DONE marker",
            ));
        }
        let mut message = json!({"role":"assistant","content":stream.text});
        if !stream.calls.is_empty() {
            message["tool_calls"]=Value::Array(stream.calls.into_values().map(|c|json!({"type":"function","id":c.id,"function":{"name":c.name,"arguments":c.arguments}})).collect());
        }
        Ok((message, finish))
    };
    let interrupted = async {
        loop {
            check(cancel, deadline)?;
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    };
    tokio::select! { value=work=>value, value=interrupted=>value }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Witness {
        events: Vec<RuntimeEvent>,
        calls: usize,
        cancel_on_text: Option<std::sync::Arc<AtomicBool>>,
    }
    impl Host for Witness {
        fn event(&mut self, event: RuntimeEvent) -> std::result::Result<(), String> {
            if matches!(event, RuntimeEvent::Text { .. })
                && let Some(cancel) = &self.cancel_on_text
            {
                cancel.store(true, Ordering::SeqCst);
            }
            self.events.push(event);
            Ok(())
        }
        fn tool(
            &mut self,
            id: &str,
            name: &str,
            args: Value,
        ) -> std::result::Result<ToolResult, String> {
            assert_eq!(id, "internal-0-0");
            assert_eq!(name, "read");
            assert_eq!(args, json!({"path":"file"}));
            self.calls += 1;
            Ok(ToolResult {
                success: true,
                text: "workspace contents".into(),
                data: None,
                receipt: Some("receipt".into()),
            })
        }
        fn approval(&mut self, _: &str, _: &str, _: &Value) -> std::result::Result<bool, String> {
            panic!("internal loop must use broker approvals")
        }
        fn question(&mut self, _: &str, _: &Value) -> std::result::Result<Value, String> {
            panic!("internal loop must use broker input")
        }
    }
    fn fixture(url: &str, dir: &std::path::Path) -> (ProviderConfig, RuntimeJobRequest) {
        let profile=serde_json::from_value(json!({"revision":"1","kind":"internal","working_dir":dir,"binary":"/unused","execution_mode":"brokered","timeout_secs":5,"model_endpoint":{"url":url,"max_iterations":4,"max_output_tokens":512}})).unwrap();
        let request=serde_json::from_value(json!({"id":"job","session_id":"session","turn_id":"turn","generation":1,"binding":{"runtime":"runtime","profile":"local","revision":"1","kind":"internal","model":"fixture","execution_mode":"brokered"},"workspace":{"id":"w","sandbox_id":"s","executor":"e","generation":1,"site":"test","root":"/remote-only","platform":"test","confinement":"test"},"input":{"origin":{"kind":"human","principal":"user"},"text":"read file"},"instructions":"Use supplied tools","tools":[{"name":"read","description":"read","input_schema":{"type":"object"},"read_only":true,"destination":{"kind":"workspace","operation":"read"}}],"approval_policy":"ask_writes"})).unwrap();
        (profile, request)
    }
    #[test]
    fn model_authentication_rejects_ambiguous_and_invalid_sources() {
        let (mut profile, request) = fixture(
            "https://example.com/v1/chat/completions",
            std::path::Path::new("/tmp"),
        );
        profile.model_endpoint.as_mut().unwrap().token_env = Some("ROMAN_MODEL_KEY".into());
        validate(&profile, &request).unwrap();
        profile.model_endpoint.as_mut().unwrap().token_file = Some("/private/key".into());
        assert!(validate(&profile, &request).is_err());
        profile.model_endpoint.as_mut().unwrap().token_file = None;
        for name in ["", "1KEY", "KEY=secret", "KEY\n"] {
            profile.model_endpoint.as_mut().unwrap().token_env = Some(name.into());
            assert!(validate(&profile, &request).is_err());
        }
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streams_tools_and_resumes_the_exact_durable_conversation() {
        use axum::{Json, Router, routing::post};
        let bodies = std::sync::Arc::new(std::sync::Mutex::new(Vec::<Value>::new()));
        let observed = bodies.clone();
        let app=Router::new().route("/v1/chat/completions",post(move |Json(body):Json<Value>| {
            let observed=observed.clone();
            async move {
                let mut requests=observed.lock().unwrap(); requests.push(body);
                let chunks=if requests.len()==1 { vec![
                    json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"native","function":{"name":"read","arguments":"{\"path\":"}}]},"finish_reason":null}]}),
                    json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"file\"}"}}]},"finish_reason":"tool_calls"}]})
                ] } else { vec![json!({"choices":[{"index":0,"delta":{"content":"Finished"},"finish_reason":"stop"}]})] };
                let mut body=chunks.iter().map(|v|format!("data: {v}\n\n")).collect::<String>(); body.push_str("data: [DONE]\n\n");
                ([("content-type","text/event-stream")],body)
            }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "http://{}/v1/chat/completions",
            listener.local_addr().unwrap()
        );
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let dir = tempfile::tempdir().unwrap();
        let (mut profile, mut request) = fixture(&url, dir.path());
        profile.model_endpoint.as_mut().unwrap().reasoning_effort = Some("none".into());
        let store = Store::open(&dir.path().join("runtime.redb")).unwrap();
        tokio::task::spawn_blocking(move || {
            let cancel = AtomicBool::new(false);
            let mut host = Witness::default();
            assert_eq!(
                run(&store, &profile, &request, &mut host, &cancel).unwrap(),
                "Finished"
            );
            assert_eq!(host.calls, 1);
            request.provider_session = host.events.iter().find_map(|e| {
                if let RuntimeEvent::Bound {
                    provider_session, ..
                } = e
                {
                    Some(provider_session.clone())
                } else {
                    None
                }
            });
            request.id = "job-2".into();
            request.turn_id = "turn-2".into();
            request.input.text = "continue".into();
            assert_eq!(
                run(&store, &profile, &request, &mut host, &cancel).unwrap(),
                "Finished"
            );
            request.session_id = "another-session".into();
            assert!(
                run(&store, &profile, &request, &mut host, &cancel)
                    .unwrap_err()
                    .contains("another session")
            );
            request.session_id = "session".into();
            request.provider_session = Some("missing".into());
            assert!(
                run(&store, &profile, &request, &mut host, &cancel)
                    .unwrap_err()
                    .contains("unavailable")
            );
        })
        .await
        .unwrap();
        let requests = bodies.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(requests.iter().all(|body| body["reasoning_effort"] == "none"));
        assert_eq!(requests[1]["messages"][3]["tool_call_id"], "native");
        assert!(
            requests[2]["messages"]
                .as_array()
                .unwrap()
                .iter()
                .any(|m| m["content"] == "continue")
        );
        server.abort();
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn buffered_stream_checks_cancellation_between_events() {
        use axum::{Router, routing::post};
        let body = [
            json!({"choices":[{"index":0,"delta":{"content":"first"},"finish_reason":null}]}),
            json!({"choices":[{"index":0,"delta":{"content":"must not be emitted"},"finish_reason":"stop"}]}),
        ].iter().map(|v| format!("data: {v}\n\n")).collect::<String>() + "data: [DONE]\n\n";
        let app = Router::new().route(
            "/",
            post(move || {
                let body = body.clone();
                async move { ([("content-type", "text/event-stream")], body) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let cancel = std::sync::Arc::new(AtomicBool::new(false));
        let mut host = Witness {
            cancel_on_text: Some(cancel.clone()),
            ..Default::default()
        };
        let error = completion(
            reqwest::Client::new().post(url),
            &mut host,
            &cancel,
            Instant::now() + Duration::from_secs(5),
        )
        .await
        .unwrap_err();
        assert_eq!(error.code, "cancelled");
        assert_eq!(
            host.events.len(),
            1,
            "buffered output continued after cancellation"
        );
        server.abort();
    }

    #[test]
    fn incomplete_streams_and_wrong_tool_finish_are_not_success() {
        let mut host = Witness::default();
        let mut stream = Stream::default();
        stream.line(b"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}",&mut host).unwrap();
        assert!(stream.finish.is_none());
        assert!(!stream.done);
        assert!(
            stream
                .line(b"data: {\"error\":{\"message\":\"sensitive\"}}", &mut host)
                .unwrap_err()
                .message
                .find("sensitive")
                .is_none()
        );
    }
}
