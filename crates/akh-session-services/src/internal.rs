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
    if endpoint
        .reasoning_effort
        .as_deref()
        .is_some_and(|effort| !matches!(effort, "none" | "low" | "medium" | "high" | "max"))
    {
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
        let span = tracing::info_span!(
            "runtime.model.iteration",
            iteration,
            otel.status_code = tracing::field::Empty
        );
        let _entered = span.enter();
        check(cancel, deadline).map_err(model_boundary)?;
        let mut messages = vec![json!({"role":"system","content":request.instructions})];
        messages.extend(conversation.messages.iter().cloned());
        let mut body = json!({"model":request.binding.model,"messages":messages,"stream":true,"stream_options":{"include_usage":true},"max_tokens":endpoint.max_output_tokens});
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
        let (message, finish) = handle
            .block_on(completion(call, host, cancel, deadline))
            .map_err(model_boundary)?;
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
        // Validate the entire model plan before persisting it or executing its
        // first call. Malformed model output is not an uncertain tool outcome.
        let mut ids = HashSet::new();
        let mut plan = Vec::new();
        for call in &calls {
            let raw_id = call["id"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 512)
                .ok_or_else(|| Fault::new("model_incomplete", "model tool call has no valid ID"))?;
            if !ids.insert(raw_id) {
                return Err(Fault::new(
                    "model_incomplete",
                    "duplicate model tool call ID",
                ));
            }
            let name = call["function"]["name"]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| Fault::new("model_incomplete", "tool name missing"))?;
            let arguments: Value =
                serde_json::from_str(call["function"]["arguments"].as_str().ok_or_else(|| {
                    Fault::new("model_incomplete", "tool arguments must be JSON text")
                })?)
                .map_err(|_| {
                    Fault::new("model_incomplete", "model tool arguments are invalid JSON")
                })?;
            if !arguments.is_object() {
                return Err(Fault::new(
                    "model_incomplete",
                    "model tool arguments must be an object",
                ));
            }
            plan.push((raw_id.to_owned(), name.to_owned(), arguments));
        }
        let output = message["content"].as_str().unwrap_or("").to_owned();
        conversation.messages.push(message);
        // A validated tool plan is durable before any callback can have effects.
        store.transaction(|tx| tx.put(&k, &conversation))?;
        if plan.is_empty() {
            conversation.active_job = None;
            store.transaction(|tx| tx.put(&k, &conversation))?;
            return Ok(output);
        }
        for (index, (raw_id, name, arguments)) in plan.into_iter().enumerate() {
            check(cancel, deadline).map_err(model_boundary)?;
            let id = format!("internal-{iteration}-{index}");
            let result = if request.tools.iter().any(|t| t.name == name) {
                host.tool(&id, &name, arguments)
                    .map_err(|e| Fault::new("tool_callback", e))?
            } else {
                ToolResult::denied(&name, None)
            };
            conversation.messages.push(json!({"role":"tool","tool_call_id":raw_id,"content":serde_json::to_string(&result).map_err(|_| Fault::invalid("cannot encode tool result"))?}));
            store.transaction(|tx| tx.put(&k, &conversation))?;
        }
    }
    Err(Fault::new(
        "model_incomplete",
        "model iteration limit reached without a final answer",
    ))
}

// Only used where no broker callback is outstanding and all previous tool
// results are durable. Never wrap callback or storage failures with this code.
fn model_boundary(error: Fault) -> Fault {
    Fault::new(
        if error.code == "cancelled" {
            "model_cancelled"
        } else {
            "model_incomplete"
        },
        error.to_string(),
    )
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
    reasoning: BTreeMap<String, String>,
    calls: BTreeMap<(usize, usize), Call>,
    generations: BTreeMap<usize, usize>,
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
        if let Some(usage) = value.get("usage") {
            crate::telemetry::usage(usage);
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
        // Preserve the provider's exact field and text across tool rounds.
        for field in ["reasoning_content", "reasoning"] {
            if let Some(text) = choice["delta"][field]
                .as_str()
                .filter(|text| !text.is_empty())
            {
                self.reasoning
                    .entry(field.into())
                    .or_default()
                    .push_str(text);
                host.event(RuntimeEvent::Reasoning { text: text.into() })
                    .map_err(Fault::invalid)?;
            }
        }
        if let Some(text) = choice["delta"]["content"]
            .as_str()
            .filter(|text| !text.is_empty())
        {
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
                let generation = self.generations.entry(index).or_default();
                // Ollama can emit several complete calls with index=0 in one
                // delta. A distinct full ID plus two complete argument objects
                // establishes a new call, never a fragment of the old call.
                let complete_object =
                    |text: &str| serde_json::from_str::<Value>(text).is_ok_and(|v| v.is_object());
                if let Some(previous) = self.calls.get(&(index, *generation)) {
                    let new_id = delta["id"].as_str().unwrap_or("");
                    if !new_id.is_empty()
                        && !previous.id.is_empty()
                        && new_id != previous.id
                        && delta["function"]["name"]
                            .as_str()
                            .is_some_and(|s| !s.is_empty())
                        && complete_object(&previous.arguments)
                        && delta["function"]["arguments"]
                            .as_str()
                            .is_some_and(complete_object)
                    {
                        *generation += 1;
                    }
                }
                if !self.calls.contains_key(&(index, *generation)) && self.calls.len() >= 128 {
                    return Err(Fault::new("model_incomplete", "too many tool calls"));
                }
                let call = self.calls.entry((index, *generation)).or_default();
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
        let mut response = None;
        for attempt in 1..=3u64 {
            check(cancel, deadline)?;
            let span = tracing::info_span!(
                "runtime.provider.attempt",
                attempt,
                http.response.status_code = tracing::field::Empty,
                otel.status_code = tracing::field::Empty
            );
            let result = async {
                let mut operation = crate::telemetry::Operation::new("provider_request");
                let sent = call
                    .try_clone()
                    .ok_or_else(|| Fault::invalid("model request is not replayable"))?
                    .send()
                    .await
                    .map_err(|_| {
                        Fault::new(
                            "transport",
                            "model request failed; ambiguous transport failures are not retried",
                        )
                    })?;
                span.record("http.response.status_code", sent.status().as_u16());
                if sent.status().is_success() {
                    operation.success();
                } else {
                    span.record("otel.status_code", "ERROR");
                }
                Ok::<_, Fault>(sent)
            };
            let sent = tracing::Instrument::instrument(result, span.clone()).await?;
            if sent.status().is_success() {
                response = Some(sent);
                break;
            }
            let status = sent.status().as_u16();
            let retry_after = sent
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned);
            let request_id = sent
                .headers()
                .get("x-request-id")
                .or_else(|| sent.headers().get("request-id"))
                .and_then(|v| v.to_str().ok())
                .map(safe_identifier)
                .unwrap_or_default();
            let (code, kind) = provider_error(sent).await;
            span.in_scope(|| tracing::warn!(event.name="runtime.provider.error", http.response.status_code=status,
                provider.error.code=%code, provider.error.kind=%kind, provider.request_id=%request_id,
                retry_after=?retry_after, "provider rejected request"));
            let delay = retry_delay(status, attempt, retry_after.as_deref());
            if let Some(delay) = delay {
                tracing::info!(
                    event.name = "runtime.provider.backoff",
                    attempt,
                    delay_seconds = delay,
                    "waiting before retry of rejected request"
                );
                tokio::time::sleep(Duration::from_secs(delay)).await;
            } else {
                return Err(Fault::new(
                    "model_error",
                    format!(
                        "model returned HTTP {status}; code={code}; type={kind}; request_id={request_id}; attempts={attempt}"
                    ),
                ));
            }
        }
        let mut response =
            response.ok_or_else(|| Fault::new("model_error", "model attempt budget exhausted"))?;
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
        let mut operation = crate::telemetry::Operation::new("provider_stream");
        let started = Instant::now();
        let mut first_chunk = true;
        let mut stream = Stream::default();
        let mut buffer = vec![];
        let mut bytes = 0;
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|_| Fault::new("transport", "model stream interrupted"))?
        {
            check(cancel, deadline)?;
            if first_chunk {
                tracing::info!(
                    event.name = "runtime.provider.first_chunk",
                    duration_seconds = started.elapsed().as_secs_f64(),
                    "first provider stream chunk"
                );
                first_chunk = false;
            }
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
        for (field, text) in stream.reasoning {
            message[&field] = json!(text);
        }
        if !stream.calls.is_empty() {
            message["tool_calls"]=Value::Array(stream.calls.into_values().map(|c|json!({"type":"function","id":c.id,"function":{"name":c.name,"arguments":c.arguments}})).collect());
        }
        operation.success();
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

// Only explicit 429 rejections are replayed, never partial streams or tools.
fn retry_delay(status: u16, attempt: u64, retry_after: Option<&str>) -> Option<u64> {
    if status != 429 || attempt >= 3 {
        return None;
    }
    match retry_after {
        Some(value) => value.parse::<u64>().ok().filter(|seconds| *seconds <= 30),
        None => Some(1 << (attempt - 1)),
    }
}
fn safe_identifier(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
        .take(128)
        .collect()
}
async fn provider_error(mut response: reqwest::Response) -> (String, String) {
    let mut bytes = Vec::new();
    while let Ok(Some(chunk)) = response.chunk().await {
        if bytes.len() + chunk.len() > 16 * 1024 {
            break;
        }
        bytes.extend_from_slice(&chunk);
    }
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    let identifier = |field: &str| {
        value["error"][field]
            .as_str()
            .map(safe_identifier)
            .unwrap_or_default()
    };
    let kind = identifier("type");
    let kind = if kind.is_empty() {
        classify_provider_message(
            value["error"]["message"]
                .as_str()
                .or_else(|| value["error"].as_str())
                .unwrap_or(""),
        )
        .to_owned()
    } else {
        kind
    };
    (identifier("code"), kind)
}
fn classify_provider_message(message: &str) -> &'static str {
    let text = message.to_ascii_lowercase();
    if text.contains("quota") || text.contains("credit") || text.contains("billing") {
        "quota_or_billing"
    } else if text.contains("rate limit") || text.contains("too many requests") {
        "rate_limit"
    } else if text.contains("usage limit") {
        "usage_limit"
    } else {
        "unclassified"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retry_policy_never_replays_ambiguous_or_long_delayed_requests() {
        assert_eq!(
            classify_provider_message("account quota exhausted"),
            "quota_or_billing"
        );
        assert_eq!(
            classify_provider_message("Rate limit reached"),
            "rate_limit"
        );
        assert_eq!(
            classify_provider_message("monthly usage limit"),
            "usage_limit"
        );

        assert_eq!(retry_delay(429, 1, None), Some(1));
        assert_eq!(retry_delay(429, 2, Some("0")), Some(0));
        assert_eq!(retry_delay(429, 3, None), None);
        assert_eq!(retry_delay(500, 1, None), None);
        assert_eq!(retry_delay(429, 1, Some("31")), None);
        assert_eq!(
            retry_delay(429, 1, Some("Wed, 21 Oct 2026 07:28:00 GMT")),
            None
        );
    }
    #[tokio::test]
    async fn rejected_requests_have_bounded_attempts_and_safe_diagnostics() {
        use axum::{Router, routing::post};
        use std::sync::{Arc, atomic::AtomicUsize};
        let attempts = Arc::new(AtomicUsize::new(0));
        let observed = attempts.clone();
        let app = Router::new().route("/", post(move || {
            observed.fetch_add(1, Ordering::SeqCst);
            async { (axum::http::StatusCode::TOO_MANY_REQUESTS,
                [("retry-after", "0"), ("x-request-id", "provider-123")],
                r#"{"error":{"code":"insufficient_quota","type":"quota_error","message":"SECRET must not leak"}}"#) }
        }));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let mut host = Witness::default();
        let error = completion(
            reqwest::Client::new().post(url).json(&json!({})),
            &mut host,
            &AtomicBool::new(false),
            Instant::now() + Duration::from_secs(5),
        )
        .await
        .unwrap_err();
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
        assert!(error.message.contains("insufficient_quota"));
        assert!(error.message.contains("provider-123"));
        assert!(!error.message.contains("SECRET"));
        assert!(host.events.is_empty());
        assert_eq!(host.calls, 0);
        server.abort();
    }
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
    fn complete_calls_reusing_ollama_index_remain_separate() {
        let mut stream = Stream::default();
        let mut witness = Witness::default();
        let delta = json!({"choices":[{"index":0,"delta":{"tool_calls":[
            {"index":0,"id":"one","function":{"name":"memory_recall","arguments":"{\"query_symbols\":\"ledger\"}"}},
            {"index":0,"id":"two","function":{"name":"monitoring_snapshot","arguments":"{}"}}
        ]},"finish_reason":"tool_calls"}]});
        stream
            .line(format!("data: {delta}").as_bytes(), &mut witness)
            .unwrap();
        let calls = stream.calls.values().collect::<Vec<_>>();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            (calls[0].id.as_str(), calls[0].name.as_str()),
            ("one", "memory_recall")
        );
        assert_eq!(
            (
                calls[1].id.as_str(),
                calls[1].name.as_str(),
                calls[1].arguments.as_str()
            ),
            ("two", "monitoring_snapshot", "{}")
        );
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
                    json!({"choices":[{"index":0,"delta":{"reasoning_content":"Check the source first.","tool_calls":[{"index":0,"id":"native","function":{"name":"read","arguments":"{\"path\":"}}]},"finish_reason":null}]}),
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
            assert!(host.events.iter().any(|e| matches!(e, RuntimeEvent::Reasoning { text } if text == "Check the source first.")));
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
        assert!(
            requests
                .iter()
                .all(|body| body["reasoning_effort"] == "none")
        );
        assert_eq!(requests[1]["messages"][3]["tool_call_id"], "native");
        assert_eq!(
            requests[1]["messages"][2]["reasoning_content"],
            "Check the source first."
        );
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
