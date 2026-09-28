//! OpenAI-compatible chat/agentic backing (ADR 070 third peer).
//!
//! Ollama speaks its own `/api/chat` dialect and Candle runs in-process; this
//! module adds the third shape an akh can be backed by: any host that serves the
//! **OpenAI `/chat/completions` contract**. That covers vLLM, SGLang, llama.cpp's
//! server, OpenRouter, DeepSeek's own API, and Hetzner's inference experiment
//! (`https://inference.hetzner.com/api/v1`), which is what motivated it.
//!
//! # Why a translation layer and not a pass-through
//!
//! [`agentic_chat`](super::agentic_chat) builds its conversation in the **Ollama**
//! shape, because that was the only wire format when the loop was written:
//!
//! ```text
//! assistant: {"role":"assistant","content":…,"tool_calls":[{"function":{"name":…,"arguments":{…}}}]}
//! tool:      {"role":"tool","content":"…"}
//! ```
//!
//! A strict OpenAI server rejects both: assistant tool calls need a per-call `id`
//! and a **stringified** `arguments`, and every `role:"tool"` message must carry
//! the `tool_call_id` it answers. Rather than reshape the shared loop (and force
//! the Ollama path to grow ids it does not want), the translation lives here, at
//! the boundary that actually needs it — [`to_openai_messages`]. The loop stays
//! backend-agnostic and the [`ChatBackend`] contract is unchanged.
//!
//! # Error handling
//!
//! Failures surface as [`OpenAiError`] with miette codes/help, then narrow into
//! [`ChatBackendError`] at the trait boundary so the agentic loop's existing
//! health-gated fallback (LLM → symbolic pipeline) keeps working untouched.

use std::sync::RwLock;

use miette::Diagnostic;
use serde_json::{Value, json};
use thiserror::Error;

use super::chat_backend::{ChatBackend, ChatBackendError, ModelCapabilities, PreparedChat};
use super::ollama::{AssistantTurn, StreamEvent, ToolCall};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Where an OpenAI-compatible model lives and how to talk to it.
///
/// `base_url` is the API root **including** any version segment — the client
/// appends `/chat/completions` and `/models` to it verbatim. For Hetzner that is
/// `https://inference.hetzner.com/api/v1`; for OpenAI proper,
/// `https://api.openai.com/v1`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiConfig {
    /// API root, version segment included. No trailing slash required.
    pub base_url: String,
    /// Bearer token. Supports the `env:VAR` indirection resolved at call time
    /// (same convention as [`OllamaConfig`](super::ollama::OllamaConfig)), so a
    /// secret never has to be written into `akh.toml`.
    pub api_key: Option<String>,
    /// The model to request, e.g. `"DeepSeek-V4-Flash-0731"`.
    pub model: String,
    /// Per-request timeout. Coding turns are long; the default is generous.
    pub timeout_secs: u64,
    /// Context window in tokens. Unlike Ollama's `/api/show`, the OpenAI
    /// contract exposes **no** way to read this, so it is configuration rather
    /// than discovery. `None` leaves the Phase-44 budget on its own default.
    pub context_window: Option<usize>,
    /// Cap on generated tokens per turn. `None` ⇒ the server's default.
    pub max_tokens: Option<u32>,
    /// Sampling temperature. `None` ⇒ the server's default.
    pub temperature: Option<f32>,
    /// Reasoning-effort hint (`"low" | "medium" | "high"`), sent as
    /// `reasoning_effort`. Ignored by servers that don't implement it.
    pub reasoning_effort: Option<String>,
}

impl Default for OpenAiConfig {
    fn default() -> Self {
        Self {
            base_url: "https://inference.hetzner.com/api/v1".into(),
            api_key: None,
            model: String::new(),
            timeout_secs: 600,
            context_window: None,
            max_tokens: None,
            temperature: None,
            reasoning_effort: None,
        }
    }
}

impl OpenAiConfig {
    /// Resolve the bearer token, expanding an `env:VAR` reference.
    ///
    /// Returns `None` — never the literal `"env:VAR"` — when the referenced
    /// variable is unset or empty, so an unresolved indirection can't be
    /// mistakenly sent as a token.
    pub fn resolved_api_key(&self) -> Option<String> {
        let raw = self.api_key.as_ref()?;
        match raw.strip_prefix("env:") {
            Some(var) => match std::env::var(var) {
                Ok(val) if !val.is_empty() => Some(val),
                _ => {
                    tracing::warn!(
                        env_var = var,
                        "openai api_key references env:{var} but it is unset/empty; \
                         sending no Authorization header"
                    );
                    None
                }
            },
            None => Some(raw.clone()),
        }
    }

    /// `{base_url}/{path}` with exactly one separating slash.
    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url.trim_end_matches('/'), path)
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Failures talking to an OpenAI-compatible endpoint.
#[derive(Debug, Error, Diagnostic)]
pub enum OpenAiError {
    /// The host could not be reached at all (DNS, TLS, connection refused).
    #[error("OpenAI-compatible endpoint {url} is unreachable: {reason}")]
    #[diagnostic(
        code(akh::openai::unreachable),
        help(
            "Check the akh's `[openai_endpoint] base_url` and that this host has \
             network access to it. `curl -sS -H \"Authorization: Bearer $KEY\" \
             <base_url>/models` reproduces the call outside the daemon."
        )
    )]
    Unreachable { url: String, reason: String },

    /// The host answered with a non-2xx status.
    #[error(
        "OpenAI-compatible request to {url} failed with HTTP {status}{}",
        message.as_deref().map(|m| format!(": {m}")).unwrap_or_default()
    )]
    #[diagnostic(
        code(akh::openai::request_failed),
        help(
            "401/403 ⇒ the bearer token is missing, expired, or lacks access to \
             model `{model}`. 404 ⇒ `base_url` is probably missing its version \
             segment (it must end in e.g. `/v1`). 400 ⇒ the model rejected the \
             request body; the message above (when present) says which field."
        )
    )]
    RequestFailed {
        url: String,
        status: u16,
        model: String,
        /// Full, untruncated response body, kept for callers that want to
        /// inspect it programmatically. Never logged directly — use
        /// `message` (below) for display, which is bounded and redacted.
        body: String,
        /// Truncated (~600 chars), redacted copy of the response body's
        /// `error.message`, or the raw body when it isn't JSON / carries no
        /// `error.message`. `None` when the body was empty. Built only from
        /// the response body — never from request headers or the API key —
        /// and defensively scrubbed of anything that looks like a bearer
        /// token, in case a misbehaving proxy echoes the request back. This
        /// is what the `Display` impl (and therefore the WARN log at the
        /// call site) shows.
        message: Option<String>,
        /// `Retry-After` and structured `error.code`/`error.type`, for the
        /// provider-failure taxonomy (Phase 53g PR 4). Not part of the message.
        /// Boxed so the error stays small on every `Result` path.
        detail: Box<super::provider_failure::ProviderErrorDetail>,
    },

    /// A 2xx response that did not match the OpenAI response contract.
    #[error("OpenAI-compatible response could not be parsed: {reason}")]
    #[diagnostic(
        code(akh::openai::parse_failed),
        help(
            "The endpoint returned 2xx but not the documented shape \
             (`choices[0].message`). It may be an incompatible dialect — verify \
             it really implements /chat/completions."
        )
    )]
    ParseFailed { reason: String },

    /// The endpoint requires a token and none resolved.
    #[error("no API key resolved for OpenAI-compatible endpoint {url}")]
    #[diagnostic(
        code(akh::openai::no_api_key),
        help(
            "Set `[openai_endpoint] api_key` in the akh's config — either a \
             literal token or `env:VAR`. If `env:VAR`, make sure the daemon's \
             environment actually exports VAR (launchd does not inherit your \
             shell profile)."
        )
    )]
    NoApiKey { url: String },
}

impl OpenAiError {
    /// What this failure says about the provider, for the provider gate.
    /// `None` for failures on the akh's side (no key, an unparseable 2xx).
    pub fn provider_evidence(&self) -> Option<super::provider_failure::ProviderFailureEvidence> {
        use super::provider_failure::ProviderFailureEvidence;
        match self {
            Self::RequestFailed { status, detail, .. } => Some(ProviderFailureEvidence::Http {
                status: *status,
                detail: detail.as_ref().clone(),
            }),
            Self::Unreachable { .. } => Some(ProviderFailureEvidence::Transport),
            Self::ParseFailed { .. } | Self::NoApiKey { .. } => None,
        }
    }
}

/// Result alias for this module.
pub type OpenAiResult<T> = Result<T, OpenAiError>;

// ---------------------------------------------------------------------------
// Message translation (Ollama-shaped loop → OpenAI wire)
// ---------------------------------------------------------------------------

/// Deterministic id for the `idx`-th tool call of the `round`-th assistant turn.
///
/// Deterministic (not random) so a replayed conversation produces byte-identical
/// requests — which is what makes the A/B context-strategy harness comparable.
fn tool_call_id(round: usize, idx: usize) -> String {
    format!("call_{round}_{idx}")
}

/// Translate the loop's Ollama-shaped conversation into strict OpenAI messages.
///
/// Six fixes are applied during translation:
///
/// 1. **Assistant tool calls** gain an `id` and a `type: "function"`, and their
///    `arguments` object is stringified (OpenAI transports arguments as a JSON
///    *string*; Ollama uses a JSON *object*).
/// 2. **Tool results** gain the `tool_call_id` they answer, paired positionally
///    with the calls of the most recent assistant turn — which is exactly the
///    order [`agentic_chat`](super::agentic_chat) appends them in.
/// 3. **Assistant turns that only called tools** get their empty `content`
///    dropped rather than sent as `""`, which some strict servers reject.
/// 4. **System messages** are coalesced into one leading message. Some strict
///    chat templates (including Qwen) reject a system role anywhere else in
///    the conversation.
/// 5. **Tool content is always a string.** Gemma 4's pinned template calls
///    `.get()` on a tool message's `content` while resolving the response
///    block; a non-string `content` (an object or array, which nothing in
///    this codebase's own tool dispatch produces today, but a future tool or
///    a replayed/compacted history entry could) makes that call raise
///    `UndefinedError: 'str object' has no attribute 'get'` — a hard 400,
///    reproduced by rendering the pinned template offline. Anything that
///    isn't already a JSON string is serialized to one.
/// 6. **A tool result with no outstanding call** (malformed/compacted
///    history — the pending-id queue is empty) is folded into a `role:"user"`
///    message instead of emitted as `role:"tool"` with no `tool_call_id`.
///    The same pinned template raises `TypeError: can only concatenate str
///    (not "NoneType") to str` while formatting a tool-response block for a
///    call it can't resolve — also reproduced offline — so "let the server
///    complain about it" is not survivable for this template; folding keeps
///    the observation without ever sending the malformed shape. Adjacent
///    messages of the same non-tool role (this fold can produce two adjacent
///    `user` turns) are then merged into one, since some strict templates
///    also expect strict user/assistant alternation.
pub fn to_openai_messages(system_prompt: &str, conversation: &[Value]) -> Vec<Value> {
    let mut out = Vec::with_capacity(conversation.len() + 1);

    let mut system_sections = Vec::new();
    if !system_prompt.trim().is_empty() {
        system_sections.push(system_prompt.to_owned());
    }
    for msg in conversation {
        if msg.get("role").and_then(Value::as_str) == Some("system") {
            if let Some(content) = msg.get("content").and_then(Value::as_str) {
                if !content.trim().is_empty() {
                    system_sections.push(content.to_owned());
                }
            }
        }
    }
    if !system_sections.is_empty() {
        out.push(json!({"role": "system", "content": system_sections.join("\n\n")}));
    }

    // Ids minted by the most recent assistant turn, consumed in order by the
    // `role:"tool"` messages that follow it.
    let mut pending_ids: std::collections::VecDeque<String> = Default::default();
    let mut round = 0usize;

    for msg in conversation {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        match role {
            // Already coalesced above.  Never let a later system role reach a
            // strict chat template.
            "system" => {}
            "assistant" => {
                let calls = msg.get("tool_calls").and_then(Value::as_array);
                let mut translated = json!({"role": "assistant"});

                match calls.filter(|c| !c.is_empty()) {
                    Some(calls) => {
                        pending_ids.clear();
                        let wire: Vec<Value> = calls
                            .iter()
                            .enumerate()
                            .map(|(idx, tc)| {
                                let id = tool_call_id(round, idx);
                                pending_ids.push_back(id.clone());
                                let func = tc.get("function").unwrap_or(tc);
                                let name =
                                    func.get("name").and_then(Value::as_str).unwrap_or_default();
                                // Arguments may already be a string (if some
                                // caller pre-serialized); only stringify objects.
                                let args = match func.get("arguments") {
                                    Some(Value::String(s)) => s.clone(),
                                    Some(v) => v.to_string(),
                                    None => "{}".to_string(),
                                };
                                json!({
                                    "id": id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": args},
                                })
                            })
                            .collect();
                        translated["tool_calls"] = Value::Array(wire);
                        // Only carry content when the model actually said something.
                        if let Some(c) = msg.get("content").and_then(Value::as_str) {
                            if !c.is_empty() {
                                translated["content"] = json!(c);
                            }
                        }
                        round += 1;
                    }
                    None => {
                        translated["content"] =
                            msg.get("content").cloned().unwrap_or_else(|| json!(""));
                    }
                }
                out.push(translated);
            }
            "tool" => {
                let content = stringify_content(msg.get("content"));
                match pending_ids.pop_front() {
                    Some(id) => out.push(json!({
                        "role": "tool",
                        "content": content,
                        "tool_call_id": id,
                    })),
                    // No outstanding call to answer: emitting `role:"tool"`
                    // without `tool_call_id` is rejected outright by Gemma
                    // 4's pinned template (see the fix-5/6 doc comment
                    // above), so fold the observation into a `user` turn
                    // instead of losing it or sending the malformed shape.
                    None => out.push(json!({
                        "role": "user",
                        "content": format!("[unmatched tool result] {content}"),
                    })),
                }
            }
            _ => out.push(msg.clone()),
        }
    }
    merge_adjacent_same_role(out)
}

/// Coerce a possibly-absent, possibly-non-string `content` value to a JSON
/// string: pass a string through verbatim, serialize anything else (object,
/// array, number, `null`, missing) to its JSON text. See fix 5 above.
fn stringify_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Merge adjacent messages that share the same non-tool role, concatenating
/// their content with a blank line. `tool` messages are never merged (each
/// carries a distinct `tool_call_id`); `system` never reaches here (already
/// coalesced by the caller); a message with `tool_calls` is never merged
/// into a neighbor, since that would silently drop the calls' positional
/// pairing with their queued response ids.
fn merge_adjacent_same_role(messages: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let mergeable = role != "tool" && role != "system" && msg.get("tool_calls").is_none();
        if mergeable
            && let Some(prev) = out.last_mut()
            && prev.get("role").and_then(Value::as_str) == Some(role)
            && prev.get("tool_calls").is_none()
        {
            let prev_content = prev.get("content").and_then(Value::as_str).unwrap_or("");
            let next_content = msg.get("content").and_then(Value::as_str).unwrap_or("");
            let merged = match (prev_content.is_empty(), next_content.is_empty()) {
                (true, _) => next_content.to_owned(),
                (false, true) => prev_content.to_owned(),
                (false, false) => format!("{prev_content}\n\n{next_content}"),
            };
            prev["content"] = json!(merged);
            continue;
        }
        out.push(msg);
    }
    out
}

/// Parse `choices[0].message` into the backend-agnostic [`AssistantTurn`].
///
/// Mirrors the inverse of [`to_openai_messages`]: OpenAI's stringified
/// `arguments` are parsed back into a JSON object, since [`ToolCall::arguments`]
/// is what the tool dispatcher indexes into. Arguments that don't parse degrade
/// to an empty object rather than dropping the call — the tool's own validation
/// then produces a far better diagnostic than a silently vanished call would.
///
/// Only `message.content` becomes the returned content — `message.reasoning_content`
/// (present when the server runs with `--reasoning-parser`, e.g. Lilith's
/// pod, see `docs/ops/lilith-serve.md` "Reasoning parser") is read on the
/// streaming path (`chat_stream`, which surfaces it as [`StreamEvent::Thinking`])
/// but not here: [`AssistantTurn`] carries no separate reasoning field for the
/// non-streaming turn to populate. This is not a leak — a configured reasoning
/// parser keeps `content` itself clean of the `<|channel>thought…<channel|>`
/// markers regardless — it just means a non-streaming turn's separated
/// reasoning text is dropped rather than surfaced as a thinking step, unlike
/// its streaming counterpart.
fn parse_message(msg: &Value) -> (String, Vec<ToolCall>) {
    let content = msg
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let tool_calls = msg
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(|tc| {
                    let func = tc.get("function")?;
                    let name = func.get("name").and_then(Value::as_str)?.to_string();
                    let arguments = match func.get("arguments") {
                        Some(Value::String(s)) => {
                            serde_json::from_str(s).unwrap_or_else(|_| json!({}))
                        }
                        Some(v) => v.clone(),
                        None => json!({}),
                    };
                    Some(ToolCall { name, arguments })
                })
                .collect()
        })
        .unwrap_or_default();

    (content, tool_calls)
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Synchronous OpenAI-compatible chat client.
pub struct OpenAiClient {
    config: OpenAiConfig,
}

impl OpenAiClient {
    /// Wrap a config. Construction performs no I/O.
    pub fn new(config: OpenAiConfig) -> Self {
        Self { config }
    }

    /// The configured model name.
    pub fn model(&self) -> &str {
        &self.config.model
    }

    fn agent(&self) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()
    }

    /// `GET /models` — a cheap authenticated liveness probe.
    ///
    /// Used by [`ChatBackend::available`] and by operators to confirm a key
    /// works before pointing an akh at the endpoint.
    pub fn list_models(&self) -> OpenAiResult<Vec<String>> {
        let url = self.config.url("models");
        let key = self
            .config
            .resolved_api_key()
            .ok_or_else(|| OpenAiError::NoApiKey { url: url.clone() })?;

        let resp = self
            .agent()
            .get(&url)
            .set("Authorization", &format!("Bearer {key}"))
            .call()
            .map_err(|e| self.map_transport(e, &url))?;

        let body: Value = resp.into_json().map_err(|e| OpenAiError::ParseFailed {
            reason: format!("JSON decode failed: {e}"),
        })?;

        Ok(body
            .get("data")
            .and_then(Value::as_array)
            .map(|models| {
                models
                    .iter()
                    .filter_map(|m| m.get("id").and_then(Value::as_str))
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default())
    }

    /// One tool-enabled `/chat/completions` turn (non-streaming).
    pub fn chat(
        &self,
        system_prompt: &str,
        conversation: &[Value],
        tools: &[Value],
        request_seq: usize,
    ) -> OpenAiResult<AssistantTurn> {
        let url = self.config.url("chat/completions");
        let mut metrics = super::llm_metrics::LlmRequestMetrics::new(request_seq);
        let span =
            super::llm_metrics::LlmRequestMetrics::request_span(&self.config.model, request_seq);
        metrics.attach_span(span.clone());
        let _request_guard = span.entered();

        let body = self.build_body(system_prompt, conversation, tools, false);

        let key = self
            .config
            .resolved_api_key()
            .ok_or_else(|| OpenAiError::NoApiKey { url: url.clone() })?;

        let resp = self
            .agent()
            .post(&url)
            .set("Authorization", &format!("Bearer {key}"))
            .set("Content-Type", "application/json")
            .send_json(&body)
            .map_err(|e| {
                // No response (transport error or HTTP error status): not
                // billed, so not metered.
                metrics.log(
                    &self.config.model,
                    super::llm_metrics::RequestOutcome::NoResponse,
                );
                self.map_transport(e, &url)
            })?;

        let parsed: Value = resp.into_json().map_err(|e| {
            metrics.log(
                &self.config.model,
                super::llm_metrics::RequestOutcome::Partial,
            );
            OpenAiError::ParseFailed {
                reason: format!("JSON decode failed: {e}"),
            }
        })?;

        let msg = parsed
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(|c| c.get("message"))
            .ok_or_else(|| {
                metrics.log(
                    &self.config.model,
                    super::llm_metrics::RequestOutcome::Partial,
                );
                OpenAiError::ParseFailed {
                    reason: "choices[0].message missing in /chat/completions response".into(),
                }
            })?;

        let (content, tool_calls) = parse_message(msg);
        let (prompt_tokens, completion_tokens) = usage_of(&parsed);

        metrics.note_chunk(content.len());
        if !content.is_empty() {
            metrics.note_content_delta();
        }
        if !tool_calls.is_empty() {
            metrics.note_tool_call_chunk();
        }
        metrics.set_tokens(prompt_tokens, completion_tokens);
        metrics.log(
            &self.config.model,
            super::llm_metrics::RequestOutcome::Completed,
        );

        Ok(AssistantTurn {
            content,
            tool_calls,
            prompt_tokens,
            completion_tokens,
        })
    }

    /// Streaming counterpart: server-sent events, one `data:` line per delta.
    ///
    /// Content deltas are emitted as [`StreamEvent::Content`] and DeepSeek-style
    /// `reasoning_content` deltas as [`StreamEvent::Thinking`], so a reasoning
    /// model renders its chain-of-thought in the collapsible section rather than
    /// inline with the answer. Tool-call deltas arrive fragmented (the arguments
    /// string is split across chunks), so they are **accumulated by index** and
    /// only parsed once the stream closes.
    pub fn chat_stream(
        &self,
        system_prompt: &str,
        conversation: &[Value],
        tools: &[Value],
        request_seq: usize,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> OpenAiResult<AssistantTurn> {
        use std::io::BufRead;

        let url = self.config.url("chat/completions");
        let mut metrics = super::llm_metrics::LlmRequestMetrics::new(request_seq);
        let span =
            super::llm_metrics::LlmRequestMetrics::request_span(&self.config.model, request_seq);
        metrics.attach_span(span.clone());
        let _request_guard = span.entered();

        let body = self.build_body(system_prompt, conversation, tools, true);
        let key = self
            .config
            .resolved_api_key()
            .ok_or_else(|| OpenAiError::NoApiKey { url: url.clone() })?;

        let resp = self
            .agent()
            .post(&url)
            .set("Authorization", &format!("Bearer {key}"))
            .set("Content-Type", "application/json")
            .set("Accept", "text/event-stream")
            .send_json(&body)
            .map_err(|e| {
                // No response (transport error or HTTP error status): not
                // billed, so not metered.
                metrics.log(
                    &self.config.model,
                    super::llm_metrics::RequestOutcome::NoResponse,
                );
                self.map_transport(e, &url)
            })?;

        let mut content = String::new();
        // Tool calls stream in fragments keyed by `index`; keep insertion order.
        let mut acc: Vec<(String, String)> = Vec::new(); // (name, arguments-so-far)
        let (mut prompt_tokens, mut completion_tokens) = (0u64, 0u64);

        let reader = std::io::BufReader::new(resp.into_reader());
        for line in reader.lines() {
            let line = line.map_err(|e| {
                metrics.log(&self.config.model, metrics.interrupted_outcome());
                OpenAiError::Unreachable {
                    url: url.clone(),
                    reason: format!("stream read failed: {e}"),
                }
            })?;

            let Some(payload) = line.strip_prefix("data:") else {
                continue; // comments / blank separators
            };
            let payload = payload.trim();
            if payload.is_empty() || payload == "[DONE]" {
                continue;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
                continue; // a malformed frame must not abort a live turn
            };

            // Usage arrives on the final frame when stream_options asked for it.
            let (p, c) = usage_of(&chunk);
            if p > 0 || c > 0 {
                prompt_tokens = p;
                completion_tokens = c;
            }

            let delta = chunk
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|c| c.first())
                .and_then(|c| c.get("delta"));
            let Some(delta) = delta else {
                on_event(StreamEvent::Keepalive);
                continue;
            };

            let mut visible = false;
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    content.push_str(text);
                    metrics.note_content_delta();
                    on_event(StreamEvent::Content(text.to_string()));
                    visible = true;
                }
            }
            if let Some(text) = delta.get("reasoning_content").and_then(Value::as_str) {
                if !text.is_empty() {
                    metrics.note_thinking_delta();
                    on_event(StreamEvent::Thinking(text.to_string()));
                    visible = true;
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tc in calls {
                    let idx = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                    if acc.len() <= idx {
                        acc.resize(idx + 1, (String::new(), String::new()));
                    }
                    if let Some(func) = tc.get("function") {
                        if let Some(name) = func.get("name").and_then(Value::as_str) {
                            if !name.is_empty() {
                                acc[idx].0 = name.to_string();
                            }
                        }
                        if let Some(args) = func.get("arguments").and_then(Value::as_str) {
                            acc[idx].1.push_str(args);
                        }
                    }
                }
                metrics.note_tool_call_chunk();
                visible = true;
            }
            metrics.note_chunk(payload.len());
            if !visible {
                on_event(StreamEvent::Keepalive);
            }
        }

        let tool_calls: Vec<ToolCall> = acc
            .into_iter()
            .filter(|(name, _)| !name.is_empty())
            .map(|(name, args)| ToolCall {
                name,
                arguments: serde_json::from_str(&args).unwrap_or_else(|_| json!({})),
            })
            .collect();

        metrics.set_tokens(prompt_tokens, completion_tokens);
        metrics.log(
            &self.config.model,
            super::llm_metrics::RequestOutcome::Completed,
        );

        Ok(AssistantTurn {
            content,
            tool_calls,
            prompt_tokens,
            completion_tokens,
        })
    }

    /// Assemble the `/chat/completions` request body.
    fn build_body(
        &self,
        system_prompt: &str,
        conversation: &[Value],
        tools: &[Value],
        stream: bool,
    ) -> Value {
        let mut body = json!({
            "model": self.config.model,
            "messages": to_openai_messages(system_prompt, conversation),
            "stream": stream,
        });
        if !tools.is_empty() {
            body["tools"] = Value::Array(tools.to_vec());
            body["tool_choice"] = json!("auto");
        }
        if let Some(n) = self.config.max_tokens {
            body["max_tokens"] = json!(n);
        }
        if let Some(t) = self.config.temperature {
            body["temperature"] = json!(t);
        }
        if let Some(effort) = &self.config.reasoning_effort {
            body["reasoning_effort"] = json!(effort);
        }
        if stream {
            // Ask for a usage frame so streamed turns still report tokens.
            body["stream_options"] = json!({"include_usage": true});
        }
        body
    }

    /// Narrow a `ureq` transport error into the right [`OpenAiError`].
    fn map_transport(&self, e: ureq::Error, url: &str) -> OpenAiError {
        match e {
            ureq::Error::Status(status, resp) => {
                let (body, detail) =
                    super::provider_failure::ProviderErrorDetail::from_response(resp);
                let message =
                    extract_error_message(&body, self.config.resolved_api_key().as_deref());
                OpenAiError::RequestFailed {
                    url: url.to_string(),
                    status,
                    model: self.config.model.clone(),
                    body,
                    message,
                    detail: Box::new(detail),
                }
            }
            other => OpenAiError::Unreachable {
                url: url.to_string(),
                reason: other.to_string(),
            },
        }
    }
}

/// Maximum length (in `char`s) of the display message extracted from a
/// provider's error body. Long enough to show a real diagnostic, short
/// enough to keep a log line readable.
const ERROR_MESSAGE_MAX_CHARS: usize = 600;

/// Build the truncated, redacted display copy of an HTTP error's response
/// body: `error.message` when the body is the documented OpenAI error JSON
/// shape (`{"error":{"message": "..."}}`), otherwise the raw body verbatim
/// (some non-strict servers answer 400 with plain text or a different JSON
/// shape, and that text is still worth showing). `None` for an empty body.
///
/// `api_key`, when given, is redacted if it appears verbatim in the body —
/// defense in depth against a proxy that echoes the request back in its
/// error response. `Bearer <token>`-shaped substrings are always redacted
/// regardless, since a leaked body is a worse outcome than an over-eager
/// redaction.
fn extract_error_message(body: &str, api_key: Option<&str>) -> Option<String> {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return None;
    }
    let text = serde_json::from_str::<Value>(trimmed)
        .ok()
        .and_then(|v| {
            v.get("error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| trimmed.to_owned());
    let redacted = redact_secrets(&text, api_key);
    Some(truncate_chars(&redacted, ERROR_MESSAGE_MAX_CHARS))
}

/// Mask `Bearer <token>` substrings and any verbatim occurrence of `api_key`.
fn redact_secrets(text: &str, api_key: Option<&str>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.to_ascii_lowercase().find("bearer ") {
        out.push_str(&rest[..pos]);
        out.push_str("Bearer [REDACTED]");
        let after = &rest[pos + "bearer ".len()..];
        // Skip the token itself (up to the next whitespace or quote).
        let token_end = after
            .find(|c: char| c.is_whitespace() || c == '"' || c == '\'')
            .unwrap_or(after.len());
        rest = &after[token_end..];
    }
    out.push_str(rest);

    if let Some(key) = api_key.filter(|k| !k.is_empty()) {
        out = out.replace(key, "[REDACTED]");
    }
    out
}

/// Truncate `text` to at most `max_chars` characters (not bytes, so a
/// multi-byte codepoint is never split), appending `…` when truncated.
fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let mut truncated: String = text.chars().take(max_chars).collect();
    truncated.push('…');
    truncated
}

/// Read `usage.{prompt,completion}_tokens` from a response or stream frame.
fn usage_of(v: &Value) -> (u64, u64) {
    let usage = v.get("usage");
    let get = |k: &str| {
        usage
            .and_then(|u| u.get(k))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    (get("prompt_tokens"), get("completion_tokens"))
}

// ---------------------------------------------------------------------------
// ChatBackend impl
// ---------------------------------------------------------------------------

/// The OpenAI-compatible backing, as a [`ChatBackend`] the agentic loop can drive.
///
/// The system prompt is owned here (not by the model manager, which is
/// Ollama-specific) and refreshed from the akh's psyche on every
/// [`ChatBackend::prepare`], so persona edits take effect on the next turn
/// exactly as they do for the Ollama path.
pub struct OpenAiChatBackend {
    client: OpenAiClient,
    /// Explicit operator prompt from `akh.toml`, if any — composed with the
    /// machine-owned sections by [`assemble_system_prompt`](super::model_manager::assemble_system_prompt).
    explicit_prompt: Option<String>,
    /// The assembled prompt for the current psyche. `RwLock` because `prepare`
    /// takes `&self` (the trait is shared behind an `Arc`).
    system_prompt: RwLock<String>,
    /// Cached liveness, so `available()` doesn't issue an HTTP call per turn.
    reachable: std::sync::OnceLock<bool>,
}

impl OpenAiChatBackend {
    /// Build a backing from an endpoint config plus the akh's optional explicit
    /// system prompt.
    pub fn new(config: OpenAiConfig, explicit_prompt: Option<String>) -> Self {
        // Until `prepare` runs (first turn), fall back to the explicit prompt so
        // even a psyche-less akh sends something sensible.
        let initial = explicit_prompt.clone().unwrap_or_default();
        Self {
            client: OpenAiClient::new(config),
            explicit_prompt,
            system_prompt: RwLock::new(initial),
            reachable: std::sync::OnceLock::new(),
        }
    }

    fn prompt(&self) -> String {
        self.system_prompt
            .read()
            .map(|p| p.clone())
            .unwrap_or_default()
    }

    fn refresh_prompt(
        &self,
        agent_id: &str,
        psyche: Option<&crate::compartment::psyche::Psyche>,
        manifests: &[&crate::agent::tool_manifest::ToolManifest],
    ) -> String {
        let assembled = super::model_manager::assemble_system_prompt(
            agent_id,
            agent_id,
            psyche,
            self.explicit_prompt.as_deref(),
            super::model_manager::detect_prompt_mode(self.client.model()),
            manifests,
        );
        if let Ok(mut slot) = self.system_prompt.write() {
            *slot = assembled.clone();
        }
        assembled
    }
}

impl ChatBackend for OpenAiChatBackend {
    fn label(&self) -> &str {
        "openai"
    }

    fn available(&self) -> bool {
        // Probed once per process: a remote endpoint that answered `/models` at
        // startup is treated as up, and a genuine mid-session outage surfaces as
        // a `TurnFailed` (which the loop already falls back on) rather than
        // costing every turn an extra round-trip.
        *self
            .reachable
            .get_or_init(|| match self.client.list_models() {
                Ok(models) => {
                    let model = self.client.model();
                    if !models.is_empty() && !models.iter().any(|m| m == model) {
                        tracing::warn!(
                            model, advertised = ?models,
                            "openai endpoint does not advertise this akh's model; sending it anyway"
                        );
                    }
                    true
                }
                Err(e) => {
                    tracing::warn!(error = %e, "openai endpoint probe failed; backing unavailable");
                    false
                }
            })
    }

    fn prepare(
        &self,
        agent_id: &str,
        psyche: Option<&crate::compartment::psyche::Psyche>,
        manifests: &[&crate::agent::tool_manifest::ToolManifest],
    ) {
        self.refresh_prompt(agent_id, psyche, manifests);
    }

    fn prepare_turn(
        &self,
        agent_id: &str,
        psyche: Option<&crate::compartment::psyche::Psyche>,
        manifests: &[&crate::agent::tool_manifest::ToolManifest],
    ) -> PreparedChat {
        PreparedChat::with_system_prompt(self.refresh_prompt(agent_id, psyche, manifests))
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            // Not discoverable over the OpenAI contract — configured, not probed.
            context_window: self.client.config.context_window,
            supports_thinking: self.client.config.reasoning_effort.is_some(),
            supports_tools: true,
            supports_images: false,
        }
    }

    fn chat_with_tools(
        &self,
        _agent_id: &str,
        conversation: &[Value],
        tools: &[Value],
        request_seq: usize,
    ) -> Result<AssistantTurn, ChatBackendError> {
        self.client
            .chat(&self.prompt(), conversation, tools, request_seq)
            .map_err(|e| ChatBackendError::from_provider("openai", &e, e.provider_evidence()))
    }

    fn chat_with_tools_prepared(
        &self,
        prepared: &PreparedChat,
        _agent_id: &str,
        conversation: &[Value],
        tools: &[Value],
        request_seq: usize,
    ) -> Result<AssistantTurn, ChatBackendError> {
        self.client
            .chat(
                prepared.require_system_prompt("openai")?,
                conversation,
                tools,
                request_seq,
            )
            .map_err(|e| ChatBackendError::from_provider("openai", &e, e.provider_evidence()))
    }

    fn chat_with_tools_streaming(
        &self,
        _agent_id: &str,
        conversation: &[Value],
        tools: &[Value],
        request_seq: usize,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<AssistantTurn, ChatBackendError> {
        self.client
            .chat_stream(&self.prompt(), conversation, tools, request_seq, on_event)
            .map_err(|e| ChatBackendError::from_provider("openai", &e, e.provider_evidence()))
    }

    fn chat_with_tools_streaming_prepared(
        &self,
        prepared: &PreparedChat,
        _agent_id: &str,
        conversation: &[Value],
        tools: &[Value],
        request_seq: usize,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<AssistantTurn, ChatBackendError> {
        self.client
            .chat_stream(
                prepared.require_system_prompt("openai")?,
                conversation,
                tools,
                request_seq,
                on_event,
            )
            .map_err(|e| ChatBackendError::from_provider("openai", &e, e.provider_evidence()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> OpenAiConfig {
        OpenAiConfig {
            model: "DeepSeek-V4-Flash-0731".into(),
            ..Default::default()
        }
    }

    /// Phase 53g PR 4: an HTTP error keeps its message but also captures
    /// the status, `Retry-After` and structured `error.code` for the provider
    /// gate, and the backend boundary carries them as `ProviderFailed`.
    #[test]
    fn http_errors_capture_provider_evidence() {
        use crate::agent::provider_failure::{
            ProviderErrorDetail, ProviderFailureEvidence, RetryAfter,
        };
        let body = r#"{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}"#;
        let raw = format!(
            "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 30\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response: ureq::Response = raw.parse().unwrap();
        let client = OpenAiClient::new(cfg());
        let error = client.map_transport(ureq::Error::Status(429, response), "http://h/v1");
        assert_eq!(
            error.to_string(),
            "OpenAI-compatible request to http://h/v1 failed with HTTP 429: \
             You exceeded your current quota",
            "the display message now surfaces the provider's error.message"
        );
        let expected = ProviderFailureEvidence::Http {
            status: 429,
            detail: ProviderErrorDetail {
                retry_after: Some(RetryAfter::Delay(std::time::Duration::from_secs(30))),
                error_code: Some("insufficient_quota".into()),
                error_type: Some("insufficient_quota".into()),
            },
        };
        assert_eq!(error.provider_evidence(), Some(expected.clone()));

        let narrowed = ChatBackendError::from_provider("openai", &error, error.provider_evidence());
        assert_eq!(narrowed.provider_evidence(), Some(&expected));
        assert_eq!(
            narrowed.to_string(),
            format!("chat backend \"openai\" turn failed: {error}")
        );
        // Failures on the akh's side are not provider evidence.
        assert_eq!(
            OpenAiError::NoApiKey { url: "u".into() }.provider_evidence(),
            None
        );
        assert_eq!(
            OpenAiError::Unreachable {
                url: "u".into(),
                reason: "connection refused".into()
            }
            .provider_evidence(),
            Some(ProviderFailureEvidence::Transport)
        );
    }

    /// Non-JSON bodies (a plain-text 400 from a non-strict server) still
    /// surface, verbatim, rather than being swallowed to nothing.
    #[test]
    fn error_message_falls_back_to_raw_body_when_not_json() {
        assert_eq!(
            extract_error_message("Bad Request: role order invalid", None),
            Some("Bad Request: role order invalid".to_owned())
        );
        assert_eq!(extract_error_message("", None), None);
        assert_eq!(extract_error_message("   ", None), None);
    }

    /// The OpenAI error shape is unwrapped to just `error.message`, not the
    /// whole JSON body (which would duplicate `type`/`code` already carried
    /// structurally in `ProviderErrorDetail`).
    #[test]
    fn error_message_extracts_error_message_from_json() {
        let body = r#"{"error":{"message":"model does not support tools","type":"invalid_request_error"}}"#;
        assert_eq!(
            extract_error_message(body, None),
            Some("model does not support tools".to_owned())
        );
    }

    /// A JSON body with no `error.message` (e.g. a bare `{"detail": "..."}`
    /// from some other server) falls back to the raw body rather than `None`.
    #[test]
    fn error_message_falls_back_when_json_has_no_error_message() {
        let body = r#"{"detail":"invalid role sequence"}"#;
        assert_eq!(extract_error_message(body, None), Some(body.to_owned()));
    }

    #[test]
    fn error_message_is_truncated_around_600_chars() {
        let long = "x".repeat(1000);
        let body = format!(r#"{{"error":{{"message":"{long}"}}}}"#);
        let extracted = extract_error_message(&body, None).unwrap();
        assert_eq!(extracted.chars().count(), ERROR_MESSAGE_MAX_CHARS + 1); // +1 for the ellipsis
        assert!(extracted.ends_with('…'));
    }

    /// Defense in depth: a body that echoes `Authorization: Bearer ...` (a
    /// misbehaving proxy reflecting the request) never reaches the log.
    #[test]
    fn error_message_redacts_bearer_tokens_and_the_configured_key() {
        let body = r#"{"error":{"message":"rejected request with header Authorization: Bearer sk-super-secret-token-value"}}"#;
        let extracted = extract_error_message(body, None).unwrap();
        assert!(
            !extracted.contains("sk-super-secret-token-value"),
            "{extracted}"
        );
        assert!(extracted.contains("Bearer [REDACTED]"), "{extracted}");

        // Even without the "Bearer " prefix, a verbatim echo of the resolved
        // key is redacted when the caller has it to check against.
        let body2 = r#"{"error":{"message":"key literal-secret-123 is not authorized"}}"#;
        let extracted2 = extract_error_message(body2, Some("literal-secret-123")).unwrap();
        assert!(!extracted2.contains("literal-secret-123"), "{extracted2}");
        assert_eq!(extracted2, "key [REDACTED] is not authorized");
    }

    /// The redacted, truncated message is what both `Display` (and therefore
    /// the WARN log at the agentic-loop call site) and the miette diagnostic
    /// show — never headers, never the key, always bounded in length.
    #[test]
    fn request_failed_display_includes_the_redacted_message() {
        let body = r#"{"error":{"message":"invalid role sequence: user must follow tool"}}"#;
        let raw = format!(
            "HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response: ureq::Response = raw.parse().unwrap();
        let client = OpenAiClient::new(cfg());
        let error = client.map_transport(ureq::Error::Status(400, response), "http://h/v1");
        assert_eq!(
            error.to_string(),
            "OpenAI-compatible request to http://h/v1 failed with HTTP 400: \
             invalid role sequence: user must follow tool"
        );
        let report = miette::Report::new(error);
        assert!(
            report.to_string().contains("invalid role sequence"),
            "{report}"
        );
    }

    #[test]
    fn url_joins_with_exactly_one_slash() {
        let mut c = cfg();
        c.base_url = "https://inference.hetzner.com/api/v1/".into();
        assert_eq!(
            c.url("chat/completions"),
            "https://inference.hetzner.com/api/v1/chat/completions"
        );
        c.base_url = "https://inference.hetzner.com/api/v1".into();
        assert_eq!(
            c.url("chat/completions"),
            "https://inference.hetzner.com/api/v1/chat/completions"
        );
    }

    #[test]
    fn api_key_expands_env_reference() {
        const VAR: &str = "AKH_TEST_OPENAI_KEY_5521";
        // SAFETY: edition-2024 requires explicit unsafe for env mutation; this is
        // a process-local test with a unique variable name.
        unsafe { std::env::set_var(VAR, "resolved-secret") };
        let c = OpenAiConfig {
            api_key: Some(format!("env:{VAR}")),
            ..cfg()
        };
        assert_eq!(c.resolved_api_key().as_deref(), Some("resolved-secret"));
        unsafe { std::env::remove_var(VAR) };
    }

    #[test]
    fn unresolved_env_reference_never_leaks_as_a_token() {
        let c = OpenAiConfig {
            api_key: Some("env:AKH_TEST_DEFINITELY_UNSET_7741".into()),
            ..cfg()
        };
        assert_eq!(c.resolved_api_key(), None);
    }

    #[test]
    fn assistant_tool_calls_gain_ids_and_stringified_arguments() {
        let conv = vec![json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "shell_exec", "arguments": {"cmd": "ls"}}}],
        })];
        let out = to_openai_messages("sys", &conv);
        let asst = &out[1];
        let tc = &asst["tool_calls"][0];
        assert_eq!(tc["id"], "call_0_0");
        assert_eq!(tc["type"], "function");
        assert_eq!(tc["function"]["name"], "shell_exec");
        // Arguments must be a JSON *string*, not an object.
        assert_eq!(tc["function"]["arguments"], json!(r#"{"cmd":"ls"}"#));
        // A tool-only turn sends no empty content field.
        assert!(asst.get("content").is_none());
    }

    #[test]
    fn tool_results_are_paired_with_their_call_ids() {
        let conv = vec![
            json!({"role": "user", "content": "list files"}),
            json!({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "a", "arguments": {}}},
                    {"function": {"name": "b", "arguments": {}}},
                ],
            }),
            json!({"role": "tool", "content": "out-a"}),
            json!({"role": "tool", "content": "out-b"}),
        ];
        let out = to_openai_messages("sys", &conv);
        assert_eq!(out[3]["tool_call_id"], "call_0_0");
        assert_eq!(out[3]["content"], "out-a");
        assert_eq!(out[4]["tool_call_id"], "call_0_1");
        assert_eq!(out[4]["content"], "out-b");
    }

    #[test]
    fn ids_stay_unique_across_rounds() {
        let call = |n: &str| json!({"function": {"name": n, "arguments": {}}});
        let conv = vec![
            json!({"role": "assistant", "tool_calls": [call("a")]}),
            json!({"role": "tool", "content": "1"}),
            json!({"role": "assistant", "tool_calls": [call("b")]}),
            json!({"role": "tool", "content": "2"}),
        ];
        let out = to_openai_messages("", &conv);
        assert_eq!(out[0]["tool_calls"][0]["id"], "call_0_0");
        assert_eq!(out[1]["tool_call_id"], "call_0_0");
        assert_eq!(out[2]["tool_calls"][0]["id"], "call_1_0");
        assert_eq!(out[3]["tool_call_id"], "call_1_0");
    }

    #[test]
    fn plain_assistant_turn_keeps_its_content() {
        let conv = vec![json!({"role": "assistant", "content": "done", "tool_calls": []})];
        let out = to_openai_messages("", &conv);
        assert_eq!(out[0]["content"], "done");
        assert!(out[0].get("tool_calls").is_none());
    }

    // ── Lilith / Gemma-4 strict-template hardening (ADR 116, Anima WI
    // 01a0e270-314e) ──────────────────────────────────────────────────────
    //
    // Both shapes below were proven to crash the *pinned* Gemma 4 chat
    // template (`chat_template.jinja` on the Lilith serve pod), rendered
    // offline via `AutoTokenizer.apply_chat_template` with the tokenizer's
    // own Jinja engine — not a guess about what a strict template *might*
    // reject:
    //
    //   * a `role:"tool"` message whose `content` is a JSON object (not a
    //     string) → `jinja2.exceptions.UndefinedError: 'str object' has no
    //     attribute 'get'` while the template forward-scans the response
    //     block.
    //   * a `role:"tool"` message with no `tool_call_id` (the "malformed
    //     history" fallback this module used to emit deliberately) →
    //     `TypeError: can only concatenate str (not "NoneType") to str`
    //     while the template formats the response block for a call id it
    //     can't resolve.
    //
    // Both are fast, synchronous template-render failures — exactly the
    // "HTTP 400 in ~0.1s, no generation" shape from the live incident — so
    // `to_openai_messages` must never produce either shape, regardless of
    // which upstream code path (a future tool, a compacted/replayed history
    // entry, or this module's own old fallback) would otherwise cause it.

    #[test]
    fn tool_content_is_always_stringified() {
        let conv = vec![
            json!({"role": "assistant", "tool_calls": [
                {"function": {"name": "list_plans", "arguments": {}}},
            ]}),
            // A structured (non-string) tool result, as a future tool or a
            // replayed history entry could produce.
            json!({"role": "tool", "content": {"plans": [{"id": "pl1"}]}}),
        ];
        let out = to_openai_messages("", &conv);
        let content = out[1]["content"]
            .as_str()
            .expect("tool content must always serialize to a JSON string");
        assert_eq!(content, r#"{"plans":[{"id":"pl1"}]}"#);
        assert_eq!(out[1]["tool_call_id"], "call_0_0");
    }

    #[test]
    fn tool_content_missing_entirely_becomes_an_empty_string() {
        let conv = vec![
            json!({"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": {}}}]}),
            json!({"role": "tool"}),
        ];
        let out = to_openai_messages("", &conv);
        assert_eq!(out[1]["content"], "");
        assert_eq!(out[1]["tool_call_id"], "call_0_0");
    }

    #[test]
    fn orphaned_tool_result_is_folded_into_a_user_turn_not_sent_malformed() {
        // No preceding assistant tool_calls at all: pending_ids is empty
        // from the start. This is the exact "malformed/compacted history"
        // shape the old fallback used to pass through as `role:"tool"` with
        // no `tool_call_id` — which the pinned template cannot render.
        let conv = vec![json!({"role": "tool", "content": "orphaned result"})];
        let out = to_openai_messages("", &conv);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["role"], "user");
        assert!(out[0].get("tool_call_id").is_none());
        assert!(
            out[0]["content"]
                .as_str()
                .unwrap()
                .contains("orphaned result"),
            "the observation must survive the fold: {out:?}"
        );
        // Never the crashing shape: role:"tool" with no tool_call_id.
        assert!(
            !(out[0]["role"] == "tool" && out[0].get("tool_call_id").is_none()),
            "must never emit a tool message without a tool_call_id"
        );
    }

    #[test]
    fn orphaned_tool_result_merges_with_an_adjacent_user_turn() {
        // A second tool message beyond what the preceding assistant round
        // declared: the first consumes the only pending id, the second is
        // orphaned and must fold *and* merge into the neighboring user turn
        // rather than appear as a second, separate user message (which
        // would violate strict user/assistant alternation too).
        let conv = vec![
            json!({"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": {}}}]}),
            json!({"role": "tool", "content": "first"}),
            json!({"role": "tool", "content": "second (orphaned)"}),
            json!({"role": "user", "content": "please continue"}),
        ];
        let out = to_openai_messages("", &conv);
        // assistant, tool(first), then ONE merged user turn (folded orphan
        // + the real next user message) — never two adjacent user messages.
        assert_eq!(out.len(), 3, "{out:?}");
        assert_eq!(out[1]["role"], "tool");
        assert_eq!(out[1]["tool_call_id"], "call_0_0");
        assert_eq!(out[2]["role"], "user");
        let merged = out[2]["content"].as_str().unwrap();
        assert!(merged.contains("second (orphaned)"), "{merged}");
        assert!(merged.contains("please continue"), "{merged}");
    }

    #[test]
    fn adjacent_assistant_content_turns_are_merged() {
        // force_final_answer / the no-work-retry path can append a plain
        // `role:"user"` turn; symmetrically, two consecutive content-only
        // assistant turns (no tool_calls on either) are merged too, since
        // some strict templates expect alternation there as well.
        let conv = vec![
            json!({"role": "assistant", "content": "partial answer"}),
            json!({"role": "assistant", "content": "more detail"}),
        ];
        let out = to_openai_messages("", &conv);
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0]["content"], "partial answer\n\nmore detail");
    }

    #[test]
    fn merge_never_touches_tool_messages_or_assistant_tool_call_turns() {
        // Two adjacent tool results for two DIFFERENT calls must stay
        // separate (each keyed by its own tool_call_id) even though they
        // share a role — merging would silently drop one observation.
        let conv = vec![
            json!({"role": "assistant", "tool_calls": [
                {"function": {"name": "a", "arguments": {}}},
                {"function": {"name": "b", "arguments": {}}},
            ]}),
            json!({"role": "tool", "content": "out-a"}),
            json!({"role": "tool", "content": "out-b"}),
        ];
        let out = to_openai_messages("", &conv);
        assert_eq!(out.len(), 3, "{out:?}");
        assert_eq!(out[1]["content"], "out-a");
        assert_eq!(out[2]["content"], "out-b");
        assert_ne!(out[1]["tool_call_id"], out[2]["tool_call_id"]);
    }

    #[test]
    fn system_prompt_is_prepended_only_when_present() {
        let conv = vec![json!({"role": "user", "content": "hi"})];
        assert_eq!(to_openai_messages("sys", &conv).len(), 2);
        assert_eq!(to_openai_messages("", &conv).len(), 1);
    }

    #[test]
    fn conversation_system_messages_are_coalesced_at_the_front() {
        let conv = vec![
            json!({"role": "user", "content": "first"}),
            json!({"role": "system", "content": "context update"}),
            json!({"role": "assistant", "content": "noted"}),
            json!({"role": "system", "content": "memory update"}),
            json!({"role": "user", "content": "continue"}),
        ];

        let out = to_openai_messages("base instructions", &conv);

        assert_eq!(out.len(), 4);
        assert_eq!(out[0]["role"], "system");
        assert_eq!(
            out[0]["content"],
            "base instructions\n\ncontext update\n\nmemory update"
        );
        assert_eq!(out[1]["content"], "first");
        assert_eq!(out[2]["content"], "noted");
        assert_eq!(out[3]["content"], "continue");
        assert_eq!(out.iter().filter(|msg| msg["role"] == "system").count(), 1);
    }

    #[test]
    fn conversation_system_message_becomes_leading_without_base_prompt() {
        let conv = vec![
            json!({"role": "user", "content": "first"}),
            json!({"role": "system", "content": "late instructions"}),
        ];

        let out = to_openai_messages("", &conv);

        assert_eq!(
            out[0],
            json!({"role": "system", "content": "late instructions"})
        );
        assert_eq!(out[1], json!({"role": "user", "content": "first"}));
    }

    #[test]
    fn response_arguments_are_parsed_back_into_objects() {
        let msg = json!({
            "content": "",
            "tool_calls": [{
                "id": "call_x",
                "type": "function",
                "function": {"name": "grep", "arguments": "{\"pattern\":\"fn main\"}"},
            }],
        });
        let (content, calls) = parse_message(&msg);
        assert!(content.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "grep");
        assert_eq!(calls[0].arguments["pattern"], "fn main");
    }

    #[test]
    fn malformed_arguments_degrade_to_empty_object_not_a_dropped_call() {
        let msg = json!({
            "tool_calls": [{"function": {"name": "grep", "arguments": "{oops"}}],
        });
        let (_c, calls) = parse_message(&msg);
        assert_eq!(
            calls.len(),
            1,
            "the call must survive for the tool to reject"
        );
        assert_eq!(calls[0].arguments, json!({}));
    }

    /// A non-streaming response from a server with `--reasoning-parser`
    /// enabled (Lilith's pod, Anima WI 01a0e270-314e) carries the separated
    /// reasoning text in a sibling `reasoning_content` field. It must never
    /// leak into, or be concatenated with, the returned content — even
    /// though `AssistantTurn` currently has nowhere else to put it (unlike
    /// the streaming path's `StreamEvent::Thinking`).
    #[test]
    fn non_streaming_reasoning_content_never_contaminates_content() {
        let msg = json!({
            "content": "The answer is 4.",
            "reasoning_content": "The user asked 2+2, so I should compute 2+2=4.",
        });
        let (content, calls) = parse_message(&msg);
        assert_eq!(content, "The answer is 4.");
        assert!(calls.is_empty());
    }

    #[test]
    fn usage_defaults_to_zero_when_absent() {
        assert_eq!(usage_of(&json!({})), (0, 0));
        assert_eq!(
            usage_of(&json!({"usage": {"prompt_tokens": 12, "completion_tokens": 7}})),
            (12, 7)
        );
    }
}
