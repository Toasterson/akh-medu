//! Ollama REST API client for local Gemma inference.
//!
//! Provides synchronous HTTP access to an Ollama server using the `/api/chat`
//! endpoint with non-streaming response mode, matching the rest of the codebase's
//! `ureq`-based sync HTTP pattern.

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for the Ollama client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// Ollama server base URL (default: "http://localhost:11434").
    #[serde(default = "default_base_url")]
    pub base_url: String,
    /// Model to load, e.g. "gemma4:e4b" (default: "gemma4:e4b").
    #[serde(default = "default_model")]
    pub model: String,
    /// Request timeout in seconds (default: 120).
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    /// Ollama keep-alive duration after a request (default: "5m").
    /// Set to "0" to unload the model immediately after the request.
    #[serde(default = "default_keep_alive")]
    pub keep_alive: String,
    /// Optional API key for cloud endpoints (e.g. `ollama.com`). When present,
    /// requests carry an `Authorization: Bearer <key>` header. Supports the
    /// `env:VAR` indirection — `"env:OLLAMA_API_KEY"` reads the token from the
    /// named environment variable at call time. If that variable is unset (or
    /// empty), [`OllamaConfig::resolved_api_key`] logs a warning and sends **no**
    /// auth header rather than a malformed `Bearer env:VAR` token. A local
    /// endpoint leaves this `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    /// Reasoning-effort hint for the agentic `chat` path. When `Some(level)`
    /// (one of `"low"`/`"medium"`/`"high"`), [`OllamaClient::chat`] sets
    /// `think: "<level>"` in the request body — accepted by Ollama's reasoning
    /// cloud models (gpt-oss/glm). `None` (the default) omits the field, leaving
    /// the model's own default in effect. The plain [`OllamaClient::generate`]
    /// path is intentionally unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: default_base_url(),
            model: default_model(),
            timeout_secs: default_timeout(),
            keep_alive: default_keep_alive(),
            api_key: None,
            reasoning_effort: None,
        }
    }
}

fn default_base_url() -> String { "http://localhost:11434".into() }
fn default_model() -> String { "gemma4:e4b".into() }
fn default_timeout() -> u64 { 120 }
fn default_keep_alive() -> String { "5m".into() }

impl OllamaConfig {
    /// Resolve the API key, expanding an `env:VAR` reference at call time.
    ///
    /// - `None` (local endpoint, no key) → `None`.
    /// - `"env:VAR"` with `VAR` set to a non-empty value → `Some(value)`.
    /// - `"env:VAR"` with `VAR` unset or empty → logs a warning and returns
    ///   `None`. Returning the literal `"env:VAR"` as a bearer token would only
    ///   produce a confusing `401`, so we omit the header and surface the
    ///   misconfiguration in the logs instead.
    /// - any other value → used verbatim as the bearer token.
    pub fn resolved_api_key(&self) -> Option<String> {
        let raw = self.api_key.as_ref()?;
        match raw.strip_prefix("env:") {
            Some(var) => match std::env::var(var) {
                Ok(val) if !val.is_empty() => Some(val),
                _ => {
                    tracing::warn!(
                        env_var = var,
                        "Ollama api_key references env:{var} but it is unset/empty; \
                         sending no Authorization header"
                    );
                    None
                }
            },
            None => Some(raw.clone()),
        }
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors from the Ollama client.
#[derive(Debug, Error, Diagnostic)]
pub enum OllamaError {
    #[error("Ollama server unreachable at {url}: {reason}")]
    #[diagnostic(
        code(akh::ollama::unreachable),
        help("Ensure Ollama is running: `ollama serve`")
    )]
    Unreachable { url: String, reason: String },

    #[error("Ollama request failed (HTTP {status}) at {url}: {body}")]
    #[diagnostic(
        code(akh::ollama::request_failed),
        help("Check Ollama server logs. Ensure the model is pulled: `ollama pull {model}`"),
    )]
    RequestFailed { url: String, status: u16, body: String, model: String },

    #[error("Ollama returned unexpected response: {reason}")]
    #[diagnostic(
        code(akh::ollama::parse_failed),
        help("The Ollama /api/chat endpoint returned JSON without message.content.")
    )]
    ParseFailed { reason: String },

    #[error("Ollama request timed out after {timeout_secs}s")]
    #[diagnostic(
        code(akh::ollama::timeout),
        help(
            "Increase timeout_secs in OllamaConfig, or configure a faster model, \
             or use the remote LLM fallback."
        )
    )]
    Timeout { timeout_secs: u64 },
}

/// Convenience alias.
pub type OllamaResult<T> = std::result::Result<T, OllamaError>;

// ---------------------------------------------------------------------------
// Response shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ChatResponse {
    message: Option<ChatMessage>,
    /// Prompt (input) tokens evaluated — Ollama reports this on the final
    /// response. Absent ⇒ 0.
    #[serde(default)]
    prompt_eval_count: u64,
    /// Completion (output) tokens generated. Absent ⇒ 0.
    #[serde(default)]
    eval_count: u64,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    content: Option<String>,
    /// Tool calls the model wants executed (Ollama `/api/chat` tool calling).
    /// Absent on plain text turns — defaults to empty.
    #[serde(default)]
    tool_calls: Vec<RawToolCall>,
}

#[derive(Debug, Deserialize)]
struct RawToolCall {
    function: RawToolFunction,
}

#[derive(Debug, Deserialize)]
struct RawToolFunction {
    name: String,
    /// Ollama returns arguments as a JSON object (not a stringified blob).
    #[serde(default)]
    arguments: serde_json::Value,
}

/// One tool call the model requested in a [`OllamaClient::chat`] turn.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Tool name (must match a registered tool).
    pub name: String,
    /// Arguments object the model supplied.
    pub arguments: serde_json::Value,
}

/// The assistant's reply to one tool-enabled chat turn.
#[derive(Debug, Clone, Default)]
pub struct AssistantTurn {
    /// Natural-language content. May be empty when the model only calls tools.
    pub content: String,
    /// Tool calls to execute before the model can answer. Empty ⇒ final answer.
    pub tool_calls: Vec<ToolCall>,
    /// Prompt (input) tokens billed for this turn (Ollama `prompt_eval_count`);
    /// 0 when the backend did not report it.
    pub prompt_tokens: u64,
    /// Completion (output) tokens billed for this turn (Ollama `eval_count`).
    pub completion_tokens: u64,
}

/// One incremental piece of a streaming chat turn, delivered to the
/// [`OllamaClient::chat_stream`] callback as the model generates.
///
/// Reasoning ("thinking") models split their output into two channels: the
/// user-facing `content` (the answer) and a separate `thinking` stream (the
/// chain-of-thought, shown when `think` is enabled). We surface them as distinct
/// events so the UI can render the answer prominently and fold the thinking into
/// a collapsible "thinking" section.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A delta of user-facing answer text (`message.content`).
    Content(String),
    /// A delta of chain-of-thought text (`message.thinking`), emitted only by
    /// reasoning models with `think` enabled.
    Thinking(String),
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Synchronous Ollama REST client.
pub struct OllamaClient {
    config: OllamaConfig,
}

impl OllamaClient {
    /// Create a new client with the given config.
    pub fn new(config: OllamaConfig) -> Self {
        Self { config }
    }

    /// Returns `true` if the Ollama server is reachable.
    ///
    /// Uses a short 5-second timeout so callers are not blocked long
    /// when Ollama is not running.
    pub fn health_check(&self) -> bool {
        let url = format!("{}/api/tags", self.config.base_url.trim_end_matches('/'));
        let mut req = ureq::get(&url).timeout(std::time::Duration::from_secs(5));
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        req.call().is_ok()
    }

    /// List the model names the endpoint advertises via `GET /api/tags`.
    ///
    /// Carries the same `Authorization: Bearer` header as [`health_check`] /
    /// [`chat`] for cloud endpoints. Parses `{"models":[{"name":"..."},...]}`
    /// and returns the names. A cloud endpoint (e.g. `ollama.com`) may return an
    /// empty `models` array; that is reported as an empty `Vec`, not an error.
    /// Transport/HTTP/parse failures are returned as [`OllamaError`] so the
    /// caller can degrade gracefully (the picker still opens; the user types a
    /// name).
    ///
    /// [`health_check`]: Self::health_check
    /// [`chat`]: Self::chat
    pub fn list_models(&self) -> OllamaResult<Vec<String>> {
        let url = format!("{}/api/tags", self.config.base_url.trim_end_matches('/'));
        let mut req = ureq::get(&url).timeout(std::time::Duration::from_secs(self.config.timeout_secs));
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let response = req.call().map_err(|e| match e {
            ureq::Error::Status(status, resp) => OllamaError::RequestFailed {
                url: url.clone(),
                status,
                body: resp.into_string().unwrap_or_default(),
                model: self.config.model.clone(),
            },
            other => OllamaError::Unreachable {
                url: url.clone(),
                reason: other.to_string(),
            },
        })?;

        let body: serde_json::Value = response.into_json().map_err(|e| OllamaError::ParseFailed {
            reason: format!("JSON decode failed: {e}"),
        })?;

        let names = body
            .get("models")
            .and_then(serde_json::Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("name").and_then(serde_json::Value::as_str))
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(names)
    }

    /// Generate a chat response from the model.
    ///
    /// Sends a non-streaming `/api/chat` request with a `system` turn followed
    /// by the `user` turn. Returns the assistant's reply as a `String`.
    ///
    /// Error variants callers can handle:
    /// - `Unreachable` — Ollama is not running; fall back to remote API.
    /// - `RequestFailed` — model not found or server error; check model name.
    /// - `Timeout` — model too slow; reduce prompt or increase `timeout_secs`.
    pub fn generate(&self, system_prompt: &str, user_prompt: &str) -> OllamaResult<String> {
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "stream": false,
            "keep_alive": self.config.keep_alive,
        });

        let mut req = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs));
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let response = req
            .send_json(&body)
            .map_err(|e| match e {
                ureq::Error::Status(status, resp) => OllamaError::RequestFailed {
                    url: url.clone(),
                    status,
                    body: resp.into_string().unwrap_or_default(),
                    model: self.config.model.clone(),
                },
                other => OllamaError::Unreachable {
                    url: url.clone(),
                    reason: other.to_string(),
                },
            })?;

        let chat_resp: ChatResponse = response.into_json().map_err(|e| OllamaError::ParseFailed {
            reason: format!("JSON decode failed: {e}"),
        })?;

        chat_resp
            .message
            .and_then(|m| m.content)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| OllamaError::ParseFailed {
                reason: "message.content missing or empty in Ollama response".into(),
            })
    }

    /// Run one tool-enabled `/api/chat` turn.
    ///
    /// `messages` is the full conversation so far — each a JSON object with at
    /// least `role` + `content`; assistant turns may carry `tool_calls` and tool
    /// results use `{"role":"tool","content":...}`. `tools` is the function
    /// catalog (Ollama/OpenAI shape: `{"type":"function","function":{name,
    /// description,parameters}}`); pass an empty slice to disable tool calling.
    ///
    /// Returns the assistant's text plus any tool calls it wants executed. An
    /// empty `tool_calls` means the turn is a final answer. Unlike [`generate`],
    /// an empty `content` is **not** an error — the model legitimately returns no
    /// prose when it only wants to call tools.
    pub fn chat(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
    ) -> OllamaResult<AssistantTurn> {
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": false,
            "keep_alive": self.config.keep_alive,
        });
        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(tools.to_vec());
        }
        // Reasoning-effort hint (chat path only): Ollama's reasoning cloud models
        // (gpt-oss/glm) accept `think: "low"|"medium"|"high"`. Absent ⇒ model default.
        if let Some(level) = &self.config.reasoning_effort {
            body["think"] = serde_json::Value::String(level.clone());
        }

        let mut req = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs));
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let response = req.send_json(&body).map_err(|e| match e {
            ureq::Error::Status(status, resp) => OllamaError::RequestFailed {
                url: url.clone(),
                status,
                body: resp.into_string().unwrap_or_default(),
                model: self.config.model.clone(),
            },
            other => OllamaError::Unreachable {
                url: url.clone(),
                reason: other.to_string(),
            },
        })?;

        let chat_resp: ChatResponse =
            response.into_json().map_err(|e| OllamaError::ParseFailed {
                reason: format!("JSON decode failed: {e}"),
            })?;

        let msg = chat_resp.message.ok_or_else(|| OllamaError::ParseFailed {
            reason: "Ollama chat response had no message".into(),
        })?;

        Ok(AssistantTurn {
            content: msg.content.unwrap_or_default(),
            tool_calls: msg
                .tool_calls
                .into_iter()
                .map(|tc| ToolCall {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                })
                .collect(),
            prompt_tokens: chat_resp.prompt_eval_count,
            completion_tokens: chat_resp.eval_count,
        })
    }

    /// Run one tool-enabled `/api/chat` turn with **streaming** enabled.
    ///
    /// Identical to [`chat`](Self::chat) in request shape (same `messages`,
    /// `tools`, and `think` handling) but sets `"stream": true`, so the server
    /// responds with newline-delimited JSON: one object per token-ish chunk.
    /// Each chunk's `message.content` / `message.thinking` delta is delivered to
    /// `on_event` as it arrives ([`StreamEvent::Content`] / [`StreamEvent::Thinking`]),
    /// letting the caller render the answer live. Tool calls (`message.tool_calls`)
    /// are accumulated and returned in the final [`AssistantTurn`] — they are not
    /// streamed token-by-token by Ollama.
    ///
    /// The returned `AssistantTurn.content` is the concatenation of all content
    /// deltas (thinking is intentionally **not** folded in). On any transport or
    /// HTTP error the call returns an [`OllamaError`]; callers that want a
    /// non-streaming fallback should catch it and retry with [`chat`](Self::chat).
    /// Unparseable NDJSON lines are skipped defensively rather than aborting the
    /// stream.
    pub fn chat_stream(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> OllamaResult<AssistantTurn> {
        use std::io::BufRead;

        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));

        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": true,
            "keep_alive": self.config.keep_alive,
        });
        if !tools.is_empty() {
            body["tools"] = serde_json::Value::Array(tools.to_vec());
        }
        if let Some(level) = &self.config.reasoning_effort {
            body["think"] = serde_json::Value::String(level.clone());
        }

        let mut req = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs));
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let response = req.send_json(&body).map_err(|e| match e {
            ureq::Error::Status(status, resp) => OllamaError::RequestFailed {
                url: url.clone(),
                status,
                body: resp.into_string().unwrap_or_default(),
                model: self.config.model.clone(),
            },
            other => OllamaError::Unreachable {
                url: url.clone(),
                reason: other.to_string(),
            },
        })?;

        let mut content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut prompt_tokens = 0u64;
        let mut completion_tokens = 0u64;
        let reader = std::io::BufReader::new(response.into_reader());
        for line in reader.lines() {
            let line = line.map_err(|e| OllamaError::Unreachable {
                url: url.clone(),
                reason: format!("stream read error: {e}"),
            })?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            // Parse each NDJSON chunk leniently — a malformed line is skipped,
            // not fatal, so one odd frame can't abort an otherwise-good stream.
            let Ok(chunk) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if let Some(msg) = chunk.get("message") {
                if let Some(delta) = msg.get("content").and_then(serde_json::Value::as_str)
                    && !delta.is_empty()
                {
                    content.push_str(delta);
                    on_event(StreamEvent::Content(delta.to_string()));
                }
                if let Some(delta) = msg.get("thinking").and_then(serde_json::Value::as_str)
                    && !delta.is_empty()
                {
                    on_event(StreamEvent::Thinking(delta.to_string()));
                }
                if let Some(calls) = msg.get("tool_calls").and_then(serde_json::Value::as_array) {
                    for call in calls {
                        if let Some(func) = call.get("function")
                            && let Some(name) =
                                func.get("name").and_then(serde_json::Value::as_str)
                        {
                            tool_calls.push(ToolCall {
                                name: name.to_string(),
                                arguments: func
                                    .get("arguments")
                                    .cloned()
                                    .unwrap_or(serde_json::Value::Null),
                            });
                        }
                    }
                }
            }
            if chunk.get("done").and_then(serde_json::Value::as_bool) == Some(true) {
                // The final chunk carries the turn's token counts.
                prompt_tokens = chunk
                    .get("prompt_eval_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                completion_tokens = chunk
                    .get("eval_count")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                break;
            }
        }

        Ok(AssistantTurn {
            content,
            tool_calls,
            prompt_tokens,
            completion_tokens,
        })
    }

    /// Best-effort fetch of the model's context-window size via `/api/show`.
    ///
    /// Ollama reports it in `model_info` under an architecture-prefixed key (e.g.
    /// `llama.context_length`, `gemma.context_length`), so we scan for any key
    /// ending in `.context_length`. Returns `None` on any transport/parse error
    /// or if the field is absent — callers then omit the `% used` figure.
    pub fn context_length(&self) -> Option<u64> {
        let url = format!("{}/api/show", self.config.base_url.trim_end_matches('/'));
        // Send both `model` (current API) and `name` (older API) — extra keys are
        // ignored, so one body satisfies either server version.
        let body = serde_json::json!({
            "model": self.config.model,
            "name": self.config.model,
        });
        let mut req = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs));
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let resp = req.send_json(&body).ok()?;
        let value: serde_json::Value = resp.into_json().ok()?;
        let model_info = value.get("model_info")?.as_object()?;
        model_info
            .iter()
            .find(|(k, _)| k.ends_with(".context_length"))
            .and_then(|(_, v)| v.as_u64())
    }

    /// Hint Ollama to unload the model immediately by sending a request with
    /// `keep_alive: "0"`. This is best-effort — failure is logged, not returned.
    pub fn hint_unload(&self) {
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [{"role": "user", "content": ""}],
            "keep_alive": "0",
            "stream": false,
        });
        let mut req = ureq::post(&url)
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(5));
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        if let Err(e) = req.send_json(&body) {
            tracing::debug!("hint_unload: Ollama did not acknowledge unload hint: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// Tests (pure logic, no live Ollama required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// `chat_stream` parses Ollama's newline-delimited streaming response:
    /// content deltas concatenate into `AssistantTurn.content` and arrive as
    /// `Content` events in order, `thinking` deltas arrive as `Thinking` events
    /// (and are NOT folded into content), and `tool_calls` are collected.
    #[test]
    fn chat_stream_parses_ndjson_deltas_and_tool_calls() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        // NDJSON body: two content chunks, a thinking chunk, a tool call, done.
        let body = concat!(
            "{\"message\":{\"content\":\"Hel\"}}\n",
            "{\"message\":{\"content\":\"lo\"}}\n",
            "{\"message\":{\"thinking\":\"hmm\"}}\n",
            "{\"message\":{\"tool_calls\":[{\"function\":{\"name\":\"search\",\"arguments\":{\"q\":\"x\"}}}]}}\n",
            "{\"done\":true}\n",
        );
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 4096];
                let _ = stream.read(&mut buf); // drain request (best-effort)
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });

        let client = OllamaClient::new(OllamaConfig {
            base_url: format!("http://127.0.0.1:{port}"),
            ..OllamaConfig::default()
        });

        let mut events = Vec::new();
        let turn = client
            .chat_stream(
                &[serde_json::json!({"role": "user", "content": "hi"})],
                &[],
                &mut |ev| events.push(ev),
            )
            .expect("chat_stream ok");
        let _ = server.join();

        assert_eq!(turn.content, "Hello", "content deltas concatenate");
        assert_eq!(turn.tool_calls.len(), 1);
        assert_eq!(turn.tool_calls[0].name, "search");
        assert_eq!(turn.tool_calls[0].arguments["q"], "x");

        let content: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Content(s) => Some(s.as_str()),
                StreamEvent::Thinking(_) => None,
            })
            .collect();
        assert_eq!(content, ["Hel", "lo"], "content events arrive in order");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::Thinking(s) if s == "hmm")),
            "thinking delta surfaced as a Thinking event"
        );
    }

    #[test]
    fn default_config_values() {
        let cfg = OllamaConfig::default();
        assert_eq!(cfg.base_url, "http://localhost:11434");
        assert_eq!(cfg.model, "gemma4:e4b");
        assert_eq!(cfg.timeout_secs, 120);
        assert_eq!(cfg.keep_alive, "5m");
    }

    #[test]
    fn resolved_api_key_none_for_local() {
        let cfg = OllamaConfig::default();
        assert!(cfg.resolved_api_key().is_none());
    }

    #[test]
    fn resolved_api_key_used_verbatim_when_no_env_prefix() {
        let cfg = OllamaConfig {
            api_key: Some("sk-literal-token".into()),
            ..OllamaConfig::default()
        };
        assert_eq!(cfg.resolved_api_key().as_deref(), Some("sk-literal-token"));
    }

    #[test]
    fn resolved_api_key_expands_env_reference() {
        // Unique var name to avoid racing other tests that read the environment.
        const VAR: &str = "AKH_TEST_OLLAMA_KEY_8842";
        // SAFETY: edition-2024 requires explicit unsafe for env mutation; this is
        // a process-local test with a unique variable name.
        unsafe {
            std::env::set_var(VAR, "resolved-secret");
        }
        let cfg = OllamaConfig {
            api_key: Some(format!("env:{VAR}")),
            ..OllamaConfig::default()
        };
        assert_eq!(cfg.resolved_api_key().as_deref(), Some("resolved-secret"));
        unsafe {
            std::env::remove_var(VAR);
        }
    }

    #[test]
    fn resolved_api_key_none_when_env_unset() {
        // A unique, definitely-unset variable name.
        let cfg = OllamaConfig {
            api_key: Some("env:AKH_TEST_DEFINITELY_UNSET_3317".into()),
            ..OllamaConfig::default()
        };
        // Must NOT leak the literal "env:VAR" as a bearer token.
        assert_eq!(cfg.resolved_api_key(), None);
    }

    #[test]
    fn reasoning_effort_defaults_to_none_and_skips_serialization() {
        let cfg = OllamaConfig::default();
        assert!(cfg.reasoning_effort.is_none());
        // skip_serializing_if = Option::is_none ⇒ absent from the serialized form.
        let json = serde_json::to_string(&cfg).expect("serialize");
        assert!(
            !json.contains("reasoning_effort"),
            "None reasoning_effort must be skipped: {json}"
        );
    }

    #[test]
    fn reasoning_effort_round_trips_when_set() {
        let cfg = OllamaConfig {
            reasoning_effort: Some("high".into()),
            ..OllamaConfig::default()
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: OllamaConfig = serde_json::from_str(&json).expect("round-trip");
        assert_eq!(back.reasoning_effort.as_deref(), Some("high"));
    }

    #[test]
    fn list_models_returns_error_when_unreachable() {
        let cfg = OllamaConfig {
            base_url: "http://127.0.0.1:19999".into(),
            timeout_secs: 2,
            ..OllamaConfig::default()
        };
        let client = OllamaClient::new(cfg);
        // No live endpoint ⇒ the caller gets an error to handle gracefully,
        // never a panic.
        assert!(client.list_models().is_err());
    }

    #[test]
    fn health_check_returns_false_when_unreachable() {
        let cfg = OllamaConfig {
            base_url: "http://127.0.0.1:19999".into(), // nothing listening here
            ..OllamaConfig::default()
        };
        let client = OllamaClient::new(cfg);
        assert!(!client.health_check());
    }
}
