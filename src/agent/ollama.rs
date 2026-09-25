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
    /// (one of `"low"`/`"medium"`/`"high"`/`"max"` — Ollama's `/api/chat`
    /// `think` enum, see [`Self::validated_think_level`]),
    /// [`OllamaClient::chat`]/[`OllamaClient::chat_stream`] set `think:
    /// "<level>"` in the request body. Omitting `think` entirely leaves
    /// Ollama's own route-handler default (`Think = true`, which renders as
    /// `"medium"` — no reasoning-effort text injected) in effect; only
    /// `"low"` (and `"high"`/`"max"`, which the dedicated Qwen3.8 renderer
    /// both map to its "xhigh" instructions) actually change the rendered
    /// prompt, so this is a real latency/depth tradeoff, not just metadata.
    /// `None` (the default) omits the field. An unrecognized string is
    /// **rejected** (logged, field omitted) rather than sent verbatim — see
    /// [`Self::validated_think_level`]. The plain [`OllamaClient::generate`]
    /// path is intentionally unaffected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

/// The `think` values Ollama's `/api/chat` accepts as a string (it also
/// accepts a bare `bool`, which this codebase does not use — an unset
/// `reasoning_effort` already gets Ollama's own default via an omitted
/// field, so there's no case where sending an explicit `true`/`false` adds
/// anything). Kept as a slice (not an enum) because the value round-trips
/// straight from `AkhConfig`'s TOML `String` into the wire JSON `String`, and
/// an enum would just add a translation step both ends have to agree on.
///
/// `pub` so the ACP `model/set` RPC handler (`bin/akhomed.rs`) validates
/// against this exact list instead of a hand-maintained copy that could
/// drift from it.
pub const VALID_THINK_LEVELS: &[&str] = &["low", "medium", "high", "max"];

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

fn default_base_url() -> String {
    "http://localhost:11434".into()
}
fn default_model() -> String {
    "gemma4:e4b".into()
}
/// Per-request timeout. `AKH_OLLAMA_TIMEOUT_SECS` overrides the 120s default so
/// long reasoning / agentic-coding generations (which the model *does* answer,
/// just slowly) aren't cut off mid-response.
fn default_timeout() -> u64 {
    std::env::var("AKH_OLLAMA_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(120)
}
fn default_keep_alive() -> String {
    "5m".into()
}

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

    /// Validate [`Self::reasoning_effort`] against Ollama's `think` string
    /// enum (`"low"|"medium"|"high"|"max"`), case/whitespace-insensitively.
    ///
    /// Returns `None` both when unset and when set to something Ollama
    /// doesn't understand — in the latter case, after logging a warning, so
    /// a typo'd `reasoning_effort` in a workspace TOML degrades to "no
    /// think field sent" (today's behaviour, safe) rather than being handed
    /// to the model verbatim (an unrecognized `think` value that a future
    /// Ollama release might error on, or that another renderer silently
    /// mishandles). The ACP `model/set` RPC handler (`akhomed.rs`) already
    /// validates before it ever writes `reasoning_effort` to disk; this is
    /// the second gate, for the TOML-direct-edit path that handler doesn't
    /// cover.
    pub fn validated_think_level(&self) -> Option<&'static str> {
        let raw = self.reasoning_effort.as_deref()?.trim();
        match VALID_THINK_LEVELS
            .iter()
            .find(|&&v| v.eq_ignore_ascii_case(raw))
        {
            Some(&level) => Some(level),
            None => {
                tracing::warn!(
                    value = raw,
                    valid = ?VALID_THINK_LEVELS,
                    "reasoning_effort is not a recognized Ollama 'think' level; \
                     omitting the think field for this request"
                );
                None
            }
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
        help("Check Ollama server logs. Ensure the model is pulled: `ollama pull {model}`")
    )]
    RequestFailed {
        url: String,
        status: u16,
        body: String,
        model: String,
        /// `Retry-After` and structured `error.code`/`error.type`, for the
        /// provider-failure taxonomy (Phase 53g PR 4). Not part of the message.
        /// Boxed so the error stays small on every `Result` path.
        detail: Box<super::provider_failure::ProviderErrorDetail>,
    },

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

impl OllamaError {
    /// Map a `ureq` failure of a request to `url`: an HTTP status becomes
    /// [`Self::RequestFailed`] with its provider detail captured, anything
    /// else [`Self::Unreachable`].
    fn from_ureq(error: ureq::Error, url: &str, model: &str) -> Self {
        match error {
            ureq::Error::Status(status, resp) => {
                let (body, detail) =
                    super::provider_failure::ProviderErrorDetail::from_response(resp);
                Self::RequestFailed {
                    url: url.to_owned(),
                    status,
                    body,
                    model: model.to_owned(),
                    detail: Box::new(detail),
                }
            }
            other => Self::Unreachable {
                url: url.to_owned(),
                reason: other.to_string(),
            },
        }
    }

    /// What this failure says about the provider, for the provider gate.
    /// `None` for a response the provider sent but the client could not
    /// parse: the provider answered.
    pub fn provider_evidence(&self) -> Option<super::provider_failure::ProviderFailureEvidence> {
        use super::provider_failure::ProviderFailureEvidence;
        match self {
            Self::RequestFailed { status, detail, .. } => Some(ProviderFailureEvidence::Http {
                status: *status,
                detail: detail.as_ref().clone(),
            }),
            Self::Unreachable { .. } | Self::Timeout { .. } => {
                Some(ProviderFailureEvidence::Transport)
            }
            Self::ParseFailed { .. } => None,
        }
    }
}

/// Convenience alias.
pub type OllamaResult<T> = std::result::Result<T, OllamaError>;

/// Result of a [`OllamaClient::generate_with_options`] call: the generated text
/// plus the actual prompt token count Ollama reported (`prompt_eval_count`).
/// The context-rot bench needs the real count to detect silent truncation of
/// long prompts (Ollama's default `num_ctx` drops turns, faking edge recall).
#[derive(Debug, Clone)]
pub struct GenerateResult {
    /// The generated text (`message.content`).
    pub text: String,
    /// Actual prompt tokens evaluated. `0` when the server does not report it.
    pub prompt_tokens: u64,
}

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
    /// Keepalive: a chunk was received from the provider but yielded no visible
    /// delta (e.g. tool-call payloads, which Ollama accumulates). Signals
    /// transport liveness so the idle watchdog doesn't mistake an active stream
    /// for a stuck one (work item "Turns die quietly" ask 3).
    Keepalive,
}

/// Append the exact wire body to `/tmp/akh_ollama_debug.log` when
/// `AKH_OLLAMA_DEBUG_DUMP=1` (default off; same idiom as `AKH_AGENT_DEBUG`
/// for the Candle backing) — the tool that pinned down the ollama
/// `Qwen35Renderer` "no user query found in messages" 500 (a request-shape
/// bug that reproduces intermittently and leaves no server-side trace once
/// the request has already failed) by correlating the literal `messages`
/// array to the failure. Kept as a permanent, opt-in diagnostic for the next
/// time a request-shape issue needs the exact wire body. File (not stderr):
/// journald splits multi-line `eprintln!` output across separate unprefixed
/// entries, which makes a pretty-printed JSON body unreadable in
/// `journalctl`.
fn debug_dump_request(label: &str, request_seq: usize, body: &serde_json::Value) {
    if std::env::var("AKH_OLLAMA_DEBUG_DUMP").as_deref() != Ok("1") {
        return;
    }
    use std::io::Write;
    let epoch_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let line = format!(
        "\n=== [t={epoch_secs}] {label} request_seq={request_seq} ===\n{}\n",
        serde_json::to_string_pretty(body).unwrap_or_default()
    );
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("/tmp/akh_ollama_debug.log")
    {
        let _ = f.write_all(line.as_bytes());
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Synchronous Ollama REST client.
pub struct OllamaClient {
    config: OllamaConfig,
    /// Model capabilities from `/api/show`, fetched once and cached.
    capabilities: std::sync::OnceLock<super::chat_backend::ModelCapabilities>,
}

impl OllamaClient {
    /// Create a new client with the given config.
    pub fn new(config: OllamaConfig) -> Self {
        Self {
            config,
            capabilities: std::sync::OnceLock::new(),
        }
    }

    /// The configuration this client was built with.
    pub fn config(&self) -> &OllamaConfig {
        &self.config
    }

    /// HTTP agent for chat/generate requests: a short **connect** timeout (fail
    /// fast when the host is unreachable) plus a **read/idle** timeout —
    /// `timeout_secs` bounds the gap *between* response chunks, NOT the total
    /// generation. A streaming turn that keeps producing tokens is never killed
    /// for merely being long (the opencode/aider idle-timeout pattern; ureq's
    /// request `.timeout()` is a *total* cap, which was aborting live sessions).
    fn chat_agent(&self) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(20))
            .timeout_read(std::time::Duration::from_secs(
                self.config.timeout_secs.max(30),
            ))
            .build()
    }

    /// Model capabilities (context window, thinking/tools support) via
    /// `POST /api/show`, fetched once and cached. Falls back to "unknown"
    /// ([`Default`]) on any failure (offline, older server) so callers degrade.
    pub fn capabilities(&self) -> &super::chat_backend::ModelCapabilities {
        self.capabilities
            .get_or_init(|| self.fetch_capabilities().unwrap_or_default())
    }

    fn fetch_capabilities(&self) -> Option<super::chat_backend::ModelCapabilities> {
        let url = format!("{}/api/show", self.config.base_url.trim_end_matches('/'));
        let mut req =
            ureq::post(&url).timeout(std::time::Duration::from_secs(self.config.timeout_secs));
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let resp: serde_json::Value = req
            .send_json(serde_json::json!({ "model": self.config.model }))
            .ok()?
            .into_json()
            .ok()?;
        let caps = resp.get("capabilities").and_then(|c| c.as_array());
        let has = |name: &str| {
            caps.map(|a| a.iter().any(|v| v.as_str() == Some(name)))
                .unwrap_or(false)
        };
        // model_info holds `<arch>.context_length` (the key is arch-prefixed).
        let context_window = resp
            .get("model_info")
            .and_then(|m| m.as_object())
            .and_then(|m| {
                m.iter()
                    .find(|(k, _)| k.ends_with(".context_length"))
                    .and_then(|(_, v)| v.as_u64())
            })
            .map(|n| n as usize);
        let caps = super::chat_backend::ModelCapabilities {
            context_window,
            supports_thinking: has("thinking"),
            supports_tools: has("tools"),
            // The ACP bridge does not yet forward images to Ollama.  Do not
            // claim the model's wire capability as an Akh runtime capability.
            supports_images: false,
        };
        tracing::info!(
            model = %self.config.model,
            context_window = ?caps.context_window,
            thinking = caps.supports_thinking,
            tools = caps.supports_tools,
            "ollama model capabilities (/api/show)"
        );
        Some(caps)
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
        let mut req =
            ureq::get(&url).timeout(std::time::Duration::from_secs(self.config.timeout_secs));
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let response = req
            .call()
            .map_err(|e| OllamaError::from_ureq(e, &url, &self.config.model))?;

        let body: serde_json::Value =
            response.into_json().map_err(|e| OllamaError::ParseFailed {
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

        let mut req = self
            .chat_agent()
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let response = req
            .send_json(&body)
            .map_err(|e| OllamaError::from_ureq(e, &url, &self.config.model))?;

        let chat_resp: ChatResponse =
            response.into_json().map_err(|e| OllamaError::ParseFailed {
                reason: format!("JSON decode failed: {e}"),
            })?;
        // Phase 53b: meter one-shot generation (the Librarian's path) for the
        // autonomy budget; chat paths meter through `LlmRequestMetrics::log`.
        super::llm_metrics::record_usage(chat_resp.prompt_eval_count, chat_resp.eval_count);

        chat_resp
            .message
            .and_then(|m| m.content)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| OllamaError::ParseFailed {
                reason: "message.content missing or empty in Ollama response".into(),
            })
    }

    /// Like [`generate`], but sets Ollama's per-request `options.num_ctx` and
    /// returns the actual prompt token count (`prompt_eval_count`).
    ///
    /// The context-rot bench (WI-386) sweeps prompt lengths up to a model's
    /// advertised window; Ollama's default `num_ctx` is small, and a prompt
    /// longer than it is **silently truncated**, which the bench's truncation
    /// detector must catch via the returned real count. Setting `num_ctx`
    /// explicitly per probe keeps that honest instead of faking good edge
    /// recall on a prompt the model never fully saw.
    pub fn generate_with_options(
        &self,
        system_prompt: &str,
        user_prompt: &str,
        num_ctx: usize,
    ) -> OllamaResult<GenerateResult> {
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "stream": false,
            "keep_alive": self.config.keep_alive,
            "options": { "num_ctx": num_ctx },
        });

        let mut req = self
            .chat_agent()
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let response = req
            .send_json(&body)
            .map_err(|e| OllamaError::from_ureq(e, &url, &self.config.model))?;

        let chat_resp: ChatResponse =
            response.into_json().map_err(|e| OllamaError::ParseFailed {
                reason: format!("JSON decode failed: {e}"),
            })?;

        let text = chat_resp
            .message
            .and_then(|m| m.content)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| OllamaError::ParseFailed {
                reason: "message.content missing or empty in Ollama response".into(),
            })?;

        Ok(GenerateResult {
            text,
            prompt_tokens: chat_resp.prompt_eval_count,
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
        request_seq: usize,
    ) -> OllamaResult<AssistantTurn> {
        let url = format!("{}/api/chat", self.config.base_url.trim_end_matches('/'));

        // Metrics: track this non-streaming request.
        let mut metrics = super::llm_metrics::LlmRequestMetrics::new(request_seq);
        let span =
            super::llm_metrics::LlmRequestMetrics::request_span(&self.config.model, request_seq);
        metrics.attach_span(span.clone());
        let _request_guard = span.entered();

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
        // (gpt-oss/glm) and the dedicated Qwen3.8 renderer accept
        // `think: "low"|"medium"|"high"|"max"`. Absent ⇒ model default.
        if self.config.reasoning_effort.is_some() {
            if self.capabilities().supports_thinking {
                if let Some(level) = self.config.validated_think_level() {
                    body["think"] = serde_json::Value::String(level.to_string());
                }
            } else {
                tracing::debug!(
                    model = %self.config.model,
                    "reasoning_effort set but model lacks the 'thinking' capability; not sending think"
                );
            }
        }

        // See `debug_dump_request` — AKH_OLLAMA_DEBUG_DUMP=1 to capture.
        debug_dump_request("chat (non-streaming)", request_seq, &body);

        let mut req = self
            .chat_agent()
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let response = req
            .send_json(&body)
            .map_err(|e| OllamaError::from_ureq(e, &url, &self.config.model))?;

        let chat_resp: ChatResponse = response.into_json().map_err(|e| {
            metrics.log(
                &self.config.model,
                super::llm_metrics::RequestOutcome::Partial,
            );
            OllamaError::ParseFailed {
                reason: format!("JSON decode failed: {e}"),
            }
        })?;

        let msg = chat_resp.message.ok_or_else(|| {
            metrics.log(
                &self.config.model,
                super::llm_metrics::RequestOutcome::Partial,
            );
            OllamaError::ParseFailed {
                reason: "Ollama chat response had no message".into(),
            }
        })?;

        // Track the response: non-streaming means we got one "chunk" (the whole response).
        let content_len = msg.content.as_ref().map(|s| s.len()).unwrap_or(0);
        metrics.note_chunk(content_len);
        if content_len > 0 {
            metrics.note_content_delta();
        }
        if !msg.tool_calls.is_empty() {
            metrics.note_tool_call_chunk();
        }
        metrics.set_tokens(chat_resp.prompt_eval_count, chat_resp.eval_count);
        metrics.log(
            &self.config.model,
            super::llm_metrics::RequestOutcome::Completed,
        );

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

    /// Run one tool-enabled `/api/chat` turn with **streaming** enabled,
    /// retrying transient connection failures before falling back.
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
    /// deltas (thinking is intentionally **not** folded in).
    ///
    /// **Retry logic**: transient transport errors (connection reset, read
    /// timeout) trigger up to 2 retries with exponential backoff (200ms, 400ms)
    /// before surfacing the error to the caller. HTTP-level failures (4xx/5xx)
    /// are not retried. Unparseable NDJSON lines are skipped defensively rather
    /// than aborting the stream.
    pub fn chat_stream(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        request_seq: usize,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> OllamaResult<AssistantTurn> {
        const MAX_RETRIES: usize = 2;
        const INITIAL_BACKOFF_MS: u64 = 200;

        let mut attempt = 0;
        loop {
            match self.chat_stream_once(messages, tools, request_seq, on_event) {
                Ok(turn) => return Ok(turn),
                Err(e) => {
                    // Only retry transient transport errors (connection reset,
                    // network unreachable, read timeout); HTTP errors (4xx/5xx)
                    // are not retried.
                    let is_retryable = matches!(&e, OllamaError::Unreachable { reason, .. }
                        if reason.contains("reset") || reason.contains("timeout") || reason.contains("Network Error"));
                    if !is_retryable || attempt >= MAX_RETRIES {
                        return Err(e);
                    }
                    attempt += 1;
                    let backoff_ms = INITIAL_BACKOFF_MS * (1 << (attempt - 1));
                    tracing::debug!(
                        attempt,
                        backoff_ms,
                        error = %e,
                        "streaming request failed with retryable error; backing off"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms));
                }
            }
        }
    }

    /// Internal: one streaming attempt without retry.
    fn chat_stream_once(
        &self,
        messages: &[serde_json::Value],
        tools: &[serde_json::Value],
        request_seq: usize,
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> OllamaResult<AssistantTurn> {
        use std::io::BufRead;
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let request_start = std::time::Instant::now();
        // Metrics: track per-request counters (chunks, bytes, deltas) for diagnosing
        // "turns die quietly" (work item ask 1-4).
        let mut metrics = super::llm_metrics::LlmRequestMetrics::new(request_seq);
        let span =
            super::llm_metrics::LlmRequestMetrics::request_span(&self.config.model, request_seq);
        metrics.attach_span(span.clone());
        let _request_guard = span.entered();
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
        if self.config.reasoning_effort.is_some() {
            if self.capabilities().supports_thinking {
                if let Some(level) = self.config.validated_think_level() {
                    body["think"] = serde_json::Value::String(level.to_string());
                }
            } else {
                tracing::debug!(
                    model = %self.config.model,
                    "reasoning_effort set but model lacks the 'thinking' capability; not sending think"
                );
            }
        }

        // See `debug_dump_request` — AKH_OLLAMA_DEBUG_DUMP=1 to capture.
        debug_dump_request("chat_stream", request_seq, &body);

        let mut req = self
            .chat_agent()
            .post(&url)
            .set("Content-Type", "application/json");
        if let Some(key) = self.config.resolved_api_key() {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let response = req
            .send_json(&body)
            .map_err(|e| OllamaError::from_ureq(e, &url, &self.config.model))?;

        // Client-side heartbeat + decoupled streaming: heartbeats must be
        // delivered DURING provider silence, not after. The previous implementation
        // drained heartbeats inside the blocking `reader.lines()` loop, so they
        // accumulated in the channel and were flushed only after the first byte
        // arrived — which is precisely when the watchdog had already fired. This
        // inverted architecture moves the blocking read into its own thread and
        // selects over a line channel and a heartbeat channel, so heartbeats fire
        // on schedule regardless of provider behavior (work item "The watchdog is
        // blind while the model generates a tool call", review round 2).
        const HEARTBEAT_INTERVAL_SECS: u64 = 10;
        let request_alive = Arc::new(AtomicBool::new(true));

        // Line-reader thread: read NDJSON lines from the response stream and
        // send them through a channel. Signals completion by dropping the sender.
        enum LineEvent {
            Line(Result<String, std::io::Error>),
        }
        let (line_tx, line_rx) = std::sync::mpsc::channel::<LineEvent>();
        let reader_alive = Arc::clone(&request_alive);
        let reader_thread = {
            let reader = std::io::BufReader::new(response.into_reader());
            std::thread::spawn(move || {
                use std::io::BufRead;
                for line in reader.lines() {
                    if !reader_alive.load(Ordering::Relaxed) {
                        break;
                    }
                    if line_tx.send(LineEvent::Line(line)).is_err() {
                        // Receiver dropped — caller is gone.
                        break;
                    }
                }
                // Sender drops, signaling EOF to the main thread.
            })
        };

        // Heartbeat thread: emit Keepalive on a timer.
        let (heartbeat_tx, heartbeat_rx) = std::sync::mpsc::channel::<StreamEvent>();
        let (heartbeat_stop_tx, heartbeat_stop_rx) = std::sync::mpsc::channel::<()>();
        let heartbeat_alive = Arc::clone(&request_alive);
        let heartbeat_thread = std::thread::spawn(move || {
            while heartbeat_alive.load(Ordering::Relaxed) {
                match heartbeat_stop_rx
                    .recv_timeout(std::time::Duration::from_secs(HEARTBEAT_INTERVAL_SECS))
                {
                    Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                }
                if heartbeat_alive.load(Ordering::Relaxed)
                    && heartbeat_tx.send(StreamEvent::Keepalive).is_err()
                {
                    // Receiver dropped.
                    break;
                }
            }
        });

        // Main event loop: select over lines and heartbeats. This is a
        // synchronous `select!`-equivalent using `recv_timeout` — check
        // heartbeats frequently (every 100ms) so they are delivered promptly,
        // but don't busy-wait.
        let mut content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut prompt_tokens = 0u64;
        let mut completion_tokens = 0u64;
        loop {
            // Drain all pending heartbeats (non-blocking).
            while let Ok(ev) = heartbeat_rx.try_recv() {
                on_event(ev);
            }

            // Wait for the next line with a short timeout so we can check for
            // heartbeats periodically.
            let line_result = line_rx.recv_timeout(std::time::Duration::from_millis(100));
            match line_result {
                Ok(LineEvent::Line(line)) => {
                    let line = line.map_err(|e| {
                        // Capture request duration at the point of stream failure for
                        // diagnosing duration-bound connection resets (work item "Turns
                        // die quietly" defect 2, ask 4).
                        let duration = request_start.elapsed();
                        tracing::warn!(
                            duration_secs = duration.as_secs(),
                            error = %e,
                            "streaming read failed mid-response"
                        );
                        // Signal heartbeat thread to stop before returning error.
                        request_alive.store(false, Ordering::Relaxed);
                        // Log metrics on cancellation path.
                        metrics.log(&self.config.model, metrics.interrupted_outcome());
                        OllamaError::Unreachable {
                            url: url.clone(),
                            reason: format!("stream read error: {e}"),
                        }
                    })?;
                    let line = line.trim();
                    // Track chunk received (bytes).
                    metrics.note_chunk(line.len());
                    if line.is_empty() {
                        continue;
                    }
                    // Parse each NDJSON chunk leniently — a malformed line is skipped,
                    // not fatal, so one odd frame can't abort an otherwise-good stream.
                    let Ok(chunk) = serde_json::from_str::<serde_json::Value>(line) else {
                        continue;
                    };
                    if let Some(msg) = chunk.get("message") {
                        // Track whether this chunk produced any visible event so we can
                        // emit a Keepalive if it didn't (transport-liveness signal).
                        let mut emitted_something = false;
                        if let Some(delta) = msg.get("content").and_then(serde_json::Value::as_str)
                            && !delta.is_empty()
                        {
                            content.push_str(delta);
                            on_event(StreamEvent::Content(delta.to_string()));
                            metrics.note_content_delta();
                            emitted_something = true;
                        }
                        if let Some(delta) = msg.get("thinking").and_then(serde_json::Value::as_str)
                            && !delta.is_empty()
                        {
                            on_event(StreamEvent::Thinking(delta.to_string()));
                            metrics.note_thinking_delta();
                            emitted_something = true;
                        }
                        if let Some(calls) =
                            msg.get("tool_calls").and_then(serde_json::Value::as_array)
                        {
                            metrics.note_tool_call_chunk();
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
                        // If we received a chunk but emitted nothing (e.g. accumulated
                        // tool-call payloads), emit Keepalive so the idle watchdog knows
                        // the stream is alive (ask 3: "reset on any chunk received").
                        if !emitted_something {
                            metrics.note_keepalive();
                            on_event(StreamEvent::Keepalive);
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
                        metrics.set_tokens(prompt_tokens, completion_tokens);
                        break;
                    }
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    // No line received in the last 100ms — continue the loop to
                    // check for heartbeats again. This is NOT a failure; the
                    // reader thread is still working.
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                    // Line sender dropped (EOF or reader thread panicked).
                    // If we got here without seeing `done: true`, the stream
                    // ended prematurely.
                    break;
                }
            }
        }

        // Signal both threads to stop.
        request_alive.store(false, Ordering::Relaxed);
        let _ = heartbeat_stop_tx.send(());
        let _ = reader_thread.join();
        let _ = heartbeat_thread.join();

        // Drain any remaining heartbeats (one might have been sent just before
        // we set request_alive to false).
        while let Ok(heartbeat_event) = heartbeat_rx.try_recv() {
            on_event(heartbeat_event);
        }

        // Log metrics on successful completion.
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

    /// Best-effort fetch of the model's context-window size via `/api/show`.
    ///
    /// Ollama reports it in `model_info` under an architecture-prefixed key (e.g.
    /// `llama.context_length`, `gemma.context_length`), so we scan for any key
    /// ending in `.context_length`. Returns `None` on any transport/parse error
    /// or if the field is absent — callers then omit the `% used` figure.
    pub fn context_length(&self) -> Option<u64> {
        // Delegates to the cached `/api/show` capabilities (one round-trip total).
        self.capabilities().context_window.map(|n| n as u64)
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

    /// Phase 53g PR 4: an HTTP error keeps its message and captures the
    /// status and `Retry-After` (an HTTP-date here) for the provider gate.
    #[test]
    fn request_failed_captures_provider_evidence() {
        use crate::agent::provider_failure::{ProviderFailureEvidence, RetryAfter};
        let body = r#"{"error":"unauthorized"}"#;
        let raw = format!(
            "HTTP/1.1 401 Unauthorized\r\nRetry-After: Wed, 21 Oct 2037 07:28:00 GMT\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let response: ureq::Response = raw.parse().unwrap();
        let error = OllamaError::from_ureq(
            ureq::Error::Status(401, response),
            "http://h/api/chat",
            "glm",
        );
        assert_eq!(
            error.to_string(),
            format!("Ollama request failed (HTTP 401) at http://h/api/chat: {body}")
        );
        let Some(ProviderFailureEvidence::Http { status, detail }) = error.provider_evidence()
        else {
            panic!("expected HTTP evidence: {error:?}");
        };
        assert_eq!(status, 401);
        assert!(matches!(detail.retry_after, Some(RetryAfter::At(_))));
        // Ollama's string `error` is prose, not a structured identifier.
        assert_eq!((detail.error_code, detail.error_type), (None, None));
        assert_eq!(
            OllamaError::ParseFailed { reason: "x".into() }.provider_evidence(),
            None
        );
        assert_eq!(
            OllamaError::Timeout { timeout_secs: 1 }.provider_evidence(),
            Some(ProviderFailureEvidence::Transport)
        );
    }

    /// Consume the complete HTTP request before the mock peer responds. On
    /// macOS, closing a socket with unread request-body bytes can turn a valid
    /// response into `ECONNRESET` at the client.
    fn drain_http_request(stream: &mut std::net::TcpStream) {
        use std::io::Read as _;

        let mut request = Vec::new();
        let mut buffer = [0u8; 1024];
        let header_end = loop {
            let read = match stream.read(&mut buffer) {
                Ok(read) => read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => panic!("read mock Ollama request: {error}"),
            };
            assert!(read > 0, "mock Ollama request ended before its headers");
            request.extend_from_slice(&buffer[..read]);
            if let Some(offset) = request.windows(4).position(|window| window == b"\r\n\r\n") {
                break offset + 4;
            }
            assert!(
                request.len() <= 64 * 1024,
                "mock Ollama request headers are too large"
            );
        };

        let headers = std::str::from_utf8(&request[..header_end]).expect("ASCII HTTP headers");
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length").then(|| {
                    value
                        .trim()
                        .parse::<usize>()
                        .expect("numeric Content-Length")
                })
            })
            .expect("mock request includes Content-Length");
        let mut remaining = content_length.saturating_sub(request.len() - header_end);
        while remaining > 0 {
            let limit = remaining.min(buffer.len());
            let read = match stream.read(&mut buffer[..limit]) {
                Ok(read) => read,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => panic!("read mock Ollama request body: {error}"),
            };
            assert!(read > 0, "mock Ollama request body ended early");
            remaining -= read;
        }
    }

    /// `chat_stream` parses Ollama's newline-delimited streaming response:
    /// content deltas concatenate into `AssistantTurn.content` and arrive as
    /// `Content` events in order, `thinking` deltas arrive as `Thinking` events
    /// (and are NOT folded into content), and `tool_calls` are collected.
    #[test]
    // WI-306 (01a0144a-b545-7e90-b444-e72d13f20adc): real socket transport is
    // quarantined from CI; run it through the documented manual Ollama suite.
    #[ignore = "WI-306: manual Ollama suite uses real loopback TCP"]
    fn chat_stream_parses_ndjson_deltas_and_tool_calls() {
        use std::io::Write;
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
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => panic!("accept mock Ollama connection: {error}"),
                }
            };
            drain_http_request(&mut stream);
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            let _ = stream.shutdown(std::net::Shutdown::Write);
            // Keep the peer alive briefly after flushing. Dropping immediately
            // can turn an otherwise complete response into a reset/EINVAL on
            // heavily parallel CI hosts before the client drains its buffer.
            std::thread::sleep(std::time::Duration::from_millis(50));
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
                0,
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
                StreamEvent::Thinking(_) | StreamEvent::Keepalive => None,
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

    /// Regression test for work item "The watchdog is blind while the model
    /// generates a tool call": a turn whose model spends > idle_secs generating
    /// a single large tool call completes normally, not killed by the watchdog.
    ///
    /// Simulates Ollama's measured behavior: 46 total chunks over 7.5s, with the
    /// first 43 arriving in 180ms (thinking/content), then 6.9s of complete
    /// silence (tool-call generation), then 1 final chunk carrying the entire
    /// tool_calls payload. The client-side heartbeat (emitting Keepalive on a
    /// 10s timer) must keep the idle watchdog alive through the silent phase.
    #[test]
    // WI-306 (01a0144a-b545-7e90-b444-e72d13f20adc): real socket transport is
    // quarantined from CI; run it through the documented manual Ollama suite.
    #[ignore = "WI-306: manual Ollama suite uses real loopback TCP"]
    fn long_tool_call_generation_survives_via_heartbeat() {
        use std::io::Write;
        use std::net::TcpListener;
        use std::sync::{Arc, Mutex};
        use std::time::Instant;

        // Track emitted StreamEvents WITH TIMESTAMPS to verify heartbeats
        // arrive DURING the silent phase, not after (review round 2).
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = Arc::clone(&events);
        let test_start = Instant::now();

        // NDJSON body: 43 thinking chunks (streamed fast), then a 12s gap,
        // then 1 tool-call chunk, then done. This models the measured 6.9s
        // silence scaled up to exceed the 10s heartbeat interval.
        let mut body = String::new();
        for i in 0..43 {
            body.push_str(&format!("{{\"message\":{{\"thinking\":\"step {i}\"}}}}\n"));
        }
        let tool_call_chunk = "{\"message\":{\"tool_calls\":[{\"function\":{\"name\":\"write_file\",\"arguments\":{\"path\":\"/tmp/test\",\"content\":\"x\"}}}]}}\n";
        body.push_str("{\"done\":true}\n");

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\n\r\n"
        );

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(connection) => break connection,
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => panic!("accept mock Ollama connection: {error}"),
                }
            };
            drain_http_request(&mut stream);
            // Write header
            let _ = stream.write_all(response.as_bytes());
            // Write first 43 chunks fast
            for i in 0..43 {
                let chunk = format!("{{\"message\":{{\"thinking\":\"step {i}\"}}}}\n");
                let _ = stream.write_all(format!("{:x}\r\n", chunk.len()).as_bytes());
                let _ = stream.write_all(chunk.as_bytes());
                let _ = stream.write_all(b"\r\n");
            }
            // 12s gap simulating tool-call generation (no chunks)
            std::thread::sleep(std::time::Duration::from_secs(12));
            // Tool call chunk
            let _ = stream.write_all(format!("{:x}\r\n", tool_call_chunk.len()).as_bytes());
            let _ = stream.write_all(tool_call_chunk.as_bytes());
            let _ = stream.write_all(b"\r\n");
            // Done chunk
            let done = "{\"done\":true}\n";
            let _ = stream.write_all(format!("{:x}\r\n", done.len()).as_bytes());
            let _ = stream.write_all(done.as_bytes());
            let _ = stream.write_all(b"\r\n0\r\n\r\n");
            let _ = stream.flush();
            let _ = stream.shutdown(std::net::Shutdown::Write);
        });

        let client = OllamaClient::new(OllamaConfig {
            base_url: format!("http://127.0.0.1:{port}"),
            timeout_secs: 30, // socket timeout, not total
            ..OllamaConfig::default()
        });

        // Exercise one transport attempt directly. This regression is about
        // heartbeat delivery during provider silence, not retry policy; a
        // retry would need a second mock connection and can hide which stream
        // attempt actually produced the heartbeat.
        let turn = client
            .chat_stream_once(
                &[serde_json::json!({"role": "user", "content": "hi"})],
                &[serde_json::json!({"name": "write_file", "parameters": {}})],
                0,
                &mut |ev| {
                    let elapsed = test_start.elapsed();
                    events_clone.lock().unwrap().push((elapsed, ev));
                },
            )
            .expect("turn survives 12s silence via heartbeat");
        let _ = server.join();

        // Verify the turn succeeded and tool call was parsed.
        assert_eq!(turn.tool_calls.len(), 1, "tool call parsed");
        assert_eq!(turn.tool_calls[0].name, "write_file");

        // Verify heartbeats were emitted DURING the silent phase, not after.
        // The server writes 43 thinking chunks fast (< 100ms), then sleeps 12s,
        // then writes the tool call + done chunks. A heartbeat that arrives
        // only after the sleep ends (> 12s elapsed) would prove the old broken
        // behavior (heartbeats queued but not delivered until the read loop
        // woke up). We need at least one Keepalive in the window [1s, 12s].
        let events = events.lock().unwrap();
        let keepalives_during_silence: Vec<_> = events
            .iter()
            .filter_map(|(elapsed, ev)| {
                if matches!(ev, StreamEvent::Keepalive)
                    && elapsed.as_secs_f64() > 1.0  // After thinking chunks
                    && elapsed.as_secs_f64() < 12.0
                // Before tool call arrives
                {
                    Some(elapsed.as_secs_f64())
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !keepalives_during_silence.is_empty(),
            "at least 1 heartbeat must arrive DURING the 12s silent window (1s-12s), \
             got {} total Keepalives but none in the silent window. Timings: {:?}",
            events
                .iter()
                .filter(|(_, ev)| matches!(ev, StreamEvent::Keepalive))
                .count(),
            keepalives_during_silence
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
    fn validated_think_level_none_when_unset() {
        let cfg = OllamaConfig::default();
        assert_eq!(cfg.validated_think_level(), None);
    }

    #[test]
    fn validated_think_level_accepts_every_documented_level() {
        for level in VALID_THINK_LEVELS {
            let cfg = OllamaConfig {
                reasoning_effort: Some((*level).into()),
                ..OllamaConfig::default()
            };
            assert_eq!(cfg.validated_think_level(), Some(*level));
        }
    }

    #[test]
    fn validated_think_level_is_case_and_whitespace_insensitive() {
        let cfg = OllamaConfig {
            reasoning_effort: Some("  HIGH  ".into()),
            ..OllamaConfig::default()
        };
        assert_eq!(cfg.validated_think_level(), Some("high"));
    }

    #[test]
    fn validated_think_level_rejects_unknown_values_rather_than_passing_them_through() {
        // This is the "reject/log unknown ones rather than passing them
        // through blindly" requirement: a typo'd or stale value must degrade
        // to "no think field sent", never reach the wire verbatim.
        let cfg = OllamaConfig {
            reasoning_effort: Some("xhigh".into()),
            ..OllamaConfig::default()
        };
        assert_eq!(cfg.validated_think_level(), None);
    }

    #[test]
    // WI-306 (01a0144a-b545-7e90-b444-e72d13f20adc): even negative transport
    // checks are kept in the manual suite so default CI never opens Ollama TCP.
    #[ignore = "WI-306: manual Ollama suite exercises HTTP transport"]
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
    // WI-306 (01a0144a-b545-7e90-b444-e72d13f20adc): even negative transport
    // checks are kept in the manual suite so default CI never opens Ollama TCP.
    #[ignore = "WI-306: manual Ollama suite exercises HTTP transport"]
    fn health_check_returns_false_when_unreachable() {
        let cfg = OllamaConfig {
            base_url: "http://127.0.0.1:19999".into(), // nothing listening here
            ..OllamaConfig::default()
        };
        let client = OllamaClient::new(cfg);
        assert!(!client.health_check());
    }
}
