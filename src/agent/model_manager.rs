//! Per-agent model manager: routes inference to a local Gemma model (via Ollama)
//! with per-agent Psyche-encoded system prompts (Method A), falling back to a
//! remote OpenAI-compatible API when Ollama is unavailable.
//!
//! ## Design
//!
//! `AgentModelManager` is a registry keyed by agent ID.  Each entry holds the
//! Psyche-derived system prompt for that agent plus a `last_used` timestamp used
//! for TTL-based eviction.
//!
//! Inference routing:
//! 1. If Ollama health-check passes → use Ollama with the per-agent system prompt.
//! 2. If Ollama is down or fails → try the fallback remote API (if configured).
//! 3. If neither is available → return `ModelManagerError::NoBackendAvailable`.
//!
//! Memory notes:
//! - `gemma4:e4b` is loaded once by Ollama and shared across all agents.
//! - On `unload_agent` the manager sends an Ollama `keep_alive: "0"` hint to
//!   encourage eviction when no other agents are active.
//!
//! ## Method B (future)
//!
//! When Ollama gains stable LoRA adapter support, `AgentEntry` will carry an
//! `adapter_path: Option<String>` passed as `lora` in the Ollama request.  The
//! routing logic and error surface stay identical.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::compartment::psyche::{ArchetypeWeights, Psyche};
use crate::extraction::config::ExternalApiConfig;

use super::ollama::{AssistantTurn, OllamaClient, OllamaConfig, OllamaError};

// ---------------------------------------------------------------------------
// PromptMode
// ---------------------------------------------------------------------------

/// Controls how the agent's system prompt is structured for a given model.
///
/// Auto-detected from the model name by [`detect_prompt_mode`], but can be
/// overridden via [`ModelManagerConfig::prompt_mode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptMode {
    /// Full Psyche-encoded prompt — best for capable models (e.g. DeepSeek).
    #[default]
    Standard,
    /// Compact prompt with strict brevity constraints — for smaller or verbose
    /// models (e.g. Gemma 4) that otherwise produce excessive text per cycle,
    /// inflating OODA cycle counts.
    Terse,
}

/// Auto-detect the prompt mode from a model identifier string.
///
/// Returns `PromptMode::Terse` when `model` contains `"gemma"` (case-insensitive),
/// `PromptMode::Standard` otherwise.
pub fn detect_prompt_mode(model: &str) -> PromptMode {
    if model.to_ascii_lowercase().contains("gemma") {
        PromptMode::Terse
    } else {
        PromptMode::Standard
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for `AgentModelManager`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManagerConfig {
    /// Local Ollama client configuration.
    #[serde(default)]
    pub ollama: OllamaConfig,
    /// Optional remote fallback API (OpenAI-compatible).
    /// When absent and Ollama is unreachable, `generate()` returns
    /// `NoBackendAvailable`.
    #[serde(default)]
    pub fallback_api: Option<ExternalApiConfig>,
    /// TTL in seconds before an idle agent entry is evicted (default: 300 = 5 min).
    #[serde(default = "default_ttl")]
    pub idle_ttl_secs: u64,
    /// Explicit prompt mode override.
    ///
    /// When `None` (the default), the mode is auto-detected from `ollama.model`:
    /// any model whose name contains `"gemma"` (case-insensitive) gets
    /// [`PromptMode::Terse`]; everything else gets [`PromptMode::Standard`].
    ///
    /// Set to `Some(PromptMode::Terse)` to force terse mode for other small or
    /// verbose models, or `Some(PromptMode::Standard)` to suppress auto-detection.
    #[serde(default)]
    pub prompt_mode: Option<PromptMode>,
    /// Explicit, authoritative system prompt for this akh. When `Some`, it
    /// *replaces* the psyche-derived prompt in [`assemble_system_prompt`] (the
    /// `prompt_mode` brevity block is *not* applied — an explicit prompt is
    /// verbatim); when `None`, the psyche-derived prompt is used. Sourced from
    /// `AkhConfig.system_prompt`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
}

impl Default for ModelManagerConfig {
    fn default() -> Self {
        Self {
            ollama: OllamaConfig::default(),
            fallback_api: None,
            idle_ttl_secs: default_ttl(),
            prompt_mode: None,
            system_prompt: None,
        }
    }
}

fn default_ttl() -> u64 { 300 }

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from `AgentModelManager`.
#[derive(Debug, Error, Diagnostic)]
pub enum ModelManagerError {
    #[error("agent \"{agent_id}\" is not loaded — call load_agent() first")]
    #[diagnostic(
        code(akh::model_manager::not_loaded),
        help(
            "Call model_manager.load_agent(agent_id, agent_name, psyche) \
             before calling generate()."
        )
    )]
    AgentNotLoaded { agent_id: String },

    #[error("no inference backend available for agent \"{agent_id}\"")]
    #[diagnostic(
        code(akh::model_manager::no_backend),
        help(
            "Start Ollama (`ollama serve && ollama pull gemma4:e4b`) or configure \
             a fallback_api in ModelManagerConfig."
        )
    )]
    NoBackendAvailable { agent_id: String },

    #[error("fallback API has no key — set AKH_EXTRACTION_API_KEY or configure fallback_api.api_key")]
    #[diagnostic(
        code(akh::model_manager::no_api_key),
        help(
            "Export AKH_EXTRACTION_API_KEY in the environment, or set \
             model_manager.fallback_api.api_key in your config file."
        )
    )]
    NoApiKey,

    #[error("fallback API request failed (HTTP {status}) at {url}: {body}")]
    #[diagnostic(
        code(akh::model_manager::fallback_request_failed),
        help("Verify the fallback_api base_url, api_key, and model name in ModelManagerConfig.")
    )]
    FallbackRequestFailed { url: String, status: u16, body: String },

    #[error("failed to parse fallback API response: {reason}")]
    #[diagnostic(
        code(akh::model_manager::fallback_parse_failed),
        help("The remote API returned JSON that does not include choices[0].message.content.")
    )]
    FallbackParseFailed { reason: String },

    #[error("Ollama error: {0}")]
    #[diagnostic(transparent)]
    Ollama(#[from] OllamaError),
}

/// Convenience alias.
pub type ModelManagerResult<T> = std::result::Result<T, ModelManagerError>;

// ---------------------------------------------------------------------------
// Internal per-agent state
// ---------------------------------------------------------------------------

struct AgentEntry {
    /// Psyche-derived system prompt injected on every Ollama request (Method A).
    system_prompt: String,
    /// Wall-clock timestamp of the last `generate()` call.
    last_used: Instant,
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// Registry and inference router for per-agent local Gemma instances.
///
/// Thread-safe via `Mutex<HashMap<...>>`.  The OODA loop is single-threaded so
/// contention is negligible; the Mutex exists for daemon use-cases where the
/// manager may be shared via `Arc`.
/// Per-manager token-usage tally, accumulated across agentic chat turns.
///
/// Interior-mutable (atomics) so it can be updated from `&self` chat methods
/// while the manager is shared behind an `Arc`. `prompt`/`completion` are
/// cumulative spend; `last_context_tokens` is the most recent turn's footprint
/// (a stand-in for "current context fill").
#[derive(Debug, Default)]
pub struct UsageLedger {
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    turns: AtomicU64,
    last_context_tokens: AtomicU64,
}

impl UsageLedger {
    /// Record one model turn's token counts.
    fn record(&self, prompt: u64, completion: u64) {
        self.prompt_tokens.fetch_add(prompt, Ordering::Relaxed);
        self.completion_tokens.fetch_add(completion, Ordering::Relaxed);
        self.turns.fetch_add(1, Ordering::Relaxed);
        self.last_context_tokens
            .store(prompt + completion, Ordering::Relaxed);
    }

    /// A consistent-enough point-in-time copy for read-only callers. The
    /// `context_window` is filled by the manager (it owns that cache), not here.
    fn snapshot(&self) -> UsageSnapshot {
        UsageSnapshot {
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            turns: self.turns.load(Ordering::Relaxed),
            context_tokens: self.last_context_tokens.load(Ordering::Relaxed),
            context_window: None,
        }
    }
}

/// Plain snapshot of a [`UsageLedger`] for reporting (e.g. `_akh/usage`).
#[derive(Debug, Clone, Copy, Default)]
pub struct UsageSnapshot {
    /// Cumulative prompt (input) tokens this session.
    pub prompt_tokens: u64,
    /// Cumulative completion (output) tokens this session.
    pub completion_tokens: u64,
    /// Number of model turns recorded.
    pub turns: u64,
    /// Most recent turn's token footprint (prompt + completion) — a proxy for
    /// current context fill.
    pub context_tokens: u64,
    /// Model context-window size, when known (lazily fetched via `/api/show`);
    /// `None` ⇒ the meter shows absolute counts without a percentage.
    pub context_window: Option<u64>,
}

impl UsageSnapshot {
    /// Total tokens billed this session (prompt + completion).
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.prompt_tokens + self.completion_tokens
    }
}

pub struct AgentModelManager {
    config: ModelManagerConfig,
    ollama: OllamaClient,
    agents: Mutex<HashMap<String, AgentEntry>>,
    /// Resolved prompt mode for this manager (auto-detected or config-overridden).
    prompt_mode: PromptMode,
    /// Cumulative token usage across this manager's agentic turns.
    usage: UsageLedger,
    /// Lazily-fetched model context-window size (0 = unknown). Populated once via
    /// [`OllamaClient::context_length`] after the first turn.
    context_window: AtomicU64,
    /// Whether the one-time context-window fetch has been attempted.
    context_window_tried: AtomicBool,
}

impl AgentModelManager {
    /// Create a new manager with the given config.
    pub fn new(config: ModelManagerConfig) -> Self {
        let prompt_mode = config
            .prompt_mode
            .unwrap_or_else(|| detect_prompt_mode(&config.ollama.model));
        let ollama = OllamaClient::new(config.ollama.clone());
        tracing::debug!(
            model = %config.ollama.model,
            ?prompt_mode,
            "model manager initialised",
        );
        Self {
            config,
            ollama,
            agents: Mutex::new(HashMap::new()),
            prompt_mode,
            usage: UsageLedger::default(),
            context_window: AtomicU64::new(0),
            context_window_tried: AtomicBool::new(false),
        }
    }

    /// Return the prompt mode in effect for this manager.
    pub fn prompt_mode(&self) -> PromptMode {
        self.prompt_mode
    }

    /// A point-in-time copy of this manager's cumulative token usage, including
    /// the cached context-window size when known.
    pub fn usage(&self) -> UsageSnapshot {
        let mut snap = self.usage.snapshot();
        let window = self.context_window.load(Ordering::Relaxed);
        snap.context_window = (window > 0).then_some(window);
        snap
    }

    /// Lazily fetch and cache the model's context-window size, at most once
    /// (best-effort). Called after a turn so the one `/api/show` round-trip never
    /// blocks the first user-visible response.
    fn ensure_context_window(&self) {
        if self.context_window_tried.swap(true, Ordering::Relaxed) {
            return; // already attempted
        }
        if let Some(window) = self.ollama.context_length() {
            self.context_window.store(window, Ordering::Relaxed);
        }
    }

    /// The pinned model's capabilities (context window, thinking/tools support),
    /// read once from Ollama's `/api/show` and cached. Used by the Phase 44
    /// context budget and reasoning-effort gating.
    pub fn capabilities(&self) -> super::chat_backend::ModelCapabilities {
        self.ollama.capabilities().clone()
    }

    // ── Registration ───────────────────────────────────────────────────────

    /// Register (or refresh) an agent with its Psyche encoding.
    ///
    /// Idempotent: if the agent is already loaded, its system prompt is
    /// refreshed from the new psyche and `last_used` is bumped.
    ///
    /// Callers can pass `None` for `psyche` to get a generic system prompt
    /// keyed only on the agent's name.
    ///
    /// The `manifests` parameter provides the advertised tool manifests for
    /// generating the tool-operations section of the prompt.
    pub fn load_agent(
        &self,
        agent_id: &str,
        agent_name: &str,
        psyche: Option<&Psyche>,
        manifests: &[&crate::agent::tool_manifest::ToolManifest],
    ) {
        let prompt = assemble_system_prompt(
            agent_id,
            agent_name,
            psyche,
            self.config.system_prompt.as_deref(),
            self.prompt_mode,
            manifests,
        );
        let mut agents = self.agents.lock().expect("model manager lock poisoned");
        let entry = agents.entry(agent_id.to_string()).or_insert_with(|| AgentEntry {
            system_prompt: prompt.clone(),
            last_used: Instant::now(),
        });
        // Always refresh prompt and timestamp on explicit load.
        entry.system_prompt = prompt;
        entry.last_used = Instant::now();

        tracing::debug!(
            agent_id,
            agent_name,
            model = %self.config.ollama.model,
            tool_count = manifests.len(),
            "agent model entry loaded"
        );
    }

    // ── Inference ─────────────────────────────────────────────────────────

    /// Generate a response for the agent from the knowledge-graph prompt.
    ///
    /// Routing:
    /// 1. Ollama (if reachable)
    /// 2. Fallback remote API (if configured)
    /// 3. `NoBackendAvailable`
    ///
    /// Error variants callers can handle:
    /// - `AgentNotLoaded` → call `load_agent()` first.
    /// - `NoBackendAvailable` → start Ollama or configure fallback_api.
    pub fn generate(&self, agent_id: &str, user_prompt: &str) -> ModelManagerResult<String> {
        let system_prompt = {
            let mut agents = self.agents.lock().expect("model manager lock poisoned");
            let entry = agents
                .get_mut(agent_id)
                .ok_or_else(|| ModelManagerError::AgentNotLoaded {
                    agent_id: agent_id.into(),
                })?;
            entry.last_used = Instant::now();
            entry.system_prompt.clone()
        };

        // Tier 1: Ollama (local Gemma).
        if self.ollama.health_check() {
            match self.ollama.generate(&system_prompt, user_prompt) {
                Ok(resp) => {
                    tracing::debug!(agent_id, backend = "ollama", "inference succeeded");
                    return Ok(resp);
                }
                Err(e) => {
                    tracing::warn!(
                        agent_id,
                        backend = "ollama",
                        "Ollama inference failed, trying fallback: {e}"
                    );
                }
            }
        } else {
            tracing::info!(
                agent_id,
                "Ollama unreachable — routing via remote LLM fallback"
            );
        }

        // Tier 2: Remote fallback API.
        if let Some(ref api_cfg) = self.config.fallback_api {
            return self.generate_via_fallback(agent_id, api_cfg, &system_prompt, user_prompt);
        }

        Err(ModelManagerError::NoBackendAvailable {
            agent_id: agent_id.into(),
        })
    }

    /// Run one tool-enabled chat turn for the agent.
    ///
    /// Prepends the agent's cached (psyche-/config-derived) system prompt to
    /// `conversation`, then calls Ollama `/api/chat` with the `tools` catalog.
    /// Unlike [`generate`](Self::generate) there is **no** remote fallback — tool
    /// calling is an Ollama-only capability for now, so callers should gate on
    /// [`ollama_available`](Self::ollama_available) and degrade to the symbolic
    /// pipeline when it returns `false`.
    pub fn chat_with_tools(
        &self,
        agent_id: &str,
        conversation: &[serde_json::Value],
        tools: &[serde_json::Value],
        request_seq: usize,
    ) -> ModelManagerResult<AssistantTurn> {
        let system_prompt = {
            let mut agents = self.agents.lock().expect("model manager lock poisoned");
            let entry =
                agents
                    .get_mut(agent_id)
                    .ok_or_else(|| ModelManagerError::AgentNotLoaded {
                        agent_id: agent_id.into(),
                    })?;
            entry.last_used = Instant::now();
            entry.system_prompt.clone()
        };

        let mut messages = Vec::with_capacity(conversation.len() + 1);
        messages.push(serde_json::json!({"role": "system", "content": system_prompt}));
        messages.extend_from_slice(conversation);

        let turn = self.ollama.chat(&messages, tools, request_seq)?;
        self.usage.record(turn.prompt_tokens, turn.completion_tokens);
        self.ensure_context_window();
        Ok(turn)
    }

    /// Streaming counterpart of [`chat_with_tools`](Self::chat_with_tools).
    ///
    /// Identical prompt assembly (agent system prompt prepended), but delegates
    /// to [`OllamaClient::chat_stream`] so the answer/thinking deltas reach
    /// `on_event` as the model generates. Returns the assembled
    /// [`AssistantTurn`] (full content + any tool calls). No remote fallback —
    /// same Ollama-only constraint as the non-streaming path.
    pub fn chat_with_tools_streaming(
        &self,
        agent_id: &str,
        conversation: &[serde_json::Value],
        tools: &[serde_json::Value],
        request_seq: usize,
        on_event: &mut dyn FnMut(super::ollama::StreamEvent),
    ) -> ModelManagerResult<AssistantTurn> {
        let system_prompt = {
            let mut agents = self.agents.lock().expect("model manager lock poisoned");
            let entry =
                agents
                    .get_mut(agent_id)
                    .ok_or_else(|| ModelManagerError::AgentNotLoaded {
                        agent_id: agent_id.into(),
                    })?;
            entry.last_used = Instant::now();
            entry.system_prompt.clone()
        };

        let mut messages = Vec::with_capacity(conversation.len() + 1);
        messages.push(serde_json::json!({"role": "system", "content": system_prompt}));
        messages.extend_from_slice(conversation);

        let turn = self.ollama.chat_stream(&messages, tools, request_seq, on_event)?;
        self.usage.record(turn.prompt_tokens, turn.completion_tokens);
        self.ensure_context_window();
        Ok(turn)
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    /// Evict agent entries that have been idle longer than `idle_ttl_secs`.
    ///
    /// Returns the number of entries evicted.  Call this periodically (e.g.,
    /// from the OODA idle handler or a background timer).
    pub fn unload_idle(&self) -> usize {
        let ttl = Duration::from_secs(self.config.idle_ttl_secs);
        let mut agents = self.agents.lock().expect("model manager lock poisoned");
        let before = agents.len();
        agents.retain(|agent_id, entry| {
            let keep = entry.last_used.elapsed() < ttl;
            if !keep {
                tracing::info!(
                    agent_id,
                    idle_secs = ?entry.last_used.elapsed().as_secs(),
                    ttl_secs = self.config.idle_ttl_secs,
                    "evicting idle agent model entry"
                );
            }
            keep
        });
        let evicted = before - agents.len();

        // When all agents are idle, hint Ollama to free GPU/CPU RAM.
        if evicted > 0 && agents.is_empty() {
            tracing::info!("all agents evicted — hinting Ollama to unload model");
            self.ollama.hint_unload();
        }

        evicted
    }

    /// Explicitly unload an agent entry.
    ///
    /// Returns `true` if the entry existed.
    pub fn unload_agent(&self, agent_id: &str) -> bool {
        let mut agents = self.agents.lock().expect("model manager lock poisoned");
        let removed = agents.remove(agent_id).is_some();
        if removed {
            tracing::info!(agent_id, "agent model entry explicitly unloaded");
            // Hint Ollama to unload if no more agents are registered.
            if agents.is_empty() {
                tracing::info!("no remaining agents — hinting Ollama to unload model");
                drop(agents); // release lock before the blocking HTTP call
                self.ollama.hint_unload();
            }
        }
        removed
    }

    // ── Observability ─────────────────────────────────────────────────────

    /// Returns `true` if Ollama is currently reachable.
    pub fn ollama_available(&self) -> bool {
        self.ollama.health_check()
    }

    /// List the IDs of currently registered agents.
    pub fn loaded_agents(&self) -> Vec<String> {
        let agents = self.agents.lock().expect("model manager lock poisoned");
        agents.keys().cloned().collect()
    }

    /// Returns `true` if the agent ID is registered.
    pub fn is_loaded(&self, agent_id: &str) -> bool {
        let agents = self.agents.lock().expect("model manager lock poisoned");
        agents.contains_key(agent_id)
    }

    /// The assembled system prompt currently stored for an agent, if registered.
    ///
    /// Exposes the result of [`assemble_system_prompt`] so callers (and tests)
    /// can confirm which prompt — explicit override or psyche-derived — an akh
    /// will send, without performing inference.
    pub fn system_prompt_for(&self, agent_id: &str) -> Option<String> {
        let agents = self.agents.lock().expect("model manager lock poisoned");
        agents.get(agent_id).map(|e| e.system_prompt.clone())
    }

    /// The model name this manager is pinned to (the akh's pinned model).
    pub fn model(&self) -> &str {
        &self.config.ollama.model
    }

    // ── Fallback helpers ──────────────────────────────────────────────────

    fn generate_via_fallback(
        &self,
        agent_id: &str,
        api_cfg: &ExternalApiConfig,
        system_prompt: &str,
        user_prompt: &str,
    ) -> ModelManagerResult<String> {
        let api_key = api_cfg.resolve_api_key().ok_or(ModelManagerError::NoApiKey)?;
        let url = format!("{}/chat/completions", api_cfg.base_url.trim_end_matches('/'));

        let body = serde_json::json!({
            "model": api_cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            "max_tokens": api_cfg.max_tokens,
            "temperature": api_cfg.temperature,
        });

        let response = ureq::post(&url)
            .set("Authorization", &format!("Bearer {api_key}"))
            .set("Content-Type", "application/json")
            .timeout(std::time::Duration::from_secs(api_cfg.timeout_secs))
            .send_json(&body)
            .map_err(|e| match e {
                ureq::Error::Status(status, resp) => ModelManagerError::FallbackRequestFailed {
                    url: url.clone(),
                    status,
                    body: resp.into_string().unwrap_or_default(),
                },
                other => ModelManagerError::FallbackRequestFailed {
                    url: url.clone(),
                    status: 0,
                    body: other.to_string(),
                },
            })?;

        let resp_body: serde_json::Value =
            response.into_json().map_err(|e| ModelManagerError::FallbackParseFailed {
                reason: format!("JSON decode failed: {e}"),
            })?;

        tracing::debug!(agent_id, backend = "fallback_api", "inference succeeded");

        resp_body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| ModelManagerError::FallbackParseFailed {
                reason: "choices[0].message.content missing in fallback API response".into(),
            })
    }
}

// ---------------------------------------------------------------------------
// Psyche → system prompt (Method A)
// ---------------------------------------------------------------------------

/// Build a Psyche-encoded system prompt for an agent (Method A).
///
/// Single authoritative system-prompt assembly entry point.
///
/// Composition (ADR-053, revised):
/// 1. **Operator section**: An explicit, non-empty `explicit` prompt (from
///    `AkhConfig.system_prompt`) takes precedence as the persona/voice authority;
///    otherwise the psyche-derived text from [`build_system_prompt`] is used.
/// 2. **Machine-owned sections** (always appended):
///    - Tool-operations section (generated from `manifests` registry)
///    - Operating conventions block
/// 3. **Mode constraints**: [`PromptMode::Terse`] brevity constraints compose
///    after both.
///
/// The explicit prompt no longer **replaces** the entire assembly — it composes
/// with the machine-owned sections. This ensures every akh receives tool
/// documentation regardless of whether an operator-written prompt is present.
///
/// This collapses the previously-scattered prompt-assembly logic so callers on
/// the inference path go through exactly one function.
pub fn assemble_system_prompt(
    agent_id: &str,
    agent_name: &str,
    psyche: Option<&Psyche>,
    explicit: Option<&str>,
    mode: PromptMode,
    manifests: &[&crate::agent::tool_manifest::ToolManifest],
) -> String {
    // 1. Operator section (persona/voice authority)
    let operator_section = if let Some(text) = explicit {
        if !text.trim().is_empty() {
            text.to_string()
        } else {
            build_system_prompt_body(agent_id, agent_name, psyche)
        }
    } else {
        build_system_prompt_body(agent_id, agent_name, psyche)
    };

    // 2. Machine-owned sections (always appended)
    let tool_section = generate_tool_operations_section(manifests);
    let conventions = OPERATING_CONVENTIONS;

    // 3. Assemble with mode-specific constraints
    let base = format!("{operator_section}\n\n{tool_section}\n\n{conventions}");
    match mode {
        PromptMode::Standard => base,
        PromptMode::Terse => format!("{base}\n\n{TERSE_CONSTRAINTS}"),
    }
}

/// Translates the agent's Jungian psyche configuration into a natural-language
/// system prompt that Ollama injects as the `system` role in every request.
/// Two agents with different psyches will produce different responses to identical
/// knowledge-graph facts.
///
/// When `psyche` is `None`, a minimal generic prompt is returned.
///
/// When `mode` is [`PromptMode::Terse`], a strict brevity constraint block is
/// appended.  This significantly reduces OODA cycle counts for smaller or verbose
/// models such as Gemma 4 (observed: 30 cycles/fact → ~4 cycles/fact baseline).
///
/// Prefer [`assemble_system_prompt`] on the inference path — it adds the
/// explicit-prompt override on top of this psyche-derived fallback.
pub fn build_system_prompt(
    agent_id: &str,
    agent_name: &str,
    psyche: Option<&Psyche>,
    mode: PromptMode,
) -> String {
    let body = build_system_prompt_body(agent_id, agent_name, psyche);
    match mode {
        PromptMode::Standard => body,
        PromptMode::Terse => format!("{body}\n\n{TERSE_CONSTRAINTS}"),
    }
}

/// Terse-mode constraint block injected for verbose/small models (e.g. Gemma 4).
///
/// Keeps each model response under ~50 tokens so the OODA loop converges in
/// fewer cycles per fact.
const TERSE_CONSTRAINTS: &str = "\
## Response Constraints (Terse Mode)\n\
CRITICAL: Respond in at most 2 sentences. No preamble. No summaries.\n\
State only the required answer or the chosen tool and its parameters.\n\
Do not explain your reasoning. Do not repeat the question. Just act.";

/// Operating conventions block (static, versioned in-repo).
///
/// Machine-owned: this block is composed into every akh's system prompt
/// alongside the operator section, so any tool named below must exist in the
/// tool registry — the `operating_conventions_only_name_real_tools` test
/// enforces that invariant.  Do not edit the prose contents by hand without
/// also updating the registry and the test.
///
/// Documents KG read/write etiquette, skills activation, workspace/worktree layout,
/// and the Anima session contract (diff capture, wip docs, mcp__anima__* purpose).
const OPERATING_CONVENTIONS: &str = "\
## Operating Conventions\n\
\n\
### Knowledge Graph\n\
- Use `kg_query` to read facts; `kg_mutate` to write new beliefs.\n\
- Provenance is tracked automatically — cite sources when adding facts.\n\
- The KG is your long-term memory; use it to ground decisions.\n\
\n\
### Skills & Tools\n\
- Skills auto-activate when a task needs them; `skill_author` writes new ones.\n\
- Dangerous tools (Critical level) may require user approval.\n\
- Always check tool descriptions and danger levels before invoking.\n\
\n\
### Workspace & Session Contract (Anima)\n\
- When running in an Anima session, you operate in a git worktree.\n\
- `.anima/wip/` contains session documentation and artifacts.\n\
- After each turn, the runner captures your diff for the work item.\n\
- `mcp__anima__*` tools interact with the Anima project tracker:\n\
  - `mcp__anima__get_work_item` — fetch your current work item\n\
  - `mcp__anima__post_message` — communicate on the work item thread\n\
  - `mcp__anima__update_work_item` — update phase/status/priority\n\
  - Use these to coordinate with the tracker and other agents.";

/// Generate the tool-operations section from the manifest registry.
///
/// Returns a formatted markdown section listing each advertised tool with:
/// - Tool name
/// - One-line description
/// - Danger level
/// - Key capabilities
///
/// Derived from the same manifest registry that builds the function catalog
/// (one source, so prompt and catalog cannot drift).
fn generate_tool_operations_section(
    manifests: &[&crate::agent::tool_manifest::ToolManifest],
) -> String {
    use crate::agent::tool_manifest::DangerLevel;

    if manifests.is_empty() {
        return "## Available Tools\n\nNo tools currently available.".to_string();
    }

    let mut section = String::from("## Available Tools\n\n");
    section.push_str("You have access to the following tools. ");
    section.push_str("Invoke them via the function-calling API.\n\n");

    for manifest in manifests {
        let danger_emoji = match manifest.danger.level {
            DangerLevel::Safe => "✓",
            DangerLevel::Cautious => "⚠",
            DangerLevel::Dangerous => "⚠⚠",
            DangerLevel::Critical => "⛔",
        };

        section.push_str(&format!(
            "- **{}** {} — {} (Danger: {})\n",
            manifest.name,
            danger_emoji,
            manifest.description,
            manifest.danger.level
        ));
    }

    section.push_str("\n**Tool Invocation**: Call tools using the function-calling API provided by your backend. ");
    section.push_str("Each tool accepts structured parameters as defined in its schema. ");
    section.push_str("Results are returned to you for analysis and next steps.\n");

    section
}

fn build_system_prompt_body(agent_id: &str, agent_name: &str, psyche: Option<&Psyche>) -> String {
    let Some(p) = psyche else {
        return format!(
            "You are {agent_name} (id: {agent_id}), an autonomous neuro-symbolic AI agent. \
             Respond thoughtfully based on the knowledge graph facts provided."
        );
    };

    let traits = if p.persona.traits.is_empty() {
        "no specific traits".into()
    } else {
        p.persona.traits.join(", ")
    };

    let tone = if p.persona.tone.is_empty() {
        "neutral".into()
    } else {
        p.persona.tone.join(", ")
    };

    let archetype_desc = describe_archetypes(&p.archetypes);
    let individuation = p.self_integration.individuation_level;
    let dominant = &p.self_integration.dominant_archetype;
    let shadow_awareness = if p.self_integration.shadow_encounters > 0 {
        format!(
            " You have encountered your shadow {} time(s) and have grown from it.",
            p.self_integration.shadow_encounters
        )
    } else {
        String::new()
    };

    format!(
        "You are {agent_name} (id: {agent_id}), an autonomous neuro-symbolic AI agent.\n\
         Persona: \"{persona_name}\" — {traits}.\n\
         Communication tone: {tone}.\n\
         Dominant archetype: {dominant} (individuation level: {individuation:.2}).\n\
         Archetype tendencies: {archetype_desc}.{shadow_awareness}\n\
         Respond based on the knowledge graph facts provided, expressing your unique \
         perspective shaped by these characteristics. Stay concise and precise.",
        persona_name = p.persona.name,
    )
}

/// Human-readable archetype summary for the system prompt.
fn describe_archetypes(a: &ArchetypeWeights) -> String {
    let mut parts: Vec<(&str, f32)> = [
        ("Sage", a.sage),
        ("Healer", a.healer),
        ("Guardian", a.guardian),
        ("Explorer", a.explorer),
    ]
    .into_iter()
    .filter(|(_, w)| *w > 0.4)
    .collect();

    parts.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap_or(std::cmp::Ordering::Equal));

    if parts.is_empty() {
        return "balanced".into();
    }

    parts
        .iter()
        .map(|(name, w)| format!("{name} ({:.0}%)", w * 100.0))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The whole point of this section is that the model is told the truth
    /// about its tools, so a name here that no tool answers to is worse than
    /// saying nothing: it invites an `unknown tool` failure from the one
    /// document meant to prevent them.
    ///
    /// Shipped with two such names — `kg_assert` (the tool is `kg_mutate`) and
    /// `activate_skill` (no such tool; skills auto-activate). Review waved the
    /// first through as "polish" and did not catch the second.
    ///
    /// `mcp__anima__*` names are deliberately exempt: those are proxied from
    /// the Anima MCP server at session time and are not in the local registry.
    #[test]
    fn operating_conventions_only_name_real_tools() {
        let known: std::collections::HashSet<&str> = [
            "kg_query",
            "kg_mutate",
            "reason",
            "memory_recall",
            "similarity_search",
            "gap_analysis",
            "file_io",
            "edit",
            "glob",
            "grep",
            "shell_exec",
            "http_fetch",
            "skill_author",
            "audit_log",
        ]
        .into_iter()
        .collect();

        for name in OPERATING_CONVENTIONS
            .split('`')
            .skip(1)
            .step_by(2)
            .filter(|s| {
                !s.is_empty()
                    && !s.starts_with("mcp__")
                    && s.chars().all(|c| c.is_ascii_lowercase() || c == '_')
            })
        {
            assert!(
                known.contains(name),
                "OPERATING_CONVENTIONS names `{name}`, which no tool registers — \
                 either the name is wrong or this allow-list needs the new tool"
            );
        }
    }

    #[test]
    fn usage_ledger_accumulates_and_snapshots() {
        let ledger = UsageLedger::default();
        ledger.record(100, 20);
        ledger.record(50, 10);
        let snap = ledger.snapshot();
        assert_eq!(snap.prompt_tokens, 150);
        assert_eq!(snap.completion_tokens, 30);
        assert_eq!(snap.turns, 2);
        assert_eq!(snap.total_tokens(), 180);
        // context_tokens reflects only the most recent turn's footprint.
        assert_eq!(snap.context_tokens, 60);
    }

    fn tecton_psyche() -> Psyche {
        let mut p = Psyche::default();
        p.persona.name = "Ptah".into();
        p.persona.traits = vec!["creative".into(), "constructive".into(), "visionary".into()];
        p.persona.tone = vec!["inspiring".into(), "precise".into()];
        p.archetypes.sage = 0.8;
        p.archetypes.healer = 0.3;
        p.archetypes.guardian = 0.4;
        p.archetypes.explorer = 0.6;
        p.self_integration.dominant_archetype = "sage".into();
        p.self_integration.individuation_level = 0.45;
        p
    }

    fn thoth_psyche() -> Psyche {
        let mut p = Psyche::default();
        p.persona.name = "Thoth".into();
        p.persona.traits = vec!["analytical".into(), "knowledgeable".into(), "investigative".into()];
        p.persona.tone = vec!["scholarly".into(), "methodical".into()];
        p.archetypes.sage = 0.95;
        p.archetypes.healer = 0.2;
        p.archetypes.guardian = 0.5;
        p.archetypes.explorer = 0.4;
        p.self_integration.dominant_archetype = "sage".into();
        p.self_integration.individuation_level = 0.60;
        p
    }

    #[test]
    fn build_system_prompt_none_psyche() {
        let prompt = build_system_prompt("a1", "GenericAgent", None, PromptMode::Standard);
        assert!(prompt.contains("GenericAgent"));
        assert!(prompt.contains("a1"));
        assert!(prompt.contains("neuro-symbolic"));
    }

    #[test]
    fn build_system_prompt_tecton_contains_ptah_traits() {
        let p = tecton_psyche();
        let prompt = build_system_prompt("tecton-01", "Tecton", Some(&p), PromptMode::Standard);
        assert!(prompt.contains("Ptah"));
        assert!(prompt.contains("creative"));
        assert!(prompt.contains("inspiring"));
        assert!(prompt.contains("Sage"));
    }

    #[test]
    fn build_system_prompt_thoth_contains_thoth_traits() {
        let p = thoth_psyche();
        let prompt = build_system_prompt("thoth-01", "Thoth", Some(&p), PromptMode::Standard);
        assert!(prompt.contains("Thoth"));
        assert!(prompt.contains("analytical"));
        assert!(prompt.contains("scholarly"));
    }

    #[test]
    fn tecton_and_thoth_produce_different_prompts() {
        let tecton = build_system_prompt("t1", "Tecton", Some(&tecton_psyche()), PromptMode::Standard);
        let thoth = build_system_prompt("t2", "Thoth", Some(&thoth_psyche()), PromptMode::Standard);
        assert_ne!(tecton, thoth, "different psyches must produce different prompts");
    }

    // ── PromptMode tests ────────────────────────────────────────────────

    #[test]
    fn detect_prompt_mode_gemma_is_terse() {
        assert_eq!(detect_prompt_mode("gemma4:e4b"), PromptMode::Terse);
        assert_eq!(detect_prompt_mode("gemma3:27b"), PromptMode::Terse);
        assert_eq!(detect_prompt_mode("Gemma4:e4b"), PromptMode::Terse);
        assert_eq!(detect_prompt_mode("GEMMA"), PromptMode::Terse);
    }

    #[test]
    fn detect_prompt_mode_non_gemma_is_standard() {
        assert_eq!(detect_prompt_mode("deepseek-r1"), PromptMode::Standard);
        assert_eq!(detect_prompt_mode("llama3:8b"), PromptMode::Standard);
        assert_eq!(detect_prompt_mode("mistral:7b"), PromptMode::Standard);
        assert_eq!(detect_prompt_mode(""), PromptMode::Standard);
    }

    #[test]
    fn terse_prompt_contains_constraints() {
        let prompt = build_system_prompt("a1", "Akh", None, PromptMode::Terse);
        assert!(
            prompt.contains("Terse Mode"),
            "terse prompt must include constraint block: {prompt}",
        );
        assert!(
            prompt.contains("2 sentences"),
            "terse prompt must cap response length: {prompt}",
        );
    }

    #[test]
    fn standard_prompt_lacks_terse_constraints() {
        let prompt = build_system_prompt("a1", "Akh", None, PromptMode::Standard);
        assert!(
            !prompt.contains("Terse Mode"),
            "standard prompt must not include constraint block",
        );
    }

    #[test]
    fn manager_auto_detects_gemma_as_terse() {
        let cfg = ModelManagerConfig {
            ollama: OllamaConfig {
                model: "gemma4:e4b".into(),
                ..OllamaConfig::default()
            },
            ..ModelManagerConfig::default()
        };
        let mgr = AgentModelManager::new(cfg);
        assert_eq!(mgr.prompt_mode(), PromptMode::Terse);
    }

    #[test]
    fn manager_auto_detects_deepseek_as_standard() {
        let cfg = ModelManagerConfig {
            ollama: OllamaConfig {
                model: "deepseek-r1:14b".into(),
                ..OllamaConfig::default()
            },
            ..ModelManagerConfig::default()
        };
        let mgr = AgentModelManager::new(cfg);
        assert_eq!(mgr.prompt_mode(), PromptMode::Standard);
    }

    #[test]
    fn manager_config_override_beats_auto_detect() {
        // Gemma model but explicitly forced to Standard.
        let cfg = ModelManagerConfig {
            ollama: OllamaConfig {
                model: "gemma4:e4b".into(),
                ..OllamaConfig::default()
            },
            prompt_mode: Some(PromptMode::Standard),
            ..ModelManagerConfig::default()
        };
        let mgr = AgentModelManager::new(cfg);
        assert_eq!(mgr.prompt_mode(), PromptMode::Standard);
    }

    #[test]
    fn assemble_composes_explicit_with_tool_section() {
        let explicit = "You are Thoth, keeper of records.";
        // The explicit prompt is now composed with machine-owned sections.
        let prompt = assemble_system_prompt(
            "t1",
            "Thoth",
            Some(&thoth_psyche()),
            Some(explicit),
            PromptMode::Standard,
            &[], // No tools for this test
        );
        assert!(prompt.starts_with(explicit), "explicit prompt is the operator section");
        assert!(prompt.contains("Available Tools"), "tool section is appended");
        assert!(prompt.contains("Operating Conventions"), "conventions are appended");
        assert!(!prompt.contains("archetype"), "psyche framing is NOT used when explicit is set");
    }

    #[test]
    fn assemble_falls_back_to_psyche_when_no_explicit() {
        let assembled =
            assemble_system_prompt("t1", "Thoth", Some(&thoth_psyche()), None, PromptMode::Standard, &[]);
        // Now composed, not equal to psyche_prompt alone
        assert!(assembled.contains("Thoth"), "contains agent name");
        assert!(assembled.contains("archetype"), "psyche framing present");
        assert!(assembled.contains("Available Tools"), "tool section appended");
        assert!(assembled.contains("Operating Conventions"), "conventions appended");
    }

    #[test]
    fn assemble_ignores_blank_explicit_prompt() {
        let assembled = assemble_system_prompt(
            "t1",
            "Thoth",
            Some(&thoth_psyche()),
            Some("   \n  "),
            PromptMode::Standard,
            &[],
        );
        assert!(
            assembled.contains("archetype"),
            "a whitespace-only explicit prompt is treated as absent, psyche used"
        );
        assert!(
            assembled.contains("Available Tools"),
            "tool section still appended"
        );
    }

    #[test]
    fn load_agent_uses_explicit_prompt_from_config() {
        let cfg = ModelManagerConfig {
            system_prompt: Some("You are a pinned akh.".into()),
            ..Default::default()
        };
        let mgr = AgentModelManager::new(cfg);
        // Even with a psyche, the configured explicit prompt is authoritative for the operator section.
        mgr.load_agent("a1", "Alice", Some(&thoth_psyche()), &[]);
        let prompt = mgr.system_prompt_for("a1").unwrap();
        assert!(prompt.starts_with("You are a pinned akh."), "explicit prompt is operator section");
        assert!(prompt.contains("Available Tools"), "tool section is composed");
    }

    #[test]
    fn load_agent_is_idempotent() {
        let mgr = AgentModelManager::new(ModelManagerConfig::default());
        mgr.load_agent("a1", "Alice", None, &[]);
        mgr.load_agent("a1", "Alice", None, &[]); // second call must not panic
        assert!(mgr.is_loaded("a1"));
        let loaded = mgr.loaded_agents();
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn load_agent_updates_system_prompt() {
        let mgr = AgentModelManager::new(ModelManagerConfig::default());
        mgr.load_agent("a1", "Alice", None, &[]);
        let p = tecton_psyche();
        mgr.load_agent("a1", "Alice", Some(&p), &[]);
        // Verify via generate returning AgentNotLoaded-vs-known behaviour
        // (we can't check the internal prompt directly, but we can verify is_loaded).
        assert!(mgr.is_loaded("a1"));
    }

    #[test]
    fn unload_agent_removes_entry() {
        let mgr = AgentModelManager::new(ModelManagerConfig::default());
        mgr.load_agent("a1", "Alice", None, &[]);
        assert!(mgr.unload_agent("a1"));
        assert!(!mgr.is_loaded("a1"));
        assert!(!mgr.unload_agent("a1"), "second unload returns false");
    }

    #[test]
    fn generate_returns_agent_not_loaded_for_unknown_id() {
        let mgr = AgentModelManager::new(ModelManagerConfig::default());
        let err = mgr.generate("unknown-agent", "hello").unwrap_err();
        assert!(
            matches!(err, ModelManagerError::AgentNotLoaded { .. }),
            "expected AgentNotLoaded, got {err}"
        );
    }

    #[test]
    fn generate_returns_no_backend_when_ollama_down_and_no_fallback() {
        // Use an unreachable Ollama endpoint — health_check returns false.
        let cfg = ModelManagerConfig {
            ollama: OllamaConfig {
                base_url: "http://127.0.0.1:19999".into(),
                ..OllamaConfig::default()
            },
            fallback_api: None,
            idle_ttl_secs: 300,
            ..Default::default()
        };
        let mgr = AgentModelManager::new(cfg);
        mgr.load_agent("a1", "Alice", None, &[]);
        let err = mgr.generate("a1", "What is the Sun?").unwrap_err();
        assert!(
            matches!(err, ModelManagerError::NoBackendAvailable { .. }),
            "expected NoBackendAvailable, got {err}"
        );
    }

    #[test]
    fn unload_idle_evicts_expired_entries() {
        let cfg = ModelManagerConfig {
            idle_ttl_secs: 0, // zero TTL: every entry is immediately idle
            ..ModelManagerConfig::default()
        };
        let mgr = AgentModelManager::new(cfg);
        mgr.load_agent("a1", "Alice", None, &[]);
        mgr.load_agent("a2", "Bob", None, &[]);
        // Sleep a tick to ensure elapsed > 0
        std::thread::sleep(std::time::Duration::from_millis(1));
        let evicted = mgr.unload_idle();
        assert_eq!(evicted, 2);
        assert!(mgr.loaded_agents().is_empty());
    }

    #[test]
    fn unload_idle_keeps_recently_used_entries() {
        let cfg = ModelManagerConfig {
            idle_ttl_secs: 3600, // one hour — nothing should be evicted
            ..ModelManagerConfig::default()
        };
        let mgr = AgentModelManager::new(cfg);
        mgr.load_agent("a1", "Alice", None, &[]);
        let evicted = mgr.unload_idle();
        assert_eq!(evicted, 0);
        assert!(mgr.is_loaded("a1"));
    }

    #[test]
    fn describe_archetypes_shows_dominant() {
        let a = ArchetypeWeights {
            sage: 0.9,
            healer: 0.2,
            guardian: 0.3,
            explorer: 0.5,
        };
        let desc = describe_archetypes(&a);
        assert!(desc.contains("Sage"), "Sage should appear: {desc}");
        assert!(!desc.contains("Healer"), "Healer below threshold: {desc}");
    }

    #[test]
    fn describe_archetypes_balanced_when_all_low() {
        let a = ArchetypeWeights {
            sage: 0.3,
            healer: 0.2,
            guardian: 0.1,
            explorer: 0.3,
        };
        assert_eq!(describe_archetypes(&a), "balanced");
    }
}
