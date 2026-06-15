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
use std::time::{Duration, Instant};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::compartment::psyche::{ArchetypeWeights, Psyche};
use crate::extraction::config::ExternalApiConfig;

use super::ollama::{OllamaClient, OllamaConfig, OllamaError};

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
pub struct AgentModelManager {
    config: ModelManagerConfig,
    ollama: OllamaClient,
    agents: Mutex<HashMap<String, AgentEntry>>,
    /// Resolved prompt mode for this manager (auto-detected or config-overridden).
    prompt_mode: PromptMode,
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
        }
    }

    /// Return the prompt mode in effect for this manager.
    pub fn prompt_mode(&self) -> PromptMode {
        self.prompt_mode
    }

    // ── Registration ───────────────────────────────────────────────────────

    /// Register (or refresh) an agent with its Psyche encoding.
    ///
    /// Idempotent: if the agent is already loaded, its system prompt is
    /// refreshed from the new psyche and `last_used` is bumped.
    ///
    /// Callers can pass `None` for `psyche` to get a generic system prompt
    /// keyed only on the agent's name.
    pub fn load_agent(&self, agent_id: &str, agent_name: &str, psyche: Option<&Psyche>) {
        let prompt = assemble_system_prompt(
            agent_id,
            agent_name,
            psyche,
            self.config.system_prompt.as_deref(),
            self.prompt_mode,
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
/// Precedence (ADR-053):
/// 1. An explicit, non-empty `explicit` prompt (from `AkhConfig.system_prompt`)
///    is returned **verbatim** — it fully replaces the psyche-derived text, with
///    no archetype/persona framing appended.
/// 2. Otherwise the psyche-derived prompt from [`build_system_prompt`] is used.
///
/// This collapses the previously-scattered prompt-assembly logic so callers on
/// the inference path go through exactly one function.
pub fn assemble_system_prompt(
    agent_id: &str,
    agent_name: &str,
    psyche: Option<&Psyche>,
    explicit: Option<&str>,
    mode: PromptMode,
) -> String {
    if let Some(text) = explicit {
        if !text.trim().is_empty() {
            // Authoritative: returned verbatim. `mode` (incl. Terse brevity) is
            // intentionally NOT applied — the operator owns the full prompt.
            return text.to_string();
        }
    }
    build_system_prompt(agent_id, agent_name, psyche, mode)
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
    fn assemble_prefers_explicit_prompt_verbatim() {
        let explicit = "You are Thoth, keeper of records.";
        // Even with a rich psyche present, the explicit prompt wins verbatim.
        let prompt = assemble_system_prompt(
            "t1",
            "Thoth",
            Some(&thoth_psyche()),
            Some(explicit),
            PromptMode::Standard,
        );
        assert_eq!(prompt, explicit);
        assert!(!prompt.contains("archetype"), "no psyche framing is appended");
    }

    #[test]
    fn assemble_falls_back_to_psyche_when_no_explicit() {
        let psyche_prompt =
            build_system_prompt("t1", "Thoth", Some(&thoth_psyche()), PromptMode::Standard);
        let assembled =
            assemble_system_prompt("t1", "Thoth", Some(&thoth_psyche()), None, PromptMode::Standard);
        assert_eq!(assembled, psyche_prompt);
    }

    #[test]
    fn assemble_ignores_blank_explicit_prompt() {
        let psyche_prompt =
            build_system_prompt("t1", "Thoth", Some(&thoth_psyche()), PromptMode::Standard);
        let assembled = assemble_system_prompt(
            "t1",
            "Thoth",
            Some(&thoth_psyche()),
            Some("   \n  "),
            PromptMode::Standard,
        );
        assert_eq!(
            assembled, psyche_prompt,
            "a whitespace-only explicit prompt is treated as absent"
        );
    }

    #[test]
    fn load_agent_uses_explicit_prompt_from_config() {
        let cfg = ModelManagerConfig {
            system_prompt: Some("You are a pinned akh.".into()),
            ..Default::default()
        };
        let mgr = AgentModelManager::new(cfg);
        // Even with a psyche, the configured explicit prompt is authoritative.
        mgr.load_agent("a1", "Alice", Some(&thoth_psyche()));
        assert_eq!(
            mgr.system_prompt_for("a1").as_deref(),
            Some("You are a pinned akh.")
        );
    }

    #[test]
    fn load_agent_is_idempotent() {
        let mgr = AgentModelManager::new(ModelManagerConfig::default());
        mgr.load_agent("a1", "Alice", None);
        mgr.load_agent("a1", "Alice", None); // second call must not panic
        assert!(mgr.is_loaded("a1"));
        let loaded = mgr.loaded_agents();
        assert_eq!(loaded.len(), 1);
    }

    #[test]
    fn load_agent_updates_system_prompt() {
        let mgr = AgentModelManager::new(ModelManagerConfig::default());
        mgr.load_agent("a1", "Alice", None);
        let p = tecton_psyche();
        mgr.load_agent("a1", "Alice", Some(&p));
        // Verify via generate returning AgentNotLoaded-vs-known behaviour
        // (we can't check the internal prompt directly, but we can verify is_loaded).
        assert!(mgr.is_loaded("a1"));
    }

    #[test]
    fn unload_agent_removes_entry() {
        let mgr = AgentModelManager::new(ModelManagerConfig::default());
        mgr.load_agent("a1", "Alice", None);
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
        mgr.load_agent("a1", "Alice", None);
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
        mgr.load_agent("a1", "Alice", None);
        mgr.load_agent("a2", "Bob", None);
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
        mgr.load_agent("a1", "Alice", None);
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
