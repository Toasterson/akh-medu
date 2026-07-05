//! Pluggable chat/agentic backing for [`Agent::run_agentic_turn`](super::Agent::run_agentic_turn).
//!
//! An akh picks its backing in `akh.toml` (ADR 070): **Ollama** (the common one
//! — local or `ollama.com` cloud, native function-calling) or **Candle** (an
//! in-process GGUF model; device-limited — GPU helps but it runs CPU-only too).
//! Both are *full tool-calling peers*: the loop is backend-agnostic because both
//! return the same [`AssistantTurn`] `{content, tool_calls}` contract.
//!
//! Candle's tool calling is grammar-constrained: because we own the sampling
//! loop, tool-call arguments are generated under a JSON grammar
//! ([`CandleBackend::generate_constrained_json`](crate::nlu::candle_backend::CandleBackend::generate_constrained_json),
//! `tool_call.gbnf`) so a call is never dropped to malformed JSON — the model
//! supplies the semantics, the grammar guarantees the syntax.
//!
//! This is the precursor to Phase 39's full `InferenceBackend` registry/router
//! (ADR 037); the trait is intentionally shaped to forward-port cleanly.

use std::sync::Arc;

use miette::Diagnostic;
use serde_json::Value;
use thiserror::Error;

use super::ollama::{AssistantTurn, StreamEvent};

/// Errors a chat backing can surface. Miette-diagnosable per CLAUDE.md.
#[derive(Debug, Error, Diagnostic)]
pub enum ChatBackendError {
    /// The backing is configured but unreachable (Ollama down, model not loaded).
    #[error("chat backend \"{backend}\" is unavailable")]
    #[diagnostic(
        code(akh::chat_backend::unavailable),
        help(
            "For Ollama: start it (`ollama serve`) or check the akh's endpoint. \
             For Candle: ensure a GGUF model is present under the data dir."
        )
    )]
    Unavailable { backend: String },

    /// The backing was reached but the turn failed.
    #[error("chat backend \"{backend}\" turn failed: {reason}")]
    #[diagnostic(code(akh::chat_backend::turn_failed))]
    TurnFailed { backend: String, reason: String },
}

/// What a model can do, read from the model itself — not guessed.
///
/// Populated from the GGUF metadata for Candle (`<arch>.context_length`) and from
/// Ollama's `/api/show` for the Ollama backing (`capabilities` + `model_info`).
/// Consumed by the context-assembly budget (Phase 44 — `context_window`) and to
/// gate the reasoning-effort (`think`) flag (`supports_thinking`).
#[derive(Debug, Clone, Default)]
pub struct ModelCapabilities {
    /// Context window in tokens, if known (e.g. gemma4-12B = 262144,
    /// glm-5.2 = 1_000_000, gpt-oss:120b = 131072).
    pub context_window: Option<usize>,
    /// The model accepts a reasoning-effort / `think` request.
    pub supports_thinking: bool,
    /// The model supports tool / function calling.
    pub supports_tools: bool,
}

/// A chat/agentic LLM backing for the agentic turn loop.
///
/// Implementors translate a tool-enabled conversation into an [`AssistantTurn`]
/// (final content and/or tool calls). The agentic loop in
/// [`super::agentic_chat`] drives this trait object regardless of which backend
/// an akh configured.
pub trait ChatBackend: Send + Sync {
    /// Short label for logs/audit (e.g. `"ollama"`, `"candle"`).
    fn label(&self) -> &str;

    /// Whether the backing is reachable right now (health-gated fallback to the
    /// symbolic pipeline when `false`).
    fn available(&self) -> bool;

    /// Refresh per-akh state (e.g. the system prompt) before a turn. Default:
    /// no-op.
    fn prepare(&self, _agent_id: &str, _psyche: Option<&crate::compartment::psyche::Psyche>) {}

    /// The model's capabilities, read from the model (GGUF metadata / Ollama
    /// `/api/show`). Default: unknown. Consumed by the Phase 44 context budget
    /// (window) and reasoning-effort gating.
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::default()
    }

    /// One tool-enabled chat turn: the model answers `conversation`, optionally
    /// requesting tools from `tools` (OpenAI-style function schemas).
    fn chat_with_tools(
        &self,
        agent_id: &str,
        conversation: &[Value],
        tools: &[Value],
    ) -> Result<AssistantTurn, ChatBackendError>;

    /// Streaming counterpart. Default: run [`chat_with_tools`](Self::chat_with_tools)
    /// and emit the assembled content as one [`StreamEvent::Content`] (backends
    /// without native streaming still satisfy the loop, which also has a
    /// non-streaming fallback).
    fn chat_with_tools_streaming(
        &self,
        agent_id: &str,
        conversation: &[Value],
        tools: &[Value],
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<AssistantTurn, ChatBackendError> {
        let turn = self.chat_with_tools(agent_id, conversation, tools)?;
        if !turn.content.is_empty() {
            on_event(StreamEvent::Content(turn.content.clone()));
        }
        Ok(turn)
    }
}

// ── Ollama backing ──────────────────────────────────────────────────────────

/// The Ollama backing: wraps the existing [`AgentModelManager`](super::model_manager::AgentModelManager)
/// (local or `ollama.com` cloud, native function-calling). Behaviour is exactly
/// what ADR 062 shipped — this is just the trait face over it.
pub struct OllamaChatBackend {
    mgr: Arc<super::model_manager::AgentModelManager>,
}

impl OllamaChatBackend {
    pub fn new(mgr: Arc<super::model_manager::AgentModelManager>) -> Self {
        Self { mgr }
    }
}

impl ChatBackend for OllamaChatBackend {
    fn label(&self) -> &str {
        "ollama"
    }

    fn available(&self) -> bool {
        self.mgr.ollama_available()
    }

    fn prepare(&self, agent_id: &str, psyche: Option<&crate::compartment::psyche::Psyche>) {
        // Refresh the akh's cached system prompt from its current psyche.
        self.mgr.load_agent(agent_id, agent_id, psyche);
    }

    fn capabilities(&self) -> ModelCapabilities {
        self.mgr.capabilities()
    }

    fn chat_with_tools(
        &self,
        agent_id: &str,
        conversation: &[Value],
        tools: &[Value],
    ) -> Result<AssistantTurn, ChatBackendError> {
        self.mgr
            .chat_with_tools(agent_id, conversation, tools)
            .map_err(|e| ChatBackendError::TurnFailed {
                backend: "ollama".into(),
                reason: e.to_string(),
            })
    }

    fn chat_with_tools_streaming(
        &self,
        agent_id: &str,
        conversation: &[Value],
        tools: &[Value],
        on_event: &mut dyn FnMut(StreamEvent),
    ) -> Result<AssistantTurn, ChatBackendError> {
        self.mgr
            .chat_with_tools_streaming(agent_id, conversation, tools, on_event)
            .map_err(|e| ChatBackendError::TurnFailed {
                backend: "ollama".into(),
                reason: e.to_string(),
            })
    }
}

// ── Candle backing ───────────────────────────────────────────────────────────

/// The Candle backing: an in-process GGUF model with **full, grammar-constrained
/// tool calling** (ADR 070). Device-limited — runs CPU-only, uses the GPU when
/// the `rocm`/`metal`/`cuda` Candle device is available.
#[cfg(feature = "candle-backend")]
pub struct CandleChatBackend {
    backend: Arc<std::sync::Mutex<crate::nlu::candle_backend::CandleBackend>>,
    system_prompt: String,
}

#[cfg(feature = "candle-backend")]
impl CandleChatBackend {
    /// Wrap a loaded Candle backend with the akh's authoritative system prompt.
    pub fn new(
        backend: Arc<std::sync::Mutex<crate::nlu::candle_backend::CandleBackend>>,
        system_prompt: String,
    ) -> Self {
        Self {
            backend,
            system_prompt,
        }
    }
}

#[cfg(feature = "candle-backend")]
impl ChatBackend for CandleChatBackend {
    fn label(&self) -> &str {
        "candle"
    }

    fn available(&self) -> bool {
        self.backend.lock().is_ok()
    }

    fn capabilities(&self) -> ModelCapabilities {
        // Context window from the GGUF metadata; Gemma/Qwen GGUFs here support
        // tool calling via the family template but are not "thinking" models.
        let context_window = self.backend.lock().ok().map(|b| b.context_length());
        ModelCapabilities {
            context_window,
            supports_thinking: false,
            supports_tools: true,
        }
    }

    fn chat_with_tools(
        &self,
        _agent_id: &str,
        conversation: &[Value],
        tools: &[Value],
    ) -> Result<AssistantTurn, ChatBackendError> {
        let mut be = self.backend.lock().map_err(|_| ChatBackendError::Unavailable {
            backend: "candle".into(),
        })?;
        let family = be.family();
        let prompt = render_tool_prompt(&self.system_prompt, conversation, tools, family);

        // Free-form generation: the model emits either an answer or a `<tool_call>` block. Use
        // the RAW (uncleaned) text — Gemma 4 puts its tool call in the thought channel, which the
        // answer-channel cleaning would strip before we can parse it.
        let raw = be
            .generate_raw(&prompt, 512)
            .map_err(|e| ChatBackendError::TurnFailed {
                backend: "candle".into(),
                reason: e.to_string(),
            })?;

        let (content, mut tool_calls) = parse_tool_calls(&raw);
        // When there is no tool call the content is the answer — clean it (answer channel only,
        // markers stripped) for display, matching the non-agentic `generate` path.
        let content = crate::nlu::channel::ChannelStream::parse(&content).answer();

        // Opt-in trace of the raw agentic generation (AKH_AGENT_DEBUG=1): the exact text the
        // model produced and how many tool calls were parsed from it. Invaluable for diagnosing
        // tool-call format mismatches without a full tracing subscriber.
        if std::env::var("AKH_AGENT_DEBUG").as_deref() == Ok("1") {
            eprintln!(
                "[candle agentic] parsed {} tool_call(s); raw generation =\n---\n{raw}\n---",
                tool_calls.len()
            );
        }

        // If the model opened a tool call but the JSON didn't parse, regenerate
        // the arguments under the JSON grammar constraint — guaranteed valid.
        if tool_calls.is_empty() && raw.contains("<tool_call>") {
            let forced = format!("{prompt}<tool_call>\n");
            if let Ok(json) = be.generate_constrained_json(&forced, 256) {
                if let Some(tc) = tool_call_from_json(json.trim()) {
                    tool_calls.push(tc);
                }
            }
        }

        Ok(AssistantTurn {
            content: if tool_calls.is_empty() { content } else { String::new() },
            tool_calls,
            ..Default::default()
        })
    }
}

/// Render a tool-enabled chat prompt in the loaded model's native template.
///
/// Tool schemas go in the system turn with the family's tool-call convention;
/// the conversation (user / assistant / tool messages) follows; the prompt ends
/// on an open assistant turn.
#[cfg(feature = "candle-backend")]
fn render_tool_prompt(
    system_prompt: &str,
    conversation: &[Value],
    tools: &[Value],
    family: crate::nlu::candle_backend::ModelFamily,
) -> String {
    use crate::nlu::candle_backend::ModelFamily;

    let tools_json = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".into());
    let sys = format!(
        "{system_prompt}\n\n# Tools\nYou can call functions. Available tools (JSON schema):\n{tools_json}\n\nTo call a tool, reply with exactly:\n<tool_call>\n{{\"name\": <tool-name>, \"arguments\": <args-object>}}\n</tool_call>\nOtherwise, answer directly."
    );

    let (sys_open, sys_close, user_open, asst_open, turn_close) = match family {
        ModelFamily::Qwen2 => (
            "<|im_start|>system\n",
            "<|im_end|>\n",
            "<|im_start|>user\n",
            "<|im_start|>assistant\n",
            "<|im_end|>\n",
        ),
        ModelFamily::Gemma | ModelFamily::Gemma4 => (
            "<start_of_turn>user\n",
            "<end_of_turn>\n",
            "<start_of_turn>user\n",
            "<start_of_turn>model\n",
            "<end_of_turn>\n",
        ),
    };

    let mut p = String::new();
    p.push_str(sys_open);
    p.push_str(&sys);
    p.push_str(sys_close);
    for msg in conversation {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = msg.get("content").and_then(Value::as_str).unwrap_or("");
        match role {
            "assistant" => {
                p.push_str(asst_open);
                p.push_str(content);
                p.push_str(turn_close);
            }
            "tool" => {
                // Tool result fed back to the model.
                p.push_str(user_open);
                p.push_str("<tool_response>\n");
                p.push_str(content);
                p.push_str("\n</tool_response>");
                p.push_str(turn_close);
            }
            _ => {
                p.push_str(user_open);
                p.push_str(content);
                p.push_str(turn_close);
            }
        }
    }
    p.push_str(asst_open);
    p
}

/// Extract `<tool_call>{json}</tool_call>` blocks from a model turn.
///
/// Returns `(content, tool_calls)` where `content` is the text outside any tool
/// call. A block whose JSON doesn't parse is left for the constrained-repair
/// path (it won't appear in `tool_calls`).
#[cfg(feature = "candle-backend")]
fn parse_tool_calls(raw: &str) -> (String, Vec<super::ollama::ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut content = String::new();
    let mut rest = raw;
    while let Some(start) = rest.find("<tool_call>") {
        content.push_str(&rest[..start]);
        let after = &rest[start + "<tool_call>".len()..];
        let (block, tail) = match after.find("</tool_call>") {
            Some(end) => (&after[..end], &after[end + "</tool_call>".len()..]),
            None => (after, ""),
        };
        if let Some(tc) = tool_call_from_json(block.trim()) {
            tool_calls.push(tc);
        }
        rest = tail;
    }
    content.push_str(rest);
    (content.trim().to_string(), tool_calls)
}

/// Parse one `{"name": ..., "arguments": {...}}` object into a [`ToolCall`].
#[cfg(feature = "candle-backend")]
fn tool_call_from_json(s: &str) -> Option<super::ollama::ToolCall> {
    let v: Value = serde_json::from_str(s).ok()?;
    let name = v.get("name").and_then(Value::as_str)?.to_string();
    let arguments = v
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    Some(super::ollama::ToolCall { name, arguments })
}

#[cfg(all(test, feature = "candle-backend"))]
mod tests {
    use super::*;

    #[test]
    fn parses_a_tool_call_block() {
        let raw = "Let me check.<tool_call>\n{\"name\": \"kg_query\", \"arguments\": {\"q\": \"Sun\"}}\n</tool_call>";
        let (content, calls) = parse_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "kg_query");
        assert_eq!(calls[0].arguments["q"], "Sun");
        assert_eq!(content, "Let me check.");
    }

    #[test]
    fn plain_answer_has_no_tool_calls() {
        let (content, calls) = parse_tool_calls("The Sun is a star.");
        assert!(calls.is_empty());
        assert_eq!(content, "The Sun is a star.");
    }

    #[test]
    fn malformed_json_block_is_skipped_for_repair() {
        let raw = "<tool_call>\n{\"name\": \"x\", \"arguments\": {oops}}\n</tool_call>";
        let (_content, calls) = parse_tool_calls(raw);
        assert!(calls.is_empty());
    }
}
