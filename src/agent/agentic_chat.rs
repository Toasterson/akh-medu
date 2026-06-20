//! LLM-driven agentic chat turn: the akh's pinned model (Ollama, local or
//! cloud) answers a prompt, calling registered tools in a loop until it produces
//! a final answer.
//!
//! This is the live half of the "first-class akh" picture (ADR-053 / the parked
//! P2 work): [`ChatProcessor`](crate::chat::ChatProcessor) is the *symbolic*
//! responder; this is the *neural* one. [`Agent::run_agentic_turn`] is called
//! from the ACP `session/prompt` handler and falls back to the symbolic pipeline
//! when no model backend is reachable, so a daemon with no Ollama configured
//! behaves exactly as before.
//!
//! ## Concurrency
//!
//! The whole turn is synchronous — blocking Ollama HTTP (`ureq`) interleaved
//! with synchronous tool execution. Callers run it inside
//! [`tokio::task::spawn_blocking`] holding the agent `Mutex` (ADR-027); no guard
//! crosses an `.await`. The per-turn permission asker, if any, must already be
//! installed on the tool registry (`Agent::set_permission_asker`) so `Ask` tools
//! can bridge to the ACP client.

use serde_json::{Value, json};

use super::Agent;
use super::ollama::StreamEvent;
use super::tool::{ToolInput, ToolRegistry};
use crate::engine::Engine;
use crate::message::{AkhMessage, MessageSink};

/// Hard ceiling on tool-call rounds in one turn — a backstop against a model
/// that loops calling tools without ever answering.
const DEFAULT_MAX_ROUNDS: usize = 8;

/// Characters of a tool-call argument blob surfaced in the "calling tool" line.
const ARG_PREVIEW_LEN: usize = 160;

impl Agent {
    /// Run one LLM-driven agentic turn, **streaming** its narration to `sink`.
    ///
    /// The akh's pinned model answers `user_prompt`, optionally calling any
    /// registered tool (gated by the permission policy) in a loop until it
    /// returns a final answer or [`DEFAULT_MAX_ROUNDS`] is reached. Rather than
    /// buffering, each piece of the turn is emitted to `sink` the instant it is
    /// produced: answer text arrives as [`AkhMessage::Narrative`] deltas as the
    /// model generates, chain-of-thought as [`AkhMessage::Reasoning`] deltas, and
    /// each tool call + result as a collapsible [`AkhMessage::ToolResult`] step.
    ///
    /// Returns `Some(())` when the turn was handled (answer emitted, or an error
    /// surfaced through `sink`), or `None` when there is **no usable model
    /// backend** — no model manager attached, or Ollama unreachable — signalling
    /// the caller to fall back to the symbolic
    /// [`ChatProcessor`](crate::chat::ChatProcessor). `agent_id` keys the model
    /// manager's per-akh system-prompt cache.
    ///
    /// Each round tries the streaming Ollama path and, on a transport/HTTP
    /// failure, falls back to one non-streaming `chat` call for that round (whose
    /// text is then emitted as a single message) so a streaming hiccup degrades
    /// gracefully instead of dropping the turn.
    ///
    /// ## Concurrency
    ///
    /// `sink.emit` is synchronous and non-blocking (it hands the message to a
    /// channel / notification send), so emitting from inside the
    /// [`spawn_blocking`](tokio::task::spawn_blocking) turn while the agent
    /// `Mutex` is held does not violate ADR-027 — no guard ever crosses an
    /// `.await`.
    pub fn run_agentic_turn(
        &mut self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
        sink: &dyn MessageSink,
    ) -> Option<()> {
        // Resolve the akh's chat/agentic backing (ADR 070): an explicitly
        // configured backend (e.g. Candle) wins; otherwise wrap the Ollama model
        // manager. Either way the loop is backend-agnostic. `None` ⇒ no usable
        // backend, so the caller falls back to the symbolic ChatProcessor.
        let backend: std::sync::Arc<dyn super::chat_backend::ChatBackend> =
            match self.chat_backend.clone() {
                Some(b) => b,
                None => {
                    let mgr = self.model_manager.clone()?;
                    std::sync::Arc::new(super::chat_backend::OllamaChatBackend::new(mgr))
                }
            };
        if !backend.available() {
            return None;
        }

        // Refresh per-akh state (e.g. the Ollama system-prompt cache) from psyche.
        backend.prepare(agent_id, self.psyche.as_ref());

        let mut tools = ollama_tool_catalog(&self.tool_registry);
        // Optional cap on the tool catalog (AKH_AGENT_MAX_TOOLS). The full catalog
        // is several thousand tokens — too large for a small-context in-process
        // model (e.g. Candle Gemma on a constrained host), where it dominates
        // prefill cost. `0` runs the akh tool-free (pure chat). A precursor to the
        // Phase 44 budget-driven tool selection.
        if let Some(cap) = std::env::var("AKH_AGENT_MAX_TOOLS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            && cap < tools.len()
        {
            tracing::debug!(cap, from = tools.len(), "capping tool catalog (AKH_AGENT_MAX_TOOLS)");
            tools.truncate(cap);
        }

        // Phase 44a (ADR 065): assemble the working context through the akh's
        // `ContextManager` instead of building the prompt ad hoc. The default
        // pass-through manager reproduces the single-user-turn behaviour
        // byte-for-byte; richer managers (history, KG, recall) slot in here
        // without the loop changing. The loop then appends assistant/tool turns
        // to this base across rounds, exactly as before.
        let mut conversation: Vec<Value> = {
            use super::context::{
                ContextManager, EpisodeState, HeuristicTokenCounter, PassThroughManager,
                TokenBudget, TokenCounter,
            };
            static PASS_THROUGH: PassThroughManager = PassThroughManager;
            let counter = HeuristicTokenCounter;
            // TODO(44b): size the window from the akh's model config once measured.
            let budget = TokenBudget::default();
            let episode = EpisodeState {
                agent_id,
                user_prompt,
                engine,
                conversation: Some(self.conversation_state()),
            };
            let manager: &dyn ContextManager =
                self.context_manager.as_deref().unwrap_or(&PASS_THROUGH);
            match manager.assemble(&episode, &budget, &counter) {
                Ok(assembled) => {
                    super::context::budget::record_assembly_telemetry(
                        &manager.id(),
                        &assembled,
                        &budget,
                        counter.label(),
                    );
                    assembled.messages
                }
                Err(e) => {
                    tracing::error!(
                        agent_id,
                        error = %e,
                        "context assembly failed; using single-turn fallback"
                    );
                    vec![json!({"role": "user", "content": user_prompt})]
                }
            }
        };

        for _round in 0..DEFAULT_MAX_ROUNDS {
            // Stream this round. Content deltas become live answer text, thinking
            // deltas become live reasoning. `used_fallback` records whether the
            // streaming path failed and we silently retried non-streaming — in
            // which case nothing was streamed and we emit the result ourselves.
            let mut used_fallback = false;
            let turn = {
                let mut on_event = |ev: StreamEvent| match ev {
                    StreamEvent::Content(delta) => sink.emit(&AkhMessage::narrative(delta, "acp")),
                    StreamEvent::Thinking(delta) => sink.emit(&AkhMessage::Reasoning {
                        step: delta,
                        expression: None,
                    }),
                };
                match backend.chat_with_tools_streaming(agent_id, &conversation, &tools, &mut on_event) {
                    Ok(t) => t,
                    Err(stream_err) => {
                        tracing::warn!(
                            agent_id,
                            "streaming chat failed ({stream_err}); falling back to non-streaming"
                        );
                        used_fallback = true;
                        match backend.chat_with_tools(agent_id, &conversation, &tools) {
                            Ok(t) => t,
                            Err(e) => {
                                sink.emit(&AkhMessage::error("llm", e.to_string()));
                                return Some(());
                            }
                        }
                    }
                }
            };

            // Echo the assistant turn (content + tool calls) back into the
            // conversation so the model sees its own prior step.
            let tool_calls_json: Vec<Value> = turn
                .tool_calls
                .iter()
                .map(|tc| json!({"function": {"name": tc.name, "arguments": tc.arguments}}))
                .collect();
            conversation.push(json!({
                "role": "assistant",
                "content": turn.content,
                "tool_calls": tool_calls_json,
            }));

            if turn.tool_calls.is_empty() {
                let answer = turn.content.trim();
                if !answer.is_empty() {
                    self.conversation_state_mut().record_agent_turn(answer);
                    // Streaming already emitted the answer as deltas; only the
                    // non-streaming fallback still needs to emit it.
                    if used_fallback {
                        sink.emit(&AkhMessage::narrative(answer, "acp"));
                    }
                }
                return Some(());
            }

            // Tool round. Any prose the model emitted alongside its tool calls is
            // reasoning; on the fallback path it was not streamed, so surface it
            // as a thinking step here (the streaming path already did).
            if used_fallback {
                let reasoning = turn.content.trim();
                if !reasoning.is_empty() {
                    sink.emit(&AkhMessage::Reasoning {
                        step: reasoning.to_string(),
                        expression: None,
                    });
                }
            }

            // Execute each requested tool; the call (args) and its result form one
            // collapsible step the TUI folds under "▸ N steps".
            for tc in &turn.tool_calls {
                let input = tool_input_from_args(&tc.arguments);
                let (success, result) = match self.tool_registry.execute(&tc.name, input, engine) {
                    Ok(o) => (o.success, o.result),
                    Err(e) => (false, format!("tool error: {e}")),
                };

                sink.emit(&AkhMessage::tool_result(
                    tc.name.clone(),
                    success,
                    format!("({}) → {}", preview_args(&tc.arguments), result),
                ));
                conversation.push(json!({"role": "tool", "content": result}));
            }
        }

        sink.emit(&AkhMessage::system(format!(
            "(stopped after {DEFAULT_MAX_ROUNDS} tool-call rounds without a final answer)"
        )));
        Some(())
    }
}

/// Build the Ollama/OpenAI `tools` catalog from every registered tool manifest.
///
/// Each tool becomes `{"type":"function","function":{name,description,parameters}}`
/// with a JSON-Schema `parameters` object derived from the manifest's parameter
/// list. Execution is still gated by the permission policy, so exposing every
/// tool here is safe — a dangerous one the model picks is denied at dispatch.
fn ollama_tool_catalog(registry: &ToolRegistry) -> Vec<Value> {
    registry
        .list_manifests()
        .into_iter()
        .map(|m| {
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();
            for p in &m.parameters {
                let ty = if p.param_type.is_empty() {
                    "string"
                } else {
                    p.param_type.as_str()
                };
                properties.insert(
                    p.name.clone(),
                    json!({"type": ty, "description": p.description}),
                );
                if p.required {
                    required.push(Value::String(p.name.clone()));
                }
            }
            json!({
                "type": "function",
                "function": {
                    "name": m.name,
                    "description": m.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    }
                }
            })
        })
        .collect()
}

/// Turn a model-supplied arguments value into a [`ToolInput`].
///
/// Ollama returns arguments as a JSON object; a non-object (or absent) value
/// yields an empty input, letting the tool report its own "missing param" error.
fn tool_input_from_args(args: &Value) -> ToolInput {
    match args {
        Value::Object(map) => ToolInput {
            params: map.clone(),
        },
        _ => ToolInput::new(),
    }
}

/// Compact one-line preview of a tool call's arguments for the narration line.
fn preview_args(args: &Value) -> String {
    let s = args.to_string();
    if s.chars().count() > ARG_PREVIEW_LEN {
        let truncated: String = s.chars().take(ARG_PREVIEW_LEN).collect();
        format!("{truncated}…")
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::error::AgentResult;
    use crate::agent::tool::{Tool, ToolOutput, ToolSignature};
    use crate::agent::tool_manifest::{
        DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
    };
    use std::collections::HashSet;

    struct DummyTool;
    impl Tool for DummyTool {
        fn signature(&self) -> ToolSignature {
            ToolSignature {
                name: "echo".into(),
                description: "Echo the text back".into(),
                parameters: vec![],
            }
        }
        fn execute(&self, _e: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
            Ok(ToolOutput::ok(format!(
                "echoed: {}",
                input.get("text").unwrap_or("")
            )))
        }
        fn manifest(&self) -> ToolManifest {
            ToolManifest {
                name: "echo".into(),
                description: "Echo the text back".into(),
                parameters: vec![ToolParamSchema::required("text", "text to echo")],
                danger: DangerInfo {
                    level: DangerLevel::Safe,
                    capabilities: HashSet::new(),
                    description: String::new(),
                    shadow_triggers: vec![],
                },
                source: ToolSource::Native,
            }
        }
    }

    #[test]
    fn catalog_maps_manifest_to_function_schema() {
        let mut reg = ToolRegistry::new();
        reg.register(Box::new(DummyTool));
        let catalog = ollama_tool_catalog(&reg);
        assert_eq!(catalog.len(), 1);
        let f = &catalog[0]["function"];
        assert_eq!(f["name"], "echo");
        assert_eq!(f["parameters"]["properties"]["text"]["type"], "string");
        assert_eq!(f["parameters"]["required"][0], "text");
    }

    #[test]
    fn args_object_becomes_tool_input() {
        let args = json!({"text": "hi", "n": 3});
        let input = tool_input_from_args(&args);
        assert_eq!(input.get("text"), Some("hi"));
        assert_eq!(input.get_i64("n"), Some(3));
    }

    #[test]
    fn non_object_args_yield_empty_input() {
        assert!(tool_input_from_args(&Value::Null).params.is_empty());
        assert!(tool_input_from_args(&json!("oops")).params.is_empty());
    }

    #[test]
    fn preview_truncates_long_args() {
        let big = json!({"blob": "x".repeat(500)});
        let p = preview_args(&big);
        assert!(p.ends_with('…'));
        assert!(p.chars().count() <= ARG_PREVIEW_LEN + 1);
    }
}
