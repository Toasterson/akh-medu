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
use super::tool::{ToolInput, ToolRegistry};
use crate::engine::Engine;
use crate::message::AkhMessage;

/// Hard ceiling on tool-call rounds in one turn — a backstop against a model
/// that loops calling tools without ever answering.
const DEFAULT_MAX_ROUNDS: usize = 8;

/// Characters of a tool-call argument blob surfaced in the "calling tool" line.
const ARG_PREVIEW_LEN: usize = 160;

impl Agent {
    /// Run one LLM-driven agentic turn.
    ///
    /// The akh's pinned model answers `user_prompt`, optionally calling any
    /// registered tool (gated by the permission policy) in a loop until it
    /// returns a final answer or [`DEFAULT_MAX_ROUNDS`] is reached. The returned
    /// [`AkhMessage`]s narrate the turn: each tool call and its result, then the
    /// final answer.
    ///
    /// Returns `None` when there is **no usable model backend** — either no model
    /// manager is attached or Ollama is unreachable — signalling the caller to
    /// fall back to the symbolic [`ChatProcessor`](crate::chat::ChatProcessor).
    /// `agent_id` keys the model manager's per-akh system-prompt cache.
    pub fn run_agentic_turn(
        &mut self,
        engine: &Engine,
        agent_id: &str,
        user_prompt: &str,
    ) -> Option<Vec<AkhMessage>> {
        // Clone the Arc out so the rest of the method can borrow other `self`
        // fields (tool_registry, psyche, conversation_state) freely.
        let mgr = self.model_manager.clone()?;
        if !mgr.ollama_available() {
            return None;
        }

        // Refresh the akh's system prompt from its current psyche.
        mgr.load_agent(agent_id, agent_id, self.psyche.as_ref());

        let tools = ollama_tool_catalog(&self.tool_registry);

        let mut conversation: Vec<Value> = vec![json!({"role": "user", "content": user_prompt})];
        let mut out: Vec<AkhMessage> = Vec::new();

        for _round in 0..DEFAULT_MAX_ROUNDS {
            let turn = match mgr.chat_with_tools(agent_id, &conversation, &tools) {
                Ok(t) => t,
                Err(e) => {
                    out.push(AkhMessage::error("llm", e.to_string()));
                    return Some(out);
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
                    out.push(AkhMessage::narrative(answer, "acp"));
                }
                return Some(out);
            }

            // Execute each requested tool, feeding the result back to the model.
            for tc in &turn.tool_calls {
                out.push(AkhMessage::system(format!(
                    "→ {}({})",
                    tc.name,
                    preview_args(&tc.arguments)
                )));

                let input = tool_input_from_args(&tc.arguments);
                let (success, result) = match self.tool_registry.execute(&tc.name, input, engine) {
                    Ok(o) => (o.success, o.result),
                    Err(e) => (false, format!("tool error: {e}")),
                };

                out.push(AkhMessage::tool_result(tc.name.clone(), success, result.clone()));
                conversation.push(json!({"role": "tool", "content": result}));
            }
        }

        out.push(AkhMessage::system(format!(
            "(stopped after {DEFAULT_MAX_ROUNDS} tool-call rounds without a final answer)"
        )));
        Some(out)
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
