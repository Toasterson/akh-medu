//! Inference-config tool: inspect and toggle runtime inference options (Phase 47 / ADR 075).
//!
//! Lets the agent (or an operator driving it) read and flip this akh's live inference toggles —
//! currently speculative (prompt-lookup) **greedy** decode — without a restart. The setting lives
//! in the engine's shared [`InferenceOptions`](crate::inference_options::InferenceOptions); the
//! Candle backend reads it on the next turn (no model-mutex lock needed — the flag is lock-free).
//! To persist a choice across restarts, set `spec_decode` in the akh's `akh.toml`.

use std::collections::HashSet;

use crate::agent::error::{AgentError, AgentResult};
use crate::agent::tool::{Tool, ToolInput, ToolOutput, ToolParam, ToolSignature};
use crate::agent::tool_manifest::{
    DangerInfo, DangerLevel, ToolManifest, ToolParamSchema, ToolSource,
};
use crate::engine::Engine;

const TOOL: &str = "inference_config";

/// Agent tool to inspect and change runtime inference options.
pub struct InferenceConfigTool;

impl Tool for InferenceConfigTool {
    fn signature(&self) -> ToolSignature {
        ToolSignature {
            name: TOOL.into(),
            description: "Inspect or change this akh's runtime inference options without a \
                          restart. action=get reports the current state; action=set with \
                          option + value toggles it live. Currently supports option=spec_decode \
                          (speculative greedy decode, gemma4-only, ~2x on echo-heavy output, \
                          lossless). Persist across restarts via spec_decode in akh.toml."
                .into(),
            parameters: vec![
                ToolParam {
                    name: "action".into(),
                    description: "get (report current options) or set (change one).".into(),
                    required: true,
                },
                ToolParam {
                    name: "option".into(),
                    description: "Option to change (required for set). Known: spec_decode.".into(),
                    required: false,
                },
                ToolParam {
                    name: "value".into(),
                    description: "New value (required for set): on/off (or true/false, 1/0)."
                        .into(),
                    required: false,
                },
            ],
        }
    }

    fn execute(&self, engine: &Engine, input: ToolInput) -> AgentResult<ToolOutput> {
        let action = input.require("action", TOOL)?;
        let opts = engine.inference_options();
        match action {
            "get" => Ok(ToolOutput::ok(format!(
                "Inference options:\n  spec_decode = {} (speculative greedy decode; gemma4-only)",
                opts.spec_decode()
            ))),
            "set" => {
                let option = input.require("option", TOOL)?;
                let value = input.require("value", TOOL)?;
                let on = parse_bool(value).ok_or_else(|| AgentError::ToolExecution {
                    tool_name: TOOL.into(),
                    message: format!("\"value\" must be on/off (true/false, 1/0); got {value:?}"),
                })?;
                match option {
                    "spec_decode" => {
                        let prev = opts.set_spec_decode(on);
                        let note = if on {
                            "  (applies to greedy gemma4 turns; pair with AKH_GEMV_BATCH=1 for the speedup)"
                        } else {
                            ""
                        };
                        Ok(ToolOutput::ok(format!(
                            "spec_decode: {prev} -> {on}{note}\nThis affects the running akh now; \
                             set spec_decode in akh.toml to persist across restarts."
                        )))
                    }
                    other => Err(AgentError::ToolExecution {
                        tool_name: TOOL.into(),
                        message: format!("unknown option {other:?}; known options: spec_decode"),
                    }),
                }
            }
            other => Err(AgentError::ToolExecution {
                tool_name: TOOL.into(),
                message: format!("unknown action {other:?}; use \"get\" or \"set\""),
            }),
        }
    }

    fn manifest(&self) -> ToolManifest {
        ToolManifest {
            name: TOOL.into(),
            description: "Inspect/toggle runtime inference options (e.g. speculative decode)."
                .into(),
            parameters: vec![
                ToolParamSchema::required("action", "get or set."),
                ToolParamSchema::optional("option", "Option to change (set): spec_decode."),
                ToolParamSchema::optional("value", "New value (set): on/off (true/false, 1/0)."),
            ],
            danger: DangerInfo {
                // Read-only or a decode-strategy switch; speculative decode is greedy-lossless, so
                // this never changes *what* is generated, only how fast — and writes no KG/state.
                level: DangerLevel::Safe,
                capabilities: HashSet::new(),
                description: "Toggles inference options (decode strategy). No KG or state writes; \
                              speculative decode is greedy-lossless (same output as plain decode)."
                    .into(),
                shadow_triggers: vec![],
            },
            source: ToolSource::Native,
        }
    }
}

/// Parse a permissive boolean: `on/off`, `true/false`, `yes/no`, `1/0` (case-insensitive).
fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "yes" | "1" | "enable" | "enabled" => Some(true),
        "off" | "false" | "no" | "0" | "disable" | "disabled" => Some(false),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn parses_permissive_booleans() {
        for t in ["on", "true", "YES", "1", "Enable"] {
            assert_eq!(parse_bool(t), Some(true), "{t} should be true");
        }
        for f in ["off", "false", "No", "0", "disabled"] {
            assert_eq!(parse_bool(f), Some(false), "{f} should be false");
        }
        assert_eq!(parse_bool("maybe"), None);
    }

    /// The tool flips the engine's shared `InferenceOptions` — the exact `Arc` the Candle backend
    /// holds — so the change is observed at runtime. `get` reports the live state.
    #[test]
    fn set_and_get_toggle_the_live_flag() {
        let engine = test_engine();
        let tool = InferenceConfigTool;
        assert!(!engine.inference_options().spec_decode(), "off by default");

        // get reports the current (off) state.
        let out = tool
            .execute(&engine, ToolInput::new().with_param("action", "get"))
            .unwrap();
        assert!(out.success && out.result.contains("spec_decode = false"));

        // set on -> the engine's shared flag flips.
        let out = tool
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("action", "set")
                    .with_param("option", "spec_decode")
                    .with_param("value", "on"),
            )
            .unwrap();
        assert!(out.success, "set should succeed: {}", out.result);
        assert!(
            engine.inference_options().spec_decode(),
            "the live shared flag is now on"
        );

        // set off -> back to false.
        tool.execute(
            &engine,
            ToolInput::new()
                .with_param("action", "set")
                .with_param("option", "spec_decode")
                .with_param("value", "off"),
        )
        .unwrap();
        assert!(!engine.inference_options().spec_decode());
    }

    #[test]
    fn rejects_bad_input() {
        let engine = test_engine();
        let tool = InferenceConfigTool;
        // unknown option
        assert!(tool
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("action", "set")
                    .with_param("option", "nope")
                    .with_param("value", "on"),
            )
            .is_err());
        // non-boolean value
        assert!(tool
            .execute(
                &engine,
                ToolInput::new()
                    .with_param("action", "set")
                    .with_param("option", "spec_decode")
                    .with_param("value", "perhaps"),
            )
            .is_err());
        // unknown action
        assert!(tool
            .execute(&engine, ToolInput::new().with_param("action", "frobnicate"))
            .is_err());
    }
}
