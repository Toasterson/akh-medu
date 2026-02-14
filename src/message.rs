//! Structured message protocol for agent output.
//!
//! `AkhMessage` replaces raw `println!()` calls with structured, typed messages
//! that can be rendered by different sinks: terminal (styled), JSON (streaming),
//! or collected in memory (testing).

use serde::{Deserialize, Serialize};
use std::sync::Mutex;

// ── Message types ───────────────────────────────────────────────────────

/// A structured message emitted by the agent or engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AkhMessage {
    /// KG query result or factual answer.
    Fact {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provenance: Option<String>,
    },
    /// Reasoning trace.
    Reasoning {
        step: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        expression: Option<String>,
    },
    /// Knowledge gap identified.
    Gap {
        entity: String,
        description: String,
    },
    /// Tool execution result.
    ToolResult {
        tool: String,
        success: bool,
        output: String,
    },
    /// Synthesized narrative/document.
    Narrative {
        text: String,
        grammar: String,
    },
    /// System status or informational message.
    System {
        text: String,
    },
    /// Error message.
    Error {
        code: String,
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        help: Option<String>,
    },
    /// Goal progress update.
    GoalProgress {
        goal: String,
        status: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    /// Prompt for user input.
    Prompt {
        question: String,
    },
}

// ── MessageSink trait ───────────────────────────────────────────────────

/// A destination for structured agent messages.
pub trait MessageSink: Send + Sync {
    /// Emit a single message.
    fn emit(&self, msg: &AkhMessage);

    /// Emit a batch of messages.
    fn emit_batch(&self, msgs: &[AkhMessage]) {
        for m in msgs {
            self.emit(m);
        }
    }
}

// ── StdoutSink ──────────────────────────────────────────────────────────

/// Renders messages as styled terminal output.
pub struct StdoutSink;

impl MessageSink for StdoutSink {
    fn emit(&self, msg: &AkhMessage) {
        match msg {
            AkhMessage::Fact {
                text,
                confidence,
                provenance,
            } => {
                print!("[fact] {text}");
                if let Some(c) = confidence {
                    print!(" (confidence: {c:.2})");
                }
                if let Some(p) = provenance {
                    print!(" [via {p}]");
                }
                println!();
            }
            AkhMessage::Reasoning { step, expression } => {
                print!("[reasoning] {step}");
                if let Some(expr) = expression {
                    print!(" :: {expr}");
                }
                println!();
            }
            AkhMessage::Gap {
                entity,
                description,
            } => {
                println!("[gap] {entity}: {description}");
            }
            AkhMessage::ToolResult {
                tool,
                success,
                output,
            } => {
                let status = if *success { "ok" } else { "FAIL" };
                println!("[{tool}:{status}] {output}");
            }
            AkhMessage::Narrative { text, grammar } => {
                println!("[{grammar}] {text}");
            }
            AkhMessage::System { text } => {
                println!("{text}");
            }
            AkhMessage::Error {
                code,
                message,
                help,
            } => {
                eprintln!("[error:{code}] {message}");
                if let Some(h) = help {
                    eprintln!("  help: {h}");
                }
            }
            AkhMessage::GoalProgress {
                goal,
                status,
                detail,
            } => {
                print!("[goal] {goal} — {status}");
                if let Some(d) = detail {
                    print!(": {d}");
                }
                println!();
            }
            AkhMessage::Prompt { question } => {
                println!("{question}");
            }
        }
    }
}

// ── JsonSink ────────────────────────────────────────────────────────────

/// Emits messages as newline-delimited JSON (for server streaming).
pub struct JsonSink;

impl MessageSink for JsonSink {
    fn emit(&self, msg: &AkhMessage) {
        if let Ok(json) = serde_json::to_string(msg) {
            println!("{json}");
        }
    }
}

// ── VecSink ─────────────────────────────────────────────────────────────

/// Collects messages into a `Vec<AkhMessage>` for testing.
pub struct VecSink {
    messages: Mutex<Vec<AkhMessage>>,
}

impl VecSink {
    pub fn new() -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
        }
    }

    /// Get all collected messages.
    pub fn messages(&self) -> Vec<AkhMessage> {
        self.messages.lock().unwrap().clone()
    }

    /// Number of collected messages.
    pub fn len(&self) -> usize {
        self.messages.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for VecSink {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageSink for VecSink {
    fn emit(&self, msg: &AkhMessage) {
        self.messages.lock().unwrap().push(msg.clone());
    }
}

// ── Convenience constructors ────────────────────────────────────────────

impl AkhMessage {
    pub fn fact(text: impl Into<String>) -> Self {
        Self::Fact {
            text: text.into(),
            confidence: None,
            provenance: None,
        }
    }

    pub fn system(text: impl Into<String>) -> Self {
        Self::System { text: text.into() }
    }

    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Error {
            code: code.into(),
            message: message.into(),
            help: None,
        }
    }

    pub fn goal_progress(
        goal: impl Into<String>,
        status: impl Into<String>,
    ) -> Self {
        Self::GoalProgress {
            goal: goal.into(),
            status: status.into(),
            detail: None,
        }
    }

    pub fn tool_result(
        tool: impl Into<String>,
        success: bool,
        output: impl Into<String>,
    ) -> Self {
        Self::ToolResult {
            tool: tool.into(),
            success,
            output: output.into(),
        }
    }

    pub fn narrative(text: impl Into<String>, grammar: impl Into<String>) -> Self {
        Self::Narrative {
            text: text.into(),
            grammar: grammar.into(),
        }
    }

    pub fn gap(entity: impl Into<String>, description: impl Into<String>) -> Self {
        Self::Gap {
            entity: entity.into(),
            description: description.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_sink_collects_messages() {
        let sink = VecSink::new();
        sink.emit(&AkhMessage::system("hello"));
        sink.emit(&AkhMessage::fact("dogs are mammals"));
        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn message_serializes_to_json() {
        let msg = AkhMessage::Fact {
            text: "dogs are mammals".into(),
            confidence: Some(0.95),
            provenance: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"type\":\"Fact\""));
        assert!(json.contains("dogs are mammals"));
    }

    #[test]
    fn message_deserializes_from_json() {
        let json = r#"{"type":"System","text":"hello"}"#;
        let msg: AkhMessage = serde_json::from_str(json).unwrap();
        assert!(matches!(msg, AkhMessage::System { text } if text == "hello"));
    }

    #[test]
    fn batch_emit() {
        let sink = VecSink::new();
        let msgs = vec![
            AkhMessage::system("one"),
            AkhMessage::system("two"),
            AkhMessage::system("three"),
        ];
        sink.emit_batch(&msgs);
        assert_eq!(sink.len(), 3);
    }
}
