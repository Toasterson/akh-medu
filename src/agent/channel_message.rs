//! Protocol message types for the communication channel layer.
//!
//! Defines the inbound/outbound message types that flow through `CommChannel`
//! implementations, plus bridge conversions to/from the existing `AkhMessage`
//! protocol.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::message::AkhMessage;
use crate::symbol::SymbolId;

// ── InterlocutorId ───────────────────────────────────────────────────────

/// Identifies who sent an inbound message.
///
/// The operator is a well-known singleton; other interlocutors are identified
/// by opaque string IDs (email address, federation handle, session token, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InterlocutorId(String);

impl InterlocutorId {
    /// The well-known operator interlocutor.
    pub fn operator() -> Self {
        Self("operator".to_string())
    }

    /// An anonymous/unknown sender.
    pub fn anonymous() -> Self {
        Self("anonymous".to_string())
    }

    /// Create a named interlocutor.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the raw ID string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Whether this is the operator.
    pub fn is_operator(&self) -> bool {
        self.0 == "operator"
    }

    /// Whether this is anonymous.
    pub fn is_anonymous(&self) -> bool {
        self.0 == "anonymous"
    }
}

impl fmt::Display for InterlocutorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ── MessageContent ───────────────────────────────────────────────────────

/// The content of an inbound message.
///
/// Currently supports text and structured commands. GoalProposal, Activity,
/// and Reaction variants will be added in Phases 12d/12e.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageContent {
    /// Free-form text input.
    Text(String),
    /// A structured command (e.g. `/goals`, `/status`).
    Command {
        name: String,
        args: Option<String>,
    },
}

// ── InboundMessage ───────────────────────────────────────────────────────

/// A message received from an interlocutor through a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InboundMessage {
    /// Which channel this message arrived on.
    pub channel_id: String,
    /// Who sent this message.
    pub sender: InterlocutorId,
    /// The message content.
    pub content: MessageContent,
    /// Unix timestamp (seconds since epoch) when the message was received.
    pub timestamp: u64,
}

impl InboundMessage {
    /// Create a new inbound message with the current timestamp.
    pub fn new(channel_id: impl Into<String>, sender: InterlocutorId, content: MessageContent) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            channel_id: channel_id.into(),
            sender,
            content,
            timestamp,
        }
    }

    /// Classify this message's intent using the existing NLP classifier.
    ///
    /// For `Text` content, delegates to `nlp::classify_intent()`.
    /// For `Command` content, maps to the corresponding `UserIntent` directly.
    pub fn classify(&self) -> crate::agent::nlp::UserIntent {
        match &self.content {
            MessageContent::Text(text) => crate::agent::nlp::classify_intent(text),
            MessageContent::Command { name, args } => {
                match name.as_str() {
                    "help" | "h" => crate::agent::nlp::UserIntent::Help,
                    "status" | "goals" => crate::agent::nlp::UserIntent::ShowStatus,
                    "run" | "cycle" => {
                        let cycles = args
                            .as_ref()
                            .and_then(|a| a.trim().parse::<usize>().ok());
                        crate::agent::nlp::UserIntent::RunAgent { cycles }
                    }
                    "show" | "render" | "graph" => {
                        let entity = args.as_ref().map(|a| a.trim().to_string()).filter(|s| !s.is_empty());
                        crate::agent::nlp::UserIntent::RenderHiero { entity }
                    }
                    _ => crate::agent::nlp::UserIntent::Freeform {
                        text: format!("/{name}{}", args.as_deref().map(|a| format!(" {a}")).unwrap_or_default()),
                    },
                }
            }
        }
    }

    /// Extract the text content, if this is a text message.
    pub fn text(&self) -> Option<&str> {
        match &self.content {
            MessageContent::Text(t) => Some(t),
            MessageContent::Command { .. } => None,
        }
    }
}

// ── ConstraintCheckStatus ────────────────────────────────────────────────

/// Status of pre-communication constraint checking (Phase 12c).
///
/// Currently a placeholder; will be extended with `Passed`, `Failed`, etc.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum ConstraintCheckStatus {
    /// No constraint checking has been performed.
    #[default]
    Unchecked,
}

// ── ResponseContent ──────────────────────────────────────────────────────

/// The content of an outbound response.
///
/// Currently wraps existing `AkhMessage` types for backward compatibility.
/// A `Grounded` variant (response backed by KG provenance) will be added in
/// Phase 12b.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseContent {
    /// One or more existing AkhMessage values (backward-compatible wrapper).
    Messages(Vec<AkhMessage>),
}

// ── OutboundMessage ──────────────────────────────────────────────────────

/// A message being sent outward through a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutboundMessage {
    /// The response content.
    pub content: ResponseContent,
    /// Provenance chain: SymbolIds that contributed to this response.
    pub provenance: Vec<SymbolId>,
    /// Optional confidence score (0.0–1.0).
    pub confidence: Option<f32>,
    /// Whether pre-communication constraints were checked.
    pub constraint_check: ConstraintCheckStatus,
}

impl OutboundMessage {
    /// Create an outbound message wrapping existing `AkhMessage` values.
    pub fn from_akh_messages(msgs: Vec<AkhMessage>) -> Self {
        Self {
            content: ResponseContent::Messages(msgs),
            provenance: Vec::new(),
            confidence: None,
            constraint_check: ConstraintCheckStatus::Unchecked,
        }
    }

    /// Convenience: create an outbound message from a single `AkhMessage`.
    pub fn single(msg: AkhMessage) -> Self {
        Self::from_akh_messages(vec![msg])
    }

    /// Extract the wrapped `AkhMessage` values (for backward-compatible rendering).
    pub fn to_akh_messages(&self) -> Vec<AkhMessage> {
        match &self.content {
            ResponseContent::Messages(msgs) => msgs.clone(),
        }
    }

    /// Set the provenance chain on this message.
    pub fn with_provenance(mut self, provenance: Vec<SymbolId>) -> Self {
        self.provenance = provenance;
        self
    }

    /// Set the confidence score on this message.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence);
        self
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interlocutor_operator() {
        let id = InterlocutorId::operator();
        assert!(id.is_operator());
        assert!(!id.is_anonymous());
        assert_eq!(id.as_str(), "operator");
        assert_eq!(id.to_string(), "operator");
    }

    #[test]
    fn interlocutor_anonymous() {
        let id = InterlocutorId::anonymous();
        assert!(id.is_anonymous());
        assert!(!id.is_operator());
    }

    #[test]
    fn interlocutor_named() {
        let id = InterlocutorId::new("alice@example.com");
        assert_eq!(id.as_str(), "alice@example.com");
        assert!(!id.is_operator());
        assert!(!id.is_anonymous());
    }

    #[test]
    fn inbound_message_classify_text() {
        let msg = InboundMessage::new(
            "ch-1",
            InterlocutorId::operator(),
            MessageContent::Text("What is a dog?".to_string()),
        );
        let intent = msg.classify();
        assert!(matches!(intent, crate::agent::nlp::UserIntent::Query { .. }));
    }

    #[test]
    fn inbound_message_classify_goal() {
        let msg = InboundMessage::new(
            "ch-1",
            InterlocutorId::operator(),
            MessageContent::Text("find similar animals to Cat".to_string()),
        );
        let intent = msg.classify();
        assert!(matches!(intent, crate::agent::nlp::UserIntent::SetGoal { .. }));
    }

    #[test]
    fn inbound_message_classify_command_help() {
        let msg = InboundMessage::new(
            "ch-1",
            InterlocutorId::operator(),
            MessageContent::Command {
                name: "help".to_string(),
                args: None,
            },
        );
        let intent = msg.classify();
        assert!(matches!(intent, crate::agent::nlp::UserIntent::Help));
    }

    #[test]
    fn inbound_message_classify_command_run() {
        let msg = InboundMessage::new(
            "ch-1",
            InterlocutorId::operator(),
            MessageContent::Command {
                name: "run".to_string(),
                args: Some("5".to_string()),
            },
        );
        let intent = msg.classify();
        assert!(matches!(
            intent,
            crate::agent::nlp::UserIntent::RunAgent { cycles: Some(5) }
        ));
    }

    #[test]
    fn inbound_message_text_accessor() {
        let msg = InboundMessage::new(
            "ch",
            InterlocutorId::operator(),
            MessageContent::Text("hello".to_string()),
        );
        assert_eq!(msg.text(), Some("hello"));

        let cmd = InboundMessage::new(
            "ch",
            InterlocutorId::operator(),
            MessageContent::Command {
                name: "help".to_string(),
                args: None,
            },
        );
        assert_eq!(cmd.text(), None);
    }

    #[test]
    fn outbound_message_round_trip() {
        let original = vec![
            AkhMessage::system("hello"),
            AkhMessage::fact("dogs are mammals"),
        ];
        let out = OutboundMessage::from_akh_messages(original.clone());
        let recovered = out.to_akh_messages();
        assert_eq!(recovered.len(), 2);
    }

    #[test]
    fn outbound_message_single() {
        let msg = OutboundMessage::single(AkhMessage::system("test"));
        let recovered = msg.to_akh_messages();
        assert_eq!(recovered.len(), 1);
    }

    #[test]
    fn outbound_message_builders() {
        let msg = OutboundMessage::single(AkhMessage::system("test"))
            .with_confidence(0.95)
            .with_provenance(vec![]);
        assert_eq!(msg.confidence, Some(0.95));
        assert!(msg.provenance.is_empty());
    }

    #[test]
    fn constraint_check_defaults_to_unchecked() {
        let msg = OutboundMessage::single(AkhMessage::system("test"));
        assert!(matches!(msg.constraint_check, ConstraintCheckStatus::Unchecked));
    }
}
