//! Conversation persistence for the chat REPL.
//!
//! Stores conversation turns (user input + agent response) and serializes
//! them to the engine's durable store for session continuity.

use serde::{Deserialize, Serialize};

/// A single conversation turn (user input + agent response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// User's input text.
    pub user_input: String,
    /// Agent's response text.
    pub agent_response: String,
    /// Timestamp (milliseconds since epoch).
    pub timestamp_ms: u64,
}

/// A conversation session with ordered turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    turns: Vec<ConversationTurn>,
    max_turns: usize,
}

impl Conversation {
    /// Create a new empty conversation.
    pub fn new(max_turns: usize) -> Self {
        Self {
            turns: Vec::new(),
            max_turns,
        }
    }

    /// Add a turn to the conversation, evicting the oldest if at capacity.
    pub fn add_turn(&mut self, user_input: String, agent_response: String) {
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        self.turns.push(ConversationTurn {
            user_input,
            agent_response,
            timestamp_ms,
        });

        // Evict oldest turns if over capacity.
        while self.turns.len() > self.max_turns {
            self.turns.remove(0);
        }
    }

    /// Get all turns.
    pub fn turns(&self) -> &[ConversationTurn] {
        &self.turns
    }

    /// Number of turns stored.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Whether the conversation is empty.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Serialize to bytes via bincode.
    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        bincode::serialize(self)
    }

    /// Deserialize from bytes via bincode.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }
}

impl Default for Conversation {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_add_and_retrieve() {
        let mut conv = Conversation::new(10);
        conv.add_turn("hello".into(), "hi there".into());
        conv.add_turn("what is a dog?".into(), "a mammal".into());

        assert_eq!(conv.len(), 2);
        assert_eq!(conv.turns()[0].user_input, "hello");
        assert_eq!(conv.turns()[1].agent_response, "a mammal");
    }

    #[test]
    fn conversation_eviction() {
        let mut conv = Conversation::new(2);
        conv.add_turn("a".into(), "1".into());
        conv.add_turn("b".into(), "2".into());
        conv.add_turn("c".into(), "3".into());

        assert_eq!(conv.len(), 2);
        assert_eq!(conv.turns()[0].user_input, "b");
        assert_eq!(conv.turns()[1].user_input, "c");
    }

    #[test]
    fn conversation_serialize_deserialize() {
        let mut conv = Conversation::new(10);
        conv.add_turn("test input".into(), "test output".into());

        let bytes = conv.to_bytes().unwrap();
        let restored = Conversation::from_bytes(&bytes).unwrap();

        assert_eq!(restored.len(), 1);
        assert_eq!(restored.turns()[0].user_input, "test input");
        assert_eq!(restored.turns()[0].agent_response, "test output");
    }

    #[test]
    fn empty_conversation() {
        let conv = Conversation::default();
        assert!(conv.is_empty());
        assert_eq!(conv.len(), 0);
    }
}
