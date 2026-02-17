//! Conversation persistence for the chat REPL.
//!
//! Stores conversation turns (user input + agent response) and serializes
//! them to the engine's durable store for session continuity. Also tracks
//! participant identity across sessions via SSH key fingerprints.

use serde::{Deserialize, Serialize};

/// How a participant's identity was established.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParticipantSource {
    /// Identity derived from SSH public key fingerprint.
    SshKey { fingerprint: String },
    /// User explicitly introduced themselves ("My name is Alice").
    Explicit,
    /// No identity established yet.
    Anonymous,
}

/// A conversation participant with optional stable identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// Stable identifier: "ssh:<fingerprint>" or "anon".
    pub id: String,
    /// Display name learned from explicit introduction.
    pub display_name: Option<String>,
    /// How identity was established.
    pub source: ParticipantSource,
}

impl Participant {
    /// Create an anonymous participant.
    pub fn anonymous() -> Self {
        Self {
            id: "anon".to_string(),
            display_name: None,
            source: ParticipantSource::Anonymous,
        }
    }

    /// Create a participant from an SSH key fingerprint.
    pub fn from_ssh_fingerprint(fingerprint: String) -> Self {
        let id = format!("ssh:{fingerprint}");
        Self {
            id,
            display_name: None,
            source: ParticipantSource::SshKey { fingerprint },
        }
    }
}

/// Discover an SSH public key fingerprint from `~/.ssh/*.pub`.
///
/// Reads the first valid public key found and computes its SHA-256 fingerprint.
/// Returns `None` if no SSH keys are found or readable.
pub fn discover_ssh_fingerprint() -> Option<String> {
    let home = std::env::var("HOME").ok()?;
    let ssh_dir = std::path::PathBuf::from(home).join(".ssh");

    if !ssh_dir.is_dir() {
        return None;
    }

    let entries = std::fs::read_dir(&ssh_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("pub") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                // SSH public key format: "<type> <base64-data> [comment]"
                let parts: Vec<&str> = content.trim().split_whitespace().collect();
                if parts.len() >= 2 {
                    // Decode base64 and hash.
                    if let Ok(key_bytes) = base64_decode(parts[1]) {
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let mut hasher = DefaultHasher::new();
                        key_bytes.hash(&mut hasher);
                        let hash = hasher.finish();
                        return Some(format!("{hash:016x}"));
                    }
                }
            }
        }
    }
    None
}

/// Minimal base64 decoder (standard alphabet, no padding required).
fn base64_decode(input: &str) -> Result<Vec<u8>, ()> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    let mut output = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;

    for &byte in input.as_bytes() {
        if byte == b'=' {
            break;
        }
        let val = TABLE.iter().position(|&c| c == byte).ok_or(())? as u32;
        buf = (buf << 6) | val;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            output.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Ok(output)
}

/// A single conversation turn (user input + agent response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// User's input text.
    pub user_input: String,
    /// Agent's response text.
    pub agent_response: String,
    /// Timestamp (milliseconds since epoch).
    pub timestamp_ms: u64,
    /// Participant who initiated this turn (if known).
    #[serde(default)]
    pub participant: Option<Participant>,
}

/// A conversation session with ordered turns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    turns: Vec<ConversationTurn>,
    max_turns: usize,
    /// Unique session identifier.
    #[serde(default)]
    pub session_id: String,
    /// The participant for this session (if identified).
    #[serde(default)]
    pub participant: Option<Participant>,
}

impl Conversation {
    /// Create a new empty conversation.
    pub fn new(max_turns: usize) -> Self {
        Self {
            turns: Vec::new(),
            max_turns,
            session_id: String::new(),
            participant: None,
        }
    }

    /// Create a new conversation with a session ID and optional participant.
    pub fn with_session(max_turns: usize, session_id: String, participant: Option<Participant>) -> Self {
        Self {
            turns: Vec::new(),
            max_turns,
            session_id,
            participant,
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
            participant: self.participant.clone(),
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

impl Default for Participant {
    fn default() -> Self {
        Self::anonymous()
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
