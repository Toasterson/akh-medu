//! Multi-agent communication: capability tokens, protocol messages, trust bootstrap (Phase 12g).
//!
//! Provides structured agent-to-agent interaction with OCapN-inspired capability
//! tokens scoped to specific permissions and topics. Agents communicate via
//! `AgentProtocolMessage` variants carried as `MessageContent::AgentMessage`
//! through the existing `CommChannel` infrastructure.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::symbol::SymbolId;

use super::channel::ChannelKind;
use super::conversation::GroundedTriple;

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors from the multi-agent communication subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum MultiAgentError {
    #[error("capability token expired: \"{token_id}\"")]
    #[diagnostic(
        code(akh::multi_agent::token_expired),
        help("The capability token has expired. Request a new one from the operator.")
    )]
    TokenExpired { token_id: String },

    #[error("capability denied: action \"{action}\" not in token scope")]
    #[diagnostic(
        code(akh::multi_agent::capability_denied),
        help(
            "The capability token does not grant this action. \
             Ask the operator to grant a token with the required scope."
        )
    )]
    CapabilityDenied { action: String, token_id: String },

    #[error("unknown agent: \"{agent_id}\"")]
    #[diagnostic(
        code(akh::multi_agent::unknown_agent),
        help("This agent has not been introduced. The operator must introduce agents first.")
    )]
    UnknownAgent { agent_id: String },

    #[error("token not found: \"{token_id}\"")]
    #[diagnostic(
        code(akh::multi_agent::token_not_found),
        help("No capability token exists with this ID.")
    )]
    TokenNotFound { token_id: String },

    #[error("invalid protocol message: {message}")]
    #[diagnostic(
        code(akh::multi_agent::invalid_message),
        help("The agent protocol message could not be parsed or validated.")
    )]
    InvalidMessage { message: String },
}

/// Convenience alias.
pub type MultiAgentResult<T> = std::result::Result<T, MultiAgentError>;

// ── CapabilityScope ─────────────────────────────────────────────────────

/// What a capability token permits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityScope {
    /// May query the KG about any topic.
    QueryAll,
    /// May query only about specific topics (SymbolIds).
    QueryTopics(Vec<SymbolId>),
    /// May assert (propose) facts about specific topics.
    AssertTopics(Vec<SymbolId>),
    /// May propose goals.
    ProposeGoals,
    /// May subscribe to topic updates.
    Subscribe,
    /// May view provenance and explanation chains.
    ViewProvenance,
}

impl fmt::Display for CapabilityScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueryAll => write!(f, "query:all"),
            Self::QueryTopics(topics) => write!(f, "query:{} topic(s)", topics.len()),
            Self::AssertTopics(topics) => write!(f, "assert:{} topic(s)", topics.len()),
            Self::ProposeGoals => write!(f, "propose-goals"),
            Self::Subscribe => write!(f, "subscribe"),
            Self::ViewProvenance => write!(f, "view-provenance"),
        }
    }
}

// ── CapabilityToken ─────────────────────────────────────────────────────

/// A scoped permission token granted by the operator to enable agent-to-agent
/// interaction.
///
/// Tokens are immutable once created. They can be revoked but not modified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityToken {
    /// Unique token identifier (e.g. "cap-{uuid}").
    pub token_id: String,
    /// Which agent this token grants access to interact with.
    pub target_agent: String,
    /// The agent that holds this token (the requester).
    pub holder_agent: String,
    /// What actions this token permits.
    pub scopes: Vec<CapabilityScope>,
    /// Unix timestamp when the token was issued.
    pub issued_at: u64,
    /// Unix timestamp when the token expires. `None` = no expiry.
    pub expires_at: Option<u64>,
    /// Whether this token has been revoked.
    pub revoked: bool,
}

impl CapabilityToken {
    /// Create a new capability token.
    pub fn new(
        token_id: impl Into<String>,
        target_agent: impl Into<String>,
        holder_agent: impl Into<String>,
        scopes: Vec<CapabilityScope>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            token_id: token_id.into(),
            target_agent: target_agent.into(),
            holder_agent: holder_agent.into(),
            scopes,
            issued_at: now,
            expires_at: None,
            revoked: false,
        }
    }

    /// Set an expiry time.
    pub fn with_expiry(mut self, expires_at: u64) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Whether the token is currently valid (not expired, not revoked).
    pub fn is_valid(&self) -> bool {
        if self.revoked {
            return false;
        }
        if let Some(expiry) = self.expires_at {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            if now > expiry {
                return false;
            }
        }
        true
    }

    /// Check whether a specific action is permitted by this token.
    pub fn permits(&self, action: &str, topic: Option<SymbolId>) -> bool {
        if !self.is_valid() {
            return false;
        }
        for scope in &self.scopes {
            match (action, scope) {
                ("query", CapabilityScope::QueryAll) => return true,
                ("query", CapabilityScope::QueryTopics(topics)) => {
                    if let Some(t) = topic {
                        if topics.contains(&t) {
                            return true;
                        }
                    }
                }
                ("assert", CapabilityScope::AssertTopics(topics)) => {
                    if let Some(t) = topic {
                        if topics.contains(&t) {
                            return true;
                        }
                    }
                }
                ("propose_goals", CapabilityScope::ProposeGoals) => return true,
                ("subscribe", CapabilityScope::Subscribe) => return true,
                ("view_provenance", CapabilityScope::ViewProvenance) => return true,
                _ => {}
            }
        }
        false
    }

    /// Revoke this token.
    pub fn revoke(&mut self) {
        self.revoked = true;
    }
}

// ── AgentProtocolMessage ────────────────────────────────────────────────

/// Structured message types for agent-to-agent communication.
///
/// These are carried as `MessageContent::AgentMessage` and dispatched
/// directly to the multi-agent handler, bypassing the NLP intent classifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentProtocolMessage {
    /// Query the other agent's KG about a subject.
    Query {
        /// The subject to query about.
        subject: String,
        /// Optional topic SymbolId for capability scoping.
        topic_id: Option<u64>,
        /// Capability token ID authorizing this query.
        token_id: String,
    },

    /// Respond to a query with grounded triples.
    QueryResponse {
        /// The original subject queried.
        subject: String,
        /// Grounded triples from the KG.
        triples: Vec<GroundedTriple>,
        /// Aggregate confidence score.
        confidence: Option<f32>,
        /// Provenance SymbolIds (for audit).
        provenance_ids: Vec<u64>,
    },

    /// Propose a fact (assertion) to the other agent.
    Assert {
        /// The fact text.
        claim: String,
        /// Supporting evidence as grounded triples.
        evidence: Vec<GroundedTriple>,
        /// Confidence in the claim.
        confidence: f32,
        /// Capability token ID authorizing this assertion.
        token_id: String,
    },

    /// Propose a goal to the other agent (requires operator approval).
    ProposeGoal {
        /// The goal description.
        description: String,
        /// Why the goal is being proposed.
        rationale: String,
        /// Capability token ID authorizing this proposal.
        token_id: String,
    },

    /// Subscribe to updates about a topic.
    Subscribe {
        /// Topic SymbolId to subscribe to.
        topic_id: u64,
        /// Capability token ID.
        token_id: String,
    },

    /// Unsubscribe from a topic.
    Unsubscribe {
        /// Topic SymbolId.
        topic_id: u64,
    },

    /// Grant a capability token to the other agent.
    GrantCapability {
        /// The token being granted.
        token: CapabilityToken,
    },

    /// Revoke a previously granted capability token.
    RevokeCapability {
        /// The token ID being revoked.
        token_id: String,
    },

    /// Acknowledge receipt of a message.
    Ack {
        /// What was acknowledged.
        regarding: String,
    },

    /// Error response.
    Error {
        /// Error code.
        code: String,
        /// Human-readable message.
        message: String,
    },
}

impl AgentProtocolMessage {
    /// Whether this message requires a capability token.
    pub fn requires_token(&self) -> bool {
        matches!(
            self,
            Self::Query { .. }
                | Self::Assert { .. }
                | Self::ProposeGoal { .. }
                | Self::Subscribe { .. }
        )
    }

    /// Extract the token ID from this message, if it carries one.
    pub fn token_id(&self) -> Option<&str> {
        match self {
            Self::Query { token_id, .. }
            | Self::Assert { token_id, .. }
            | Self::ProposeGoal { token_id, .. }
            | Self::Subscribe { token_id, .. } => Some(token_id),
            _ => None,
        }
    }
}

// ── InterlocutorKind ────────────────────────────────────────────────────

/// Whether an interlocutor is a human or another agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterlocutorKind {
    /// A human user.
    Human,
    /// Another autonomous agent.
    Agent,
}

impl Default for InterlocutorKind {
    fn default() -> Self {
        Self::Human
    }
}

impl fmt::Display for InterlocutorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Human => write!(f, "human"),
            Self::Agent => write!(f, "agent"),
        }
    }
}

// ── TokenRegistry ───────────────────────────────────────────────────────

/// Registry of granted capability tokens.
///
/// Manages tokens that have been issued by the operator to enable
/// agent-to-agent communication. Indexed by token ID and by agent pair.
#[derive(Debug, Default)]
pub struct TokenRegistry {
    /// Tokens indexed by token_id.
    tokens: HashMap<String, CapabilityToken>,
    /// Token IDs indexed by holder agent → target agent.
    by_pair: HashMap<(String, String), Vec<String>>,
}

impl TokenRegistry {
    /// Create an empty token registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Grant a new capability token.
    pub fn grant(&mut self, token: CapabilityToken) {
        let pair_key = (token.holder_agent.clone(), token.target_agent.clone());
        self.by_pair
            .entry(pair_key)
            .or_default()
            .push(token.token_id.clone());
        self.tokens.insert(token.token_id.clone(), token);
    }

    /// Revoke a token by ID.
    pub fn revoke(&mut self, token_id: &str) -> MultiAgentResult<()> {
        let token = self.tokens.get_mut(token_id).ok_or_else(|| {
            MultiAgentError::TokenNotFound {
                token_id: token_id.to_string(),
            }
        })?;
        token.revoke();
        Ok(())
    }

    /// Get a token by ID.
    pub fn get(&self, token_id: &str) -> Option<&CapabilityToken> {
        self.tokens.get(token_id)
    }

    /// Find all valid tokens for a given agent pair (holder → target).
    pub fn tokens_for_pair(
        &self,
        holder: &str,
        target: &str,
    ) -> Vec<&CapabilityToken> {
        let pair_key = (holder.to_string(), target.to_string());
        self.by_pair
            .get(&pair_key)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.tokens.get(id))
                    .filter(|t| t.is_valid())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Validate a protocol message against its token.
    ///
    /// Returns `Ok(())` if the message is authorized, or an error describing
    /// the failure.
    pub fn validate_message(
        &self,
        msg: &AgentProtocolMessage,
        sender: &str,
    ) -> MultiAgentResult<()> {
        let token_id = match msg.token_id() {
            Some(id) => id,
            None => return Ok(()), // Messages without tokens don't need validation.
        };

        let token = self.tokens.get(token_id).ok_or_else(|| {
            MultiAgentError::TokenNotFound {
                token_id: token_id.to_string(),
            }
        })?;

        // Check the holder matches the sender.
        if token.holder_agent != sender {
            return Err(MultiAgentError::CapabilityDenied {
                action: "use_token".to_string(),
                token_id: token_id.to_string(),
            });
        }

        // Check validity (expiry + revocation).
        if !token.is_valid() {
            return Err(MultiAgentError::TokenExpired {
                token_id: token_id.to_string(),
            });
        }

        // Check action-specific scope.
        let (action, topic) = match msg {
            AgentProtocolMessage::Query { topic_id, .. } => {
                ("query", topic_id.and_then(SymbolId::new))
            }
            AgentProtocolMessage::Assert { .. } => ("assert", None),
            AgentProtocolMessage::ProposeGoal { .. } => ("propose_goals", None),
            AgentProtocolMessage::Subscribe { topic_id, .. } => {
                ("subscribe", SymbolId::new(*topic_id))
            }
            _ => return Ok(()),
        };

        if !token.permits(action, topic) {
            return Err(MultiAgentError::CapabilityDenied {
                action: action.to_string(),
                token_id: token_id.to_string(),
            });
        }

        Ok(())
    }

    /// Total number of tokens (including revoked).
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Count of currently valid (non-expired, non-revoked) tokens.
    pub fn valid_count(&self) -> usize {
        self.tokens.values().filter(|t| t.is_valid()).count()
    }
}

// ── Trust bootstrap ─────────────────────────────────────────────────────

/// Determine the initial trust level for an agent based on how it was
/// introduced.
///
/// - If the operator explicitly introduced the agent via a `GrantCapability`
///   message, the agent starts at `Trusted`.
/// - If the agent was encountered through federation without introduction,
///   it starts at `Public`.
pub fn initial_trust_for_agent(has_operator_introduction: bool) -> ChannelKind {
    if has_operator_introduction {
        ChannelKind::Trusted
    } else {
        ChannelKind::Public
    }
}

/// Check if a protocol message should promote an interlocutor from Public
/// to Trusted (via capability grant from the operator).
pub fn should_promote_trust(msg: &AgentProtocolMessage) -> bool {
    matches!(msg, AgentProtocolMessage::GrantCapability { .. })
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(n: u64) -> SymbolId {
        SymbolId::new(n).unwrap()
    }

    // ── CapabilityToken ──────────────────────────────────────────────

    #[test]
    fn token_valid_by_default() {
        let token = CapabilityToken::new(
            "cap-1", "agent-b", "agent-a",
            vec![CapabilityScope::QueryAll],
        );
        assert!(token.is_valid());
        assert!(token.permits("query", None));
        assert!(!token.permits("assert", None));
    }

    #[test]
    fn token_expired() {
        let token = CapabilityToken::new(
            "cap-2", "agent-b", "agent-a",
            vec![CapabilityScope::QueryAll],
        ).with_expiry(0); // Expired at epoch.
        assert!(!token.is_valid());
        assert!(!token.permits("query", None));
    }

    #[test]
    fn token_revoked() {
        let mut token = CapabilityToken::new(
            "cap-3", "agent-b", "agent-a",
            vec![CapabilityScope::QueryAll],
        );
        assert!(token.is_valid());
        token.revoke();
        assert!(!token.is_valid());
    }

    #[test]
    fn token_scoped_to_topics() {
        let token = CapabilityToken::new(
            "cap-4", "agent-b", "agent-a",
            vec![CapabilityScope::QueryTopics(vec![sym(42), sym(43)])],
        );
        assert!(token.permits("query", Some(sym(42))));
        assert!(token.permits("query", Some(sym(43))));
        assert!(!token.permits("query", Some(sym(99))));
        assert!(!token.permits("query", None)); // No topic specified.
    }

    #[test]
    fn token_multiple_scopes() {
        let token = CapabilityToken::new(
            "cap-5", "agent-b", "agent-a",
            vec![
                CapabilityScope::QueryAll,
                CapabilityScope::ProposeGoals,
                CapabilityScope::ViewProvenance,
            ],
        );
        assert!(token.permits("query", None));
        assert!(token.permits("propose_goals", None));
        assert!(token.permits("view_provenance", None));
        assert!(!token.permits("assert", None));
    }

    // ── TokenRegistry ────────────────────────────────────────────────

    #[test]
    fn registry_grant_and_lookup() {
        let mut registry = TokenRegistry::new();
        assert!(registry.is_empty());

        let token = CapabilityToken::new(
            "cap-1", "agent-b", "agent-a",
            vec![CapabilityScope::QueryAll],
        );
        registry.grant(token);

        assert_eq!(registry.len(), 1);
        assert!(registry.get("cap-1").is_some());
    }

    #[test]
    fn registry_tokens_for_pair() {
        let mut registry = TokenRegistry::new();
        registry.grant(CapabilityToken::new(
            "cap-1", "agent-b", "agent-a",
            vec![CapabilityScope::QueryAll],
        ));
        registry.grant(CapabilityToken::new(
            "cap-2", "agent-c", "agent-a",
            vec![CapabilityScope::ProposeGoals],
        ));

        let tokens = registry.tokens_for_pair("agent-a", "agent-b");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token_id, "cap-1");

        let tokens = registry.tokens_for_pair("agent-a", "agent-c");
        assert_eq!(tokens.len(), 1);

        let tokens = registry.tokens_for_pair("agent-b", "agent-a");
        assert!(tokens.is_empty());
    }

    #[test]
    fn registry_revoke() {
        let mut registry = TokenRegistry::new();
        registry.grant(CapabilityToken::new(
            "cap-1", "agent-b", "agent-a",
            vec![CapabilityScope::QueryAll],
        ));
        assert_eq!(registry.valid_count(), 1);

        registry.revoke("cap-1").unwrap();
        assert_eq!(registry.valid_count(), 0);
        assert_eq!(registry.len(), 1); // Still stored, just revoked.
    }

    #[test]
    fn registry_revoke_not_found() {
        let mut registry = TokenRegistry::new();
        let result = registry.revoke("nonexistent");
        assert!(matches!(result, Err(MultiAgentError::TokenNotFound { .. })));
    }

    #[test]
    fn registry_validate_query() {
        let mut registry = TokenRegistry::new();
        registry.grant(CapabilityToken::new(
            "cap-1", "agent-b", "agent-a",
            vec![CapabilityScope::QueryAll],
        ));

        let msg = AgentProtocolMessage::Query {
            subject: "dogs".to_string(),
            topic_id: None,
            token_id: "cap-1".to_string(),
        };

        // Valid: sender matches holder.
        assert!(registry.validate_message(&msg, "agent-a").is_ok());

        // Invalid: wrong sender.
        assert!(registry.validate_message(&msg, "agent-c").is_err());
    }

    #[test]
    fn registry_validate_expired_token() {
        let mut registry = TokenRegistry::new();
        registry.grant(
            CapabilityToken::new(
                "cap-expired", "agent-b", "agent-a",
                vec![CapabilityScope::QueryAll],
            )
            .with_expiry(0),
        );

        let msg = AgentProtocolMessage::Query {
            subject: "dogs".to_string(),
            topic_id: None,
            token_id: "cap-expired".to_string(),
        };
        assert!(matches!(
            registry.validate_message(&msg, "agent-a"),
            Err(MultiAgentError::TokenExpired { .. })
        ));
    }

    #[test]
    fn registry_validate_scope_denied() {
        let mut registry = TokenRegistry::new();
        registry.grant(CapabilityToken::new(
            "cap-query-only", "agent-b", "agent-a",
            vec![CapabilityScope::QueryAll],
        ));

        let msg = AgentProtocolMessage::ProposeGoal {
            description: "learn Rust".to_string(),
            rationale: "relevant to current work".to_string(),
            token_id: "cap-query-only".to_string(),
        };
        assert!(matches!(
            registry.validate_message(&msg, "agent-a"),
            Err(MultiAgentError::CapabilityDenied { .. })
        ));
    }

    // ── AgentProtocolMessage ─────────────────────────────────────────

    #[test]
    fn message_requires_token() {
        assert!(AgentProtocolMessage::Query {
            subject: "test".to_string(),
            topic_id: None,
            token_id: "cap-1".to_string(),
        }
        .requires_token());

        assert!(!AgentProtocolMessage::Ack {
            regarding: "query".to_string(),
        }
        .requires_token());

        assert!(!AgentProtocolMessage::QueryResponse {
            subject: "test".to_string(),
            triples: vec![],
            confidence: None,
            provenance_ids: vec![],
        }
        .requires_token());
    }

    #[test]
    fn message_token_id_extraction() {
        let msg = AgentProtocolMessage::Query {
            subject: "test".to_string(),
            topic_id: None,
            token_id: "cap-42".to_string(),
        };
        assert_eq!(msg.token_id(), Some("cap-42"));

        let ack = AgentProtocolMessage::Ack {
            regarding: "test".to_string(),
        };
        assert_eq!(ack.token_id(), None);
    }

    #[test]
    fn protocol_message_serde_round_trip() {
        let msg = AgentProtocolMessage::Query {
            subject: "Rust programming".to_string(),
            topic_id: Some(42),
            token_id: "cap-1".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let recovered: AgentProtocolMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(recovered, AgentProtocolMessage::Query { ref subject, .. } if subject == "Rust programming"));
    }

    // ── InterlocutorKind ─────────────────────────────────────────────

    #[test]
    fn interlocutor_kind_default_is_human() {
        assert_eq!(InterlocutorKind::default(), InterlocutorKind::Human);
    }

    #[test]
    fn interlocutor_kind_display() {
        assert_eq!(InterlocutorKind::Human.to_string(), "human");
        assert_eq!(InterlocutorKind::Agent.to_string(), "agent");
    }

    // ── Trust bootstrap ──────────────────────────────────────────────

    #[test]
    fn trust_bootstrap_with_introduction() {
        assert_eq!(
            initial_trust_for_agent(true),
            ChannelKind::Trusted,
        );
    }

    #[test]
    fn trust_bootstrap_without_introduction() {
        assert_eq!(
            initial_trust_for_agent(false),
            ChannelKind::Public,
        );
    }

    #[test]
    fn should_promote_on_grant() {
        let msg = AgentProtocolMessage::GrantCapability {
            token: CapabilityToken::new(
                "cap-1", "agent-b", "agent-a",
                vec![CapabilityScope::QueryAll],
            ),
        };
        assert!(should_promote_trust(&msg));

        let msg = AgentProtocolMessage::Query {
            subject: "test".to_string(),
            topic_id: None,
            token_id: "cap-1".to_string(),
        };
        assert!(!should_promote_trust(&msg));
    }

    // ── CapabilityScope ──────────────────────────────────────────────

    #[test]
    fn scope_display() {
        assert_eq!(CapabilityScope::QueryAll.to_string(), "query:all");
        assert_eq!(
            CapabilityScope::QueryTopics(vec![sym(1), sym(2)]).to_string(),
            "query:2 topic(s)"
        );
        assert_eq!(CapabilityScope::ProposeGoals.to_string(), "propose-goals");
    }
}
