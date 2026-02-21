//! Communication channel abstraction with OCapN-inspired capability model.
//!
//! Provides the `CommChannel` trait that all interaction surfaces (TUI, headless REPL,
//! WebSocket, future federation) implement, plus a `ChannelRegistry` that the Agent
//! uses to manage open channels.
//!
//! Each channel carries an immutable `ChannelCapabilities` determined by its
//! `ChannelKind` (Operator / Trusted / Social / Public).

use std::collections::HashMap;
use std::fmt;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::channel_message::{InboundMessage, OutboundMessage};

// ── Errors ───────────────────────────────────────────────────────────────

/// Errors specific to the communication channel layer.
#[derive(Debug, Error, Diagnostic)]
pub enum ChannelError {
    #[error("capability denied: action \"{action}\" on channel \"{channel_id}\" (kind: {kind})")]
    #[diagnostic(
        code(akh::channel::capability_denied),
        help(
            "The channel's kind ({kind}) does not permit this action. \
             Operator channels have full access; social/public channels are restricted."
        )
    )]
    CapabilityDenied {
        action: String,
        channel_id: String,
        kind: ChannelKind,
    },

    #[error("rate limit exceeded on channel \"{channel_id}\": {limit} messages/minute")]
    #[diagnostic(
        code(akh::channel::rate_limit),
        help("Wait before sending more messages, or use an Operator/Trusted channel.")
    )]
    RateLimitExceeded { channel_id: String, limit: u32 },

    #[error("duplicate operator channel: an operator channel is already registered")]
    #[diagnostic(
        code(akh::channel::duplicate_operator),
        help(
            "Only one Operator channel may be registered at a time. \
             Unregister the existing operator channel first."
        )
    )]
    DuplicateOperator,

    #[error("channel not found: \"{channel_id}\"")]
    #[diagnostic(
        code(akh::channel::not_found),
        help("The channel ID does not exist in the registry. Check with `channel_ids()`.")
    )]
    ChannelNotFound { channel_id: String },

    #[error("transport error: {message}")]
    #[diagnostic(
        code(akh::channel::transport),
        help("The underlying transport (stdin, WebSocket, etc.) encountered an error.")
    )]
    Transport { message: String },
}

/// Convenience alias for channel operations.
pub type ChannelResult<T> = std::result::Result<T, ChannelError>;

// ── ChannelKind ──────────────────────────────────────────────────────────

/// The trust level / role of a communication channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChannelKind {
    /// The primary operator (full control, exactly one per agent).
    Operator,
    /// A trusted peer (e.g. another akh instance in the same federation).
    Trusted,
    /// A social contact (e.g. an external user via chat).
    Social,
    /// A public/anonymous channel (read-only, heavily restricted).
    Public,
}

impl fmt::Display for ChannelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operator => write!(f, "Operator"),
            Self::Trusted => write!(f, "Trusted"),
            Self::Social => write!(f, "Social"),
            Self::Public => write!(f, "Public"),
        }
    }
}

// ── ChannelCapabilities ──────────────────────────────────────────────────

/// Immutable capability set for a channel, determined by its [`ChannelKind`].
///
/// Capabilities are boolean flags that gate what a channel is allowed to do.
/// Once created, they cannot be changed — the trust model is fixed at channel
/// construction time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelCapabilities {
    /// Can set/modify agent goals.
    pub can_set_goals: bool,
    /// Can query the knowledge graph.
    pub can_query: bool,
    /// Can assert new facts into the KG.
    pub can_assert: bool,
    /// Can trigger OODA cycles.
    pub can_run_cycles: bool,
    /// Can invoke tools directly.
    pub can_invoke_tools: bool,
    /// Can read agent status and working memory.
    pub can_read_status: bool,
    /// Can modify agent configuration.
    pub can_configure: bool,
    /// Can persist/resume sessions.
    pub can_persist: bool,
    /// Can register/unregister other channels.
    pub can_manage_channels: bool,
    /// Can view provenance and reasoning traces.
    pub can_view_provenance: bool,
    /// Can propose goals to the agent via agent protocol messages (Phase 12g).
    pub can_propose_goals: bool,
    /// Optional rate limit (messages per minute). None = unlimited.
    pub rate_limit: Option<u32>,
}

impl ChannelCapabilities {
    /// Full capabilities for the operator channel.
    pub fn operator() -> Self {
        Self {
            can_set_goals: true,
            can_query: true,
            can_assert: true,
            can_run_cycles: true,
            can_invoke_tools: true,
            can_read_status: true,
            can_configure: true,
            can_persist: true,
            can_manage_channels: true,
            can_view_provenance: true,
            can_propose_goals: true,
            rate_limit: None,
        }
    }

    /// Capabilities for a trusted peer.
    pub fn trusted() -> Self {
        Self {
            can_set_goals: true,
            can_query: true,
            can_assert: true,
            can_run_cycles: true,
            can_invoke_tools: false,
            can_read_status: true,
            can_configure: false,
            can_persist: false,
            can_manage_channels: false,
            can_view_provenance: true,
            can_propose_goals: true,
            rate_limit: None,
        }
    }

    /// Capabilities for a social contact.
    pub fn social() -> Self {
        Self {
            can_set_goals: false,
            can_query: true,
            can_assert: false,
            can_run_cycles: false,
            can_invoke_tools: false,
            can_read_status: false,
            can_configure: false,
            can_persist: false,
            can_manage_channels: false,
            can_view_provenance: false,
            can_propose_goals: false,
            rate_limit: Some(30),
        }
    }

    /// Capabilities for a public/anonymous channel (read-only).
    pub fn public_readonly() -> Self {
        Self {
            can_set_goals: false,
            can_query: true,
            can_assert: false,
            can_run_cycles: false,
            can_invoke_tools: false,
            can_read_status: false,
            can_configure: false,
            can_persist: false,
            can_manage_channels: false,
            can_view_provenance: false,
            can_propose_goals: false,
            rate_limit: Some(10),
        }
    }

    /// Create capabilities for a given channel kind.
    pub fn for_kind(kind: ChannelKind) -> Self {
        match kind {
            ChannelKind::Operator => Self::operator(),
            ChannelKind::Trusted => Self::trusted(),
            ChannelKind::Social => Self::social(),
            ChannelKind::Public => Self::public_readonly(),
        }
    }

    /// Check that a capability-gated action is permitted. Returns `Ok(())` if
    /// the action is allowed, or `Err(CapabilityDenied)` if not.
    ///
    /// `action` should be a human-readable name like `"set_goals"`, `"query"`, etc.
    pub fn require(
        &self,
        action: &str,
        channel_id: &str,
        kind: ChannelKind,
    ) -> ChannelResult<()> {
        let allowed = match action {
            "set_goals" => self.can_set_goals,
            "query" => self.can_query,
            "assert" => self.can_assert,
            "run_cycles" => self.can_run_cycles,
            "invoke_tools" => self.can_invoke_tools,
            "read_status" => self.can_read_status,
            "configure" => self.can_configure,
            "persist" => self.can_persist,
            "manage_channels" => self.can_manage_channels,
            "view_provenance" => self.can_view_provenance,
            "propose_goals" => self.can_propose_goals,
            _ => false,
        };

        if allowed {
            Ok(())
        } else {
            Err(ChannelError::CapabilityDenied {
                action: action.to_string(),
                channel_id: channel_id.to_string(),
                kind,
            })
        }
    }
}

// ── CommChannel trait ────────────────────────────────────────────────────

/// A communication channel through which the agent interacts with the outside
/// world. Each channel has a kind (trust level), immutable capabilities, and
/// can send/receive messages.
///
/// Channels are `Send` but not necessarily `Sync` — they are owned by the
/// registry and accessed via `&mut` borrows.
pub trait CommChannel: Send {
    /// Unique identifier for this channel instance.
    fn channel_id(&self) -> &str;

    /// The kind (trust level) of this channel.
    fn channel_kind(&self) -> ChannelKind;

    /// The immutable capability set for this channel.
    fn capabilities(&self) -> &ChannelCapabilities;

    /// Try to receive a pending inbound message (non-blocking).
    ///
    /// Returns `Ok(None)` if no message is available, `Ok(Some(msg))` if one
    /// is ready, or `Err` on transport failure.
    fn try_receive(&mut self) -> ChannelResult<Option<InboundMessage>>;

    /// Send an outbound message through this channel.
    fn send(&self, msg: &OutboundMessage) -> ChannelResult<()>;

    /// Whether the channel is still connected/alive.
    fn is_connected(&self) -> bool;
}

// ── ChannelRegistry ─────────────────────────────────────────────────────

/// Registry of open communication channels, owned by the Agent.
///
/// Enforces the invariant that at most one `Operator` channel may be
/// registered at any time.
pub struct ChannelRegistry {
    channels: HashMap<String, Box<dyn CommChannel>>,
    /// Cached operator channel ID for fast lookup.
    operator_id: Option<String>,
}

impl ChannelRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            channels: HashMap::new(),
            operator_id: None,
        }
    }

    /// Register a channel. Enforces the exactly-one-operator invariant.
    pub fn register(&mut self, channel: Box<dyn CommChannel>) -> ChannelResult<()> {
        let id = channel.channel_id().to_string();
        let kind = channel.channel_kind();

        if kind == ChannelKind::Operator {
            if let Some(ref existing) = self.operator_id {
                if existing != &id {
                    return Err(ChannelError::DuplicateOperator);
                }
            }
            self.operator_id = Some(id.clone());
        }

        self.channels.insert(id, channel);
        Ok(())
    }

    /// Unregister a channel by ID.
    pub fn unregister(&mut self, channel_id: &str) -> ChannelResult<()> {
        if self.channels.remove(channel_id).is_none() {
            return Err(ChannelError::ChannelNotFound {
                channel_id: channel_id.to_string(),
            });
        }

        if self.operator_id.as_deref() == Some(channel_id) {
            self.operator_id = None;
        }

        Ok(())
    }

    /// Get a shared reference to a channel by ID.
    pub fn get(&self, channel_id: &str) -> Option<&dyn CommChannel> {
        self.channels.get(channel_id).map(|c| c.as_ref())
    }

    /// Get a mutable reference to a channel by ID.
    pub fn get_mut(&mut self, channel_id: &str) -> Option<&mut (dyn CommChannel + 'static)> {
        self.channels.get_mut(channel_id).map(|c| &mut **c)
    }

    /// Get a shared reference to the operator channel (if registered).
    pub fn operator(&self) -> Option<&dyn CommChannel> {
        self.operator_id
            .as_ref()
            .and_then(|id| self.channels.get(id))
            .map(|c| c.as_ref())
    }

    /// Get a mutable reference to the operator channel (if registered).
    pub fn operator_mut(&mut self) -> Option<&mut (dyn CommChannel + 'static)> {
        if let Some(ref id) = self.operator_id {
            self.channels.get_mut(id.as_str()).map(|c| &mut **c)
        } else {
            None
        }
    }

    /// Drain all pending inbound messages from all channels, in registration order.
    pub fn drain_all(&mut self) -> Vec<(String, InboundMessage)> {
        let mut messages = Vec::new();
        let ids: Vec<String> = self.channels.keys().cloned().collect();
        for id in ids {
            if let Some(ch) = self.channels.get_mut(&id) {
                while let Ok(Some(msg)) = ch.try_receive() {
                    messages.push((id.clone(), msg));
                }
            }
        }
        messages
    }

    /// List all registered channel IDs.
    pub fn channel_ids(&self) -> Vec<&str> {
        self.channels.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered channels.
    pub fn len(&self) -> usize {
        self.channels.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.channels.is_empty()
    }
}

impl Default for ChannelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for ChannelRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChannelRegistry")
            .field("channels", &self.channels.len())
            .field("operator_id", &self.operator_id)
            .finish()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal test channel for unit tests.
    struct TestChannel {
        id: String,
        kind: ChannelKind,
        capabilities: ChannelCapabilities,
        connected: bool,
    }

    impl TestChannel {
        fn new(id: &str, kind: ChannelKind) -> Self {
            Self {
                id: id.to_string(),
                kind,
                capabilities: ChannelCapabilities::for_kind(kind),
                connected: true,
            }
        }
    }

    impl CommChannel for TestChannel {
        fn channel_id(&self) -> &str {
            &self.id
        }
        fn channel_kind(&self) -> ChannelKind {
            self.kind
        }
        fn capabilities(&self) -> &ChannelCapabilities {
            &self.capabilities
        }
        fn try_receive(&mut self) -> ChannelResult<Option<InboundMessage>> {
            Ok(None)
        }
        fn send(&self, _msg: &OutboundMessage) -> ChannelResult<()> {
            Ok(())
        }
        fn is_connected(&self) -> bool {
            self.connected
        }
    }

    #[test]
    fn capability_presets_operator() {
        let caps = ChannelCapabilities::operator();
        assert!(caps.can_set_goals);
        assert!(caps.can_query);
        assert!(caps.can_assert);
        assert!(caps.can_run_cycles);
        assert!(caps.can_invoke_tools);
        assert!(caps.can_read_status);
        assert!(caps.can_configure);
        assert!(caps.can_persist);
        assert!(caps.can_manage_channels);
        assert!(caps.can_view_provenance);
        assert!(caps.rate_limit.is_none());
    }

    #[test]
    fn capability_presets_social() {
        let caps = ChannelCapabilities::social();
        assert!(!caps.can_set_goals);
        assert!(caps.can_query);
        assert!(!caps.can_assert);
        assert!(!caps.can_run_cycles);
        assert_eq!(caps.rate_limit, Some(30));
    }

    #[test]
    fn capability_presets_public() {
        let caps = ChannelCapabilities::public_readonly();
        assert!(!caps.can_set_goals);
        assert!(caps.can_query);
        assert!(!caps.can_assert);
        assert_eq!(caps.rate_limit, Some(10));
    }

    #[test]
    fn require_passes_for_allowed_action() {
        let caps = ChannelCapabilities::operator();
        assert!(caps.require("set_goals", "op-1", ChannelKind::Operator).is_ok());
        assert!(caps.require("query", "op-1", ChannelKind::Operator).is_ok());
    }

    #[test]
    fn require_fails_for_denied_action() {
        let caps = ChannelCapabilities::public_readonly();
        let result = caps.require("set_goals", "pub-1", ChannelKind::Public);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ChannelError::CapabilityDenied { .. }
        ));
    }

    #[test]
    fn require_fails_for_unknown_action() {
        let caps = ChannelCapabilities::operator();
        let result = caps.require("fly_to_mars", "op-1", ChannelKind::Operator);
        assert!(result.is_err());
    }

    #[test]
    fn for_kind_matches_factory() {
        assert_eq!(
            ChannelCapabilities::for_kind(ChannelKind::Operator),
            ChannelCapabilities::operator()
        );
        assert_eq!(
            ChannelCapabilities::for_kind(ChannelKind::Social),
            ChannelCapabilities::social()
        );
    }

    #[test]
    fn registry_register_and_lookup() {
        let mut reg = ChannelRegistry::new();
        assert!(reg.is_empty());

        reg.register(Box::new(TestChannel::new("op", ChannelKind::Operator)))
            .unwrap();
        assert_eq!(reg.len(), 1);
        assert!(reg.get("op").is_some());
        assert!(reg.operator().is_some());
    }

    #[test]
    fn registry_enforces_single_operator() {
        let mut reg = ChannelRegistry::new();
        reg.register(Box::new(TestChannel::new("op-1", ChannelKind::Operator)))
            .unwrap();

        let result = reg.register(Box::new(TestChannel::new("op-2", ChannelKind::Operator)));
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ChannelError::DuplicateOperator));
    }

    #[test]
    fn registry_allows_reregister_same_operator() {
        let mut reg = ChannelRegistry::new();
        reg.register(Box::new(TestChannel::new("op", ChannelKind::Operator)))
            .unwrap();
        // Re-registering the same ID should succeed (replace).
        reg.register(Box::new(TestChannel::new("op", ChannelKind::Operator)))
            .unwrap();
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn registry_unregister() {
        let mut reg = ChannelRegistry::new();
        reg.register(Box::new(TestChannel::new("ch-1", ChannelKind::Social)))
            .unwrap();
        assert_eq!(reg.len(), 1);

        reg.unregister("ch-1").unwrap();
        assert!(reg.is_empty());
        assert!(reg.get("ch-1").is_none());
    }

    #[test]
    fn registry_unregister_operator_clears_cache() {
        let mut reg = ChannelRegistry::new();
        reg.register(Box::new(TestChannel::new("op", ChannelKind::Operator)))
            .unwrap();
        assert!(reg.operator().is_some());

        reg.unregister("op").unwrap();
        assert!(reg.operator().is_none());
    }

    #[test]
    fn registry_unregister_not_found() {
        let mut reg = ChannelRegistry::new();
        let result = reg.unregister("nonexistent");
        assert!(matches!(
            result.unwrap_err(),
            ChannelError::ChannelNotFound { .. }
        ));
    }

    #[test]
    fn registry_multiple_non_operator_channels() {
        let mut reg = ChannelRegistry::new();
        reg.register(Box::new(TestChannel::new("s-1", ChannelKind::Social)))
            .unwrap();
        reg.register(Box::new(TestChannel::new("s-2", ChannelKind::Social)))
            .unwrap();
        reg.register(Box::new(TestChannel::new("p-1", ChannelKind::Public)))
            .unwrap();
        assert_eq!(reg.len(), 3);
        assert!(reg.operator().is_none());
    }

    #[test]
    fn registry_drain_all_empty() {
        let mut reg = ChannelRegistry::new();
        reg.register(Box::new(TestChannel::new("ch", ChannelKind::Social)))
            .unwrap();
        let msgs = reg.drain_all();
        assert!(msgs.is_empty());
    }

    #[test]
    fn registry_channel_ids() {
        let mut reg = ChannelRegistry::new();
        reg.register(Box::new(TestChannel::new("a", ChannelKind::Social)))
            .unwrap();
        reg.register(Box::new(TestChannel::new("b", ChannelKind::Trusted)))
            .unwrap();
        let mut ids = reg.channel_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn channel_kind_display() {
        assert_eq!(ChannelKind::Operator.to_string(), "Operator");
        assert_eq!(ChannelKind::Trusted.to_string(), "Trusted");
        assert_eq!(ChannelKind::Social.to_string(), "Social");
        assert_eq!(ChannelKind::Public.to_string(), "Public");
    }
}
