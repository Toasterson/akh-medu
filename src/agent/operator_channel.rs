//! Concrete operator channel implementation.
//!
//! `OperatorChannel` wraps the existing `MessageSink` to implement `CommChannel`
//! with full `Operator` capabilities. An `InboundHandle` allows the UI event loop
//! (TUI, headless REPL) to push inbound messages that the agent can drain.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::message::MessageSink;

use super::channel::{ChannelCapabilities, ChannelKind, ChannelResult, CommChannel};
use super::channel_message::{
    InboundMessage, InterlocutorId, MessageContent, OutboundMessage,
};

// ── InboundHandle ────────────────────────────────────────────────────────

/// Cloneable handle for pushing inbound messages into an `OperatorChannel`.
///
/// The TUI or headless event loop holds this handle and enqueues messages;
/// the agent drains them via `CommChannel::try_receive()`.
#[derive(Debug, Clone)]
pub struct InboundHandle {
    channel_id: String,
    queue: Arc<Mutex<VecDeque<InboundMessage>>>,
}

impl InboundHandle {
    /// Push a text message from the operator.
    pub fn push_text(&self, text: impl Into<String>) {
        let msg = InboundMessage::new(
            &self.channel_id,
            InterlocutorId::operator(),
            MessageContent::Text(text.into()),
        );
        self.queue.lock().unwrap().push_back(msg);
    }

    /// Push a command message from the operator.
    pub fn push_command(&self, name: impl Into<String>, args: Option<String>) {
        let msg = InboundMessage::new(
            &self.channel_id,
            InterlocutorId::operator(),
            MessageContent::Command {
                name: name.into(),
                args,
            },
        );
        self.queue.lock().unwrap().push_back(msg);
    }

    /// Number of pending messages in the queue.
    pub fn pending(&self) -> usize {
        self.queue.lock().unwrap().len()
    }
}

// ── OperatorChannel ──────────────────────────────────────────────────────

/// The operator's communication channel.
///
/// Bridges the existing `MessageSink` (used for outbound rendering) with the
/// new `CommChannel` protocol. Inbound messages are enqueued via `InboundHandle`.
pub struct OperatorChannel {
    id: String,
    capabilities: ChannelCapabilities,
    sink: Arc<dyn MessageSink>,
    inbound: Arc<Mutex<VecDeque<InboundMessage>>>,
    operator_id: InterlocutorId,
}

impl OperatorChannel {
    /// Create a new operator channel wrapping an existing message sink.
    pub fn new(sink: Arc<dyn MessageSink>) -> Self {
        let id = "operator".to_string();
        let inbound = Arc::new(Mutex::new(VecDeque::new()));
        Self {
            id,
            capabilities: ChannelCapabilities::operator(),
            sink,
            inbound,
            operator_id: InterlocutorId::operator(),
        }
    }

    /// Get a cloneable inbound handle for the UI event loop to push messages.
    pub fn inbound_handle(&self) -> InboundHandle {
        InboundHandle {
            channel_id: self.id.clone(),
            queue: Arc::clone(&self.inbound),
        }
    }

    /// Access the underlying message sink (for legacy code paths that still
    /// need direct `MessageSink` access).
    pub fn sink(&self) -> &dyn MessageSink {
        self.sink.as_ref()
    }

    /// The operator's interlocutor ID.
    pub fn operator_id(&self) -> &InterlocutorId {
        &self.operator_id
    }
}

impl CommChannel for OperatorChannel {
    fn channel_id(&self) -> &str {
        &self.id
    }

    fn channel_kind(&self) -> ChannelKind {
        ChannelKind::Operator
    }

    fn capabilities(&self) -> &ChannelCapabilities {
        &self.capabilities
    }

    fn try_receive(&mut self) -> ChannelResult<Option<InboundMessage>> {
        Ok(self.inbound.lock().unwrap().pop_front())
    }

    fn send(&self, msg: &OutboundMessage) -> ChannelResult<()> {
        let akh_messages = msg.to_akh_messages();
        self.sink.emit_batch(&akh_messages);
        Ok(())
    }

    fn is_connected(&self) -> bool {
        // The operator channel is always connected while the process runs.
        true
    }
}

impl std::fmt::Debug for OperatorChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OperatorChannel")
            .field("id", &self.id)
            .field("operator_id", &self.operator_id)
            .field(
                "pending_inbound",
                &self.inbound.lock().unwrap().len(),
            )
            .finish()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{AkhMessage, VecSink};

    #[test]
    fn try_receive_fifo_order() {
        let sink = Arc::new(VecSink::new());
        let mut channel = OperatorChannel::new(sink);
        let handle = channel.inbound_handle();

        handle.push_text("first");
        handle.push_text("second");
        handle.push_text("third");

        let m1 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m1.text(), Some("first"));

        let m2 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m2.text(), Some("second"));

        let m3 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m3.text(), Some("third"));

        assert!(channel.try_receive().unwrap().is_none());
    }

    #[test]
    fn send_emits_through_sink() {
        let sink = Arc::new(VecSink::new());
        let channel = OperatorChannel::new(Arc::clone(&sink) as Arc<dyn MessageSink>);

        let out = OutboundMessage::single(AkhMessage::system("hello from channel"));
        channel.send(&out).unwrap();

        let collected = sink.messages();
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn send_batch_emits_all() {
        let sink = Arc::new(VecSink::new());
        let channel = OperatorChannel::new(Arc::clone(&sink) as Arc<dyn MessageSink>);

        let out = OutboundMessage::from_akh_messages(vec![
            AkhMessage::system("one"),
            AkhMessage::system("two"),
            AkhMessage::fact("three"),
        ]);
        channel.send(&out).unwrap();

        let collected = sink.messages();
        assert_eq!(collected.len(), 3);
    }

    #[test]
    fn inbound_handle_push_command() {
        let sink = Arc::new(VecSink::new());
        let mut channel = OperatorChannel::new(sink);
        let handle = channel.inbound_handle();

        handle.push_command("status", None);
        handle.push_command("run", Some("5".to_string()));

        let m1 = channel.try_receive().unwrap().unwrap();
        assert!(matches!(
            m1.content,
            MessageContent::Command { ref name, .. } if name == "status"
        ));

        let m2 = channel.try_receive().unwrap().unwrap();
        assert!(matches!(
            m2.content,
            MessageContent::Command { ref name, ref args } if name == "run" && args.as_deref() == Some("5")
        ));
    }

    #[test]
    fn inbound_handle_is_cloneable() {
        let sink = Arc::new(VecSink::new());
        let mut channel = OperatorChannel::new(sink);
        let h1 = channel.inbound_handle();
        let h2 = h1.clone();

        h1.push_text("from h1");
        h2.push_text("from h2");

        assert_eq!(handle_pending(&channel), 2);

        let m1 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m1.text(), Some("from h1"));
        let m2 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m2.text(), Some("from h2"));
    }

    fn handle_pending(ch: &OperatorChannel) -> usize {
        ch.inbound.lock().unwrap().len()
    }

    #[test]
    fn channel_metadata() {
        let sink = Arc::new(VecSink::new());
        let channel = OperatorChannel::new(sink);

        assert_eq!(channel.channel_id(), "operator");
        assert_eq!(channel.channel_kind(), ChannelKind::Operator);
        assert!(channel.is_connected());
        assert!(channel.capabilities().can_set_goals);
        assert!(channel.capabilities().can_configure);
        assert!(channel.operator_id().is_operator());
    }

    #[test]
    fn pending_count() {
        let sink = Arc::new(VecSink::new());
        let channel = OperatorChannel::new(sink);
        let handle = channel.inbound_handle();

        assert_eq!(handle.pending(), 0);
        handle.push_text("msg");
        assert_eq!(handle.pending(), 1);
    }
}
