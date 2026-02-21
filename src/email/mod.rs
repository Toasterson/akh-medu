//! Email channel subsystem: JMAP/IMAP connector, MIME parsing, JWZ threading,
//! composition, and CommChannel implementation.
//!
//! Feature-gated under `email`. Provides `EmailChannel` (implementing the
//! `CommChannel` trait from Phase 12a) with background polling via `std::thread`.

pub mod classify;
pub mod compose;
pub mod connector;
pub mod error;
pub mod parser;
pub mod threading;
pub mod triage;

pub use classify::{ClassificationResult, SpamClassifier, SpamDecision};
pub use compose::{ComposedEmail, compose_new, compose_reply, to_mime};
pub use connector::{
    ConnectionType, EmailConfig, EmailConnector, EmailCredentials, ImapConnector, JmapConnector,
    MockConnector, RawEmail,
};
pub use error::{EmailError, EmailResult};
pub use parser::{ParsedEmail, extract_domain, parse_raw};
pub use threading::{ThreadNode, ThreadTree, build_threads};
pub use triage::{EmailRoute, SenderStats, TriageEngine, TriagePredicates, TriageResult};

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::agent::channel::{ChannelCapabilities, ChannelKind, ChannelResult, CommChannel};
use crate::agent::channel_message::{
    InboundMessage, InterlocutorId, MessageContent, OutboundMessage,
};
use crate::engine::Engine;
use crate::error::AkhResult;
use crate::symbol::SymbolId;

// ── EmailPredicates ─────────────────────────────────────────────────────

/// Well-known KG predicates for email message modeling.
///
/// Follows the same pattern as `InterlocutorPredicates` and `AgentPredicates`.
#[derive(Debug, Clone)]
pub struct EmailPredicates {
    /// `email:message-id` — RFC 5322 Message-ID.
    pub message_id: SymbolId,
    /// `email:from` — sender address.
    pub from: SymbolId,
    /// `email:to` — recipient address.
    pub to: SymbolId,
    /// `email:cc` — CC address.
    pub cc: SymbolId,
    /// `email:subject` — subject line.
    pub subject: SymbolId,
    /// `email:date` — send date (unix timestamp).
    pub date: SymbolId,
    /// `email:thread-id` — JWZ thread root message ID.
    pub thread_id: SymbolId,
    /// `email:in-reply-to` — parent message ID.
    pub in_reply_to: SymbolId,
    /// `email:has-attachment` — whether attachments are present.
    pub has_attachment: SymbolId,
    /// `email:content-type` — MIME content type.
    pub content_type: SymbolId,
    /// `email:body-text` — plain text body.
    pub body_text: SymbolId,
    /// `email:list-id` — mailing list identifier.
    pub list_id: SymbolId,
    /// `email:dkim-pass` — DKIM verification status.
    pub dkim_pass: SymbolId,
    /// `email:spf-pass` — SPF verification status.
    pub spf_pass: SymbolId,
}

impl EmailPredicates {
    /// Resolve or create all email predicates in the engine.
    pub fn init(engine: &Engine) -> AkhResult<Self> {
        Ok(Self {
            message_id: engine.resolve_or_create_relation("email:message-id")?,
            from: engine.resolve_or_create_relation("email:from")?,
            to: engine.resolve_or_create_relation("email:to")?,
            cc: engine.resolve_or_create_relation("email:cc")?,
            subject: engine.resolve_or_create_relation("email:subject")?,
            date: engine.resolve_or_create_relation("email:date")?,
            thread_id: engine.resolve_or_create_relation("email:thread-id")?,
            in_reply_to: engine.resolve_or_create_relation("email:in-reply-to")?,
            has_attachment: engine.resolve_or_create_relation("email:has-attachment")?,
            content_type: engine.resolve_or_create_relation("email:content-type")?,
            body_text: engine.resolve_or_create_relation("email:body-text")?,
            list_id: engine.resolve_or_create_relation("email:list-id")?,
            dkim_pass: engine.resolve_or_create_relation("email:dkim-pass")?,
            spf_pass: engine.resolve_or_create_relation("email:spf-pass")?,
        })
    }
}

// ── EmailInboundHandle ──────────────────────────────────────────────────

/// Cloneable handle for pushing inbound email messages into an `EmailChannel`.
///
/// The background polling thread and tests use this handle to enqueue
/// parsed emails; the agent drains them via `CommChannel::try_receive()`.
#[derive(Debug, Clone)]
pub struct EmailInboundHandle {
    channel_id: String,
    queue: Arc<Mutex<VecDeque<InboundMessage>>>,
}

impl EmailInboundHandle {
    /// Push a parsed email as an inbound message.
    ///
    /// Converts the email's subject + body into `MessageContent::Text`.
    /// The sender becomes the `InterlocutorId`.
    pub fn push_email(&self, parsed: &ParsedEmail) {
        let text = format!(
            "[Email] {}\n\n{}",
            parsed.subject,
            parsed.best_body().unwrap_or("[no body]")
        );
        let msg = InboundMessage::new(
            &self.channel_id,
            InterlocutorId::new(&parsed.from),
            MessageContent::Text(text),
        );
        self.queue.lock().unwrap().push_back(msg);
    }

    /// Push a raw text message (for testing or direct injection).
    pub fn push_text(&self, text: impl Into<String>, sender: &str) {
        let msg = InboundMessage::new(
            &self.channel_id,
            InterlocutorId::new(sender),
            MessageContent::Text(text.into()),
        );
        self.queue.lock().unwrap().push_back(msg);
    }

    /// Number of pending messages in the queue.
    pub fn pending(&self) -> usize {
        self.queue.lock().unwrap().len()
    }
}

// ── EmailChannel ────────────────────────────────────────────────────────

/// An email-backed communication channel implementing `CommChannel`.
///
/// Polls for new messages via a background `std::thread` and fills
/// a shared `VecDeque<InboundMessage>` that `try_receive()` drains.
pub struct EmailChannel {
    id: String,
    capabilities: ChannelCapabilities,
    predicates: EmailPredicates,
    inbound: Arc<Mutex<VecDeque<InboundMessage>>>,
    config: EmailConfig,
    poll_handle: Option<std::thread::JoinHandle<()>>,
    connected: Arc<AtomicBool>,
    /// Shutdown signal for the polling thread.
    shutdown: Arc<AtomicBool>,
}

impl EmailChannel {
    /// Create a new email channel.
    ///
    /// Initializes predicates in the engine but does not start polling.
    /// Call `start_polling()` to begin background fetching.
    pub fn new(config: EmailConfig, engine: &Engine) -> EmailResult<Self> {
        config.validate()?;
        let predicates = EmailPredicates::init(engine)?;

        let user = match &config.credentials {
            EmailCredentials::AppPassword { user, .. } => user.clone(),
            EmailCredentials::OAuth2 { .. } => "oauth-user".to_string(),
        };
        let id = format!("email:{}", user);

        Ok(Self {
            id,
            capabilities: ChannelCapabilities::social(),
            predicates,
            inbound: Arc::new(Mutex::new(VecDeque::new())),
            config,
            poll_handle: None,
            connected: Arc::new(AtomicBool::new(false)),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Get the email predicates for KG operations.
    pub fn predicates(&self) -> &EmailPredicates {
        &self.predicates
    }

    /// Get a cloneable inbound handle for the polling thread or tests.
    pub fn inbound_handle(&self) -> EmailInboundHandle {
        EmailInboundHandle {
            channel_id: self.id.clone(),
            queue: Arc::clone(&self.inbound),
        }
    }

    /// Start the background polling thread.
    ///
    /// The thread creates a connector (JMAP or IMAP based on config),
    /// polls at `config.poll_interval`, parses results, and pushes
    /// them to the inbound queue.
    pub fn start_polling(&mut self) {
        if self.poll_handle.is_some() {
            return; // Already polling.
        }

        let config = self.config.clone();
        let handle = self.inbound_handle();
        let connected = Arc::clone(&self.connected);
        let shutdown = Arc::clone(&self.shutdown);

        let join_handle = std::thread::spawn(move || {
            let mut connector: Box<dyn EmailConnector> = match config.connection_type {
                ConnectionType::Jmap => {
                    let mut jmap = JmapConnector::new(config.clone());
                    if let Err(e) = jmap.discover() {
                        tracing::error!("JMAP discovery failed: {e}");
                        return;
                    }
                    Box::new(jmap)
                }
                ConnectionType::Imap => Box::new(ImapConnector::new(config.clone())),
            };

            connected.store(true, Ordering::SeqCst);

            let interval = config.poll_interval();
            while !shutdown.load(Ordering::SeqCst) {
                match connector.fetch_new() {
                    Ok(raw_emails) => {
                        for raw in &raw_emails {
                            match parse_raw(raw) {
                                Ok(parsed) => handle.push_email(&parsed),
                                Err(e) => {
                                    tracing::warn!(
                                        "Failed to parse email uid={}: {e}",
                                        raw.uid
                                    );
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Email fetch failed: {e}");
                    }
                }
                std::thread::sleep(interval);
            }

            connected.store(false, Ordering::SeqCst);
        });

        self.poll_handle = Some(join_handle);
    }

    /// Stop the background polling thread.
    pub fn stop_polling(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.poll_handle.take() {
            // Don't block indefinitely — the thread will exit on next poll.
            let _ = handle.join();
        }
    }

    /// The email configuration.
    pub fn config(&self) -> &EmailConfig {
        &self.config
    }
}

impl CommChannel for EmailChannel {
    fn channel_id(&self) -> &str {
        &self.id
    }

    fn channel_kind(&self) -> ChannelKind {
        ChannelKind::Social
    }

    fn capabilities(&self) -> &ChannelCapabilities {
        &self.capabilities
    }

    fn try_receive(&mut self) -> ChannelResult<Option<InboundMessage>> {
        Ok(self.inbound.lock().unwrap().pop_front())
    }

    fn send(&self, msg: &OutboundMessage) -> ChannelResult<()> {
        // Outbound email is sent via the compose + SMTP pipeline,
        // not through the CommChannel::send() method directly.
        // This method is a no-op — the agent uses the email tools instead.
        let _ = msg;
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for EmailChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmailChannel")
            .field("id", &self.id)
            .field("connected", &self.connected.load(Ordering::Relaxed))
            .field(
                "pending_inbound",
                &self.inbound.lock().unwrap().len(),
            )
            .field("config_host", &self.config.host)
            .finish()
    }
}

impl Drop for EmailChannel {
    fn drop(&mut self) {
        self.stop_polling();
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    fn test_engine() -> Engine {
        Engine::new(EngineConfig::default()).unwrap()
    }

    fn test_config() -> EmailConfig {
        EmailConfig {
            connection_type: ConnectionType::Imap,
            host: "imap.test.com".to_string(),
            port: 993,
            credentials: EmailCredentials::AppPassword {
                user: "testuser".to_string(),
                pass: "testpass".to_string(),
            },
            poll_interval_secs: 60,
            mailboxes: vec!["INBOX".to_string()],
            smtp_host: None,
            smtp_port: None,
        }
    }

    #[test]
    fn predicates_init() {
        let engine = test_engine();
        let preds = EmailPredicates::init(&engine).unwrap();

        // All predicate IDs should be distinct.
        let ids = [
            preds.message_id,
            preds.from,
            preds.to,
            preds.cc,
            preds.subject,
            preds.date,
            preds.thread_id,
            preds.in_reply_to,
            preds.has_attachment,
            preds.content_type,
            preds.body_text,
            preds.list_id,
            preds.dkim_pass,
            preds.spf_pass,
        ];
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 14);
    }

    #[test]
    fn predicates_idempotent() {
        let engine = test_engine();
        let p1 = EmailPredicates::init(&engine).unwrap();
        let p2 = EmailPredicates::init(&engine).unwrap();
        assert_eq!(p1.message_id, p2.message_id);
        assert_eq!(p1.from, p2.from);
    }

    #[test]
    fn email_channel_construction() {
        let engine = test_engine();
        let config = test_config();
        let channel = EmailChannel::new(config, &engine).unwrap();

        assert_eq!(channel.channel_id(), "email:testuser");
        assert_eq!(channel.channel_kind(), ChannelKind::Social);
        assert!(!channel.is_connected());
    }

    #[test]
    fn email_channel_invalid_config() {
        let engine = test_engine();
        let config = EmailConfig {
            host: String::new(),
            ..test_config()
        };
        assert!(EmailChannel::new(config, &engine).is_err());
    }

    #[test]
    fn inbound_handle_push_and_receive() {
        let engine = test_engine();
        let config = test_config();
        let mut channel = EmailChannel::new(config, &engine).unwrap();
        let handle = channel.inbound_handle();

        handle.push_text("hello from email", "alice@example.com");
        assert_eq!(handle.pending(), 1);

        let msg = channel.try_receive().unwrap().unwrap();
        assert_eq!(msg.text(), Some("hello from email"));
        assert_eq!(msg.sender.as_str(), "alice@example.com");

        assert!(channel.try_receive().unwrap().is_none());
    }

    #[test]
    fn inbound_handle_push_email() {
        let engine = test_engine();
        let config = test_config();
        let mut channel = EmailChannel::new(config, &engine).unwrap();
        let handle = channel.inbound_handle();

        let parsed = ParsedEmail {
            message_id: "<test@example>".to_string(),
            from: "sender@example.com".to_string(),
            from_display: Some("Sender".to_string()),
            to: vec!["me@example.com".to_string()],
            cc: Vec::new(),
            subject: "Test Subject".to_string(),
            date: Some(1700000000),
            in_reply_to: None,
            references: Vec::new(),
            body_text: Some("Email body here".to_string()),
            body_html: None,
            has_attachments: false,
            list_id: None,
            content_type: "text/plain".to_string(),
        };

        handle.push_email(&parsed);
        let msg = channel.try_receive().unwrap().unwrap();
        let text = msg.text().unwrap();
        assert!(text.contains("Test Subject"));
        assert!(text.contains("Email body here"));
        assert_eq!(msg.sender.as_str(), "sender@example.com");
    }

    #[test]
    fn inbound_handle_fifo_order() {
        let engine = test_engine();
        let config = test_config();
        let mut channel = EmailChannel::new(config, &engine).unwrap();
        let handle = channel.inbound_handle();

        handle.push_text("first", "a@x.com");
        handle.push_text("second", "b@x.com");
        handle.push_text("third", "c@x.com");

        let m1 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m1.text(), Some("first"));
        let m2 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m2.text(), Some("second"));
        let m3 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m3.text(), Some("third"));
        assert!(channel.try_receive().unwrap().is_none());
    }

    #[test]
    fn inbound_handle_is_cloneable() {
        let engine = test_engine();
        let config = test_config();
        let mut channel = EmailChannel::new(config, &engine).unwrap();
        let h1 = channel.inbound_handle();
        let h2 = h1.clone();

        h1.push_text("from h1", "a@x.com");
        h2.push_text("from h2", "b@x.com");

        assert_eq!(h1.pending(), 2);
        assert_eq!(h2.pending(), 2); // Same queue.

        let m1 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m1.text(), Some("from h1"));
        let m2 = channel.try_receive().unwrap().unwrap();
        assert_eq!(m2.text(), Some("from h2"));
    }

    #[test]
    fn channel_capabilities_are_social() {
        let engine = test_engine();
        let config = test_config();
        let channel = EmailChannel::new(config, &engine).unwrap();
        let caps = channel.capabilities();

        assert!(caps.can_query);
        assert!(!caps.can_set_goals);
        assert!(!caps.can_assert);
        assert!(!caps.can_configure);
        assert!(caps.rate_limit.is_some());
    }

    #[test]
    fn channel_send_is_noop() {
        let engine = test_engine();
        let config = test_config();
        let channel = EmailChannel::new(config, &engine).unwrap();

        let msg = OutboundMessage::from_akh_messages(vec![]);
        assert!(channel.send(&msg).is_ok());
    }

    #[test]
    fn channel_debug_format() {
        let engine = test_engine();
        let config = test_config();
        let channel = EmailChannel::new(config, &engine).unwrap();
        let debug = format!("{channel:?}");
        assert!(debug.contains("EmailChannel"));
        assert!(debug.contains("imap.test.com"));
    }
}
