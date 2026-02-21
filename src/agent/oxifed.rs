//! Federation via oxifed — ActivityPub integration (Phase 12e).
//!
//! Akh-medu is an **application inside the oxifed system**. Oxifed handles all
//! ActivityPub protocol details (HTTP signatures, WebFinger, inbox/outbox,
//! federation, delivery). Akh-medu communicates through oxifed's AMQP message
//! bus and REST admin API.
//!
//! The [`OxifedChannel`] implements [`CommChannel`] with `ChannelKind::Social`
//! capabilities. A background tokio task consumes the AMQP inbox queue and
//! feeds a sync `VecDeque`; outbound messages are sent via an `mpsc` channel
//! to a publisher task.
//!
//! # Feature gate
//!
//! This module is only available with `--features oxifed`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::channel::{ChannelCapabilities, ChannelError, ChannelKind, ChannelResult, CommChannel};
use super::channel_message::{
    InboundMessage, InterlocutorId, MessageContent, OutboundMessage, ResponseContent,
};

// ── Error types ─────────────────────────────────────────────────────────

/// Errors from the oxifed federation subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum OxifedError {
    #[error("AMQP connection failed: {message}")]
    #[diagnostic(
        code(akh::oxifed::amqp_connect),
        help("Check that the AMQP broker (RabbitMQ/LavinMQ) is running and the URL is correct.")
    )]
    AmqpConnect { message: String },

    #[error("AMQP publish failed: {message}")]
    #[diagnostic(
        code(akh::oxifed::amqp_publish),
        help("The outbound message could not be published to the AMQP exchange.")
    )]
    AmqpPublish { message: String },

    #[error("oxifed REST API error: {status} — {message}")]
    #[diagnostic(
        code(akh::oxifed::api_error),
        help("Check that the oxifed admin API is reachable and the API token is valid.")
    )]
    ApiError { status: u16, message: String },

    #[error("channel not connected")]
    #[diagnostic(
        code(akh::oxifed::not_connected),
        help("Call OxifedChannel::connect() before using the channel.")
    )]
    NotConnected,

    #[error("deserialization failed: {message}")]
    #[diagnostic(
        code(akh::oxifed::deserialize),
        help("The AMQP message payload could not be parsed. Check oxifed version compatibility.")
    )]
    Deserialize { message: String },
}

pub type OxifedResult<T> = Result<T, OxifedError>;

// ── AMQP constants (matching oxifed) ────────────────────────────────────

/// Exchange for internal service messages (fanout).
pub const EXCHANGE_INTERNAL_PUBLISH: &str = "oxifed.internal.publish";

/// Exchange for ActivityPub activities ready for federation delivery (fanout).
pub const EXCHANGE_ACTIVITYPUB_PUBLISH: &str = "oxifed.activitypub.publish";

/// Exchange for incoming federation messages to process (fanout).
pub const EXCHANGE_INCOMING_PROCESS: &str = "oxifed.incoming.process";

/// Queue for internal activity messages.
pub const QUEUE_ACTIVITIES: &str = "oxifed.activities";

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for connecting to an oxifed instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OxifedConfig {
    /// AMQP broker URL (e.g., `amqp://guest:guest@localhost:5672`).
    pub amqp_url: String,
    /// Oxifed admin API base URL (e.g., `http://localhost:8081`).
    pub admin_api_url: String,
    /// Bearer token for admin API authentication.
    pub api_token: String,
    /// Domain name (e.g., `yourdomain.example`).
    pub domain: String,
    /// Actor username (e.g., `akh`). Combined with domain: `@akh@yourdomain.example`.
    pub actor_username: String,
    /// Custom inbox queue name. Defaults to `oxifed.app.{actor_username}`.
    pub inbox_queue: Option<String>,
}

impl OxifedConfig {
    /// The actor's full subject string (`username@domain`).
    pub fn subject(&self) -> String {
        format!("{}@{}", self.actor_username, self.domain)
    }

    /// The AMQP inbox queue name for this actor.
    pub fn inbox_queue_name(&self) -> String {
        self.inbox_queue
            .clone()
            .unwrap_or_else(|| format!("oxifed.app.{}", self.actor_username))
    }

    /// The actor's ActivityPub ID URL.
    pub fn actor_id(&self) -> String {
        format!("https://{}/users/{}", self.domain, self.actor_username)
    }
}

// ── Oxifed-compatible message types ─────────────────────────────────────
//
// These are serde-compatible with the oxifed crate's `messaging` module.
// We define them locally to avoid depending on oxifed's heavy dependency
// chain (mongodb, ring, axum, etc.).

/// Top-level message envelope matching oxifed's `MessageEnum`.
///
/// Only the variants akh-medu needs to produce/consume are included.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OxifedMessage {
    NoteCreateMessage(NoteCreate),
    NoteUpdateMessage(NoteUpdate),
    NoteDeleteMessage(NoteDelete),
    ProfileCreateMessage(ProfileCreate),
    ProfileUpdateMessage(ProfileUpdate),
    FollowActivityMessage(FollowActivity),
    LikeActivityMessage(LikeActivity),
    AnnounceActivityMessage(AnnounceActivity),
    IncomingObjectMessage(IncomingObject),
    IncomingActivityMessage(IncomingActivity),
}

/// Create a Note (post) on behalf of the akh actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteCreate {
    /// Author subject (`username@domain`).
    pub author: String,
    /// Note text content.
    pub content: String,
    /// Optional content warning / summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    /// Space-separated mention list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mentions: Option<String>,
    /// Space-separated tag list.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<String>,
    /// Additional JSON properties.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Value>,
}

/// Update an existing Note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteUpdate {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Value>,
}

/// Delete a Note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteDelete {
    pub id: String,
}

/// Create a profile for the akh actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCreate {
    /// Subject (`username@domain`).
    pub subject: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Value>,
}

/// Update the akh actor's profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileUpdate {
    pub subject: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub icon: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attachments: Option<Vec<serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<serde_json::Value>,
}

/// Follow activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FollowActivity {
    /// The actor initiating the follow.
    pub actor: String,
    /// The actor being followed.
    pub object: String,
}

/// Like activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LikeActivity {
    pub actor: String,
    pub object: String,
}

/// Announce (boost/share) activity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnounceActivity {
    pub action: String,
    #[serde(rename = "type")]
    pub activity_type: String,
    pub actor: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cc: Option<String>,
}

/// Incoming ActivityPub object from the fediverse (via oxifed pipeline).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncomingObject {
    /// Raw ActivityPub JSON.
    pub object: serde_json::Value,
    /// Type of the object (Note, Article, etc.).
    pub object_type: String,
    /// The actor who created this object.
    pub attributed_to: String,
    /// The domain this was received for.
    pub target_domain: String,
    /// The username this was addressed to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_username: Option<String>,
    /// ISO 8601 timestamp.
    pub received_at: String,
    /// Source IP or identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

/// Incoming ActivityPub activity from the fediverse (via oxifed pipeline).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncomingActivity {
    /// Raw ActivityPub JSON.
    pub activity: serde_json::Value,
    /// Activity type (Create, Follow, Like, Announce, Undo, etc.).
    pub activity_type: String,
    /// The actor who performed this activity.
    pub actor: String,
    /// The domain this was received for.
    pub target_domain: String,
    /// The username this was addressed to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_username: Option<String>,
    /// ISO 8601 timestamp.
    pub received_at: String,
    /// Source IP or identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
}

// ── Activity ↔ InboundMessage bridges ───────────────────────────────────

/// Convert an incoming oxifed object message to an `InboundMessage`.
///
/// Extracts text content from Note/Article objects and creates a
/// `MessageContent::Text` message from the attributed author.
pub fn incoming_object_to_inbound(
    obj: &IncomingObject,
    channel_id: &str,
) -> Option<InboundMessage> {
    // Extract text content from the raw AP JSON.
    let content = obj
        .object
        .get("content")
        .and_then(|v| v.as_str())
        .map(|s| strip_html_tags(s));

    let content = match content {
        Some(c) if !c.is_empty() => c,
        _ => return None, // No text content to process.
    };

    let sender = InterlocutorId::new(&obj.attributed_to);
    Some(InboundMessage::new(
        channel_id,
        sender,
        MessageContent::Text(content),
    ))
}

/// Convert an incoming oxifed activity message to an `InboundMessage`.
///
/// Handles Create (extracts object content), Follow, Like, Announce, etc.
pub fn incoming_activity_to_inbound(
    act: &IncomingActivity,
    channel_id: &str,
) -> Option<InboundMessage> {
    let sender = InterlocutorId::new(&act.actor);

    match act.activity_type.as_str() {
        "Create" => {
            // Extract content from the nested object.
            let content = act
                .activity
                .get("object")
                .and_then(|obj| obj.get("content"))
                .and_then(|v| v.as_str())
                .map(|s| strip_html_tags(s));

            let content = match content {
                Some(c) if !c.is_empty() => c,
                _ => return None,
            };

            Some(InboundMessage::new(
                channel_id,
                sender,
                MessageContent::Text(content),
            ))
        }
        "Follow" => {
            let object = act
                .activity
                .get("object")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Some(InboundMessage::new(
                channel_id,
                sender,
                MessageContent::Command {
                    name: "follow".to_string(),
                    args: Some(object.to_string()),
                },
            ))
        }
        "Like" | "Announce" => {
            let object = act
                .activity
                .get("object")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Some(InboundMessage::new(
                channel_id,
                sender,
                MessageContent::Command {
                    name: act.activity_type.to_lowercase(),
                    args: Some(object.to_string()),
                },
            ))
        }
        "Undo" => {
            // Undo wraps another activity — extract the inner type.
            let inner_type = act
                .activity
                .get("object")
                .and_then(|obj| obj.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            Some(InboundMessage::new(
                channel_id,
                sender,
                MessageContent::Command {
                    name: format!("undo-{}", inner_type.to_lowercase()),
                    args: None,
                },
            ))
        }
        _ => None, // Unhandled activity types are silently ignored.
    }
}

/// Convert an `OutboundMessage` to an oxifed `NoteCreate` for publishing.
///
/// Linearizes the response content to plain text for the Note body.
pub fn outbound_to_note(msg: &OutboundMessage, config: &OxifedConfig) -> NoteCreate {
    let content = match &msg.content {
        ResponseContent::Messages(messages) => {
            let mut text = String::new();
            for m in messages {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(&akh_message_text(m));
            }
            text
        }
        ResponseContent::Grounded { rendered, .. } => rendered.clone(),
    };

    NoteCreate {
        author: config.subject(),
        content,
        summary: None,
        mentions: None,
        tags: None,
        properties: None,
    }
}

/// Wrap a `NoteCreate` in an `OxifedMessage` envelope for AMQP publishing.
pub fn wrap_note(note: NoteCreate) -> OxifedMessage {
    OxifedMessage::NoteCreateMessage(note)
}

/// Wrap a `ProfileCreate` in an `OxifedMessage` envelope.
pub fn wrap_profile_create(profile: ProfileCreate) -> OxifedMessage {
    OxifedMessage::ProfileCreateMessage(profile)
}

// ── Message text extraction ─────────────────────────────────────────────

/// Extract the primary text content from an `AkhMessage`.
fn akh_message_text(msg: &crate::message::AkhMessage) -> String {
    use crate::message::AkhMessage;
    match msg {
        AkhMessage::Fact { text, .. } => text.clone(),
        AkhMessage::Reasoning { step, .. } => step.clone(),
        AkhMessage::Gap { description, .. } => description.clone(),
        AkhMessage::ToolResult { output, .. } => output.clone(),
        AkhMessage::Narrative { text, .. } => text.clone(),
        AkhMessage::System { text } => text.clone(),
        AkhMessage::Error { message, .. } => message.clone(),
        AkhMessage::GoalProgress { goal, status, .. } => format!("{goal}: {status}"),
        AkhMessage::Prompt { question } => question.clone(),
    }
}

// ── Simple HTML tag stripping ───────────────────────────────────────────

/// Strip HTML tags from content (ActivityPub content is often HTML).
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    result.trim().to_string()
}

// ── Outbound payload for the publisher task ─────────────────────────────

/// Payload sent through the mpsc channel to the AMQP publisher task.
#[derive(Debug)]
pub struct OutboundPayload {
    /// Serialized JSON of the `OxifedMessage` envelope.
    pub json: String,
    /// AMQP exchange to publish to.
    pub exchange: String,
    /// AMQP routing key (empty for fanout exchanges).
    pub routing_key: String,
}

// ── OxifedChannel ───────────────────────────────────────────────────────

/// A `CommChannel` implementation for oxifed federation.
///
/// Uses a background tokio task to consume AMQP messages and feed them
/// to a sync `VecDeque`. Outbound messages are sent via an `mpsc` channel
/// to a background publisher task.
///
/// # Construction
///
/// Use [`OxifedChannel::connect()`] from an async context (e.g., inside
/// `akhomed.rs`). It spawns the background tasks and returns a sync channel.
pub struct OxifedChannel {
    channel_id: String,
    capabilities: ChannelCapabilities,
    config: OxifedConfig,
    /// Inbox fed by the AMQP consumer background task.
    inbox: Arc<Mutex<VecDeque<InboundMessage>>>,
    /// Sender for outbound payloads (consumed by the publisher task).
    outbox_tx: std::sync::mpsc::Sender<OutboundPayload>,
    /// Whether the AMQP connection is alive.
    connected: Arc<AtomicBool>,
}

impl OxifedChannel {
    /// Create a new (disconnected) OxifedChannel.
    ///
    /// Use [`connect()`] to establish the AMQP connection and spawn
    /// background tasks. This constructor is useful for testing.
    pub fn new_disconnected(config: OxifedConfig) -> Self {
        let (outbox_tx, _outbox_rx) = std::sync::mpsc::channel();
        let channel_id = format!("oxifed:{}", config.actor_username);
        Self {
            channel_id,
            capabilities: ChannelCapabilities::social(),
            config,
            inbox: Arc::new(Mutex::new(VecDeque::new())),
            outbox_tx,
            connected: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Connect to oxifed's AMQP broker and spawn background tasks.
    ///
    /// Must be called from an async context (inside a tokio runtime).
    /// Returns the channel and a join handle for the consumer task.
    #[cfg(feature = "oxifed")]
    pub async fn connect(
        config: OxifedConfig,
    ) -> OxifedResult<Self> {
        use lapin::{Connection, ConnectionProperties, options::*, types::FieldTable};
        use futures_util::StreamExt;

        let channel_id = format!("oxifed:{}", config.actor_username);
        let inbox: Arc<Mutex<VecDeque<InboundMessage>>> =
            Arc::new(Mutex::new(VecDeque::new()));
        let connected = Arc::new(AtomicBool::new(false));

        // Connect to AMQP.
        let conn = Connection::connect(&config.amqp_url, ConnectionProperties::default())
            .await
            .map_err(|e| OxifedError::AmqpConnect {
                message: e.to_string(),
            })?;

        // Consumer channel.
        let consumer_ch = conn.create_channel().await.map_err(|e| {
            OxifedError::AmqpConnect {
                message: format!("consumer channel: {e}"),
            }
        })?;

        // Publisher channel with confirms.
        let publisher_ch = conn.create_channel().await.map_err(|e| {
            OxifedError::AmqpConnect {
                message: format!("publisher channel: {e}"),
            }
        })?;
        publisher_ch
            .confirm_select(ConfirmSelectOptions::default())
            .await
            .map_err(|e| OxifedError::AmqpConnect {
                message: format!("publisher confirms: {e}"),
            })?;

        // Declare the app's inbox queue (durable, auto-delete when no consumers).
        let queue_name = config.inbox_queue_name();
        consumer_ch
            .queue_declare(
                &queue_name,
                QueueDeclareOptions {
                    durable: true,
                    ..Default::default()
                },
                FieldTable::default(),
            )
            .await
            .map_err(|e| OxifedError::AmqpConnect {
                message: format!("queue declare: {e}"),
            })?;

        // Bind to the incoming process exchange.
        consumer_ch
            .queue_bind(
                &queue_name,
                EXCHANGE_INCOMING_PROCESS,
                "",
                QueueBindOptions::default(),
                FieldTable::default(),
            )
            .await
            .map_err(|e| OxifedError::AmqpConnect {
                message: format!("queue bind: {e}"),
            })?;

        connected.store(true, Ordering::SeqCst);

        // Extract values needed by background tasks and the return struct.
        let return_domain = config.domain.clone();
        let return_username = config.actor_username.clone();
        let return_inbox_queue = config.inbox_queue.clone();
        let consumer_tag = format!("akh-{}", config.actor_username);

        // Spawn consumer task.
        let inbox_clone = inbox.clone();
        let connected_clone = connected.clone();
        let channel_id_clone = channel_id.clone();
        tokio::spawn(async move {
            let mut consumer = match consumer_ch
                .basic_consume(
                    &queue_name,
                    &consumer_tag,
                    BasicConsumeOptions::default(),
                    FieldTable::default(),
                )
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!("AMQP consume failed: {e}");
                    connected_clone.store(false, Ordering::SeqCst);
                    return;
                }
            };

            while let Some(delivery) = consumer.next().await {
                match delivery {
                    Ok(delivery) => {
                        // Try to parse the message.
                        if let Ok(msg) =
                            serde_json::from_slice::<OxifedMessage>(&delivery.data)
                        {
                            let inbound = match &msg {
                                OxifedMessage::IncomingObjectMessage(obj) => {
                                    incoming_object_to_inbound(obj, &channel_id_clone)
                                }
                                OxifedMessage::IncomingActivityMessage(act) => {
                                    incoming_activity_to_inbound(act, &channel_id_clone)
                                }
                                _ => None,
                            };

                            if let Some(inbound) = inbound {
                                if let Ok(mut inbox) = inbox_clone.lock() {
                                    inbox.push_back(inbound);
                                }
                            }
                        }

                        // Acknowledge regardless (don't requeue unparseable messages).
                        let _ = delivery
                            .ack(BasicAckOptions::default())
                            .await;
                    }
                    Err(e) => {
                        tracing::warn!("AMQP delivery error: {e}");
                    }
                }
            }

            // Consumer stream ended — connection lost.
            connected_clone.store(false, Ordering::SeqCst);
        });

        // Spawn publisher task.
        let (outbox_tx, outbox_rx) = std::sync::mpsc::channel::<OutboundPayload>();

        tokio::spawn(async move {
            // Block on receiving from the sync mpsc channel.
            loop {
                let payload = match tokio::task::block_in_place(|| outbox_rx.recv()) {
                    Ok(p) => p,
                    Err(_) => break, // Channel closed.
                };

                let result = publisher_ch
                    .basic_publish(
                        &payload.exchange,
                        &payload.routing_key,
                        BasicPublishOptions::default(),
                        payload.json.as_bytes(),
                        lapin::BasicProperties::default()
                            .with_content_type("application/json".into())
                            .with_delivery_mode(2), // Persistent.
                    )
                    .await;

                if let Err(e) = result {
                    tracing::error!("AMQP publish failed: {e}");
                }
            }
        });

        Ok(Self {
            channel_id,
            capabilities: ChannelCapabilities::social(),
            config: OxifedConfig {
                amqp_url: String::new(), // Don't keep credentials in memory.
                admin_api_url: String::new(),
                api_token: String::new(),
                domain: return_domain,
                actor_username: return_username,
                inbox_queue: return_inbox_queue,
            },
            inbox,
            outbox_tx,
            connected,
        })
    }

    /// Get the OxifedConfig.
    pub fn config(&self) -> &OxifedConfig {
        &self.config
    }

    /// Get a cloneable handle for pushing inbound messages (for testing).
    pub fn inbound_handle(&self) -> OxifedInboundHandle {
        OxifedInboundHandle {
            inbox: self.inbox.clone(),
            channel_id: self.channel_id.clone(),
        }
    }
}

impl CommChannel for OxifedChannel {
    fn channel_id(&self) -> &str {
        &self.channel_id
    }

    fn channel_kind(&self) -> ChannelKind {
        ChannelKind::Social
    }

    fn capabilities(&self) -> &ChannelCapabilities {
        &self.capabilities
    }

    fn try_receive(&mut self) -> ChannelResult<Option<InboundMessage>> {
        let mut inbox = self.inbox.lock().map_err(|_| ChannelError::Transport {
            message: "inbox mutex poisoned".to_string(),
        })?;
        Ok(inbox.pop_front())
    }

    fn send(&self, msg: &OutboundMessage) -> ChannelResult<()> {
        if !self.connected.load(Ordering::SeqCst) {
            return Err(ChannelError::Transport {
                message: "oxifed channel not connected".to_string(),
            });
        }

        // Check constraint status — only emit passed messages.
        if !msg.constraint_check.is_passed() {
            return Ok(()); // Silently drop failed constraint checks.
        }

        let note = outbound_to_note(msg, &self.config);
        let envelope = wrap_note(note);
        let json = serde_json::to_string(&envelope).map_err(|e| ChannelError::Transport {
            message: format!("serialize: {e}"),
        })?;

        self.outbox_tx
            .send(OutboundPayload {
                json,
                exchange: EXCHANGE_INTERNAL_PUBLISH.to_string(),
                routing_key: String::new(),
            })
            .map_err(|e| ChannelError::Transport {
                message: format!("outbox send: {e}"),
            })?;

        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }
}

impl std::fmt::Debug for OxifedChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OxifedChannel")
            .field("channel_id", &self.channel_id)
            .field("domain", &self.config.domain)
            .field("actor", &self.config.actor_username)
            .field("connected", &self.connected.load(Ordering::SeqCst))
            .field("inbox_len", &self.inbox.lock().map(|q| q.len()).unwrap_or(0))
            .finish()
    }
}

// ── OxifedInboundHandle ─────────────────────────────────────────────────

/// Cloneable handle for pushing inbound messages into the channel.
///
/// Useful for testing or for bridging from non-AMQP sources.
#[derive(Clone)]
pub struct OxifedInboundHandle {
    inbox: Arc<Mutex<VecDeque<InboundMessage>>>,
    channel_id: String,
}

impl OxifedInboundHandle {
    /// Push a text message from a remote actor.
    pub fn push_text(&self, actor_id: &str, text: &str) {
        let msg = InboundMessage::new(
            &self.channel_id,
            InterlocutorId::new(actor_id),
            MessageContent::Text(text.to_string()),
        );
        if let Ok(mut inbox) = self.inbox.lock() {
            inbox.push_back(msg);
        }
    }

    /// Push a follow event.
    pub fn push_follow(&self, actor_id: &str) {
        let msg = InboundMessage::new(
            &self.channel_id,
            InterlocutorId::new(actor_id),
            MessageContent::Command {
                name: "follow".to_string(),
                args: None,
            },
        );
        if let Ok(mut inbox) = self.inbox.lock() {
            inbox.push_back(msg);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_subject() {
        let config = OxifedConfig {
            amqp_url: String::new(),
            admin_api_url: String::new(),
            api_token: String::new(),
            domain: "example.com".to_string(),
            actor_username: "akh".to_string(),
            inbox_queue: None,
        };
        assert_eq!(config.subject(), "akh@example.com");
        assert_eq!(config.actor_id(), "https://example.com/users/akh");
        assert_eq!(config.inbox_queue_name(), "oxifed.app.akh");
    }

    #[test]
    fn config_custom_inbox_queue() {
        let config = OxifedConfig {
            amqp_url: String::new(),
            admin_api_url: String::new(),
            api_token: String::new(),
            domain: "example.com".to_string(),
            actor_username: "akh".to_string(),
            inbox_queue: Some("custom.queue".to_string()),
        };
        assert_eq!(config.inbox_queue_name(), "custom.queue");
    }

    #[test]
    fn strip_html_basic() {
        assert_eq!(strip_html_tags("<p>Hello <b>world</b></p>"), "Hello world");
        assert_eq!(strip_html_tags("no tags"), "no tags");
        assert_eq!(strip_html_tags("<br/>"), "");
    }

    #[test]
    fn incoming_object_to_inbound_note() {
        let obj = IncomingObject {
            object: serde_json::json!({
                "type": "Note",
                "content": "<p>Hello from the fediverse!</p>",
                "attributedTo": "https://remote.com/users/alice"
            }),
            object_type: "Note".to_string(),
            attributed_to: "https://remote.com/users/alice".to_string(),
            target_domain: "example.com".to_string(),
            target_username: Some("akh".to_string()),
            received_at: "2026-02-21T12:00:00Z".to_string(),
            source: None,
        };

        let msg = incoming_object_to_inbound(&obj, "oxifed:akh").unwrap();
        assert_eq!(msg.sender.as_str(), "https://remote.com/users/alice");
        match &msg.content {
            MessageContent::Text(t) => assert_eq!(t, "Hello from the fediverse!"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn incoming_object_empty_content_returns_none() {
        let obj = IncomingObject {
            object: serde_json::json!({ "type": "Image" }),
            object_type: "Image".to_string(),
            attributed_to: "https://remote.com/users/bob".to_string(),
            target_domain: "example.com".to_string(),
            target_username: None,
            received_at: "2026-02-21T12:00:00Z".to_string(),
            source: None,
        };

        assert!(incoming_object_to_inbound(&obj, "oxifed:akh").is_none());
    }

    #[test]
    fn incoming_activity_create() {
        let act = IncomingActivity {
            activity: serde_json::json!({
                "type": "Create",
                "actor": "https://remote.com/users/alice",
                "object": {
                    "type": "Note",
                    "content": "<p>A reply!</p>"
                }
            }),
            activity_type: "Create".to_string(),
            actor: "https://remote.com/users/alice".to_string(),
            target_domain: "example.com".to_string(),
            target_username: Some("akh".to_string()),
            received_at: "2026-02-21T12:00:00Z".to_string(),
            source: None,
        };

        let msg = incoming_activity_to_inbound(&act, "oxifed:akh").unwrap();
        match &msg.content {
            MessageContent::Text(t) => assert_eq!(t, "A reply!"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn incoming_activity_follow() {
        let act = IncomingActivity {
            activity: serde_json::json!({
                "type": "Follow",
                "actor": "https://remote.com/users/alice",
                "object": "https://example.com/users/akh"
            }),
            activity_type: "Follow".to_string(),
            actor: "https://remote.com/users/alice".to_string(),
            target_domain: "example.com".to_string(),
            target_username: Some("akh".to_string()),
            received_at: "2026-02-21T12:00:00Z".to_string(),
            source: None,
        };

        let msg = incoming_activity_to_inbound(&act, "oxifed:akh").unwrap();
        match &msg.content {
            MessageContent::Command { name, args } => {
                assert_eq!(name, "follow");
                assert_eq!(
                    args.as_deref(),
                    Some("https://example.com/users/akh")
                );
            }
            _ => panic!("expected Command"),
        }
    }

    #[test]
    fn incoming_activity_like() {
        let act = IncomingActivity {
            activity: serde_json::json!({
                "type": "Like",
                "actor": "https://remote.com/users/bob",
                "object": "https://example.com/objects/123"
            }),
            activity_type: "Like".to_string(),
            actor: "https://remote.com/users/bob".to_string(),
            target_domain: "example.com".to_string(),
            target_username: None,
            received_at: "2026-02-21T12:00:00Z".to_string(),
            source: None,
        };

        let msg = incoming_activity_to_inbound(&act, "oxifed:akh").unwrap();
        match &msg.content {
            MessageContent::Command { name, .. } => assert_eq!(name, "like"),
            _ => panic!("expected Command"),
        }
    }

    #[test]
    fn incoming_activity_undo() {
        let act = IncomingActivity {
            activity: serde_json::json!({
                "type": "Undo",
                "actor": "https://remote.com/users/alice",
                "object": { "type": "Follow" }
            }),
            activity_type: "Undo".to_string(),
            actor: "https://remote.com/users/alice".to_string(),
            target_domain: "example.com".to_string(),
            target_username: None,
            received_at: "2026-02-21T12:00:00Z".to_string(),
            source: None,
        };

        let msg = incoming_activity_to_inbound(&act, "oxifed:akh").unwrap();
        match &msg.content {
            MessageContent::Command { name, .. } => assert_eq!(name, "undo-follow"),
            _ => panic!("expected Command"),
        }
    }

    #[test]
    fn incoming_activity_unknown_returns_none() {
        let act = IncomingActivity {
            activity: serde_json::json!({ "type": "Move" }),
            activity_type: "Move".to_string(),
            actor: "https://remote.com/users/x".to_string(),
            target_domain: "example.com".to_string(),
            target_username: None,
            received_at: "2026-02-21T12:00:00Z".to_string(),
            source: None,
        };

        assert!(incoming_activity_to_inbound(&act, "oxifed:akh").is_none());
    }

    #[test]
    fn outbound_to_note_from_messages() {
        let config = OxifedConfig {
            amqp_url: String::new(),
            admin_api_url: String::new(),
            api_token: String::new(),
            domain: "example.com".to_string(),
            actor_username: "akh".to_string(),
            inbox_queue: None,
        };

        let msg = OutboundMessage::single(crate::message::AkhMessage::system("test reply"));
        let note = outbound_to_note(&msg, &config);
        assert_eq!(note.author, "akh@example.com");
        assert!(note.content.contains("test reply"));
    }

    #[test]
    fn disconnected_channel_try_receive_empty() {
        let config = OxifedConfig {
            amqp_url: String::new(),
            admin_api_url: String::new(),
            api_token: String::new(),
            domain: "example.com".to_string(),
            actor_username: "akh".to_string(),
            inbox_queue: None,
        };

        let mut ch = OxifedChannel::new_disconnected(config);
        assert_eq!(ch.channel_kind(), ChannelKind::Social);
        assert!(!ch.is_connected());
        assert!(ch.try_receive().unwrap().is_none());
    }

    #[test]
    fn inbound_handle_push_and_receive() {
        let config = OxifedConfig {
            amqp_url: String::new(),
            admin_api_url: String::new(),
            api_token: String::new(),
            domain: "example.com".to_string(),
            actor_username: "akh".to_string(),
            inbox_queue: None,
        };

        let mut ch = OxifedChannel::new_disconnected(config);
        let handle = ch.inbound_handle();
        handle.push_text("https://remote.com/users/alice", "Hello!");
        handle.push_follow("https://remote.com/users/bob");

        let msg1 = ch.try_receive().unwrap().unwrap();
        assert_eq!(msg1.sender.as_str(), "https://remote.com/users/alice");

        let msg2 = ch.try_receive().unwrap().unwrap();
        match &msg2.content {
            MessageContent::Command { name, .. } => assert_eq!(name, "follow"),
            _ => panic!("expected follow command"),
        }

        assert!(ch.try_receive().unwrap().is_none());
    }

    #[test]
    fn note_create_serde_round_trip() {
        let note = NoteCreate {
            author: "akh@example.com".to_string(),
            content: "Hello fediverse!".to_string(),
            summary: None,
            mentions: Some("@alice@remote.com".to_string()),
            tags: None,
            properties: None,
        };

        let envelope = wrap_note(note.clone());
        let json = serde_json::to_string(&envelope).unwrap();
        let parsed: OxifedMessage = serde_json::from_str(&json).unwrap();

        match parsed {
            OxifedMessage::NoteCreateMessage(n) => {
                assert_eq!(n.author, "akh@example.com");
                assert_eq!(n.content, "Hello fediverse!");
                assert_eq!(n.mentions.as_deref(), Some("@alice@remote.com"));
            }
            _ => panic!("expected NoteCreateMessage"),
        }
    }

    #[test]
    fn message_serde_compatibility() {
        // Test that our types produce JSON compatible with oxifed's format.
        let follow = FollowActivity {
            actor: "https://example.com/users/akh".to_string(),
            object: "https://remote.com/users/alice".to_string(),
        };
        let json = serde_json::to_string(&follow).unwrap();
        assert!(json.contains("\"actor\""));
        assert!(json.contains("\"object\""));

        // Parse back.
        let parsed: FollowActivity = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.actor, follow.actor);
    }
}
