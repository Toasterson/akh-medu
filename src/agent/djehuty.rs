//! Typed client for the Djehuty Board agent-coordination service.
//!
//! The board (see <https://code.aopc.cloud/toasterson/djehuty-board>) is an
//! API-first coordination surface for agents: authenticated agents post typed
//! messages into private inboxes and public threads, and read threads through
//! replayable cursor pagination. This module is the **client side only** — it
//! owns the typed models mirroring the board's wire format, authenticated
//! posting, deterministic forward polling, and a standalone sidecar
//! checkpoint API. The resident daemon integration uses the same typed page
//! API but stores its fan-out cursor in engine metadata (`board_reactor`) so
//! cursor advancement and durable reactor events share one storage boundary.
//!
//! # Wire contract (board main, commit `a83d7f97` "feat: add replayable
//! thread cursors")
//!
//! * Thread reads are `GET /v1/threads/{thread_id}/messages` with optional
//!   exclusive `after_message_id` / `before_message_id` boundaries and a
//!   `limit` (service bounds pages to 200). Pages are sorted by
//!   `(created_at, message_id)` so equal timestamps have a stable
//!   tie-breaker — forward polling is therefore deterministic.
//! * While another page exists the response carries a `next-cursor`
//!   header (`MESSAGE_UUID`); the header is absent at the current end. The
//!   body is the plain JSON array of messages.
//! * Writes (`POST /v1/threads/{thread_id}/messages`,
//!   `POST /v1/inbox/messages`) take a `MessageContent` envelope with a
//!   caller-selected `idempotency_key` that makes each write retry-safe:
//!   replays return the originally stored message instead of a duplicate.
//! * Identity comes from a hashed bearer credential; the author is derived
//!   server-side and cannot be chosen by the caller.
//! * Errors are `{ "error": { "code": <symbol>, "message": <detail> } }`.
//!
//! # Crash safety
//!
//! The subscription checkpoint is written through the engine's standard
//! durable-metadata pattern (JSON sidecar written to a sibling `.tmp` file,
//! then `rename(2)`d into place, as in `vsa::item_memory::dump_to_disk`).
//! Rename within a directory is atomic, so a crash mid-write leaves either
//! the previous checkpoint or the new one — never a torn file. A load
//! tolerates a missing or corrupt file by returning the default (resume
//! from the thread head) rather than failing startup. The watermark
//! advances only after a page has been fully handed to the consumer and the
//! checkpoint re-saved, so a crash between receiving and saving replays at
//! most one page.
//!
//! Configuration follows the `openclaw` convention: credentials and endpoint
//! come from environment variables so secrets never enter source or config
//! files:
//!   `DJEHUTY_BOARD_TOKEN`   — bearer token (required when enabled)
//!   `DJEHUTY_BOARD_URL`     — board base URL (default `http://127.0.0.1:8080`)
//!   `DJEHUTY_AGENT_ID`      — authenticated agent UUID (required by reactor ingress)

use std::path::Path;
use std::time::Duration;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors raised by the Djehuty Board client.
#[derive(Debug, Error, Diagnostic)]
pub enum DjehutyError {
    /// The board bearer token was not configured.
    #[error("Djehuty Board token not configured")]
    #[diagnostic(
        code(akh::djehuty::no_token),
        help("Set the DJEHUTY_BOARD_TOKEN environment variable to the board bearer token.")
    )]
    NoToken,

    /// The board rejected a request with a non-envelope error body.
    #[error("Djehuty Board request failed (HTTP {status}): {message}")]
    #[diagnostic(
        code(akh::djehuty::api_error),
        help(
            "Check the bearer token (DJEHUTY_BOARD_TOKEN) and the payload against the board's registered message types."
        )
    )]
    ApiError { status: u16, message: String },

    /// A structured board error envelope with a known symbolic code.
    #[error("Djehuty Board error (HTTP {status}, {code}): {message}")]
    #[diagnostic(
        code(akh::djehuty::board_error),
        help(
            "The board rejected the request with a structured error envelope (invalid_input, not_found, conflict, ...)."
        )
    )]
    BoardError {
        status: u16,
        code: String,
        message: String,
    },

    /// The board could not be reached.
    #[error("Djehuty Board transport error: {message}")]
    #[diagnostic(
        code(akh::djehuty::transport),
        help(
            "Verify DJEHUTY_BOARD_URL is reachable and the board service is running (/health/ready)."
        )
    )]
    Transport { message: String },

    /// The response could not be decoded against the typed models.
    #[error("Djehuty Board response parse error: {message}")]
    #[diagnostic(
        code(akh::djehuty::parse),
        help("The board returned unexpected JSON. Check for board/client version mismatches.")
    )]
    Parse { message: String },

    /// The client refused to run because its configuration is inconsistent.
    #[error("Djehuty Board client misconfigured: {reason}")]
    #[diagnostic(
        code(akh::djehuty::invalid_config),
        help(
            "Fix the client configuration before use (e.g. an unbound checkpoint, a rebind attempt, or an unwritable checkpoint directory)."
        )
    )]
    InvalidConfig { reason: String },
}

pub type DjehutyResult<T> = std::result::Result<T, DjehutyError>;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Default board base URL (the single-binary default listener).
const DEFAULT_BOARD_URL: &str = "http://127.0.0.1:8080";

/// Environment variable holding the board bearer token.
const TOKEN_ENV: &str = "DJEHUTY_BOARD_TOKEN";

/// Environment variable holding the board base URL.
const URL_ENV: &str = "DJEHUTY_BOARD_URL";

/// Runtime configuration for the Djehuty Board client.
#[derive(Debug, Clone)]
pub struct DjehutyConfig {
    /// Bearer token issued by `agent create` on the board.
    pub token: String,
    /// Board base URL without a trailing slash.
    pub base_url: String,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Page size for thread polling (the board bounds it to 200).
    pub poll_limit: u64,
}

impl DjehutyConfig {
    /// Read config from environment variables.
    ///
    /// Returns `None` when `DJEHUTY_BOARD_TOKEN` is unset or empty, so the
    /// caller can silently skip the subscription rather than fail.
    pub fn from_env() -> Option<Self> {
        let token = std::env::var(TOKEN_ENV).ok()?;
        if token.is_empty() {
            return None;
        }
        let base_url = std::env::var(URL_ENV).unwrap_or_else(|_| DEFAULT_BOARD_URL.into());
        Some(Self::new(token, base_url))
    }

    /// Build an explicit configuration.
    pub fn new(token: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            token: token.into(),
            base_url: base_url.into(),
            timeout: Duration::from_secs(30),
            poll_limit: 200,
        }
    }
}

// ---------------------------------------------------------------------------
// Typed models (mirror of the board's domain wire types)
// ---------------------------------------------------------------------------

macro_rules! board_id {
    ($name:ident, $docs:literal) => {
        #[doc = $docs]
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub uuid::Uuid);

        impl $name {
            /// Generate a new time-ordered identifier.
            #[must_use]
            pub fn generate() -> Self {
                Self(uuid::Uuid::now_v7())
            }
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.0.fmt(f)
            }
        }
    };
}

board_id!(
    DjehutyThreadId,
    "Stable identity of a public thread on the Djehuty Board."
);
board_id!(
    DjehutyMessageId,
    "Stable identity of a message envelope on the Djehuty Board."
);
board_id!(
    DjehutyAgentId,
    "Stable identity of an authenticated agent on the board."
);

/// Retention requested for a new message (board `RetentionPolicy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DjehutyRetention {
    /// Preserved until an explicit future governance policy removes it.
    Durable,
    /// Automatically removed after the supplied bounded lifetime.
    Temporary {
        /// Lifetime from acceptance time.
        ttl_seconds: u64,
    },
}

/// The stable message envelope shared by inbox and thread writes
/// (board `MessageContent`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DjehutyMessageContent {
    /// Registered protocol governing the payload, `namespace/name@vN`.
    pub message_type: String,
    /// Payload validated against the registered protocol schema.
    pub payload: serde_json::Value,
    /// Compact indexable description, separate from protocol data.
    pub summary: String,
    /// Durable or time-bounded retention.
    pub retention: DjehutyRetention,
    /// Optional causal predecessor.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reply_to: Option<DjehutyMessageId>,
    /// Retry-safety key unique to the author.
    pub idempotency_key: String,
}

/// A stored message as returned by the board (board `MessageView`; unknown
/// fields are tolerated on read, and inbox messages omit `thread_id`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DjehutyMessageView {
    /// Stable message identity.
    pub id: DjehutyMessageId,
    /// Authenticated author; callers cannot choose this value.
    pub author_id: DjehutyAgentId,
    /// Thread identity for public posts; absent for inbox messages.
    #[serde(default)]
    pub thread_id: Option<DjehutyThreadId>,
    /// Causal predecessor, when supplied.
    #[serde(default)]
    pub reply_to: Option<DjehutyMessageId>,
    /// Registered message protocol.
    pub message_type: String,
    /// Schema-validated payload.
    pub payload: serde_json::Value,
    /// Indexable summary.
    pub summary: String,
    /// RFC 3339 creation instant.
    pub created_at: String,
    /// Expiry instant for temporary content, when present.
    #[serde(default)]
    pub expires_at: Option<String>,
}

/// Public thread metadata (board `ThreadView`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DjehutyThreadView {
    /// Stable thread identity.
    pub id: DjehutyThreadId,
    /// Display title.
    pub title: String,
    /// Creating agent.
    pub created_by: DjehutyAgentId,
    /// RFC 3339 creation instant.
    pub created_at: String,
    /// Closing instant, when replies are no longer accepted.
    #[serde(default)]
    pub closed_at: Option<String>,
}

/// One deterministic forward page of thread messages.
#[derive(Clone, Debug, PartialEq)]
pub struct DjehutyPage {
    /// Messages ordered by `(created_at, id)` ascending.
    pub messages: Vec<DjehutyMessageView>,
    /// Cursor for the next forward page, or `None` at the current end.
    pub next_cursor: Option<DjehutyMessageId>,
}

// ---------------------------------------------------------------------------
// Subscription checkpoint
// ---------------------------------------------------------------------------

/// Filename of the crash-safe subscription checkpoint sidecar.
const CHECKPOINT_FILE: &str = "djehuty-subscription.json";

/// Durable polling state for one thread subscription.
///
/// Persisted through the engine's durable-metadata pattern (JSON sidecar +
/// tmp-rename) so the forward cursor survives restarts and never rewinds.
/// The stored `last_message_id` is the *exclusive* forward boundary
/// (`after_message_id`): only messages strictly after it are requested, so
/// the cursor is also the identity of the last consumed message — a crash
/// redelivers a page in full rather than partially skipping it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DjehutyCheckpoint {
    /// Identifier of the subscribed thread.
    pub thread_id: Option<DjehutyThreadId>,
    /// Exclusive forward boundary for the next poll
    /// (`after_message_id`); `None` starts from the thread head.
    pub last_message_id: Option<DjehutyMessageId>,
    /// Total messages handed to the consumer, for observability.
    pub messages_consumed: u64,
}

impl DjehutyCheckpoint {
    /// Load the checkpoint if the sidecar exists and parses; otherwise fall
    /// back to the default without failing (missing/corrupt = fresh start).
    ///
    /// Callers that want the default directory root use
    /// [`crate::paths::AkhPaths::resolve().data_dir`]-derived paths; the
    /// checkpoint lives beside the other durable engine metadata.
    pub fn load(dir: &Path) -> Self {
        let path = dir.join(CHECKPOINT_FILE);
        std::fs::read_to_string(path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    /// Persist the checkpoint crash-safely: write a sibling `.tmp` file,
    /// then atomically rename it over the checkpoint path.
    pub fn save(&self, dir: &Path) -> DjehutyResult<()> {
        std::fs::create_dir_all(dir).map_err(|e| DjehutyError::InvalidConfig {
            reason: format!("cannot create checkpoint dir {}: {e}", dir.display()),
        })?;
        let path = dir.join(CHECKPOINT_FILE);
        let tmp = path.with_extension("json.tmp");
        let json = serde_json::to_string_pretty(self).map_err(|e| DjehutyError::Parse {
            message: format!("checkpoint serialize: {e}"),
        })?;
        std::fs::write(&tmp, json).map_err(|e| DjehutyError::InvalidConfig {
            reason: format!("cannot write checkpoint {}: {e}", tmp.display()),
        })?;
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            DjehutyError::InvalidConfig {
                reason: format!(
                    "cannot rename checkpoint into place {}: {e}",
                    path.display()
                ),
            }
        })
    }

    /// Advance the watermark to a page end. The caller must have durably
    /// handed the page to its consumer first.
    pub fn advance(&mut self, thread_id: DjehutyThreadId, page: &DjehutyPage) {
        self.thread_id = Some(thread_id);
        // The next forward poll must begin strictly after everything that
        // was just delivered: prefer the board's next-cursor, which for a
        // full page is the trailing message id; fall back to the last
        // message of the page when the board signals end-of-thread.
        if let Some(cursor) = page.next_cursor {
            self.last_message_id = Some(cursor);
        } else if let Some(last) = page.messages.last() {
            self.last_message_id = Some(last.id);
        }
        self.messages_consumed += page.messages.len() as u64;
    }
}

// ---------------------------------------------------------------------------
// Retry policy
// ---------------------------------------------------------------------------

/// Bounded exponential backoff for polling and posting.
#[derive(Clone, Debug)]
pub struct DjehutyRetryPolicy {
    /// Total attempts per operation (the first try included).
    pub max_attempts: u32,
    /// Delay before the first retry.
    pub initial_delay: Duration,
    /// Ceiling for a single delay.
    pub max_delay: Duration,
}

impl Default for DjehutyRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 4,
            initial_delay: Duration::from_millis(250),
            max_delay: Duration::from_secs(8),
        }
    }
}

impl DjehutyRetryPolicy {
    /// Backoff before the retry following failed attempt `attempt`
    /// (0-based). Exponential growth saturating at `max_delay`.
    fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let shift = attempt.min(16);
        let base = self.initial_delay.saturating_mul(1u32 << shift);
        base.min(self.max_delay)
    }

    /// Run `op`, retrying only transient failures: transport errors, 429,
    /// and 5xx. 4xx responses are final (the request itself is wrong), and
    /// a successful response is returned immediately. The caller-supplied
    /// `idempotency_key` makes replaying a timed-out write safe.
    fn with_retry<T>(&self, op: &mut dyn FnMut() -> DjehutyResult<T>) -> DjehutyResult<T> {
        let mut last: Option<DjehutyError> = None;
        for attempt in 0..self.max_attempts {
            match op() {
                Ok(value) => return Ok(value),
                Err(err) if is_retryable(&err) => {
                    let delay = self.delay_for_attempt(attempt);
                    tracing::warn!(
                        attempt,
                        max_attempts = self.max_attempts,
                        delay_ms = delay.as_millis() as u64,
                        error = %err,
                        "djehuty: retrying transient board failure"
                    );
                    std::thread::sleep(delay);
                    last = Some(err);
                }
                Err(err) => return Err(err),
            }
        }
        Err(last.unwrap_or_else(|| DjehutyError::Transport {
            message: "retry loop produced no error".into(),
        }))
    }
}

/// Transient failures worth retrying with backoff.
fn is_retryable(err: &DjehutyError) -> bool {
    match err {
        DjehutyError::Transport { .. } => true,
        DjehutyError::ApiError { status, .. } => *status == 429 || *status >= 500,
        DjehutyError::BoardError { status, .. } => *status == 429 || *status >= 500,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/// Synchronous HTTP client for the Djehuty Board typed API.
pub struct DjehutyClient {
    config: DjehutyConfig,
    retry: DjehutyRetryPolicy,
    agent: ureq::Agent,
}

impl DjehutyClient {
    /// Build a client for `config` with the default retry policy.
    pub fn new(config: DjehutyConfig) -> Self {
        Self::with_retry_policy(config, DjehutyRetryPolicy::default())
    }

    /// Build a client with an explicit retry policy (tests use attempts=1).
    pub fn with_retry_policy(config: DjehutyConfig, retry: DjehutyRetryPolicy) -> Self {
        let agent = ureq::AgentBuilder::new().timeout(config.timeout).build();
        let mut config = config;
        config.base_url = config.base_url.trim_end_matches('/').to_owned();
        Self {
            config,
            retry,
            agent,
        }
    }

    fn auth(&self) -> String {
        format!("Bearer {}", self.config.token)
    }

    /// List the public threads visible to this identity.
    pub fn list_threads(&self) -> DjehutyResult<Vec<DjehutyThreadView>> {
        let url = format!("{}/v1/threads", self.config.base_url);
        self.retry.with_retry(&mut || {
            let response = self
                .agent
                .get(&url)
                .set("Authorization", &self.auth())
                .set("Accept", "application/json")
                .call()
                .map_err(map_ureq)?;
            response
                .into_json::<Vec<DjehutyThreadView>>()
                .map_err(|e| DjehutyError::Parse {
                    message: e.to_string(),
                })
        })
    }

    /// Post one typed message to a public thread.
    ///
    /// `content.idempotency_key` makes the write retry-safe: the board
    /// de-duplicates on it, so a timeout followed by a replay returns the
    /// originally stored message instead of posting twice.
    pub fn post_thread_message(
        &self,
        thread_id: DjehutyThreadId,
        content: DjehutyMessageContent,
    ) -> DjehutyResult<DjehutyMessageView> {
        let url = format!("{}/v1/threads/{}/messages", self.config.base_url, thread_id);
        let body = serde_json::to_string(&content).map_err(|e| DjehutyError::Parse {
            message: format!("envelope serialize: {e}"),
        })?;
        self.retry.with_retry(&mut || {
            let response = self
                .agent
                .post(&url)
                .set("Authorization", &self.auth())
                .set("Content-Type", "application/json")
                .set("Accept", "application/json")
                .send_string(&body)
                .map_err(map_ureq)?;
            response
                .into_json::<DjehutyMessageView>()
                .map_err(|e| DjehutyError::Parse {
                    message: e.to_string(),
                })
        })
    }

    /// Post one typed message into other agents' private inboxes.
    ///
    /// The wire shape is the board's `PostInboxMessage`: the recipients
    /// array alongside the flattened `MessageContent` envelope.
    pub fn post_inbox_message(
        &self,
        recipients: Vec<DjehutyAgentId>,
        content: DjehutyMessageContent,
    ) -> DjehutyResult<DjehutyMessageView> {
        let url = format!("{}/v1/inbox/messages", self.config.base_url);
        let body = serde_json::json!({
            "recipients": recipients,
            "message_type": content.message_type,
            "payload": content.payload,
            "summary": content.summary,
            "retention": content.retention,
            "reply_to": content.reply_to,
            "idempotency_key": content.idempotency_key,
        })
        .to_string();
        self.retry.with_retry(&mut || {
            let response = self
                .agent
                .post(&url)
                .set("Authorization", &self.auth())
                .set("Content-Type", "application/json")
                .set("Accept", "application/json")
                .send_string(&body)
                .map_err(map_ureq)?;
            response
                .into_json::<DjehutyMessageView>()
                .map_err(|e| DjehutyError::Parse {
                    message: e.to_string(),
                })
        })
    }

    /// Read the next deterministic forward page of a thread.
    ///
    /// With `after = None` the page starts at the thread head; with a cursor
    /// the board returns messages strictly after it in `(created_at, id)`
    /// order. `page.next_cursor` feeds the next call and is what the
    /// subscription checkpoint persists.
    pub fn read_page(
        &self,
        thread_id: DjehutyThreadId,
        after: Option<DjehutyMessageId>,
        limit: u64,
    ) -> DjehutyResult<DjehutyPage> {
        let mut url = format!(
            "{}/v1/threads/{}/messages?limit={}",
            self.config.base_url, thread_id, limit
        );
        if let Some(cursor) = after {
            url.push_str(&format!("&after_message_id={cursor}"));
        }
        self.retry.with_retry(&mut || {
            let response = self
                .agent
                .get(&url)
                .set("Authorization", &self.auth())
                .set("Accept", "application/json")
                .call()
                .map_err(map_ureq)?;
            let next_cursor = response
                .header("next-cursor")
                .and_then(|raw| uuid::Uuid::parse_str(raw).ok())
                .map(DjehutyMessageId);
            let messages = response
                .into_json::<Vec<DjehutyMessageView>>()
                .map_err(|e| DjehutyError::Parse {
                    message: e.to_string(),
                })?;
            Ok(DjehutyPage {
                messages,
                next_cursor,
            })
        })
    }

    /// Deterministically drain a thread forward, honoring the persisted
    /// checkpoint, and persist the new checkpoint after each page.
    ///
    /// The watermark only advances once a page has been handed to `consumer`
    /// and the checkpoint re-saved — the same poison-safe ordering the daily
    /// Librarian pass uses — so a crash replays at most one page and the
    /// cursor never skips messages. The consumer must return `Ok(())` only
    /// after the messages are safe; returning `Err` stops the drain before
    /// the checkpoint advances (the poisoned page is retried next poll).
    /// Returns the number of messages delivered this call.
    pub fn poll_thread(
        &self,
        checkpoint_dir: &Path,
        consumer: &mut dyn FnMut(&[DjehutyMessageView]) -> DjehutyResult<()>,
    ) -> DjehutyResult<u64> {
        let mut checkpoint = DjehutyCheckpoint::load(checkpoint_dir);
        let Some(thread_id) = checkpoint.thread_id else {
            return Err(DjehutyError::InvalidConfig {
                reason: "checkpoint has no thread_id; call subscribe_thread first".into(),
            });
        };
        let mut delivered = 0u64;
        loop {
            let page = self.read_page(
                thread_id,
                checkpoint.last_message_id,
                self.config.poll_limit,
            )?;
            if page.messages.is_empty() {
                return Ok(delivered);
            }
            consumer(&page.messages)?;
            delivered += page.messages.len() as u64;
            checkpoint.advance(thread_id, &page);
            checkpoint.save(checkpoint_dir)?;
            if page.next_cursor.is_none() {
                return Ok(delivered);
            }
        }
    }

    /// Point the checkpoint at a thread without changing its cursor.
    ///
    /// Existing cursor state is preserved (re-subscribing must not rewind a
    /// durable watermark); the thread binding is durable immediately so a
    /// crash cannot lose it.
    pub fn subscribe_thread(
        &self,
        checkpoint_dir: &Path,
        thread_id: DjehutyThreadId,
    ) -> DjehutyResult<()> {
        let mut checkpoint = DjehutyCheckpoint::load(checkpoint_dir);
        if checkpoint
            .thread_id
            .is_some_and(|existing| existing != thread_id)
        {
            return Err(DjehutyError::InvalidConfig {
                reason: format!(
                    "checkpoint already tracks thread {existing}; refusing to rebind to {thread_id}",
                    existing = checkpoint.thread_id.unwrap()
                ),
            });
        }
        checkpoint.thread_id = Some(thread_id);
        checkpoint.save(checkpoint_dir)
    }
}

fn decode_board_error(status: u16, body: &str) -> DjehutyError {
    match serde_json::from_str::<BoardErrorEnvelope>(body) {
        Ok(envelope) => DjehutyError::BoardError {
            status,
            code: envelope.error.code,
            message: envelope.error.message,
        },
        Err(_) => DjehutyError::ApiError {
            status,
            message: body.chars().take(500).collect(),
        },
    }
}

/// Map a transport/status failure into the typed error.
fn map_ureq(err: ureq::Error) -> DjehutyError {
    match err {
        ureq::Error::Status(status, response) => {
            let body = response.into_string().unwrap_or_default();
            decode_board_error(status, &body)
        }
        ureq::Error::Transport(t) => DjehutyError::Transport {
            message: t.to_string(),
        },
    }
}

/// Outer board error envelope (`{"error": {"code": ..., "message": ...}}`).
#[derive(Debug, Deserialize)]
struct BoardErrorEnvelope {
    error: BoardErrorBody,
}

/// Symbolic code and safe detail of a board error.
#[derive(Debug, Deserialize)]
struct BoardErrorBody {
    code: String,
    message: String,
}

/// Built-in board protocol used by the convenience constructor.
pub const BOARD_NOTE_V1: &str = "board/note@v1";

/// Convenience envelope for a durable `board/note@v1` post with an explicit
/// idempotency key.
///
/// The key is derived by the caller from a (logical write name, day, seq)
/// tuple so the same logical write retries to a single stored message.
pub fn board_note(
    text: impl Into<String>,
    summary: impl Into<String>,
    idempotency_key: impl Into<String>,
) -> DjehutyMessageContent {
    DjehutyMessageContent {
        message_type: BOARD_NOTE_V1.to_owned(),
        payload: serde_json::json!({ "text": text.into() }),
        summary: summary.into(),
        retention: DjehutyRetention::Durable,
        reply_to: None,
        idempotency_key: idempotency_key.into(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    // -- fixtures -----------------------------------------------------------

    /// Deterministic UUID for fixture use (seeded, unique per seed).
    fn fixed_uuid(seed: usize) -> uuid::Uuid {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&(seed as u64).to_be_bytes());
        bytes[8..].copy_from_slice(&(seed as u64 ^ 0xdead_beef).to_be_bytes());
        uuid::Uuid::from_bytes(bytes)
    }

    fn message(id_seed: usize, created_at: &str, text: &str) -> DjehutyMessageView {
        DjehutyMessageView {
            id: DjehutyMessageId(fixed_uuid(id_seed)),
            author_id: DjehutyAgentId(fixed_uuid(999)),
            thread_id: Some(DjehutyThreadId(fixed_uuid(1))),
            reply_to: None,
            message_type: "board/note@v1".into(),
            payload: serde_json::json!({ "text": text }),
            summary: text.to_owned(),
            created_at: created_at.to_owned(),
            expires_at: None,
        }
    }

    fn note_json(id_seed: usize, text: &str, created_at: &str) -> String {
        let view = message(id_seed, created_at, text);
        serde_json::to_string(&view).unwrap()
    }

    fn captured_log() -> Arc<Mutex<Vec<String>>> {
        Arc::new(Mutex::new(Vec::new()))
    }

    /// One-shot HTTP mock: read one request (recording it), reply with a
    /// canned status/headers/body. The recorder is shared with the test.
    fn serve(
        status_line: &'static str,
        body: String,
        extra_header: Option<(&'static str, &'static str)>,
        log: Arc<Mutex<Vec<String>>>,
    ) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
            let mut request = Vec::new();
            let mut buf = [0u8; 4096];
            loop {
                let n = stream.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break;
                }
                request.extend_from_slice(&buf[..n]);
                let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n")
                else {
                    continue;
                };
                let header_end = header_end + 4;
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers.lines().find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                });
                if content_length.is_none_or(|length| request.len() >= header_end + length) {
                    break;
                }
            }
            if let Ok(mut l) = log.lock() {
                l.push(String::from_utf8_lossy(&request).to_string());
            }
            let mut response = format!(
                "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
                body.len()
            );
            if let Some((name, value)) = extra_header {
                response.push_str(&format!("{name}: {value}\r\n"));
            }
            response.push_str("\r\n");
            response.push_str(&body);
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        });
        (format!("http://127.0.0.1:{port}"), handle)
    }

    fn no_retry() -> DjehutyRetryPolicy {
        DjehutyRetryPolicy {
            max_attempts: 1,
            ..DjehutyRetryPolicy::default()
        }
    }

    fn client_at(url: String) -> DjehutyClient {
        DjehutyClient::with_retry_policy(DjehutyConfig::new("tok", url), no_retry())
    }

    fn request_body(request: &str) -> serde_json::Value {
        let payload_start = request.find("\r\n\r\n").expect("headers/body split") + 4;
        serde_json::from_str(request[payload_start..].trim()).expect("request body is JSON")
    }

    // -- model tests --------------------------------------------------------

    #[test]
    fn thread_view_round_trips_board_json() {
        let raw = format!(
            r#"{{"id":"{}","title":"Impl board client","created_by":"{}","created_at":"2026-08-29T10:00:00Z","closed_at":null}}"#,
            fixed_uuid(1),
            fixed_uuid(2)
        );
        let view: DjehutyThreadView = serde_json::from_str(&raw).unwrap();
        assert_eq!(view.id, DjehutyThreadId(fixed_uuid(1)));
        assert_eq!(view.title, "Impl board client");
        assert!(view.closed_at.is_none());
        let round: DjehutyThreadView =
            serde_json::from_str(&serde_json::to_string(&view).unwrap()).unwrap();
        assert_eq!(round, view);
    }

    #[test]
    fn message_view_tolerates_unknown_fields_and_missing_optionals() {
        // Board may add fields; optional thread_id/reply_to can be absent
        // (inbox messages have no thread).
        let raw = format!(
            r#"{{"id":"{}","author_id":"{}","message_type":"board/note@v1","payload":{{"text":"hi"}},"summary":"hi","created_at":"2026-08-29T10:00:00Z","x-extra":true}}"#,
            fixed_uuid(3),
            fixed_uuid(4)
        );
        let view: DjehutyMessageView = serde_json::from_str(&raw).unwrap();
        assert_eq!(view.id, DjehutyMessageId(fixed_uuid(3)));
        assert!(view.thread_id.is_none());
        assert!(view.reply_to.is_none());
        assert_eq!(view.payload["text"], "hi");
    }

    #[test]
    fn retention_serializes_tagged_kind() {
        let durable = serde_json::to_value(DjehutyRetention::Durable).unwrap();
        assert_eq!(durable, serde_json::json!({"kind": "durable"}));
        let temp = serde_json::to_value(DjehutyRetention::Temporary { ttl_seconds: 60 }).unwrap();
        assert_eq!(
            temp,
            serde_json::json!({"kind": "temporary", "ttl_seconds": 60})
        );
    }

    #[test]
    fn board_note_uses_note_protocol_and_durable_retention() {
        let content = board_note("hello", "hello", "2026-08-29:hello:0001");
        assert_eq!(content.message_type, "board/note@v1");
        assert_eq!(content.retention, DjehutyRetention::Durable);
        assert_eq!(content.idempotency_key, "2026-08-29:hello:0001");
        let json = serde_json::to_value(&content).unwrap();
        assert_eq!(json["message_type"], "board/note@v1");
        assert_eq!(json["retention"]["kind"], "durable");
        assert!(json["idempotency_key"].is_string());
    }

    // -- checkpoint tests ---------------------------------------------------

    #[test]
    fn checkpoint_round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let mut checkpoint = DjehutyCheckpoint {
            thread_id: Some(DjehutyThreadId(fixed_uuid(10))),
            last_message_id: Some(DjehutyMessageId(fixed_uuid(11))),
            messages_consumed: 7,
        };
        checkpoint.save(dir.path()).unwrap();
        assert_eq!(DjehutyCheckpoint::load(dir.path()), checkpoint);
        // Advance replaces the boundary with the page cursor.
        let page = DjehutyPage {
            messages: vec![
                message(12, "2026-08-29T10:00:00Z", "a"),
                message(13, "2026-08-29T10:00:01Z", "b"),
            ],
            next_cursor: Some(DjehutyMessageId(fixed_uuid(13))),
        };
        checkpoint.advance(DjehutyThreadId(fixed_uuid(10)), &page);
        checkpoint.save(dir.path()).unwrap();
        let loaded = DjehutyCheckpoint::load(dir.path());
        assert_eq!(
            loaded.last_message_id,
            Some(DjehutyMessageId(fixed_uuid(13)))
        );
        assert_eq!(loaded.messages_consumed, 9);
    }

    #[test]
    fn checkpoint_load_is_tolerant_of_missing_and_corrupt_files() {
        let dir = tempfile::tempdir().unwrap();
        // Missing sidecar: fresh default.
        assert_eq!(
            DjehutyCheckpoint::load(dir.path()),
            DjehutyCheckpoint::default()
        );
        // Corrupt sidecar: tolerated, default returned.
        std::fs::write(dir.path().join(CHECKPOINT_FILE), "{not json").unwrap();
        assert_eq!(
            DjehutyCheckpoint::load(dir.path()),
            DjehutyCheckpoint::default()
        );
    }

    #[test]
    fn checkpoint_save_leaves_no_temporary_residue() {
        let dir = tempfile::tempdir().unwrap();
        let mut checkpoint = DjehutyCheckpoint {
            thread_id: Some(DjehutyThreadId(fixed_uuid(20))),
            ..DjehutyCheckpoint::default()
        };
        checkpoint.save(dir.path()).unwrap();
        checkpoint.messages_consumed += 1;
        checkpoint.save(dir.path()).unwrap();
        let residue = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(residue, 0, "no tmp residue after successful saves");
        assert_eq!(DjehutyCheckpoint::load(dir.path()).messages_consumed, 1);
    }

    #[test]
    fn checkpoint_advance_uses_last_message_when_no_cursor() {
        let mut checkpoint = DjehutyCheckpoint::default();
        let page = DjehutyPage {
            messages: vec![message(30, "2026-08-29T10:00:00Z", "a")],
            next_cursor: None,
        };
        checkpoint.advance(DjehutyThreadId(fixed_uuid(10)), &page);
        assert_eq!(
            checkpoint.last_message_id,
            Some(DjehutyMessageId(fixed_uuid(30)))
        );
    }

    // -- retry tests --------------------------------------------------------

    #[test]
    fn retry_policy_backs_off_bounded() {
        let policy = DjehutyRetryPolicy {
            max_attempts: 6,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(400),
        };
        assert_eq!(policy.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(policy.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(400));
        // Saturates at max_delay, never exceeds it.
        assert_eq!(policy.delay_for_attempt(10), Duration::from_millis(400));
        assert_eq!(policy.delay_for_attempt(30), Duration::from_millis(400));
    }

    #[test]
    fn retry_only_transient_failures() {
        assert!(is_retryable(&DjehutyError::Transport {
            message: "reset".into()
        }));
        assert!(is_retryable(&DjehutyError::ApiError {
            status: 503,
            message: "down".into()
        }));
        assert!(is_retryable(&DjehutyError::ApiError {
            status: 429,
            message: "slow".into()
        }));
        assert!(!is_retryable(&DjehutyError::ApiError {
            status: 422,
            message: "bad payload".into()
        }));
        assert!(!is_retryable(&DjehutyError::NoToken));
        assert!(!is_retryable(&DjehutyError::InvalidConfig {
            reason: "nope".into()
        }));
    }

    // -- HTTP client tests (local one-shot board mock) -----------------------

    #[test]
    fn list_threads_sends_bearer_and_parses_array() {
        let log = captured_log();
        let body = format!(
            r#"[{{"id":"{}","title":"Impl board client","created_by":"{}","created_at":"2026-08-29T10:00:00Z"}}]"#,
            fixed_uuid(1),
            fixed_uuid(2)
        );
        let (url, handle) = serve("HTTP/1.1 200 OK", body, None, std::sync::Arc::clone(&log));
        let threads = client_at(url).list_threads().expect("list ok");
        let _ = handle.join();
        assert_eq!(threads.len(), 1);
        assert_eq!(threads[0].id, DjehutyThreadId(fixed_uuid(1)));
        let requests = log.lock().unwrap();
        assert!(
            requests[0].starts_with("GET /v1/threads"),
            "got: {}",
            requests[0]
        );
        assert!(requests[0].contains("Authorization: Bearer tok"));
        assert!(requests[0].contains("Accept: application/json"));
    }

    #[test]
    fn read_page_follows_next_cursor_header() {
        let log = captured_log();
        let body = note_json(70, "a", "2026-08-29T10:00:00Z");
        let body = format!("[{body}]");
        let next = "11111111-1111-1111-1111-111111111111";
        let (url, handle) = serve(
            "HTTP/1.1 200 OK",
            body,
            Some(("next-cursor", next)),
            std::sync::Arc::clone(&log),
        );
        let thread = DjehutyThreadId(fixed_uuid(1));
        let page = client_at(url).read_page(thread, None, 50).expect("page ok");
        let _ = handle.join();
        assert_eq!(page.messages.len(), 1);
        assert_eq!(
            page.next_cursor,
            Some(DjehutyMessageId(next.parse().unwrap()))
        );
        let requests = log.lock().unwrap();
        assert!(
            requests[0].starts_with(&format!(
                "GET /v1/threads/{}/messages?limit=50",
                fixed_uuid(1)
            )),
            "got: {}",
            requests[0]
        );
    }

    #[test]
    fn read_page_sends_exclusive_after_boundary() {
        let log = captured_log();
        let (url, handle) = serve(
            "HTTP/1.1 200 OK",
            "[]".to_owned(),
            None,
            std::sync::Arc::clone(&log),
        );
        let thread = DjehutyThreadId(fixed_uuid(1));
        let after = DjehutyMessageId(fixed_uuid(77));
        let page = client_at(url)
            .read_page(thread, Some(after), 200)
            .expect("page ok");
        let _ = handle.join();
        assert!(page.messages.is_empty());
        assert!(page.next_cursor.is_none(), "absent header means no cursor");
        let requests = log.lock().unwrap();
        assert!(
            requests[0].contains(&format!("after_message_id={}", fixed_uuid(77))),
            "got: {}",
            requests[0]
        );
    }

    #[test]
    fn post_thread_message_sends_envelope_and_parses_view() {
        let log = captured_log();
        let response =
            serde_json::to_string(&message(40, "2026-08-29T10:00:00Z", "posted")).unwrap();
        let (url, handle) = serve(
            "HTTP/1.1 201 Created",
            response,
            None,
            std::sync::Arc::clone(&log),
        );
        let thread = DjehutyThreadId(fixed_uuid(1));
        let view = client_at(url)
            .post_thread_message(
                thread,
                board_note("from test", "posted", "2026-08-29:test:0001"),
            )
            .expect("post ok");
        let _ = handle.join();
        assert_eq!(view.id, DjehutyMessageId(fixed_uuid(40)));
        let requests = log.lock().unwrap();
        assert!(requests[0].starts_with(&format!("POST /v1/threads/{}/messages", fixed_uuid(1))));
        assert!(requests[0].contains("Content-Type: application/json"));
        assert!(requests[0].contains("Authorization: Bearer tok"));
        let sent = request_body(&requests[0]);
        assert_eq!(sent["message_type"], "board/note@v1");
        assert_eq!(sent["retention"]["kind"], "durable");
        assert_eq!(sent["idempotency_key"], "2026-08-29:test:0001");
        assert_eq!(sent["payload"]["text"], "from test");
    }

    #[test]
    fn post_inbox_message_merges_recipients_with_envelope() {
        let log = captured_log();
        let mut dm = message(50, "2026-08-29T10:00:00Z", "dm");
        dm.thread_id = None;
        let response = serde_json::to_string(&dm).unwrap();
        let (url, handle) = serve(
            "HTTP/1.1 201 Created",
            response,
            None,
            std::sync::Arc::clone(&log),
        );
        let recipient = DjehutyAgentId(fixed_uuid(60));
        let view = client_at(url)
            .post_inbox_message(
                vec![recipient],
                board_note("dm", "dm", "2026-08-29:dm:0001"),
            )
            .expect("dm ok");
        let _ = handle.join();
        assert_eq!(view.author_id, DjehutyAgentId(fixed_uuid(999)));
        let requests = log.lock().unwrap();
        assert!(requests[0].starts_with("POST /v1/inbox/messages"));
        let sent = request_body(&requests[0]);
        assert_eq!(sent["recipients"], serde_json::json!([recipient]));
        assert_eq!(sent["message_type"], "board/note@v1");
        assert_eq!(sent["idempotency_key"], "2026-08-29:dm:0001");
    }

    #[test]
    fn api_error_envelope_decodes_to_board_error_and_is_final() {
        let log = captured_log();
        let (url, handle) = serve(
            "HTTP/1.1 422 Unprocessable Entity",
            r#"{"error":{"code":"invalid_input","message":"after_message_id and before_message_id are mutually exclusive"}}"#.to_owned(),
            None,
            std::sync::Arc::clone(&log),
        );
        let err = client_at(url)
            .read_page(DjehutyThreadId(fixed_uuid(1)), None, 50)
            .expect_err("422 is final");
        let _ = handle.join();
        match err {
            DjehutyError::BoardError {
                status,
                code,
                message,
            } => {
                assert_eq!(status, 422);
                assert_eq!(code, "invalid_input");
                assert!(message.contains("mutually exclusive"));
            }
            other => panic!("expected BoardError, got {other:?}"),
        }
    }

    #[test]
    fn unauthorized_maps_to_api_error() {
        let log = captured_log();
        let (url, handle) = serve(
            "HTTP/1.1 401 Unauthorized",
            r#"{"error":{"code":"unauthenticated","message":"a valid agent bearer token is required"}}"#.to_owned(),
            None,
            std::sync::Arc::clone(&log),
        );
        let err = client_at(url).list_threads().expect_err("401 is final");
        let _ = handle.join();
        match err {
            DjehutyError::BoardError { code, .. } => assert_eq!(code, "unauthenticated"),
            other => panic!("expected BoardError, got {other:?}"),
        }
    }

    #[test]
    fn transient_5xx_retries_then_succeeds() {
        // Serve two 503s then one empty 200, logging each request.
        let log = captured_log();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        serve_5xx_then_ok(listener, std::sync::Arc::clone(&log));
        let url = format!("http://127.0.0.1:{port}");
        let client = DjehutyClient::with_retry_policy(
            DjehutyConfig::new("tok", url),
            DjehutyRetryPolicy {
                max_attempts: 3,
                initial_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(2),
            },
        );
        let page = client
            .read_page(DjehutyThreadId(fixed_uuid(1)), None, 50)
            .expect("eventual success after 503s");
        assert!(page.messages.is_empty());
        assert_eq!(log.lock().unwrap().len(), 3, "three attempts made");
    }

    /// Serve two 503s then one empty 200 on `listener`, logging requests.
    fn serve_5xx_then_ok(listener: TcpListener, log: Arc<Mutex<Vec<String>>>) {
        std::thread::spawn(move || {
            let script: [(&str, &str); 3] = [
                (
                    "HTTP/1.1 503 Service Unavailable",
                    r#"{"error":{"code":"internal","message":"try again"}}"#,
                ),
                (
                    "HTTP/1.1 503 Service Unavailable",
                    r#"{"error":{"code":"internal","message":"try again"}}"#,
                ),
                ("HTTP/1.1 200 OK", "[]"),
            ];
            for (status, body) in script {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut buf = [0u8; 8192];
                let _ = stream.read(&mut buf);
                if let Ok(mut l) = log.lock() {
                    l.push("served".into());
                }
                let response = format!(
                    "{status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });
    }

    // -- poll/subscribe tests (checkpoint-driven) ----------------------------

    #[test]
    fn subscribe_then_poll_delivers_in_order_and_persists_cursor() {
        // Two pages: first returns 2 messages + next-cursor, second returns
        // 1 message with no header. The consumer records delivery order.
        let log = captured_log();
        let server_log = std::sync::Arc::clone(&log);
        let page1 = format!(
            "[{},{}]",
            note_json(70, "one", "2026-08-29T10:00:00Z"),
            note_json(71, "two", "2026-08-29T10:00:01Z")
        );
        let page2 = format!("[{}]", note_json(72, "three", "2026-08-29T10:00:02Z"));
        let cursor = "33333333-3333-3333-3333-333333333333";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let handle = std::thread::spawn(move || {
            for (body, next) in [(&page1, Some(cursor)), (&page2, None)] {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut buf = [0u8; 16384];
                let n = stream.read(&mut buf).unwrap_or(0);
                if let Ok(mut l) = server_log.lock() {
                    l.push(String::from_utf8_lossy(&buf[..n]).to_string());
                }
                let mut response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
                    body.len()
                );
                if let Some(c) = next {
                    response.push_str(&format!("next-cursor: {c}\r\n"));
                }
                response.push_str("Connection: close\r\n\r\n");
                response.push_str(body);
                let _ = stream.write_all(response.as_bytes());
                let _ = stream.flush();
            }
        });

        let client = client_at(format!("http://127.0.0.1:{port}"));
        let dir = tempfile::tempdir().unwrap();
        let thread = DjehutyThreadId(fixed_uuid(1));
        client.subscribe_thread(dir.path(), thread).unwrap();

        let seen = Arc::new(Mutex::new(Vec::new()));
        let consumer_seen = Arc::clone(&seen);
        let delivered = client
            .poll_thread(dir.path(), &mut move |messages| {
                consumer_seen
                    .lock()
                    .unwrap()
                    .extend(messages.iter().map(|m| m.summary.clone()));
                Ok(())
            })
            .unwrap();
        handle.join().unwrap();

        assert_eq!(delivered, 3);
        assert_eq!(*seen.lock().unwrap(), vec!["one", "two", "three"]);
        // First request had no cursor; second resumed from page1's cursor.
        let requests = log.lock().unwrap();
        assert!(
            !requests[0].contains("after_message_id"),
            "got: {}",
            requests[0]
        );
        assert!(
            requests[1].contains("after_message_id=33333333-3333-3333-3333-333333333333"),
            "got: {}",
            requests[1]
        );
        // Durable cursor persisted for the next poll: page 2 ended the
        // thread (no next-cursor), so the boundary is its last message id.
        let persisted = DjehutyCheckpoint::load(dir.path());
        assert_eq!(persisted.thread_id, Some(thread));
        assert_eq!(
            persisted.last_message_id,
            Some(DjehutyMessageId(fixed_uuid(72)))
        );
        assert_eq!(persisted.messages_consumed, 3);
    }

    #[test]
    fn poll_resumes_from_checkpoint_without_reprocessing_backlog() {
        // A durable checkpoint mid-thread means poll starts strictly after it.
        let log = captured_log();
        let after_body = format!("[{}]", note_json(80, "later", "2026-08-29T10:05:00Z"));
        let cursor_after_page = "88888888-8888-8888-8888-888888888888";
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let log_for_server = Arc::clone(&log);
        let handle = std::thread::spawn(move || {
            // Request 1: cursor page (one message + next-cursor header).
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 16384];
            let n = stream.read(&mut buf).unwrap_or(0);
            if let Ok(mut l) = log_for_server.lock() {
                l.push(String::from_utf8_lossy(&buf[..n]).to_string());
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nnext-cursor: {cursor_after_page}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{after_body}",
                after_body.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
            // Request 2: poll continues from the new cursor; the thread is
            // exhausted, so answer with an empty page and no header.
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let mut buf = [0u8; 16384];
            let n = stream.read(&mut buf).unwrap_or(0);
            if let Ok(mut l) = log_for_server.lock() {
                l.push(String::from_utf8_lossy(&buf[..n]).to_string());
            }
            let tail = "[]";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{tail}",
                tail.len()
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        });

        let client = client_at(format!("http://127.0.0.1:{port}"));
        let dir = tempfile::tempdir().unwrap();
        let thread = DjehutyThreadId(fixed_uuid(1));
        // Pre-existing durable checkpoint: 5 consumed up to cursor 7777...
        let persisted = DjehutyCheckpoint {
            thread_id: Some(thread),
            last_message_id: Some(DjehutyMessageId(
                "77777777-7777-7777-7777-777777777777".parse().unwrap(),
            )),
            messages_consumed: 5,
        };
        persisted.save(dir.path()).unwrap();

        let delivered = client.poll_thread(dir.path(), &mut |_| Ok(())).unwrap();
        handle.join().unwrap();

        assert_eq!(delivered, 1);
        let requests = log.lock().unwrap();
        assert!(
            requests[0].contains("after_message_id=77777777-7777-7777-7777-777777777777"),
            "poll resumes strictly after the durable watermark: {}",
            requests[0]
        );
        let updated = DjehutyCheckpoint::load(dir.path());
        assert_eq!(
            updated.last_message_id,
            Some(DjehutyMessageId(cursor_after_page.parse().unwrap()))
        );
        assert_eq!(updated.messages_consumed, 6);
    }

    #[test]
    fn poll_refuses_to_rebind_to_another_thread() {
        let client = client_at("http://127.0.0.1:1".to_owned());
        let dir = tempfile::tempdir().unwrap();
        client
            .subscribe_thread(dir.path(), DjehutyThreadId(fixed_uuid(1)))
            .unwrap();
        let err = client
            .subscribe_thread(dir.path(), DjehutyThreadId(fixed_uuid(2)))
            .expect_err("rebind refused");
        assert!(matches!(err, DjehutyError::InvalidConfig { .. }));
    }

    #[test]
    fn poll_without_subscribe_is_a_clear_config_error() {
        let client = client_at("http://127.0.0.1:1".to_owned());
        let dir = tempfile::tempdir().unwrap();
        let err = client
            .poll_thread(dir.path(), &mut |_| Ok(()))
            .expect_err("no thread bound");
        assert!(matches!(err, DjehutyError::InvalidConfig { .. }));
    }

    #[test]
    fn consumer_error_stops_poll_before_checkpoint_advance() {
        // The consumer rejects the page; the checkpoint must not advance.
        let page1 = format!("[{}]", note_json(90, "poison", "2026-08-29T10:00:00Z"));
        let cursor = "90909090-9090-9090-9090-909090909090";
        let (url, handle) = serve(
            "HTTP/1.1 200 OK",
            page1,
            Some(("next-cursor", cursor)),
            captured_log(),
        );
        let client = client_at(url);
        let dir = tempfile::tempdir().unwrap();
        let thread = DjehutyThreadId(fixed_uuid(1));
        client.subscribe_thread(dir.path(), thread).unwrap();
        let before = DjehutyCheckpoint::load(dir.path());

        let err = client
            .poll_thread(dir.path(), &mut |_| {
                Err(DjehutyError::Transport {
                    message: "consumer refused (poison page)".into(),
                })
            })
            .expect_err("consumer error propagates");
        let _ = handle.join();

        assert!(matches!(err, DjehutyError::Transport { .. }));
        let after = DjehutyCheckpoint::load(dir.path());
        assert_eq!(after, before, "checkpoint unchanged on consumer failure");
        assert_eq!(after.messages_consumed, 0);
    }

    #[test]
    fn config_from_env_requires_token() {
        // SAFETY: single-threaded test; no other threads read this var.
        unsafe { std::env::remove_var("DJEHUTY_BOARD_TOKEN") };
        assert!(DjehutyConfig::from_env().is_none());
    }
}
