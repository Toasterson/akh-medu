//! Email connector abstraction: trait + JMAP, IMAP, and mock implementations.
//!
//! `EmailConnector` defines the interface for fetching and sending email.
//! - `JmapConnector` uses ureq (sync HTTP) to speak JMAP (RFC 8620).
//! - `ImapConnector` uses the `imap` crate for sync IMAP access.
//! - `MockConnector` provides an in-memory queue for unit testing.

use std::collections::VecDeque;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::compose::ComposedEmail;
use super::error::{EmailError, EmailResult};

// ── RawEmail ────────────────────────────────────────────────────────────

/// A raw email message as fetched from a mail server.
#[derive(Debug, Clone)]
pub struct RawEmail {
    /// Server-assigned unique identifier (IMAP UID or JMAP id).
    pub uid: String,
    /// The mailbox this message was fetched from (e.g. "INBOX").
    pub mailbox: String,
    /// Server-side flags (e.g. "\Seen", "\Flagged").
    pub flags: Vec<String>,
    /// Raw RFC 5322 message bytes.
    pub data: Vec<u8>,
}

// ── EmailConfig ─────────────────────────────────────────────────────────

/// Which protocol to use for connecting to the mail server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// JMAP (RFC 8620) over HTTPS.
    Jmap,
    /// IMAP (RFC 3501) over TLS.
    Imap,
}

/// Credentials for authenticating with the mail server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EmailCredentials {
    /// Username + app password (or regular password).
    AppPassword { user: String, pass: String },
    /// OAuth2 bearer token.
    OAuth2 { token: String },
}

/// Configuration for the email connector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    /// Protocol to use.
    pub connection_type: ConnectionType,
    /// Mail server hostname (e.g. "imap.example.com" or "jmap.example.com").
    pub host: String,
    /// Server port (993 for IMAPS, 443 for JMAP).
    pub port: u16,
    /// Authentication credentials.
    pub credentials: EmailCredentials,
    /// How often to poll for new messages (seconds). Minimum 10.
    pub poll_interval_secs: u64,
    /// Which mailbox(es) to monitor (defaults to "INBOX").
    pub mailboxes: Vec<String>,
    /// SMTP host for sending (e.g. "smtp.example.com").
    pub smtp_host: Option<String>,
    /// SMTP port (defaults to 587 for STARTTLS).
    pub smtp_port: Option<u16>,
}

impl EmailConfig {
    /// Validate this configuration, returning an error if invalid.
    pub fn validate(&self) -> EmailResult<()> {
        if self.host.is_empty() {
            return Err(EmailError::Config {
                message: "host must not be empty".to_string(),
            });
        }
        if self.port == 0 {
            return Err(EmailError::Config {
                message: "port must be non-zero".to_string(),
            });
        }
        if self.poll_interval_secs < 10 {
            return Err(EmailError::Config {
                message: format!(
                    "poll_interval_secs must be at least 10, got {}",
                    self.poll_interval_secs
                ),
            });
        }
        if self.mailboxes.is_empty() {
            return Err(EmailError::Config {
                message: "at least one mailbox must be specified".to_string(),
            });
        }
        Ok(())
    }

    /// The poll interval as a `Duration`.
    pub fn poll_interval(&self) -> Duration {
        Duration::from_secs(self.poll_interval_secs)
    }
}

impl Default for EmailConfig {
    fn default() -> Self {
        Self {
            connection_type: ConnectionType::Imap,
            host: String::new(),
            port: 993,
            credentials: EmailCredentials::AppPassword {
                user: String::new(),
                pass: String::new(),
            },
            poll_interval_secs: 60,
            mailboxes: vec!["INBOX".to_string()],
            smtp_host: None,
            smtp_port: None,
        }
    }
}

// ── EmailConnector trait ────────────────────────────────────────────────

/// Trait for fetching and sending email via different protocols.
///
/// Implementations must be `Send` for background polling threads.
pub trait EmailConnector: Send {
    /// Fetch new (unseen) messages since the last sync point.
    fn fetch_new(&mut self) -> EmailResult<Vec<RawEmail>>;

    /// Fetch a specific message by its server-assigned ID.
    fn fetch_by_id(&self, id: &str) -> EmailResult<Option<RawEmail>>;

    /// Send a composed email via the configured transport.
    fn send_email(&self, message: &ComposedEmail, from: &str) -> EmailResult<()>;

    /// Opaque sync state for delta synchronization (e.g. JMAP state token,
    /// IMAP highest-seen UID). Returns `None` if no sync has occurred.
    fn sync_state(&self) -> Option<String>;
}

// ── JmapConnector ───────────────────────────────────────────────────────

/// JMAP connector using ureq for sync HTTP.
///
/// Performs session discovery via `GET /.well-known/jmap`, then uses
/// `Email/query` + `Email/get` for fetching and `Email/set` for updates.
pub struct JmapConnector {
    config: EmailConfig,
    /// The JMAP session URL (discovered from .well-known).
    session_url: Option<String>,
    /// The JMAP API endpoint URL.
    api_url: Option<String>,
    /// Account ID from the JMAP session.
    account_id: Option<String>,
    /// Delta sync state from last `Email/changes` call.
    state: Option<String>,
}

impl JmapConnector {
    /// Create a new JMAP connector (does not connect yet).
    pub fn new(config: EmailConfig) -> Self {
        Self {
            config,
            session_url: None,
            api_url: None,
            account_id: None,
            state: None,
        }
    }

    /// Discover the JMAP session endpoint and extract API URL + account ID.
    pub fn discover(&mut self) -> EmailResult<()> {
        let well_known = format!("https://{}/.well-known/jmap", self.config.host);
        let (user, pass) = match &self.config.credentials {
            EmailCredentials::AppPassword { user, pass } => (user.clone(), pass.clone()),
            EmailCredentials::OAuth2 { .. } => {
                return Err(EmailError::Config {
                    message: "JMAP OAuth2 not yet implemented".to_string(),
                });
            }
        };

        let resp = ureq::get(&well_known)
            .set("Authorization", &basic_auth(&user, &pass))
            .call()
            .map_err(|e| EmailError::Connection {
                message: format!("JMAP discovery failed: {e}"),
            })?;

        let body: serde_json::Value =
            resp.into_json().map_err(|e| EmailError::Connection {
                message: format!("JMAP discovery response parse failed: {e}"),
            })?;

        self.api_url = body["apiUrl"].as_str().map(|s| s.to_string());
        self.session_url = Some(well_known);

        // Extract primary account ID.
        if let Some(accounts) = body["primaryAccounts"].as_object() {
            if let Some(mail_account) = accounts.get("urn:ietf:params:jmap:mail") {
                self.account_id = mail_account.as_str().map(|s| s.to_string());
            }
        }

        if self.api_url.is_none() {
            return Err(EmailError::Connection {
                message: "JMAP session missing apiUrl".to_string(),
            });
        }

        Ok(())
    }
}

impl std::fmt::Debug for JmapConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JmapConnector")
            .field("host", &self.config.host)
            .field("api_url", &self.api_url)
            .field("account_id", &self.account_id)
            .field("state", &self.state)
            .finish()
    }
}

impl EmailConnector for JmapConnector {
    fn fetch_new(&mut self) -> EmailResult<Vec<RawEmail>> {
        let api_url = self.api_url.as_ref().ok_or(EmailError::Connection {
            message: "JMAP not discovered yet — call discover() first".to_string(),
        })?;
        let account_id = self.account_id.as_ref().ok_or(EmailError::Connection {
            message: "JMAP account ID not available".to_string(),
        })?;

        let (user, pass) = match &self.config.credentials {
            EmailCredentials::AppPassword { user, pass } => (user.clone(), pass.clone()),
            EmailCredentials::OAuth2 { .. } => {
                return Err(EmailError::Config {
                    message: "JMAP OAuth2 not yet implemented".to_string(),
                });
            }
        };

        // Build Email/query + Email/get request.
        let request_body = serde_json::json!({
            "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
            "methodCalls": [
                ["Email/query", {
                    "accountId": account_id,
                    "filter": {
                        "inMailbox": self.config.mailboxes.first().unwrap_or(&"INBOX".to_string()),
                        "notKeyword": "$seen"
                    },
                    "sort": [{"property": "receivedAt", "isAscending": false}],
                    "limit": 50
                }, "q0"],
                ["Email/get", {
                    "accountId": account_id,
                    "#ids": {
                        "resultOf": "q0",
                        "name": "Email/query",
                        "path": "/ids"
                    },
                    "properties": ["id", "blobId", "threadId", "mailboxIds",
                                   "keywords", "from", "to", "cc", "subject",
                                   "receivedAt", "messageId", "inReplyTo",
                                   "references", "bodyValues", "textBody",
                                   "hasAttachment", "header:list-id:asText",
                                   "header:content-type:asText"],
                    "fetchAllBodyValues": true,
                    "maxBodyValueBytes": 4096
                }, "g0"]
            ]
        });

        let resp = ureq::post(api_url)
            .set("Authorization", &basic_auth(&user, &pass))
            .set("Content-Type", "application/json")
            .send_json(request_body)
            .map_err(|e| EmailError::Connection {
                message: format!("JMAP Email/query failed: {e}"),
            })?;

        let body: serde_json::Value = resp.into_json().map_err(|e| EmailError::Parse {
            message: format!("JMAP response parse failed: {e}"),
        })?;

        let mut emails = Vec::new();

        // Extract Email/get results.
        if let Some(method_responses) = body["methodResponses"].as_array() {
            for response in method_responses {
                if let Some(arr) = response.as_array() {
                    if arr.first().and_then(|v| v.as_str()) == Some("Email/get") {
                        if let Some(list) = arr.get(1).and_then(|v| v["list"].as_array()) {
                            for email in list {
                                let id =
                                    email["id"].as_str().unwrap_or("unknown").to_string();
                                // JMAP returns structured data, not raw RFC5322.
                                // We serialize the JSON as the "raw" data for downstream parsing.
                                let data =
                                    serde_json::to_vec(email).unwrap_or_default();
                                emails.push(RawEmail {
                                    uid: id,
                                    mailbox: self
                                        .config
                                        .mailboxes
                                        .first()
                                        .cloned()
                                        .unwrap_or_else(|| "INBOX".to_string()),
                                    flags: Vec::new(),
                                    data,
                                });
                            }
                        }
                        // Update sync state.
                        if let Some(state) = arr.get(1).and_then(|v| v["state"].as_str()) {
                            self.state = Some(state.to_string());
                        }
                    }
                }
            }
        }

        Ok(emails)
    }

    fn fetch_by_id(&self, id: &str) -> EmailResult<Option<RawEmail>> {
        let api_url = self.api_url.as_ref().ok_or(EmailError::Connection {
            message: "JMAP not discovered yet".to_string(),
        })?;
        let account_id = self.account_id.as_ref().ok_or(EmailError::Connection {
            message: "JMAP account ID not available".to_string(),
        })?;

        let (user, pass) = match &self.config.credentials {
            EmailCredentials::AppPassword { user, pass } => (user.clone(), pass.clone()),
            EmailCredentials::OAuth2 { .. } => {
                return Err(EmailError::Config {
                    message: "JMAP OAuth2 not yet implemented".to_string(),
                });
            }
        };

        let request_body = serde_json::json!({
            "using": ["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
            "methodCalls": [
                ["Email/get", {
                    "accountId": account_id,
                    "ids": [id],
                    "properties": ["id", "blobId", "from", "to", "cc", "subject",
                                   "receivedAt", "messageId", "inReplyTo",
                                   "references", "bodyValues", "textBody",
                                   "hasAttachment"],
                    "fetchAllBodyValues": true,
                    "maxBodyValueBytes": 4096
                }, "g0"]
            ]
        });

        let resp = ureq::post(api_url)
            .set("Authorization", &basic_auth(&user, &pass))
            .set("Content-Type", "application/json")
            .send_json(request_body)
            .map_err(|e| EmailError::Connection {
                message: format!("JMAP Email/get failed: {e}"),
            })?;

        let body: serde_json::Value = resp.into_json().map_err(|e| EmailError::Parse {
            message: format!("JMAP response parse failed: {e}"),
        })?;

        if let Some(method_responses) = body["methodResponses"].as_array() {
            for response in method_responses {
                if let Some(arr) = response.as_array() {
                    if arr.first().and_then(|v| v.as_str()) == Some("Email/get") {
                        if let Some(list) = arr.get(1).and_then(|v| v["list"].as_array()) {
                            if let Some(email) = list.first() {
                                let data = serde_json::to_vec(email).unwrap_or_default();
                                return Ok(Some(RawEmail {
                                    uid: id.to_string(),
                                    mailbox: "INBOX".to_string(),
                                    flags: Vec::new(),
                                    data,
                                }));
                            }
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    fn send_email(&self, _message: &ComposedEmail, _from: &str) -> EmailResult<()> {
        // JMAP Email/set submission — future enhancement.
        // For now, email sending goes through SMTP (handled at EmailChannel level).
        Err(EmailError::Send {
            message: "JMAP Email/set submission not yet implemented — use SMTP".to_string(),
        })
    }

    fn sync_state(&self) -> Option<String> {
        self.state.clone()
    }
}

// ── ImapConnector ───────────────────────────────────────────────────────

/// IMAP connector using the `imap` crate (sync, TLS).
///
/// Connects via IMAPS, uses `SEARCH UNSEEN` for new messages, and tracks
/// the highest seen UID for delta sync.
pub struct ImapConnector {
    config: EmailConfig,
    /// Highest UID seen so far (for delta sync).
    last_uid: Option<u32>,
}

impl ImapConnector {
    /// Create a new IMAP connector (does not connect yet).
    pub fn new(config: EmailConfig) -> Self {
        Self {
            config,
            last_uid: None,
        }
    }
}

impl std::fmt::Debug for ImapConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImapConnector")
            .field("host", &self.config.host)
            .field("port", &self.config.port)
            .field("last_uid", &self.last_uid)
            .finish()
    }
}

impl ImapConnector {
    /// Establish a TLS connection and login.
    fn connect_and_login(
        &self,
    ) -> EmailResult<imap::Session<native_tls::TlsStream<std::net::TcpStream>>> {
        let (user, pass) = match &self.config.credentials {
            EmailCredentials::AppPassword { user, pass } => (user.clone(), pass.clone()),
            EmailCredentials::OAuth2 { .. } => {
                return Err(EmailError::Config {
                    message: "IMAP OAuth2 not yet implemented".to_string(),
                });
            }
        };

        let tls = native_tls::TlsConnector::builder()
            .build()
            .map_err(|e| EmailError::Connection {
                message: format!("TLS connector build failed: {e}"),
            })?;

        let addr = (&*self.config.host, self.config.port);
        let client =
            imap::connect(addr, &self.config.host, &tls).map_err(|e| {
                EmailError::Connection {
                    message: format!("IMAP connection failed: {e}"),
                }
            })?;

        let session = client.login(&user, &pass).map_err(|e| {
            EmailError::Authentication {
                message: format!("IMAP login failed: {}", e.0),
            }
        })?;

        Ok(session)
    }
}

impl EmailConnector for ImapConnector {
    fn fetch_new(&mut self) -> EmailResult<Vec<RawEmail>> {
        let mut session = self.connect_and_login()?;

        let mailbox = self
            .config
            .mailboxes
            .first()
            .cloned()
            .unwrap_or_else(|| "INBOX".to_string());

        // EXAMINE (read-only) to avoid marking messages as seen.
        session.examine(&mailbox).map_err(|e| EmailError::Connection {
            message: format!("IMAP EXAMINE {mailbox} failed: {e}"),
        })?;

        // Search for unseen messages, optionally above our last UID.
        let search_query = if let Some(uid) = self.last_uid {
            format!("UID {}:* UNSEEN", uid + 1)
        } else {
            "UNSEEN".to_string()
        };

        let uids = session.uid_search(&search_query).map_err(|e| {
            EmailError::Connection {
                message: format!("IMAP UID SEARCH failed: {e}"),
            }
        })?;

        if uids.is_empty() {
            session.logout().ok();
            return Ok(Vec::new());
        }

        // Build UID set string.
        let uid_set: String = uids
            .iter()
            .map(|u: &u32| u.to_string())
            .collect::<Vec<_>>()
            .join(",");

        // Fetch full RFC822 messages.
        let fetches = session
            .uid_fetch(&uid_set, "RFC822")
            .map_err(|e| EmailError::Connection {
                message: format!("IMAP UID FETCH failed: {e}"),
            })?;

        let mut emails = Vec::new();
        for fetch in fetches.iter() {
            if let Some(body) = fetch.body() {
                let uid: u32 = fetch.uid.unwrap_or(0);
                emails.push(RawEmail {
                    uid: uid.to_string(),
                    mailbox: mailbox.clone(),
                    flags: Vec::new(),
                    data: body.to_vec(),
                });
                // Track highest UID.
                if self.last_uid.is_none() || self.last_uid < Some(uid) {
                    self.last_uid = Some(uid);
                }
            }
        }

        session.logout().ok();
        Ok(emails)
    }

    fn fetch_by_id(&self, id: &str) -> EmailResult<Option<RawEmail>> {
        let uid: u32 = id.parse().map_err(|_| EmailError::Config {
            message: format!("invalid IMAP UID: {id}"),
        })?;

        let mut session = self.connect_and_login()?;

        let mailbox = self
            .config
            .mailboxes
            .first()
            .cloned()
            .unwrap_or_else(|| "INBOX".to_string());

        session.examine(&mailbox).map_err(|e| EmailError::Connection {
            message: format!("IMAP EXAMINE failed: {e}"),
        })?;

        let fetches = session
            .uid_fetch(uid.to_string(), "RFC822")
            .map_err(|e| EmailError::Connection {
                message: format!("IMAP UID FETCH failed: {e}"),
            })?;

        let result = fetches
            .iter()
            .next()
            .and_then(|f: &imap::types::Fetch| {
                f.body().map(|body: &[u8]| RawEmail {
                    uid: id.to_string(),
                    mailbox: mailbox.clone(),
                    flags: Vec::new(),
                    data: body.to_vec(),
                })
            });

        session.logout().ok();
        Ok(result)
    }

    fn send_email(&self, _message: &ComposedEmail, _from: &str) -> EmailResult<()> {
        // IMAP doesn't send — SMTP is used at the EmailChannel level.
        Err(EmailError::Send {
            message: "IMAP cannot send email — use SMTP transport".to_string(),
        })
    }

    fn sync_state(&self) -> Option<String> {
        self.last_uid.map(|uid| uid.to_string())
    }
}

// ── MockConnector ───────────────────────────────────────────────────────

/// In-memory mock connector for unit testing.
///
/// Push raw emails to simulate incoming mail; inspect `sent()` to verify outbound.
pub struct MockConnector {
    inbox: VecDeque<RawEmail>,
    sent: Vec<(ComposedEmail, String)>,
    state: Option<String>,
}

impl MockConnector {
    /// Create an empty mock connector.
    pub fn new() -> Self {
        Self {
            inbox: VecDeque::new(),
            sent: Vec::new(),
            state: None,
        }
    }

    /// Push a raw email to simulate an incoming message.
    pub fn push_raw(&mut self, raw: RawEmail) {
        self.inbox.push_back(raw);
    }

    /// Get all sent emails (for test assertions).
    pub fn sent(&self) -> &[(ComposedEmail, String)] {
        &self.sent
    }
}

impl Default for MockConnector {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for MockConnector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockConnector")
            .field("inbox_len", &self.inbox.len())
            .field("sent_len", &self.sent.len())
            .finish()
    }
}

impl EmailConnector for MockConnector {
    fn fetch_new(&mut self) -> EmailResult<Vec<RawEmail>> {
        let emails: Vec<RawEmail> = self.inbox.drain(..).collect();
        if let Some(last) = emails.last() {
            self.state = Some(last.uid.clone());
        }
        Ok(emails)
    }

    fn fetch_by_id(&self, id: &str) -> EmailResult<Option<RawEmail>> {
        Ok(self.inbox.iter().find(|e| e.uid == id).cloned())
    }

    fn send_email(&self, message: &ComposedEmail, from: &str) -> EmailResult<()> {
        // MockConnector needs &mut for sending — cast via interior mutability
        // is avoided; tests should use the direct push pattern instead.
        // For trait compliance, we accept &self but cannot actually store.
        // Tests should use the mutable handle directly.
        let _ = (message, from);
        Ok(())
    }

    fn sync_state(&self) -> Option<String> {
        self.state.clone()
    }
}

impl MockConnector {
    /// Send an email via the mock (mutable version for tests).
    pub fn mock_send(&mut self, message: ComposedEmail, from: String) {
        self.sent.push((message, from));
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Build a basic HTTP Authorization header value.
fn basic_auth(user: &str, pass: &str) -> String {
    use std::io::Write;
    let mut buf = Vec::new();
    write!(buf, "{user}:{pass}").unwrap();
    format!("Basic {}", base64_encode(&buf))
}

/// Minimal base64 encoder (avoids adding a base64 crate dependency).
fn base64_encode(input: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((input.len() + 2) / 3 * 4);
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_raw(uid: &str, data: &[u8]) -> RawEmail {
        RawEmail {
            uid: uid.to_string(),
            mailbox: "INBOX".to_string(),
            flags: vec!["\\Recent".to_string()],
            data: data.to_vec(),
        }
    }

    #[test]
    fn mock_connector_fifo() {
        let mut mock = MockConnector::new();
        mock.push_raw(make_raw("1", b"first"));
        mock.push_raw(make_raw("2", b"second"));
        mock.push_raw(make_raw("3", b"third"));

        let fetched = mock.fetch_new().unwrap();
        assert_eq!(fetched.len(), 3);
        assert_eq!(fetched[0].uid, "1");
        assert_eq!(fetched[1].uid, "2");
        assert_eq!(fetched[2].uid, "3");
    }

    #[test]
    fn mock_connector_fetch_drains() {
        let mut mock = MockConnector::new();
        mock.push_raw(make_raw("1", b"msg"));

        let first = mock.fetch_new().unwrap();
        assert_eq!(first.len(), 1);

        let second = mock.fetch_new().unwrap();
        assert!(second.is_empty());
    }

    #[test]
    fn mock_connector_fetch_by_id() {
        let mut mock = MockConnector::new();
        mock.push_raw(make_raw("42", b"target"));
        mock.push_raw(make_raw("99", b"other"));

        let found = mock.fetch_by_id("42").unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().uid, "42");

        let not_found = mock.fetch_by_id("999").unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn mock_connector_sync_state() {
        let mut mock = MockConnector::new();
        assert!(mock.sync_state().is_none());

        mock.push_raw(make_raw("5", b"msg"));
        mock.fetch_new().unwrap();
        assert_eq!(mock.sync_state(), Some("5".to_string()));
    }

    #[test]
    fn mock_connector_send() {
        let mut mock = MockConnector::new();
        let email = ComposedEmail {
            to: vec!["bob@example.com".to_string()],
            cc: Vec::new(),
            subject: "test".to_string(),
            body_text: "hello".to_string(),
            in_reply_to: None,
            references: Vec::new(),
        };
        mock.mock_send(email, "alice@example.com".to_string());
        assert_eq!(mock.sent().len(), 1);
        assert_eq!(mock.sent()[0].0.subject, "test");
    }

    #[test]
    fn config_validate_valid() {
        let config = EmailConfig {
            host: "imap.example.com".to_string(),
            port: 993,
            poll_interval_secs: 60,
            mailboxes: vec!["INBOX".to_string()],
            ..Default::default()
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_validate_empty_host() {
        let config = EmailConfig {
            host: String::new(),
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("host"));
    }

    #[test]
    fn config_validate_zero_port() {
        let config = EmailConfig {
            host: "example.com".to_string(),
            port: 0,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("port"));
    }

    #[test]
    fn config_validate_low_poll_interval() {
        let config = EmailConfig {
            host: "example.com".to_string(),
            port: 993,
            poll_interval_secs: 5,
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("poll_interval"));
    }

    #[test]
    fn config_validate_empty_mailboxes() {
        let config = EmailConfig {
            host: "example.com".to_string(),
            port: 993,
            poll_interval_secs: 60,
            mailboxes: Vec::new(),
            ..Default::default()
        };
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("mailbox"));
    }

    #[test]
    fn config_poll_interval_duration() {
        let config = EmailConfig {
            poll_interval_secs: 120,
            ..Default::default()
        };
        assert_eq!(config.poll_interval(), Duration::from_secs(120));
    }

    #[test]
    fn base64_encode_basic() {
        assert_eq!(base64_encode(b"user:pass"), "dXNlcjpwYXNz");
        assert_eq!(base64_encode(b"a"), "YQ==");
        assert_eq!(base64_encode(b"ab"), "YWI=");
        assert_eq!(base64_encode(b"abc"), "YWJj");
    }

    #[test]
    fn basic_auth_header() {
        let header = basic_auth("user", "pass");
        assert!(header.starts_with("Basic "));
        assert_eq!(header, "Basic dXNlcjpwYXNz");
    }
}
