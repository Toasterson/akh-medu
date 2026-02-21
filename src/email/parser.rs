//! MIME email parsing via `mail-parser`.
//!
//! Converts raw RFC 5322 bytes into a structured `ParsedEmail` with
//! extracted headers, body text, and threading metadata.

use mail_parser::{MimeHeaders, MessageParser};

use super::connector::RawEmail;
use super::error::{EmailError, EmailResult};

/// Maximum body text size (4 KB) — prevents memory blow-up on large emails.
const MAX_BODY_TEXT: usize = 4096;

/// Maximum HTML body size (8 KB).
const MAX_BODY_HTML: usize = 8192;

// ── ParsedEmail ─────────────────────────────────────────────────────────

/// Structured representation of a parsed email message.
#[derive(Debug, Clone)]
pub struct ParsedEmail {
    /// RFC 5322 Message-ID (e.g. `<abc@example.com>`).
    pub message_id: String,
    /// Sender email address.
    pub from: String,
    /// Sender display name (if available).
    pub from_display: Option<String>,
    /// Recipient addresses.
    pub to: Vec<String>,
    /// CC addresses.
    pub cc: Vec<String>,
    /// Subject line.
    pub subject: String,
    /// Date as unix timestamp (seconds since epoch).
    pub date: Option<u64>,
    /// In-Reply-To header (Message-ID of parent).
    pub in_reply_to: Option<String>,
    /// References header (chain of Message-IDs).
    pub references: Vec<String>,
    /// Plain text body (truncated to `MAX_BODY_TEXT`).
    pub body_text: Option<String>,
    /// HTML body (truncated to `MAX_BODY_HTML`).
    pub body_html: Option<String>,
    /// Whether the message has attachments.
    pub has_attachments: bool,
    /// List-Id header (for mailing list detection).
    pub list_id: Option<String>,
    /// Top-level Content-Type header value.
    pub content_type: String,
}

impl ParsedEmail {
    /// A short summary suitable for display: "Subject (from sender)".
    pub fn summary(&self) -> String {
        let from_display = self
            .from_display
            .as_deref()
            .unwrap_or(self.from.as_str());
        format!("{} (from {})", self.subject, from_display)
    }

    /// The body text, preferring plain text over HTML.
    pub fn best_body(&self) -> Option<&str> {
        self.body_text
            .as_deref()
            .or(self.body_html.as_deref())
    }
}

// ── parse_raw ───────────────────────────────────────────────────────────

/// Parse a `RawEmail` into a `ParsedEmail`.
///
/// Uses `mail-parser` for full RFC 5322 / MIME conformance. Handles
/// multipart/alternative (prefers text/plain), multipart/mixed (detects
/// attachments), and nested MIME structures.
pub fn parse_raw(raw: &RawEmail) -> EmailResult<ParsedEmail> {
    let message = MessageParser::default()
        .parse(&raw.data)
        .ok_or_else(|| EmailError::Parse {
            message: format!(
                "failed to parse MIME message (uid: {}, {} bytes)",
                raw.uid,
                raw.data.len()
            ),
        })?;

    // Message-ID
    let message_id = message
        .message_id()
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("<generated-{}>", raw.uid));

    // From — mail-parser returns Option<&Address>
    let (from, from_display) = extract_sender(&message);

    // To — Option<&Address>
    let to = message
        .to()
        .map(|addr| extract_addresses(addr))
        .unwrap_or_default();

    // Cc — Option<&Address>
    let cc = message
        .cc()
        .map(|addr| extract_addresses(addr))
        .unwrap_or_default();

    // Subject
    let subject = message
        .subject()
        .unwrap_or("[no subject]")
        .to_string();

    // Date
    let date = message.date().map(|dt| dt.to_timestamp() as u64);

    // In-Reply-To — returns &HeaderValue (not Option)
    let in_reply_to = extract_first_text_or_id(message.in_reply_to());

    // References — returns &HeaderValue (not Option)
    let references = extract_text_list(message.references());

    // Body text (plain)
    let body_text = message.body_text(0).map(|s| truncate(&s, MAX_BODY_TEXT));

    // Body HTML
    let body_html = message.body_html(0).map(|s| truncate(&s, MAX_BODY_HTML));

    // Attachments
    let has_attachments = message.attachment_count() > 0;

    // List-Id — header_raw returns Option<&str>
    let list_id = message
        .header_raw("List-Id")
        .map(|s| s.trim().to_string());

    // Content-Type
    let content_type = message
        .content_type()
        .map(|ct| {
            let ctype = ct.ctype();
            if let Some(subtype) = ct.subtype() {
                format!("{ctype}/{subtype}")
            } else {
                ctype.to_string()
            }
        })
        .unwrap_or_else(|| "text/plain".to_string());

    Ok(ParsedEmail {
        message_id,
        from,
        from_display,
        to,
        cc,
        subject,
        date,
        in_reply_to,
        references,
        body_text,
        body_html,
        has_attachments,
        list_id,
        content_type,
    })
}

// ── extract_domain ──────────────────────────────────────────────────────

/// Extract the domain part from an email address.
///
/// Returns the part after `@`, or the whole string if no `@` is found.
pub fn extract_domain(email_address: &str) -> &str {
    // Handle angle-bracket wrapped addresses: <user@domain.com>
    let addr = email_address
        .trim()
        .trim_start_matches('<')
        .trim_end_matches('>');

    match addr.rsplit_once('@') {
        Some((_, domain)) => domain,
        None => addr,
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Extract sender address and display name from a parsed message.
fn extract_sender(message: &mail_parser::Message<'_>) -> (String, Option<String>) {
    if let Some(from_addr) = message.from() {
        // Address enum can be List or Group.
        if let Some(first) = from_addr.first() {
            let email = first
                .address()
                .unwrap_or("unknown@unknown")
                .to_string();
            let display = first.name().map(|n| n.to_string());
            (email, display)
        } else {
            ("unknown@unknown".to_string(), None)
        }
    } else {
        ("unknown@unknown".to_string(), None)
    }
}

/// Extract email addresses from an `Address` value (To, Cc, etc.).
fn extract_addresses(addr: &mail_parser::Address<'_>) -> Vec<String> {
    addr.iter()
        .filter_map(|a| a.address().map(|s| s.to_string()))
        .collect()
}

/// Extract the first text value from a HeaderValue (for In-Reply-To).
fn extract_first_text_or_id(hv: &mail_parser::HeaderValue<'_>) -> Option<String> {
    match hv {
        mail_parser::HeaderValue::Text(s) => Some(s.to_string()),
        mail_parser::HeaderValue::TextList(list) => list.first().map(|s| s.to_string()),
        _ => None,
    }
}

/// Extract all text values from a HeaderValue (for References).
fn extract_text_list(hv: &mail_parser::HeaderValue<'_>) -> Vec<String> {
    match hv {
        mail_parser::HeaderValue::Text(s) => vec![s.to_string()],
        mail_parser::HeaderValue::TextList(list) => {
            list.iter().map(|s| s.to_string()).collect()
        }
        _ => Vec::new(),
    }
}

/// Truncate a string to at most `max_bytes` bytes (on a char boundary).
fn truncate(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        s.to_string()
    } else {
        let mut end = max_bytes;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        let mut result = s[..end].to_string();
        result.push_str("...");
        result
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_from_str(uid: &str, text: &str) -> RawEmail {
        RawEmail {
            uid: uid.to_string(),
            mailbox: "INBOX".to_string(),
            flags: Vec::new(),
            data: text.as_bytes().to_vec(),
        }
    }

    const SIMPLE_EMAIL: &str = "\
From: Alice <alice@example.com>\r\n\
To: Bob <bob@example.com>\r\n\
Subject: Hello Bob\r\n\
Message-ID: <msg-001@example.com>\r\n\
Date: Sat, 20 Nov 2021 14:22:01 -0800\r\n\
Content-Type: text/plain; charset=utf-8\r\n\
\r\n\
Hi Bob, this is a test email.\r\n";

    #[test]
    fn parse_simple_text_email() {
        let raw = raw_from_str("1", SIMPLE_EMAIL);
        let parsed = parse_raw(&raw).unwrap();

        assert_eq!(parsed.message_id, "msg-001@example.com");
        assert_eq!(parsed.from, "alice@example.com");
        assert_eq!(parsed.from_display, Some("Alice".to_string()));
        assert_eq!(parsed.to, vec!["bob@example.com"]);
        assert!(parsed.cc.is_empty());
        assert_eq!(parsed.subject, "Hello Bob");
        assert!(parsed.date.is_some());
        assert!(parsed.in_reply_to.is_none());
        assert!(parsed.references.is_empty());
        assert!(parsed.body_text.is_some());
        assert!(parsed.body_text.unwrap().contains("test email"));
        assert!(!parsed.has_attachments);
        assert_eq!(parsed.content_type, "text/plain");
    }

    #[test]
    fn parse_reply_with_references() {
        let email = "\
From: Bob <bob@example.com>\r\n\
To: Alice <alice@example.com>\r\n\
Subject: Re: Hello Bob\r\n\
Message-ID: <msg-002@example.com>\r\n\
In-Reply-To: <msg-001@example.com>\r\n\
References: <msg-001@example.com>\r\n\
Date: Sun, 21 Nov 2021 10:00:00 -0800\r\n\
Content-Type: text/plain\r\n\
\r\n\
Hi Alice, thanks for writing!\r\n";

        let raw = raw_from_str("2", email);
        let parsed = parse_raw(&raw).unwrap();

        assert_eq!(parsed.in_reply_to, Some("msg-001@example.com".to_string()));
        assert_eq!(parsed.references, vec!["msg-001@example.com"]);
        assert!(parsed.subject.starts_with("Re:"));
    }

    #[test]
    fn parse_multipart_alternative() {
        let email = "\
From: sender@example.com\r\n\
To: recipient@example.com\r\n\
Subject: Multipart test\r\n\
Message-ID: <multi-001@example.com>\r\n\
Content-Type: multipart/alternative; boundary=\"boundary42\"\r\n\
\r\n\
--boundary42\r\n\
Content-Type: text/plain\r\n\
\r\n\
Plain text body\r\n\
--boundary42\r\n\
Content-Type: text/html\r\n\
\r\n\
<p>HTML body</p>\r\n\
--boundary42--\r\n";

        let raw = raw_from_str("3", email);
        let parsed = parse_raw(&raw).unwrap();

        // Should have both text and HTML.
        assert!(parsed.body_text.is_some());
        assert!(parsed.body_html.is_some());
        assert!(parsed.body_text.unwrap().contains("Plain text body"));
        assert!(parsed.body_html.unwrap().contains("HTML body"));
    }

    #[test]
    fn parse_missing_message_id() {
        let email = "\
From: sender@example.com\r\n\
To: recipient@example.com\r\n\
Subject: No message ID\r\n\
Content-Type: text/plain\r\n\
\r\n\
Body text\r\n";

        let raw = raw_from_str("99", email);
        let parsed = parse_raw(&raw).unwrap();

        // Should generate a fallback message ID.
        assert!(parsed.message_id.contains("99"));
    }

    #[test]
    fn parse_cc_addresses() {
        let email = "\
From: sender@example.com\r\n\
To: alice@example.com\r\n\
Cc: bob@example.com, carol@example.com\r\n\
Subject: CC test\r\n\
Message-ID: <cc-001@example.com>\r\n\
Content-Type: text/plain\r\n\
\r\n\
Body\r\n";

        let raw = raw_from_str("4", email);
        let parsed = parse_raw(&raw).unwrap();

        assert_eq!(parsed.cc.len(), 2);
        assert!(parsed.cc.contains(&"bob@example.com".to_string()));
        assert!(parsed.cc.contains(&"carol@example.com".to_string()));
    }

    #[test]
    fn extract_domain_simple() {
        assert_eq!(extract_domain("alice@example.com"), "example.com");
    }

    #[test]
    fn extract_domain_angle_brackets() {
        assert_eq!(extract_domain("<alice@example.com>"), "example.com");
    }

    #[test]
    fn extract_domain_no_at() {
        assert_eq!(extract_domain("localhost"), "localhost");
    }

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate("hello", 100), "hello");
    }

    #[test]
    fn truncate_long_string() {
        let long = "a".repeat(5000);
        let truncated = truncate(&long, 100);
        assert!(truncated.len() <= 103); // 100 + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn parsed_email_summary() {
        let raw = raw_from_str("1", SIMPLE_EMAIL);
        let parsed = parse_raw(&raw).unwrap();
        let summary = parsed.summary();
        assert!(summary.contains("Hello Bob"));
        assert!(summary.contains("Alice"));
    }

    #[test]
    fn parsed_email_best_body() {
        let raw = raw_from_str("1", SIMPLE_EMAIL);
        let parsed = parse_raw(&raw).unwrap();
        assert!(parsed.best_body().is_some());
    }
}
