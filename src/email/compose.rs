//! Email composition: replies and new messages rendered to RFC 5322 MIME.
//!
//! Uses `lettre::message::Message` for standards-compliant MIME generation.

use super::error::{EmailError, EmailResult};
use super::parser::ParsedEmail;

// ── ComposedEmail ───────────────────────────────────────────────────────

/// A composed email ready for SMTP delivery.
#[derive(Debug, Clone)]
pub struct ComposedEmail {
    /// Recipient addresses.
    pub to: Vec<String>,
    /// CC addresses.
    pub cc: Vec<String>,
    /// Subject line.
    pub subject: String,
    /// Plain text body.
    pub body_text: String,
    /// In-Reply-To header (for threading).
    pub in_reply_to: Option<String>,
    /// References chain (for threading).
    pub references: Vec<String>,
}

// ── compose_reply ───────────────────────────────────────────────────────

/// Compose a reply to an existing email.
///
/// Sets proper `In-Reply-To`, `References`, `Re:` subject prefix,
/// and quotes the original body.
pub fn compose_reply(original: &ParsedEmail, body: &str) -> ComposedEmail {
    // Subject: add "Re: " prefix if not already present.
    let subject = if original
        .subject
        .to_lowercase()
        .starts_with("re:")
    {
        original.subject.clone()
    } else {
        format!("Re: {}", original.subject)
    };

    // Build References chain: original's references + original's message ID.
    let mut references = original.references.clone();
    if !references.contains(&original.message_id) {
        references.push(original.message_id.clone());
    }

    // Quote the original body.
    let quoted = quote_body(original);
    let full_body = format!("{body}\n\n{quoted}");

    ComposedEmail {
        to: vec![original.from.clone()],
        cc: Vec::new(),
        subject,
        body_text: full_body,
        in_reply_to: Some(original.message_id.clone()),
        references,
    }
}

// ── compose_new ─────────────────────────────────────────────────────────

/// Compose a new email (not a reply).
pub fn compose_new(to: Vec<String>, subject: &str, body: &str) -> ComposedEmail {
    ComposedEmail {
        to,
        cc: Vec::new(),
        subject: subject.to_string(),
        body_text: body.to_string(),
        in_reply_to: None,
        references: Vec::new(),
    }
}

// ── to_mime ─────────────────────────────────────────────────────────────

/// Render a `ComposedEmail` to an RFC 5322 MIME string via `lettre`.
pub fn to_mime(email: &ComposedEmail, from: &str) -> EmailResult<String> {
    use lettre::message::header;
    use lettre::message::Message;

    let from_mailbox: lettre::message::Mailbox =
        from.parse().map_err(|e| EmailError::Send {
            message: format!("invalid From address \"{from}\": {e}"),
        })?;

    let mut builder = Message::builder()
        .from(from_mailbox)
        .subject(&email.subject);

    // Add recipients.
    for addr in &email.to {
        let mailbox: lettre::message::Mailbox =
            addr.parse().map_err(|e| EmailError::Send {
                message: format!("invalid To address \"{addr}\": {e}"),
            })?;
        builder = builder.to(mailbox);
    }

    // Add CC.
    for addr in &email.cc {
        let mailbox: lettre::message::Mailbox =
            addr.parse().map_err(|e| EmailError::Send {
                message: format!("invalid Cc address \"{addr}\": {e}"),
            })?;
        builder = builder.cc(mailbox);
    }

    // In-Reply-To header.
    if let Some(ref reply_to) = email.in_reply_to {
        builder = builder.in_reply_to(reply_to.clone());
    }

    // References headers.
    for ref_id in &email.references {
        builder = builder.references(ref_id.clone());
    }

    // Set Content-Type to UTF-8 plain text.
    builder = builder.header(header::ContentType::TEXT_PLAIN);

    let message = builder
        .body(email.body_text.clone())
        .map_err(|e| EmailError::Send {
            message: format!("failed to build MIME message: {e}"),
        })?;

    let formatted = message.formatted();
    String::from_utf8(formatted).map_err(|e| EmailError::Send {
        message: format!("MIME output is not valid UTF-8: {e}"),
    })
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Quote the original message body for inclusion in a reply.
fn quote_body(original: &ParsedEmail) -> String {
    let sender = original
        .from_display
        .as_deref()
        .unwrap_or(&original.from);

    let body = original
        .body_text
        .as_deref()
        .unwrap_or("[no text]");

    let quoted_lines: String = body
        .lines()
        .map(|line| format!("> {line}"))
        .collect::<Vec<_>>()
        .join("\n");

    format!("On {date}, {sender} wrote:\n{quoted_lines}",
        date = original
            .date
            .map(|ts| format!("{ts}"))
            .unwrap_or_else(|| "[unknown date]".to_string()),
    )
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_email() -> ParsedEmail {
        ParsedEmail {
            message_id: "<orig-001@example.com>".to_string(),
            from: "alice@example.com".to_string(),
            from_display: Some("Alice".to_string()),
            to: vec!["bob@example.com".to_string()],
            cc: Vec::new(),
            subject: "Hello".to_string(),
            date: Some(1700000000),
            in_reply_to: None,
            references: Vec::new(),
            body_text: Some("How are you?".to_string()),
            body_html: None,
            has_attachments: false,
            list_id: None,
            content_type: "text/plain".to_string(),
        }
    }

    #[test]
    fn reply_adds_re_prefix() {
        let original = sample_email();
        let reply = compose_reply(&original, "I'm fine!");
        assert!(reply.subject.starts_with("Re: "));
        assert_eq!(reply.subject, "Re: Hello");
    }

    #[test]
    fn reply_no_double_re() {
        let mut original = sample_email();
        original.subject = "Re: Hello".to_string();
        let reply = compose_reply(&original, "Thanks!");
        assert_eq!(reply.subject, "Re: Hello");
    }

    #[test]
    fn reply_sets_in_reply_to() {
        let original = sample_email();
        let reply = compose_reply(&original, "Reply body");
        assert_eq!(reply.in_reply_to, Some("<orig-001@example.com>".to_string()));
    }

    #[test]
    fn reply_builds_references_chain() {
        let mut original = sample_email();
        original.references = vec!["<ancestor@example.com>".to_string()];
        let reply = compose_reply(&original, "Reply");
        assert_eq!(reply.references.len(), 2);
        assert!(reply.references.contains(&"<ancestor@example.com>".to_string()));
        assert!(reply.references.contains(&"<orig-001@example.com>".to_string()));
    }

    #[test]
    fn reply_quotes_original() {
        let original = sample_email();
        let reply = compose_reply(&original, "Great!");
        assert!(reply.body_text.contains("> How are you?"));
        assert!(reply.body_text.contains("Alice wrote:"));
    }

    #[test]
    fn reply_sends_to_original_sender() {
        let original = sample_email();
        let reply = compose_reply(&original, "Reply");
        assert_eq!(reply.to, vec!["alice@example.com"]);
    }

    #[test]
    fn compose_new_email() {
        let email = compose_new(
            vec!["recipient@example.com".to_string()],
            "New topic",
            "Hello!",
        );
        assert_eq!(email.subject, "New topic");
        assert_eq!(email.body_text, "Hello!");
        assert!(email.in_reply_to.is_none());
        assert!(email.references.is_empty());
    }

    #[test]
    fn to_mime_produces_valid_output() {
        let email = compose_new(
            vec!["bob@example.com".to_string()],
            "Test subject",
            "Test body",
        );
        let mime = to_mime(&email, "alice@example.com").unwrap();
        assert!(mime.contains("From:"));
        assert!(mime.contains("To:"));
        assert!(mime.contains("Subject:"));
        assert!(mime.contains("Test body"));
    }

    #[test]
    fn to_mime_reply_includes_threading_headers() {
        let original = sample_email();
        let reply = compose_reply(&original, "Reply text");
        let mime = to_mime(&reply, "bob@example.com").unwrap();
        assert!(mime.contains("In-Reply-To:"));
        assert!(mime.contains("References:"));
    }

    #[test]
    fn to_mime_invalid_from_returns_error() {
        let email = compose_new(
            vec!["bob@example.com".to_string()],
            "Test",
            "Body",
        );
        let result = to_mime(&email, "not-an-email");
        assert!(result.is_err());
    }
}
