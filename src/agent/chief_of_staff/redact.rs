//! Credential redaction for announced tool arguments.
//!
//! An announcement carries the tool's real input so a mis-declared action is
//! visible, but it goes to the operator's thread and a durable log, so values
//! shaped like credentials are masked first. Detection is by **credential
//! format** only: provider key prefixes, JWT and PEM structure, bearer tokens
//! and URL userinfo. The set follows the measured detector in
//! `crates/akh-transcripts/src/redact.rs`. That detector's cue-word rule is
//! left out because it matches natural-language words, so a key name alone
//! never triggers redaction.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

/// Longest string kept verbatim in an announcement; longer values are cut.
const MAX_ANNOUNCED_STRING: usize = 4096;

static PATTERNS: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    [
        ("aws_access_key", r"AKIA[0-9A-Z]{16}"),
        ("github_token", r"gh[posru]_[A-Za-z0-9]{30,}"),
        ("anthropic_key", r"sk-ant-[A-Za-z0-9_\-]{20,}"),
        ("openai_key", r"\bsk-[A-Za-z0-9_\-]{32,}"),
        ("slack_token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
        (
            "jwt",
            r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]*",
        ),
        (
            "private_key",
            r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)",
        ),
        ("bearer", r"Bearer\s+[A-Za-z0-9._\-]{20,}"),
        (
            "url_password",
            r"[A-Za-z][A-Za-z0-9+.\-]*://[^:/\s@]+:[^@\s]{3,}@",
        ),
    ]
    .into_iter()
    .map(|(name, pattern)| {
        (
            name,
            Regex::new(pattern).expect("fixed, hand-verified credential pattern"),
        )
    })
    .collect()
});

/// Mask credential-shaped spans in `text` and bound its length.
pub fn redact_str(text: &str) -> String {
    let mut out = text.to_string();
    for (name, pattern) in PATTERNS.iter() {
        if pattern.is_match(&out) {
            out = pattern
                .replace_all(&out, format!("[REDACTED:{name}]").as_str())
                .into_owned();
        }
    }
    if out.len() > MAX_ANNOUNCED_STRING {
        let mut cut = MAX_ANNOUNCED_STRING;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
        out.push_str("…[truncated]");
    }
    out
}

/// Redact every string inside `value`, recursively.
pub fn redact_value(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(redact_str(text)),
        Value::Array(items) => Value::Array(items.iter().map(redact_value).collect()),
        Value::Object(map) => Value::Object(redact_arguments(map)),
        other => other.clone(),
    }
}

/// Redact a tool's argument map for announcement.
pub fn redact_arguments(arguments: &Map<String, Value>) -> Map<String, Value> {
    arguments
        .iter()
        .map(|(key, value)| (key.clone(), redact_value(value)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks_credential_formats_and_keeps_the_rest() {
        let token = format!("ghp_{}", "a".repeat(36));
        let args = serde_json::json!({
            "command": format!("curl -H 'Authorization: Bearer {}' https://x", "b".repeat(30)),
            "env": [format!("GH={token}")],
            "url": "postgres://akh:hunter22@db/akh",
            "count": 3,
        });
        let redacted = redact_value(&args);
        let text = redacted.to_string();
        assert!(!text.contains(&token), "{text}");
        assert!(!text.contains("hunter22"), "{text}");
        assert!(!text.contains(&"b".repeat(30)), "{text}");
        assert!(text.contains("[REDACTED:github_token]"));
        assert!(text.contains("curl -H"));
        assert_eq!(redacted["count"], 3);
    }

    #[test]
    fn private_key_bodies_are_masked_whole() {
        let pem =
            "-----BEGIN OPENSSH PRIVATE KEY-----\nAAAAsecret\n-----END OPENSSH PRIVATE KEY-----";
        assert_eq!(redact_str(pem), "[REDACTED:private_key]");
    }
}
