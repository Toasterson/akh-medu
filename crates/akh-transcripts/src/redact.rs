//! redact — the credential detector. Pattern set, what's excluded, and the
//! mask-vs-withhold decision: measured against the whole 68k-record corpus
//! in `docs/wayfinder/transcript-harvest/tickets/redaction-detector.md`,
//! ported from `assets/scripts/redaction_prototype.py`.
//!
//! Deliberately **no** entropy fallback and **no** cue-word rule: both were
//! measured and rejected. The entropy rule would flag ~15% of the corpus,
//! 68.4% of it sitting on `signature`/`encrypted_content` fields (model
//! ciphertext, not secrets). The cue-word rule collides head-on with this
//! corpus's own vocabulary — "token" alone drives 61.4% cue-word hits,
//! because `token_count`/`token_usage_record` mean *LLM tokens* here, not
//! credentials.
//!
//! `scan` is applied only to a [`crate::normalize::NormalizedEvent::text`]
//! value. By construction (see `normalize::claude_code` and
//! `normalize::codex`), that field never holds a Claude thinking-block
//! `signature` or a Codex `encrypted_content` payload — those fields are
//! simply never read into `text` in the first place, so there is nothing
//! here to specially exclude.

use std::sync::LazyLock;

use regex::Regex;

use crate::normalize::{SecretAction, SecretFlag};

/// Ten named patterns are the whole detector — see the module docs for why
/// entropy and cue-word rules were measured and rejected. Regex literals are
/// compile-time constants; `expect` here documents "this is a fixed,
/// hand-verified pattern," not "an operation whose success I haven't
/// checked."
static AWS_ACCESS_KEY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"AKIA[0-9A-Z]{16}").expect("static aws_access_key pattern"));
static GITHUB_PAT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"gh[posru]_[A-Za-z0-9]{30,}").expect("static github_pat pattern"));
static ANTHROPIC_KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"sk-ant-[A-Za-z0-9_\-]{20,}").expect("static anthropic_key pattern")
});
static OPENAI_KEY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\bsk-[A-Za-z0-9]{32,}").expect("static openai_key pattern"));
static SLACK_TOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"xox[baprs]-[A-Za-z0-9-]{10,}").expect("static slack_token pattern")
});
static JWT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.").expect("static jwt pattern")
});
static PRIVATE_KEY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-----BEGIN [A-Z ]*PRIVATE KEY-----").expect("static private_key pattern")
});
static BEARER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Bearer\s+[A-Za-z0-9._\-]{20,}").expect("static bearer pattern"));
static PG_URL_PASSWORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"postgres(?:ql)?://[^:/\s]+:[^@\s]{3,}@").expect("static pg_url_password pattern")
});
static ASSIGNMENT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b\s*[=:]\s*"?'?[^\s"',}]{8,}"#,
    )
    .expect("static assignment pattern")
});

fn patterns() -> [(&'static str, &'static LazyLock<Regex>, SecretAction); 10] {
    [
        ("aws_access_key", &AWS_ACCESS_KEY, SecretAction::Mask),
        ("github_pat", &GITHUB_PAT, SecretAction::Mask),
        ("anthropic_key", &ANTHROPIC_KEY, SecretAction::Mask),
        ("openai_key", &OPENAI_KEY, SecretAction::Mask),
        ("slack_token", &SLACK_TOKEN, SecretAction::Mask),
        ("jwt", &JWT, SecretAction::Mask),
        // Alone among the ten: the match is the unboundable *header* of a
        // multi-line body, so the whole text is withheld rather than masked.
        ("private_key", &PRIVATE_KEY, SecretAction::Withhold),
        ("bearer", &BEARER, SecretAction::Mask),
        ("pg_url_password", &PG_URL_PASSWORD, SecretAction::Mask),
        ("assignment", &ASSIGNMENT, SecretAction::Mask),
    ]
}

/// Scan `text` for credential-shaped matches. Only ever called on a
/// [`crate::normalize::NormalizedEvent::text`] value — never on a
/// `signature` or `encrypted_content` field, which are excluded by
/// construction upstream in `normalize`.
pub fn scan(text: &str) -> Vec<SecretFlag> {
    let mut flags = Vec::new();
    for (name, pattern, action) in patterns() {
        for m in pattern.find_iter(text) {
            flags.push(SecretFlag {
                pattern: name.to_string(),
                start: m.start(),
                end: m.end(),
                action,
            });
        }
    }
    flags.sort_by_key(|f| f.start);
    flags
}

/// Apply flags found by [`scan`] to `text`.
///
/// If any flag is [`SecretAction::Withhold`] (only `private_key` produces
/// one), the *entire* text is replaced — masking just the PEM header would
/// still emit the key material beneath it, which is worse than withholding.
/// Otherwise every flagged span is masked in place with
/// `[REDACTED:<pattern>]`, leaving the rest of the text — usually a small
/// fraction of it — intact.
pub fn apply(text: &str, flags: &[SecretFlag]) -> String {
    if let Some(withheld) = flags.iter().find(|f| f.action == SecretAction::Withhold) {
        return format!("[WITHHELD: {}]", withheld.pattern);
    }
    if flags.is_empty() {
        return text.to_string();
    }

    let mut sorted: Vec<&SecretFlag> = flags.iter().collect();
    sorted.sort_by_key(|f| f.start);

    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    for flag in sorted {
        if flag.start < cursor {
            // Overlapping matches (two patterns hitting the same span) —
            // keep the first winner, don't double-mask.
            continue;
        }
        out.push_str(&text[cursor..flag.start]);
        out.push_str(&format!("[REDACTED:{}]", flag.pattern));
        cursor = flag.end;
    }
    out.push_str(&text[cursor..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::SecretAction;

    fn pattern_names(flags: &[SecretFlag]) -> Vec<&str> {
        flags.iter().map(|f| f.pattern.as_str()).collect()
    }

    #[test]
    fn aws_access_key_positive_and_negative() {
        assert_eq!(
            pattern_names(&scan("key=AKIAIOSFODNN7EXAMPLE end")),
            vec!["aws_access_key"]
        );
        assert!(scan("not a key at all").is_empty());
    }

    #[test]
    fn github_pat_positive_and_negative() {
        let hit = format!("token gho_{} in transcript", "a".repeat(36));
        assert_eq!(pattern_names(&scan(&hit)), vec!["github_pat"]);
        assert!(scan("gh_too_short").is_empty());
    }

    #[test]
    fn anthropic_key_positive_and_negative() {
        let hit = format!("sk-ant-{}", "a".repeat(25));
        assert_eq!(pattern_names(&scan(&hit)), vec!["anthropic_key"]);
        assert!(scan("sk-ant-").is_empty());
    }

    #[test]
    fn openai_key_positive_and_negative() {
        let hit = format!("sk-{}", "a".repeat(40));
        assert_eq!(pattern_names(&scan(&hit)), vec!["openai_key"]);
        assert!(scan("sk-short").is_empty());
    }

    #[test]
    fn slack_token_positive_and_negative() {
        let hit = "xoxb-1234567890-abcdefghij";
        assert_eq!(pattern_names(&scan(hit)), vec!["slack_token"]);
        assert!(scan("xoxz-not-a-real-prefix").is_empty());
    }

    #[test]
    fn jwt_positive_and_negative() {
        let hit = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature";
        assert_eq!(pattern_names(&scan(hit)), vec!["jwt"]);
        assert!(scan("eyJnotarealjwt").is_empty());
    }

    #[test]
    fn private_key_positive_and_negative_and_withholds() {
        let hit = "-----BEGIN OPENSSH PRIVATE KEY-----\nabc123\n-----END-----";
        let flags = scan(hit);
        assert_eq!(pattern_names(&flags), vec!["private_key"]);
        assert_eq!(flags[0].action, SecretAction::Withhold);
        assert_eq!(apply(hit, &flags), "[WITHHELD: private_key]");
        assert!(scan("BEGIN PRIVATE KEY without dashes").is_empty());
    }

    #[test]
    fn bearer_positive_and_negative() {
        let hit = format!("Bearer {}", "a".repeat(25));
        assert_eq!(pattern_names(&scan(&hit)), vec!["bearer"]);
        assert!(scan("Bearer short").is_empty());
    }

    #[test]
    fn pg_url_password_positive_and_negative() {
        let hit = "postgres://akh:supersecret@db.internal/akhomed";
        assert_eq!(pattern_names(&scan(hit)), vec!["pg_url_password"]);
        assert!(scan("postgres://akh@db.internal/akhomed").is_empty());
    }

    #[test]
    fn assignment_positive_and_negative() {
        let hit = "api_key = abcdefgh12345";
        assert_eq!(pattern_names(&scan(hit)), vec!["assignment"]);
        assert!(scan("this paragraph mentions a password policy").is_empty());
    }

    #[test]
    fn mask_replaces_only_the_matched_span() {
        let text = "prefix key=AKIAIOSFODNN7EXAMPLE suffix".to_string();
        let flags = scan(&text);
        let masked = apply(&text, &flags);
        assert_eq!(masked, "prefix key=[REDACTED:aws_access_key] suffix");
    }

    #[test]
    fn no_flags_returns_text_unchanged() {
        let text = "nothing secret here";
        assert_eq!(apply(text, &scan(text)), text);
    }

    #[test]
    fn signature_and_encrypted_content_blobs_produce_zero_flags() {
        // These fields are never fed into `scan` by construction (see
        // `normalize`), but the detector itself must also not be fooled if
        // it ever saw them: they are high-entropy base64, not one of the
        // ten named shapes.
        let claude_signature =
            "EqEBCkYIBRgCKkAX9y2p1kQmM3zR7bV8wJcT4nF6hL0aPq2sK9dY5uZxW1oCbEjRtGvHnMxA".repeat(4);
        assert!(scan(&claude_signature).is_empty());

        let codex_encrypted_content =
            "gAAAAABkX3ZqZvY2c3F1aGVsbG93b3JsZHRoaXNpc25vdGFzZWNyZXQtaXQtaXMtY2lwaGVy".repeat(4);
        assert!(scan(&codex_encrypted_content).is_empty());
    }

    #[test]
    fn token_count_prose_does_not_flag_as_a_secret() {
        // The sharpest lesson of the prototype: "token" in this corpus means
        // LLM tokens, not credentials. There is no cue-word rule to trip.
        assert!(scan("token_count 17311").is_empty());
        assert!(scan("token_usage_record: total_tokens=482").is_empty());
    }
}
