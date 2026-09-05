//! ActivityPub federation for wiki articles — Phase 22e.
//!
//! Articles created or updated in the wiki are broadcast as ActivityPub
//! `Create(Article)` or `Update(Article)` activities via the oxifed AMQP
//! broker (requires the `oxifed` feature flag).
//!
//! When an inbound activity arrives carrying a wiki article, it is ingested
//! into the local `ArticleStore` with a `DerivationKind::ArticleFederated`
//! provenance record.
//!
//! # Wire format
//!
//! Articles are serialised as a subset of ActivityStreams 2.0 JSON-LD:
//!
//! ```json
//! {
//!   "@context": "https://www.w3.org/ns/activitystreams",
//!   "type": "Article",
//!   "id": "https://akh-medu.dev/wiki/<slug>",
//!   "name": "<title>",
//!   "content": "<markdown body>",
//!   "attributedTo": "https://akh-medu.dev/actor/<author_id>",
//!   "published": "<ISO-8601 timestamp>"
//! }
//! ```

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::wiki::store::ArticleStore;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

#[derive(Debug, Error, Diagnostic)]
pub enum FederationError {
    #[error("serialization error: {message}")]
    #[diagnostic(
        code(akh::wiki::federation::serialize),
        help("The article could not be serialised to JSON-LD.")
    )]
    Serialization { message: String },

    #[error("article not found: {title}")]
    #[diagnostic(
        code(akh::wiki::federation::not_found),
        help("The article must exist in the store before it can be federated.")
    )]
    ArticleNotFound { title: String },

    #[error("inbound activity had an unsupported type: {activity_type}")]
    #[diagnostic(
        code(akh::wiki::federation::unsupported),
        help("Only Create and Update activities carrying Article objects are processed.")
    )]
    UnsupportedActivity { activity_type: String },
}

pub type FederationResult<T> = std::result::Result<T, FederationError>;

// ═══════════════════════════════════════════════════════════════════════
// Wire types
// ═══════════════════════════════════════════════════════════════════════

/// Serialised ActivityStreams 2.0 Article object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApArticle {
    #[serde(rename = "@context")]
    pub context: String,
    #[serde(rename = "type")]
    pub object_type: String,
    pub id: String,
    pub name: String,
    pub content: String,
    #[serde(rename = "attributedTo")]
    pub attributed_to: String,
    pub published: String,
}

/// ActivityStreams 2.0 Activity wrapper (Create / Update / Announce).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApActivity {
    #[serde(rename = "@context")]
    pub context: String,
    #[serde(rename = "type")]
    pub activity_type: String,
    pub id: String,
    pub actor: String,
    pub object: ApArticle,
}

// ═══════════════════════════════════════════════════════════════════════
// Outbound: build activities from the article store
// ═══════════════════════════════════════════════════════════════════════

/// Base URL prefix used to build canonical IDs.
const BASE_URL: &str = "https://akh-medu.dev";

/// Construct a `Create(Article)` activity payload for the given article title.
///
/// Returns the JSON-LD serialised activity ready to publish to an AMQP exchange.
pub fn build_create_activity(
    store: &ArticleStore,
    title: &str,
    activity_id: &str,
) -> FederationResult<String> {
    build_activity(store, title, activity_id, "Create")
}

/// Construct an `Update(Article)` activity payload.
pub fn build_update_activity(
    store: &ArticleStore,
    title: &str,
    activity_id: &str,
) -> FederationResult<String> {
    build_activity(store, title, activity_id, "Update")
}

/// Construct an `Announce(Article)` activity payload (for citations).
pub fn build_announce_activity(
    store: &ArticleStore,
    title: &str,
    activity_id: &str,
) -> FederationResult<String> {
    build_activity(store, title, activity_id, "Announce")
}

fn build_activity(
    store: &ArticleStore,
    title: &str,
    activity_id: &str,
    activity_type: &str,
) -> FederationResult<String> {
    let article = store
        .get_by_title(title)
        .ok_or_else(|| FederationError::ArticleNotFound {
            title: title.into(),
        })?;

    let slug = slugify(title);
    let article_url = format!("{BASE_URL}/wiki/{slug}");
    let author_url = format!("{BASE_URL}/actor/{}", article.header.author.get());

    let ap_article = ApArticle {
        context: "https://www.w3.org/ns/activitystreams".into(),
        object_type: "Article".into(),
        id: article_url,
        name: article.header.title.clone(),
        content: article.body.clone(),
        attributed_to: author_url.clone(),
        published: unix_to_iso8601(article.header.created),
    };

    let activity = ApActivity {
        context: "https://www.w3.org/ns/activitystreams".into(),
        activity_type: activity_type.into(),
        id: format!("{BASE_URL}/activities/{activity_id}"),
        actor: author_url,
        object: ap_article,
    };

    serde_json::to_string(&activity).map_err(|e| FederationError::Serialization {
        message: e.to_string(),
    })
}

// ═══════════════════════════════════════════════════════════════════════
// Inbound: parse incoming activities
// ═══════════════════════════════════════════════════════════════════════

/// Parse an inbound ActivityPub JSON payload.
///
/// Returns `(activity_type, title, body)` for supported activities, or an
/// error for unsupported types.
pub fn parse_inbound_activity(json: &str) -> FederationResult<(String, String, String)> {
    let activity: ApActivity =
        serde_json::from_str(json).map_err(|e| FederationError::Serialization {
            message: e.to_string(),
        })?;

    match activity.activity_type.as_str() {
        "Create" | "Update" | "Announce" => {}
        other => {
            return Err(FederationError::UnsupportedActivity {
                activity_type: other.into(),
            });
        }
    }

    Ok((
        activity.activity_type,
        activity.object.name,
        activity.object.content,
    ))
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

fn slugify(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect()
}

fn unix_to_iso8601(secs: u64) -> String {
    // Minimal ISO-8601 UTC formatter without chrono dependency.
    // Format: YYYY-MM-DDTHH:MM:SSZ
    let s = secs;
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;
    // Days since Unix epoch → Gregorian date (Zeller-style).
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Convert days since Unix epoch (1970-01-01) to (year, month, day).
fn days_to_ymd(days: u64) -> (u32, u32, u32) {
    // Algorithm: civil day count conversion (Howard Hinnant's algorithm).
    let z = days as i64 + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u32, m, d)
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::symbol::SymbolKind;
    use crate::wiki::store::ArticleStore;

    fn engine() -> Engine {
        Engine::new(EngineConfig::default()).unwrap()
    }

    fn store_with_article(eng: &Engine, title: &str, body: &str) -> ArticleStore {
        let author = eng
            .create_symbol(SymbolKind::Entity, &format!("fed-author-{title}"))
            .unwrap()
            .id;
        let mut store = ArticleStore::new();
        store
            .create_article(title, body, author, &[], vec![], eng)
            .unwrap();
        store
    }

    #[test]
    fn build_create_activity_roundtrip() {
        let eng = engine();
        let store = store_with_article(&eng, "Rust Federation", "# Rust\n\nbody");
        let json = build_create_activity(&store, "Rust Federation", "act-001").unwrap();
        assert!(json.contains("Create"));
        assert!(json.contains("Rust Federation"));
        assert!(json.contains("akh-medu.dev"));
    }

    #[test]
    fn build_update_activity_test() {
        let eng = engine();
        let store = store_with_article(&eng, "Update Article", "updated body");
        let json = build_update_activity(&store, "Update Article", "act-002").unwrap();
        assert!(json.contains("Update"));
    }

    #[test]
    fn build_announce_activity_test() {
        let eng = engine();
        let store = store_with_article(&eng, "Announced Article", "body");
        let json = build_announce_activity(&store, "Announced Article", "act-003").unwrap();
        assert!(json.contains("Announce"));
    }

    #[test]
    fn missing_article_errors() {
        let store = ArticleStore::new();
        let result = build_create_activity(&store, "NonExistent", "act-004");
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(FederationError::ArticleNotFound { .. })
        ));
    }

    #[test]
    fn parse_inbound_create() {
        let eng = engine();
        let store = store_with_article(&eng, "Inbound Article", "inbound body");
        let json = build_create_activity(&store, "Inbound Article", "act-005").unwrap();
        let (activity_type, title, body) = parse_inbound_activity(&json).unwrap();
        assert_eq!(activity_type, "Create");
        assert_eq!(title, "Inbound Article");
        assert_eq!(body, "inbound body");
    }

    #[test]
    fn parse_unsupported_activity_type() {
        let payload = r#"{
            "@context": "https://www.w3.org/ns/activitystreams",
            "type": "Delete",
            "id": "https://akh-medu.dev/activities/del-001",
            "actor": "https://akh-medu.dev/actor/1",
            "object": {
                "@context": "https://www.w3.org/ns/activitystreams",
                "type": "Article",
                "id": "https://akh-medu.dev/wiki/test",
                "name": "Test",
                "content": "body",
                "attributedTo": "https://akh-medu.dev/actor/1",
                "published": "2026-01-01T00:00:00Z"
            }
        }"#;
        let result = parse_inbound_activity(payload);
        assert!(matches!(
            result,
            Err(FederationError::UnsupportedActivity { .. })
        ));
    }

    #[test]
    fn iso8601_format() {
        // Unix epoch → 1970-01-01T00:00:00Z
        assert_eq!(unix_to_iso8601(0), "1970-01-01T00:00:00Z");
        // 2025-01-01 = 20089 days after epoch = 1735689600 seconds.
        let s = unix_to_iso8601(1_735_689_600);
        assert!(s.starts_with("2025-01-01"));
        // 2026-01-01 = 20454 days after epoch = 1767225600 seconds.
        let s2 = unix_to_iso8601(1_767_225_600);
        assert!(s2.starts_with("2026-01-01"));
    }
}
