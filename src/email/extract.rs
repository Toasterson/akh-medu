//! Structured extraction from email messages (Phase 13d).
//!
//! Extracts dates, tracking numbers, URLs, phone numbers, email addresses,
//! and action items from email content. Uses a regex + grammar hybrid approach:
//!
//! - **Regex** handles structured patterns (ISO dates, tracking numbers, URLs,
//!   phone numbers, email addresses) where the format is fixed.
//! - **Grammar framework** handles natural language extraction (relative dates,
//!   action items) where multi-language support and symbol resolution matter.
//!
//! Extraction triples are stored in compartment-scoped microtheories following
//! the interlocutor pattern from Phase 12d.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::grammar::detect::detect_language;
use crate::grammar::lexer::Language;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::error::{EmailError, EmailResult};
use super::parser::ParsedEmail;

// ── Regex patterns ──────────────────────────────────────────────────────

static RE_DATE_ISO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}(?::\d{2})?)?)\b").unwrap()
});

static RE_DATE_US: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d{1,2}/\d{1,2}/\d{4})\b").unwrap()
});

static RE_DATE_EU: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d{1,2}\.\d{2}\.\d{4})\b").unwrap()
});

static RE_DATE_WRITTEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b((?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}(?:,\s*)?\d{4}|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b",
    )
    .unwrap()
});

static RE_TRACKING_UPS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(1Z[A-Z0-9]{16})\b").unwrap()
});

static RE_TRACKING_FEDEX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(\d{12,22})\b").unwrap()
});

static RE_TRACKING_USPS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(9[2-5]\d{18,20})\b").unwrap()
});

static RE_URL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"https?://[^\s<>"')\]]+"#).unwrap()
});

static RE_PHONE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:\+\d{1,3}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b").unwrap()
});

static RE_EMAIL: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b").unwrap()
});

/// FedEx context keywords used to gate false-positive FedEx tracking numbers.
const FEDEX_CONTEXT_KEYWORDS: &[&str] = &[
    "tracking", "shipment", "delivery", "fedex", "package", "shipped", "carrier",
];

/// Context window radius (chars) for FedEx context gating.
const FEDEX_CONTEXT_WINDOW: usize = 100;

// ── Types ───────────────────────────────────────────────────────────────

/// Kind of extracted information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExtractedItemKind {
    Date,
    RelativeDate,
    TrackingNumber,
    Url,
    PhoneNumber,
    ActionItem,
    EmailAddress,
}

impl std::fmt::Display for ExtractedItemKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Date => write!(f, "Date"),
            Self::RelativeDate => write!(f, "RelativeDate"),
            Self::TrackingNumber => write!(f, "TrackingNumber"),
            Self::Url => write!(f, "Url"),
            Self::PhoneNumber => write!(f, "PhoneNumber"),
            Self::ActionItem => write!(f, "ActionItem"),
            Self::EmailAddress => write!(f, "EmailAddress"),
        }
    }
}

/// Which part of the email the item was found in.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SourceField {
    Subject,
    BodyText,
}

impl std::fmt::Display for SourceField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Subject => write!(f, "Subject"),
            Self::BodyText => write!(f, "BodyText"),
        }
    }
}

/// A single extracted item from an email.
#[derive(Debug, Clone)]
pub struct ExtractedItem {
    /// What kind of item this is.
    pub kind: ExtractedItemKind,
    /// Raw text as found in the email.
    pub raw_text: String,
    /// Normalized/canonical form for deduplication and downstream use.
    pub normalized: String,
    /// KG symbol ID (assigned during `store_extractions`).
    pub symbol_id: Option<SymbolId>,
    /// Byte offset in the source field.
    pub offset: usize,
    /// Confidence (0.0–1.0).
    pub confidence: f32,
    /// Which part of the email this was found in.
    pub source_field: SourceField,
    /// Detected language (for grammar-extracted items).
    pub language: Option<Language>,
}

/// Result of running extraction on an email.
#[derive(Debug, Clone)]
pub struct ExtractionResult {
    /// All extracted items.
    pub items: Vec<ExtractedItem>,
    /// Count by kind.
    pub counts: HashMap<ExtractedItemKind, usize>,
    /// Human-readable reasoning summary.
    pub reasoning: String,
    /// Detected language of the email body.
    pub detected_language: Option<Language>,
}

// ── ExtractionPredicates ────────────────────────────────────────────────

/// Well-known KG predicates for extraction results.
#[derive(Debug, Clone)]
pub struct ExtractionPredicates {
    pub mentions_date: SymbolId,
    pub mentions_tracking: SymbolId,
    pub mentions_url: SymbolId,
    pub mentions_phone: SymbolId,
    pub has_action_item: SymbolId,
    pub mentions_email: SymbolId,
    pub value: SymbolId,
    pub carrier: SymbolId,
}

impl ExtractionPredicates {
    /// Resolve or create all extraction predicates in the engine.
    pub fn init(engine: &Engine) -> EmailResult<Self> {
        Ok(Self {
            mentions_date: engine
                .resolve_or_create_relation("extract:mentions-date")
                .map_err(EmailError::from)?,
            mentions_tracking: engine
                .resolve_or_create_relation("extract:mentions-tracking")
                .map_err(EmailError::from)?,
            mentions_url: engine
                .resolve_or_create_relation("extract:mentions-url")
                .map_err(EmailError::from)?,
            mentions_phone: engine
                .resolve_or_create_relation("extract:mentions-phone")
                .map_err(EmailError::from)?,
            has_action_item: engine
                .resolve_or_create_relation("extract:has-action-item")
                .map_err(EmailError::from)?,
            mentions_email: engine
                .resolve_or_create_relation("extract:mentions-email")
                .map_err(EmailError::from)?,
            value: engine
                .resolve_or_create_relation("extract:value")
                .map_err(EmailError::from)?,
            carrier: engine
                .resolve_or_create_relation("extract:carrier")
                .map_err(EmailError::from)?,
        })
    }
}

// ── ExtractionScope ─────────────────────────────────────────────────────

/// Compartment scope for storing extraction triples.
#[derive(Debug, Clone)]
pub struct ExtractionScope {
    /// Account-level microtheory label: `"mt:email:account:{email}"`.
    pub account_compartment: String,
    /// Correspondent-level microtheory label: `"mt:email:correspondent:{sender}"`.
    pub correspondent_compartment: String,
}

/// Create or resolve account + correspondent microtheories for extraction scoping.
///
/// Account MT is the parent; correspondent MT specializes it.
pub fn ensure_extraction_scope(
    engine: &Engine,
    account_email: &str,
    sender_email: &str,
) -> EmailResult<ExtractionScope> {
    let account_label = format!("mt:email:account:{account_email}");
    let correspondent_label = format!("mt:email:correspondent:{sender_email}");

    // Create account microtheory (idempotent — resolve_or_create).
    let account_mt = engine
        .create_context(
            &account_label,
            crate::compartment::ContextDomain::Belief,
            &[],
        )
        .map_err(EmailError::from)?;

    // Create correspondent microtheory that specializes account.
    let _correspondent_mt = engine
        .create_context(
            &correspondent_label,
            crate::compartment::ContextDomain::Belief,
            &[account_mt.id],
        )
        .map_err(EmailError::from)?;

    Ok(ExtractionScope {
        account_compartment: account_label,
        correspondent_compartment: correspondent_label,
    })
}

// ── ActionItemGoalSpec ──────────────────────────────────────────────────

/// Specification for creating a goal from an extracted action item.
///
/// Does NOT create the goal — the caller decides whether to act on it.
#[derive(Debug, Clone)]
pub struct ActionItemGoalSpec {
    /// Goal description: `"[Action from {sender}] {action_text}"`.
    pub description: String,
    /// Success criteria: `"{action_text} completed"`.
    pub criteria: String,
    /// Priority (5 = medium, 7 = urgent).
    pub priority: u8,
    /// Source email message ID.
    pub email_message_id: String,
    /// Source email KG symbol.
    pub email_symbol: SymbolId,
    /// Raw action text.
    pub action_text: String,
}

/// Urgency keywords (multi-language) that boost action item priority.
const URGENCY_KEYWORDS: &[&str] = &[
    "urgent",
    "asap",
    "immediately",
    "critical",
    "срочно",
    "немедленно",
    "urgente",
    "inmediatamente",
    "عاجل",
    "فوري",
    "urgent",     // French same as English
    "immédiat",
    "immédiatement",
];

// ── Temporal keyword maps ───────────────────────────────────────────────

struct TemporalPattern {
    keyword: &'static str,
    canonical: &'static str,
    language: Language,
}

const TEMPORAL_PATTERNS: &[TemporalPattern] = &[
    // English
    TemporalPattern { keyword: "tomorrow", canonical: "temporal:tomorrow", language: Language::English },
    TemporalPattern { keyword: "today", canonical: "temporal:today", language: Language::English },
    TemporalPattern { keyword: "tonight", canonical: "temporal:tonight", language: Language::English },
    TemporalPattern { keyword: "this weekend", canonical: "temporal:this-weekend", language: Language::English },
    TemporalPattern { keyword: "next week", canonical: "temporal:next-week", language: Language::English },
    TemporalPattern { keyword: "next monday", canonical: "temporal:next-monday", language: Language::English },
    TemporalPattern { keyword: "next tuesday", canonical: "temporal:next-tuesday", language: Language::English },
    TemporalPattern { keyword: "next wednesday", canonical: "temporal:next-wednesday", language: Language::English },
    TemporalPattern { keyword: "next thursday", canonical: "temporal:next-thursday", language: Language::English },
    TemporalPattern { keyword: "next friday", canonical: "temporal:next-friday", language: Language::English },
    TemporalPattern { keyword: "next saturday", canonical: "temporal:next-saturday", language: Language::English },
    TemporalPattern { keyword: "next sunday", canonical: "temporal:next-sunday", language: Language::English },
    TemporalPattern { keyword: "next month", canonical: "temporal:next-month", language: Language::English },
    // Russian
    TemporalPattern { keyword: "завтра", canonical: "temporal:tomorrow", language: Language::Russian },
    TemporalPattern { keyword: "послезавтра", canonical: "temporal:day-after-tomorrow", language: Language::Russian },
    TemporalPattern { keyword: "сегодня", canonical: "temporal:today", language: Language::Russian },
    TemporalPattern { keyword: "на следующей неделе", canonical: "temporal:next-week", language: Language::Russian },
    // French
    TemporalPattern { keyword: "demain", canonical: "temporal:tomorrow", language: Language::French },
    TemporalPattern { keyword: "après-demain", canonical: "temporal:day-after-tomorrow", language: Language::French },
    TemporalPattern { keyword: "aujourd'hui", canonical: "temporal:today", language: Language::French },
    TemporalPattern { keyword: "la semaine prochaine", canonical: "temporal:next-week", language: Language::French },
    // Spanish
    TemporalPattern { keyword: "mañana", canonical: "temporal:tomorrow", language: Language::Spanish },
    TemporalPattern { keyword: "pasado mañana", canonical: "temporal:day-after-tomorrow", language: Language::Spanish },
    TemporalPattern { keyword: "hoy", canonical: "temporal:today", language: Language::Spanish },
    TemporalPattern { keyword: "la próxima semana", canonical: "temporal:next-week", language: Language::Spanish },
    // Arabic
    TemporalPattern { keyword: "غداً", canonical: "temporal:tomorrow", language: Language::Arabic },
    TemporalPattern { keyword: "غدا", canonical: "temporal:tomorrow", language: Language::Arabic },
    TemporalPattern { keyword: "بعد غد", canonical: "temporal:day-after-tomorrow", language: Language::Arabic },
    TemporalPattern { keyword: "اليوم", canonical: "temporal:today", language: Language::Arabic },
    TemporalPattern { keyword: "الأسبوع القادم", canonical: "temporal:next-week", language: Language::Arabic },
];

// ── "in N days/weeks" regex patterns ────────────────────────────────────

static RE_IN_N_DAYS_EN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bin\s+(\d{1,3})\s+(days?|weeks?)\b").unwrap()
});

static RE_IN_N_DAYS_FR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bdans\s+(\d{1,3})\s+(jours?|semaines?)\b").unwrap()
});

static RE_IN_N_DAYS_ES: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\ben\s+(\d{1,3})\s+(días?|semanas?)\b").unwrap()
});

static RE_IN_N_DAYS_RU: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bчерез\s+(\d{1,3})\s+(дн(?:ей|я)|недел[ьюи])\b").unwrap()
});

// ── Action item patterns ────────────────────────────────────────────────

struct ActionPattern {
    regex: &'static LazyLock<Regex>,
    language: Language,
    base_confidence: f32,
}

static RE_ACTION_EN_PLEASE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bplease\s+(.{5,80})").unwrap()
});

static RE_ACTION_EN_COULD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bcould you\s+(.{5,80})").unwrap()
});

static RE_ACTION_EN_NEED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bneed to\s+(.{5,80})").unwrap()
});

static RE_ACTION_EN_TODO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bTODO:?\s*(.{3,80})").unwrap()
});

static RE_ACTION_FR_SVP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bs'il vous plaît\s+(.{5,80})").unwrap()
});

static RE_ACTION_FR_VEUILLEZ: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bveuillez\s+(.{5,80})").unwrap()
});

static RE_ACTION_ES_POR_FAVOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bpor favor\s+(.{5,80})").unwrap()
});

static RE_ACTION_ES_NECESITO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bnecesito\s+(.{5,80})").unwrap()
});

static RE_ACTION_RU_POZH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bпожалуйста\s+(.{5,80})").unwrap()
});

static RE_ACTION_RU_NUZHNO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\bнужно\s+(.{5,80})").unwrap()
});

static RE_ACTION_AR_PLEASE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:من فضلك|يرجى)\s+(.{5,80})").unwrap()
});

static RE_ACTION_DEADLINE_EN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\b(?:deadline|due|by)\s+(.{5,60})").unwrap()
});

const ACTION_PATTERNS: &[ActionPattern] = &[
    ActionPattern { regex: &RE_ACTION_EN_PLEASE, language: Language::English, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_EN_COULD, language: Language::English, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_EN_NEED, language: Language::English, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_EN_TODO, language: Language::English, base_confidence: 0.8 },
    ActionPattern { regex: &RE_ACTION_FR_SVP, language: Language::French, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_FR_VEUILLEZ, language: Language::French, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_ES_POR_FAVOR, language: Language::Spanish, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_ES_NECESITO, language: Language::Spanish, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_RU_POZH, language: Language::Russian, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_RU_NUZHNO, language: Language::Russian, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_AR_PLEASE, language: Language::Arabic, base_confidence: 0.7 },
    ActionPattern { regex: &RE_ACTION_DEADLINE_EN, language: Language::English, base_confidence: 0.6 },
];

// ── Regex-based extractors ──────────────────────────────────────────────

fn extract_dates_regex(text: &str, source: SourceField) -> Vec<ExtractedItem> {
    let mut items = Vec::new();

    for m in RE_DATE_ISO.find_iter(text) {
        items.push(ExtractedItem {
            kind: ExtractedItemKind::Date,
            raw_text: m.as_str().to_string(),
            normalized: m.as_str().to_string(),
            symbol_id: None,
            offset: m.start(),
            confidence: 0.95,
            source_field: source.clone(),
            language: None,
        });
    }

    for m in RE_DATE_US.find_iter(text) {
        items.push(ExtractedItem {
            kind: ExtractedItemKind::Date,
            raw_text: m.as_str().to_string(),
            normalized: m.as_str().to_string(),
            symbol_id: None,
            offset: m.start(),
            confidence: 0.9,
            source_field: source.clone(),
            language: None,
        });
    }

    for m in RE_DATE_EU.find_iter(text) {
        items.push(ExtractedItem {
            kind: ExtractedItemKind::Date,
            raw_text: m.as_str().to_string(),
            normalized: m.as_str().to_string(),
            symbol_id: None,
            offset: m.start(),
            confidence: 0.9,
            source_field: source.clone(),
            language: None,
        });
    }

    for m in RE_DATE_WRITTEN.find_iter(text) {
        items.push(ExtractedItem {
            kind: ExtractedItemKind::Date,
            raw_text: m.as_str().to_string(),
            normalized: m.as_str().to_string(),
            symbol_id: None,
            offset: m.start(),
            confidence: 0.85,
            source_field: source.clone(),
            language: None,
        });
    }

    items
}

fn extract_tracking_regex(text: &str, source: SourceField) -> Vec<ExtractedItem> {
    let mut items = Vec::new();
    let lower = text.to_lowercase();

    // UPS: 1Z + 16 alphanumeric
    for m in RE_TRACKING_UPS.find_iter(text) {
        items.push(ExtractedItem {
            kind: ExtractedItemKind::TrackingNumber,
            raw_text: m.as_str().to_string(),
            normalized: format!("tracking:ups:{}", m.as_str()),
            symbol_id: None,
            offset: m.start(),
            confidence: 0.95,
            source_field: source.clone(),
            language: None,
        });
    }

    // USPS: 9[2-5] + 18-20 digits
    for m in RE_TRACKING_USPS.find_iter(text) {
        items.push(ExtractedItem {
            kind: ExtractedItemKind::TrackingNumber,
            raw_text: m.as_str().to_string(),
            normalized: format!("tracking:usps:{}", m.as_str()),
            symbol_id: None,
            offset: m.start(),
            confidence: 0.9,
            source_field: source.clone(),
            language: None,
        });
    }

    // FedEx: 12-22 digits with context gating
    for m in RE_TRACKING_FEDEX.find_iter(text) {
        let matched = m.as_str();
        let len = matched.len();
        // Skip if it's a USPS match (starts with 9[2-5]) — already captured above
        if len >= 20 && matched.starts_with('9') {
            let second = matched.as_bytes().get(1).copied().unwrap_or(b'0');
            if (b'2'..=b'5').contains(&second) {
                continue;
            }
        }

        // Context gating: check for shipping keywords nearby
        let start = m.start().saturating_sub(FEDEX_CONTEXT_WINDOW);
        let end = (m.end() + FEDEX_CONTEXT_WINDOW).min(lower.len());
        // Ensure we don't split in the middle of a char
        let start = lower.floor_char_boundary(start);
        let end = lower.floor_char_boundary(end);
        let window = &lower[start..end];

        let has_context = FEDEX_CONTEXT_KEYWORDS
            .iter()
            .any(|kw| window.contains(kw));

        let confidence = if has_context { 0.9 } else { 0.5 };

        // Only include if confidence is reasonable
        if confidence >= 0.5 && len >= 12 {
            items.push(ExtractedItem {
                kind: ExtractedItemKind::TrackingNumber,
                raw_text: matched.to_string(),
                normalized: format!("tracking:fedex:{matched}"),
                symbol_id: None,
                offset: m.start(),
                confidence,
                source_field: source.clone(),
                language: None,
            });
        }
    }

    items
}

fn extract_urls_regex(text: &str, source: SourceField) -> Vec<ExtractedItem> {
    RE_URL
        .find_iter(text)
        .map(|m| ExtractedItem {
            kind: ExtractedItemKind::Url,
            raw_text: m.as_str().to_string(),
            normalized: m.as_str().to_string(),
            symbol_id: None,
            offset: m.start(),
            confidence: 0.95,
            source_field: source.clone(),
            language: None,
        })
        .collect()
}

fn extract_phones_regex(text: &str, source: SourceField) -> Vec<ExtractedItem> {
    RE_PHONE
        .find_iter(text)
        .map(|m| {
            // Normalize: strip non-digits except leading +
            let raw = m.as_str();
            let normalized: String = raw
                .chars()
                .filter(|c| c.is_ascii_digit() || *c == '+')
                .collect();
            ExtractedItem {
                kind: ExtractedItemKind::PhoneNumber,
                raw_text: raw.to_string(),
                normalized,
                symbol_id: None,
                offset: m.start(),
                confidence: 0.85,
                source_field: source.clone(),
                language: None,
            }
        })
        .collect()
}

fn extract_emails_regex(text: &str, source: SourceField) -> Vec<ExtractedItem> {
    RE_EMAIL
        .find_iter(text)
        .map(|m| ExtractedItem {
            kind: ExtractedItemKind::EmailAddress,
            raw_text: m.as_str().to_string(),
            normalized: m.as_str().to_lowercase(),
            symbol_id: None,
            offset: m.start(),
            confidence: 0.95,
            source_field: source.clone(),
            language: None,
        })
        .collect()
}

// ── Grammar-based extractors ────────────────────────────────────────────

/// Extract relative date mentions via multi-language temporal keyword matching.
///
/// Uses `grammar::detect::detect_language()` for language identification, then
/// language-specific temporal keyword maps. Canonical forms are language-independent.
pub fn extract_temporal_via_grammar(text: &str, source: SourceField) -> Vec<ExtractedItem> {
    let detection = detect_language(text);
    let detected_lang = detection.language;

    let lower = text.to_lowercase();
    let mut items = Vec::new();

    // Match temporal keyword patterns
    for pattern in TEMPORAL_PATTERNS {
        // Accept patterns that match the detected language, or always try
        // language-specific matches (a Russian word in English text still valid)
        if let Some(offset) = lower.find(pattern.keyword) {
            items.push(ExtractedItem {
                kind: ExtractedItemKind::RelativeDate,
                raw_text: text[offset..offset + pattern.keyword.len()].to_string(),
                normalized: pattern.canonical.to_string(),
                symbol_id: None,
                offset,
                confidence: 0.8,
                source_field: source.clone(),
                language: Some(pattern.language),
            });
        }
    }

    // "in N days/weeks" patterns per language
    let in_n_regexes: &[(&LazyLock<Regex>, Language)] = &[
        (&RE_IN_N_DAYS_EN, Language::English),
        (&RE_IN_N_DAYS_FR, Language::French),
        (&RE_IN_N_DAYS_ES, Language::Spanish),
        (&RE_IN_N_DAYS_RU, Language::Russian),
    ];

    for (re, lang) in in_n_regexes {
        for caps in re.captures_iter(text) {
            if let (Some(n_match), Some(unit_match)) = (caps.get(1), caps.get(2)) {
                let n = n_match.as_str();
                let unit_str = unit_match.as_str().to_lowercase();
                let unit = if unit_str.starts_with('d')
                    || unit_str.starts_with("día")
                    || unit_str.starts_with("jour")
                    || unit_str.starts_with("дн")
                {
                    "days"
                } else {
                    "weeks"
                };
                let canonical = format!("temporal:in-{n}-{unit}");
                let full = caps.get(0).unwrap();
                items.push(ExtractedItem {
                    kind: ExtractedItemKind::RelativeDate,
                    raw_text: full.as_str().to_string(),
                    normalized: canonical,
                    symbol_id: None,
                    offset: full.start(),
                    confidence: 0.8,
                    source_field: source.clone(),
                    language: Some(*lang),
                });
            }
        }
    }

    // Set language to detected if not language-specific
    for item in &mut items {
        if item.language.is_none() {
            item.language = Some(detected_lang);
        }
    }

    items
}

/// Extract action items via multi-language pattern matching.
///
/// Uses language detection + per-language action verb patterns.
/// Confidence is boosted by urgency keywords.
pub fn extract_actions_via_grammar(text: &str, source: SourceField) -> Vec<ExtractedItem> {
    let detection = detect_language(text);
    let detected_lang = detection.language;

    let lower = text.to_lowercase();
    let mut items = Vec::new();

    for pattern in ACTION_PATTERNS {
        for caps in pattern.regex.captures_iter(text) {
            if let Some(action_match) = caps.get(1) {
                let action_text = action_match.as_str().trim();
                // Trim trailing punctuation from action text
                let action_text = action_text
                    .trim_end_matches('.')
                    .trim_end_matches(',')
                    .trim_end_matches('!')
                    .trim();

                if action_text.len() < 3 {
                    continue;
                }

                // Boost confidence if urgency keywords present
                let has_urgency = URGENCY_KEYWORDS.iter().any(|kw| lower.contains(kw));
                let confidence = if has_urgency {
                    (pattern.base_confidence + 0.1).min(0.9)
                } else {
                    pattern.base_confidence
                };

                let full = caps.get(0).unwrap();
                items.push(ExtractedItem {
                    kind: ExtractedItemKind::ActionItem,
                    raw_text: full.as_str().to_string(),
                    normalized: action_text.to_string(),
                    symbol_id: None,
                    offset: full.start(),
                    confidence,
                    source_field: source.clone(),
                    language: Some(if pattern.language == Language::English {
                        detected_lang
                    } else {
                        pattern.language
                    }),
                });
            }
        }
    }

    items
}

// ── Main extraction pipeline ────────────────────────────────────────────

/// Extract all structured information from an email.
///
/// Pipeline:
/// 1. Detect language via `grammar::detect::detect_language()`
/// 2. Run regex extractors (dates, tracking, URLs, phones, emails) on subject + body
/// 3. Run `extract_temporal_via_grammar` on subject + body
/// 4. Run `extract_actions_via_grammar` on subject + body
/// 5. Deduplicate by `(kind, normalized)` — keep highest confidence
/// 6. Build counts HashMap
/// 7. Generate reasoning summary
pub fn extract_all(email: &ParsedEmail) -> ExtractionResult {
    let body = email.best_body().unwrap_or("");
    let subject = &email.subject;

    // Detect language from body
    let detection = if !body.is_empty() {
        Some(detect_language(body))
    } else if !subject.is_empty() {
        Some(detect_language(subject))
    } else {
        None
    };
    let detected_language = detection.as_ref().map(|d| d.language);

    let mut all_items: Vec<ExtractedItem> = Vec::new();

    // Regex extractors on subject
    all_items.extend(extract_dates_regex(subject, SourceField::Subject));
    all_items.extend(extract_tracking_regex(subject, SourceField::Subject));
    all_items.extend(extract_urls_regex(subject, SourceField::Subject));
    all_items.extend(extract_phones_regex(subject, SourceField::Subject));
    all_items.extend(extract_emails_regex(subject, SourceField::Subject));

    // Regex extractors on body
    all_items.extend(extract_dates_regex(body, SourceField::BodyText));
    all_items.extend(extract_tracking_regex(body, SourceField::BodyText));
    all_items.extend(extract_urls_regex(body, SourceField::BodyText));
    all_items.extend(extract_phones_regex(body, SourceField::BodyText));
    all_items.extend(extract_emails_regex(body, SourceField::BodyText));

    // Grammar-based temporal extraction on subject + body
    all_items.extend(extract_temporal_via_grammar(subject, SourceField::Subject));
    all_items.extend(extract_temporal_via_grammar(body, SourceField::BodyText));

    // Grammar-based action item extraction on subject + body
    all_items.extend(extract_actions_via_grammar(subject, SourceField::Subject));
    all_items.extend(extract_actions_via_grammar(body, SourceField::BodyText));

    // Deduplicate by (kind, normalized) — keep highest confidence
    let mut dedup: HashMap<(ExtractedItemKind, String), ExtractedItem> = HashMap::new();
    for item in all_items {
        let key = (item.kind.clone(), item.normalized.clone());
        match dedup.get(&key) {
            Some(existing) if existing.confidence >= item.confidence => {}
            _ => {
                dedup.insert(key, item);
            }
        }
    }

    let items: Vec<ExtractedItem> = dedup.into_values().collect();

    // Build counts
    let mut counts: HashMap<ExtractedItemKind, usize> = HashMap::new();
    for item in &items {
        *counts.entry(item.kind.clone()).or_insert(0) += 1;
    }

    // Reasoning summary
    let kinds_summary: Vec<String> = counts
        .iter()
        .map(|(k, v)| format!("{k}: {v}"))
        .collect();
    let lang_str = detected_language
        .map(|l| l.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let reasoning = format!(
        "extracted {} items from email [{}]; language: {lang_str}",
        items.len(),
        kinds_summary.join(", "),
    );

    ExtractionResult {
        items,
        counts,
        reasoning,
        detected_language,
    }
}

// ── KG storage ──────────────────────────────────────────────────────────

/// Store extraction results as compartment-scoped KG triples.
///
/// For each `ExtractedItem`:
/// - Creates an entity symbol via `engine.resolve_or_create_entity()`
/// - Creates a triple `(email_symbol, extract:mentions-X, item_symbol)`
/// - Applies compartment scoping from `scope`
/// - For tracking numbers, adds a carrier triple
///
/// Returns the list of created symbol IDs.
pub fn store_extractions(
    engine: &Engine,
    email_symbol: SymbolId,
    result: &mut ExtractionResult,
    predicates: &ExtractionPredicates,
    scope: &ExtractionScope,
) -> EmailResult<Vec<SymbolId>> {
    let mut created_symbols = Vec::new();

    for item in &mut result.items {
        // Create entity for the extracted item
        let item_sym = engine
            .resolve_or_create_entity(&item.normalized)
            .map_err(EmailError::from)?;
        item.symbol_id = Some(item_sym);
        created_symbols.push(item_sym);

        // Choose predicate based on kind
        let predicate = match item.kind {
            ExtractedItemKind::Date | ExtractedItemKind::RelativeDate => predicates.mentions_date,
            ExtractedItemKind::TrackingNumber => predicates.mentions_tracking,
            ExtractedItemKind::Url => predicates.mentions_url,
            ExtractedItemKind::PhoneNumber => predicates.mentions_phone,
            ExtractedItemKind::ActionItem => predicates.has_action_item,
            ExtractedItemKind::EmailAddress => predicates.mentions_email,
        };

        // Create compartment-scoped triple
        let triple = Triple::new(email_symbol, predicate, item_sym)
            .with_compartment(scope.correspondent_compartment.clone());
        engine.add_triple(&triple).map_err(EmailError::from)?;

        // For tracking numbers, add carrier triple
        if item.kind == ExtractedItemKind::TrackingNumber {
            let carrier_name = if item.normalized.starts_with("tracking:ups:") {
                "UPS"
            } else if item.normalized.starts_with("tracking:usps:") {
                "USPS"
            } else if item.normalized.starts_with("tracking:fedex:") {
                "FedEx"
            } else {
                "Unknown"
            };
            let carrier_sym = engine
                .resolve_or_create_entity(carrier_name)
                .map_err(EmailError::from)?;
            let carrier_triple = Triple::new(item_sym, predicates.carrier, carrier_sym)
                .with_compartment(scope.correspondent_compartment.clone());
            engine
                .add_triple(&carrier_triple)
                .map_err(EmailError::from)?;
        }
    }

    Ok(created_symbols)
}

// ── Provenance ──────────────────────────────────────────────────────────

/// Record provenance for email extraction.
pub fn record_extraction_provenance(
    engine: &Engine,
    email_symbol: SymbolId,
    result: &ExtractionResult,
    email: &ParsedEmail,
) -> EmailResult<()> {
    let kinds_found: Vec<String> = result.counts.keys().map(|k| k.to_string()).collect();
    let mut record = ProvenanceRecord::new(
        email_symbol,
        DerivationKind::EmailExtracted {
            email_message_id: email.message_id.clone(),
            item_count: result.items.len(),
            kinds_found: kinds_found.join(", "),
        },
    );
    record.confidence = 0.9;

    engine
        .store_provenance(&mut record)
        .map_err(|e| EmailError::Engine(Box::new(e)))?;
    Ok(())
}

// ── Goal spec generation ────────────────────────────────────────────────

/// Convert extracted action items into goal specifications.
///
/// Does NOT create goals — returns specs for the caller to decide.
pub fn action_items_to_goals(
    items: &[ExtractedItem],
    email: &ParsedEmail,
    email_symbol: SymbolId,
) -> Vec<ActionItemGoalSpec> {
    items
        .iter()
        .filter(|i| i.kind == ExtractedItemKind::ActionItem)
        .map(|item| {
            let action_text = &item.normalized;
            let lower = action_text.to_lowercase();

            // Check for urgency keywords (multi-language)
            let has_urgency = URGENCY_KEYWORDS.iter().any(|kw| lower.contains(kw));
            let priority = if has_urgency { 7 } else { 5 };

            ActionItemGoalSpec {
                description: format!("[Action from {}] {action_text}", email.from),
                criteria: format!("{action_text} completed"),
                priority,
                email_message_id: email.message_id.clone(),
                email_symbol,
                action_text: action_text.clone(),
            }
        })
        .collect()
}

// ── Quick predicates ────────────────────────────────────────────────────

/// Quick check: does the extraction result contain action items?
pub fn has_action_items(result: &ExtractionResult) -> bool {
    result
        .counts
        .get(&ExtractedItemKind::ActionItem)
        .is_some_and(|&c| c > 0)
}

/// Quick check: does the extraction result contain date or relative-date mentions?
pub fn has_calendar_event(result: &ExtractionResult) -> bool {
    result
        .counts
        .get(&ExtractedItemKind::Date)
        .is_some_and(|&c| c > 0)
        || result
            .counts
            .get(&ExtractedItemKind::RelativeDate)
            .is_some_and(|&c| c > 0)
}

/// Quick check: does the extraction result contain tracking numbers?
pub fn has_shipment_info(result: &ExtractionResult) -> bool {
    result
        .counts
        .get(&ExtractedItemKind::TrackingNumber)
        .is_some_and(|&c| c > 0)
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};

    fn test_engine() -> Engine {
        Engine::new(EngineConfig::default()).unwrap()
    }

    fn make_email(subject: &str, body: &str) -> ParsedEmail {
        ParsedEmail {
            message_id: "<test-001@example.com>".to_string(),
            from: "sender@example.com".to_string(),
            from_display: Some("Test Sender".to_string()),
            to: vec!["me@example.com".to_string()],
            cc: Vec::new(),
            subject: subject.to_string(),
            date: Some(1700000000),
            in_reply_to: None,
            references: Vec::new(),
            body_text: Some(body.to_string()),
            body_html: None,
            has_attachments: false,
            list_id: None,
            content_type: "text/plain".to_string(),
        }
    }

    // ── Date extraction ─────────────────────────────────────────────

    #[test]
    fn extract_iso_date() {
        let items = extract_dates_regex("Meeting on 2026-02-21 at noon", SourceField::BodyText);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].normalized, "2026-02-21");
        assert_eq!(items[0].confidence, 0.95);
    }

    #[test]
    fn extract_iso_datetime() {
        let items = extract_dates_regex("Event at 2026-02-21T14:30", SourceField::BodyText);
        assert_eq!(items.len(), 1);
        assert!(items[0].normalized.contains("T14:30"));
    }

    #[test]
    fn extract_written_date() {
        let items = extract_dates_regex("Due by February 21, 2026", SourceField::BodyText);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, ExtractedItemKind::Date);
        assert_eq!(items[0].confidence, 0.85);
    }

    #[test]
    fn extract_us_date() {
        let items = extract_dates_regex("Received 02/21/2026", SourceField::BodyText);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].normalized, "02/21/2026");
    }

    #[test]
    fn extract_eu_date() {
        let items = extract_dates_regex("Datum: 21.02.2026", SourceField::BodyText);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].normalized, "21.02.2026");
    }

    // ── Relative date extraction ────────────────────────────────────

    #[test]
    fn extract_relative_date_english() {
        let items = extract_temporal_via_grammar(
            "Let's meet tomorrow at noon",
            SourceField::BodyText,
        );
        assert!(items.iter().any(|i| i.normalized == "temporal:tomorrow"));
    }

    #[test]
    fn extract_relative_date_russian() {
        let items = extract_temporal_via_grammar(
            "Встретимся завтра в офисе",
            SourceField::BodyText,
        );
        assert!(items.iter().any(|i| i.normalized == "temporal:tomorrow"));
        assert!(items.iter().any(|i| i.language == Some(Language::Russian)));
    }

    #[test]
    fn extract_relative_date_french() {
        let items = extract_temporal_via_grammar(
            "Le rendez-vous est demain",
            SourceField::BodyText,
        );
        assert!(items.iter().any(|i| i.normalized == "temporal:tomorrow"));
        assert!(items.iter().any(|i| i.language == Some(Language::French)));
    }

    #[test]
    fn extract_relative_date_spanish() {
        let items = extract_temporal_via_grammar(
            "La reunión es mañana por la mañana",
            SourceField::BodyText,
        );
        assert!(items.iter().any(|i| i.normalized == "temporal:tomorrow"));
    }

    #[test]
    fn extract_relative_date_arabic() {
        let items = extract_temporal_via_grammar(
            "الاجتماع غداً في المكتب",
            SourceField::BodyText,
        );
        assert!(items.iter().any(|i| i.normalized == "temporal:tomorrow"));
    }

    #[test]
    fn extract_in_n_days() {
        let items = extract_temporal_via_grammar(
            "The report is due in 3 days",
            SourceField::BodyText,
        );
        assert!(items
            .iter()
            .any(|i| i.normalized == "temporal:in-3-days"));
    }

    // ── Tracking number extraction ──────────────────────────────────

    #[test]
    fn extract_tracking_ups() {
        let items = extract_tracking_regex(
            "Your UPS tracking: 1Z999AA10123456784",
            SourceField::BodyText,
        );
        assert_eq!(items.len(), 1);
        assert!(items[0].normalized.starts_with("tracking:ups:"));
        assert_eq!(items[0].confidence, 0.95);
    }

    #[test]
    fn extract_tracking_usps() {
        let items = extract_tracking_regex(
            "USPS tracking number: 92748901234567890123",
            SourceField::BodyText,
        );
        assert!(items.iter().any(|i| i.normalized.starts_with("tracking:usps:")));
    }

    #[test]
    fn extract_tracking_fedex_with_context() {
        let items = extract_tracking_regex(
            "Your FedEx shipment tracking number is 123456789012",
            SourceField::BodyText,
        );
        let fedex_items: Vec<_> = items
            .iter()
            .filter(|i| i.normalized.starts_with("tracking:fedex:"))
            .collect();
        assert!(!fedex_items.is_empty());
        // With context keywords, confidence should be 0.9
        assert!(fedex_items[0].confidence >= 0.9);
    }

    #[test]
    fn no_false_positive_numbers() {
        // A 12-digit number without shipping context should have low confidence
        let items = extract_tracking_regex(
            "The account balance is 123456789012",
            SourceField::BodyText,
        );
        let fedex_items: Vec<_> = items
            .iter()
            .filter(|i| i.normalized.starts_with("tracking:fedex:"))
            .collect();
        if !fedex_items.is_empty() {
            assert!(
                fedex_items[0].confidence <= 0.5,
                "without context, FedEx confidence should be low: {}",
                fedex_items[0].confidence
            );
        }
    }

    // ── URL extraction ──────────────────────────────────────────────

    #[test]
    fn extract_urls() {
        let items = extract_urls_regex(
            "Visit https://example.com/page and http://test.org",
            SourceField::BodyText,
        );
        assert_eq!(items.len(), 2);
        assert!(items.iter().all(|i| i.kind == ExtractedItemKind::Url));
    }

    // ── Phone extraction ────────────────────────────────────────────

    #[test]
    fn extract_phone_us() {
        let items = extract_phones_regex(
            "Call us at (555) 123-4567",
            SourceField::BodyText,
        );
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].kind, ExtractedItemKind::PhoneNumber);
        assert_eq!(items[0].normalized, "5551234567");
    }

    #[test]
    fn extract_phone_international() {
        let items = extract_phones_regex(
            "Reach me at +1 555-123-4567",
            SourceField::BodyText,
        );
        assert_eq!(items.len(), 1);
        assert!(items[0].normalized.starts_with("+1"));
    }

    // ── Email address extraction ────────────────────────────────────

    #[test]
    fn extract_email_addresses() {
        let items = extract_emails_regex(
            "Contact alice@example.com or bob@test.org",
            SourceField::BodyText,
        );
        assert_eq!(items.len(), 2);
        assert!(items
            .iter()
            .any(|i| i.normalized == "alice@example.com"));
    }

    // ── Action item extraction ──────────────────────────────────────

    #[test]
    fn extract_action_items_english() {
        let items = extract_actions_via_grammar(
            "Please send the report by Friday",
            SourceField::BodyText,
        );
        assert!(!items.is_empty());
        assert_eq!(items[0].kind, ExtractedItemKind::ActionItem);
    }

    #[test]
    fn extract_action_items_french() {
        let items = extract_actions_via_grammar(
            "Veuillez envoyer le rapport vendredi",
            SourceField::BodyText,
        );
        assert!(!items.is_empty());
        assert!(items.iter().any(|i| i.language == Some(Language::French)));
    }

    #[test]
    fn extract_action_items_deadline() {
        let items = extract_actions_via_grammar(
            "The deadline is next Friday",
            SourceField::BodyText,
        );
        assert!(!items.is_empty());
    }

    #[test]
    fn extract_action_items_urgency_boost() {
        let items = extract_actions_via_grammar(
            "Please send the report urgently",
            SourceField::BodyText,
        );
        assert!(!items.is_empty());
        // Urgency should boost confidence above base 0.7
        assert!(items[0].confidence > 0.7);
    }

    // ── Full pipeline ───────────────────────────────────────────────

    #[test]
    fn extract_all_from_email() {
        let email = make_email(
            "Meeting on 2026-02-21",
            "Hi, please review the document at https://example.com/doc. \
             Your tracking number is 1Z999AA10123456784. \
             Call me at (555) 123-4567. Let's discuss tomorrow.",
        );
        let result = extract_all(&email);

        assert!(result.items.len() >= 4, "expected >=4 items, got {}", result.items.len());
        assert!(result.counts.contains_key(&ExtractedItemKind::Date));
        assert!(result.counts.contains_key(&ExtractedItemKind::Url));
        assert!(result.counts.contains_key(&ExtractedItemKind::TrackingNumber));
        assert!(result.counts.contains_key(&ExtractedItemKind::PhoneNumber));
    }

    #[test]
    fn extract_all_deduplicates() {
        let email = make_email(
            "Visit https://example.com",
            "Check out https://example.com for details",
        );
        let result = extract_all(&email);

        let url_count = result
            .items
            .iter()
            .filter(|i| i.kind == ExtractedItemKind::Url && i.normalized == "https://example.com")
            .count();
        assert_eq!(url_count, 1, "duplicate URL should be deduplicated");
    }

    #[test]
    fn extract_all_empty() {
        let email = make_email("Hello", "Just a plain message with nothing to extract.");
        let result = extract_all(&email);
        // May have some false positives but should be minimal
        assert!(result.items.len() <= 2);
    }

    // ── Extraction predicates ───────────────────────────────────────

    #[test]
    fn extraction_predicates_init() {
        let engine = test_engine();
        let preds = ExtractionPredicates::init(&engine).unwrap();

        let ids = [
            preds.mentions_date,
            preds.mentions_tracking,
            preds.mentions_url,
            preds.mentions_phone,
            preds.has_action_item,
            preds.mentions_email,
            preds.value,
            preds.carrier,
        ];
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 8, "all predicates should be distinct");
    }

    // ── Extraction scope ────────────────────────────────────────────

    #[test]
    fn ensure_extraction_scope_creates_mts() {
        let engine = test_engine();
        let scope = ensure_extraction_scope(&engine, "me@gmail.com", "alice@example.com").unwrap();

        assert_eq!(scope.account_compartment, "mt:email:account:me@gmail.com");
        assert_eq!(
            scope.correspondent_compartment,
            "mt:email:correspondent:alice@example.com"
        );
    }

    // ── Store extractions ───────────────────────────────────────────

    #[test]
    fn store_extractions_creates_scoped_triples() {
        let engine = test_engine();
        let predicates = ExtractionPredicates::init(&engine).unwrap();
        let scope =
            ensure_extraction_scope(&engine, "me@gmail.com", "sender@example.com").unwrap();

        let email_sym = engine.resolve_or_create_entity("test-email").unwrap();

        let mut result = ExtractionResult {
            items: vec![ExtractedItem {
                kind: ExtractedItemKind::Url,
                raw_text: "https://example.com".to_string(),
                normalized: "https://example.com".to_string(),
                symbol_id: None,
                offset: 0,
                confidence: 0.95,
                source_field: SourceField::BodyText,
                language: None,
            }],
            counts: {
                let mut m = HashMap::new();
                m.insert(ExtractedItemKind::Url, 1);
                m
            },
            reasoning: "test".to_string(),
            detected_language: None,
        };

        let created = store_extractions(&engine, email_sym, &mut result, &predicates, &scope).unwrap();
        assert_eq!(created.len(), 1);
        // Item should now have symbol_id assigned
        assert!(result.items[0].symbol_id.is_some());
    }

    // ── Provenance ──────────────────────────────────────────────────

    #[test]
    fn record_extraction_provenance_works() {
        let engine = Engine::new(EngineConfig {
            data_dir: Some(tempfile::tempdir().unwrap().into_path()),
            ..EngineConfig::default()
        })
        .unwrap();

        let email_sym = engine.resolve_or_create_entity("test-email").unwrap();
        let email = make_email("Test", "Body");

        let result = ExtractionResult {
            items: Vec::new(),
            counts: HashMap::new(),
            reasoning: "test".to_string(),
            detected_language: None,
        };

        record_extraction_provenance(&engine, email_sym, &result, &email).unwrap();
    }

    // ── Action items to goals ───────────────────────────────────────

    #[test]
    fn action_items_to_goals_basic() {
        let email = make_email("Test", "Please send the report");
        let email_sym = SymbolId::new(1).unwrap();

        let items = vec![ExtractedItem {
            kind: ExtractedItemKind::ActionItem,
            raw_text: "please send the report".to_string(),
            normalized: "send the report".to_string(),
            symbol_id: None,
            offset: 0,
            confidence: 0.7,
            source_field: SourceField::BodyText,
            language: Some(Language::English),
        }];

        let goals = action_items_to_goals(&items, &email, email_sym);
        assert_eq!(goals.len(), 1);
        assert!(goals[0].description.contains("Action from sender@example.com"));
        assert_eq!(goals[0].priority, 5); // No urgency
    }

    #[test]
    fn action_items_urgency_boost_multilang() {
        let email = make_email("Test", "Urgent: send the report");
        let email_sym = SymbolId::new(1).unwrap();

        let items = vec![ExtractedItem {
            kind: ExtractedItemKind::ActionItem,
            raw_text: "urgent: send the report".to_string(),
            normalized: "urgent: send the report".to_string(),
            symbol_id: None,
            offset: 0,
            confidence: 0.7,
            source_field: SourceField::BodyText,
            language: Some(Language::English),
        }];

        let goals = action_items_to_goals(&items, &email, email_sym);
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].priority, 7); // Urgency boosts to 7
    }

    // ── Quick predicates ────────────────────────────────────────────

    #[test]
    fn quick_predicate_checks() {
        let mut counts = HashMap::new();
        counts.insert(ExtractedItemKind::ActionItem, 2);
        counts.insert(ExtractedItemKind::Date, 1);
        counts.insert(ExtractedItemKind::TrackingNumber, 1);

        let result = ExtractionResult {
            items: Vec::new(),
            counts,
            reasoning: String::new(),
            detected_language: None,
        };

        assert!(has_action_items(&result));
        assert!(has_calendar_event(&result));
        assert!(has_shipment_info(&result));
    }

    #[test]
    fn quick_predicate_empty() {
        let result = ExtractionResult {
            items: Vec::new(),
            counts: HashMap::new(),
            reasoning: String::new(),
            detected_language: None,
        };

        assert!(!has_action_items(&result));
        assert!(!has_calendar_event(&result));
        assert!(!has_shipment_info(&result));
    }

    // ── Display traits ──────────────────────────────────────────────

    #[test]
    fn display_traits() {
        assert_eq!(ExtractedItemKind::Date.to_string(), "Date");
        assert_eq!(ExtractedItemKind::RelativeDate.to_string(), "RelativeDate");
        assert_eq!(ExtractedItemKind::TrackingNumber.to_string(), "TrackingNumber");
        assert_eq!(ExtractedItemKind::Url.to_string(), "Url");
        assert_eq!(ExtractedItemKind::PhoneNumber.to_string(), "PhoneNumber");
        assert_eq!(ExtractedItemKind::ActionItem.to_string(), "ActionItem");
        assert_eq!(ExtractedItemKind::EmailAddress.to_string(), "EmailAddress");
    }

    #[test]
    fn source_field_display() {
        assert_eq!(SourceField::Subject.to_string(), "Subject");
        assert_eq!(SourceField::BodyText.to_string(), "BodyText");
    }
}
