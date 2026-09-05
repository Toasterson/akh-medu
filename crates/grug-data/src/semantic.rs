//! Semantic gloss lexicon (Phase 51 P3d) — move the retrieval signal off the
//! tool *name*.
//!
//! The P3a corpus embeds each tool's literal `verb_noun` in every request
//! (`fetch_ledger` → "fetch the ledger") and lets the description merely echo
//! the name. That makes retrieval trivial (recall@8 = 1.000) and abstention
//! artificially crisp (0.998) — every headline carries the "surface coupling"
//! asterisk (see [`crate::retriever::SemanticRetriever`] lever-2 findings and
//! ADR 089's M1 caveat). P3d breaks the coupling with a controlled **gloss
//! lexicon**: each `verb` and `noun` in the procedural vocab gets two synonym
//! glosses — a *primary* and an *alt* — chosen so **neither shares a whole-word
//! token with the name** (enforced by test), so a glossed request/description
//! carries the tool's *meaning* without its surface form.
//!
//! Three [`SignalMode`] tiers, graded by difficulty:
//!
//! - [`SignalMode::Name`] (**D0**, control): request + description use the
//!   literal name — reproduces the current corpus exactly.
//! - [`SignalMode::Description`] (**D1**): request *and* description use the
//!   **primary** gloss. The name is absent from the request; the retrieval
//!   signal has moved to the description (which the retriever must now read, as
//!   the real system does). Still surface — but keyed on description↔request
//!   overlap, not the name.
//! - [`SignalMode::Paraphrase`] (**D2**): the **request** uses the **alt**
//!   gloss while the **description** keeps the **primary** gloss, so the two
//!   share *no* surface tokens for the verb/noun — only meaning bridges them.
//!   No surface retriever can win; a learned embedding is required (P3d-2).
//!
//! Argument *values* are never glossed — they still appear verbatim in the
//! request — so value-grounding is a control that must not move across tiers.

/// Which retrieval-signal tier a corpus is generated under (P3d). See the
/// module docs for the D0/D1/D2 mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalMode {
    /// D0 — the literal `verb_noun` name in request + description (current).
    Name,
    /// D1 — primary gloss in both request and description (signal → description).
    Description,
    /// D2 — alt gloss in the request, primary gloss in the description
    /// (surface-disjoint; only meaning bridges them).
    Paraphrase,
}

impl SignalMode {
    /// Parse a CLI label (`name` | `description` | `paraphrase`, or the short
    /// `d0` | `d1` | `d2`).
    pub fn parse_label(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "name" | "d0" => Some(SignalMode::Name),
            "description" | "desc" | "d1" => Some(SignalMode::Description),
            "paraphrase" | "para" | "d2" => Some(SignalMode::Paraphrase),
            _ => None,
        }
    }
}

/// `(verb, primary_gloss, alt_gloss)` — glosses share no whole-word token with
/// the verb, and (by test) the primary and alt are themselves surface-disjoint.
/// One row per entry in `procedural::VERBS` (coupling checked by test).
pub(crate) const VERB_GLOSS: &[(&str, &str, &str)] = &[
    ("get", "obtain", "acquire"),
    ("fetch", "retrieve", "pull up"),
    ("list", "enumerate", "show all"),
    ("create", "make", "set up"),
    ("update", "modify", "revise"),
    ("delete", "remove", "purge"),
    ("archive", "move to storage", "put into cold vault"),
    ("restore", "bring back", "recover"),
    ("search", "look for", "seek"),
    ("schedule", "book", "plan a time for"),
    ("cancel", "call off", "abort"),
    ("start", "begin", "kick off"),
    ("stop", "halt", "cease"),
    ("pause", "suspend", "hold"),
    ("resume", "continue", "carry on"),
    ("sync", "reconcile", "align"),
    ("export", "send out", "extract"),
    ("import", "bring in", "load"),
    ("validate", "verify", "confirm"),
    ("check", "inspect", "examine"),
    ("reset", "revert", "zero out"),
    ("rotate", "cycle", "turn over"),
    ("merge", "combine", "join"),
    ("split", "divide", "break apart"),
    ("clone", "duplicate", "copy"),
    ("publish", "make live", "post"),
    ("unpublish", "take down", "retract"),
    ("lock", "secure", "freeze"),
    ("unlock", "grant access", "unseal"),
    ("tag", "mark", "flag"),
    ("untag", "unmark", "clear marking"),
    ("assign", "allocate", "hand to"),
    ("unassign", "unallocate", "take away from"),
    ("notify", "alert", "inform"),
    ("escalate", "raise", "bump up"),
    ("resolve", "settle", "fix"),
    ("close", "shut", "wrap up"),
    ("reopen", "open again", "revive"),
    ("approve", "authorize", "sign off"),
    ("reject", "decline", "turn down"),
    ("queue", "line up", "add to the waitlist"),
    ("dequeue", "pull from the line", "take off the waitlist"),
    ("compress", "shrink", "pack down"),
    ("decompress", "expand", "unpack"),
    ("encrypt", "scramble", "encode"),
    ("decrypt", "unscramble", "decode"),
    ("index", "catalog", "build a lookup for"),
    ("reindex", "rebuild the catalog for", "recatalog"),
    ("compile", "build", "assemble"),
    ("deploy", "roll out", "ship"),
];

/// `(noun, primary_gloss, alt_gloss)` — one row per entry in
/// `procedural::NOUNS`. Same no-name-token + primary/alt-disjoint rules; for
/// the multi-token key `cluster_node`, *both* underscore-split tokens are kept
/// out of the glosses.
pub(crate) const NOUN_GLOSS: &[(&str, &str, &str)] = &[
    ("ledger", "accounting records", "account book"),
    ("beacon", "signal transmitter", "locator broadcast"),
    ("manifest", "contents list", "shipping inventory"),
    ("cluster", "server group", "node collection"),
    ("shard", "data partition", "database slice"),
    ("widget", "interface element", "screen control"),
    ("module", "software unit", "plugin"),
    ("profile", "account details", "user information"),
    ("session", "login period", "active connection"),
    ("invoice", "bill", "payment request"),
    ("ticket", "support case", "help request"),
    ("snapshot", "point-in-time copy", "saved state"),
    ("pipeline", "processing workflow", "build sequence"),
    ("endpoint", "api address", "service url"),
    ("artifact", "build output", "generated file"),
    ("record", "database entry", "stored row"),
    ("workspace", "project area", "team environment"),
    ("channel", "message stream", "communication line"),
    ("queue", "task line", "waiting list"),
    ("policy", "rule set", "governance directive"),
    ("template", "reusable pattern", "starter layout"),
    ("dataset", "data collection", "table of rows"),
    ("credential", "access secret", "login key"),
    ("webhook", "event callback", "http notifier"),
    ("schedule", "timetable", "calendar plan"),
    ("report", "summary document", "analysis writeup"),
    ("dashboard", "metrics overview", "control panel"),
    ("device", "hardware unit", "physical gadget"),
    ("sensor", "measurement probe", "detector unit"),
    ("package", "software bundle", "distributable kit"),
    ("release", "software version", "product rollout"),
    ("backup", "safety copy", "stored duplicate"),
    ("cache", "fast-access store", "temporary memory"),
    ("route", "network path", "traffic rule"),
    ("zone", "geographic area", "isolated segment"),
    ("region", "territory", "geographic division"),
    ("cluster_node", "worker machine", "compute instance"),
    ("token", "access key", "auth string"),
    (
        "certificate",
        "digital identity proof",
        "signed key document",
    ),
    ("subscription", "recurring plan", "membership"),
    ("playlist", "track collection", "song lineup"),
    ("thread", "message chain", "discussion series"),
    ("annotation", "note", "margin comment"),
    ("bookmark", "saved link", "favorite pointer"),
    ("label", "category marker", "classification name"),
    ("tag", "keyword marker", "sortable descriptor"),
    ("attachment", "enclosed file", "appended document"),
    ("form", "input document", "data-entry sheet"),
    ("survey", "questionnaire", "feedback poll"),
    ("campaign", "marketing drive", "outreach initiative"),
    ("contact", "person entry", "address-book listing"),
];

/// The primary/alt gloss for a verb, if known.
fn verb_gloss(verb: &str, primary: bool) -> Option<&'static str> {
    VERB_GLOSS
        .iter()
        .find(|(w, ..)| *w == verb)
        .map(|(_, p, a)| if primary { *p } else { *a })
}

/// The primary/alt gloss for a noun key, if known.
fn noun_gloss(noun_key: &str, primary: bool) -> Option<&'static str> {
    NOUN_GLOSS
        .iter()
        .find(|(w, ..)| *w == noun_key)
        .map(|(_, p, a)| if primary { *p } else { *a })
}

/// Strip a trailing `_<digits>` disambiguation suffix from a procedural noun
/// key (the namer appends `_{n}` once the `verb×noun` space is exhausted), so
/// `ledger_37` looks up as `ledger`. Multi-word keys like `cluster_node` (no
/// trailing digits) pass through unchanged.
fn noun_key(rest: &str) -> &str {
    match rest.rsplit_once('_') {
        Some((head, tail)) if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) => head,
        _ => rest,
    }
}

/// The natural-language verb+noun phrase for a tool name under `mode`:
///
/// - `Name`: `fetch_ledger` → `"fetch the ledger"` (the current `verb_phrase_for`).
/// - `Description`: primary glosses → `"retrieve the accounting records"`.
/// - `Paraphrase`: alt glosses → `"pull up the account book"`.
///
/// Falls back to the raw word if a gloss is somehow missing (keeps the path
/// total for numeric-suffixed names whose noun key still resolves via
/// [`noun_key`]).
pub fn glossed_phrase(tool_name: &str, mode: SignalMode) -> String {
    let (verb, rest) = tool_name.split_once('_').unwrap_or((tool_name, ""));
    if rest.is_empty() {
        return verb.to_string();
    }
    match mode {
        SignalMode::Name => format!("{verb} the {}", rest.replace('_', " ")),
        SignalMode::Description | SignalMode::Paraphrase => {
            let primary = matches!(mode, SignalMode::Description);
            let raw_noun = rest.replace('_', " ");
            let vg = verb_gloss(verb, primary).unwrap_or(verb);
            let ng = noun_gloss(noun_key(rest), primary).unwrap_or(&raw_noun);
            format!("{vg} the {ng}")
        }
    }
}

/// The tool's human description under `mode`. `Name` keeps the current echo
/// form (`Procedurally generated tool \`{name}\` with N parameter(s).`);
/// `Description` and `Paraphrase` *both* use the **primary** gloss (only the
/// request differs between D1 and D2), a name-free sentence: `"Retrieve the
/// accounting records."`.
pub fn describe_tool(tool_name: &str, n_params: usize, mode: SignalMode) -> String {
    match mode {
        SignalMode::Name => {
            format!("Procedurally generated tool `{tool_name}` with {n_params} parameter(s).")
        }
        SignalMode::Description | SignalMode::Paraphrase => {
            // Description always uses the primary gloss, regardless of D1/D2.
            let phrase = glossed_phrase(tool_name, SignalMode::Description);
            format!("{}.", capitalize_first(&phrase))
        }
    }
}

/// Uppercase the first character of `s` (ASCII), leaving the rest untouched.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedural::{NOUNS, VERBS};

    /// Split a gloss into lowercased whole-word tokens (alnum runs).
    fn tokens(s: &str) -> Vec<String> {
        s.split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_ascii_lowercase())
            .collect()
    }

    /// Ubiquitous function words that carry no discriminative retrieval signal
    /// — a surface retriever keying on "the" learns nothing, so glosses sharing
    /// only these are still surface-disjoint in the sense D2 needs.
    const STOPWORDS: &[&str] = &[
        "the", "a", "an", "to", "of", "for", "up", "off", "in", "out", "on", "into", "from",
        "with", "and",
    ];

    fn content_tokens(s: &str) -> Vec<String> {
        tokens(s)
            .into_iter()
            .filter(|t| !STOPWORDS.contains(&t.as_str()))
            .collect()
    }

    #[test]
    fn every_verb_and_noun_has_exactly_one_gloss_row() {
        assert_eq!(
            VERB_GLOSS.len(),
            VERBS.len(),
            "VERB_GLOSS must cover every procedural verb"
        );
        assert_eq!(
            NOUN_GLOSS.len(),
            NOUNS.len(),
            "NOUN_GLOSS must cover every procedural noun"
        );
        for v in VERBS {
            assert!(
                verb_gloss(v, true).is_some() && verb_gloss(v, false).is_some(),
                "verb {v} missing a gloss"
            );
        }
        for n in NOUNS {
            assert!(
                noun_gloss(n, true).is_some() && noun_gloss(n, false).is_some(),
                "noun {n} missing a gloss"
            );
        }
    }

    #[test]
    fn glosses_share_no_whole_word_token_with_their_name() {
        // The load-bearing property: a glossed request/description must not
        // re-leak the name's surface form. For multi-token keys (`cluster_node`)
        // every underscore-split token is forbidden.
        for (word, p, a) in VERB_GLOSS.iter().chain(NOUN_GLOSS.iter()) {
            let name_toks: Vec<String> = word.split('_').map(|t| t.to_ascii_lowercase()).collect();
            for gloss in [p, a] {
                for gt in tokens(gloss) {
                    assert!(
                        !name_toks.contains(&gt),
                        "gloss {gloss:?} for {word:?} re-leaks name token {gt:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn primary_and_alt_are_surface_disjoint() {
        // D2 offers the alt gloss in the request and the primary in the
        // description; if they shared a token, D2 would leak surface signal.
        for (word, p, a) in VERB_GLOSS.iter().chain(NOUN_GLOSS.iter()) {
            let pt = content_tokens(p);
            for at in content_tokens(a) {
                assert!(
                    !pt.contains(&at),
                    "primary/alt gloss for {word:?} share content token {at:?} — D2 would not be surface-disjoint"
                );
            }
        }
    }

    #[test]
    fn glossed_phrase_and_description_carry_no_name_tokens_in_d1_d2() {
        for mode in [SignalMode::Description, SignalMode::Paraphrase] {
            let phrase = glossed_phrase("fetch_ledger", mode);
            for tok in tokens(&phrase) {
                assert!(
                    tok != "fetch" && tok != "ledger",
                    "D{mode:?} phrase leaked a name token: {phrase:?}"
                );
            }
        }
        // Description uses the primary gloss and is identical across D1/D2.
        let d1 = describe_tool("fetch_ledger", 2, SignalMode::Description);
        let d2 = describe_tool("fetch_ledger", 2, SignalMode::Paraphrase);
        assert_eq!(d1, d2);
        assert_eq!(d1, "Retrieve the accounting records.");
        assert_eq!(
            glossed_phrase("fetch_ledger", SignalMode::Paraphrase),
            "pull up the account book"
        );
        assert_eq!(
            glossed_phrase("fetch_ledger", SignalMode::Name),
            "fetch the ledger"
        );
    }

    #[test]
    fn numeric_suffix_noun_key_resolves() {
        // `ledger_37` (a disambiguated name) must still gloss via `ledger`.
        assert_eq!(
            glossed_phrase("fetch_ledger_37", SignalMode::Description),
            "retrieve the accounting records"
        );
        // Multi-word key `cluster_node` resolves as a whole.
        assert_eq!(
            glossed_phrase("get_cluster_node", SignalMode::Description),
            "obtain the worker machine"
        );
    }
}
