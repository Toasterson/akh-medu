//! Narrative synthesis: transforms working memory entries into human-readable prose.
//!
//! The pipeline:
//! 1. `extract_facts()` — parse WM ToolResult entries into structured facts
//! 2. `group_facts()` — group by topic (entity, code structure, similarity, etc.)
//! 3. `render_template()` — build a `NarrativeSummary` from grouped facts
//! 4. `polish_with_llm()` — optionally refine prose via Ollama (graceful fallback)

use std::collections::{BTreeMap, HashSet};

use super::llm::OllamaClient;
use super::memory::{WorkingMemoryEntry, WorkingMemoryKind};

// ── Public types ─────────────────────────────────────────────────────────

/// Classification of an extracted fact.
#[derive(Debug, Clone)]
pub enum FactKind {
    Triple {
        subject: String,
        predicate: String,
        object: String,
    },
    Similarity {
        entity: String,
        similar_to: String,
        score: f32,
    },
    Gap {
        entity: String,
        description: String,
    },
    Inference {
        expression: String,
        simplified: String,
    },
    Derivation {
        count: usize,
        iterations: usize,
    },
    CodeFact {
        kind: String,
        name: String,
        detail: String,
    },
    Raw(String),
}

/// A fact extracted from a working memory tool result.
#[derive(Debug, Clone)]
pub struct ExtractedFact {
    pub kind: FactKind,
    pub source_tool: String,
    pub source_cycle: u64,
}

/// A section of the narrative summary.
#[derive(Debug, Clone)]
pub struct NarrativeSection {
    pub heading: String,
    pub prose: String,
}

/// The final human-readable synthesis of agent findings.
#[derive(Debug, Clone)]
pub struct NarrativeSummary {
    pub overview: String,
    pub sections: Vec<NarrativeSection>,
    pub gaps: Vec<String>,
    pub facts_count: usize,
}

// ── Metadata filtering ───────────────────────────────────────────────────

/// Labels that are agent-internal metadata and should be excluded from narratives.
pub fn is_metadata_label(label: &str) -> bool {
    label.starts_with("desc:")
        || label.starts_with("status:")
        || label.starts_with("priority:")
        || label.starts_with("criteria:")
        || label.starts_with("goal:")
        || label.starts_with("agent:")
        || label.starts_with("episode:")
        || label.starts_with("summary:")
        || label.starts_with("tag:")
}

/// Whether a line references agent metadata labels (even embedded in a sentence).
fn contains_metadata_ref(line: &str) -> bool {
    let lower = line.to_lowercase();

    // Exact metadata tokens
    let metadata_tokens = [
        "status:active", "status:completed", "status:failed", "status:pending",
        "status:suspended",
        "agent:has_description", "agent:has_status", "agent:has_priority",
        "agent:has_criteria", "agent:is_goal",
        "priority:128", "priority:200",
        "goal:", "episode:", "summary:", "tag:", "desc:",
    ];
    if metadata_tokens.iter().any(|tok| lower.contains(tok)) {
        return true;
    }

    // "X has only N connection(s)" — always noise in gap output, not useful in narrative.
    if lower.contains("has only") && lower.contains("connection") {
        return true;
    }

    false
}

/// Strip a leading relevance score like `[0.90] ` from a gap line.
fn strip_leading_score(s: &str) -> &str {
    if s.starts_with('[') {
        if let Some(bracket_end) = s.find("] ") {
            let inner = &s[1..bracket_end];
            if inner.parse::<f32>().is_ok() {
                return s[bracket_end + 2..].trim();
            }
        }
    }
    s
}

/// Structural code predicates worth showing in narrative (NOT annotations).
const CODE_STRUCTURE_PREDICATES: &[&str] = &[
    "code:contains-mod", "code:defines-fn", "code:defines-struct",
    "code:defines-enum", "code:defines-type", "code:defines-mod",
    "code:depends-on", "code:defined-in",
    "code:has-variant", "code:has-method", "code:has-field",
    "code:derives-trait", "code:implements-trait",
    "defines-fn", "defines-type", "defines-mod",
    "depends-on", "implements", "contains",
];

/// Code annotation predicates to skip (noise in narrative).
const CODE_ANNOTATION_PREDICATES: &[&str] = &[
    "code:has-visibility", "code:has-attribute", "code:has-doc",
    "code:line-count", "code:has-return-type", "code:returns-type",
];

/// Inferred predicates that are redundant with structural code predicates.
const INFERRED_NOISE_PREDICATES: &[&str] = &[
    "child-of", "part-of", "similar-to", "has-a",
];

/// Whether a predicate indicates code structure (worth showing).
fn is_code_predicate(pred: &str) -> bool {
    CODE_STRUCTURE_PREDICATES.iter().any(|p| pred == *p)
}

/// Whether a predicate is a code annotation to skip.
fn is_code_annotation(pred: &str) -> bool {
    CODE_ANNOTATION_PREDICATES.iter().any(|p| pred == *p)
}

/// Whether a predicate is inferred noise (redundant with structural predicates).
fn is_inferred_noise(pred: &str) -> bool {
    INFERRED_NOISE_PREDICATES.iter().any(|p| pred == *p)
}

// ── Pipeline ─────────────────────────────────────────────────────────────

/// Top-level synthesis: extract facts from WM entries, group, render, optionally polish.
pub fn synthesize(
    goal: &str,
    entries: &[WorkingMemoryEntry],
    engine: &crate::engine::Engine,
    llm: Option<&OllamaClient>,
) -> NarrativeSummary {
    let facts = extract_facts(entries);
    let groups = group_facts(&facts);
    let mut summary = render_template(goal, &groups, &facts, engine);

    if let Some(client) = llm {
        if client.is_available() {
            summary = polish_with_llm(summary, client);
        }
    }

    summary
}

// ── Step 1: Extract facts ────────────────────────────────────────────────

/// Parse each WM ToolResult entry into structured `ExtractedFact` values.
fn extract_facts(entries: &[WorkingMemoryEntry]) -> Vec<ExtractedFact> {
    let mut facts = Vec::new();

    for entry in entries {
        if !matches!(entry.kind, WorkingMemoryKind::ToolResult) {
            continue;
        }

        let content = &entry.content;

        // Determine source tool from the "Tool result (xxx):" prefix.
        let source_tool = extract_tool_name(content);
        let body = strip_tool_prefix(content);

        match source_tool.as_str() {
            "kg_query" => {
                facts.extend(parse_kg_query_facts(&body, &source_tool, entry.source_cycle));
            }
            "similarity_search" => {
                facts.extend(parse_similarity_facts(
                    &body,
                    &source_tool,
                    entry.source_cycle,
                ));
            }
            "reason" => {
                facts.extend(parse_reason_facts(&body, &source_tool, entry.source_cycle));
            }
            "infer_rules" => {
                facts.extend(parse_infer_facts(&body, &source_tool, entry.source_cycle));
            }
            "gap_analysis" => {
                facts.extend(parse_gap_facts(&body, &source_tool, entry.source_cycle));
            }
            _ => {
                // Treat all other tools as raw facts (file_io, http_fetch, etc.)
                let trimmed = body.trim();
                if !trimmed.is_empty() && !is_metadata_noise(trimmed) {
                    facts.push(ExtractedFact {
                        kind: FactKind::Raw(trimmed.to_string()),
                        source_tool,
                        source_cycle: entry.source_cycle,
                    });
                }
            }
        }
    }

    // Deduplicate: keep first occurrence of each unique fact signature.
    let mut seen = HashSet::new();
    facts.retain(|f| {
        let key = match &f.kind {
            FactKind::Triple {
                subject,
                predicate,
                object,
            } => format!("T:{subject}:{predicate}:{object}"),
            FactKind::CodeFact { kind, name, detail } => format!("C:{kind}:{name}:{detail}"),
            FactKind::Similarity {
                entity,
                similar_to,
                ..
            } => format!("S:{entity}:{similar_to}"),
            FactKind::Gap {
                entity,
                description,
            } => format!("G:{entity}:{description}"),
            FactKind::Inference {
                expression,
                simplified,
            } => format!("I:{expression}:{simplified}"),
            FactKind::Derivation { count, iterations } => format!("D:{count}:{iterations}"),
            FactKind::Raw(s) => format!("R:{s}"),
        };
        seen.insert(key)
    });

    facts
}

fn extract_tool_name(content: &str) -> String {
    if let Some(start) = content.find("Tool result (") {
        let after = &content[start + 13..];
        if let Some(end) = after.find("):") {
            return after[..end].to_string();
        }
    }
    "unknown".to_string()
}

fn strip_tool_prefix(content: &str) -> String {
    if let Some(pos) = content.find("):\n") {
        content[pos + 3..].to_string()
    } else if let Some(pos) = content.find("):") {
        content[pos + 2..].to_string()
    } else {
        content.to_string()
    }
}

/// Whether a raw string is metadata noise that should be skipped.
fn is_metadata_noise(s: &str) -> bool {
    // Lines that are purely metadata labels
    if s.lines().all(|line| {
        let trimmed = line.trim();
        trimmed.is_empty() || is_metadata_label(trimmed)
    }) {
        return true;
    }
    // Agent metadata triples from kg_mutate: "goal:..." -> agent:... -> ...
    if s.starts_with("Added triple:") {
        let lower = s.to_lowercase();
        if lower.contains("goal:") || lower.contains("agent:") {
            return true;
        }
    }
    false
}

/// Parse kg_query results. Typical format:
/// `  subject -> predicate -> object`
/// or `Found N triples for "X":`
fn parse_kg_query_facts(body: &str, tool: &str, cycle: u64) -> Vec<ExtractedFact> {
    let mut facts = Vec::new();

    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("Found ") || trimmed.starts_with("Queried ") {
            continue;
        }
        if is_metadata_label(trimmed) {
            continue;
        }

        // Try parsing "subject -> predicate -> object" format
        if let Some(fact) = parse_arrow_triple(trimmed) {
            // Classify as code or regular triple
            let (s, p, o) = match &fact {
                FactKind::Triple {
                    subject,
                    predicate,
                    object,
                } => (subject.clone(), predicate.clone(), object.clone()),
                _ => unreachable!(),
            };

            // Skip code annotations (visibility, attributes, etc.) and inferred noise.
            if is_code_annotation(&p) || is_inferred_noise(&p) {
                continue;
            }

            if is_code_predicate(&p) {
                facts.push(ExtractedFact {
                    kind: FactKind::CodeFact {
                        kind: p.clone(),
                        name: s,
                        detail: o,
                    },
                    source_tool: tool.to_string(),
                    source_cycle: cycle,
                });
            } else if !is_metadata_label(&s)
                && !is_metadata_label(&p)
                && !is_metadata_label(&o)
            {
                facts.push(ExtractedFact {
                    kind: fact,
                    source_tool: tool.to_string(),
                    source_cycle: cycle,
                });
            }
        }
    }

    facts
}

/// Strip surrounding quotes and trailing confidence like `[0.85]` from a label.
///
/// KG query output looks like: `"Vsa" -> code:contains-mod -> "encode"  [1.00]`
fn clean_label(s: &str) -> String {
    let mut s = s.trim().to_string();
    // Strip trailing confidence: `  [0.85]` or ` [1.00]`
    if let Some(bracket_pos) = s.rfind("  [") {
        // Verify it ends with `]` and the content looks like a float
        let rest = &s[bracket_pos + 3..];
        if rest.ends_with(']') {
            let inner = &rest[..rest.len() - 1];
            if inner.parse::<f32>().is_ok() {
                s.truncate(bracket_pos);
            }
        }
    }
    // Strip surrounding quotes
    let trimmed = s.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

/// Try to parse `"A" -> B -> "C"  [0.85]` into a Triple fact.
fn parse_arrow_triple(line: &str) -> Option<FactKind> {
    let parts: Vec<&str> = line.split(" -> ").collect();
    if parts.len() >= 3 {
        let subject = clean_label(parts[0]);
        let predicate = clean_label(parts[1]);
        let object = clean_label(&parts[2..].join(" -> "));
        if subject.is_empty() || predicate.is_empty() || object.is_empty() {
            return None;
        }
        Some(FactKind::Triple {
            subject,
            predicate,
            object,
        })
    } else {
        None
    }
}

/// Parse similarity_search results. Typical format:
/// `  entity ~ similar_to (score: 0.85)`
fn parse_similarity_facts(body: &str, tool: &str, cycle: u64) -> Vec<ExtractedFact> {
    let mut facts = Vec::new();

    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("Similar")
            || trimmed.starts_with("Found")
            || is_metadata_label(trimmed)
        {
            continue;
        }

        // Try "entity ~ similar (score: N)" or "entity ~ similar (N)"
        if let Some(tilde_pos) = trimmed.find(" ~ ") {
            let entity = trimmed[..tilde_pos].trim().to_string();
            let rest = trimmed[tilde_pos + 3..].trim();

            let (similar_to, score) = if let Some(paren_pos) = rest.rfind('(') {
                let name = rest[..paren_pos].trim().to_string();
                let score_str = rest[paren_pos + 1..].trim_end_matches(')').trim();
                let score = score_str
                    .strip_prefix("score: ")
                    .or(Some(score_str))
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                (name, score)
            } else {
                (rest.to_string(), 0.0)
            };

            if !is_metadata_label(&entity) && !is_metadata_label(&similar_to) {
                facts.push(ExtractedFact {
                    kind: FactKind::Similarity {
                        entity,
                        similar_to,
                        score,
                    },
                    source_tool: tool.to_string(),
                    source_cycle: cycle,
                });
            }
        }
    }

    facts
}

/// Parse reason (egg) results.
fn parse_reason_facts(body: &str, tool: &str, cycle: u64) -> Vec<ExtractedFact> {
    let mut facts = Vec::new();
    let mut expression = String::new();
    let mut simplified = String::new();

    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Expression: ") {
            expression = rest.to_string();
        } else if let Some(rest) = trimmed.strip_prefix("Simplified: ") {
            simplified = rest.to_string();
        } else if let Some(rest) = trimmed.strip_prefix("Best: ") {
            simplified = rest.to_string();
        }
    }

    if !expression.is_empty() || !simplified.is_empty() {
        facts.push(ExtractedFact {
            kind: FactKind::Inference {
                expression,
                simplified,
            },
            source_tool: tool.to_string(),
            source_cycle: cycle,
        });
    }

    facts
}

/// Parse infer_rules results.
fn parse_infer_facts(body: &str, tool: &str, cycle: u64) -> Vec<ExtractedFact> {
    let mut count = 0usize;
    let mut iterations = 0usize;

    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Derived ") {
            // "Derived 5 new triples in 3 iterations"
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some(n) = parts.first().and_then(|s| s.parse::<usize>().ok()) {
                count = n;
            }
            if let Some(pos) = parts.iter().position(|&w| w == "in") {
                if let Some(i) = parts.get(pos + 1).and_then(|s| s.parse::<usize>().ok()) {
                    iterations = i;
                }
            }
        }
        // Also try to parse individual new triples as facts
        if let Some(fact) = parse_arrow_triple(trimmed) {
            facts_from_infer_triple(&mut Vec::new(), fact, tool, cycle);
        }
    }

    if count > 0 || iterations > 0 {
        vec![ExtractedFact {
            kind: FactKind::Derivation { count, iterations },
            source_tool: tool.to_string(),
            source_cycle: cycle,
        }]
    } else {
        Vec::new()
    }
}

fn facts_from_infer_triple(_facts: &mut Vec<ExtractedFact>, _fact: FactKind, _tool: &str, _cycle: u64) {
    // Individual infer triples are bundled into the Derivation count.
}

/// Parse gap_analysis results.
fn parse_gap_facts(body: &str, tool: &str, cycle: u64) -> Vec<ExtractedFact> {
    let mut facts = Vec::new();

    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("Gap analysis")
            || trimmed.starts_with("Found")
            || trimmed.starts_with("Analyzed")
            || is_metadata_label(trimmed)
        {
            continue;
        }

        // Skip lines that reference agent metadata labels anywhere.
        if contains_metadata_ref(trimmed) {
            continue;
        }

        // Skip generic schema-completeness gaps ("X is missing predicate 'Y'").
        if trimmed.contains("is missing predicate") {
            continue;
        }

        // Lines like "- entity: description" or "entity has no X"
        let cleaned = trimmed.strip_prefix("- ").unwrap_or(trimmed);
        // Strip leading relevance scores like "[0.90] "
        let cleaned = strip_leading_score(cleaned);
        // Skip entries where score stripping left an empty or truncated stub.
        if cleaned.is_empty()
            || cleaned == "..."
            || (cleaned.ends_with("...") && cleaned.len() < 10)
        {
            continue;
        }
        if let Some(colon_pos) = cleaned.find(": ") {
            let entity = cleaned[..colon_pos].trim().to_string();
            let description = cleaned[colon_pos + 2..].trim().to_string();
            if !is_metadata_label(&entity) {
                facts.push(ExtractedFact {
                    kind: FactKind::Gap {
                        entity,
                        description,
                    },
                    source_tool: tool.to_string(),
                    source_cycle: cycle,
                });
            }
        } else if !is_metadata_label(cleaned) {
            facts.push(ExtractedFact {
                kind: FactKind::Gap {
                    entity: String::new(),
                    description: cleaned.to_string(),
                },
                source_tool: tool.to_string(),
                source_cycle: cycle,
            });
        }
    }

    facts
}

// ── Step 2: Group facts ──────────────────────────────────────────────────

/// Grouping key for fact organization.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum GroupKey {
    Entity(String),
    CodeStructure,
    RelatedConcepts,
    KnowledgeGaps,
    ReasoningResults,
    Other,
}

struct FactGroup {
    key: GroupKey,
    facts: Vec<ExtractedFact>,
}

fn group_facts(facts: &[ExtractedFact]) -> Vec<FactGroup> {
    let mut groups: BTreeMap<GroupKey, Vec<ExtractedFact>> = BTreeMap::new();

    for fact in facts {
        let key = match &fact.kind {
            FactKind::Triple { subject, .. } => GroupKey::Entity(subject.clone()),
            FactKind::CodeFact { .. } => GroupKey::CodeStructure,
            FactKind::Similarity { .. } => GroupKey::RelatedConcepts,
            FactKind::Gap { .. } => GroupKey::KnowledgeGaps,
            FactKind::Inference { .. } | FactKind::Derivation { .. } => {
                GroupKey::ReasoningResults
            }
            FactKind::Raw(_) => GroupKey::Other,
        };
        groups.entry(key).or_default().push(fact.clone());
    }

    groups
        .into_iter()
        .map(|(key, facts)| FactGroup { key, facts })
        .collect()
}

// ── Step 3: Render template ──────────────────────────────────────────────

fn render_template(
    goal: &str,
    groups: &[FactGroup],
    all_facts: &[ExtractedFact],
    engine: &crate::engine::Engine,
) -> NarrativeSummary {
    let total_facts = all_facts.len();
    let max_cycle = all_facts
        .iter()
        .map(|f| f.source_cycle)
        .max()
        .unwrap_or(0);
    let topic_count = groups
        .iter()
        .filter(|g| !matches!(g.key, GroupKey::Other | GroupKey::KnowledgeGaps))
        .count();

    let overview = if total_facts == 0 {
        format!("Explored \"{goal}\" but found no concrete facts yet.")
    } else {
        format!(
            "Explored \"{goal}\" over {max_cycle} cycle{}, found {total_facts} fact{} about {topic_count} topic{}.",
            if max_cycle == 1 { "" } else { "s" },
            if total_facts == 1 { "" } else { "s" },
            if topic_count == 1 { "" } else { "s" },
        )
    };

    let mut sections = Vec::new();
    let mut gaps = Vec::new();

    for group in groups {
        match &group.key {
            GroupKey::Entity(entity) => {
                sections.push(render_entity_section(entity, &group.facts));
            }
            GroupKey::CodeStructure => {
                sections.push(render_code_section(&group.facts, engine));
            }
            GroupKey::RelatedConcepts => {
                sections.push(render_similarity_section(&group.facts));
            }
            GroupKey::KnowledgeGaps => {
                for fact in &group.facts {
                    if let FactKind::Gap {
                        entity,
                        description,
                    } = &fact.kind
                    {
                        if entity.is_empty() {
                            gaps.push(description.clone());
                        } else {
                            gaps.push(format!("{entity}: {description}"));
                        }
                    }
                }
            }
            GroupKey::ReasoningResults => {
                sections.push(render_reasoning_section(&group.facts));
            }
            GroupKey::Other => {
                if let Some(sec) = render_other_section(&group.facts) {
                    sections.push(sec);
                }
            }
        }
    }

    // Drop sections with only placeholder prose (nothing useful to show).
    // Also drop trivially uninformative entity sections ("X is a Module.").
    sections.retain(|s| {
        if s.prose.starts_with("No ") {
            return false;
        }
        // Entity sections that only state "is a Module/Function/Struct" are noise.
        is_informative_section(s)
    });

    NarrativeSummary {
        overview,
        sections,
        gaps,
        facts_count: total_facts,
    }
}

fn render_entity_section(entity: &str, facts: &[ExtractedFact]) -> NarrativeSection {
    let mut predicates: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for fact in facts {
        if let FactKind::Triple {
            predicate, object, ..
        } = &fact.kind
        {
            // Skip code predicates here — they belong in the Code Structure section.
            if is_code_predicate(predicate) || is_code_annotation(predicate) {
                continue;
            }
            predicates
                .entry(predicate.clone())
                .or_default()
                .push(object.clone());
        }
    }

    let mut prose_parts = Vec::new();

    for (pred, objects) in &predicates {
        let obj_list = format_list(objects);
        // Humanize common predicates
        let sentence = match pred.as_str() {
            "is-a" | "is_a" | "rdf:type" => format!("is a {obj_list}"),
            "has" | "has-part" => format!("has {obj_list}"),
            "depends-on" => format!("depends on {obj_list}"),
            "implements" => format!("implements {obj_list}"),
            "contains" => format!("contains {obj_list}"),
            "defines-fn" | "code:defines-fn" | "code:has-method" => {
                format!("defines functions {obj_list}")
            }
            "defines-type" | "code:defines-type" | "code:defines-struct" => {
                format!("defines types {obj_list}")
            }
            "defines-mod" | "code:defines-mod" | "code:contains-mod" => {
                format!("contains modules {obj_list}")
            }
            "code:defines-enum" => format!("defines enums {obj_list}"),
            "code:has-variant" => format!("has variants {obj_list}"),
            "code:has-field" => format!("has fields {obj_list}"),
            "code:depends-on" => format!("depends on {obj_list}"),
            "code:defined-in" => format!("is defined in {obj_list}"),
            "code:derives-trait" => format!("derives {obj_list}"),
            "code:implements-trait" => format!("implements {obj_list}"),
            _ => format!("{pred} {obj_list}"),
        };
        prose_parts.push(sentence);
    }

    let prose = if prose_parts.is_empty() {
        format!("No details found for **{entity}**.")
    } else {
        format!("**{entity}** {}.", prose_parts.join(", "))
    };

    NarrativeSection {
        heading: entity.to_string(),
        prose,
    }
}

fn render_code_section(facts: &[ExtractedFact], engine: &crate::engine::Engine) -> NarrativeSection {
    let mut modules: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut functions: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut types: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut deps: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut prose_parts = Vec::new();

    for fact in facts {
        if let FactKind::CodeFact { kind, name, detail } = &fact.kind {
            match kind.as_str() {
                "code:contains-mod" | "code:defines-mod" | "defines-mod" | "contains" => {
                    modules
                        .entry(name.clone())
                        .or_default()
                        .push(detail.clone());
                }
                "code:defines-fn" | "defines-fn" | "code:has-method" => {
                    functions
                        .entry(name.clone())
                        .or_default()
                        .push(detail.clone());
                }
                "code:defines-struct" | "code:defines-enum" | "code:defines-type"
                | "defines-type" | "implements" | "code:has-variant" | "code:has-field"
                | "code:derives-trait" | "code:implements-trait" => {
                    types
                        .entry(name.clone())
                        .or_default()
                        .push(detail.clone());
                }
                "depends-on" | "code:depends-on" => {
                    deps.entry(name.clone()).or_default().push(detail.clone());
                }
                "code:defined-in" => {
                    // "defined-in" is informational — add as module metadata
                    prose_parts.push(format!("**{name}** is defined in `{detail}`."));
                }
                _ => {
                    // Genuinely unknown code predicates — show as-is
                    functions
                        .entry(name.clone())
                        .or_default()
                        .push(detail.clone());
                }
            }
        }
    }

    // Identify the primary entity (most code facts) and its children.
    // Filter out noise from unrelated entities that happen to reference the same symbols.
    let mut fact_counts: BTreeMap<String, usize> = BTreeMap::new();
    for map in [&modules, &functions, &types, &deps] {
        for (name, items) in map {
            *fact_counts.entry(name.clone()).or_default() += items.len();
        }
    }
    let primary = fact_counts
        .iter()
        .max_by_key(|(_, count)| **count)
        .map(|(name, _)| name.clone());
    let primary_children: HashSet<String> = primary
        .as_ref()
        .and_then(|p| modules.get(p))
        .map(|children| children.iter().cloned().collect())
        .unwrap_or_default();

    // Only keep entries from the primary entity, its children, or entities with 2+ facts.
    let is_relevant = |name: &str| -> bool {
        primary.as_deref() == Some(name)
            || primary_children.contains(name)
            || fact_counts.get(name).copied().unwrap_or(0) >= 2
    };
    modules.retain(|name, _| is_relevant(name));
    functions.retain(|name, _| is_relevant(name));
    types.retain(|name, _| is_relevant(name));
    deps.retain(|name, _| is_relevant(name));

    // ── Semantic enrichment lookups ──────────────────────────────────────
    // Try to load semantic predicates for enrichment data (roles, importance, flow).
    // If enrichment hasn't run, these will gracefully return None/empty.
    let sem_preds = super::semantic_enrichment::SemanticPredicates::init(engine).ok();

    // Helper: resolve a label to its SymbolId (best-effort).
    let resolve_sym = |label: &str| -> Option<crate::symbol::SymbolId> {
        engine.resolve_symbol(label).ok()
    };

    // Helper: get role label for a name.
    let get_role = |name: &str| -> Option<String> {
        let preds = sem_preds.as_ref()?;
        let sym = resolve_sym(name)?;
        super::semantic_enrichment::lookup_role(engine, sym, preds)
    };

    // Helper: get importance for a name.
    let get_importance = |name: &str| -> Option<f32> {
        let preds = sem_preds.as_ref()?;
        let sym = resolve_sym(name)?;
        super::semantic_enrichment::lookup_importance(engine, sym, preds)
    };

    // Build hierarchical output: group child details under the primary entity.
    let mut rendered_children: HashSet<String> = HashSet::new();

    if let Some(ref primary_name) = primary {
        // ── Purpose header from primary roles ──
        let primary_roles: Vec<String> = sem_preds.as_ref().and_then(|preds| {
            let sym = resolve_sym(primary_name)?;
            let triples = engine.triples_from(sym);
            let roles: Vec<String> = triples
                .iter()
                .filter(|t| t.predicate == preds.has_role)
                .map(|t| {
                    engine
                        .resolve_label(t.object)
                        .trim_start_matches("role:")
                        .to_string()
                })
                .collect();
            if roles.is_empty() { None } else { Some(roles) }
        }).unwrap_or_default();

        if let Some(children) = modules.get(primary_name) {
            // Sort children by importance (highest first), falling back to name order.
            let mut sorted_children: Vec<&String> = children.iter().collect();
            sorted_children.sort_by(|a, b| {
                let imp_a = get_importance(a).unwrap_or(0.0);
                let imp_b = get_importance(b).unwrap_or(0.0);
                imp_b
                    .partial_cmp(&imp_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut child_lines = Vec::new();
            for child in &sorted_children {
                let mut parts = Vec::new();
                if let Some(fns) = functions.get(*child) {
                    parts.push(format!("functions: {}", format_code_list(fns)));
                }
                if let Some(ts) = types.get(*child) {
                    parts.push(format!("types: {}", format_code_list(ts)));
                }
                if let Some(ds) = deps.get(*child) {
                    parts.push(format!("depends on {}", format_list(ds)));
                }

                // Enrichment annotations: role label and importance star.
                let role_tag = get_role(child)
                    .map(|r| format!(" ({r})"))
                    .unwrap_or_default();
                let star = if get_importance(child).unwrap_or(0.0) > 0.7 {
                    ", \u{2605}"
                } else {
                    ""
                };

                if parts.is_empty() {
                    child_lines.push(format!("- **{child}**{role_tag}{star}"));
                } else {
                    child_lines.push(format!(
                        "- **{child}**{role_tag}{star} \u{2014} {}",
                        parts.join(". "),
                    ));
                }
                rendered_children.insert((*child).clone());
            }

            // Truncation: > 8 children → show top 5 + "and N more"
            let total_children = child_lines.len();
            if total_children > 8 {
                child_lines.truncate(5);
                child_lines.push(format!("- ...and {} more", total_children - 5));
            }

            if !child_lines.is_empty() {
                let purpose = if !primary_roles.is_empty() {
                    let role_desc = primary_roles.join(" and ");
                    format!(
                        "The **{primary_name}** module is a {role_desc} layer.\n\nIt contains {} submodule{}:\n{}",
                        total_children,
                        if total_children == 1 { "" } else { "s" },
                        child_lines.join("\n"),
                    )
                } else {
                    format!(
                        "The **{primary_name}** module contains:\n{}",
                        child_lines.join("\n"),
                    )
                };
                prose_parts.push(purpose);
            }

            // Data flow line from semantic:flows-to triples.
            if let Some(ref preds) = sem_preds {
                let child_syms: Vec<crate::symbol::SymbolId> = sorted_children
                    .iter()
                    .filter_map(|c| resolve_sym(c))
                    .collect();
                let flow_chain =
                    super::semantic_enrichment::build_flow_chain(engine, preds, &child_syms);
                if flow_chain.len() >= 2 {
                    let flow_parts: Vec<String> = flow_chain
                        .iter()
                        .map(|(name, via)| {
                            if let Some(t) = via {
                                format!("`{name}` \u{2192} {t}")
                            } else {
                                format!("`{name}`")
                            }
                        })
                        .collect();
                    prose_parts.push(format!("Data flow: {}", flow_parts.join(" \u{2192} ")));
                }
            }

            // Render primary entity's own types and deps (not child-level).
            if let Some(ts) = types.get(primary_name) {
                // Add star to high-importance types.
                let typed_list: Vec<String> = ts
                    .iter()
                    .map(|t| {
                        let star = if get_importance(t).unwrap_or(0.0) > 0.7 {
                            " (\u{2605})"
                        } else {
                            ""
                        };
                        format!("`{t}`{star}")
                    })
                    .collect();
                prose_parts.push(format!("Key types: {}.", format_list(&typed_list)));
            }
            if let Some(ds) = deps.get(primary_name) {
                let list = format_list(ds);
                prose_parts.push(format!("Depends on {list}."));
            }
            rendered_children.insert(primary_name.clone());
        }
    }

    // Render any remaining entities not covered by the hierarchical tree.
    for (parent, children) in &modules {
        if rendered_children.contains(parent) {
            continue;
        }
        let list = format_code_list(children);
        prose_parts.push(format!("The **{parent}** module contains {list}."));
    }

    for (owner, fns) in &functions {
        if rendered_children.contains(owner) {
            continue;
        }
        let list = format_code_list(fns);
        prose_parts.push(format!("**{owner}** defines functions {list}."));
    }

    for (owner, ts) in &types {
        if rendered_children.contains(owner) {
            continue;
        }
        let list = format_code_list(ts);
        prose_parts.push(format!("**{owner}** defines types {list}."));
    }

    for (module, dep_list) in &deps {
        if rendered_children.contains(module) {
            continue;
        }
        let list = format_list(dep_list);
        prose_parts.push(format!("**{module}** depends on {list}."));
    }

    let prose = if prose_parts.is_empty() {
        "No code structure details found.".to_string()
    } else {
        prose_parts.join(" ")
    };

    // Use "Code Architecture" heading when enrichment data is present,
    // fall back to "Code Structure" when it's purely structural.
    let has_enrichment = sem_preds.is_some()
        && primary.as_ref().is_some_and(|p| get_role(p).is_some());
    let heading = if has_enrichment {
        "Code Architecture"
    } else {
        "Code Structure"
    };

    NarrativeSection {
        heading: heading.to_string(),
        prose,
    }
}

fn render_similarity_section(facts: &[ExtractedFact]) -> NarrativeSection {
    let mut lines = Vec::new();

    for fact in facts {
        if let FactKind::Similarity {
            entity,
            similar_to,
            score,
        } = &fact.kind
        {
            if *score > 0.0 {
                lines.push(format!(
                    "**{entity}** is related to **{similar_to}** (similarity: {score:.2})."
                ));
            } else {
                lines.push(format!("**{entity}** is related to **{similar_to}**."));
            }
        }
    }

    NarrativeSection {
        heading: "Related Concepts".to_string(),
        prose: if lines.is_empty() {
            "No related concepts found.".to_string()
        } else {
            lines.join(" ")
        },
    }
}

fn render_reasoning_section(facts: &[ExtractedFact]) -> NarrativeSection {
    let mut lines = Vec::new();

    for fact in facts {
        match &fact.kind {
            FactKind::Inference {
                expression,
                simplified,
            } => {
                // Skip noise: goal-derived expressions or trivial rewrites
                let is_noise = simplified.contains("goal_")
                    || expression.contains("goal_")
                    || expression.is_empty()
                    || simplified.is_empty()
                    || simplified == expression;
                if !is_noise && !simplified.is_empty() {
                    lines.push(format!("Simplified `{expression}` to `{simplified}`."));
                }
            }
            FactKind::Derivation { count, iterations } => {
                // Skip zero derivations (noise) and counts > 100 (internal e-graph expansion).
                if *count > 0 && *count <= 100 {
                    lines.push(format!(
                        "Derived {count} new fact{} in {iterations} iteration{}.",
                        if *count == 1 { "" } else { "s" },
                        if *iterations == 1 { "" } else { "s" },
                    ));
                }
            }
            _ => {}
        }
    }

    NarrativeSection {
        heading: "Reasoning Results".to_string(),
        prose: if lines.is_empty() {
            "No reasoning results.".to_string()
        } else {
            lines.join(" ")
        },
    }
}

fn render_other_section(facts: &[ExtractedFact]) -> Option<NarrativeSection> {
    let lines: Vec<String> = facts
        .iter()
        .filter_map(|f| match &f.kind {
            FactKind::Raw(s) => {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            }
            _ => None,
        })
        .collect();

    if lines.is_empty() {
        return None;
    }

    Some(NarrativeSection {
        heading: "Other Findings".to_string(),
        prose: lines.join("\n"),
    })
}

// ── Step 4: Optional LLM polish ──────────────────────────────────────────

fn polish_with_llm(summary: NarrativeSummary, client: &OllamaClient) -> NarrativeSummary {
    let template_text = format_summary_as_text(&summary);

    let system = "Rewrite the following agent findings into clear, concise prose. \
        Preserve all facts exactly. Do not add information not present in the input. \
        Do NOT speculate about what modules, functions, or types do — only describe \
        relationships and names that appear in the input. \
        Preserve computed annotations: role labels like (transformation), (storage) \
        and importance markers like (\u{2605}). These are derived from graph analysis, \
        not speculation — keep them exactly as written. \
        Keep ## Markdown headings. Be conversational but factual.";

    match client.generate(&template_text, Some(system)) {
        Ok(polished) => parse_polished_response(&polished, &summary),
        Err(_) => summary, // Graceful degradation to template
    }
}

fn format_summary_as_text(summary: &NarrativeSummary) -> String {
    let mut text = summary.overview.clone();
    text.push('\n');
    for section in &summary.sections {
        text.push_str(&format!("\n## {}\n{}\n", section.heading, section.prose));
    }
    if !summary.gaps.is_empty() {
        text.push_str("\n## Knowledge Gaps\n");
        for gap in &summary.gaps {
            text.push_str(&format!("- {gap}\n"));
        }
    }
    text
}

fn parse_polished_response(polished: &str, original: &NarrativeSummary) -> NarrativeSummary {
    // Split polished text by ## headings
    let mut sections = Vec::new();
    let mut overview = String::new();
    let mut current_heading = String::new();
    let mut current_body = String::new();
    let mut gaps = original.gaps.clone();

    for line in polished.lines() {
        if let Some(heading) = line.strip_prefix("## ") {
            // Flush previous section
            if !current_heading.is_empty() {
                if current_heading == "Knowledge Gaps" {
                    // Parse gap lines
                    gaps = current_body
                        .lines()
                        .filter_map(|l| {
                            let t = l.trim().strip_prefix("- ").unwrap_or(l.trim());
                            if t.is_empty() {
                                None
                            } else {
                                Some(t.to_string())
                            }
                        })
                        .collect();
                } else {
                    sections.push(NarrativeSection {
                        heading: current_heading.clone(),
                        prose: current_body.trim().to_string(),
                    });
                }
            } else if !current_body.trim().is_empty() {
                overview = current_body.trim().to_string();
            }
            current_heading = heading.trim().to_string();
            current_body.clear();
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    // Flush last section
    if !current_heading.is_empty() {
        if current_heading == "Knowledge Gaps" {
            gaps = current_body
                .lines()
                .filter_map(|l| {
                    let t = l.trim().strip_prefix("- ").unwrap_or(l.trim());
                    if t.is_empty() {
                        None
                    } else {
                        Some(t.to_string())
                    }
                })
                .collect();
        } else {
            sections.push(NarrativeSection {
                heading: current_heading,
                prose: current_body.trim().to_string(),
            });
        }
    } else if overview.is_empty() && !current_body.trim().is_empty() {
        overview = current_body.trim().to_string();
    }

    // Fallback to original overview if parsing failed
    if overview.is_empty() {
        overview = original.overview.clone();
    }

    NarrativeSummary {
        overview,
        sections: if sections.is_empty() {
            original.sections.clone()
        } else {
            sections
        },
        gaps,
        facts_count: original.facts_count,
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────

/// Trivial type-only section: entity sections that say only "X is a Module/Function/Struct".
const TRIVIAL_TYPES: &[&str] = &[
    "Module", "Function", "Struct", "Enum", "Type", "Trait",
];

/// Whether a section is informative enough to show (not just "X is a Module.").
fn is_informative_section(section: &NarrativeSection) -> bool {
    // Code Structure/Architecture, Related Concepts, etc. are always informative.
    if section.heading == "Code Structure"
        || section.heading == "Code Architecture"
        || section.heading == "Related Concepts"
        || section.heading == "Reasoning Results"
    {
        return true;
    }
    // Entity sections: check if they say more than just "is a Module".
    let prose = &section.prose;
    // Pattern: "**entity** is a X." or "**entity** is a X and Y."
    if let Some(rest) = prose.strip_prefix(&format!("**{}** is a ", section.heading)) {
        let rest = rest.trim_end_matches('.');
        // Split by " and " — if all parts are trivial types, skip.
        let parts: Vec<&str> = rest.split(" and ").collect();
        if parts
            .iter()
            .all(|p| TRIVIAL_TYPES.iter().any(|t| p.trim() == *t))
        {
            return false;
        }
    }
    true
}

/// Format a list of items as "A, B, and C".
fn format_list(items: &[String]) -> String {
    match items.len() {
        0 => "nothing".to_string(),
        1 => items[0].clone(),
        2 => format!("{} and {}", items[0], items[1]),
        _ => {
            let last = &items[items.len() - 1];
            let rest = &items[..items.len() - 1];
            format!("{}, and {last}", rest.join(", "))
        }
    }
}

/// Format a list of code identifiers with backticks.
fn format_code_list(items: &[String]) -> String {
    let coded: Vec<String> = items.iter().map(|i| format!("`{i}`")).collect();
    format_list(&coded)
}

impl std::fmt::Display for NarrativeSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.overview)?;
        for section in &self.sections {
            writeln!(f, "\n## {}", section.heading)?;
            writeln!(f, "{}", section.prose)?;
        }
        if !self.gaps.is_empty() {
            writeln!(f, "\nOpen questions:")?;
            for gap in &self.gaps {
                writeln!(f, "  - {gap}")?;
            }
        }
        Ok(())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::WorkingMemoryKind;

    fn make_tool_entry(content: &str, cycle: u64) -> WorkingMemoryEntry {
        WorkingMemoryEntry {
            id: 1,
            content: content.to_string(),
            symbols: vec![],
            kind: WorkingMemoryKind::ToolResult,
            timestamp: 0,
            relevance: 1.0,
            source_cycle: cycle,
            reference_count: 0,
        }
    }

    #[test]
    fn extract_tool_name_parses_prefix() {
        assert_eq!(
            extract_tool_name("Tool result (kg_query):\nstuff"),
            "kg_query"
        );
        assert_eq!(
            extract_tool_name("Tool result (similarity_search):\nstuff"),
            "similarity_search"
        );
        assert_eq!(extract_tool_name("no prefix here"), "unknown");
    }

    #[test]
    fn strip_tool_prefix_removes_header() {
        assert_eq!(
            strip_tool_prefix("Tool result (kg_query):\nfoo bar"),
            "foo bar"
        );
    }

    #[test]
    fn parse_arrow_triple_works() {
        let fact = parse_arrow_triple("Engine -> defines-fn -> create_symbol");
        assert!(matches!(
            fact,
            Some(FactKind::Triple { subject, predicate, object })
            if subject == "Engine" && predicate == "defines-fn" && object == "create_symbol"
        ));
    }

    #[test]
    fn metadata_labels_filtered() {
        assert!(is_metadata_label("desc:some description"));
        assert!(is_metadata_label("status:active"));
        assert!(is_metadata_label("priority:128"));
        assert!(is_metadata_label("agent:goal"));
        assert!(!is_metadata_label("Engine"));
        assert!(!is_metadata_label("defines-fn"));
    }

    #[test]
    fn code_predicates_detected() {
        assert!(is_code_predicate("code:defines-fn"));
        assert!(is_code_predicate("code:contains-mod"));
        assert!(is_code_predicate("code:defines-struct"));
        assert!(is_code_predicate("defines-fn"));
        assert!(is_code_predicate("depends-on"));
        assert!(!is_code_predicate("is-a"));
        assert!(!is_code_predicate("has"));
        // Annotations should NOT be code predicates
        assert!(!is_code_predicate("code:has-visibility"));
        assert!(!is_code_predicate("code:has-attribute"));
        // But they should be annotations
        assert!(is_code_annotation("code:has-visibility"));
        assert!(is_code_annotation("code:has-attribute"));
    }

    #[test]
    fn extract_facts_from_kg_query() {
        let entry = make_tool_entry(
            "Tool result (kg_query):\nFound 2 triples for \"Engine\":\n  Engine -> defines-fn -> create_symbol\n  Engine -> is-a -> Struct",
            1,
        );
        let facts = extract_facts(&[entry]);
        assert_eq!(facts.len(), 2);
        // First should be CodeFact (defines-fn is code predicate)
        assert!(matches!(&facts[0].kind, FactKind::CodeFact { kind, .. } if kind == "defines-fn"));
        // Second should be Triple (is-a is not code predicate)
        assert!(matches!(&facts[1].kind, FactKind::Triple { predicate, .. } if predicate == "is-a"));
    }

    #[test]
    fn extract_facts_filters_metadata() {
        let entry = make_tool_entry(
            "Tool result (kg_query):\n  status:active -> is-a -> status\n  Engine -> defines-fn -> search",
            1,
        );
        let facts = extract_facts(&[entry]);
        // Only the Engine fact should remain (status:active is metadata)
        assert_eq!(facts.len(), 1);
    }

    #[test]
    fn visibility_annotations_filtered() {
        let entry = make_tool_entry(
            "Tool result (kg_query):\n  \"MyMod\" -> code:has-visibility -> \"public\"  [1.00]\n  \"MyMod\" -> code:defines-fn -> \"my_func\"  [1.00]",
            1,
        );
        let facts = extract_facts(&[entry]);
        // Only the defines-fn fact should survive, has-visibility is annotation noise
        assert_eq!(facts.len(), 1);
        assert!(matches!(&facts[0].kind, FactKind::CodeFact { kind, .. } if kind == "code:defines-fn"));
    }

    #[test]
    fn similarity_facts_parsed() {
        let entry = make_tool_entry(
            "Tool result (similarity_search):\nSimilar to \"vsa\":\n  HyperVec ~ VsaOps (score: 0.85)\n  encode ~ decode (0.72)",
            2,
        );
        let facts = extract_facts(&[entry]);
        assert_eq!(facts.len(), 2);
        assert!(matches!(&facts[0].kind, FactKind::Similarity { score, .. } if (*score - 0.85).abs() < 0.01));
    }

    #[test]
    fn group_facts_organizes_by_topic() {
        let facts = vec![
            ExtractedFact {
                kind: FactKind::Triple {
                    subject: "Engine".into(),
                    predicate: "is-a".into(),
                    object: "Struct".into(),
                },
                source_tool: "kg_query".into(),
                source_cycle: 1,
            },
            ExtractedFact {
                kind: FactKind::Triple {
                    subject: "Engine".into(),
                    predicate: "has".into(),
                    object: "registry".into(),
                },
                source_tool: "kg_query".into(),
                source_cycle: 1,
            },
            ExtractedFact {
                kind: FactKind::Gap {
                    entity: "VSA".into(),
                    description: "no trait impls found".into(),
                },
                source_tool: "gap_analysis".into(),
                source_cycle: 2,
            },
        ];

        let groups = group_facts(&facts);
        // Should have Entity("Engine") and KnowledgeGaps groups
        assert!(groups.iter().any(|g| g.key == GroupKey::Entity("Engine".into())));
        assert!(groups.iter().any(|g| g.key == GroupKey::KnowledgeGaps));
    }

    #[test]
    fn format_list_handles_counts() {
        assert_eq!(format_list(&[]), "nothing");
        assert_eq!(format_list(&["A".into()]), "A");
        assert_eq!(format_list(&["A".into(), "B".into()]), "A and B");
        assert_eq!(
            format_list(&["A".into(), "B".into(), "C".into()]),
            "A, B, and C"
        );
    }

    #[test]
    fn narrative_summary_display() {
        let summary = NarrativeSummary {
            overview: "Found stuff.".into(),
            sections: vec![NarrativeSection {
                heading: "Engine".into(),
                prose: "**Engine** is a Struct.".into(),
            }],
            gaps: vec!["VSA traits unknown".into()],
            facts_count: 5,
        };
        let text = format!("{summary}");
        assert!(text.contains("Found stuff."));
        assert!(text.contains("## Engine"));
        assert!(text.contains("**Engine** is a Struct."));
        assert!(text.contains("VSA traits unknown"));
    }

    #[test]
    fn clean_label_strips_quotes_and_confidence() {
        assert_eq!(clean_label("\"Vsa\""), "Vsa");
        assert_eq!(clean_label("\"encode\"  [1.00]"), "encode");
        assert_eq!(clean_label("code:contains-mod"), "code:contains-mod");
        assert_eq!(clean_label("\"HyperVec\"  [0.85]"), "HyperVec");
        assert_eq!(clean_label("plain"), "plain");
        assert_eq!(clean_label("\"quoted\""), "quoted");
    }

    #[test]
    fn parse_arrow_triple_with_quotes_and_confidence() {
        let fact = parse_arrow_triple("\"Vsa\" -> code:contains-mod -> \"encode\"  [1.00]");
        assert!(matches!(
            fact,
            Some(FactKind::Triple { subject, predicate, object })
            if subject == "Vsa" && predicate == "code:contains-mod" && object == "encode"
        ));
    }

    #[test]
    fn metadata_noise_filters_agent_triples() {
        assert!(is_metadata_noise("Added triple: \"goal:test\" -> agent:has_criteria -> \"data\" [0.70]"));
        assert!(!is_metadata_noise("Added triple: \"Vsa\" -> is-a -> \"Module\" [1.00]"));
    }

    #[test]
    fn contains_metadata_ref_catches_embedded() {
        assert!(contains_metadata_ref("[0.90] status:completed has only 1 connection"));
        assert!(contains_metadata_ref("agent:has_description is sparse"));
        assert!(!contains_metadata_ref("Engine has 5 functions"));
    }

    #[test]
    fn strip_leading_score_works() {
        assert_eq!(strip_leading_score("[0.90] some text"), "some text");
        assert_eq!(strip_leading_score("no score here"), "no score here");
        assert_eq!(strip_leading_score("[invalid] text"), "[invalid] text");
    }

    #[test]
    fn gap_facts_filter_metadata_refs() {
        let entry = make_tool_entry(
            "Tool result (gap_analysis):\nAnalyzed 10 entities: 5 dead ends\n[0.90] status:completed has only 1 connection(s)\n[0.85] Engine: has no documentation",
            3,
        );
        let facts = extract_facts(&[entry]);
        // Only the Engine gap should survive — status:completed is metadata, Analyzed is skipped
        assert_eq!(facts.len(), 1);
        assert!(matches!(&facts[0].kind, FactKind::Gap { entity, .. } if entity == "Engine"));
    }

    #[test]
    fn duplicate_facts_deduplicated() {
        // Same triple from two different cycles should appear only once.
        let e1 = make_tool_entry(
            "Tool result (kg_query):\n\"Vsa\" -> code:contains-mod -> \"encode\"  [1.00]",
            1,
        );
        let e2 = make_tool_entry(
            "Tool result (kg_query):\n\"Vsa\" -> code:contains-mod -> \"encode\"  [1.00]",
            3,
        );
        let facts = extract_facts(&[e1, e2]);
        assert_eq!(facts.len(), 1, "duplicate triples should be merged");
    }

    #[test]
    fn low_connection_gaps_filtered() {
        let entry = make_tool_entry(
            "Tool result (gap_analysis):\n[0.80] web has only 1 connection(s) (in=1, out=0)\n[0.75] api has only 2 connection(s) (in=1, out=1)",
            2,
        );
        let facts = extract_facts(&[entry]);
        assert_eq!(facts.len(), 0, "low-connection gaps should be filtered");
    }

    #[test]
    fn zero_derivation_filtered() {
        let entry = make_tool_entry(
            "Tool result (infer_rules):\nDerived 0 new triples in 1 iterations",
            1,
        );
        let facts = extract_facts(&[entry]);
        // Zero derivations should produce no Derivation facts (filtered at extract level).
        // The Derivation fact IS created but the rendering skips count==0.
        let section = render_reasoning_section(&facts);
        assert!(
            !section.prose.contains("Derived 0"),
            "zero-derivation should not appear in prose: {}",
            section.prose,
        );
    }

    #[test]
    fn gap_truncated_scores_filtered() {
        let entry = make_tool_entry(
            "Tool result (gap_analysis):\n- [0.90] ...\n- [0.85] ab...\n- [0.80] Engine: valid gap description",
            3,
        );
        let facts = extract_facts(&[entry]);
        // Only the Engine gap should survive — truncated stubs are skipped.
        assert_eq!(facts.len(), 1, "truncated gaps should be filtered, got: {facts:?}");
        assert!(matches!(&facts[0].kind, FactKind::Gap { entity, .. } if entity == "Engine"));
    }

    fn test_engine() -> crate::engine::Engine {
        crate::engine::Engine::new(crate::engine::EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn hierarchical_code_rendering() {
        let engine = test_engine();
        let facts = vec![
            ExtractedFact {
                kind: FactKind::CodeFact {
                    kind: "code:contains-mod".into(),
                    name: "Vsa".into(),
                    detail: "encode".into(),
                },
                source_tool: "kg_query".into(),
                source_cycle: 1,
            },
            ExtractedFact {
                kind: FactKind::CodeFact {
                    kind: "code:contains-mod".into(),
                    name: "Vsa".into(),
                    detail: "ops".into(),
                },
                source_tool: "kg_query".into(),
                source_cycle: 1,
            },
            ExtractedFact {
                kind: FactKind::CodeFact {
                    kind: "code:defines-fn".into(),
                    name: "encode".into(),
                    detail: "encode_symbol".into(),
                },
                source_tool: "kg_query".into(),
                source_cycle: 2,
            },
            ExtractedFact {
                kind: FactKind::CodeFact {
                    kind: "code:defines-type".into(),
                    name: "ops".into(),
                    detail: "VsaOps".into(),
                },
                source_tool: "kg_query".into(),
                source_cycle: 2,
            },
            ExtractedFact {
                kind: FactKind::CodeFact {
                    kind: "code:defines-type".into(),
                    name: "Vsa".into(),
                    detail: "HyperVec".into(),
                },
                source_tool: "kg_query".into(),
                source_cycle: 1,
            },
            ExtractedFact {
                kind: FactKind::CodeFact {
                    kind: "code:depends-on".into(),
                    name: "Vsa".into(),
                    detail: "serde".into(),
                },
                source_tool: "kg_query".into(),
                source_cycle: 1,
            },
        ];

        let section = render_code_section(&facts, &engine);

        // Should contain hierarchical structure
        assert!(
            section.prose.contains("- **encode**"),
            "should have encode as child: {}",
            section.prose,
        );
        assert!(
            section.prose.contains("- **ops**"),
            "should have ops as child: {}",
            section.prose,
        );
        // Child details should be inline
        assert!(
            section.prose.contains("`encode_symbol`"),
            "encode child should show functions: {}",
            section.prose,
        );
        assert!(
            section.prose.contains("`VsaOps`"),
            "ops child should show types: {}",
            section.prose,
        );
        // Primary entity's own types/deps should be separate
        assert!(
            section.prose.contains("Key types:"),
            "primary should show own types: {}",
            section.prose,
        );
        assert!(
            section.prose.contains("serde"),
            "primary should show deps: {}",
            section.prose,
        );
    }

    #[test]
    fn derives_trait_predicates_handled() {
        assert!(is_code_predicate("code:derives-trait"));
        assert!(is_code_predicate("code:implements-trait"));
    }
}
