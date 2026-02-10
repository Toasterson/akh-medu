//! Narrative synthesis: transforms working memory entries into human-readable prose.
//!
//! The pipeline:
//! 1. `extract_facts()` — parse WM ToolResult entries into structured facts
//! 2. `group_facts()` — group by topic (entity, code structure, similarity, etc.)
//! 3. `render_template()` — build a `NarrativeSummary` from grouped facts
//! 4. `polish_with_llm()` — optionally refine prose via Ollama (graceful fallback)

use std::collections::BTreeMap;

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

/// Code-related predicate prefixes that indicate code structure triples.
fn is_code_predicate(pred: &str) -> bool {
    pred.starts_with("code:")
        || pred == "defines-fn"
        || pred == "defines-type"
        || pred == "defines-mod"
        || pred == "depends-on"
        || pred == "implements"
        || pred == "contains"
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
    s.lines().all(|line| {
        let trimmed = line.trim();
        trimmed.is_empty() || is_metadata_label(trimmed)
    })
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

/// Try to parse "A -> B -> C" into a Triple fact.
fn parse_arrow_triple(line: &str) -> Option<FactKind> {
    let parts: Vec<&str> = line.split(" -> ").collect();
    if parts.len() >= 3 {
        let subject = parts[0].trim().to_string();
        let predicate = parts[1].trim().to_string();
        let object = parts[2..].join(" -> ").trim().to_string();
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
            || is_metadata_label(trimmed)
        {
            continue;
        }

        // Lines like "- entity: description" or "entity has no X"
        let cleaned = trimmed.strip_prefix("- ").unwrap_or(trimmed);
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
    _engine: &crate::engine::Engine,
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
                sections.push(render_code_section(&group.facts));
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
            "defines-fn" | "code:defines-fn" => format!("defines functions {obj_list}"),
            "defines-type" | "code:defines-type" => format!("defines types {obj_list}"),
            "defines-mod" | "code:defines-mod" => format!("defines modules {obj_list}"),
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

fn render_code_section(facts: &[ExtractedFact]) -> NarrativeSection {
    let mut modules: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut functions: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut types: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut deps: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for fact in facts {
        if let FactKind::CodeFact { kind, name, detail } = &fact.kind {
            match kind.as_str() {
                "code:defines-mod" | "defines-mod" | "contains" => {
                    modules
                        .entry(name.clone())
                        .or_default()
                        .push(detail.clone());
                }
                "code:defines-fn" | "defines-fn" => {
                    functions
                        .entry(name.clone())
                        .or_default()
                        .push(detail.clone());
                }
                "code:defines-type" | "defines-type" | "implements" => {
                    types
                        .entry(name.clone())
                        .or_default()
                        .push(detail.clone());
                }
                "depends-on" | "code:depends-on" => {
                    deps.entry(name.clone()).or_default().push(detail.clone());
                }
                _ => {
                    // Other code facts go under functions as a catch-all
                    functions
                        .entry(name.clone())
                        .or_default()
                        .push(format!("{kind}: {detail}"));
                }
            }
        }
    }

    let mut prose_parts = Vec::new();

    if !modules.is_empty() {
        for (parent, children) in &modules {
            let list = format_code_list(children);
            prose_parts.push(format!("The **{parent}** module contains {list}."));
        }
    }

    if !functions.is_empty() {
        for (owner, fns) in &functions {
            let list = format_code_list(fns);
            prose_parts.push(format!("**{owner}** defines functions {list}."));
        }
    }

    if !types.is_empty() {
        for (owner, ts) in &types {
            let list = format_code_list(ts);
            prose_parts.push(format!("**{owner}** defines types {list}."));
        }
    }

    if !deps.is_empty() {
        for (module, dep_list) in &deps {
            let list = format_list(dep_list);
            prose_parts.push(format!("**{module}** depends on {list}."));
        }
    }

    let prose = if prose_parts.is_empty() {
        "No code structure details found.".to_string()
    } else {
        prose_parts.join(" ")
    };

    NarrativeSection {
        heading: "Code Structure".to_string(),
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
                if !simplified.is_empty() && simplified != expression {
                    lines.push(format!("Simplified `{expression}` to `{simplified}`."));
                } else if !expression.is_empty() {
                    lines.push(format!("Analyzed expression: `{expression}`."));
                }
            }
            FactKind::Derivation { count, iterations } => {
                lines.push(format!(
                    "Derived {count} new fact{} in {iterations} iteration{}.",
                    if *count == 1 { "" } else { "s" },
                    if *iterations == 1 { "" } else { "s" },
                ));
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
        Preserve all facts and structure (headings, sections). \
        Do not add facts not present in the input. \
        Keep headings as ## Markdown headings. \
        Be conversational but accurate.";

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
        assert!(is_code_predicate("defines-fn"));
        assert!(is_code_predicate("depends-on"));
        assert!(!is_code_predicate("is-a"));
        assert!(!is_code_predicate("has"));
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
}
