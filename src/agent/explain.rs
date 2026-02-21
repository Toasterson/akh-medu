//! Transparent reasoning: provenance-to-prose explanation pipeline (Phase 12f).
//!
//! Provides structured explanation queries (Why, How, WhatKnown, HowConfident,
//! WhatChanged) that walk the provenance ledger and render derivation chains
//! as human-readable prose or indented trees.

use std::collections::HashSet;
use std::sync::Arc;

use miette::Diagnostic;
use thiserror::Error;

use crate::engine::Engine;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;

// ── Errors ──────────────────────────────────────────────────────────────

/// Errors from the explanation pipeline.
#[derive(Debug, Error, Diagnostic)]
pub enum ExplainError {
    #[error("entity not found: \"{name}\"")]
    #[diagnostic(
        code(akh::explain::entity_not_found),
        help("The entity could not be resolved in the knowledge graph. Check the name.")
    )]
    EntityNotFound { name: String },

    #[error("no provenance records for symbol {symbol_id}")]
    #[diagnostic(
        code(akh::explain::no_provenance),
        help("This knowledge has no recorded derivation history. It may be seed data.")
    )]
    NoProvenance { symbol_id: u64 },

    #[error("provenance error: {message}")]
    #[diagnostic(
        code(akh::explain::provenance),
        help("An error occurred while querying the provenance ledger.")
    )]
    Provenance { message: String },
}

/// Convenience alias.
pub type ExplainResult<T> = std::result::Result<T, ExplainError>;

// ── ExplanationQuery ────────────────────────────────────────────────────

/// What the user is asking for an explanation of.
#[derive(Debug, Clone)]
pub enum ExplanationQuery {
    /// "Why X?" — trace provenance for entity/triple X.
    Why { subject: String },
    /// "How did you decide X?" — show the decision reasoning.
    How { subject: String },
    /// "What do you know about X?" — enumerate all triples with provenance.
    WhatKnown { subject: String },
    /// "How confident are you about X?" — confidence + evidence chain.
    HowConfident { subject: String },
    /// "What changed?" — diff KG state since last summary.
    WhatChanged,
}

impl ExplanationQuery {
    /// Try to parse an explanation query from user text.
    ///
    /// Returns `None` if the text doesn't match an explanation pattern.
    pub fn parse(input: &str) -> Option<Self> {
        let trimmed = input.trim();
        let lower = trimmed.to_lowercase();

        // "why X?" / "why is X?"
        if lower.starts_with("why ") {
            let rest = strip_question_prefix(&trimmed[4..]);
            if !rest.is_empty() {
                return Some(Self::Why {
                    subject: rest.to_string(),
                });
            }
        }

        // "how did you decide X?" / "how was X derived?"
        if lower.starts_with("how did you ")
            || lower.starts_with("how was ")
            || lower.starts_with("how is ")
        {
            let lowered = input.trim().to_lowercase();
            let rest = strip_how_prefix(&lowered);
            if !rest.is_empty() {
                return Some(Self::How {
                    subject: rest.to_string(),
                });
            }
        }

        // "what do you know about X?" / "what is known about X?"
        if lower.contains("know about ") || lower.contains("known about ") {
            if let Some(idx) = lower.find("about ") {
                let rest = strip_question_suffix(&trimmed[idx + 6..]);
                if !rest.is_empty() {
                    return Some(Self::WhatKnown {
                        subject: rest.to_string(),
                    });
                }
            }
        }

        // "how confident are you about X?" / "confidence of X?"
        if lower.contains("confident") || lower.starts_with("confidence ") {
            if let Some(idx) = lower.find("about ") {
                let rest = strip_question_suffix(&trimmed[idx + 6..]);
                if !rest.is_empty() {
                    return Some(Self::HowConfident {
                        subject: rest.to_string(),
                    });
                }
            }
            if lower.starts_with("confidence ") {
                let rest = strip_question_suffix(&trimmed[11..]);
                if !rest.is_empty() {
                    return Some(Self::HowConfident {
                        subject: rest.to_string(),
                    });
                }
            }
        }

        // "what changed?" / "what's new?" / "what changed since last time?"
        if lower.starts_with("what changed")
            || lower.starts_with("what's new")
            || lower.starts_with("what is new")
        {
            return Some(Self::WhatChanged);
        }

        // "explain X" — generic explanation request, treated as Why
        if lower.starts_with("explain ") {
            let rest = strip_question_suffix(&trimmed[8..]);
            if !rest.is_empty() {
                return Some(Self::Why {
                    subject: rest.to_string(),
                });
            }
        }

        None
    }
}

// ── DerivationNode ──────────────────────────────────────────────────────

/// A node in a derivation tree, built by walking provenance records.
#[derive(Debug, Clone)]
pub struct DerivationNode {
    /// The symbol this node represents.
    pub symbol_id: SymbolId,
    /// Human-readable label.
    pub label: String,
    /// How this symbol was derived.
    pub derivation: Option<DerivationKind>,
    /// Confidence score.
    pub confidence: f32,
    /// Depth in the derivation tree.
    pub depth: usize,
    /// Child nodes (sources of this derivation).
    pub children: Vec<DerivationNode>,
}

/// Build a derivation tree by recursively walking provenance records.
///
/// The `max_depth` parameter prevents infinite recursion on circular provenance chains.
pub fn build_derivation_tree(
    engine: &Arc<Engine>,
    symbol: SymbolId,
    max_depth: usize,
) -> ExplainResult<DerivationNode> {
    let mut visited = HashSet::new();
    build_node(engine, symbol, 0, max_depth, &mut visited)
}

fn build_node(
    engine: &Arc<Engine>,
    symbol: SymbolId,
    depth: usize,
    max_depth: usize,
    visited: &mut HashSet<u64>,
) -> ExplainResult<DerivationNode> {
    let label = engine.resolve_label(symbol);

    // Prevent cycles.
    if !visited.insert(symbol.get()) || depth > max_depth {
        return Ok(DerivationNode {
            symbol_id: symbol,
            label,
            derivation: None,
            confidence: 1.0,
            depth,
            children: Vec::new(),
        });
    }

    // Query provenance for this symbol.
    let records = engine.provenance_of(symbol).map_err(|e| ExplainError::Provenance {
        message: e.to_string(),
    })?;

    if records.is_empty() {
        return Ok(DerivationNode {
            symbol_id: symbol,
            label,
            derivation: None,
            confidence: 1.0,
            depth,
            children: Vec::new(),
        });
    }

    // Use the most recent (last) provenance record.
    let record = &records[records.len() - 1];

    // Recursively build children from sources.
    let mut children = Vec::new();
    for source in &record.sources {
        let child = build_node(engine, *source, depth + 1, max_depth, visited)?;
        children.push(child);
    }

    Ok(DerivationNode {
        symbol_id: symbol,
        label,
        derivation: Some(record.kind.clone()),
        confidence: record.confidence,
        depth,
        children,
    })
}

// ── Rendering ───────────────────────────────────────────────────────────

/// Render a derivation tree as an indented hierarchy.
///
/// ```text
/// [0.92] concept:rust is-a programming-language
///   └─ [0.95] Inferred via rule: transitivity from:
///        ├─ [1.00] concept:rust has-paradigm systems-programming (seed)
///        └─ [0.88] concept:systems-programming is-a programming-paradigm
///             └─ [0.88] Extracted from document ingest
/// ```
pub fn render_derivation_tree(node: &DerivationNode) -> String {
    let mut lines = Vec::new();
    render_node(node, &mut lines, "", true);
    lines.join("\n")
}

fn render_node(node: &DerivationNode, lines: &mut Vec<String>, prefix: &str, is_last: bool) {
    let conf = format!("[{:.2}]", node.confidence);
    let derivation_tag = node
        .derivation
        .as_ref()
        .map(|d| format!(" ({})", derivation_kind_prose(d)))
        .unwrap_or_default();

    if node.depth == 0 {
        lines.push(format!("{conf} {}{derivation_tag}", node.label));
    } else {
        let connector = if is_last { "└─ " } else { "├─ " };
        lines.push(format!("{prefix}{connector}{conf} {}{derivation_tag}", node.label));
    }

    let child_count = node.children.len();
    for (i, child) in node.children.iter().enumerate() {
        let is_child_last = i == child_count - 1;
        let new_prefix = if node.depth == 0 {
            "  ".to_string()
        } else if is_last {
            format!("{prefix}   ")
        } else {
            format!("{prefix}│  ")
        };
        render_node(child, lines, &new_prefix, is_child_last);
    }
}

/// Render a derivation tree as prose (for concise/normal detail levels).
pub fn render_derivation_prose(node: &DerivationNode) -> String {
    let mut parts = Vec::new();
    collect_prose(node, &mut parts, 0);
    parts.join(", ")
}

fn collect_prose(node: &DerivationNode, parts: &mut Vec<String>, depth: usize) {
    if depth == 0 {
        let tag = node
            .derivation
            .as_ref()
            .map(derivation_kind_prose)
            .unwrap_or_else(|| "asserted".to_string());
        parts.push(format!(
            "{} was {} (confidence: {:.2})",
            node.label, tag, node.confidence
        ));
    } else {
        let tag = node
            .derivation
            .as_ref()
            .map(derivation_kind_prose)
            .unwrap_or_else(|| "asserted".to_string());
        parts.push(format!("from {} ({})", node.label, tag));
    }

    for child in &node.children {
        collect_prose(child, parts, depth + 1);
    }
}

// ── Explanation Functions ───────────────────────────────────────────────

/// Explain why a particular entity has its current state in the KG.
///
/// Returns a prose explanation with a derivation tree.
pub fn explain_entity(engine: &Arc<Engine>, subject: &str) -> ExplainResult<String> {
    let sym_id = engine
        .resolve_symbol(subject)
        .map_err(|_| ExplainError::EntityNotFound {
            name: subject.to_string(),
        })?;

    // Collect triples involving this entity.
    let from_triples = engine.triples_from(sym_id);
    let to_triples = engine.triples_to(sym_id);

    if from_triples.is_empty() && to_triples.is_empty() {
        return Ok(format!("No knowledge found about \"{subject}\"."));
    }

    let tree = build_derivation_tree(engine, sym_id, 5)?;
    let rendered = render_derivation_tree(&tree);

    // Build triple summary.
    let mut triple_lines = Vec::new();
    for triple in from_triples.iter().chain(to_triples.iter()) {
        let subj = engine.resolve_label(triple.subject);
        let pred = engine.resolve_label(triple.predicate);
        let obj = engine.resolve_label(triple.object);

        // Skip metadata.
        if super::synthesize::is_metadata_label(&pred)
            || super::synthesize::is_metadata_label(&obj)
            || super::synthesize::is_metadata_label(&subj)
        {
            continue;
        }

        let prov_tag = collect_provenance_tag(engine, triple);
        triple_lines.push(format!(
            "  [{:.2}] {subj} {pred} {obj} ({prov_tag})",
            triple.confidence,
        ));
    }

    if triple_lines.is_empty() {
        return Ok(format!("No non-metadata knowledge found about \"{subject}\"."));
    }

    let mut result = format!("Explanation for \"{subject}\":\n\n");
    result.push_str("Derivation:\n");
    result.push_str(&rendered);
    result.push_str("\n\nKnown facts:\n");
    result.push_str(&triple_lines.join("\n"));

    Ok(result)
}

/// Explain the confidence level for an entity.
pub fn explain_confidence(engine: &Arc<Engine>, subject: &str) -> ExplainResult<String> {
    let sym_id = engine
        .resolve_symbol(subject)
        .map_err(|_| ExplainError::EntityNotFound {
            name: subject.to_string(),
        })?;

    let from_triples = engine.triples_from(sym_id);
    let to_triples = engine.triples_to(sym_id);

    let all_triples: Vec<_> = from_triples
        .iter()
        .chain(to_triples.iter())
        .filter(|t| {
            let pred = engine.resolve_label(t.predicate);
            !super::synthesize::is_metadata_label(&pred)
        })
        .collect();

    if all_triples.is_empty() {
        return Ok(format!("No knowledge found about \"{subject}\" to assess confidence."));
    }

    // Compute aggregate confidence.
    let sum: f32 = all_triples.iter().map(|t| t.confidence).sum();
    let avg = sum / all_triples.len() as f32;
    let min = all_triples
        .iter()
        .map(|t| t.confidence)
        .fold(f32::INFINITY, f32::min);
    let max = all_triples
        .iter()
        .map(|t| t.confidence)
        .fold(f32::NEG_INFINITY, f32::max);

    // Categorize confidence level.
    let assessment = if avg >= 0.95 {
        "Very high confidence — well-established knowledge"
    } else if avg >= 0.8 {
        "High confidence — strong evidence"
    } else if avg >= 0.6 {
        "Moderate confidence — some uncertainty remains"
    } else if avg >= 0.4 {
        "Low confidence — limited evidence"
    } else {
        "Very low confidence — highly uncertain"
    };

    // Show provenance breakdown.
    let mut provenance_counts: Vec<(String, usize)> = Vec::new();
    for triple in &all_triples {
        let tag = collect_provenance_tag(engine, triple);
        if let Some(entry) = provenance_counts.iter_mut().find(|(t, _)| t == &tag) {
            entry.1 += 1;
        } else {
            provenance_counts.push((tag, 1));
        }
    }

    let mut result = format!("Confidence assessment for \"{subject}\":\n\n");
    result.push_str(&format!("  Average confidence: {avg:.2}\n"));
    result.push_str(&format!("  Range: [{min:.2}, {max:.2}]\n"));
    result.push_str(&format!("  Based on {} fact(s)\n", all_triples.len()));
    result.push_str(&format!("  Assessment: {assessment}\n"));

    if !provenance_counts.is_empty() {
        result.push_str("\nEvidence sources:\n");
        for (tag, count) in &provenance_counts {
            result.push_str(&format!("  - {tag}: {count} fact(s)\n"));
        }
    }

    Ok(result)
}

/// List all known facts about an entity with provenance.
pub fn explain_known(engine: &Arc<Engine>, subject: &str) -> ExplainResult<String> {
    let sym_id = engine
        .resolve_symbol(subject)
        .map_err(|_| ExplainError::EntityNotFound {
            name: subject.to_string(),
        })?;

    let from_triples = engine.triples_from(sym_id);
    let to_triples = engine.triples_to(sym_id);

    let mut lines = Vec::new();
    let mut fact_count = 0;

    for triple in from_triples.iter().chain(to_triples.iter()) {
        let subj = engine.resolve_label(triple.subject);
        let pred = engine.resolve_label(triple.predicate);
        let obj = engine.resolve_label(triple.object);

        if super::synthesize::is_metadata_label(&pred)
            || super::synthesize::is_metadata_label(&obj)
            || super::synthesize::is_metadata_label(&subj)
        {
            continue;
        }

        let prov_tag = collect_provenance_tag(engine, triple);
        lines.push(format!(
            "  [{:.2}] {subj} {pred} {obj} — {prov_tag}",
            triple.confidence,
        ));
        fact_count += 1;
    }

    if lines.is_empty() {
        return Ok(format!("No non-metadata knowledge found about \"{subject}\"."));
    }

    let mut result = format!("Known facts about \"{subject}\" ({fact_count} total):\n\n");
    result.push_str(&lines.join("\n"));

    Ok(result)
}

/// Report what changed in the KG since a given timestamp.
///
/// `since_timestamp` is seconds since UNIX epoch. If `None`, reports all
/// knowledge (useful for first-run).
pub fn explain_changes(
    engine: &Arc<Engine>,
    since_timestamp: Option<u64>,
) -> ExplainResult<String> {
    let all = engine.all_triples();
    let threshold = since_timestamp.unwrap_or(0);

    let changed: Vec<_> = all
        .iter()
        .filter(|t| t.timestamp > threshold)
        .collect();

    if changed.is_empty() {
        return Ok("No changes since last check.".to_string());
    }

    let mut lines = Vec::new();
    for triple in &changed {
        let subj = engine.resolve_label(triple.subject);
        let pred = engine.resolve_label(triple.predicate);
        let obj = engine.resolve_label(triple.object);

        if super::synthesize::is_metadata_label(&pred)
            || super::synthesize::is_metadata_label(&obj)
            || super::synthesize::is_metadata_label(&subj)
        {
            continue;
        }

        let prov_tag = collect_provenance_tag(engine, triple);
        lines.push(format!(
            "  [{:.2}] {subj} {pred} {obj} — {prov_tag}",
            triple.confidence,
        ));
    }

    if lines.is_empty() {
        return Ok("No non-metadata changes since last check.".to_string());
    }

    let mut result = format!("{} change(s) since last check:\n\n", lines.len());
    result.push_str(&lines.join("\n"));

    Ok(result)
}

/// Execute an explanation query and return the prose result.
pub fn execute_query(
    engine: &Arc<Engine>,
    query: &ExplanationQuery,
    since_timestamp: Option<u64>,
) -> ExplainResult<String> {
    match query {
        ExplanationQuery::Why { subject } | ExplanationQuery::How { subject } => {
            explain_entity(engine, subject)
        }
        ExplanationQuery::WhatKnown { subject } => explain_known(engine, subject),
        ExplanationQuery::HowConfident { subject } => explain_confidence(engine, subject),
        ExplanationQuery::WhatChanged => explain_changes(engine, since_timestamp),
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Collect a human-readable provenance tag for a triple.
fn collect_provenance_tag(engine: &Engine, triple: &crate::graph::Triple) -> String {
    if let Some(prov_id) = triple.provenance_id {
        if let Some(id) = SymbolId::new(prov_id) {
            if let Ok(records) = engine.provenance_of(id) {
                if let Some(rec) = records.first() {
                    return derivation_kind_prose(&rec.kind);
                }
            }
        }
    }

    if triple.confidence >= 0.99 {
        "asserted".to_string()
    } else if triple.confidence >= 0.8 {
        "high-confidence inference".to_string()
    } else {
        "inferred".to_string()
    }
}

/// Convert a `DerivationKind` to a short human-readable prose string.
pub fn derivation_kind_prose(kind: &DerivationKind) -> String {
    match kind {
        DerivationKind::Extracted => "extracted from source material".to_string(),
        DerivationKind::Seed => "seed knowledge (asserted)".to_string(),
        DerivationKind::Reasoned => "derived via symbolic reasoning".to_string(),
        DerivationKind::Aggregated => "aggregated from multiple sources".to_string(),
        DerivationKind::GraphEdge { .. } => "inferred via graph edge traversal".to_string(),
        DerivationKind::VsaRecovery { similarity, .. } => {
            format!("recovered via VSA (similarity: {similarity:.2})")
        }
        DerivationKind::Analogy { .. } => "inferred via analogy".to_string(),
        DerivationKind::FillerRecovery { .. } => "recovered as role filler via unbind".to_string(),
        DerivationKind::RuleInference { rule_name, .. } => {
            format!("inferred via rule: {rule_name}")
        }
        DerivationKind::FusedInference {
            path_count,
            interference_signal,
            ..
        } => {
            format!(
                "fused from {path_count} inference path(s) (interference: {interference_signal:.2})"
            )
        }
        DerivationKind::GapIdentified { gap_kind, .. } => {
            format!("identified as knowledge gap: {gap_kind}")
        }
        DerivationKind::AgentDecision { cycle, .. } => {
            format!("agent decision in OODA cycle {cycle}")
        }
        DerivationKind::AgentConsolidation { reason, .. } => {
            format!("consolidated from working memory: {reason}")
        }
        DerivationKind::SchemaDiscovered { pattern_type } => {
            format!("discovered schema pattern: {pattern_type}")
        }
        DerivationKind::SemanticEnrichment { source } => {
            format!("semantic enrichment: {source}")
        }
        DerivationKind::CompartmentLoaded { source_file, .. } => {
            format!("loaded from compartment: {source_file}")
        }
        DerivationKind::ShadowVeto { pattern_name, .. } => {
            format!("shadow veto: {pattern_name}")
        }
        DerivationKind::PsycheEvolution { trigger, .. } => {
            format!("psyche evolution: {trigger}")
        }
        DerivationKind::WasmToolExecution { tool_name, .. } => {
            format!("WASM tool execution: {tool_name}")
        }
        DerivationKind::CliToolExecution { tool_name, .. } => {
            format!("CLI tool execution: {tool_name}")
        }
        DerivationKind::DocumentIngested { format, .. } => {
            format!("ingested from document ({format})")
        }
        DerivationKind::ConceptExtracted {
            extraction_method, ..
        } => {
            format!("concept extraction: {extraction_method}")
        }
        DerivationKind::ContextInheritance { .. } => {
            "inherited from ancestor context".to_string()
        }
        DerivationKind::ContextLifting { condition, .. } => {
            format!("lifted between contexts: {condition}")
        }
        DerivationKind::PredicateGeneralization { .. } => {
            "inferred via predicate generalization".to_string()
        }
        DerivationKind::PredicateInverse { .. } => {
            "inferred via predicate inverse".to_string()
        }
        DerivationKind::DefeasibleOverride { reason, .. } => {
            format!("defeasible override: {reason}")
        }
        DerivationKind::DispatchRoute { reasoner, .. } => {
            format!("dispatched to {reasoner}")
        }
        DerivationKind::ArgumentVerdict { decisive_rule, .. } => {
            format!("argumentation verdict: {decisive_rule}")
        }
        DerivationKind::RuleMacroExpansion { macro_name, .. } => {
            format!("expanded from macro: {macro_name}")
        }
        DerivationKind::TemporalDecay {
            original_confidence,
            decayed_confidence,
            ..
        } => {
            format!(
                "temporal decay: {original_confidence:.2} → {decayed_confidence:.2}"
            )
        }
        DerivationKind::ContradictionDetected { kind, .. } => {
            format!("contradiction detected: {kind}")
        }
        DerivationKind::SkolemWitness { .. } => {
            "Skolem witness (existential)".to_string()
        }
        DerivationKind::SkolemGrounding { .. } => {
            "Skolem grounded to concrete entity".to_string()
        }
        DerivationKind::CwaQuery { .. } => {
            "closed-world assumption query".to_string()
        }
        DerivationKind::SecondOrderInstantiation { rule_name, .. } => {
            format!("second-order rule: {rule_name}")
        }
        DerivationKind::NartCreation { arg_count, .. } => {
            format!("NART creation ({arg_count} args)")
        }
        DerivationKind::CodeGenerated { scope, .. } => {
            format!("code generated: {scope}")
        }
        DerivationKind::CodeRefinement { attempt, .. } => {
            format!("code refined (attempt {attempt})")
        }
        DerivationKind::LibraryLearning { pattern_name, .. } => {
            format!("library learning: {pattern_name}")
        }
        DerivationKind::AutonomousGoalGeneration { drive, .. } => {
            format!("autonomous goal from {drive} drive")
        }
        DerivationKind::HtnDecomposition { method_name, .. } => {
            format!("HTN decomposition: {method_name}")
        }
        DerivationKind::PriorityArgumentation { .. } => {
            "priority re-evaluated via argumentation".to_string()
        }
        DerivationKind::ProjectCreated { name } => {
            format!("project created: {name}")
        }
        DerivationKind::WatchFired { watch_id, .. } => {
            format!("watch fired: {watch_id}")
        }
        DerivationKind::MetacognitiveEvaluation { signal, .. } => {
            format!("metacognitive evaluation: {signal}")
        }
        DerivationKind::ResourceAssessment { voc, .. } => {
            format!("resource assessment (VOC: {voc:.2})")
        }
        DerivationKind::ProceduralLearning { method_name, .. } => {
            format!("procedural learning: {method_name}")
        }
    }
}

/// Strip trailing question mark and whitespace.
fn strip_question_suffix(s: &str) -> &str {
    s.trim().trim_end_matches('?').trim()
}

/// Strip "is ", "are ", "was ", "were " prefix from a why-question remainder,
/// and also strip a trailing question mark.
fn strip_question_prefix(s: &str) -> &str {
    let trimmed = strip_question_suffix(s);
    let lower = trimmed.to_lowercase();
    if lower.starts_with("is ") {
        trimmed[3..].trim()
    } else if lower.starts_with("are ") {
        trimmed[4..].trim()
    } else if lower.starts_with("was ") {
        trimmed[4..].trim()
    } else if lower.starts_with("were ") {
        trimmed[5..].trim()
    } else {
        trimmed
    }
}

/// Strip "how did you decide ", "how was ", "how is " prefix (and optional
/// leading "how ").
fn strip_how_prefix(s: &str) -> &str {
    let s = strip_question_suffix(s);
    // Strip leading "how " if present.
    let s = if s.starts_with("how ") { &s[4..] } else { s };
    if s.starts_with("did you decide ") {
        s[15..].trim()
    } else if s.starts_with("did you ") {
        s[8..].trim()
    } else if s.starts_with("was ") {
        s[4..].trim()
    } else if s.starts_with("is ") {
        s[3..].trim()
    } else {
        s
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ExplanationQuery::parse ──────────────────────────────────────

    #[test]
    fn parse_why_query() {
        let q = ExplanationQuery::parse("why is Rust a programming language?").unwrap();
        assert!(matches!(q, ExplanationQuery::Why { ref subject } if subject == "Rust a programming language"));
    }

    #[test]
    fn parse_why_simple() {
        let q = ExplanationQuery::parse("why dogs?").unwrap();
        assert!(matches!(q, ExplanationQuery::Why { ref subject } if subject == "dogs"));
    }

    #[test]
    fn parse_how_query() {
        let q = ExplanationQuery::parse("how did you decide Rust?").unwrap();
        assert!(matches!(q, ExplanationQuery::How { ref subject } if subject == "rust"));
    }

    #[test]
    fn parse_how_was() {
        let q = ExplanationQuery::parse("how was X derived?").unwrap();
        assert!(matches!(q, ExplanationQuery::How { ref subject } if subject == "x derived"));
    }

    #[test]
    fn parse_what_known() {
        let q = ExplanationQuery::parse("what do you know about dogs?").unwrap();
        assert!(matches!(q, ExplanationQuery::WhatKnown { ref subject } if subject == "dogs"));
    }

    #[test]
    fn parse_what_is_known() {
        let q = ExplanationQuery::parse("what is known about cats?").unwrap();
        assert!(matches!(q, ExplanationQuery::WhatKnown { ref subject } if subject == "cats"));
    }

    #[test]
    fn parse_how_confident() {
        let q = ExplanationQuery::parse("how confident are you about Rust?").unwrap();
        assert!(matches!(q, ExplanationQuery::HowConfident { ref subject } if subject == "Rust"));
    }

    #[test]
    fn parse_confidence_shorthand() {
        let q = ExplanationQuery::parse("confidence Rust").unwrap();
        assert!(matches!(q, ExplanationQuery::HowConfident { ref subject } if subject == "Rust"));
    }

    #[test]
    fn parse_what_changed() {
        let q = ExplanationQuery::parse("what changed?").unwrap();
        assert!(matches!(q, ExplanationQuery::WhatChanged));
    }

    #[test]
    fn parse_whats_new() {
        let q = ExplanationQuery::parse("what's new?").unwrap();
        assert!(matches!(q, ExplanationQuery::WhatChanged));
    }

    #[test]
    fn parse_explain_shorthand() {
        let q = ExplanationQuery::parse("explain dogs").unwrap();
        assert!(matches!(q, ExplanationQuery::Why { ref subject } if subject == "dogs"));
    }

    #[test]
    fn parse_non_explanation() {
        assert!(ExplanationQuery::parse("hello world").is_none());
        assert!(ExplanationQuery::parse("find dogs").is_none());
        assert!(ExplanationQuery::parse("status").is_none());
    }

    // ── DerivationNode rendering ─────────────────────────────────────

    #[test]
    fn render_leaf_node() {
        let node = DerivationNode {
            symbol_id: SymbolId::new(1).unwrap(),
            label: "concept:rust".to_string(),
            derivation: Some(DerivationKind::Seed),
            confidence: 1.0,
            depth: 0,
            children: Vec::new(),
        };
        let rendered = render_derivation_tree(&node);
        assert!(rendered.contains("[1.00]"));
        assert!(rendered.contains("concept:rust"));
        assert!(rendered.contains("seed knowledge"));
    }

    #[test]
    fn render_tree_with_children() {
        let child1 = DerivationNode {
            symbol_id: SymbolId::new(2).unwrap(),
            label: "concept:systems-programming".to_string(),
            derivation: Some(DerivationKind::Seed),
            confidence: 1.0,
            depth: 1,
            children: Vec::new(),
        };
        let child2 = DerivationNode {
            symbol_id: SymbolId::new(3).unwrap(),
            label: "concept:memory-safety".to_string(),
            derivation: Some(DerivationKind::Extracted),
            confidence: 0.88,
            depth: 1,
            children: Vec::new(),
        };
        let root = DerivationNode {
            symbol_id: SymbolId::new(1).unwrap(),
            label: "concept:rust".to_string(),
            derivation: Some(DerivationKind::Reasoned),
            confidence: 0.92,
            depth: 0,
            children: vec![child1, child2],
        };

        let rendered = render_derivation_tree(&root);
        assert!(rendered.contains("├─"));
        assert!(rendered.contains("└─"));
        assert!(rendered.contains("concept:rust"));
        assert!(rendered.contains("concept:systems-programming"));
        assert!(rendered.contains("concept:memory-safety"));
    }

    // ── Prose rendering ──────────────────────────────────────────────

    #[test]
    fn render_prose_leaf() {
        let node = DerivationNode {
            symbol_id: SymbolId::new(1).unwrap(),
            label: "concept:rust".to_string(),
            derivation: Some(DerivationKind::Seed),
            confidence: 1.0,
            depth: 0,
            children: Vec::new(),
        };
        let prose = render_derivation_prose(&node);
        assert!(prose.contains("concept:rust"));
        assert!(prose.contains("seed knowledge"));
        assert!(prose.contains("1.00"));
    }

    // ── derivation_kind_prose ────────────────────────────────────────

    #[test]
    fn derivation_prose_coverage() {
        // Test a selection of kinds to ensure the match arms work.
        assert!(derivation_kind_prose(&DerivationKind::Seed).contains("seed"));
        assert!(derivation_kind_prose(&DerivationKind::Extracted).contains("extracted"));
        assert!(derivation_kind_prose(&DerivationKind::Reasoned).contains("reasoning"));
        assert!(derivation_kind_prose(&DerivationKind::Aggregated).contains("aggregated"));
        assert!(derivation_kind_prose(&DerivationKind::RuleInference {
            rule_name: "transitivity".to_string(),
            antecedents: vec![],
        })
        .contains("transitivity"));
    }

    // ── Helper tests ─────────────────────────────────────────────────

    #[test]
    fn strip_question_suffix_works() {
        assert_eq!(strip_question_suffix("dogs?"), "dogs");
        assert_eq!(strip_question_suffix("dogs"), "dogs");
        assert_eq!(strip_question_suffix("dogs? "), "dogs");
    }

    #[test]
    fn strip_question_prefix_works() {
        assert_eq!(strip_question_prefix("is Rust good?"), "Rust good");
        assert_eq!(strip_question_prefix("are dogs mammals?"), "dogs mammals");
        assert_eq!(strip_question_prefix("dogs?"), "dogs");
    }
}
