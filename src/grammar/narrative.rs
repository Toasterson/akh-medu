//! Narrative/Exploratory grammar archetype.
//!
//! Produces flowing, story-like prose for interactive sessions:
//! - Active voice with varied sentence openers
//! - Transition phrases between statements
//! - Engaging, exploratory tone
//!
//! ## Examples
//!
//! - Triple → "Dogs belong to the broader family of mammals."
//! - Gap → "An open question remains: what habitats do dogs prefer?"
//! - Similarity → "Interestingly, dogs share a close resemblance to wolves."

use std::sync::atomic::{AtomicUsize, Ordering};

use super::abs::{AbsTree, CompareOrd, Modality, ProvenanceTag, Quantifier, TemporalExpr};
use super::cat::Cat;
use super::concrete::{ConcreteGrammar, LinContext, ParseContext};
use super::discourse::{PointOfView, QueryFocus};
use super::error::GrammarResult;
use super::morpho;
use super::parser;

/// Transition phrases cycled for variety.
const TRANSITIONS: &[&str] = &[
    "",
    "Furthermore, ",
    "Notably, ",
    "Interestingly, ",
    "Additionally, ",
    "In turn, ",
    "Building on this, ",
    "Along similar lines, ",
];

/// Gap opening phrases cycled for variety.
const GAP_OPENERS: &[&str] = &[
    "An open question remains",
    "It remains unclear",
    "Further investigation is needed",
    "A gap in our knowledge exists",
];

/// The narrative/exploratory grammar archetype.
///
/// Uses an internal counter to cycle through transition phrases,
/// producing varied and engaging prose.
pub struct NarrativeGrammar {
    transition_counter: AtomicUsize,
}

impl Default for NarrativeGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl NarrativeGrammar {
    /// Create a new narrative grammar with the transition counter at zero.
    pub fn new() -> Self {
        Self {
            transition_counter: AtomicUsize::new(0),
        }
    }

    fn next_transition(&self) -> &'static str {
        let idx = self.transition_counter.fetch_add(1, Ordering::Relaxed);
        TRANSITIONS[idx % TRANSITIONS.len()]
    }

    fn next_gap_opener(&self) -> &'static str {
        let idx = self.transition_counter.load(Ordering::Relaxed);
        GAP_OPENERS[idx % GAP_OPENERS.len()]
    }

    fn linearize_inner(&self, tree: &AbsTree, ctx: &LinContext) -> GrammarResult<String> {
        match tree {
            AbsTree::EntityRef { label, symbol_id } => Ok(ctx.resolve_label(label, *symbol_id)),

            AbsTree::RelationRef { label, symbol_id } => {
                let resolved = ctx.resolve_label(label, *symbol_id);
                Ok(morpho::humanize_predicate(&resolved))
            }

            AbsTree::Freeform(text) => Ok(text.clone()),

            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => {
                let s = self.linearize_inner(subject, ctx)?;
                let p = self.linearize_inner(predicate, ctx)?;
                let o = self.linearize_inner(object, ctx)?;

                let transition = self.next_transition();
                Ok(format!("{transition}{s} {p} {o}."))
            }

            AbsTree::Similarity {
                entity,
                similar_to,
                score,
            } => {
                let e = self.linearize_inner(entity, ctx)?;
                let s = self.linearize_inner(similar_to, ctx)?;
                let transition = self.next_transition();
                let strength = if *score > 0.9 {
                    "a striking resemblance"
                } else if *score > 0.7 {
                    "a close resemblance"
                } else if *score > 0.5 {
                    "some similarity"
                } else {
                    "a faint resemblance"
                };
                Ok(format!("{transition}{e} shares {strength} to {s}."))
            }

            AbsTree::Gap {
                entity,
                description,
            } => {
                let e = self.linearize_inner(entity, ctx)?;
                let opener = self.next_gap_opener();
                Ok(format!("{opener}: regarding {e}, {description}."))
            }

            AbsTree::Inference {
                expression,
                simplified,
            } => {
                let transition = self.next_transition();
                Ok(format!(
                    "{transition}Through symbolic reasoning, `{expression}` reduces to `{simplified}`."
                ))
            }

            AbsTree::CodeFact { kind, name, detail } => {
                let transition = self.next_transition();
                Ok(format!(
                    "{transition}The {kind} `{name}` serves as {detail}."
                ))
            }

            AbsTree::CodeModule {
                name,
                role,
                importance,
                doc_summary,
                children,
            } => {
                let transition = self.next_transition();
                let desc = doc_summary
                    .as_deref()
                    .or(role.as_deref())
                    .unwrap_or("plays a role in the system");
                let key_note = if importance.unwrap_or(0.0) > 0.7 {
                    " It is a key component."
                } else {
                    ""
                };
                let mut out = format!("{transition}The **{name}** module {desc}.{key_note}");
                if !children.is_empty() {
                    out.push_str("\nIts contents include:\n");
                    for child in children {
                        let line = self.linearize_inner(child, ctx)?;
                        out.push_str("- ");
                        out.push_str(&line);
                        out.push('\n');
                    }
                }
                Ok(out)
            }

            AbsTree::CodeSignature {
                kind,
                name,
                doc_summary,
                params_or_fields,
                return_type,
                traits,
                ..
            } => {
                let transition = self.next_transition();
                let doc = doc_summary
                    .as_deref()
                    .map(|d| format!("{d}, "))
                    .unwrap_or_default();
                let params = if params_or_fields.is_empty() {
                    String::new()
                } else {
                    format!("takes {}, ", params_or_fields.join(", "))
                };
                let ret = return_type
                    .as_ref()
                    .map(|r| format!("returning `{r}`"))
                    .unwrap_or_default();
                let derives = if traits.is_empty() {
                    String::new()
                } else {
                    format!(", deriving {}", traits.join(", "))
                };
                Ok(format!(
                    "{transition}The {kind} `{name}` {doc}{params}{ret}{derives}."
                ))
            }

            AbsTree::DataFlow { steps } => {
                let transition = self.next_transition();
                let parts: Vec<String> = steps
                    .iter()
                    .map(|s| match &s.via_type {
                        Some(t) => format!("`{}` (via {t})", s.name),
                        None => format!("`{}`", s.name),
                    })
                    .collect();
                Ok(format!(
                    "{transition}Data flows through the system from {}.",
                    parts.join(" to ")
                ))
            }

            AbsTree::WithConfidence { inner, confidence } => {
                let text = self.linearize_inner(inner, ctx)?;
                let qualifier = if *confidence > 0.9 {
                    "with high confidence"
                } else if *confidence > 0.7 {
                    "with moderate confidence"
                } else if *confidence > 0.5 {
                    "tentatively"
                } else {
                    "speculatively"
                };
                // Insert qualifier before final period
                if let Some(stripped) = text.strip_suffix('.') {
                    Ok(format!("{stripped}, {qualifier}."))
                } else {
                    Ok(format!("{text}, {qualifier}"))
                }
            }

            AbsTree::WithProvenance { inner, tag } => {
                let text = self.linearize_inner(inner, ctx)?;
                let prov = format_provenance_narrative(tag);
                if let Some(stripped) = text.strip_suffix('.') {
                    Ok(format!("{stripped} ({prov})."))
                } else {
                    Ok(format!("{text} ({prov})"))
                }
            }

            AbsTree::Conjunction { items, is_and } => {
                let parts: Vec<String> = items
                    .iter()
                    .map(|i| self.linearize_inner(i, ctx))
                    .collect::<GrammarResult<Vec<_>>>()?;
                if *is_and {
                    Ok(parts.join(" "))
                } else {
                    let items_str: Vec<String> = parts
                        .iter()
                        .map(|p| {
                            if let Some(stripped) = p.strip_suffix('.') {
                                stripped.to_string()
                            } else {
                                p.clone()
                            }
                        })
                        .collect();
                    Ok(format!(
                        "Either {}, depending on the context.",
                        morpho::join_list(&items_str, "or")
                    ))
                }
            }

            AbsTree::Section { heading, body } => {
                let mut out = format!("## {heading}\n\n");
                // Reset transition counter at section boundary for variety
                self.transition_counter.store(0, Ordering::Relaxed);
                for item in body {
                    let line = self.linearize_inner(item, ctx)?;
                    out.push_str(&line);
                    if !line.ends_with('\n') {
                        out.push('\n');
                    }
                }
                Ok(out)
            }

            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => {
                let mut out = String::new();

                let overview_text = self.linearize_inner(overview, ctx)?;
                out.push_str(&overview_text);
                if !overview_text.ends_with('\n') {
                    out.push_str("\n\n");
                }

                for section in sections {
                    let section_text = self.linearize_inner(section, ctx)?;
                    out.push_str(&section_text);
                    if !section_text.ends_with('\n') {
                        out.push('\n');
                    }
                }

                if !gaps.is_empty() {
                    out.push_str("\n## Open Questions\n\n");
                    for gap in gaps {
                        let gap_text = self.linearize_inner(gap, ctx)?;
                        out.push_str("- ");
                        out.push_str(&gap_text);
                        if !gap_text.ends_with('\n') {
                            out.push('\n');
                        }
                    }
                }

                Ok(out)
            }

            AbsTree::DiscourseFrame { pov, focus, inner } => match pov {
                PointOfView::FirstPerson => self.linearize_first_person(inner, focus, ctx),
                PointOfView::SecondPerson => self.linearize_second_person(inner, ctx),
                PointOfView::ThirdPerson => self.linearize_inner(inner, ctx),
            },

            // ── NLU extensions ────────────────────────────────────────
            AbsTree::Negation { inner } => {
                let transition = self.next_transition();
                let text = self.linearize_inner(inner, ctx)?;
                Ok(format!("{transition}That is not the case: {text}"))
            }
            AbsTree::Quantified { quantifier, scope } => {
                let transition = self.next_transition();
                let text = self.linearize_inner(scope, ctx)?;
                match quantifier {
                    Quantifier::Universal => Ok(format!("{transition}In every case, {text}")),
                    Quantifier::Existential => {
                        Ok(format!("{transition}There are cases where {text}"))
                    }
                    Quantifier::Most => Ok(format!("{transition}In most cases, {text}")),
                    Quantifier::None => Ok(format!("{transition}In no case {text}")),
                    Quantifier::Specific(n) => {
                        Ok(format!("{transition}In {n} specific instances, {text}"))
                    }
                }
            }
            AbsTree::Comparison {
                entity_a,
                entity_b,
                property,
                ordering,
            } => {
                let transition = self.next_transition();
                let a = self.linearize_inner(entity_a, ctx)?;
                let b = self.linearize_inner(entity_b, ctx)?;
                match ordering {
                    CompareOrd::GreaterThan => {
                        Ok(format!("{transition}{a} is more {property} than {b}."))
                    }
                    CompareOrd::LessThan => {
                        Ok(format!("{transition}{a} is less {property} than {b}."))
                    }
                    CompareOrd::Equal => {
                        Ok(format!("{transition}{a} is equally {property} as {b}."))
                    }
                }
            }
            AbsTree::Conditional {
                condition,
                consequent,
            } => {
                let transition = self.next_transition();
                let cond = self.linearize_inner(condition, ctx)?;
                let cons = self.linearize_inner(consequent, ctx)?;
                Ok(format!("{transition}If {cond}, then {cons}."))
            }
            AbsTree::Temporal { time_expr, inner } => {
                let transition = self.next_transition();
                let text = self.linearize_inner(inner, ctx)?;
                let t = match time_expr {
                    TemporalExpr::Named(n) => n.clone(),
                    TemporalExpr::Recurring(r) => format!("on a recurring basis ({r})"),
                    TemporalExpr::Absolute(ts) => format!("at time {ts}"),
                    TemporalExpr::Relative(delta) if *delta > 0 => format!("in {delta} units"),
                    TemporalExpr::Relative(delta) => format!("{} units ago", delta.unsigned_abs()),
                };
                Ok(format!("{transition}{t}, {text}"))
            }
            AbsTree::Modal { modality, inner } => {
                let transition = self.next_transition();
                let text = self.linearize_inner(inner, ctx)?;
                let m = match modality {
                    Modality::Want => "wants to",
                    Modality::Can => "is able to",
                    Modality::Should => "should",
                    Modality::Must => "must",
                    Modality::May => "may",
                };
                Ok(format!("{transition}One {m} {text}."))
            }
            AbsTree::RelativeClause { head, clause } => {
                let h = self.linearize_inner(head, ctx)?;
                let c = self.linearize_inner(clause, ctx)?;
                Ok(format!("{h}, which {c}."))
            }
        }
    }

    /// Linearize inner content from a first-person perspective.
    ///
    /// Transforms each triple: "self is-a akh" → "I am an Akh"
    fn linearize_first_person(
        &self,
        tree: &AbsTree,
        _focus: &QueryFocus,
        ctx: &LinContext,
    ) -> GrammarResult<String> {
        match tree {
            AbsTree::Triple {
                predicate, object, ..
            } => {
                let pred_raw = self.linearize_inner(predicate, ctx)?;
                let obj_raw = self.linearize_inner(object, ctx)?;
                let obj_name = morpho::capitalize_entity(&obj_raw);

                let pred_1p = morpho::first_person_predicate(&pred_raw);

                // For "is a" predicates, insert the article if needed.
                if pred_1p.starts_with("am a ")
                    || pred_1p.starts_with("am an ")
                    || pred_1p == "am"
                {
                    if pred_1p == "am" {
                        Ok(format!("I am {obj_name}."))
                    } else {
                        // "am a" / "am an" — replace with correct article
                        let article = morpho::article(&obj_name);
                        Ok(format!("I am {article} {obj_name}."))
                    }
                } else {
                    Ok(format!("I {pred_1p} {obj_name}."))
                }
            }

            AbsTree::Conjunction { items, .. } => {
                let mut sentences = Vec::new();
                for item in items {
                    let s = self.linearize_first_person(item, _focus, ctx)?;
                    sentences.push(s);
                }
                Ok(join_first_person_sentences(&sentences))
            }

            AbsTree::WithConfidence { inner, confidence } => {
                let text = self.linearize_first_person(inner, _focus, ctx)?;
                let qualifier = if *confidence > 0.9 {
                    "with high confidence"
                } else if *confidence > 0.7 {
                    "with moderate confidence"
                } else if *confidence > 0.5 {
                    "tentatively"
                } else {
                    "speculatively"
                };
                if let Some(stripped) = text.strip_suffix('.') {
                    Ok(format!("{stripped}, {qualifier}."))
                } else {
                    Ok(format!("{text}, {qualifier}"))
                }
            }

            AbsTree::WithProvenance { inner, .. } => {
                self.linearize_first_person(inner, _focus, ctx)
            }

            // For non-triple nodes, fall back to standard linearization.
            _ => self.linearize_inner(tree, ctx),
        }
    }

    /// Linearize inner content from a second-person perspective ("You are...").
    fn linearize_second_person(&self, tree: &AbsTree, ctx: &LinContext) -> GrammarResult<String> {
        // For now, delegate to standard linearization.
        // Future: "You are..." transformations.
        self.linearize_inner(tree, ctx)
    }
}

/// Join first-person sentences into flowing prose.
///
/// Instead of "Furthermore, I am..." transitions, joins with commas
/// and proper sentence structure.
fn join_first_person_sentences(sentences: &[String]) -> String {
    if sentences.is_empty() {
        return String::new();
    }
    if sentences.len() == 1 {
        return sentences[0].clone();
    }

    // Collect the sentences. For identity focus, merge "I am X." sentences
    // into a flowing list: "I am X, powered by Y. I am capable of Z."
    let mut merged = Vec::new();
    let mut i_am_parts: Vec<String> = Vec::new();

    for sentence in sentences {
        let stripped = sentence.strip_suffix('.').unwrap_or(sentence);
        if stripped.starts_with("I am ") || stripped.starts_with("I have ") {
            i_am_parts.push(stripped.to_string());
        } else {
            // Non "I am" sentence — flush accumulated I am parts first.
            if !i_am_parts.is_empty() {
                merged.push(merge_i_am_clauses(&i_am_parts));
                i_am_parts.clear();
            }
            merged.push(sentence.clone());
        }
    }

    if !i_am_parts.is_empty() {
        merged.push(merge_i_am_clauses(&i_am_parts));
    }

    merged.join(" ")
}

/// Merge multiple "I am X" clauses into flowing prose.
///
/// "I am Akh" + "I am powered by Akh-Medu" → "I am Akh, powered by Akh-Medu."
fn merge_i_am_clauses(clauses: &[String]) -> String {
    if clauses.is_empty() {
        return String::new();
    }
    if clauses.len() == 1 {
        let c = &clauses[0];
        if c.ends_with('.') {
            return c.clone();
        }
        return format!("{c}.");
    }

    // First clause stays as-is; subsequent "I am" / "I have" clauses
    // become trailing phrases by stripping "I am " / "I have ".
    let first = &clauses[0];
    let mut parts = vec![first.clone()];

    for clause in &clauses[1..] {
        if let Some(rest) = clause.strip_prefix("I am ") {
            parts.push(rest.to_string());
        } else if let Some(rest) = clause.strip_prefix("I have ") {
            // Oxford comma join already adds "and" before the last item.
            parts.push(format!("have {rest}"));
        } else {
            parts.push(clause.clone());
        }
    }

    // Join with commas.
    if parts.len() == 2 {
        format!("{}, {}.", parts[0], parts[1])
    } else {
        let last = parts.last().unwrap();
        let rest = &parts[..parts.len() - 1];
        format!("{}, and {last}.", rest.join(", "))
    }
}

fn format_provenance_narrative(tag: &ProvenanceTag) -> String {
    match tag {
        ProvenanceTag::Extracted => "drawn from source material".to_string(),
        ProvenanceTag::GraphInferred => "inferred from the knowledge graph".to_string(),
        ProvenanceTag::VsaInferred { similarity } => {
            format!("suggested by vector similarity at {similarity:.0}%")
        }
        ProvenanceTag::Reasoned => "derived through symbolic reasoning".to_string(),
        ProvenanceTag::AgentDerived => "discovered by the agent".to_string(),
        ProvenanceTag::Enrichment => "identified through semantic analysis".to_string(),
        ProvenanceTag::UserAsserted => "as stated by the user".to_string(),
    }
}

impl ConcreteGrammar for NarrativeGrammar {
    fn name(&self) -> &str {
        "narrative"
    }

    fn description(&self) -> &str {
        "Flowing, story-like prose with varied transitions for interactive sessions"
    }

    fn linearize(&self, tree: &AbsTree, ctx: &LinContext) -> GrammarResult<String> {
        self.linearize_inner(tree, ctx)
    }

    fn parse(
        &self,
        input: &str,
        _expected_cat: Option<Cat>,
        ctx: &ParseContext,
    ) -> GrammarResult<AbsTree> {
        parser::parse_universal(input, ctx)
    }

    fn supported_categories(&self) -> &[Cat] {
        &[
            Cat::Entity,
            Cat::Relation,
            Cat::Statement,
            Cat::Similarity,
            Cat::Gap,
            Cat::Inference,
            Cat::CodeFact,
            Cat::CodeModule,
            Cat::CodeSignature,
            Cat::DataFlow,
            Cat::Confidence,
            Cat::Provenance,
            Cat::Conjunction,
            Cat::Section,
            Cat::Document,
            Cat::DiscourseFrame,
            Cat::Freeform,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linearize_triple_narrative() {
        let g = NarrativeGrammar::new();
        let ctx = LinContext::default();
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let result = g.linearize(&tree, &ctx).unwrap();
        // First call uses empty transition
        assert!(result.contains("Dog"));
        assert!(result.contains("Mammal"));
        assert!(result.ends_with('.'));
    }

    #[test]
    fn transitions_vary() {
        let g = NarrativeGrammar::new();
        let ctx = LinContext::default();
        let tree = AbsTree::triple(
            AbsTree::entity("A"),
            AbsTree::relation("r"),
            AbsTree::entity("B"),
        );
        let first = g.linearize(&tree, &ctx).unwrap();
        let second = g.linearize(&tree, &ctx).unwrap();
        // Second should have a different transition prefix
        assert_ne!(first, second);
    }

    #[test]
    fn linearize_gap_narrative() {
        let g = NarrativeGrammar::new();
        let ctx = LinContext::default();
        let tree = AbsTree::gap(AbsTree::entity("Dog"), "what habitats do dogs prefer");
        let result = g.linearize(&tree, &ctx).unwrap();
        assert!(result.contains("Dog"));
        assert!(result.contains("habitat"));
    }

    #[test]
    fn confidence_qualifiers() {
        let g = NarrativeGrammar::new();
        let ctx = LinContext::default();

        let high = AbsTree::triple_with_confidence(
            AbsTree::entity("A"),
            AbsTree::relation("r"),
            AbsTree::entity("B"),
            0.95,
        );
        let result = g.linearize(&high, &ctx).unwrap();
        assert!(result.contains("high confidence"));

        let low = AbsTree::triple_with_confidence(
            AbsTree::entity("A"),
            AbsTree::relation("r"),
            AbsTree::entity("B"),
            0.3,
        );
        let result = g.linearize(&low, &ctx).unwrap();
        assert!(result.contains("speculatively"));
    }

    #[test]
    fn similarity_strength_varies() {
        let g = NarrativeGrammar::new();
        let ctx = LinContext::default();

        let high = AbsTree::similarity(AbsTree::entity("A"), AbsTree::entity("B"), 0.95);
        let result = g.linearize(&high, &ctx).unwrap();
        assert!(result.contains("striking resemblance"));

        let low = AbsTree::similarity(AbsTree::entity("A"), AbsTree::entity("B"), 0.4);
        let result = g.linearize(&low, &ctx).unwrap();
        assert!(result.contains("faint resemblance"));
    }

    #[test]
    fn merge_i_am_clauses_no_double_and() {
        let clauses = vec![
            "I am Akh".to_string(),
            "I am powered by Akh-Medu".to_string(),
            "I have reasoning".to_string(),
        ];
        let result = merge_i_am_clauses(&clauses);
        // Must not produce "and and have" — the Oxford comma join adds "and"
        // before the last item, so the have-clause itself must not start with "and".
        assert!(
            !result.contains("and and"),
            "double conjunction in: {result}"
        );
        // The final form should be: "I am Akh, powered by Akh-Medu, and have reasoning."
        assert!(result.contains("have reasoning"), "missing have-clause in: {result}");
        assert!(result.contains("and have reasoning"), "Oxford comma join should produce 'and have': {result}");
    }
}
