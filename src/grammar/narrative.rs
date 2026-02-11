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

use super::abs::{AbsTree, ProvenanceTag};
use super::cat::Cat;
use super::concrete::{ConcreteGrammar, LinContext, ParseContext};
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
            AbsTree::EntityRef { label, symbol_id } => {
                Ok(ctx.resolve_label(label, *symbol_id))
            }

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

            AbsTree::Gap { entity, description } => {
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
                    let items_str: Vec<String> = parts.iter().map(|p| {
                        if let Some(stripped) = p.strip_suffix('.') {
                            stripped.to_string()
                        } else {
                            p.clone()
                        }
                    }).collect();
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
        }
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
            Cat::Confidence,
            Cat::Provenance,
            Cat::Conjunction,
            Cat::Section,
            Cat::Document,
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
}
