//! Terse/Compact grammar archetype.
//!
//! Produces minimal prose with dense, symbol-heavy notation:
//! - Arrows and brackets instead of natural language
//! - Omits articles and filler words
//! - Confidence as bracketed decimal
//!
//! ## Examples
//!
//! - Triple → "Dog → is-a → Mammal [0.95]"
//! - Gap → "? Dog: no habitat data"
//! - Similarity → "Dog ~ Wolf (0.87)"

use super::abs::{AbsTree, ProvenanceTag};
use super::cat::Cat;
use super::concrete::{ConcreteGrammar, LinContext, ParseContext};
use super::error::GrammarResult;
use super::parser;

/// The terse/compact grammar archetype.
///
/// Produces dense, symbol-heavy notation optimized for
/// information density over readability.
pub struct TerseGrammar;

impl TerseGrammar {
    fn linearize_inner(&self, tree: &AbsTree, ctx: &LinContext) -> GrammarResult<String> {
        match tree {
            AbsTree::EntityRef { label, symbol_id } => {
                Ok(ctx.resolve_label(label, *symbol_id))
            }

            AbsTree::RelationRef { label, symbol_id } => {
                Ok(ctx.resolve_label(label, *symbol_id))
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
                Ok(format!("{s} → {p} → {o}"))
            }

            AbsTree::Similarity {
                entity,
                similar_to,
                score,
            } => {
                let e = self.linearize_inner(entity, ctx)?;
                let s = self.linearize_inner(similar_to, ctx)?;
                Ok(format!("{e} ~ {s} ({score:.2})"))
            }

            AbsTree::Gap { entity, description } => {
                let e = self.linearize_inner(entity, ctx)?;
                Ok(format!("? {e}: {description}"))
            }

            AbsTree::Inference {
                expression,
                simplified,
            } => Ok(format!("{expression} ⇒ {simplified}")),

            AbsTree::CodeFact { kind, name, detail } => {
                Ok(format!("{kind}:{name} — {detail}"))
            }

            AbsTree::WithConfidence { inner, confidence } => {
                let text = self.linearize_inner(inner, ctx)?;
                Ok(format!("{text} [{confidence:.2}]"))
            }

            AbsTree::WithProvenance { inner, tag } => {
                let text = self.linearize_inner(inner, ctx)?;
                let prov = format_provenance_terse(tag);
                Ok(format!("{text} ({prov})"))
            }

            AbsTree::Conjunction { items, is_and } => {
                let parts: Vec<String> = items
                    .iter()
                    .map(|i| self.linearize_inner(i, ctx))
                    .collect::<GrammarResult<Vec<_>>>()?;
                let sep = if *is_and { "; " } else { " | " };
                Ok(parts.join(sep))
            }

            AbsTree::Section { heading, body } => {
                let mut out = format!("── {heading} ──\n");
                for item in body {
                    let line = self.linearize_inner(item, ctx)?;
                    out.push_str("  ");
                    out.push_str(&line);
                    out.push('\n');
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
                out.push_str("\n\n");

                for section in sections {
                    let section_text = self.linearize_inner(section, ctx)?;
                    out.push_str(&section_text);
                    out.push('\n');
                }

                if !gaps.is_empty() {
                    out.push_str("── Gaps ──\n");
                    for gap in gaps {
                        let gap_text = self.linearize_inner(gap, ctx)?;
                        out.push_str("  ");
                        out.push_str(&gap_text);
                        out.push('\n');
                    }
                }

                Ok(out)
            }
        }
    }
}

fn format_provenance_terse(tag: &ProvenanceTag) -> String {
    match tag {
        ProvenanceTag::Extracted => "ext".to_string(),
        ProvenanceTag::GraphInferred => "graph".to_string(),
        ProvenanceTag::VsaInferred { similarity } => format!("vsa:{similarity:.2}"),
        ProvenanceTag::Reasoned => "reas".to_string(),
        ProvenanceTag::AgentDerived => "agent".to_string(),
        ProvenanceTag::Enrichment => "enrich".to_string(),
        ProvenanceTag::UserAsserted => "user".to_string(),
    }
}

impl ConcreteGrammar for TerseGrammar {
    fn name(&self) -> &str {
        "terse"
    }

    fn description(&self) -> &str {
        "Minimal, symbol-heavy notation optimized for information density"
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
        // Try terse-specific patterns first: "X → Y → Z" arrow notation
        if let Some(tree) = try_parse_arrow(input) {
            return Ok(tree);
        }
        // Fall back to the universal parser
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

/// Try to parse terse arrow notation: "X → Y → Z" or "X -> Y -> Z"
fn try_parse_arrow(input: &str) -> Option<AbsTree> {
    let trimmed = input.trim();

    // Try Unicode arrows first, then ASCII
    let parts: Vec<&str> = if trimmed.contains('→') {
        trimmed.split('→').map(str::trim).collect()
    } else if trimmed.contains("->") {
        trimmed.split("->").map(str::trim).collect()
    } else {
        return None;
    };

    if parts.len() != 3 {
        return None;
    }

    // Check for trailing confidence: "[0.95]"
    let (object_str, confidence) = parse_trailing_confidence(parts[2]);

    let subject = AbsTree::entity(parts[0]);
    let predicate = AbsTree::relation(parts[1]);
    let object = AbsTree::entity(object_str);

    let triple = AbsTree::triple(subject, predicate, object);

    if let Some(conf) = confidence {
        Some(AbsTree::WithConfidence {
            inner: Box::new(triple),
            confidence: conf,
        })
    } else {
        Some(triple)
    }
}

/// Parse trailing confidence like "[0.95]" from the end of a string.
fn parse_trailing_confidence(s: &str) -> (&str, Option<f32>) {
    let trimmed = s.trim();
    if let Some(bracket_start) = trimmed.rfind('[') {
        if trimmed.ends_with(']') {
            let inner = &trimmed[bracket_start + 1..trimmed.len() - 1];
            if let Ok(conf) = inner.parse::<f32>() {
                let before = trimmed[..bracket_start].trim();
                return (before, Some(conf));
            }
        }
    }
    (trimmed, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linearize_triple_arrow() {
        let g = TerseGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let result = g.linearize(&tree, &ctx).unwrap();
        assert_eq!(result, "Dog → is-a → Mammal");
    }

    #[test]
    fn linearize_with_confidence() {
        let g = TerseGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::triple_with_confidence(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
            0.95,
        );
        let result = g.linearize(&tree, &ctx).unwrap();
        assert_eq!(result, "Dog → is-a → Mammal [0.95]");
    }

    #[test]
    fn linearize_gap() {
        let g = TerseGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::gap(AbsTree::entity("Dog"), "no habitat data");
        let result = g.linearize(&tree, &ctx).unwrap();
        assert_eq!(result, "? Dog: no habitat data");
    }

    #[test]
    fn linearize_similarity() {
        let g = TerseGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::similarity(
            AbsTree::entity("Dog"),
            AbsTree::entity("Wolf"),
            0.87,
        );
        let result = g.linearize(&tree, &ctx).unwrap();
        assert_eq!(result, "Dog ~ Wolf (0.87)");
    }

    #[test]
    fn parse_arrow_notation() {
        let tree = try_parse_arrow("Dog → is-a → Mammal").unwrap();
        assert_eq!(tree.cat(), Cat::Statement);
    }

    #[test]
    fn parse_arrow_with_confidence() {
        let tree = try_parse_arrow("Dog → is-a → Mammal [0.95]").unwrap();
        assert_eq!(tree.cat(), Cat::Confidence);
    }

    #[test]
    fn parse_ascii_arrows() {
        let tree = try_parse_arrow("Dog -> is-a -> Mammal").unwrap();
        assert_eq!(tree.cat(), Cat::Statement);
    }

    #[test]
    fn linearize_conjunction_semicolons() {
        let g = TerseGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::and(vec![
            AbsTree::triple(
                AbsTree::entity("Dog"),
                AbsTree::relation("is-a"),
                AbsTree::entity("Mammal"),
            ),
            AbsTree::triple(
                AbsTree::entity("Cat"),
                AbsTree::relation("is-a"),
                AbsTree::entity("Mammal"),
            ),
        ]);
        let result = g.linearize(&tree, &ctx).unwrap();
        assert!(result.contains("; "));
    }

    #[test]
    fn linearize_inference() {
        let g = TerseGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::Inference {
            expression: "(bind x (bind x y))".to_string(),
            simplified: "y".to_string(),
        };
        let result = g.linearize(&tree, &ctx).unwrap();
        assert_eq!(result, "(bind x (bind x y)) ⇒ y");
    }
}
