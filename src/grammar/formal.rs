//! Formal/Technical grammar archetype.
//!
//! Produces precise, structured, academic-style output:
//! - Passive/hedging voice
//! - Explicit provenance and confidence
//! - Section-structured documents
//!
//! ## Examples
//!
//! - Triple → "The entity 'Dog' is classified as a subtype of 'Mammal' (confidence: 0.95)."
//! - Gap → "Knowledge gap identified for 'Dog': no habitat data available."
//! - Similarity → "'Dog' exhibits similarity to 'Wolf' (score: 0.87)."

use super::abs::{AbsTree, ProvenanceTag};
use super::cat::Cat;
use super::concrete::{ConcreteGrammar, LinContext, ParseContext};
use super::error::GrammarResult;
use super::morpho;
use super::parser;

/// The formal/technical grammar archetype.
///
/// Produces precise, structured prose with explicit confidence
/// and provenance annotations.
pub struct FormalGrammar;

impl FormalGrammar {
    fn linearize_inner(&self, tree: &AbsTree, ctx: &LinContext) -> GrammarResult<String> {
        match tree {
            AbsTree::EntityRef { label, symbol_id } => {
                let resolved = ctx.resolve_label(label, *symbol_id);
                Ok(format!("'{resolved}'"))
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

                // Formal style: "The entity X <predicate> Y."
                Ok(format!("The entity {s} {p} {o}."))
            }

            AbsTree::Similarity {
                entity,
                similar_to,
                score,
            } => {
                let e = self.linearize_inner(entity, ctx)?;
                let s = self.linearize_inner(similar_to, ctx)?;
                Ok(format!(
                    "{e} exhibits similarity to {s} (score: {score:.2})."
                ))
            }

            AbsTree::Gap {
                entity,
                description,
            } => {
                let e = self.linearize_inner(entity, ctx)?;
                Ok(format!("Knowledge gap identified for {e}: {description}."))
            }

            AbsTree::Inference {
                expression,
                simplified,
            } => Ok(format!(
                "Reasoning result: the expression `{expression}` simplifies to `{simplified}`."
            )),

            AbsTree::CodeFact { kind, name, detail } => {
                Ok(format!("Code structure: {kind} `{name}` — {detail}."))
            }

            AbsTree::CodeModule {
                name,
                role,
                importance,
                doc_summary,
                children,
            } => {
                let desc = doc_summary
                    .as_deref()
                    .or(role.as_deref())
                    .unwrap_or("serves an unspecified role");
                let imp_tag = if importance.unwrap_or(0.0) > 0.7 {
                    format!(" (importance: {:.2})", importance.unwrap())
                } else {
                    String::new()
                };
                let mut out = format!("The module `{name}` {desc}.{imp_tag}");
                if !children.is_empty() {
                    out.push_str(&format!("\nContains {} items:\n", children.len()));
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
                importance,
            } => {
                let star = if importance.unwrap_or(0.0) > 0.7 {
                    "\u{2605} "
                } else {
                    ""
                };
                let doc = doc_summary.as_deref().unwrap_or("no documentation");
                let params = if params_or_fields.is_empty() {
                    String::new()
                } else {
                    format!(" params: ({}),", params_or_fields.join(", "))
                };
                let ret = return_type
                    .as_ref()
                    .map(|r| format!(" returns `{r}`."))
                    .unwrap_or_else(|| ".".to_string());
                let derives = if traits.is_empty() {
                    String::new()
                } else {
                    format!(" [derives: {}]", traits.join(", "))
                };
                Ok(format!(
                    "{star}{kind} `{name}` \u{2014} {doc}.{params}{ret}{derives}"
                ))
            }

            AbsTree::DataFlow { steps } => {
                let parts: Vec<String> = steps
                    .iter()
                    .map(|s| match &s.via_type {
                        Some(t) => format!("`{}` \u{2192} {t}", s.name),
                        None => format!("`{}`", s.name),
                    })
                    .collect();
                Ok(format!("Data flow: {}", parts.join(" \u{2192} ")))
            }

            AbsTree::WithConfidence { inner, confidence } => {
                let text = self.linearize_inner(inner, ctx)?;
                // Insert confidence before the final period
                if let Some(stripped) = text.strip_suffix('.') {
                    Ok(format!("{stripped} (confidence: {confidence:.2})."))
                } else {
                    Ok(format!("{text} (confidence: {confidence:.2})"))
                }
            }

            AbsTree::WithProvenance { inner, tag } => {
                let text = self.linearize_inner(inner, ctx)?;
                let prov = format_provenance_formal(tag);
                if let Some(stripped) = text.strip_suffix('.') {
                    Ok(format!("{stripped} [{prov}]."))
                } else {
                    Ok(format!("{text} [{prov}]"))
                }
            }

            AbsTree::Conjunction { items, is_and } => {
                let parts: Vec<String> = items
                    .iter()
                    .map(|i| self.linearize_inner(i, ctx))
                    .collect::<GrammarResult<Vec<_>>>()?;
                let conj = if *is_and { "and" } else { "or" };
                Ok(morpho::join_list(&parts, conj))
            }

            AbsTree::Section { heading, body } => {
                let mut out = format!("## {heading}\n\n");
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
                    out.push_str("\n## Knowledge Gaps\n\n");
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

            // Discourse frames: delegate to inner content (formal doesn't
            // do POV transformation — that's narrative's job).
            AbsTree::DiscourseFrame { inner, .. } => self.linearize_inner(inner, ctx),
        }
    }
}

fn format_provenance_formal(tag: &ProvenanceTag) -> String {
    match tag {
        ProvenanceTag::Extracted => "source: extracted".to_string(),
        ProvenanceTag::GraphInferred => "source: graph inference".to_string(),
        ProvenanceTag::VsaInferred { similarity } => {
            format!("source: VSA inference (similarity: {similarity:.2})")
        }
        ProvenanceTag::Reasoned => "source: symbolic reasoning".to_string(),
        ProvenanceTag::AgentDerived => "source: agent derivation".to_string(),
        ProvenanceTag::Enrichment => "source: semantic enrichment".to_string(),
        ProvenanceTag::UserAsserted => "source: user assertion".to_string(),
    }
}

impl ConcreteGrammar for FormalGrammar {
    fn name(&self) -> &str {
        "formal"
    }

    fn description(&self) -> &str {
        "Precise, structured, academic-style output with explicit confidence and provenance"
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
        // Delegate to the shared parser — the formal grammar doesn't have
        // its own special parse patterns beyond the universal ones.
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
            Cat::Freeform,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linearize_triple() {
        let g = FormalGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let result = g.linearize(&tree, &ctx).unwrap();
        assert_eq!(result, "The entity 'Dog' is a 'Mammal'.");
    }

    #[test]
    fn linearize_triple_with_confidence() {
        let g = FormalGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::triple_with_confidence(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
            0.95,
        );
        let result = g.linearize(&tree, &ctx).unwrap();
        assert!(result.contains("confidence: 0.95"));
    }

    #[test]
    fn linearize_gap() {
        let g = FormalGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::gap(AbsTree::entity("Dog"), "no habitat data available");
        let result = g.linearize(&tree, &ctx).unwrap();
        assert!(result.contains("Knowledge gap identified for 'Dog'"));
        assert!(result.contains("no habitat data available"));
    }

    #[test]
    fn linearize_similarity() {
        let g = FormalGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::similarity(AbsTree::entity("Dog"), AbsTree::entity("Wolf"), 0.87);
        let result = g.linearize(&tree, &ctx).unwrap();
        assert!(result.contains("exhibits similarity to"));
        assert!(result.contains("0.87"));
    }

    #[test]
    fn linearize_conjunction() {
        let g = FormalGrammar;
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
        assert!(result.contains("and"));
    }

    #[test]
    fn linearize_provenance() {
        let g = FormalGrammar;
        let ctx = LinContext::default();
        let tree = AbsTree::WithProvenance {
            inner: Box::new(AbsTree::triple(
                AbsTree::entity("Dog"),
                AbsTree::relation("is-a"),
                AbsTree::entity("Mammal"),
            )),
            tag: ProvenanceTag::GraphInferred,
        };
        let result = g.linearize(&tree, &ctx).unwrap();
        assert!(result.contains("[source: graph inference]"));
    }
}
