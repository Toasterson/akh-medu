//! Bridge: conversions between [`AbsTree`] and existing akh-medu types.
//!
//! This module provides bidirectional conversions so the grammar system
//! can interoperate with the existing synthesis pipeline:
//!
//! - `fact_to_abs()` — `ExtractedFact` → `AbsTree`
//! - `triple_to_abs()` — `Triple` → `AbsTree`
//! - `summary_to_abs()` — `NarrativeSummary` → `AbsTree`
//! - `abs_to_summary()` — `AbsTree` → `NarrativeSummary`

use crate::agent::synthesize::{ExtractedFact, FactKind, NarrativeSection, NarrativeSummary};
use crate::graph::Triple;
use crate::registry::SymbolRegistry;

use super::abs::AbsTree;
use super::concrete::{ConcreteGrammar, LinContext};
use super::error::GrammarResult;

/// Convert an `ExtractedFact` (from the synthesis pipeline) to an `AbsTree`.
pub fn fact_to_abs(fact: &ExtractedFact) -> AbsTree {
    match &fact.kind {
        FactKind::Triple {
            subject,
            predicate,
            object,
        } => AbsTree::triple(
            AbsTree::entity(subject),
            AbsTree::relation(predicate),
            AbsTree::entity(object),
        ),
        FactKind::Similarity {
            entity,
            similar_to,
            score,
        } => AbsTree::similarity(AbsTree::entity(entity), AbsTree::entity(similar_to), *score),
        FactKind::Gap {
            entity,
            description,
        } => AbsTree::gap(AbsTree::entity(entity), description),
        FactKind::Inference {
            expression,
            simplified,
        } => AbsTree::Inference {
            expression: expression.clone(),
            simplified: simplified.clone(),
        },
        FactKind::CodeFact { kind, name, detail } => AbsTree::CodeFact {
            kind: kind.clone(),
            name: name.clone(),
            detail: detail.clone(),
        },
        FactKind::Derivation { count, iterations } => AbsTree::Freeform(format!(
            "Derived {count} new facts over {iterations} iterations."
        )),
        FactKind::Raw(text) => AbsTree::Freeform(text.clone()),
    }
}

/// Convert a graph `Triple` to an `AbsTree`, resolving labels via the registry.
pub fn triple_to_abs(triple: &Triple, registry: &SymbolRegistry) -> AbsTree {
    let subject_label = registry
        .get(triple.subject)
        .map(|m| m.label)
        .unwrap_or_else(|| format!("sym:{}", triple.subject.get()));
    let predicate_label = registry
        .get(triple.predicate)
        .map(|m| m.label)
        .unwrap_or_else(|| format!("sym:{}", triple.predicate.get()));
    let object_label = registry
        .get(triple.object)
        .map(|m| m.label)
        .unwrap_or_else(|| format!("sym:{}", triple.object.get()));

    let base = AbsTree::triple(
        AbsTree::entity_resolved(subject_label, triple.subject),
        AbsTree::RelationRef {
            label: predicate_label,
            symbol_id: Some(triple.predicate),
        },
        AbsTree::entity_resolved(object_label, triple.object),
    );

    if (triple.confidence - 1.0).abs() > f32::EPSILON {
        AbsTree::WithConfidence {
            inner: Box::new(base),
            confidence: triple.confidence,
        }
    } else {
        base
    }
}

/// Convert a `NarrativeSummary` (from the synthesis pipeline) to an `AbsTree`.
pub fn summary_to_abs(summary: &NarrativeSummary) -> AbsTree {
    let overview = AbsTree::Freeform(summary.overview.clone());

    let sections: Vec<AbsTree> = summary
        .sections
        .iter()
        .map(|s| AbsTree::section(&s.heading, vec![AbsTree::Freeform(s.prose.clone())]))
        .collect();

    let gaps: Vec<AbsTree> = summary
        .gaps
        .iter()
        .map(|g| AbsTree::Freeform(g.clone()))
        .collect();

    AbsTree::Document {
        overview: Box::new(overview),
        sections,
        gaps,
    }
}

/// Convert an `AbsTree` to a `NarrativeSummary`, linearizing through a grammar.
pub fn abs_to_summary(
    tree: &AbsTree,
    grammar: &dyn ConcreteGrammar,
    ctx: &LinContext,
) -> GrammarResult<NarrativeSummary> {
    match tree {
        AbsTree::Document {
            overview,
            sections,
            gaps,
        } => {
            let overview_text = grammar.linearize(overview, ctx)?;

            let rendered_sections: Vec<NarrativeSection> = sections
                .iter()
                .map(|s| match s {
                    AbsTree::Section { heading, body } => {
                        // Linearize each body item separately to avoid
                        // the heading being duplicated by the grammar's
                        // Section rendering.
                        let lines: Vec<String> = body
                            .iter()
                            .map(|item| grammar.linearize(item, ctx))
                            .collect::<GrammarResult<Vec<_>>>()?;
                        Ok(NarrativeSection {
                            heading: heading.clone(),
                            prose: lines.join("\n"),
                        })
                    }
                    _ => {
                        let text = grammar.linearize(s, ctx)?;
                        Ok(NarrativeSection {
                            heading: String::new(),
                            prose: text,
                        })
                    }
                })
                .collect::<GrammarResult<Vec<_>>>()?;

            let rendered_gaps: Vec<String> = gaps
                .iter()
                .map(|g| grammar.linearize(g, ctx))
                .collect::<GrammarResult<Vec<_>>>()?;

            let facts_count = count_facts(tree);

            Ok(NarrativeSummary {
                overview: overview_text,
                sections: rendered_sections,
                gaps: rendered_gaps,
                facts_count,
            })
        }
        // Non-document trees: wrap in a single-section summary
        _ => {
            let text = grammar.linearize(tree, ctx)?;
            let facts_count = count_facts(tree);
            Ok(NarrativeSummary {
                overview: text,
                sections: vec![],
                gaps: vec![],
                facts_count,
            })
        }
    }
}

/// Convert an `AbsTree` fact back to an `ExtractedFact`.
pub fn abs_to_fact(tree: &AbsTree, source_tool: &str, source_cycle: u64) -> Option<ExtractedFact> {
    let kind = match tree {
        AbsTree::Triple {
            subject,
            predicate,
            object,
        } => {
            let s = subject.label().unwrap_or("?").to_string();
            let p = predicate.label().unwrap_or("?").to_string();
            let o = object.label().unwrap_or("?").to_string();
            FactKind::Triple {
                subject: s,
                predicate: p,
                object: o,
            }
        }
        AbsTree::Similarity {
            entity,
            similar_to,
            score,
        } => FactKind::Similarity {
            entity: entity.label().unwrap_or("?").to_string(),
            similar_to: similar_to.label().unwrap_or("?").to_string(),
            score: *score,
        },
        AbsTree::Gap { entity, description } => FactKind::Gap {
            entity: entity.label().unwrap_or("?").to_string(),
            description: description.clone(),
        },
        AbsTree::Inference {
            expression,
            simplified,
        } => FactKind::Inference {
            expression: expression.clone(),
            simplified: simplified.clone(),
        },
        AbsTree::CodeFact { kind, name, detail } => FactKind::CodeFact {
            kind: kind.clone(),
            name: name.clone(),
            detail: detail.clone(),
        },
        AbsTree::CodeModule { name, .. } => FactKind::CodeFact {
            kind: "module".to_string(),
            name: name.clone(),
            detail: "module".to_string(),
        },
        AbsTree::CodeSignature { kind, name, .. } => FactKind::CodeFact {
            kind: kind.clone(),
            name: name.clone(),
            detail: kind.clone(),
        },
        AbsTree::DataFlow { .. } => return None,
        AbsTree::WithConfidence { inner, .. } | AbsTree::WithProvenance { inner, .. } => {
            return abs_to_fact(inner, source_tool, source_cycle);
        }
        AbsTree::Freeform(text) => FactKind::Raw(text.clone()),
        _ => return None,
    };

    Some(ExtractedFact {
        kind,
        source_tool: source_tool.to_string(),
        source_cycle,
    })
}

/// Count statement-like nodes in the tree.
fn count_facts(tree: &AbsTree) -> usize {
    match tree {
        AbsTree::Triple { .. }
        | AbsTree::Similarity { .. }
        | AbsTree::Gap { .. }
        | AbsTree::Inference { .. }
        | AbsTree::CodeFact { .. }
        | AbsTree::CodeSignature { .. } => 1,

        AbsTree::CodeModule { children, .. } => {
            1 + children.iter().map(count_facts).sum::<usize>()
        }

        AbsTree::DataFlow { .. } => 0,

        AbsTree::WithConfidence { inner, .. } | AbsTree::WithProvenance { inner, .. } => {
            count_facts(inner)
        }

        AbsTree::Conjunction { items, .. } => items.iter().map(count_facts).sum(),

        AbsTree::Section { body, .. } => body.iter().map(count_facts).sum(),

        AbsTree::Document {
            overview,
            sections,
            gaps,
        } => {
            count_facts(overview)
                + sections.iter().map(count_facts).sum::<usize>()
                + gaps.iter().map(count_facts).sum::<usize>()
        }

        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fact_to_abs_triple() {
        let fact = ExtractedFact {
            kind: FactKind::Triple {
                subject: "Dog".into(),
                predicate: "is-a".into(),
                object: "Mammal".into(),
            },
            source_tool: "test".into(),
            source_cycle: 1,
        };
        let tree = fact_to_abs(&fact);
        assert_eq!(tree.cat(), super::super::cat::Cat::Statement);
        assert_eq!(tree.collect_labels(), vec!["Dog", "is-a", "Mammal"]);
    }

    #[test]
    fn fact_to_abs_similarity() {
        let fact = ExtractedFact {
            kind: FactKind::Similarity {
                entity: "Dog".into(),
                similar_to: "Wolf".into(),
                score: 0.87,
            },
            source_tool: "test".into(),
            source_cycle: 1,
        };
        let tree = fact_to_abs(&fact);
        assert_eq!(tree.cat(), super::super::cat::Cat::Similarity);
    }

    #[test]
    fn summary_roundtrip() {
        let summary = NarrativeSummary {
            overview: "Test overview.".into(),
            sections: vec![NarrativeSection {
                heading: "Section 1".into(),
                prose: "Some prose.".into(),
            }],
            gaps: vec!["A knowledge gap.".into()],
            facts_count: 3,
        };

        let tree = summary_to_abs(&summary);
        assert_eq!(tree.cat(), super::super::cat::Cat::Document);
    }

    #[test]
    fn abs_to_fact_roundtrip() {
        let fact = ExtractedFact {
            kind: FactKind::Triple {
                subject: "Dog".into(),
                predicate: "is-a".into(),
                object: "Mammal".into(),
            },
            source_tool: "grammar".into(),
            source_cycle: 0,
        };
        let tree = fact_to_abs(&fact);
        let restored = abs_to_fact(&tree, "grammar", 0);
        assert!(restored.is_some());
        let restored = restored.unwrap();
        assert!(matches!(restored.kind, FactKind::Triple { .. }));
    }

    #[test]
    fn count_facts_in_document() {
        let doc = AbsTree::Document {
            overview: Box::new(AbsTree::Freeform("Overview".into())),
            sections: vec![AbsTree::section(
                "Test",
                vec![
                    AbsTree::triple(
                        AbsTree::entity("A"),
                        AbsTree::relation("r"),
                        AbsTree::entity("B"),
                    ),
                    AbsTree::triple(
                        AbsTree::entity("C"),
                        AbsTree::relation("r"),
                        AbsTree::entity("D"),
                    ),
                ],
            )],
            gaps: vec![AbsTree::gap(AbsTree::entity("E"), "missing data")],
        };
        assert_eq!(count_facts(&doc), 3);
    }
}
