//! Abstract syntax tree — the shared semantic representation.
//!
//! All concrete grammars (archetypes) linearize from the same [`AbsTree`].
//! The abstract syntax captures *what* to say; the concrete grammar controls
//! *how* to say it.
//!
//! This is the GF-inspired "interlingua": one representation, many voices.

use serde::{Deserialize, Serialize};

use crate::symbol::SymbolId;

use super::cat::Cat;
use super::error::{GrammarError, GrammarResult};

/// How a piece of knowledge was derived (simplified provenance tag).
///
/// Lighter-weight than `DerivationKind` — carries just enough information
/// for the grammar to mention provenance in prose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProvenanceTag {
    /// Directly extracted from source material.
    Extracted,
    /// Inferred by graph traversal.
    GraphInferred,
    /// Inferred by VSA similarity/analogy.
    VsaInferred { similarity: f32 },
    /// Derived by symbolic reasoning (e-graph).
    Reasoned,
    /// Agent decision or consolidation.
    AgentDerived,
    /// Semantic enrichment (role classification, importance).
    Enrichment,
    /// User-asserted via the grammar parser.
    UserAsserted,
}

impl std::fmt::Display for ProvenanceTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProvenanceTag::Extracted => write!(f, "extracted"),
            ProvenanceTag::GraphInferred => write!(f, "graph-inferred"),
            ProvenanceTag::VsaInferred { similarity } => {
                write!(f, "vsa-inferred({similarity:.2})")
            }
            ProvenanceTag::Reasoned => write!(f, "reasoned"),
            ProvenanceTag::AgentDerived => write!(f, "agent-derived"),
            ProvenanceTag::Enrichment => write!(f, "enrichment"),
            ProvenanceTag::UserAsserted => write!(f, "user-asserted"),
        }
    }
}

/// Abstract syntax tree node — the shared semantic representation.
///
/// All concrete grammars linearize from these nodes. The same tree can be
/// rendered as formal prose, terse notation, or narrative flow depending
/// on the chosen archetype.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AbsTree {
    // ── Leaves ──────────────────────────────────────────────────────────

    /// A reference to an entity (resolved or unresolved).
    EntityRef {
        label: String,
        symbol_id: Option<SymbolId>,
    },

    /// A reference to a relation (resolved or unresolved).
    RelationRef {
        label: String,
        symbol_id: Option<SymbolId>,
    },

    /// Free-form text that didn't parse into a structured node.
    Freeform(String),

    // ── Composites ──────────────────────────────────────────────────────

    /// A subject-predicate-object statement.
    Triple {
        subject: Box<AbsTree>,
        predicate: Box<AbsTree>,
        object: Box<AbsTree>,
    },

    /// A similarity assertion between two entities.
    Similarity {
        entity: Box<AbsTree>,
        similar_to: Box<AbsTree>,
        score: f32,
    },

    /// A knowledge gap — something unknown or missing.
    Gap {
        entity: Box<AbsTree>,
        description: String,
    },

    /// An inference or reasoning result.
    Inference {
        expression: String,
        simplified: String,
    },

    /// A code structure fact.
    CodeFact {
        kind: String,
        name: String,
        detail: String,
    },

    // ── Modifiers ───────────────────────────────────────────────────────

    /// Wrap an inner node with a confidence score.
    WithConfidence {
        inner: Box<AbsTree>,
        confidence: f32,
    },

    /// Wrap an inner node with provenance information.
    WithProvenance {
        inner: Box<AbsTree>,
        tag: ProvenanceTag,
    },

    // ── Structure ───────────────────────────────────────────────────────

    /// A conjunction (and) or disjunction (or) of multiple items.
    Conjunction {
        items: Vec<AbsTree>,
        /// `true` for conjunction ("and"), `false` for disjunction ("or").
        is_and: bool,
    },

    /// A titled section containing body items.
    Section {
        heading: String,
        body: Vec<AbsTree>,
    },

    /// A complete document with overview, sections, and gaps.
    Document {
        overview: Box<AbsTree>,
        sections: Vec<AbsTree>,
        gaps: Vec<AbsTree>,
    },
}

impl AbsTree {
    /// Return the grammatical category of this node.
    pub fn cat(&self) -> Cat {
        match self {
            AbsTree::EntityRef { .. } => Cat::Entity,
            AbsTree::RelationRef { .. } => Cat::Relation,
            AbsTree::Freeform(_) => Cat::Freeform,
            AbsTree::Triple { .. } => Cat::Statement,
            AbsTree::Similarity { .. } => Cat::Similarity,
            AbsTree::Gap { .. } => Cat::Gap,
            AbsTree::Inference { .. } => Cat::Inference,
            AbsTree::CodeFact { .. } => Cat::CodeFact,
            AbsTree::WithConfidence { .. } => Cat::Confidence,
            AbsTree::WithProvenance { .. } => Cat::Provenance,
            AbsTree::Conjunction { .. } => Cat::Conjunction,
            AbsTree::Section { .. } => Cat::Section,
            AbsTree::Document { .. } => Cat::Document,
        }
    }

    /// Type-check the tree structure, ensuring children have valid categories.
    ///
    /// Returns `Ok(())` if the tree is well-formed, or a `TypeMismatch` error
    /// identifying the first invalid child.
    pub fn validate(&self) -> GrammarResult<()> {
        match self {
            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => {
                let s_cat = subject.cat();
                if !s_cat.valid_in_statement() {
                    return Err(GrammarError::TypeMismatch {
                        expected: Cat::Entity,
                        actual: s_cat,
                    });
                }
                let p_cat = predicate.cat();
                if p_cat != Cat::Relation && p_cat != Cat::Freeform {
                    return Err(GrammarError::TypeMismatch {
                        expected: Cat::Relation,
                        actual: p_cat,
                    });
                }
                let o_cat = object.cat();
                if !o_cat.valid_in_statement() {
                    return Err(GrammarError::TypeMismatch {
                        expected: Cat::Entity,
                        actual: o_cat,
                    });
                }
                subject.validate()?;
                predicate.validate()?;
                object.validate()
            }
            AbsTree::Similarity { entity, similar_to, .. } => {
                entity.validate()?;
                similar_to.validate()
            }
            AbsTree::Gap { entity, .. } => entity.validate(),
            AbsTree::WithConfidence { inner, confidence } => {
                if !(*confidence >= 0.0 && *confidence <= 1.0) {
                    return Err(GrammarError::TypeMismatch {
                        expected: Cat::Confidence,
                        actual: Cat::Freeform,
                    });
                }
                inner.validate()
            }
            AbsTree::WithProvenance { inner, .. } => inner.validate(),
            AbsTree::Conjunction { items, .. } => {
                for item in items {
                    item.validate()?;
                }
                Ok(())
            }
            AbsTree::Section { body, .. } => {
                for item in body {
                    item.validate()?;
                }
                Ok(())
            }
            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => {
                overview.validate()?;
                for s in sections {
                    s.validate()?;
                }
                for g in gaps {
                    g.validate()?;
                }
                Ok(())
            }
            // Leaves and simple nodes are always valid.
            _ => Ok(()),
        }
    }

    /// Extract the label from a leaf node, if it is one.
    pub fn label(&self) -> Option<&str> {
        match self {
            AbsTree::EntityRef { label, .. } | AbsTree::RelationRef { label, .. } => Some(label),
            AbsTree::Freeform(text) => Some(text),
            _ => None,
        }
    }

    /// Extract the symbol ID from a leaf node, if resolved.
    pub fn symbol_id(&self) -> Option<SymbolId> {
        match self {
            AbsTree::EntityRef { symbol_id, .. } | AbsTree::RelationRef { symbol_id, .. } => {
                *symbol_id
            }
            _ => None,
        }
    }

    /// Count total nodes in this tree (including self).
    pub fn node_count(&self) -> usize {
        match self {
            AbsTree::EntityRef { .. }
            | AbsTree::RelationRef { .. }
            | AbsTree::Freeform(_)
            | AbsTree::Inference { .. }
            | AbsTree::CodeFact { .. } => 1,

            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => 1 + subject.node_count() + predicate.node_count() + object.node_count(),

            AbsTree::Similarity {
                entity, similar_to, ..
            } => 1 + entity.node_count() + similar_to.node_count(),

            AbsTree::Gap { entity, .. } => 1 + entity.node_count(),

            AbsTree::WithConfidence { inner, .. } | AbsTree::WithProvenance { inner, .. } => {
                1 + inner.node_count()
            }

            AbsTree::Conjunction { items, .. } => {
                1 + items.iter().map(|i| i.node_count()).sum::<usize>()
            }

            AbsTree::Section { body, .. } => {
                1 + body.iter().map(|i| i.node_count()).sum::<usize>()
            }

            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => {
                1 + overview.node_count()
                    + sections.iter().map(|s| s.node_count()).sum::<usize>()
                    + gaps.iter().map(|g| g.node_count()).sum::<usize>()
            }
        }
    }

    /// Collect all entity/relation labels referenced in this tree.
    pub fn collect_labels(&self) -> Vec<&str> {
        let mut labels = Vec::new();
        self.collect_labels_inner(&mut labels);
        labels
    }

    fn collect_labels_inner<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            AbsTree::EntityRef { label, .. } | AbsTree::RelationRef { label, .. } => {
                out.push(label);
            }
            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => {
                subject.collect_labels_inner(out);
                predicate.collect_labels_inner(out);
                object.collect_labels_inner(out);
            }
            AbsTree::Similarity {
                entity, similar_to, ..
            } => {
                entity.collect_labels_inner(out);
                similar_to.collect_labels_inner(out);
            }
            AbsTree::Gap { entity, .. } => {
                entity.collect_labels_inner(out);
            }
            AbsTree::WithConfidence { inner, .. } | AbsTree::WithProvenance { inner, .. } => {
                inner.collect_labels_inner(out);
            }
            AbsTree::Conjunction { items, .. } => {
                for item in items {
                    item.collect_labels_inner(out);
                }
            }
            AbsTree::Section { body, .. } => {
                for item in body {
                    item.collect_labels_inner(out);
                }
            }
            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => {
                overview.collect_labels_inner(out);
                for s in sections {
                    s.collect_labels_inner(out);
                }
                for g in gaps {
                    g.collect_labels_inner(out);
                }
            }
            _ => {}
        }
    }
}

// ── Convenience constructors ────────────────────────────────────────────

impl AbsTree {
    /// Create an entity reference.
    pub fn entity(label: impl Into<String>) -> Self {
        AbsTree::EntityRef {
            label: label.into(),
            symbol_id: None,
        }
    }

    /// Create an entity reference with a resolved symbol ID.
    pub fn entity_resolved(label: impl Into<String>, id: SymbolId) -> Self {
        AbsTree::EntityRef {
            label: label.into(),
            symbol_id: Some(id),
        }
    }

    /// Create a relation reference.
    pub fn relation(label: impl Into<String>) -> Self {
        AbsTree::RelationRef {
            label: label.into(),
            symbol_id: None,
        }
    }

    /// Create a statement (triple).
    pub fn triple(subject: AbsTree, predicate: AbsTree, object: AbsTree) -> Self {
        AbsTree::Triple {
            subject: Box::new(subject),
            predicate: Box::new(predicate),
            object: Box::new(object),
        }
    }

    /// Create a statement with confidence.
    pub fn triple_with_confidence(
        subject: AbsTree,
        predicate: AbsTree,
        object: AbsTree,
        confidence: f32,
    ) -> Self {
        AbsTree::WithConfidence {
            inner: Box::new(Self::triple(subject, predicate, object)),
            confidence,
        }
    }

    /// Create a similarity assertion.
    pub fn similarity(entity: AbsTree, similar_to: AbsTree, score: f32) -> Self {
        AbsTree::Similarity {
            entity: Box::new(entity),
            similar_to: Box::new(similar_to),
            score,
        }
    }

    /// Create a knowledge gap.
    pub fn gap(entity: AbsTree, description: impl Into<String>) -> Self {
        AbsTree::Gap {
            entity: Box::new(entity),
            description: description.into(),
        }
    }

    /// Create a conjunction (items joined by "and").
    pub fn and(items: Vec<AbsTree>) -> Self {
        AbsTree::Conjunction {
            items,
            is_and: true,
        }
    }

    /// Create a disjunction (items joined by "or").
    pub fn or(items: Vec<AbsTree>) -> Self {
        AbsTree::Conjunction {
            items,
            is_and: false,
        }
    }

    /// Create a document section.
    pub fn section(heading: impl Into<String>, body: Vec<AbsTree>) -> Self {
        AbsTree::Section {
            heading: heading.into(),
            body,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cat_returns_correct_category() {
        let e = AbsTree::entity("Dog");
        assert_eq!(e.cat(), Cat::Entity);

        let r = AbsTree::relation("is-a");
        assert_eq!(r.cat(), Cat::Relation);

        let t = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        assert_eq!(t.cat(), Cat::Statement);
    }

    #[test]
    fn validate_well_formed_triple() {
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        assert!(tree.validate().is_ok());
    }

    #[test]
    fn validate_rejects_statement_as_subject() {
        let inner = AbsTree::triple(
            AbsTree::entity("A"),
            AbsTree::relation("r"),
            AbsTree::entity("B"),
        );
        let tree = AbsTree::triple(inner, AbsTree::relation("r"), AbsTree::entity("C"));
        assert!(tree.validate().is_err());
    }

    #[test]
    fn node_count_is_correct() {
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        assert_eq!(tree.node_count(), 4); // triple + 3 leaves
    }

    #[test]
    fn collect_labels_finds_all() {
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let labels = tree.collect_labels();
        assert_eq!(labels, vec!["Dog", "is-a", "Mammal"]);
    }

    #[test]
    fn serde_round_trip() {
        let tree = AbsTree::triple_with_confidence(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
            0.95,
        );
        let json = serde_json::to_string(&tree).unwrap();
        let restored: AbsTree = serde_json::from_str(&json).unwrap();
        assert_eq!(tree, restored);
    }

    #[test]
    fn conjunction_collects_all_labels() {
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
        let labels = tree.collect_labels();
        assert_eq!(labels, vec!["Dog", "is-a", "Mammal", "Cat", "is-a", "Mammal"]);
    }
}
