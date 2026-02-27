//! Abstract syntax tree — the shared semantic representation.
//!
//! All concrete grammars (archetypes) linearize from the same [`AbsTree`].
//! The abstract syntax captures *what* to say; the concrete grammar controls
//! *how* to say it.
//!
//! This is the GF-inspired "interlingua": one representation, many voices.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::registry::SymbolRegistry;
use crate::symbol::SymbolId;
use crate::vsa::HyperVec;
use crate::vsa::encode::{encode_label, encode_token};
use crate::vsa::item_memory::ItemMemory;
use crate::vsa::ops::VsaOps;

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

/// Create a synthetic [`SymbolId`] from a string by hashing it.
///
/// Uses the high-bit convention (`hash | (1<<63)`) to avoid collisions
/// with real allocated SymbolIds. The same string always produces the
/// same synthetic ID.
pub fn synthetic_id(name: &str) -> SymbolId {
    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    let hash = hasher.finish();
    SymbolId::new(hash | (1u64 << 63)).expect("non-zero hash with high bit set")
}

/// Well-known synthetic [`SymbolId`]s for structural role encoding in VSA.
///
/// These IDs are used as "role" vectors when encoding structured `AbsTree`
/// nodes (triples, similarity, sections) into compositional hypervectors.
/// Follows the same pattern as `AgentPredicates` — resolved once at init.
#[derive(Debug, Clone)]
pub struct VsaRoleSymbols {
    pub role_subject: SymbolId,
    pub role_predicate: SymbolId,
    pub role_object: SymbolId,
    pub role_entity: SymbolId,
    pub role_similar_to: SymbolId,
    pub role_heading: SymbolId,
    // NLU extension roles (Phase 14j)
    pub role_negation: SymbolId,
    pub role_quantifier: SymbolId,
    pub role_scope: SymbolId,
    pub role_entity_a: SymbolId,
    pub role_entity_b: SymbolId,
    pub role_property: SymbolId,
    pub role_ordering: SymbolId,
    pub role_condition: SymbolId,
    pub role_consequent: SymbolId,
    pub role_temporal: SymbolId,
    pub role_modality: SymbolId,
    pub role_head: SymbolId,
    pub role_clause: SymbolId,
}

impl VsaRoleSymbols {
    /// Create the well-known role symbols using deterministic hashing.
    pub fn new() -> Self {
        Self {
            role_subject: synthetic_id("vsa-role:subject"),
            role_predicate: synthetic_id("vsa-role:predicate"),
            role_object: synthetic_id("vsa-role:object"),
            role_entity: synthetic_id("vsa-role:entity"),
            role_similar_to: synthetic_id("vsa-role:similar-to"),
            role_heading: synthetic_id("vsa-role:heading"),
            role_negation: synthetic_id("vsa-role:negation"),
            role_quantifier: synthetic_id("vsa-role:quantifier"),
            role_scope: synthetic_id("vsa-role:scope"),
            role_entity_a: synthetic_id("vsa-role:entity-a"),
            role_entity_b: synthetic_id("vsa-role:entity-b"),
            role_property: synthetic_id("vsa-role:property"),
            role_ordering: synthetic_id("vsa-role:ordering"),
            role_condition: synthetic_id("vsa-role:condition"),
            role_consequent: synthetic_id("vsa-role:consequent"),
            role_temporal: synthetic_id("vsa-role:temporal"),
            role_modality: synthetic_id("vsa-role:modality"),
            role_head: synthetic_id("vsa-role:head"),
            role_clause: synthetic_id("vsa-role:clause"),
        }
    }
}

impl Default for VsaRoleSymbols {
    fn default() -> Self {
        Self::new()
    }
}

/// A step in a data flow chain between code components.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataFlowStep {
    pub name: String,
    pub via_type: Option<String>,
}

/// Logical quantifier for quantified expressions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Quantifier {
    Universal,
    Existential,
    Most,
    None,
    Specific(u32),
}

/// Comparison ordering direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompareOrd {
    GreaterThan,
    LessThan,
    Equal,
}

/// Temporal expression types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TemporalExpr {
    Absolute(u64),
    Relative(i64),
    Named(String),
    Recurring(String),
}

/// Deontic/epistemic modality.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Modality {
    Want,
    Can,
    Should,
    Must,
    May,
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

    /// A code module with optional semantic enrichment and child items.
    CodeModule {
        name: String,
        role: Option<String>,
        importance: Option<f32>,
        doc_summary: Option<String>,
        children: Vec<AbsTree>,
    },

    /// A code item (function/struct/enum/trait) with signature details.
    CodeSignature {
        kind: String,
        name: String,
        doc_summary: Option<String>,
        params_or_fields: Vec<String>,
        return_type: Option<String>,
        traits: Vec<String>,
        importance: Option<f32>,
    },

    /// A directed data flow chain between code components.
    DataFlow { steps: Vec<DataFlowStep> },

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
    Section { heading: String, body: Vec<AbsTree> },

    /// A complete document with overview, sections, and gaps.
    Document {
        overview: Box<AbsTree>,
        sections: Vec<AbsTree>,
        gaps: Vec<AbsTree>,
    },

    /// A discourse-framed response with point-of-view and query focus.
    DiscourseFrame {
        pov: super::discourse::PointOfView,
        focus: super::discourse::QueryFocus,
        inner: Box<AbsTree>,
    },

    // ── NLU extensions (Phase 14j) ──────────────────────────────────────

    /// Logical negation of an inner expression.
    Negation { inner: Box<AbsTree> },

    /// Quantified expression (universal, existential, etc.).
    Quantified {
        quantifier: Quantifier,
        scope: Box<AbsTree>,
    },

    /// Comparison between two entities on a property.
    Comparison {
        entity_a: Box<AbsTree>,
        entity_b: Box<AbsTree>,
        property: String,
        ordering: CompareOrd,
    },

    /// Conditional: if condition then consequent.
    Conditional {
        condition: Box<AbsTree>,
        consequent: Box<AbsTree>,
    },

    /// Temporal scoping of an inner expression.
    Temporal {
        time_expr: TemporalExpr,
        inner: Box<AbsTree>,
    },

    /// Modal scoping (want, can, should, must, may).
    Modal {
        modality: Modality,
        inner: Box<AbsTree>,
    },

    /// Relative clause: head noun modified by a subordinate clause.
    RelativeClause {
        head: Box<AbsTree>,
        clause: Box<AbsTree>,
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
            AbsTree::CodeModule { .. } => Cat::CodeModule,
            AbsTree::CodeSignature { .. } => Cat::CodeSignature,
            AbsTree::DataFlow { .. } => Cat::DataFlow,
            AbsTree::WithConfidence { .. } => Cat::Confidence,
            AbsTree::WithProvenance { .. } => Cat::Provenance,
            AbsTree::Conjunction { .. } => Cat::Conjunction,
            AbsTree::Section { .. } => Cat::Section,
            AbsTree::Document { .. } => Cat::Document,
            AbsTree::DiscourseFrame { .. } => Cat::DiscourseFrame,
            AbsTree::Negation { .. } => Cat::Negation,
            AbsTree::Quantified { .. } => Cat::Quantifier,
            AbsTree::Comparison { .. } => Cat::Comparison,
            AbsTree::Conditional { .. } => Cat::Conditional,
            AbsTree::Temporal { .. } => Cat::Temporal,
            AbsTree::Modal { .. } => Cat::Modal,
            AbsTree::RelativeClause { .. } => Cat::RelativeClause,
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
            AbsTree::Similarity {
                entity, similar_to, ..
            } => {
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
            AbsTree::CodeModule { children, .. } => {
                for child in children {
                    child.validate()?;
                }
                Ok(())
            }
            AbsTree::DiscourseFrame { inner, .. } => inner.validate(),
            AbsTree::Negation { inner } => inner.validate(),
            AbsTree::Quantified { scope, .. } => scope.validate(),
            AbsTree::Comparison {
                entity_a, entity_b, ..
            } => {
                entity_a.validate()?;
                entity_b.validate()
            }
            AbsTree::Conditional {
                condition,
                consequent,
            } => {
                condition.validate()?;
                consequent.validate()
            }
            AbsTree::Temporal { inner, .. } => inner.validate(),
            AbsTree::Modal { inner, .. } => inner.validate(),
            AbsTree::RelativeClause { head, clause } => {
                head.validate()?;
                clause.validate()
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
            | AbsTree::CodeFact { .. }
            | AbsTree::CodeSignature { .. }
            | AbsTree::DataFlow { .. } => 1,

            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => 1 + subject.node_count() + predicate.node_count() + object.node_count(),

            AbsTree::Similarity {
                entity, similar_to, ..
            } => 1 + entity.node_count() + similar_to.node_count(),

            AbsTree::Gap { entity, .. } => 1 + entity.node_count(),

            AbsTree::WithConfidence { inner, .. }
            | AbsTree::WithProvenance { inner, .. }
            | AbsTree::DiscourseFrame { inner, .. } => 1 + inner.node_count(),

            AbsTree::Conjunction { items, .. } => {
                1 + items.iter().map(|i| i.node_count()).sum::<usize>()
            }

            AbsTree::CodeModule { children, .. } => {
                1 + children.iter().map(|c| c.node_count()).sum::<usize>()
            }

            AbsTree::Section { body, .. } => 1 + body.iter().map(|i| i.node_count()).sum::<usize>(),

            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => {
                1 + overview.node_count()
                    + sections.iter().map(|s| s.node_count()).sum::<usize>()
                    + gaps.iter().map(|g| g.node_count()).sum::<usize>()
            }

            AbsTree::Negation { inner }
            | AbsTree::Quantified { scope: inner, .. }
            | AbsTree::Temporal { inner, .. }
            | AbsTree::Modal { inner, .. } => 1 + inner.node_count(),

            AbsTree::Comparison {
                entity_a, entity_b, ..
            } => 1 + entity_a.node_count() + entity_b.node_count(),

            AbsTree::Conditional {
                condition,
                consequent,
            } => 1 + condition.node_count() + consequent.node_count(),

            AbsTree::RelativeClause { head, clause } => {
                1 + head.node_count() + clause.node_count()
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
            AbsTree::WithConfidence { inner, .. }
            | AbsTree::WithProvenance { inner, .. }
            | AbsTree::DiscourseFrame { inner, .. } => {
                inner.collect_labels_inner(out);
            }
            AbsTree::Conjunction { items, .. } => {
                for item in items {
                    item.collect_labels_inner(out);
                }
            }
            AbsTree::CodeModule { children, .. } => {
                for child in children {
                    child.collect_labels_inner(out);
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
            AbsTree::Negation { inner }
            | AbsTree::Quantified { scope: inner, .. }
            | AbsTree::Temporal { inner, .. }
            | AbsTree::Modal { inner, .. } => {
                inner.collect_labels_inner(out);
            }
            AbsTree::Comparison {
                entity_a, entity_b, ..
            } => {
                entity_a.collect_labels_inner(out);
                entity_b.collect_labels_inner(out);
            }
            AbsTree::Conditional {
                condition,
                consequent,
            } => {
                condition.collect_labels_inner(out);
                consequent.collect_labels_inner(out);
            }
            AbsTree::RelativeClause { head, clause } => {
                head.collect_labels_inner(out);
                clause.collect_labels_inner(out);
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

    /// Create a code module node with optional enrichment.
    pub fn code_module(
        name: impl Into<String>,
        role: Option<String>,
        importance: Option<f32>,
        doc_summary: Option<String>,
        children: Vec<AbsTree>,
    ) -> Self {
        AbsTree::CodeModule {
            name: name.into(),
            role,
            importance,
            doc_summary,
            children,
        }
    }

    /// Create a code signature node (other fields default to None/empty).
    pub fn code_signature(kind: impl Into<String>, name: impl Into<String>) -> Self {
        AbsTree::CodeSignature {
            kind: kind.into(),
            name: name.into(),
            doc_summary: None,
            params_or_fields: Vec::new(),
            return_type: None,
            traits: Vec::new(),
            importance: None,
        }
    }

    /// Create a data flow chain from steps.
    pub fn data_flow(steps: Vec<DataFlowStep>) -> Self {
        AbsTree::DataFlow { steps }
    }

    /// Create a negation wrapper.
    pub fn negation(inner: AbsTree) -> Self {
        AbsTree::Negation {
            inner: Box::new(inner),
        }
    }

    /// Create a quantified expression.
    pub fn quantified(quantifier: Quantifier, scope: AbsTree) -> Self {
        AbsTree::Quantified {
            quantifier,
            scope: Box::new(scope),
        }
    }

    /// Create a comparison.
    pub fn comparison(
        entity_a: AbsTree,
        entity_b: AbsTree,
        property: impl Into<String>,
        ordering: CompareOrd,
    ) -> Self {
        AbsTree::Comparison {
            entity_a: Box::new(entity_a),
            entity_b: Box::new(entity_b),
            property: property.into(),
            ordering,
        }
    }

    /// Create a conditional.
    pub fn conditional(condition: AbsTree, consequent: AbsTree) -> Self {
        AbsTree::Conditional {
            condition: Box::new(condition),
            consequent: Box::new(consequent),
        }
    }

    /// Create a temporal scoping.
    pub fn temporal(time_expr: TemporalExpr, inner: AbsTree) -> Self {
        AbsTree::Temporal {
            time_expr,
            inner: Box::new(inner),
        }
    }

    /// Create a modal expression.
    pub fn modal(modality: Modality, inner: AbsTree) -> Self {
        AbsTree::Modal {
            modality,
            inner: Box::new(inner),
        }
    }

    /// Create a relative clause.
    pub fn relative_clause(head: AbsTree, clause: AbsTree) -> Self {
        AbsTree::RelativeClause {
            head: Box::new(head),
            clause: Box::new(clause),
        }
    }
}

// ── Grounding & VSA encoding ────────────────────────────────────────────

impl AbsTree {
    /// Resolve labels in this tree against a [`SymbolRegistry`].
    ///
    /// Returns a new tree with `symbol_id` fields filled in where the
    /// registry knows the label. Leaves unknown labels as `None`.
    /// This is non-destructive — the original tree is not modified.
    pub fn ground(&self, registry: &SymbolRegistry) -> Self {
        match self {
            AbsTree::EntityRef { label, symbol_id } => AbsTree::EntityRef {
                label: label.clone(),
                symbol_id: symbol_id.or_else(|| registry.lookup(label)),
            },
            AbsTree::RelationRef { label, symbol_id } => AbsTree::RelationRef {
                label: label.clone(),
                symbol_id: symbol_id.or_else(|| registry.lookup(label)),
            },
            AbsTree::Freeform(text) => AbsTree::Freeform(text.clone()),
            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => AbsTree::Triple {
                subject: Box::new(subject.ground(registry)),
                predicate: Box::new(predicate.ground(registry)),
                object: Box::new(object.ground(registry)),
            },
            AbsTree::Similarity {
                entity,
                similar_to,
                score,
            } => AbsTree::Similarity {
                entity: Box::new(entity.ground(registry)),
                similar_to: Box::new(similar_to.ground(registry)),
                score: *score,
            },
            AbsTree::Gap {
                entity,
                description,
            } => AbsTree::Gap {
                entity: Box::new(entity.ground(registry)),
                description: description.clone(),
            },
            AbsTree::Inference {
                expression,
                simplified,
            } => AbsTree::Inference {
                expression: expression.clone(),
                simplified: simplified.clone(),
            },
            AbsTree::CodeFact { kind, name, detail } => AbsTree::CodeFact {
                kind: kind.clone(),
                name: name.clone(),
                detail: detail.clone(),
            },
            AbsTree::CodeModule {
                name,
                role,
                importance,
                doc_summary,
                children,
            } => AbsTree::CodeModule {
                name: name.clone(),
                role: role.clone(),
                importance: *importance,
                doc_summary: doc_summary.clone(),
                children: children.iter().map(|c| c.ground(registry)).collect(),
            },
            AbsTree::CodeSignature {
                kind,
                name,
                doc_summary,
                params_or_fields,
                return_type,
                traits,
                importance,
            } => AbsTree::CodeSignature {
                kind: kind.clone(),
                name: name.clone(),
                doc_summary: doc_summary.clone(),
                params_or_fields: params_or_fields.clone(),
                return_type: return_type.clone(),
                traits: traits.clone(),
                importance: *importance,
            },
            AbsTree::DataFlow { steps } => AbsTree::DataFlow {
                steps: steps.clone(),
            },
            AbsTree::WithConfidence { inner, confidence } => AbsTree::WithConfidence {
                inner: Box::new(inner.ground(registry)),
                confidence: *confidence,
            },
            AbsTree::WithProvenance { inner, tag } => AbsTree::WithProvenance {
                inner: Box::new(inner.ground(registry)),
                tag: tag.clone(),
            },
            AbsTree::Conjunction { items, is_and } => AbsTree::Conjunction {
                items: items.iter().map(|i| i.ground(registry)).collect(),
                is_and: *is_and,
            },
            AbsTree::Section { heading, body } => AbsTree::Section {
                heading: heading.clone(),
                body: body.iter().map(|i| i.ground(registry)).collect(),
            },
            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => AbsTree::Document {
                overview: Box::new(overview.ground(registry)),
                sections: sections.iter().map(|s| s.ground(registry)).collect(),
                gaps: gaps.iter().map(|g| g.ground(registry)).collect(),
            },
            AbsTree::DiscourseFrame { pov, focus, inner } => AbsTree::DiscourseFrame {
                pov: *pov,
                focus: focus.clone(),
                inner: Box::new(inner.ground(registry)),
            },
            AbsTree::Negation { inner } => AbsTree::Negation {
                inner: Box::new(inner.ground(registry)),
            },
            AbsTree::Quantified { quantifier, scope } => AbsTree::Quantified {
                quantifier: quantifier.clone(),
                scope: Box::new(scope.ground(registry)),
            },
            AbsTree::Comparison {
                entity_a,
                entity_b,
                property,
                ordering,
            } => AbsTree::Comparison {
                entity_a: Box::new(entity_a.ground(registry)),
                entity_b: Box::new(entity_b.ground(registry)),
                property: property.clone(),
                ordering: ordering.clone(),
            },
            AbsTree::Conditional {
                condition,
                consequent,
            } => AbsTree::Conditional {
                condition: Box::new(condition.ground(registry)),
                consequent: Box::new(consequent.ground(registry)),
            },
            AbsTree::Temporal { time_expr, inner } => AbsTree::Temporal {
                time_expr: time_expr.clone(),
                inner: Box::new(inner.ground(registry)),
            },
            AbsTree::Modal { modality, inner } => AbsTree::Modal {
                modality: modality.clone(),
                inner: Box::new(inner.ground(registry)),
            },
            AbsTree::RelativeClause { head, clause } => AbsTree::RelativeClause {
                head: Box::new(head.ground(registry)),
                clause: Box::new(clause.ground(registry)),
            },
        }
    }

    /// Count unresolved leaf nodes (EntityRef/RelationRef with `symbol_id: None`).
    pub fn unresolved_count(&self) -> usize {
        match self {
            AbsTree::EntityRef {
                symbol_id: None, ..
            }
            | AbsTree::RelationRef {
                symbol_id: None, ..
            } => 1,
            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => {
                subject.unresolved_count()
                    + predicate.unresolved_count()
                    + object.unresolved_count()
            }
            AbsTree::Similarity {
                entity, similar_to, ..
            } => entity.unresolved_count() + similar_to.unresolved_count(),
            AbsTree::Gap { entity, .. } => entity.unresolved_count(),
            AbsTree::WithConfidence { inner, .. }
            | AbsTree::WithProvenance { inner, .. }
            | AbsTree::DiscourseFrame { inner, .. } => inner.unresolved_count(),
            AbsTree::Conjunction { items, .. } => items.iter().map(|i| i.unresolved_count()).sum(),
            AbsTree::CodeModule { children, .. } => {
                children.iter().map(|c| c.unresolved_count()).sum()
            }
            AbsTree::Section { body, .. } => body.iter().map(|i| i.unresolved_count()).sum(),
            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => {
                overview.unresolved_count()
                    + sections.iter().map(|s| s.unresolved_count()).sum::<usize>()
                    + gaps.iter().map(|g| g.unresolved_count()).sum::<usize>()
            }
            AbsTree::Negation { inner }
            | AbsTree::Quantified { scope: inner, .. }
            | AbsTree::Temporal { inner, .. }
            | AbsTree::Modal { inner, .. } => inner.unresolved_count(),
            AbsTree::Comparison {
                entity_a, entity_b, ..
            } => entity_a.unresolved_count() + entity_b.unresolved_count(),
            AbsTree::Conditional {
                condition,
                consequent,
            } => condition.unresolved_count() + consequent.unresolved_count(),
            AbsTree::RelativeClause { head, clause } => {
                head.unresolved_count() + clause.unresolved_count()
            }
            _ => 0,
        }
    }

    /// Find the label of the first unresolved leaf node, if any.
    pub fn first_unresolved(&self) -> Option<&str> {
        match self {
            AbsTree::EntityRef {
                label,
                symbol_id: None,
            }
            | AbsTree::RelationRef {
                label,
                symbol_id: None,
            } => Some(label),
            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => subject
                .first_unresolved()
                .or_else(|| predicate.first_unresolved())
                .or_else(|| object.first_unresolved()),
            AbsTree::Similarity {
                entity, similar_to, ..
            } => entity
                .first_unresolved()
                .or_else(|| similar_to.first_unresolved()),
            AbsTree::Gap { entity, .. } => entity.first_unresolved(),
            AbsTree::WithConfidence { inner, .. }
            | AbsTree::WithProvenance { inner, .. }
            | AbsTree::DiscourseFrame { inner, .. } => inner.first_unresolved(),
            AbsTree::Conjunction { items, .. } => items.iter().find_map(|i| i.first_unresolved()),
            AbsTree::CodeModule { children, .. } => {
                children.iter().find_map(|c| c.first_unresolved())
            }
            AbsTree::Section { body, .. } => body.iter().find_map(|i| i.first_unresolved()),
            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => overview
                .first_unresolved()
                .or_else(|| sections.iter().find_map(|s| s.first_unresolved()))
                .or_else(|| gaps.iter().find_map(|g| g.first_unresolved())),
            AbsTree::Negation { inner }
            | AbsTree::Quantified { scope: inner, .. }
            | AbsTree::Temporal { inner, .. }
            | AbsTree::Modal { inner, .. } => inner.first_unresolved(),
            AbsTree::Comparison {
                entity_a, entity_b, ..
            } => entity_a
                .first_unresolved()
                .or_else(|| entity_b.first_unresolved()),
            AbsTree::Conditional {
                condition,
                consequent,
            } => condition
                .first_unresolved()
                .or_else(|| consequent.first_unresolved()),
            AbsTree::RelativeClause { head, clause } => {
                head.first_unresolved()
                    .or_else(|| clause.first_unresolved())
            }
            _ => None,
        }
    }

    /// Encode this abstract syntax tree as a compositional hypervector.
    ///
    /// Grounded leaves use their known vector from `item_memory`; ungrounded
    /// leaves fall back to hash-based `encode_token`. Composite nodes use
    /// role-filler binding to preserve structure.
    pub fn to_vsa(
        &self,
        ops: &VsaOps,
        item_memory: &ItemMemory,
        roles: &VsaRoleSymbols,
    ) -> GrammarResult<HyperVec> {
        match self {
            // ── Leaves ──────────────────────────────────────────────────
            AbsTree::EntityRef {
                symbol_id: Some(id),
                ..
            }
            | AbsTree::RelationRef {
                symbol_id: Some(id),
                ..
            } => Ok(item_memory.get_or_create(ops, *id)),

            AbsTree::EntityRef {
                label,
                symbol_id: None,
            }
            | AbsTree::RelationRef {
                label,
                symbol_id: None,
            } => Ok(encode_token(ops, label)),

            AbsTree::Freeform(text) => {
                encode_label(ops, text).map_err(|e| GrammarError::VsaError {
                    message: e.to_string(),
                })
            }

            // ── Composites ──────────────────────────────────────────────
            AbsTree::Triple {
                subject,
                predicate,
                object,
            } => {
                let s_vec = subject.to_vsa(ops, item_memory, roles)?;
                let p_vec = predicate.to_vsa(ops, item_memory, roles)?;
                let o_vec = object.to_vsa(ops, item_memory, roles)?;

                let role_s = vsa_encode_symbol(ops, roles.role_subject);
                let role_p = vsa_encode_symbol(ops, roles.role_predicate);
                let role_o = vsa_encode_symbol(ops, roles.role_object);

                let bound_s = ops.bind(&role_s, &s_vec).map_err(vsa_err)?;
                let bound_p = ops.bind(&role_p, &p_vec).map_err(vsa_err)?;
                let bound_o = ops.bind(&role_o, &o_vec).map_err(vsa_err)?;

                ops.bundle(&[&bound_s, &bound_p, &bound_o]).map_err(vsa_err)
            }

            AbsTree::Similarity {
                entity, similar_to, ..
            } => {
                let e_vec = entity.to_vsa(ops, item_memory, roles)?;
                let s_vec = similar_to.to_vsa(ops, item_memory, roles)?;

                let role_e = vsa_encode_symbol(ops, roles.role_entity);
                let role_st = vsa_encode_symbol(ops, roles.role_similar_to);

                let bound_e = ops.bind(&role_e, &e_vec).map_err(vsa_err)?;
                let bound_s = ops.bind(&role_st, &s_vec).map_err(vsa_err)?;

                ops.bundle(&[&bound_e, &bound_s]).map_err(vsa_err)
            }

            AbsTree::Gap {
                entity,
                description,
            } => {
                let e_vec = entity.to_vsa(ops, item_memory, roles)?;
                let d_vec = encode_label(ops, description).map_err(vsa_err)?;
                ops.bundle(&[&e_vec, &d_vec]).map_err(vsa_err)
            }

            AbsTree::Inference {
                expression,
                simplified,
            } => {
                let expr_vec = encode_label(ops, expression).map_err(vsa_err)?;
                let simp_vec = encode_label(ops, simplified).map_err(vsa_err)?;
                ops.bundle(&[&expr_vec, &simp_vec]).map_err(vsa_err)
            }

            AbsTree::CodeFact { kind, name, detail } => {
                let k_vec = encode_token(ops, kind);
                let n_vec = encode_token(ops, name);
                let d_vec = encode_label(ops, detail).map_err(vsa_err)?;
                ops.bundle(&[&k_vec, &n_vec, &d_vec]).map_err(vsa_err)
            }

            AbsTree::CodeModule { name, children, .. } => {
                let n_vec = encode_token(ops, name);
                let mut all_vecs = vec![n_vec];
                for child in children {
                    all_vecs.push(child.to_vsa(ops, item_memory, roles)?);
                }
                let refs: Vec<&HyperVec> = all_vecs.iter().collect();
                ops.bundle(&refs).map_err(vsa_err)
            }

            AbsTree::CodeSignature { kind, name, .. } => {
                let k_vec = encode_token(ops, kind);
                let n_vec = encode_token(ops, name);
                ops.bundle(&[&k_vec, &n_vec]).map_err(vsa_err)
            }

            AbsTree::DataFlow { steps } => {
                if steps.is_empty() {
                    return Err(GrammarError::VsaError {
                        message: "cannot encode empty data flow".into(),
                    });
                }
                let vecs: Vec<HyperVec> =
                    steps.iter().map(|s| encode_token(ops, &s.name)).collect();
                let refs: Vec<&HyperVec> = vecs.iter().collect();
                ops.bundle(&refs).map_err(vsa_err)
            }

            // ── Modifiers (transparent) ─────────────────────────────────
            AbsTree::WithConfidence { inner, .. }
            | AbsTree::WithProvenance { inner, .. }
            | AbsTree::DiscourseFrame { inner, .. } => inner.to_vsa(ops, item_memory, roles),

            // ── NLU extensions ────────────────────────────────────────
            AbsTree::Negation { inner } => {
                let inner_vec = inner.to_vsa(ops, item_memory, roles)?;
                let role_neg = vsa_encode_symbol(ops, roles.role_negation);
                ops.bind(&role_neg, &inner_vec).map_err(vsa_err)
            }

            AbsTree::Quantified { quantifier, scope } => {
                let scope_vec = scope.to_vsa(ops, item_memory, roles)?;
                let q_label = match quantifier {
                    Quantifier::Universal => "universal",
                    Quantifier::Existential => "existential",
                    Quantifier::Most => "most",
                    Quantifier::None => "none",
                    Quantifier::Specific(n) => {
                        // Encode the number as a string for simplicity
                        let _ = n;
                        "specific"
                    }
                };
                let q_vec = encode_token(ops, q_label);
                let role_q = vsa_encode_symbol(ops, roles.role_quantifier);
                let role_s = vsa_encode_symbol(ops, roles.role_scope);
                let bound_q = ops.bind(&role_q, &q_vec).map_err(vsa_err)?;
                let bound_s = ops.bind(&role_s, &scope_vec).map_err(vsa_err)?;
                ops.bundle(&[&bound_q, &bound_s]).map_err(vsa_err)
            }

            AbsTree::Comparison {
                entity_a,
                entity_b,
                property,
                ordering,
            } => {
                let a_vec = entity_a.to_vsa(ops, item_memory, roles)?;
                let b_vec = entity_b.to_vsa(ops, item_memory, roles)?;
                let p_vec = encode_token(ops, property);
                let ord_label = match ordering {
                    CompareOrd::GreaterThan => "greater",
                    CompareOrd::LessThan => "less",
                    CompareOrd::Equal => "equal",
                };
                let o_vec = encode_token(ops, ord_label);
                let role_a = vsa_encode_symbol(ops, roles.role_entity_a);
                let role_b = vsa_encode_symbol(ops, roles.role_entity_b);
                let role_p = vsa_encode_symbol(ops, roles.role_property);
                let role_o = vsa_encode_symbol(ops, roles.role_ordering);
                let bound_a = ops.bind(&role_a, &a_vec).map_err(vsa_err)?;
                let bound_b = ops.bind(&role_b, &b_vec).map_err(vsa_err)?;
                let bound_p = ops.bind(&role_p, &p_vec).map_err(vsa_err)?;
                let bound_o = ops.bind(&role_o, &o_vec).map_err(vsa_err)?;
                ops.bundle(&[&bound_a, &bound_b, &bound_p, &bound_o])
                    .map_err(vsa_err)
            }

            AbsTree::Conditional {
                condition,
                consequent,
            } => {
                let cond_vec = condition.to_vsa(ops, item_memory, roles)?;
                let cons_vec = consequent.to_vsa(ops, item_memory, roles)?;
                let role_c = vsa_encode_symbol(ops, roles.role_condition);
                let role_cs = vsa_encode_symbol(ops, roles.role_consequent);
                let bound_c = ops.bind(&role_c, &cond_vec).map_err(vsa_err)?;
                let bound_cs = ops.bind(&role_cs, &cons_vec).map_err(vsa_err)?;
                ops.bundle(&[&bound_c, &bound_cs]).map_err(vsa_err)
            }

            AbsTree::Temporal { time_expr, inner } => {
                let inner_vec = inner.to_vsa(ops, item_memory, roles)?;
                let t_label = match time_expr {
                    TemporalExpr::Named(n) => n.as_str(),
                    TemporalExpr::Recurring(r) => r.as_str(),
                    _ => "temporal",
                };
                let t_vec = encode_token(ops, t_label);
                let role_t = vsa_encode_symbol(ops, roles.role_temporal);
                let bound_t = ops.bind(&role_t, &t_vec).map_err(vsa_err)?;
                ops.bundle(&[&bound_t, &inner_vec]).map_err(vsa_err)
            }

            AbsTree::Modal { modality, inner } => {
                let inner_vec = inner.to_vsa(ops, item_memory, roles)?;
                let m_label = match modality {
                    Modality::Want => "want",
                    Modality::Can => "can",
                    Modality::Should => "should",
                    Modality::Must => "must",
                    Modality::May => "may",
                };
                let m_vec = encode_token(ops, m_label);
                let role_m = vsa_encode_symbol(ops, roles.role_modality);
                let bound_m = ops.bind(&role_m, &m_vec).map_err(vsa_err)?;
                ops.bundle(&[&bound_m, &inner_vec]).map_err(vsa_err)
            }

            AbsTree::RelativeClause { head, clause } => {
                let h_vec = head.to_vsa(ops, item_memory, roles)?;
                let c_vec = clause.to_vsa(ops, item_memory, roles)?;
                let role_h = vsa_encode_symbol(ops, roles.role_head);
                let role_c = vsa_encode_symbol(ops, roles.role_clause);
                let bound_h = ops.bind(&role_h, &h_vec).map_err(vsa_err)?;
                let bound_c = ops.bind(&role_c, &c_vec).map_err(vsa_err)?;
                ops.bundle(&[&bound_h, &bound_c]).map_err(vsa_err)
            }

            // ── Structure ───────────────────────────────────────────────
            AbsTree::Conjunction { items, .. } => {
                if items.is_empty() {
                    return Err(GrammarError::VsaError {
                        message: "cannot encode empty conjunction".into(),
                    });
                }
                let vecs: Vec<HyperVec> = items
                    .iter()
                    .map(|i| i.to_vsa(ops, item_memory, roles))
                    .collect::<Result<_, _>>()?;
                let refs: Vec<&HyperVec> = vecs.iter().collect();
                ops.bundle(&refs).map_err(vsa_err)
            }

            AbsTree::Section { heading, body } => {
                let heading_vec = encode_label(ops, heading).map_err(vsa_err)?;
                let role_h = vsa_encode_symbol(ops, roles.role_heading);
                let bound_h = ops.bind(&role_h, &heading_vec).map_err(vsa_err)?;

                let mut all_vecs = vec![bound_h];
                for item in body {
                    all_vecs.push(item.to_vsa(ops, item_memory, roles)?);
                }
                let refs: Vec<&HyperVec> = all_vecs.iter().collect();
                ops.bundle(&refs).map_err(vsa_err)
            }

            AbsTree::Document {
                overview,
                sections,
                gaps,
            } => {
                let mut all_vecs = vec![overview.to_vsa(ops, item_memory, roles)?];
                for s in sections {
                    all_vecs.push(s.to_vsa(ops, item_memory, roles)?);
                }
                for g in gaps {
                    all_vecs.push(g.to_vsa(ops, item_memory, roles)?);
                }
                let refs: Vec<&HyperVec> = all_vecs.iter().collect();
                ops.bundle(&refs).map_err(vsa_err)
            }
        }
    }
}

/// Helper: encode a SymbolId into a HyperVec (for role vectors).
fn vsa_encode_symbol(ops: &VsaOps, id: SymbolId) -> HyperVec {
    crate::vsa::encode::encode_symbol(ops, id)
}

/// Helper: convert VsaError → GrammarError.
fn vsa_err(e: crate::error::VsaError) -> GrammarError {
    GrammarError::VsaError {
        message: e.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd;
    use crate::symbol::{SymbolKind, SymbolMeta};
    use crate::vsa::item_memory::ItemMemory;
    use crate::vsa::{Dimension, Encoding};

    fn test_ops() -> VsaOps {
        VsaOps::new(simd::best_kernel(), Dimension::TEST, Encoding::Bipolar)
    }

    fn test_item_memory() -> ItemMemory {
        ItemMemory::new(Dimension::TEST, Encoding::Bipolar, 1000)
    }

    fn test_registry_with_symbols() -> SymbolRegistry {
        let reg = SymbolRegistry::new();
        reg.register(SymbolMeta::new(
            SymbolId::new(1).unwrap(),
            SymbolKind::Entity,
            "Dog",
        ))
        .unwrap();
        reg.register(SymbolMeta::new(
            SymbolId::new(2).unwrap(),
            SymbolKind::Entity,
            "Mammal",
        ))
        .unwrap();
        reg.register(SymbolMeta::new(
            SymbolId::new(3).unwrap(),
            SymbolKind::Relation,
            "is-a",
        ))
        .unwrap();
        reg
    }

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
        assert_eq!(
            labels,
            vec!["Dog", "is-a", "Mammal", "Cat", "is-a", "Mammal"]
        );
    }

    // ── VsaRoleSymbols tests ────────────────────────────────────────────

    #[test]
    fn role_symbols_are_distinct() {
        let roles = VsaRoleSymbols::new();
        let ids = [
            roles.role_subject,
            roles.role_predicate,
            roles.role_object,
            roles.role_entity,
            roles.role_similar_to,
            roles.role_heading,
        ];
        // All roles must be unique
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "role {i} and {j} collide");
            }
        }
    }

    #[test]
    fn role_symbols_deterministic() {
        let r1 = VsaRoleSymbols::new();
        let r2 = VsaRoleSymbols::new();
        assert_eq!(r1.role_subject, r2.role_subject);
        assert_eq!(r1.role_heading, r2.role_heading);
    }

    // ── ground() tests ──────────────────────────────────────────────────

    #[test]
    fn ground_resolves_known_labels() {
        let reg = test_registry_with_symbols();
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let grounded = tree.ground(&reg);

        // Subject should resolve to Dog (id=1)
        if let AbsTree::Triple {
            subject,
            predicate,
            object,
        } = &grounded
        {
            assert_eq!(subject.symbol_id(), Some(SymbolId::new(1).unwrap()));
            assert_eq!(predicate.symbol_id(), Some(SymbolId::new(3).unwrap()));
            assert_eq!(object.symbol_id(), Some(SymbolId::new(2).unwrap()));
        } else {
            panic!("expected Triple");
        }
    }

    #[test]
    fn ground_leaves_unknown_labels() {
        let reg = test_registry_with_symbols();
        let tree = AbsTree::entity("UnknownThing");
        let grounded = tree.ground(&reg);
        assert_eq!(grounded.symbol_id(), None);
    }

    #[test]
    fn ground_preserves_existing_ids() {
        let reg = test_registry_with_symbols();
        let existing_id = SymbolId::new(999).unwrap();
        let tree = AbsTree::entity_resolved("Dog", existing_id);
        let grounded = tree.ground(&reg);
        // Should keep the existing ID, not overwrite with registry's
        assert_eq!(grounded.symbol_id(), Some(existing_id));
    }

    #[test]
    fn ground_is_non_destructive() {
        let reg = test_registry_with_symbols();
        let tree = AbsTree::entity("Dog");
        let grounded = tree.ground(&reg);
        // Original should still be unresolved
        assert_eq!(tree.symbol_id(), None);
        // Grounded should be resolved
        assert_eq!(grounded.symbol_id(), Some(SymbolId::new(1).unwrap()));
    }

    #[test]
    fn unresolved_count_correct() {
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity_resolved("Mammal", SymbolId::new(2).unwrap()),
        );
        assert_eq!(tree.unresolved_count(), 2); // Dog and is-a
    }

    // ── to_vsa() tests ──────────────────────────────────────────────────

    #[test]
    fn to_vsa_grounded_entity_correct_dim() {
        let ops = test_ops();
        let im = test_item_memory();
        let roles = VsaRoleSymbols::new();
        let id = SymbolId::new(42).unwrap();
        let tree = AbsTree::entity_resolved("Dog", id);
        let vec = tree.to_vsa(&ops, &im, &roles).unwrap();
        assert_eq!(vec.dim(), ops.dim());
    }

    #[test]
    fn to_vsa_ungrounded_entity_hash_encodes() {
        let ops = test_ops();
        let im = test_item_memory();
        let roles = VsaRoleSymbols::new();
        let tree = AbsTree::entity("Dog");
        let vec = tree.to_vsa(&ops, &im, &roles).unwrap();
        assert_eq!(vec.dim(), ops.dim());
        // Same label should produce same vector
        let vec2 = AbsTree::entity("Dog").to_vsa(&ops, &im, &roles).unwrap();
        assert_eq!(vec, vec2);
    }

    #[test]
    fn to_vsa_triple_valid_vector() {
        let ops = test_ops();
        let im = test_item_memory();
        let roles = VsaRoleSymbols::new();
        let tree = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let vec = tree.to_vsa(&ops, &im, &roles).unwrap();
        assert_eq!(vec.dim(), ops.dim());
    }

    #[test]
    fn to_vsa_role_filler_recoverable_from_triple() {
        let ops = test_ops();
        let im = test_item_memory();
        let roles = VsaRoleSymbols::new();

        let subj_id = SymbolId::new(10).unwrap();
        let tree = AbsTree::triple(
            AbsTree::entity_resolved("Dog", subj_id),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let triple_vec = tree.to_vsa(&ops, &im, &roles).unwrap();

        // Unbind with role_subject to recover the subject
        let role_s_vec = vsa_encode_symbol(&ops, roles.role_subject);
        let recovered = ops.unbind(&triple_vec, &role_s_vec).unwrap();
        let expected = im.get_or_create(&ops, subj_id);
        let sim = ops.similarity(&recovered, &expected).unwrap();
        // With 3 components bundled, recovery won't be exact but should
        // be well above random
        assert!(sim > 0.55, "recovered subject sim={sim}, expected > 0.55");
    }

    #[test]
    fn to_vsa_similar_triples_higher_similarity() {
        let ops = test_ops();
        let im = test_item_memory();
        let roles = VsaRoleSymbols::new();

        let t1 = AbsTree::triple(
            AbsTree::entity("Dog"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let t2 = AbsTree::triple(
            AbsTree::entity("Cat"),
            AbsTree::relation("is-a"),
            AbsTree::entity("Mammal"),
        );
        let t3 = AbsTree::triple(
            AbsTree::entity("Paris"),
            AbsTree::relation("capital-of"),
            AbsTree::entity("France"),
        );

        let v1 = t1.to_vsa(&ops, &im, &roles).unwrap();
        let v2 = t2.to_vsa(&ops, &im, &roles).unwrap();
        let v3 = t3.to_vsa(&ops, &im, &roles).unwrap();

        let sim_related = ops.similarity(&v1, &v2).unwrap();
        let sim_unrelated = ops.similarity(&v1, &v3).unwrap();

        // Two triples sharing predicate+object should be more similar
        // than two completely different triples
        assert!(
            sim_related > sim_unrelated,
            "related={sim_related:.3} should be > unrelated={sim_unrelated:.3}"
        );
    }

    #[test]
    fn to_vsa_modifiers_transparent() {
        let ops = test_ops();
        let im = test_item_memory();
        let roles = VsaRoleSymbols::new();

        let inner = AbsTree::entity("Dog");
        let with_conf = AbsTree::WithConfidence {
            inner: Box::new(inner.clone()),
            confidence: 0.9,
        };

        let v_inner = inner.to_vsa(&ops, &im, &roles).unwrap();
        let v_conf = with_conf.to_vsa(&ops, &im, &roles).unwrap();
        assert_eq!(v_inner, v_conf);
    }

    #[test]
    fn to_vsa_conjunction_bundles() {
        let ops = test_ops();
        let im = test_item_memory();
        let roles = VsaRoleSymbols::new();

        let tree = AbsTree::and(vec![AbsTree::entity("Dog"), AbsTree::entity("Cat")]);
        let vec = tree.to_vsa(&ops, &im, &roles).unwrap();
        assert_eq!(vec.dim(), ops.dim());
    }

    #[test]
    fn to_vsa_empty_conjunction_errors() {
        let ops = test_ops();
        let im = test_item_memory();
        let roles = VsaRoleSymbols::new();

        let tree = AbsTree::Conjunction {
            items: vec![],
            is_and: true,
        };
        assert!(tree.to_vsa(&ops, &im, &roles).is_err());
    }
}
