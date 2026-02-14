//! AbsTree builder for narrative synthesis.
//!
//! Converts grouped facts + engine enrichment into an [`AbsTree::Document`]
//! that can be linearized through any concrete grammar. This module replaces
//! the manual Markdown rendering in `synthesize::render_template()` with
//! structured abstract syntax trees.

use std::collections::{BTreeMap, HashSet};

use crate::engine::Engine;
use crate::grammar::abs::{AbsTree, DataFlowStep};
use crate::grammar::bridge::fact_to_abs;

use super::semantic_enrichment::SemanticPredicates;
use super::synthesize::{
    ExtractedFact, FactGroup, FactKind, GroupKey, is_code_annotation, is_code_predicate,
    is_semantic_predicate,
};
use super::tools::code_predicates::CodePredicates;

/// Build a complete `AbsTree::Document` from grouped facts and engine.
pub(crate) fn build_document(
    goal: &str,
    groups: &[FactGroup],
    all_facts: &[ExtractedFact],
    engine: &Engine,
) -> AbsTree {
    let overview = build_overview(goal, all_facts, groups);

    let mut sections = Vec::new();
    let mut gaps = Vec::new();

    for group in groups {
        match &group.key {
            GroupKey::Entity(entity) => {
                sections.push(build_entity_section(entity, &group.facts));
            }
            GroupKey::CodeStructure => {
                sections.push(build_code_section(&group.facts, engine));
            }
            GroupKey::RelatedConcepts => {
                sections.push(build_similarity_section(&group.facts));
            }
            GroupKey::KnowledgeGaps => {
                for fact in &group.facts {
                    if let FactKind::Gap {
                        entity,
                        description,
                    } = &fact.kind
                    {
                        if entity.is_empty() {
                            gaps.push(AbsTree::Freeform(description.clone()));
                        } else {
                            gaps.push(AbsTree::gap(AbsTree::entity(entity), description.as_str()));
                        }
                    }
                }
            }
            GroupKey::ReasoningResults => {
                sections.push(build_reasoning_section(&group.facts));
            }
            GroupKey::Other => {
                if let Some(sec) = build_other_section(&group.facts) {
                    sections.push(sec);
                }
            }
        }
    }

    AbsTree::Document {
        overview: Box::new(overview),
        sections,
        gaps,
    }
}

/// Build the overview Freeform node.
fn build_overview(goal: &str, all_facts: &[ExtractedFact], groups: &[FactGroup]) -> AbsTree {
    let total_facts = all_facts.len();
    let max_cycle = all_facts.iter().map(|f| f.source_cycle).max().unwrap_or(0);
    let topic_count = groups
        .iter()
        .filter(|g| !matches!(g.key, GroupKey::Other | GroupKey::KnowledgeGaps))
        .count();

    let text = if total_facts == 0 {
        format!("Explored \"{goal}\" but found no concrete facts yet.")
    } else {
        format!(
            "Explored \"{goal}\" over {max_cycle} cycle{}, found {total_facts} fact{} about {topic_count} topic{}.",
            if max_cycle == 1 { "" } else { "s" },
            if total_facts == 1 { "" } else { "s" },
            if topic_count == 1 { "" } else { "s" },
        )
    };

    AbsTree::Freeform(text)
}

/// Build a Section of Triple nodes for an entity.
fn build_entity_section(entity: &str, facts: &[ExtractedFact]) -> AbsTree {
    let mut body = Vec::new();

    for fact in facts {
        if let FactKind::Triple { predicate, .. } = &fact.kind {
            // Skip predicates handled by Code Architecture section.
            if is_code_predicate(predicate)
                || is_code_annotation(predicate)
                || is_semantic_predicate(predicate)
            {
                continue;
            }
            body.push(fact_to_abs(fact));
        }
    }

    AbsTree::section(entity, body)
}

/// Build the Code Architecture/Structure section using enrichment data.
fn build_code_section(facts: &[ExtractedFact], engine: &Engine) -> AbsTree {
    let mut modules: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut functions: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut types: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut deps: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut extra_items: Vec<AbsTree> = Vec::new();

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
                "code:defines-struct"
                | "code:defines-enum"
                | "code:defines-type"
                | "defines-type"
                | "implements"
                | "code:has-variant" => {
                    types.entry(name.clone()).or_default().push(detail.clone());
                }
                "depends-on" | "code:depends-on" => {
                    deps.entry(name.clone()).or_default().push(detail.clone());
                }
                "code:defined-in" => {
                    extra_items.push(AbsTree::CodeFact {
                        kind: "defined-in".to_string(),
                        name: name.clone(),
                        detail: detail.clone(),
                    });
                }
                _ => {
                    functions
                        .entry(name.clone())
                        .or_default()
                        .push(detail.clone());
                }
            }
        }
    }

    // Identify the primary entity.
    let mut fact_counts: BTreeMap<String, usize> = BTreeMap::new();
    for map in [&modules, &functions, &types, &deps] {
        for (name, items) in map {
            *fact_counts.entry(name.clone()).or_default() += items.len();
        }
    }
    let primary = if modules.is_empty() {
        fact_counts
            .iter()
            .max_by_key(|(_, c)| **c)
            .map(|(n, _)| n.clone())
    } else {
        modules
            .iter()
            .max_by_key(|(name, children)| {
                children.len()
                    + functions.get(*name).map_or(0, |v| v.len())
                    + types.get(*name).map_or(0, |v| v.len())
                    + deps.get(*name).map_or(0, |v| v.len())
            })
            .map(|(name, _)| name.clone())
    };
    let primary_children: HashSet<String> = primary
        .as_ref()
        .and_then(|p| modules.get(p))
        .map(|children| children.iter().cloned().collect())
        .unwrap_or_default();

    // Keep only relevant entries.
    let is_relevant = |name: &str| -> bool {
        primary.as_deref() == Some(name)
            || primary_children.contains(name)
            || fact_counts.get(name).copied().unwrap_or(0) >= 2
    };
    modules.retain(|name, _| is_relevant(name));
    functions.retain(|name, _| is_relevant(name));
    types.retain(|name, _| is_relevant(name));
    deps.retain(|name, _| is_relevant(name));

    // Semantic enrichment lookups.
    let sem_preds = SemanticPredicates::init(engine).ok();
    let code_preds = CodePredicates::init(engine).ok();

    let resolve_sym = |label: &str| -> Option<crate::symbol::SymbolId> {
        if let Ok(sym) = engine.resolve_symbol(label) {
            return Some(sym);
        }
        let lower = label.to_lowercase();
        engine.all_symbols().iter().find_map(|sym| {
            let sym_lower = sym.label.to_lowercase();
            if sym_lower == lower || sym_lower.ends_with(&format!("::{lower}")) {
                Some(sym.id)
            } else {
                None
            }
        })
    };

    let get_role = |name: &str| -> Option<String> {
        let preds = sem_preds.as_ref()?;
        let sym = resolve_sym(name)?;
        super::semantic_enrichment::lookup_role(engine, sym, preds)
    };

    let get_importance = |name: &str| -> Option<f32> {
        let preds = sem_preds.as_ref()?;
        let sym = resolve_sym(name)?;
        super::semantic_enrichment::lookup_importance(engine, sym, preds)
    };

    let get_doc = |name: &str| -> Option<String> {
        let preds = code_preds.as_ref()?;
        let sym = resolve_sym(name)?;
        let triples = engine.triples_from(sym);
        triples
            .iter()
            .find(|t| t.predicate == preds.has_doc)
            .map(|t| engine.resolve_label(t.object).to_string())
    };

    let get_fn_signature = |name: &str| -> (Vec<String>, Option<String>) {
        let Some(preds) = code_preds.as_ref() else {
            return (Vec::new(), None);
        };
        let Some(sym) = resolve_sym(name) else {
            return (Vec::new(), None);
        };
        let triples = engine.triples_from(sym);
        let params: Vec<String> = triples
            .iter()
            .filter(|t| t.predicate == preds.has_param)
            .map(|t| engine.resolve_label(t.object).to_string())
            .collect();
        let ret = triples
            .iter()
            .find(|t| t.predicate == preds.returns_type)
            .map(|t| engine.resolve_label(t.object).to_string());
        (params, ret)
    };

    let get_fields = |name: &str| -> Vec<String> {
        let Some(preds) = code_preds.as_ref() else {
            return Vec::new();
        };
        let Some(sym) = resolve_sym(name) else {
            return Vec::new();
        };
        engine
            .triples_from(sym)
            .iter()
            .filter(|t| t.predicate == preds.has_field)
            .map(|t| engine.resolve_label(t.object).to_string())
            .collect()
    };

    let get_derives = |name: &str| -> Vec<String> {
        let Some(preds) = code_preds.as_ref() else {
            return Vec::new();
        };
        let Some(sym) = resolve_sym(name) else {
            return Vec::new();
        };
        engine
            .triples_from(sym)
            .iter()
            .filter(|t| t.predicate == preds.derives_trait)
            .map(|t| engine.resolve_label(t.object).to_string())
            .collect()
    };

    // Build AbsTree items for the section body.
    let mut body: Vec<AbsTree> = Vec::new();
    let mut rendered_children: HashSet<String> = HashSet::new();

    if let Some(ref primary_name) = primary {
        if let Some(children) = modules.get(primary_name) {
            // Sort children by importance.
            let mut sorted_children: Vec<&String> = children.iter().collect();
            sorted_children.sort_by(|a, b| {
                let imp_a = get_importance(a).unwrap_or(0.0);
                let imp_b = get_importance(b).unwrap_or(0.0);
                imp_b
                    .partial_cmp(&imp_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Truncation: > 8 children → show top 5.
            let total_children = sorted_children.len();
            let display_children = if total_children > 8 {
                &sorted_children[..5]
            } else {
                &sorted_children
            };

            // Build child AbsTree nodes.
            let mut child_nodes: Vec<AbsTree> = Vec::new();
            for child in display_children {
                child_nodes.push(build_child_node(
                    child,
                    &functions,
                    &types,
                    &get_role,
                    &get_importance,
                    &get_doc,
                    &get_fn_signature,
                    &get_fields,
                    &get_derives,
                ));
                rendered_children.insert((*child).clone());
            }

            if total_children > 8 {
                child_nodes.push(AbsTree::Freeform(format!(
                    "...and {} more",
                    total_children - 5,
                )));
            }

            // Primary module node with enrichment.
            let primary_doc = get_doc(primary_name).map(|d| first_sentence_of(&d));
            let primary_role = get_role(primary_name);
            let primary_imp = get_importance(primary_name);

            body.push(AbsTree::code_module(
                primary_name.clone(),
                primary_role,
                primary_imp,
                primary_doc,
                child_nodes,
            ));

            // Data flow chain.
            if let Some(ref preds) = sem_preds {
                let child_syms: Vec<crate::symbol::SymbolId> = sorted_children
                    .iter()
                    .filter_map(|c| resolve_sym(c))
                    .collect();
                let flow_chain =
                    super::semantic_enrichment::build_flow_chain(engine, preds, &child_syms);
                if flow_chain.len() >= 2 {
                    let steps: Vec<DataFlowStep> = flow_chain
                        .iter()
                        .map(|(name, via)| DataFlowStep {
                            name: name.clone(),
                            via_type: via.clone(),
                        })
                        .collect();
                    body.push(AbsTree::data_flow(steps));
                }
            }

            // Primary entity's own types.
            if let Some(ts) = types.get(primary_name) {
                for t in ts {
                    body.push(build_type_signature(
                        t,
                        &get_importance,
                        &get_fields,
                        &get_derives,
                        &get_doc,
                    ));
                }
            }

            // Primary entity's deps.
            if let Some(ds) = deps.get(primary_name) {
                for d in ds {
                    body.push(AbsTree::CodeFact {
                        kind: "dependency".to_string(),
                        name: primary_name.clone(),
                        detail: d.clone(),
                    });
                }
            }

            rendered_children.insert(primary_name.clone());
        }
    }

    // Render remaining entities not covered by the hierarchical tree.
    for (parent, children) in &modules {
        if rendered_children.contains(parent) {
            continue;
        }
        let doc = get_doc(parent).map(|d| first_sentence_of(&d));
        let role = get_role(parent);
        let imp = get_importance(parent);
        let child_nodes: Vec<AbsTree> = children
            .iter()
            .map(|c| AbsTree::Freeform(format!("`{c}`")))
            .collect();
        body.push(AbsTree::code_module(
            parent.clone(),
            role,
            imp,
            doc,
            child_nodes,
        ));
    }

    for (owner, fns) in &functions {
        if rendered_children.contains(owner) {
            continue;
        }
        for f in fns {
            body.push(build_fn_signature(f, &get_fn_signature, &get_doc));
        }
    }

    for (owner, ts) in &types {
        if rendered_children.contains(owner) {
            continue;
        }
        for t in ts {
            body.push(build_type_signature(
                t,
                &get_importance,
                &get_fields,
                &get_derives,
                &get_doc,
            ));
        }
    }

    for (module, dep_list) in &deps {
        if rendered_children.contains(module) {
            continue;
        }
        for d in dep_list {
            body.push(AbsTree::CodeFact {
                kind: "dependency".to_string(),
                name: module.clone(),
                detail: d.clone(),
            });
        }
    }

    // Append extra items (defined-in facts, etc.).
    body.extend(extra_items);

    // Choose heading based on enrichment availability.
    let has_enrichment =
        sem_preds.is_some() && primary.as_ref().is_some_and(|p| get_role(p).is_some());
    let heading = if has_enrichment {
        "Code Architecture"
    } else {
        "Code Structure"
    };

    AbsTree::section(heading, body)
}

/// Build a child module/signature node for a sub-component.
fn build_child_node(
    child: &str,
    functions: &BTreeMap<String, Vec<String>>,
    types: &BTreeMap<String, Vec<String>>,
    get_role: &dyn Fn(&str) -> Option<String>,
    get_importance: &dyn Fn(&str) -> Option<f32>,
    get_doc: &dyn Fn(&str) -> Option<String>,
    get_fn_signature: &dyn Fn(&str) -> (Vec<String>, Option<String>),
    get_fields: &dyn Fn(&str) -> Vec<String>,
    get_derives: &dyn Fn(&str) -> Vec<String>,
) -> AbsTree {
    let role = get_role(child);
    let imp = get_importance(child);
    let doc = get_doc(child).map(|d| first_sentence_of(&d));

    // Gather child items: key functions and types.
    let mut child_items: Vec<AbsTree> = Vec::new();

    if let Some(fns) = functions.get(child) {
        for f in fns.iter().take(3) {
            child_items.push(build_fn_signature(f, get_fn_signature, get_doc));
        }
    }

    if let Some(ts) = types.get(child) {
        for t in ts {
            child_items.push(build_type_signature(
                t,
                get_importance,
                get_fields,
                get_derives,
                get_doc,
            ));
        }
    }

    AbsTree::code_module(child.to_string(), role, imp, doc, child_items)
}

/// Build a function CodeSignature node.
fn build_fn_signature(
    name: &str,
    get_fn_signature: &dyn Fn(&str) -> (Vec<String>, Option<String>),
    get_doc: &dyn Fn(&str) -> Option<String>,
) -> AbsTree {
    let (params, ret) = get_fn_signature(name);
    let doc = get_doc(name).map(|d| first_sentence_of(&d));

    AbsTree::CodeSignature {
        kind: "function".to_string(),
        name: name.to_string(),
        doc_summary: doc,
        params_or_fields: params,
        return_type: ret,
        traits: Vec::new(),
        importance: None,
    }
}

/// Build a type CodeSignature node (struct/enum/trait).
fn build_type_signature(
    name: &str,
    get_importance: &dyn Fn(&str) -> Option<f32>,
    get_fields: &dyn Fn(&str) -> Vec<String>,
    get_derives: &dyn Fn(&str) -> Vec<String>,
    get_doc: &dyn Fn(&str) -> Option<String>,
) -> AbsTree {
    let fields = get_fields(name);
    let derives = get_derives(name);
    let imp = get_importance(name);
    let doc = get_doc(name).map(|d| first_sentence_of(&d));

    AbsTree::CodeSignature {
        kind: "type".to_string(),
        name: name.to_string(),
        doc_summary: doc,
        params_or_fields: fields,
        return_type: None,
        traits: derives,
        importance: imp,
    }
}

/// Build a Section of Similarity nodes.
fn build_similarity_section(facts: &[ExtractedFact]) -> AbsTree {
    let body: Vec<AbsTree> = facts
        .iter()
        .filter_map(|f| {
            if let FactKind::Similarity { .. } = &f.kind {
                Some(fact_to_abs(f))
            } else {
                None
            }
        })
        .collect();
    AbsTree::section("Related Concepts", body)
}

/// Build a Section of Inference/Derivation nodes.
fn build_reasoning_section(facts: &[ExtractedFact]) -> AbsTree {
    let body: Vec<AbsTree> = facts
        .iter()
        .filter_map(|f| match &f.kind {
            FactKind::Inference {
                expression,
                simplified,
            } => {
                let is_noise = simplified.contains("goal_")
                    || expression.contains("goal_")
                    || expression.is_empty()
                    || simplified.is_empty()
                    || simplified == expression;
                if is_noise { None } else { Some(fact_to_abs(f)) }
            }
            FactKind::Derivation {
                count,
                iterations: _,
            } => {
                if *count > 0 && *count <= 100 {
                    Some(fact_to_abs(f))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();
    AbsTree::section("Reasoning Results", body)
}

/// Build an optional Other section from raw facts.
fn build_other_section(facts: &[ExtractedFact]) -> Option<AbsTree> {
    let body: Vec<AbsTree> = facts
        .iter()
        .filter_map(|f| {
            if let FactKind::Raw(s) = &f.kind {
                let trimmed = s.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(AbsTree::Freeform(trimmed.to_string()))
                }
            } else {
                None
            }
        })
        .collect();

    if body.is_empty() {
        None
    } else {
        Some(AbsTree::section("Other Findings", body))
    }
}

// ── Helpers (duplicated from synthesize.rs to keep this module self-contained) ──

/// Extract the first sentence from a doc comment.
fn first_sentence_of(doc: &str) -> String {
    let trimmed = doc.trim();
    if let Some(pos) = trimmed.find(". ") {
        trimmed[..=pos].to_string()
    } else if let Some(stripped) = trimmed.strip_suffix('.') {
        format!("{stripped}.")
    } else if let Some(pos) = trimmed.find('\n') {
        trimmed[..pos].trim_end().to_string()
    } else {
        trimmed.to_string()
    }
}
