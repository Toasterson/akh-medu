//! Phase 33a — Grammar aggregation rules.
//!
//! Rule-based sentence combining: N related triples become one fluent
//! combined sentence instead of one robotic sentence per triple. Pure
//! [`AbsTree`] transformations applied *before* linearization — no model,
//! fully deterministic.
//!
//! | Rule | Before | After |
//! |---|---|---|
//! | Subject merge | "X is-a Y. X has Z." | "X, a Y, has Z." |
//! | Predicate enumeration | "X has A. X has B. X has C." | "X has A, B, and C." |
//! | Discourse ordering | random triple order | is-a first, then properties, then relations |
//! | Pronoun substitution | "X is a Y. X has Z." | "X is a Y. It has Z." |
//! | Connective insertion | no transitions | continuation sentences get a connective |
//!
//! The transformation folds shared-subject [`AbsTree::Triple`]s into
//! [`AbsTree::Aggregated`] nodes which every concrete grammar knows how to
//! linearize in its own voice. Facts are only merged when they share the
//! same **confidence band** — a speculative fact is never presented under
//! a confident sentence's qualifier.
//!
//! Plan: docs/ai/plans/2026-03-26-phase33-grounded-verbalization.md (33a)
//! ADR:  docs/ai/decisions/043-grounded-verbalization.md

use std::collections::{HashMap, HashSet};

use super::abs::{AbsTree, PredicateGroup};
use super::morpho;

/// Environment knob: set `AKH_GRAMMAR_AGGREGATE=0` (or `false`/`off`) to
/// disable aggregation and fall back to one-sentence-per-triple output.
pub const AGGREGATE_ENV: &str = "AKH_GRAMMAR_AGGREGATE";

/// Configuration for the aggregation pass.
#[derive(Debug, Clone)]
pub struct AggregationConfig {
    /// Master switch. When `false`, [`aggregate_statements`] is the identity.
    pub enabled: bool,
    /// Maximum coordinated verb phrases per sentence. Groups beyond this
    /// split into continuation sentences (pronoun substitution rule).
    pub max_groups_per_sentence: usize,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_groups_per_sentence: 3,
        }
    }
}

impl AggregationConfig {
    /// Read the config from the environment.
    ///
    /// Aggregation defaults to ON; `AKH_GRAMMAR_AGGREGATE=0|false|off`
    /// disables it (useful for A/B comparison and debugging).
    pub fn from_env() -> Self {
        let enabled = std::env::var(AGGREGATE_ENV)
            .map(|v| {
                let v = v.to_lowercase();
                v != "0" && v != "false" && v != "off"
            })
            .unwrap_or(true);
        Self {
            enabled,
            ..Self::default()
        }
    }
}

// ── Fact extraction ──────────────────────────────────────────────────────

/// A triple flattened for grouping: leaf nodes plus their labels and the
/// confidence carried by a `WithConfidence` wrapper (if any).
struct Fact<'a> {
    subject: &'a AbsTree,
    subject_key: String,
    predicate: &'a AbsTree,
    predicate_key: String,
    object: &'a AbsTree,
    object_key: String,
    confidence: Option<f32>,
}

/// Try to view an item as an aggregatable fact.
///
/// Only plain triples (optionally wrapped in a single `WithConfidence`)
/// whose subject/predicate/object are all leaves qualify. Everything else
/// (provenance-tagged facts, nested structures, non-triples) passes through
/// aggregation untouched — we never merge facts whose qualifiers we cannot
/// preserve faithfully.
fn extract_fact(item: &AbsTree) -> Option<Fact<'_>> {
    let (triple, confidence) = match item {
        AbsTree::Triple { .. } => (item, None),
        AbsTree::WithConfidence { inner, confidence } => match inner.as_ref() {
            AbsTree::Triple { .. } => (inner.as_ref(), Some(*confidence)),
            _ => return None,
        },
        _ => return None,
    };

    let AbsTree::Triple {
        subject,
        predicate,
        object,
    } = triple
    else {
        return None;
    };

    let subject_label = subject.label()?;
    let predicate_label = predicate.label()?;
    let object_label = object.label()?;

    Some(Fact {
        subject,
        subject_key: subject_label.to_lowercase(),
        predicate,
        predicate_key: predicate_label.to_lowercase(),
        object,
        object_key: object_label.to_lowercase(),
        confidence,
    })
}

/// Confidence band. Facts are only merged within one band so the combined
/// sentence's confidence qualifier stays truthful for every fact in it.
/// Thresholds mirror the narrative grammar's qualifier boundaries.
fn confidence_band(confidence: Option<f32>) -> u8 {
    match confidence {
        None => 4,
        Some(c) if c > 0.9 => 3,
        Some(c) if c > 0.7 => 2,
        Some(c) if c > 0.5 => 1,
        Some(_) => 0,
    }
}

/// Discourse ordering rank: `is-a` first, then properties, then relations.
fn predicate_rank(predicate_key: &str) -> u8 {
    if is_copula(predicate_key) {
        0
    } else if predicate_key.starts_with("has") {
        1
    } else {
        2
    }
}

/// Is this predicate the classification copula (`is-a`)?
fn is_copula(predicate_key: &str) -> bool {
    predicate_key == "is-a" || predicate_key == "is_a"
}

// ── Aggregation ──────────────────────────────────────────────────────────

/// Fold shared-subject triples in `items` into [`AbsTree::Aggregated`]
/// statements. Non-triple items and single-fact subjects pass through
/// byte-identical; original discourse order is preserved (each aggregated
/// sentence appears where its first contributing fact was).
pub fn aggregate_statements(items: &[AbsTree], cfg: &AggregationConfig) -> Vec<AbsTree> {
    if !cfg.enabled || items.len() < 2 {
        return items.to_vec();
    }

    // Pass 1: group aggregatable facts by (subject, confidence band).
    type GroupKey = (String, u8);
    let mut groups: HashMap<GroupKey, Vec<Fact<'_>>> = HashMap::new();
    // Per input item: which group it landed in (None = passthrough).
    let mut slots: Vec<Option<GroupKey>> = Vec::with_capacity(items.len());

    for item in items {
        match extract_fact(item) {
            Some(fact) => {
                let key = (fact.subject_key.clone(), confidence_band(fact.confidence));
                groups.entry(key.clone()).or_default().push(fact);
                slots.push(Some(key));
            }
            None => slots.push(None),
        }
    }

    // Pass 2: emit in original order; each group is emitted (aggregated)
    // at the position of its first member.
    let mut emitted: HashSet<GroupKey> = HashSet::new();
    let mut result: Vec<AbsTree> = Vec::with_capacity(items.len());

    for (item, slot) in items.iter().zip(&slots) {
        match slot {
            None => result.push(item.clone()),
            Some(key) => {
                if !emitted.insert(key.clone()) {
                    continue;
                }
                let facts = &groups[key];
                if facts.len() == 1 {
                    // Single fact — keep the original node untouched.
                    result.push(item.clone());
                } else {
                    result.extend(aggregate_group(facts, cfg));
                }
            }
        }
    }

    // Pass 3: pronoun substitution — consecutive sentences about the same
    // subject become continuations ("X is a Y. It has Z.").
    apply_continuations(&mut result);

    result
}

/// Fold one subject's same-band facts into one or more `Aggregated` nodes.
fn aggregate_group(facts: &[Fact<'_>], cfg: &AggregationConfig) -> Vec<AbsTree> {
    debug_assert!(facts.len() >= 2);

    // Discourse ordering: is-a first, then has-*, then other relations.
    let mut ordered: Vec<&Fact<'_>> = facts.iter().collect();
    ordered.sort_by_key(|f| predicate_rank(&f.predicate_key));

    // Predicate enumeration: fold same-predicate facts into one group,
    // deduplicating repeated objects.
    let mut pred_order: Vec<String> = Vec::new();
    let mut by_pred: HashMap<String, (&AbsTree, Vec<(&AbsTree, &str)>)> = HashMap::new();
    for fact in &ordered {
        let entry = by_pred
            .entry(fact.predicate_key.clone())
            .or_insert_with(|| {
                pred_order.push(fact.predicate_key.clone());
                (fact.predicate, Vec::new())
            });
        if !entry.1.iter().any(|(_, key)| *key == fact.object_key) {
            entry.1.push((fact.object, &fact.object_key));
        }
    }

    // Subject merge: is-a objects become appositions when other predicate
    // groups exist; a pure-is-a group stays a copula sentence.
    let mut appositions: Vec<AbsTree> = Vec::new();
    let mut vp_groups: Vec<PredicateGroup> = Vec::new();
    let has_non_copula = pred_order.iter().any(|p| !is_copula(p));

    for pred_key in &pred_order {
        let (pred_node, objects) = &by_pred[pred_key];
        let object_nodes: Vec<AbsTree> = objects.iter().map(|(node, _)| (*node).clone()).collect();
        if is_copula(pred_key) && has_non_copula {
            appositions.extend(object_nodes);
        } else {
            vp_groups.push(PredicateGroup {
                predicate: (*pred_node).clone(),
                objects: object_nodes,
            });
        }
    }

    // The combined sentence carries the group's most conservative
    // confidence (the band guarantees the qualifier reads the same).
    let min_confidence = facts
        .iter()
        .filter_map(|f| f.confidence)
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let subject = facts[0].subject.clone();
    let chunk_size = cfg.max_groups_per_sentence.max(1);

    // Sentence splitting: too many verb phrases → continuation sentences.
    let mut nodes: Vec<AbsTree> = Vec::new();
    if vp_groups.is_empty() {
        // Only reachable if every predicate was copular AND has_non_copula
        // was false — then vp_groups holds the copula group. Defensive:
        // realize appositions as a copula sentence instead of dropping them.
        nodes.push(AbsTree::aggregated(
            subject.clone(),
            appositions,
            Vec::new(),
            false,
        ));
    } else {
        for (i, chunk) in vp_groups.chunks(chunk_size).enumerate() {
            let apps = if i == 0 {
                std::mem::take(&mut appositions)
            } else {
                Vec::new()
            };
            nodes.push(AbsTree::aggregated(
                subject.clone(),
                apps,
                chunk.to_vec(),
                false,
            ));
        }
    }

    // Re-wrap with the confidence stripped during extraction.
    if let Some(conf) = min_confidence {
        nodes = nodes
            .into_iter()
            .map(|inner| AbsTree::WithConfidence {
                inner: Box::new(inner),
                confidence: conf,
            })
            .collect();
    }

    nodes
}

/// Mark aggregated sentences that immediately follow another statement
/// about the same subject as continuations (pronoun substitution rule).
fn apply_continuations(nodes: &mut [AbsTree]) {
    let mut prev_subject: Option<String> = None;
    for node in nodes.iter_mut() {
        let current = statement_subject_key(node);
        if let (Some(prev), Some(cur)) = (&prev_subject, &current) {
            if prev == cur {
                mark_continuation(node);
            }
        }
        prev_subject = current;
    }
}

/// Subject key of a statement node (unwrapping confidence/provenance).
fn statement_subject_key(node: &AbsTree) -> Option<String> {
    match node {
        AbsTree::WithConfidence { inner, .. } | AbsTree::WithProvenance { inner, .. } => {
            statement_subject_key(inner)
        }
        AbsTree::Triple { subject, .. } | AbsTree::Aggregated { subject, .. } => {
            subject.label().map(str::to_lowercase)
        }
        _ => None,
    }
}

/// Set the continuation flag on an `Aggregated` node (through wrappers).
fn mark_continuation(node: &mut AbsTree) {
    match node {
        AbsTree::WithConfidence { inner, .. } | AbsTree::WithProvenance { inner, .. } => {
            mark_continuation(inner)
        }
        AbsTree::Aggregated { continuation, .. } => *continuation = true,
        _ => {}
    }
}

// ── Shared English realization helpers ───────────────────────────────────

/// Build a coordinated verb phrase from a *humanized* predicate and
/// linearized object strings.
///
/// - `("is a", [mammal])` → "is a mammal"
/// - `("is a", [mammal, pet])` → "is a mammal and a pet"
/// - `("has", [fur, tail, whiskers])` → "has fur, tail, and whiskers"
pub fn english_verb_phrase(humanized_predicate: &str, objects: &[String]) -> String {
    if humanized_predicate == "is a" {
        let parts: Vec<String> = objects
            .iter()
            .map(|o| format!("{} {o}", morpho::article(o)))
            .collect();
        format!("is {}", morpho::join_list(&parts, "and"))
    } else {
        format!(
            "{humanized_predicate} {}",
            morpho::join_list(objects, "and")
        )
    }
}

/// Render apposition labels with articles: `[mammal]` → "a mammal",
/// `[mammal, predator]` → "a mammal and a predator".
pub fn apposition_phrase(labels: &[String]) -> String {
    let parts: Vec<String> = labels
        .iter()
        .map(|l| format!("{} {l}", morpho::article(l)))
        .collect();
    morpho::join_list(&parts, "and")
}

/// Neutral (grammar-independent) English realization of a statement node.
///
/// Used by chat surfaces that compose prose from grounded triples outside
/// the grammar registry (Phase 34 communicative pipeline, grounded-response
/// fallbacks). Returns the sentence body without terminal punctuation or
/// capitalization; `None` for nodes that are not statements. Confidence and
/// provenance wrappers are transparent (those surfaces render qualifiers
/// separately or not at all).
pub fn realize_statement(tree: &AbsTree) -> Option<String> {
    match tree {
        AbsTree::WithConfidence { inner, .. }
        | AbsTree::WithProvenance { inner, .. }
        | AbsTree::DiscourseFrame { inner, .. } => realize_statement(inner),

        AbsTree::Triple {
            subject,
            predicate,
            object,
        } => {
            let s = subject.label()?;
            let p = morpho::humanize_predicate(predicate.label()?);
            let o = object.label()?.to_string();
            Some(format!("{s} {}", english_verb_phrase(&p, &[o])))
        }

        AbsTree::Aggregated {
            subject,
            appositions,
            groups,
            continuation,
        } => {
            let mut out = if *continuation {
                "it".to_string()
            } else {
                subject.label()?.to_string()
            };

            if !appositions.is_empty() {
                let labels: Vec<String> = appositions
                    .iter()
                    .filter_map(|a| a.label().map(str::to_string))
                    .collect();
                if !labels.is_empty() {
                    out.push_str(&format!(", {},", apposition_phrase(&labels)));
                }
            }

            let vps: Vec<String> = groups
                .iter()
                .filter_map(|g| {
                    let p = morpho::humanize_predicate(g.predicate.label()?);
                    let objects: Vec<String> = g
                        .objects
                        .iter()
                        .filter_map(|o| o.label().map(str::to_string))
                        .collect();
                    if objects.is_empty() {
                        None
                    } else {
                        Some(english_verb_phrase(&p, &objects))
                    }
                })
                .collect();

            if vps.is_empty() && appositions.is_empty() {
                return None;
            }
            if !vps.is_empty() {
                out.push(' ');
                out.push_str(&morpho::join_list(&vps, "and"));
            } else {
                // Apposition-only node: realize as a copula sentence.
                let labels: Vec<String> = appositions
                    .iter()
                    .filter_map(|a| a.label().map(str::to_string))
                    .collect();
                out = if *continuation {
                    "it".to_string()
                } else {
                    subject.label()?.to_string()
                };
                out.push_str(&format!(" is {}", apposition_phrase(&labels)));
            }
            Some(out)
        }

        _ => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn t(s: &str, p: &str, o: &str) -> AbsTree {
        AbsTree::triple(AbsTree::entity(s), AbsTree::relation(p), AbsTree::entity(o))
    }

    fn tc(s: &str, p: &str, o: &str, conf: f32) -> AbsTree {
        AbsTree::WithConfidence {
            inner: Box::new(t(s, p, o)),
            confidence: conf,
        }
    }

    fn cfg() -> AggregationConfig {
        AggregationConfig::default()
    }

    fn realize_all(nodes: &[AbsTree]) -> Vec<String> {
        nodes.iter().filter_map(realize_statement).collect()
    }

    // ── Rule: predicate enumeration ──────────────────────────────────

    #[test]
    fn predicate_enumeration_merges_objects() {
        let items = vec![
            t("dog", "has-a", "fur"),
            t("dog", "has-a", "tail"),
            t("dog", "has-a", "whiskers"),
        ];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 1, "three same-predicate triples → one node");
        let prose = realize_statement(&out[0]).unwrap();
        assert_eq!(prose, "dog has fur, tail, and whiskers");
    }

    #[test]
    fn predicate_enumeration_dedups_objects() {
        let items = vec![t("dog", "has-a", "fur"), t("dog", "has-a", "fur")];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 1);
        let prose = realize_statement(&out[0]).unwrap();
        assert_eq!(prose, "dog has fur");
    }

    // ── Rule: subject merge (apposition) ─────────────────────────────

    #[test]
    fn subject_merge_makes_apposition() {
        let items = vec![t("dog", "is-a", "mammal"), t("dog", "has-a", "fur")];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 1);
        let prose = realize_statement(&out[0]).unwrap();
        assert_eq!(prose, "dog, a mammal, has fur");
    }

    #[test]
    fn pure_is_a_group_stays_copular() {
        let items = vec![t("dog", "is-a", "mammal"), t("dog", "is-a", "predator")];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 1);
        let prose = realize_statement(&out[0]).unwrap();
        assert_eq!(prose, "dog is a mammal and a predator");
    }

    // ── Rule: discourse ordering ─────────────────────────────────────

    #[test]
    fn discourse_ordering_is_a_first() {
        // is-a arrives LAST but must lead the combined sentence.
        let items = vec![
            t("dog", "lives-in", "forests"),
            t("dog", "has-a", "fur"),
            t("dog", "is-a", "mammal"),
        ];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 1);
        let prose = realize_statement(&out[0]).unwrap();
        assert_eq!(prose, "dog, a mammal, has fur and lives in forests");
    }

    // ── Rule: pronoun substitution + sentence splitting ──────────────

    #[test]
    fn sentence_split_uses_pronoun_continuation() {
        let items = vec![
            t("dog", "is-a", "mammal"),
            t("dog", "has-a", "fur"),
            t("dog", "lives-in", "forests"),
            t("dog", "chases", "cats"),
            t("dog", "guards", "houses"),
        ];
        let config = AggregationConfig {
            max_groups_per_sentence: 2,
            ..cfg()
        };
        let out = aggregate_statements(&items, &config);
        assert_eq!(out.len(), 2, "four VP groups at max 2/sentence → 2 nodes");
        let prose = realize_all(&out);
        assert_eq!(prose[0], "dog, a mammal, has fur and lives in forests");
        assert_eq!(prose[1], "it chases cats and guards houses");
    }

    #[test]
    fn continuation_across_confidence_bands() {
        // Same subject, different bands → two sentences; the second uses
        // the pronoun because it directly follows the first.
        let items = vec![
            tc("dog", "is-a", "mammal", 0.95),
            tc("dog", "has-a", "fur", 0.92),
            tc("dog", "similar-to", "wolf", 0.85),
            tc("dog", "part-of", "pack", 0.8),
        ];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 2, "two bands → two sentences");
        let prose = realize_all(&out);
        assert_eq!(prose[0], "dog, a mammal, has fur");
        assert_eq!(prose[1], "it is similar to wolf and is part of pack");
        // Confidence wrappers survive with the band's minimum.
        match &out[0] {
            AbsTree::WithConfidence { confidence, .. } => {
                assert!(
                    (confidence - 0.92).abs() < 1e-6,
                    "min of band, got {confidence}"
                )
            }
            other => panic!("expected WithConfidence wrapper, got {other:?}"),
        }
    }

    #[test]
    fn different_confidence_bands_never_merge() {
        let items = vec![
            tc("dog", "is-a", "mammal", 0.95),
            tc("dog", "is-a", "reptile", 0.3),
        ];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(
            out.len(),
            2,
            "confident and speculative facts must stay separate"
        );
    }

    // ── Grouping & passthrough behaviour ─────────────────────────────

    #[test]
    fn different_subjects_stay_separate() {
        let items = vec![
            t("dog", "has-a", "fur"),
            t("cat", "has-a", "whiskers"),
            t("dog", "is-a", "mammal"),
        ];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 2);
        let prose = realize_all(&out);
        // Dog's group is emitted at the position of its FIRST fact.
        assert_eq!(prose[0], "dog, a mammal, has fur");
        assert_eq!(prose[1], "cat has whiskers");
    }

    #[test]
    fn non_triples_pass_through_untouched() {
        let gap = AbsTree::gap(AbsTree::entity("dog"), "diet unknown");
        let items = vec![
            t("dog", "is-a", "mammal"),
            gap.clone(),
            t("dog", "has-a", "fur"),
        ];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 2);
        assert_eq!(out[1], gap, "non-statement nodes must be byte-identical");
    }

    #[test]
    fn single_facts_pass_through_untouched() {
        let items = vec![t("dog", "is-a", "mammal"), t("cat", "is-a", "mammal")];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out, items, "no shared subject → identity");
    }

    #[test]
    fn disabled_config_is_identity() {
        let items = vec![t("dog", "is-a", "mammal"), t("dog", "has-a", "fur")];
        let config = AggregationConfig {
            enabled: false,
            ..cfg()
        };
        assert_eq!(aggregate_statements(&items, &config), items);
    }

    #[test]
    fn provenance_wrapped_facts_pass_through() {
        use crate::grammar::abs::ProvenanceTag;
        let wrapped = AbsTree::WithProvenance {
            inner: Box::new(t("dog", "has-a", "fur")),
            tag: ProvenanceTag::Extracted,
        };
        let items = vec![t("dog", "is-a", "mammal"), wrapped.clone()];
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 2);
        assert_eq!(out[1], wrapped);
    }

    // ── Validation of produced trees ─────────────────────────────────

    #[test]
    fn aggregated_output_validates() {
        let items = vec![
            t("dog", "is-a", "mammal"),
            t("dog", "has-a", "fur"),
            t("dog", "has-a", "tail"),
            t("dog", "lives-in", "forests"),
        ];
        for node in aggregate_statements(&items, &cfg()) {
            node.validate()
                .expect("aggregated trees must be well-typed");
        }
    }

    // ── Before/after fixture ─────────────────────────────────────────

    #[test]
    fn before_after_fixture_n_triples_one_sentence() {
        // BEFORE (one sentence per triple):
        //   "dog is a mammal. dog has fur. dog has a tail.
        //    dog lives in forests."
        let items = vec![
            t("dog", "is-a", "mammal"),
            t("dog", "has-a", "fur"),
            t("dog", "has-a", "tail"),
            t("dog", "lives-in", "forests"),
        ];
        let before: Vec<String> = items.iter().filter_map(realize_statement).collect();
        assert_eq!(before.len(), 4);

        // AFTER: one fluent combined sentence.
        let out = aggregate_statements(&items, &cfg());
        assert_eq!(out.len(), 1);
        let after = realize_statement(&out[0]).unwrap();
        assert_eq!(
            after,
            "dog, a mammal, has fur and tail and lives in forests"
        );
    }

    // ── Env knob ─────────────────────────────────────────────────────

    #[test]
    fn env_config_default_is_enabled() {
        // NB: does not set the env var (tests run in parallel); just checks
        // the default path when the variable is absent or unrelated.
        let config = AggregationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_groups_per_sentence, 3);
    }
}
