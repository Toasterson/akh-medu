//! Forward-chaining rule engine with pattern matching.
//!
//! Runs inference rules to a fixpoint or max iteration cap,
//! producing derived triples with full provenance.

use std::collections::{HashMap, HashSet};

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};
use crate::symbol::SymbolId;

use super::error::AutonomousResult;
use super::rules::{InferenceRule, RuleSet, RuleTerm, TriplePattern};

// ---------------------------------------------------------------------------
// Noise entity filter
// ---------------------------------------------------------------------------

/// Returns `true` if an entity name matches known noise/artifact patterns from
/// agent learning and scaffolding. Entities matching these patterns should be
/// excluded from rule inference to prevent nonsense transitive closures (e.g.
/// `Tecton → is-a → Stor01` through intermediate tool-output entities).
///
/// Also used by the extraction pipeline to reject scaffolding entities before
/// they are persisted to the knowledge graph (episode bleed prevention).
pub fn is_noise_entity(name: &str) -> bool {
    // Agent learning artifacts
    name.starts_with("episode:")
    // Agent goal records
    || name.starts_with("goal:")
    // Agent criteria patterns
    || name.starts_with("criteria:")
    // Agent scaffolding markers
    || name.starts_with("expand:")
    || name.starts_with("desc:")
    // Numeric rate/certainty markers like "rate:0.8000"
    || name.starts_with("rate:")
    // Tool invocation names like "tool:similarity_search", "tool:memory_recall"
    || name.starts_with("tool:")
    // Status literals like "status:dormant", "status:active"
    || name.starts_with("status:")
    // Priority literals like "priority:69", "priority:200"
    || name.starts_with("priority:")
    // Bare numeric strings like "0.9910" or "69"
    || is_numeric_string(name)
    // File paths and document names (SOUL.md, memory/, src/agent/foo.rs)
    || is_file_path(name)
    // UUID-like identifiers used for episodes, work items, etc.
    || is_uuid_like(name)
}

/// Returns `true` if the entity is a service-topology or infrastructure type
/// that should be excluded from structural rules such as `contains-part-inverse`.
///
/// Checked by walking the entity's outgoing `is-a` triples.  Returns `false`
/// quickly when `guards` is empty or when the KG has no `is-a` predicate yet.
fn is_entity_type_guarded(entity: SymbolId, guards: &[String], engine: &Engine) -> bool {
    if guards.is_empty() {
        return false;
    }
    let isa_id = match engine.lookup_symbol("is-a") {
        Ok(id) => id,
        Err(_) => return false,
    };
    for triple in engine.triples_from(entity) {
        if triple.predicate == isa_id {
            let obj_label = engine.resolve_label(triple.object);
            if guards.iter().any(|g| g == &obj_label) {
                return true;
            }
        }
    }
    false
}

/// Returns `true` if the string is purely numeric (digits and at most one decimal point).
fn is_numeric_string(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut dot_count = 0;
    for ch in s.chars() {
        if ch == '.' {
            dot_count += 1;
            if dot_count > 1 {
                return false;
            }
        } else if !ch.is_ascii_digit() {
            return false;
        }
    }
    true
}

/// Returns `true` if the string looks like a file path or document name.
///
/// Matches:
/// - Directory-like strings ending with `/` (e.g. `memory/`)
/// - Strings ending with a known file extension (e.g. `SOUL.md`, `src/agent/foo.rs`)
fn is_file_path(s: &str) -> bool {
    if s.ends_with('/') {
        return true;
    }
    const FILE_EXTS: &[&str] = &[
        ".md", ".rs", ".toml", ".json", ".py", ".ts", ".js", ".yaml", ".yml", ".txt", ".sh",
        ".html", ".xml",
    ];
    if FILE_EXTS.iter().any(|ext| s.ends_with(ext)) {
        // Accept if it has a path separator or has no spaces (bare filename)
        return s.contains('/') || !s.contains(' ');
    }
    false
}

/// Returns `true` if the string looks like a UUID (8-4-4-4-12 lowercase hex).
///
/// Matches both hyphenated (`019e4909-cf4b-70e1-a703-287c3fdea3ef`) and
/// compact (`019e4909cf4b70e1a703287c3fdea3ef`) forms of 128-bit identifiers.
fn is_uuid_like(s: &str) -> bool {
    // Hyphenated UUID: exactly 36 chars
    if s.len() == 36 {
        let b = s.as_bytes();
        if b[8] == b'-' && b[13] == b'-' && b[18] == b'-' && b[23] == b'-' {
            return s
                .chars()
                .filter(|&c| c != '-')
                .all(|c| c.is_ascii_hexdigit());
        }
    }
    // Compact UUID: exactly 32 hex chars
    if s.len() == 32 {
        return s.chars().all(|c| c.is_ascii_hexdigit());
    }
    false
}

// ---------------------------------------------------------------------------
// Semantic type-compatibility guard
// ---------------------------------------------------------------------------

/// Returns `true` if the label looks like a snake_case tool or function name
/// (e.g. `infer_rules`, `kg_query`, `code_ingest`).  These identifiers are
/// internal artefacts and must not appear as nodes in a type hierarchy.
fn is_snake_case_tool_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.contains('_')
        && !s.contains('-')
        && !s.contains(' ')
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// Returns `true` if the label matches known CI/CD or storage infrastructure
/// entity patterns.  Such labels must not become type-hierarchy parents of
/// agent identity entities (e.g. `Ptah is-a ForgejoActions` is semantically
/// invalid).
fn is_infrastructure_entity(s: &str) -> bool {
    let lower = s.to_ascii_lowercase();
    // Forgejo CI/CD platform (ForgejoActions, Forgejo, forgejo-runner, …)
    lower.contains("forgejo")
    // Generic CI runner labels
    || lower == "ci runner"
    || lower == "ci-runner"
    || lower.starts_with("ci runner ")
    || lower.starts_with("ci-runner")
    // Numbered CI runner instances like "runner-01", "runner_prod" (not standalone "runner")
    || lower.starts_with("runner-")
    || lower.starts_with("runner_")
    // Storage node labels: "Storage", "Storage-01", "Stor01", "StorageNode"
    || lower == "storage"
    || lower.starts_with("storage ")
    || lower.starts_with("storage-")
    || lower.starts_with("storage_")
    || lower.starts_with("storagenode")
    // Numbered storage nodes like "Stor01", "Stor-1" (prefix + digit or dash)
    || (lower.starts_with("stor")
        && lower.len() > 4
        && lower[4..].starts_with(|c: char| c.is_ascii_digit() || c == '-'))
}

/// Returns `true` if the entity name is a known agent identity (persona).
///
/// Identity entities are symbolic personas drawn from mythology, fiction, or
/// history — they must not gain spurious `is-a` links to operational
/// infrastructure via transitive closure.  The VSA similarity guard uses this
/// predicate to decide when to apply the secondary (embedding-space) check.
///
/// The set covers the Ennead AI collective plus common agent names drawn from
/// the same cultural pools used by the identity bootstrap system.
pub fn is_identity_entity(name: &str) -> bool {
    matches!(
        name,
        // Egyptian Ennead and associated figures (primary Akh-Medu personas)
        "Ptah" | "Thoth" | "Ra" | "Re" | "Anubis" | "Khnum" | "Horus" | "Isis" | "Osiris"
            | "Seth" | "Nephthys" | "Nut" | "Geb" | "Tefnut" | "Shu" | "Atum" | "Khepri"
            // Tecton and the broader Akh-Medu collective
            | "Tecton" | "Akhu" | "Akh"
            // Greek / Norse agent personas
            | "Athena" | "Apollo" | "Hermes" | "Ares" | "Zeus" | "Hera"
            | "Odin" | "Thor" | "Loki" | "Freya"
            // Fictional / historical agent personas
            | "Gandalf" | "Sherlock" | "Spock" | "Turing"
    )
}

/// Returns `true` if the label is a standard UUID (8-4-4-4-12 hex groups).
fn is_uuid(s: &str) -> bool {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let expected = [8usize, 4, 4, 4, 12];
    parts
        .iter()
        .zip(expected.iter())
        .all(|(part, &len)| part.len() == len && part.chars().all(|c| c.is_ascii_hexdigit()))
}

/// Returns `true` when `entity_label` must not participate in a type-hierarchy
/// assertion (`is-a` or `child-of` predicate).
///
/// Blocks cross-domain contamination: agent identity nodes must not gain
/// spurious `is-a` / `child-of` links to CI runners, storage nodes, tool-name
/// stubs, UUIDs, or other operational artefacts that have no place in the
/// concept taxonomy.  Apply this to **both** the subject and the object of
/// any candidate derived triple.
pub fn is_invalid_type_hierarchy_participant(pred_label: &str, entity_label: &str) -> bool {
    if !matches!(pred_label, "is-a" | "child-of") {
        return false;
    }
    is_noise_entity(entity_label)
        || is_snake_case_tool_name(entity_label)
        || is_infrastructure_entity(entity_label)
        || is_uuid(entity_label)
}

// ---------------------------------------------------------------------------
// VSA similarity guard for Identity entities
// ---------------------------------------------------------------------------

/// Minimum Hamming similarity required when the proposed object of an `is-a`
/// derivation is checked against a known Identity entity subject.
///
/// Calibration: random unrelated bipolar hypervectors have expected similarity
/// ≈ 0.50.  A threshold of **0.40** therefore has no effect on freshly-allocated
/// (ungrounded) vectors — it only blocks strongly dissimilar pairs that arise
/// after semantic grounding pushes infrastructure embeddings away from identity
/// embeddings.  Genuinely related concepts ground above ≈ 0.60; contaminated
/// CI-runner / storage-node targets ground well below 0.40.
pub const IDENTITY_VSA_GUARD_THRESHOLD: f32 = 0.40;

/// VSA similarity guard for Identity entity type-hierarchy assertions.
///
/// When the proposed subject of an `is-a` or `child-of` derivation is a known
/// Identity entity (see [`is_identity_entity`]), the object must be semantically
/// plausible in VSA space.  Returns `true` (**block** the derivation) when:
///
/// * the subject label matches [`is_identity_entity`], **and**
/// * both hypervectors exist in the engine's item memory, **and**
/// * their Hamming similarity is below `threshold`.
///
/// Returns `false` (allow) if the subject is not an identity entity, or if
/// either vector is absent from item memory.  This prevents the guard from
/// accidentally suppressing valid derivations in freshly-bootstrapped workspaces
/// where vectors have not yet been semantically grounded (random pairs sit near
/// ≈ 0.50, well above the 0.40 threshold).
pub fn identity_vsa_similarity_guard(
    engine: &Engine,
    subject: SymbolId,
    object: SymbolId,
    threshold: f32,
) -> bool {
    let subj_label = engine.resolve_label(subject);
    if !is_identity_entity(&subj_label) {
        return false;
    }

    let im = engine.item_memory();
    let subj_vec = match im.get(subject) {
        Some(v) => v,
        None => return false,
    };
    let obj_vec = match im.get(object) {
        Some(v) => v,
        None => return false,
    };

    match engine.ops().similarity(&subj_vec, &obj_vec) {
        Ok(sim) => sim < threshold,
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the forward-chaining rule engine.
#[derive(Debug, Clone)]
pub struct RuleEngineConfig {
    /// Maximum forward-chaining iterations (default: 10).
    pub max_iterations: usize,
    /// Minimum confidence for derived triples (default: 0.1).
    pub min_confidence: f32,
    /// Whether to add derived triples to the KG immediately (default: true).
    pub auto_commit: bool,
    /// Hard cap on total new triples (default: 1000).
    pub max_new_triples: usize,
}

impl Default for RuleEngineConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            min_confidence: 0.1,
            auto_commit: true,
            max_new_triples: 1000,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// A single derived triple with provenance.
#[derive(Debug, Clone)]
pub struct DerivedTriple {
    pub triple: Triple,
    pub rule_name: String,
    pub antecedent_triples: Vec<Triple>,
    pub confidence: f32,
    pub iteration: usize,
}

/// Result of running the rule engine.
#[derive(Debug, Clone)]
pub struct RuleEngineResult {
    pub derived: Vec<DerivedTriple>,
    pub iterations: usize,
    pub reached_fixpoint: bool,
    /// Per-rule derivation counts.
    pub rule_stats: HashMap<String, usize>,
}

// ---------------------------------------------------------------------------
// Rule engine
// ---------------------------------------------------------------------------

/// Forward-chaining rule engine.
pub struct RuleEngine {
    config: RuleEngineConfig,
    rule_sets: Vec<RuleSet>,
}

impl RuleEngine {
    pub fn new(config: RuleEngineConfig) -> Self {
        Self {
            config,
            rule_sets: Vec::new(),
        }
    }

    pub fn with_rules(mut self, rules: RuleSet) -> Self {
        self.rule_sets.push(rules);
        self
    }

    /// Run forward-chaining on the given engine's knowledge graph.
    pub fn run(&self, engine: &Engine) -> AutonomousResult<RuleEngineResult> {
        let mut derived: Vec<DerivedTriple> = Vec::new();
        let mut rule_stats: HashMap<String, usize> = HashMap::new();
        // Track derived triples as (s, p, o) to avoid duplicates.
        let mut derived_set: HashSet<(u64, u64, u64)> = HashSet::new();

        let all_rules: Vec<&InferenceRule> = self
            .rule_sets
            .iter()
            .flat_map(|rs| rs.rules.iter())
            .filter(|r| r.enabled)
            .collect();

        let mut iteration = 0;
        let mut reached_fixpoint = false;

        for iter in 0..self.config.max_iterations {
            iteration = iter + 1;
            let mut new_this_round: Vec<DerivedTriple> = Vec::new();

            for rule in &all_rules {
                let bindings_list = self.match_antecedents(rule, engine)?;
                let mut legacy_support_counts = HashMap::<Vec<SymbolId>, usize>::new();
                for (_, matched) in &bindings_list {
                    *legacy_support_counts
                        .entry(matched.iter().map(|triple| triple.subject).collect())
                        .or_default() += 1;
                }

                let mut rule_derived_count = 0;
                for (bindings, matched_triples) in &bindings_list {
                    if rule_derived_count >= rule.max_derivations_per_iteration {
                        break;
                    }
                    let legacy_subjects = matched_triples
                        .iter()
                        .map(|premise| premise.subject)
                        .collect::<Vec<_>>();
                    let legacy_support_unique =
                        legacy_support_counts.get(&legacy_subjects) == Some(&1);

                    for consequent in &rule.consequents {
                        if let Some(triple) =
                            self.instantiate_pattern(consequent, bindings, engine)?
                        {
                            // Semantic type-compatibility guard: reject derivations
                            // that would link type-hierarchy predicates to infrastructure
                            // entities, tool-name stubs, UUIDs, or other non-taxonomic
                            // artefacts (fixes KG contamination from the P2 infer_rules bug).
                            let pred_label = engine.resolve_label(triple.predicate);
                            let subj_label = engine.resolve_label(triple.subject);
                            let obj_label = engine.resolve_label(triple.object);
                            if is_invalid_type_hierarchy_participant(&pred_label, &subj_label)
                                || is_invalid_type_hierarchy_participant(&pred_label, &obj_label)
                            {
                                continue;
                            }

                            // Self-reference guard: skip any derived triple where the
                            // subject and object resolve to the same symbol. Transitive
                            // rules (is-a-transitive, parent-child-inverse) can produce
                            // X → P → X loops; discard them before they enter the KG.
                            if triple.subject == triple.object {
                                tracing::debug!(
                                    rule = %rule.name,
                                    symbol = triple.subject.get(),
                                    predicate = %pred_label,
                                    "rule engine discarding self-referential derived triple"
                                );
                                continue;
                            }

                            // VSA similarity guard: defence-in-depth for Identity entities.
                            // After semantic grounding, identity (persona) embeddings diverge
                            // from infrastructure embeddings; this guard blocks residual
                            // cross-domain contamination that escapes the pattern filter.
                            // The threshold (0.40) is below the ≈0.50 random baseline so
                            // ungrounded fresh vectors are never falsely rejected.
                            if matches!(pred_label.as_str(), "is-a" | "child-of")
                                && identity_vsa_similarity_guard(
                                    engine,
                                    triple.subject,
                                    triple.object,
                                    IDENTITY_VSA_GUARD_THRESHOLD,
                                )
                            {
                                continue;
                            }

                            let key = (
                                triple.subject.get(),
                                triple.predicate.get(),
                                triple.object.get(),
                            );

                            if derived_set.contains(&key) {
                                continue;
                            }

                            // Upgrade an existing pre-contract rule result only
                            // when its old ledger record identifies this exact
                            // rule application unambiguously.  Never relabel an
                            // arbitrary asserted fact merely because a rule can
                            // also derive it.
                            if let Some(existing) = engine
                                .triples_from(triple.subject)
                                .into_iter()
                                .find(|candidate| {
                                    candidate.predicate == triple.predicate
                                        && candidate.object == triple.object
                                })
                            {
                                if existing.provenance_id.is_none()
                                    && let Some(provenance_id) = recover_rule_provenance_id(
                                        engine,
                                        rule,
                                        &triple,
                                        matched_triples,
                                        legacy_support_unique,
                                    )
                                {
                                    let mut repaired = existing;
                                    repaired.provenance_id = Some(provenance_id.get());
                                    engine.add_triple(&repaired)?;
                                }
                                continue;
                            }

                            let confidence =
                                rule.confidence_factor * avg_confidence(matched_triples);
                            if confidence < self.config.min_confidence {
                                continue;
                            }

                            // Per-rule antecedent confidence floor: if the
                            // weakest antecedent triple is below the threshold,
                            // skip this binding to prevent low-quality source
                            // edges from spawning derived topology.
                            if rule.min_antecedent_confidence > 0.0 {
                                let weakest = matched_triples
                                    .iter()
                                    .map(|t| t.confidence)
                                    .fold(f32::MAX, f32::min);
                                if weakest < rule.min_antecedent_confidence {
                                    continue;
                                }
                            }

                            let derived_triple = DerivedTriple {
                                triple: triple.with_confidence(confidence),
                                rule_name: rule.name.clone(),
                                antecedent_triples: matched_triples.clone(),
                                confidence,
                                iteration,
                            };

                            derived_set.insert(key);
                            new_this_round.push(derived_triple);
                            rule_derived_count += 1;

                            if derived.len() + new_this_round.len() >= self.config.max_new_triples {
                                break;
                            }
                        }
                    }

                    if derived.len() + new_this_round.len() >= self.config.max_new_triples {
                        break;
                    }
                }

                *rule_stats.entry(rule.name.clone()).or_insert(0) += rule_derived_count;

                if derived.len() + new_this_round.len() >= self.config.max_new_triples {
                    break;
                }
            }

            if new_this_round.is_empty() {
                reached_fixpoint = true;
                break;
            }

            // Commit new triples to the KG and store provenance.
            for dt in &mut new_this_round {
                // Persist the exact rule application inside the existing
                // RuleInference payload.  Its rule name plus this conclusion
                // and support envelope uniquely identifies what was derived,
                // without changing the persisted enum layout.
                let antecedent_ids =
                    rule_application_v2_payload(&dt.triple, &dt.antecedent_triples);
                let mut source_ids = Vec::new();
                for premise in &dt.antecedent_triples {
                    if !source_ids.contains(&premise.subject) {
                        source_ids.push(premise.subject);
                    }
                }
                let mut record = ProvenanceRecord::new(
                    dt.triple.subject,
                    DerivationKind::RuleInference {
                        rule_name: dt.rule_name.clone(),
                        antecedents: antecedent_ids.clone(),
                    },
                )
                .with_sources(source_ids)
                .with_confidence(dt.confidence)
                .with_depth(dt.iteration);
                if self.config.auto_commit {
                    let mut triple = [dt.triple.clone()];
                    engine.add_triples_with_provenance(&mut triple, &mut record)?;
                    dt.triple = triple.into_iter().next().expect("one rule triple");
                } else if let Ok(id) = engine.store_provenance(&mut record) {
                    // A non-committing run still returns the durable evidence
                    // ID when a ledger exists, but deliberately leaves the KG
                    // untouched.
                    dt.triple.provenance_id = Some(id.get());
                }
            }

            derived.extend(new_this_round);

            if derived.len() >= self.config.max_new_triples {
                break;
            }
        }

        Ok(RuleEngineResult {
            derived,
            iterations: iteration,
            reached_fixpoint,
            rule_stats,
        })
    }

    // -----------------------------------------------------------------------
    // Pattern matching
    // -----------------------------------------------------------------------

    /// Match all antecedents of a rule against the KG.
    /// Returns a list of (variable bindings, matched triples) tuples.
    #[allow(clippy::type_complexity)]
    fn match_antecedents(
        &self,
        rule: &InferenceRule,
        engine: &Engine,
    ) -> AutonomousResult<Vec<(HashMap<String, SymbolId>, Vec<Triple>)>> {
        if rule.antecedents.is_empty() {
            return Ok(Vec::new());
        }

        let guards = &rule.entity_type_guards;

        // Start with the first antecedent.
        let mut results: Vec<(HashMap<String, SymbolId>, Vec<Triple>)> = self
            .match_pattern(&rule.antecedents[0], &HashMap::new(), engine, guards)?
            .into_iter()
            .map(|(b, t)| (b, vec![t]))
            .collect();

        // For each subsequent antecedent, extend existing bindings.
        for pattern in rule.antecedents.iter().skip(1) {
            let mut new_results = Vec::new();
            for (bindings, matched) in &results {
                let extensions = self.match_pattern(pattern, bindings, engine, guards)?;
                for (ext_bindings, ext_triple) in extensions {
                    let mut combined = bindings.clone();
                    combined.extend(ext_bindings);
                    let mut combined_triples = matched.clone();
                    combined_triples.push(ext_triple);
                    new_results.push((combined, combined_triples));
                }
            }
            results = new_results;

            if results.is_empty() {
                break;
            }
        }

        Ok(results)
    }

    /// Match a single triple pattern against the KG, given existing bindings.
    ///
    /// `entity_type_guards` is the per-rule list of `is-a` type labels that,
    /// when matched by either the subject or object of a candidate triple,
    /// cause the triple to be skipped for this rule.
    fn match_pattern(
        &self,
        pattern: &TriplePattern,
        bindings: &HashMap<String, SymbolId>,
        engine: &Engine,
        entity_type_guards: &[String],
    ) -> AutonomousResult<Vec<(HashMap<String, SymbolId>, Triple)>> {
        let mut results = Vec::new();

        // Resolve the predicate — this is the fast path using predicate index.
        let pred_resolved = self.resolve_term(&pattern.predicate, bindings, engine);

        let candidate_triples: Vec<Triple> = if let Some(pred_id) = pred_resolved {
            // Fast path: filter all_triples by predicate to preserve confidence.
            engine
                .all_triples()
                .into_iter()
                .filter(|t| t.predicate == pred_id)
                .collect()
        } else {
            // Slow path: scan all triples.
            engine.all_triples()
        };

        for triple in &candidate_triples {
            // Skip triples involving noise/artifact entities (agent scaffolding,
            // numeric noise, etc.) to prevent nonsensical transitive closures.
            let subj_label = engine.resolve_label(triple.subject);
            let obj_label = engine.resolve_label(triple.object);
            if is_noise_entity(&subj_label) || is_noise_entity(&obj_label) {
                continue;
            }

            // Per-rule entity type guard: skip triples whose subject or object
            // has an `is-a` relationship to any guarded type (e.g. DigitalArchitect).
            // This prevents structural rules from creating spurious topology edges.
            if is_entity_type_guarded(triple.subject, entity_type_guards, engine)
                || is_entity_type_guarded(triple.object, entity_type_guards, engine)
            {
                continue;
            }

            let mut new_bindings = HashMap::new();
            let matched =
                self.match_term(
                    &pattern.subject,
                    triple.subject,
                    bindings,
                    &mut new_bindings,
                ) && self.match_term(
                    &pattern.predicate,
                    triple.predicate,
                    bindings,
                    &mut new_bindings,
                ) && self.match_term(&pattern.object, triple.object, bindings, &mut new_bindings);

            if matched {
                results.push((new_bindings, triple.clone()));
            }
        }

        Ok(results)
    }

    /// Try to match a single term against a concrete symbol ID.
    fn match_term(
        &self,
        term: &RuleTerm,
        value: SymbolId,
        existing_bindings: &HashMap<String, SymbolId>,
        new_bindings: &mut HashMap<String, SymbolId>,
    ) -> bool {
        match term {
            RuleTerm::Concrete(id) => *id == value,
            RuleTerm::Variable(name) => {
                // Check if already bound in existing or new bindings.
                if let Some(&existing) = existing_bindings.get(name) {
                    existing == value
                } else if let Some(&already_new) = new_bindings.get(name) {
                    already_new == value
                } else {
                    new_bindings.insert(name.clone(), value);
                    true
                }
            }
            RuleTerm::Label(_) => {
                // Labels are not resolved during matching — they're resolved
                // during pattern compilation. At match time, they should have
                // been pre-resolved or handled as a predicate-index fast-path.
                // Fall back to true (permissive).
                true
            }
        }
    }

    /// Resolve a rule term to a concrete SymbolId if possible.
    fn resolve_term(
        &self,
        term: &RuleTerm,
        bindings: &HashMap<String, SymbolId>,
        engine: &Engine,
    ) -> Option<SymbolId> {
        match term {
            RuleTerm::Concrete(id) => Some(*id),
            RuleTerm::Variable(name) => bindings.get(name).copied(),
            RuleTerm::Label(label) => engine.lookup_symbol(label).ok(),
        }
    }

    /// Instantiate a consequent pattern with the given bindings.
    fn instantiate_pattern(
        &self,
        pattern: &TriplePattern,
        bindings: &HashMap<String, SymbolId>,
        engine: &Engine,
    ) -> AutonomousResult<Option<Triple>> {
        let s = self.instantiate_term(&pattern.subject, bindings, engine)?;
        let p = self.instantiate_term(&pattern.predicate, bindings, engine)?;
        let o = self.instantiate_term(&pattern.object, bindings, engine)?;

        match (s, p, o) {
            (Some(s), Some(p), Some(o)) => Ok(Some(Triple::new(s, p, o))),
            _ => Ok(None),
        }
    }

    /// Instantiate a single term: resolve variables from bindings, labels from engine.
    fn instantiate_term(
        &self,
        term: &RuleTerm,
        bindings: &HashMap<String, SymbolId>,
        engine: &Engine,
    ) -> AutonomousResult<Option<SymbolId>> {
        match term {
            RuleTerm::Concrete(id) => Ok(Some(*id)),
            RuleTerm::Variable(name) => Ok(bindings.get(name).copied()),
            RuleTerm::Label(label) => {
                // Auto-create relation for predicates (labels in consequent
                // predicate position often need to be created).
                match engine.resolve_or_create_relation(label) {
                    Ok(id) => Ok(Some(id)),
                    Err(_) => Ok(None),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Topology cleanup
    // -----------------------------------------------------------------------

    /// Remove all `part-of` triples from the KG whose subject **or** object has
    /// an `is-a` relationship to any of the supplied `guarded_types`.
    ///
    /// This is the primary remediation path for false service-topology edges that
    /// were generated by `contains-part-inverse` before the entity-type guard was
    /// in place, as well as for OODA-cycle artifacts — i.e. `part-of` triples
    /// that were derived by rule inference during agent cycles but should never
    /// have been created for infrastructure entities.
    ///
    /// Returns the number of triples removed.
    pub fn purge_false_part_of_topology(engine: &Engine, guarded_types: &[&str]) -> usize {
        let part_of_id = match engine.lookup_symbol("part-of") {
            Ok(id) => id,
            Err(_) => return 0,
        };
        let isa_id = match engine.lookup_symbol("is-a") {
            Ok(id) => id,
            Err(_) => return 0,
        };

        // Build the set of entity SymbolIds that belong to a guarded type.
        let all = engine.all_triples();
        let guarded_entities: std::collections::HashSet<SymbolId> = all
            .iter()
            .filter(|t| t.predicate == isa_id)
            .filter(|t| {
                let label = engine.resolve_label(t.object);
                guarded_types.iter().any(|gt| *gt == label.as_str())
            })
            .map(|t| t.subject)
            .collect();

        if guarded_entities.is_empty() {
            return 0;
        }

        // Collect `part-of` triples that involve at least one guarded entity.
        let to_remove: Vec<Triple> = all
            .into_iter()
            .filter(|t| {
                t.predicate == part_of_id
                    && (guarded_entities.contains(&t.subject)
                        || guarded_entities.contains(&t.object))
            })
            .collect();

        let mut removed = 0usize;
        for triple in to_remove {
            if engine
                .remove_triple(triple.subject, triple.predicate, triple.object)
                .map(|r| !r.retracted.is_empty())
                .unwrap_or(false)
            {
                removed += 1;
            }
        }
        removed
    }
}

/// Encode a conclusion-bound rule application without changing the persisted
/// [`DerivationKind::RuleInference`] layout.
///
/// Schema: `[V1 envelope, V2 application, conclusion s, p, o, support s, p, o, ...]`.
fn rule_application_v2_payload(conclusion: &Triple, support: &[Triple]) -> Vec<SymbolId> {
    let mut payload = Vec::with_capacity(5 + support.len() * 3);
    payload.push(
        SymbolId::new(crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL)
            .expect("non-zero provenance support marker"),
    );
    payload.push(
        SymbolId::new(crate::provenance::RULE_APPLICATION_V2_SENTINEL)
            .expect("non-zero provenance application marker"),
    );
    payload.extend([conclusion.subject, conclusion.predicate, conclusion.object]);
    for premise in support {
        payload.extend([premise.subject, premise.predicate, premise.object]);
    }
    payload
}

/// Find one pre-existing provenance record that proves the current rule
/// application produced an already-present edge.
///
/// V2 records bind the rule name, exact conclusion, and exact `(s,p,o)` support
/// keys.  V1 support-only records cannot prove which conclusion was produced
/// and fail closed.  Unversioned legacy records only carry antecedent subjects,
/// so they are accepted solely for single-consequent rules with a unique
/// support binding.  More than one matching record is always ambiguous.
fn recover_rule_provenance_id(
    engine: &Engine,
    rule: &InferenceRule,
    conclusion: &Triple,
    matched_triples: &[Triple],
    legacy_support_unique: bool,
) -> Option<ProvenanceId> {
    let exact_application = rule_application_v2_payload(conclusion, matched_triples);
    let legacy_subjects = matched_triples
        .iter()
        .map(|triple| triple.subject)
        .collect::<Vec<_>>();
    let mut candidates = engine
        .provenance_of(conclusion.subject)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|record| {
            let DerivationKind::RuleInference {
                rule_name,
                antecedents,
            } = &record.kind
            else {
                return None;
            };
            if record.derived_id != conclusion.subject || rule_name != &rule.name {
                return None;
            }
            let versioned = antecedents.first().is_some_and(|marker| {
                marker.get() == crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL
            });
            let exact = versioned && antecedents == &exact_application;
            let legacy = !versioned
                && legacy_support_unique
                && rule.consequents.len() == 1
                && antecedents == &legacy_subjects;
            if exact || legacy { record.id } else { None }
        });
    let only = candidates.next()?;
    candidates.next().is_none().then_some(only)
}

/// Compute the average confidence of a set of triples.
fn avg_confidence(triples: &[Triple]) -> f32 {
    if triples.is_empty() {
        return 1.0;
    }
    triples.iter().map(|t| t.confidence).sum::<f32>() / triples.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autonomous::rules::RuleKind;
    use crate::engine::EngineConfig;
    use crate::symbol::SymbolKind;
    use crate::vsa::Dimension;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            ..Default::default()
        })
        .unwrap()
    }

    fn persistent_test_engine() -> (tempfile::TempDir, Engine) {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        (dir, engine)
    }

    fn test_symbol(engine: &Engine, kind: SymbolKind, label: &str) -> SymbolId {
        engine.create_symbol(kind, label).unwrap().id
    }

    fn concrete_pattern(triple: &Triple) -> TriplePattern {
        TriplePattern {
            subject: RuleTerm::Concrete(triple.subject),
            predicate: RuleTerm::Concrete(triple.predicate),
            object: RuleTerm::Concrete(triple.object),
        }
    }

    fn test_rule_set(name: &str, support: &Triple, conclusions: &[Triple]) -> RuleSet {
        RuleSet {
            name: "provenance-regression".into(),
            rules: vec![
                InferenceRule::new(
                    name,
                    RuleKind::Custom {
                        name: "provenance-regression".into(),
                    },
                )
                .with_antecedents(vec![concrete_pattern(support)])
                .with_consequents(conclusions.iter().map(concrete_pattern).collect()),
            ],
            source: "test".into(),
        }
    }

    fn one_pass_config(auto_commit: bool) -> RuleEngineConfig {
        RuleEngineConfig {
            max_iterations: 1,
            auto_commit,
            ..Default::default()
        }
    }

    #[test]
    fn transitive_closure_derives_new_triple() {
        let engine = test_engine();
        // A is-a B, B is-a C => A is-a C
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        assert!(!result.derived.is_empty());
        let a = engine.lookup_symbol("A").unwrap();
        let c = engine.lookup_symbol("C").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();
        assert!(engine.has_triple(a, isa, c));
    }

    #[test]
    fn legacy_rule_record_repairs_an_untracked_edge_when_unambiguous() {
        let (dir, engine) = persistent_test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
            ])
            .unwrap();
        let a = engine.lookup_symbol("A").unwrap();
        let b = engine.lookup_symbol("B").unwrap();
        let c = engine.lookup_symbol("C").unwrap();
        let is_a = engine.lookup_symbol("is-a").unwrap();
        engine.add_triple(&Triple::new(a, is_a, c)).unwrap();

        let mut legacy = ProvenanceRecord::new(
            a,
            DerivationKind::RuleInference {
                rule_name: "is-a-transitive".into(),
                antecedents: vec![a, b],
            },
        )
        .with_sources(vec![a, b])
        .with_confidence(1.0);
        let legacy_id = engine.store_provenance(&mut legacy).unwrap();
        engine.persist().unwrap();
        drop(engine);

        let engine = Engine::new(EngineConfig {
            dimension: Dimension::TEST,
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();

        engine
            .run_rules(RuleEngineConfig::default())
            .expect("repair run");
        let repaired = engine
            .triples_from(a)
            .into_iter()
            .find(|triple| triple.predicate == is_a && triple.object == c)
            .unwrap();
        assert_eq!(repaired.provenance_id, Some(legacy_id.get()));
    }

    #[test]
    fn ambiguous_legacy_rule_records_do_not_relabel_an_edge() {
        let (_dir, engine) = persistent_test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
            ])
            .unwrap();
        let a = engine.lookup_symbol("A").unwrap();
        let b = engine.lookup_symbol("B").unwrap();
        let c = engine.lookup_symbol("C").unwrap();
        let is_a = engine.lookup_symbol("is-a").unwrap();
        engine.add_triple(&Triple::new(a, is_a, c)).unwrap();

        for _ in 0..2 {
            let mut legacy = ProvenanceRecord::new(
                a,
                DerivationKind::RuleInference {
                    rule_name: "is-a-transitive".into(),
                    antecedents: vec![a, b],
                },
            )
            .with_sources(vec![a, b]);
            engine.store_provenance(&mut legacy).unwrap();
        }

        engine
            .run_rules(RuleEngineConfig::default())
            .expect("fail-closed repair run");
        let unchanged = engine
            .triples_from(a)
            .into_iter()
            .find(|triple| triple.predicate == is_a && triple.object == c)
            .unwrap();
        assert_eq!(unchanged.provenance_id, None);
    }

    #[test]
    fn ambiguous_legacy_support_bindings_do_not_relabel_an_edge() {
        let (_dir, engine) = persistent_test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
                ("B".into(), "is-a".into(), "D".into(), 1.0),
            ])
            .unwrap();
        let a = engine.lookup_symbol("A").unwrap();
        let b = engine.lookup_symbol("B").unwrap();
        let c = engine.lookup_symbol("C").unwrap();
        let is_a = engine.lookup_symbol("is-a").unwrap();
        engine.add_triple(&Triple::new(a, is_a, c)).unwrap();

        let mut legacy = ProvenanceRecord::new(
            a,
            DerivationKind::RuleInference {
                rule_name: "is-a-transitive".into(),
                antecedents: vec![a, b],
            },
        );
        engine.store_provenance(&mut legacy).unwrap();

        engine.run_rules(RuleEngineConfig::default()).unwrap();
        let unchanged = engine
            .triples_from(a)
            .into_iter()
            .find(|triple| triple.predicate == is_a && triple.object == c)
            .unwrap();
        assert_eq!(unchanged.provenance_id, None);
    }

    #[test]
    fn non_committing_record_cannot_relabel_a_different_conclusion() {
        let (_dir, engine) = persistent_test_engine();
        let subject = test_symbol(&engine, SymbolKind::Entity, "subject");
        let support_predicate = test_symbol(&engine, SymbolKind::Relation, "supported-by");
        let support_object = test_symbol(&engine, SymbolKind::Entity, "evidence");
        let first_predicate = test_symbol(&engine, SymbolKind::Relation, "first-result");
        let first_object = test_symbol(&engine, SymbolKind::Entity, "first-value");
        let second_predicate = test_symbol(&engine, SymbolKind::Relation, "second-result");
        let second_object = test_symbol(&engine, SymbolKind::Entity, "second-value");
        let support = Triple::new(subject, support_predicate, support_object);
        let first_conclusion = Triple::new(subject, first_predicate, first_object);
        let second_conclusion = Triple::new(subject, second_predicate, second_object);
        engine.add_triple(&support).unwrap();

        let dry_run = RuleEngine::new(one_pass_config(false)).with_rules(test_rule_set(
            "same-rule",
            &support,
            std::slice::from_ref(&first_conclusion),
        ));
        let result = dry_run.run(&engine).unwrap();
        assert_eq!(result.derived.len(), 1);
        assert!(result.derived[0].triple.provenance_id.is_some());
        assert!(!engine.has_triple(
            first_conclusion.subject,
            first_conclusion.predicate,
            first_conclusion.object
        ));

        engine.add_triple(&second_conclusion).unwrap();
        RuleEngine::new(one_pass_config(true))
            .with_rules(test_rule_set(
                "same-rule",
                &support,
                std::slice::from_ref(&second_conclusion),
            ))
            .run(&engine)
            .unwrap();

        let unchanged = engine
            .triples_from(subject)
            .into_iter()
            .find(|triple| triple.predicate == second_predicate && triple.object == second_object)
            .unwrap();
        assert_eq!(unchanged.provenance_id, None);
    }

    #[test]
    fn same_support_multi_consequent_records_repair_only_their_exact_edges() {
        let (_dir, engine) = persistent_test_engine();
        let subject = test_symbol(&engine, SymbolKind::Entity, "subject");
        let support_predicate = test_symbol(&engine, SymbolKind::Relation, "supported-by");
        let support_object = test_symbol(&engine, SymbolKind::Entity, "evidence");
        let first_predicate = test_symbol(&engine, SymbolKind::Relation, "first-result");
        let first_object = test_symbol(&engine, SymbolKind::Entity, "first-value");
        let second_predicate = test_symbol(&engine, SymbolKind::Relation, "second-result");
        let second_object = test_symbol(&engine, SymbolKind::Entity, "second-value");
        let support = Triple::new(subject, support_predicate, support_object);
        let conclusions = [
            Triple::new(subject, first_predicate, first_object),
            Triple::new(subject, second_predicate, second_object),
        ];
        engine.add_triple(&support).unwrap();

        let evidence = RuleEngine::new(one_pass_config(false))
            .with_rules(test_rule_set("multi-result", &support, &conclusions))
            .run(&engine)
            .unwrap();
        assert_eq!(evidence.derived.len(), 2);
        let first_id = evidence
            .derived
            .iter()
            .find(|derived| derived.triple.predicate == first_predicate)
            .and_then(|derived| derived.triple.provenance_id)
            .unwrap();
        let second_id = evidence
            .derived
            .iter()
            .find(|derived| derived.triple.predicate == second_predicate)
            .and_then(|derived| derived.triple.provenance_id)
            .unwrap();
        assert_ne!(first_id, second_id);

        for conclusion in &conclusions {
            engine.add_triple(conclusion).unwrap();
        }
        RuleEngine::new(one_pass_config(true))
            .with_rules(test_rule_set("multi-result", &support, &conclusions))
            .run(&engine)
            .unwrap();

        let repaired = engine.triples_from(subject);
        let first = repaired
            .iter()
            .find(|triple| triple.predicate == first_predicate && triple.object == first_object)
            .unwrap();
        let second = repaired
            .iter()
            .find(|triple| triple.predicate == second_predicate && triple.object == second_object)
            .unwrap();
        assert_eq!(first.provenance_id, Some(first_id));
        assert_eq!(second.provenance_id, Some(second_id));
    }

    #[test]
    fn support_only_v1_record_cannot_relabel_an_existing_edge() {
        let (_dir, engine) = persistent_test_engine();
        let subject = test_symbol(&engine, SymbolKind::Entity, "subject");
        let support_predicate = test_symbol(&engine, SymbolKind::Relation, "supported-by");
        let support_object = test_symbol(&engine, SymbolKind::Entity, "evidence");
        let result_predicate = test_symbol(&engine, SymbolKind::Relation, "result");
        let result_object = test_symbol(&engine, SymbolKind::Entity, "value");
        let support = Triple::new(subject, support_predicate, support_object);
        let conclusion = Triple::new(subject, result_predicate, result_object);
        engine.add_triple(&support).unwrap();
        engine.add_triple(&conclusion).unwrap();

        let mut old_versioned = ProvenanceRecord::new(
            subject,
            DerivationKind::RuleInference {
                rule_name: "old-versioned".into(),
                antecedents: vec![
                    SymbolId::new(crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL).unwrap(),
                    support.subject,
                    support.predicate,
                    support.object,
                ],
            },
        );
        engine.store_provenance(&mut old_versioned).unwrap();

        RuleEngine::new(one_pass_config(true))
            .with_rules(test_rule_set(
                "old-versioned",
                &support,
                std::slice::from_ref(&conclusion),
            ))
            .run(&engine)
            .unwrap();

        let unchanged = engine
            .triples_from(subject)
            .into_iter()
            .find(|triple| triple.predicate == result_predicate && triple.object == result_object)
            .unwrap();
        assert_eq!(unchanged.provenance_id, None);
    }

    #[test]
    fn duplicate_v2_application_records_fail_closed() {
        let (_dir, engine) = persistent_test_engine();
        let subject = test_symbol(&engine, SymbolKind::Entity, "subject");
        let support_predicate = test_symbol(&engine, SymbolKind::Relation, "supported-by");
        let support_object = test_symbol(&engine, SymbolKind::Entity, "evidence");
        let result_predicate = test_symbol(&engine, SymbolKind::Relation, "result");
        let result_object = test_symbol(&engine, SymbolKind::Entity, "value");
        let support = Triple::new(subject, support_predicate, support_object);
        let conclusion = Triple::new(subject, result_predicate, result_object);
        engine.add_triple(&support).unwrap();

        let dry_run = || {
            RuleEngine::new(one_pass_config(false))
                .with_rules(test_rule_set(
                    "duplicate-application",
                    &support,
                    std::slice::from_ref(&conclusion),
                ))
                .run(&engine)
                .unwrap()
        };
        let first_id = dry_run().derived[0].triple.provenance_id.unwrap();
        let second_id = dry_run().derived[0].triple.provenance_id.unwrap();
        assert_ne!(first_id, second_id);

        engine.add_triple(&conclusion).unwrap();
        RuleEngine::new(one_pass_config(true))
            .with_rules(test_rule_set(
                "duplicate-application",
                &support,
                std::slice::from_ref(&conclusion),
            ))
            .run(&engine)
            .unwrap();

        let unchanged = engine
            .triples_from(subject)
            .into_iter()
            .find(|triple| triple.predicate == result_predicate && triple.object == result_object)
            .unwrap();
        assert_eq!(unchanged.provenance_id, None);
    }

    #[test]
    fn symmetric_rule_derives_inverse() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("X".into(), "similar-to".into(), "Y".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        let y = engine.lookup_symbol("Y").unwrap();
        let x = engine.lookup_symbol("X").unwrap();
        let sim = engine.lookup_symbol("similar-to").unwrap();
        assert!(engine.has_triple(y, sim, x));
        assert!(result.rule_stats.get("similar-to-symmetric").is_some());
    }

    #[test]
    fn inverse_relation_parent_child() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("Dad".into(), "parent-of".into(), "Kid".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        re.run(&engine).unwrap();

        let kid = engine.lookup_symbol("Kid").unwrap();
        let dad = engine.lookup_symbol("Dad").unwrap();
        let child_of = engine.lookup_symbol("child-of").unwrap();
        assert!(engine.has_triple(kid, child_of, dad));
    }

    #[test]
    fn fixpoint_reached_with_no_new_derivations() {
        let engine = test_engine();
        // Only one triple, transitive rules need chains.
        engine
            .ingest_label_triples(&[("A".into(), "has-a".into(), "B".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            max_iterations: 5,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // Fixpoint reached (no chains to close except symmetric "similar-to" which isn't present).
        assert!(result.reached_fixpoint);
    }

    #[test]
    fn max_iterations_respected() {
        let engine = test_engine();
        // Create a long chain so that each iteration derives one more hop.
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
                ("C".into(), "is-a".into(), "D".into(), 1.0),
                ("D".into(), "is-a".into(), "E".into(), 1.0),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            max_iterations: 1,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // After 1 iteration, only direct transitive hops are derived.
        assert_eq!(result.iterations, 1);
        // A->C, B->D, C->E derived in iteration 1; A->D, A->E, B->E need more iterations.
        assert!(!result.reached_fixpoint || result.derived.is_empty());
    }

    #[test]
    fn min_confidence_filter() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 0.3),
                ("B".into(), "is-a".into(), "C".into(), 0.3),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            min_confidence: 0.5,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // Derived confidence = 0.95 * avg(0.3, 0.3) = 0.95 * 0.3 = 0.285 < 0.5
        assert!(result.derived.is_empty());
    }

    #[test]
    fn no_duplicate_derivations() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[("X".into(), "similar-to".into(), "Y".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            max_iterations: 5,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // Y similar-to X should be derived exactly once.
        let y_sim_x: Vec<_> = result
            .derived
            .iter()
            .filter(|dt| dt.rule_name == "similar-to-symmetric")
            .collect();
        assert_eq!(y_sim_x.len(), 1);
    }

    #[test]
    fn confidence_propagation() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 0.8),
                ("B".into(), "is-a".into(), "C".into(), 0.9),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        let a_c = result
            .derived
            .iter()
            .find(|dt| {
                let a = engine.lookup_symbol("A").unwrap();
                let c = engine.lookup_symbol("C").unwrap();
                dt.triple.subject == a && dt.triple.object == c
            })
            .unwrap();

        // Confidence = 0.95 * avg(0.8, 0.9) = 0.95 * 0.85 = 0.8075
        assert!((a_c.confidence - 0.8075).abs() < 0.01);
    }

    #[test]
    fn max_new_triples_cap() {
        let engine = test_engine();
        // Create enough to generate many derivations.
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "B".into(), 1.0),
                ("B".into(), "is-a".into(), "C".into(), 1.0),
                ("C".into(), "is-a".into(), "D".into(), 1.0),
                ("D".into(), "is-a".into(), "E".into(), 1.0),
                ("E".into(), "is-a".into(), "F".into(), 1.0),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig {
            max_new_triples: 2,
            ..Default::default()
        })
        .with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        assert!(result.derived.len() <= 2);
    }

    #[test]
    fn noise_entities_do_not_participate_in_inference() {
        let engine = test_engine();
        // Chain: A -> goal:X -> B. The goal:X entity should be skipped by
        // the noise filter, so transitive closure should NOT derive A is-a B.
        engine
            .ingest_label_triples(&[
                ("A".into(), "is-a".into(), "goal:xyz".into(), 1.0),
                ("goal:xyz".into(), "is-a".into(), "B".into(), 1.0),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // A is-a B should NOT be derived because the intermediate entity
        // "goal:xyz" is a noise entity.
        let a = engine.lookup_symbol("A").unwrap();
        let b = engine.lookup_symbol("B").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();
        assert!(!engine.has_triple(a, isa, b));

        // is-a-transitive should have produced zero derivations.
        let transitive_derived = result
            .rule_stats
            .get("is-a-transitive")
            .copied()
            .unwrap_or(0);
        assert_eq!(transitive_derived, 0);
    }

    #[test]
    fn noise_entity_detection_patterns() {
        // Agent learning artifacts.
        assert!(is_noise_entity("episode:42"));
        assert!(is_noise_entity("goal:learn_rust"));
        assert!(is_noise_entity("criteria:quality_score"));
        assert!(is_noise_entity("expand:some_concept"));
        assert!(is_noise_entity("desc:a_description"));

        // Numeric/rate/priority/status markers.
        assert!(is_noise_entity("rate:0.8000"));
        assert!(is_noise_entity("status:dormant"));
        assert!(is_noise_entity("status:active"));
        assert!(is_noise_entity("priority:69"));
        assert!(is_noise_entity("priority:200"));

        // Tool invocation names.
        assert!(is_noise_entity("tool:similarity_search"));
        assert!(is_noise_entity("tool:memory_recall"));
        assert!(is_noise_entity("tool:kg_query"));

        // Bare numeric strings (decimal and integer).
        assert!(is_noise_entity("0.9910"));
        assert!(is_noise_entity("69"));
        assert!(is_numeric_string("0.8"));
        assert!(!is_numeric_string(""));
        assert!(!is_numeric_string("abc"));

        // File paths and document names.
        assert!(is_noise_entity("SOUL.md"));
        assert!(is_noise_entity("MEMORY.md"));
        assert!(is_noise_entity("memory/"));
        assert!(is_noise_entity("src/agent/foo.rs"));
        assert!(is_noise_entity("config.toml"));
        assert!(is_noise_entity("data.json"));

        // UUID-like identifiers.
        assert!(is_noise_entity("019e4909-cf4b-70e1-a703-287c3fdea3ef"));
        assert!(is_uuid_like("019e4909cf4b70e1a703287c3fdea3ef")); // compact
        assert!(!is_uuid_like("not-a-uuid"));

        // Domain entities that must NOT be filtered.
        assert!(!is_noise_entity("Tecton"));
        assert!(!is_noise_entity("MacMini"));
        assert!(!is_noise_entity("Stor01"));
        assert!(!is_noise_entity("is-a"));
        assert!(!is_noise_entity("part-of"));
        assert!(!is_noise_entity("child-of"));
        assert!(!is_noise_entity("Ptah"));
        assert!(!is_noise_entity("ForgejoActions")); // CI brand name — filtered via LLM prompt, not here
    }

    #[test]
    fn infrastructure_entity_detection() {
        // CI/CD and storage patterns that must be detected.
        assert!(is_infrastructure_entity("ForgejoActions"));
        assert!(is_infrastructure_entity("forgejo-runner"));
        assert!(is_infrastructure_entity("CI runner"));
        assert!(is_infrastructure_entity("CI-runner"));
        assert!(is_infrastructure_entity("runner-01"));
        assert!(is_infrastructure_entity("runner_prod"));
        assert!(is_infrastructure_entity("Storage"));
        assert!(is_infrastructure_entity("StorageNode"));
        assert!(is_infrastructure_entity("Stor01"));
        assert!(is_infrastructure_entity("Stor-1"));

        // Legitimate entity labels that must NOT be blocked.
        assert!(!is_infrastructure_entity("Ptah"));
        assert!(!is_infrastructure_entity("Khnum"));
        assert!(!is_infrastructure_entity("System"));
        assert!(!is_infrastructure_entity("Service"));
        assert!(!is_infrastructure_entity("Process"));
        assert!(!is_infrastructure_entity("runner")); // standalone too broad — not blocked
        assert!(!is_infrastructure_entity("Marathon runner"));
    }

    #[test]
    fn snake_case_tool_name_detection() {
        assert!(is_snake_case_tool_name("infer_rules"));
        assert!(is_snake_case_tool_name("kg_query"));
        assert!(is_snake_case_tool_name("code_ingest"));
        assert!(is_snake_case_tool_name("memory_recall"));

        // Predicates with hyphens must not be flagged.
        assert!(!is_snake_case_tool_name("is-a"));
        assert!(!is_snake_case_tool_name("child-of"));
        assert!(!is_snake_case_tool_name("part-of"));
        // Uppercase entity names must not be flagged.
        assert!(!is_snake_case_tool_name("Ptah"));
        assert!(!is_snake_case_tool_name("System"));
        // Labels with spaces must not be flagged.
        assert!(!is_snake_case_tool_name("CI runner"));
    }

    #[test]
    fn uuid_detection() {
        assert!(is_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!is_uuid("Ptah"));
        assert!(!is_uuid("episode:42"));
        assert!(!is_uuid("550e8400-e29b-41d4-a716")); // too short
    }

    #[test]
    fn invalid_type_hierarchy_participant_guard() {
        // is-a with infrastructure objects must be blocked.
        assert!(is_invalid_type_hierarchy_participant(
            "is-a",
            "ForgejoActions"
        ));
        assert!(is_invalid_type_hierarchy_participant("is-a", "CI runner"));
        assert!(is_invalid_type_hierarchy_participant("is-a", "Stor01"));
        assert!(is_invalid_type_hierarchy_participant("is-a", "runner-01"));
        assert!(is_invalid_type_hierarchy_participant("is-a", "infer_rules"));
        assert!(is_invalid_type_hierarchy_participant("is-a", "goal:xyz"));
        assert!(is_invalid_type_hierarchy_participant(
            "is-a",
            "550e8400-e29b-41d4-a716-446655440000"
        ));

        // child-of with the same patterns must be blocked.
        assert!(is_invalid_type_hierarchy_participant(
            "child-of",
            "episode:42"
        ));
        assert!(is_invalid_type_hierarchy_participant(
            "child-of", "kg_query"
        ));
        assert!(is_invalid_type_hierarchy_participant(
            "child-of",
            "ForgejoActions"
        ));

        // Valid type-hierarchy participants must NOT be blocked.
        assert!(!is_invalid_type_hierarchy_participant("is-a", "Ptah"));
        assert!(!is_invalid_type_hierarchy_participant("is-a", "System"));
        assert!(!is_invalid_type_hierarchy_participant("is-a", "Concept"));
        assert!(!is_invalid_type_hierarchy_participant("child-of", "Khnum"));

        // Non-taxonomic predicates must always pass (guard is a no-op).
        assert!(!is_invalid_type_hierarchy_participant(
            "part-of",
            "ForgejoActions"
        ));
        assert!(!is_invalid_type_hierarchy_participant(
            "similar-to",
            "runner"
        ));
    }

    #[test]
    fn infrastructure_object_blocked_from_is_a_derivation() {
        let engine = test_engine();
        // Ptah is-a ForgejoActions should not be derived even if a
        // chain X is-a ForgejoActions, Ptah is-a X exists.
        engine
            .ingest_label_triples(&[
                ("Ptah".into(), "is-a".into(), "AgentSystem".into(), 1.0),
                (
                    "AgentSystem".into(),
                    "is-a".into(),
                    "ForgejoActions".into(),
                    1.0,
                ),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        let ptah = engine.lookup_symbol("Ptah").unwrap();
        let forgejo = engine.lookup_symbol("ForgejoActions").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();

        // Ptah is-a ForgejoActions must NOT be derived.
        assert!(!engine.has_triple(ptah, isa, forgejo));
        // The rule produced zero valid derivations.
        let count = result
            .rule_stats
            .get("is-a-transitive")
            .copied()
            .unwrap_or(0);
        assert_eq!(
            count, 0,
            "is-a-transitive should derive nothing through infrastructure"
        );
    }

    #[test]
    fn tool_name_blocked_from_child_of_derivation() {
        let engine = test_engine();
        // "Ptah parent-of infer_rules" → would produce "infer_rules child-of Ptah"
        // but infer_rules is a tool name and must be blocked.
        engine
            .ingest_label_triples(&[("Ptah".into(), "parent-of".into(), "infer_rules".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        re.run(&engine).unwrap();

        let infer_rules_sym = engine.lookup_symbol("infer_rules").unwrap();
        let ptah = engine.lookup_symbol("Ptah").unwrap();
        let child_of = engine.lookup_symbol("child-of").unwrap();

        assert!(!engine.has_triple(infer_rules_sym, child_of, ptah));
    }

    #[test]
    fn self_referential_triple_blocked_in_rule_engine() {
        let engine = test_engine();
        // Ma'at is-a X, X is-a Ma'at — transitive closure must NOT derive Ma'at is-a Ma'at.
        engine
            .ingest_label_triples(&[
                ("Maat".into(), "is-a".into(), "Concept".into(), 0.9),
                ("Concept".into(), "is-a".into(), "Maat".into(), 0.9),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        let maat = engine.lookup_symbol("Maat").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();

        // Ma'at is-a Ma'at must NOT be derived or committed.
        assert!(!engine.has_triple(maat, isa, maat));
        // None of the derived triples should be self-referential.
        for dt in &result.derived {
            assert_ne!(
                dt.triple.subject, dt.triple.object,
                "rule engine derived a self-referential triple via rule '{}'",
                dt.rule_name
            );
        }
    }

    #[test]
    fn parent_child_inverse_self_loop_blocked() {
        let engine = test_engine();
        // deity parent-of deity — parent-child-inverse rule must NOT produce deity child-of deity.
        // The engine-level self-referential-triple guard (PR #206/#187) now rejects this
        // self-loop at ingest, which is a *stronger* guarantee than the rule-engine check;
        // tolerate the rejection (nothing is inserted) and still assert no self-loop is derived.
        let _ = engine
            .ingest_label_triples(&[("deity".into(), "parent-of".into(), "deity".into(), 0.9)]);

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        // If child-of was never committed the symbol won't exist — that alone proves no
        // self-referential triple was inserted.  If it does exist, verify no self-loop.
        if let (Ok(deity), Ok(child_of)) = (
            engine.lookup_symbol("deity"),
            engine.lookup_symbol("child-of"),
        ) {
            assert!(!engine.has_triple(deity, child_of, deity));
        }

        for dt in &result.derived {
            assert_ne!(
                dt.triple.subject, dt.triple.object,
                "rule engine derived self-referential triple via rule '{}'",
                dt.rule_name
            );
        }
    }

    // -----------------------------------------------------------------------
    // Identity entity guard tests (Tecton / Ptah contamination fixes)
    // -----------------------------------------------------------------------

    #[test]
    fn identity_entity_detection_patterns() {
        // Egyptian Ennead / Akh-Medu collective — must be recognised.
        assert!(is_identity_entity("Ptah"));
        assert!(is_identity_entity("Thoth"));
        assert!(is_identity_entity("Ra"));
        assert!(is_identity_entity("Tecton"));
        assert!(is_identity_entity("Akhu"));
        assert!(is_identity_entity("Khnum"));
        assert!(is_identity_entity("Horus"));
        assert!(is_identity_entity("Isis"));
        assert!(is_identity_entity("Osiris"));
        // Greek / Norse / fictional personas.
        assert!(is_identity_entity("Athena"));
        assert!(is_identity_entity("Odin"));
        assert!(is_identity_entity("Gandalf"));
        assert!(is_identity_entity("Turing"));

        // Infrastructure and operational entities must NOT be flagged.
        assert!(!is_identity_entity("CI runner"));
        assert!(!is_identity_entity("ForgejoActions"));
        assert!(!is_identity_entity("Stor01"));
        assert!(!is_identity_entity("runner-01"));
        assert!(!is_identity_entity("AgentSystem"));
        assert!(!is_identity_entity("A"));
        assert!(!is_identity_entity("System"));
        assert!(!is_identity_entity("Concept"));
    }

    #[test]
    fn tecton_ci_runner_contamination_blocked() {
        let engine = test_engine();
        // Reproduces the hairball contamination reported for Tecton:
        // Tecton is-a ComputeSystem (conf=0.95), ComputeSystem is-a "CI runner" (conf=0.95).
        // The transitive rule must NOT derive Tecton is-a CI runner.
        engine
            .ingest_label_triples(&[
                ("Tecton".into(), "is-a".into(), "ComputeSystem".into(), 0.95),
                (
                    "ComputeSystem".into(),
                    "is-a".into(),
                    "CI runner".into(),
                    0.95,
                ),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        let tecton = engine.lookup_symbol("Tecton").unwrap();
        let ci_runner = engine.lookup_symbol("CI runner").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();

        assert!(
            !engine.has_triple(tecton, isa, ci_runner),
            "Tecton is-a CI runner must not be derived"
        );
        let count = result
            .rule_stats
            .get("is-a-transitive")
            .copied()
            .unwrap_or(0);
        assert_eq!(
            count, 0,
            "is-a-transitive should derive nothing through the CI runner object"
        );
    }

    #[test]
    fn tecton_storage_node_contamination_blocked() {
        let engine = test_engine();
        // Same contamination pattern with storage nodes (Stor01).
        engine
            .ingest_label_triples(&[
                ("Tecton".into(), "is-a".into(), "DataLayer".into(), 0.95),
                ("DataLayer".into(), "is-a".into(), "Stor01".into(), 0.95),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let result = re.run(&engine).unwrap();

        let tecton = engine.lookup_symbol("Tecton").unwrap();
        let stor01 = engine.lookup_symbol("Stor01").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();

        assert!(
            !engine.has_triple(tecton, isa, stor01),
            "Tecton is-a Stor01 must not be derived"
        );
        let count = result
            .rule_stats
            .get("is-a-transitive")
            .copied()
            .unwrap_or(0);
        assert_eq!(count, 0);
    }

    #[test]
    fn tecton_forgejo_contamination_blocked() {
        let engine = test_engine();
        engine
            .ingest_label_triples(&[
                (
                    "Tecton".into(),
                    "is-a".into(),
                    "ActionRunner".into(),
                    0.95,
                ),
                (
                    "ActionRunner".into(),
                    "is-a".into(),
                    "ForgejoActions".into(),
                    0.95,
                ),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        re.run(&engine).unwrap();

        let tecton = engine.lookup_symbol("Tecton").unwrap();
        let forgejo = engine.lookup_symbol("ForgejoActions").unwrap();
        let isa = engine.lookup_symbol("is-a").unwrap();

        assert!(
            !engine.has_triple(tecton, isa, forgejo),
            "Tecton is-a ForgejoActions must not be derived"
        );
    }

    #[test]
    fn identity_vsa_guard_blocks_strongly_dissimilar_vectors() {
        use crate::vsa::{Encoding, HyperVec};

        let engine = test_engine();
        // Ingest two symbols so they are registered.
        engine
            .ingest_label_triples(&[(
                "Tecton".into(),
                "is-a".into(),
                "InfraTarget".into(),
                1.0,
            )])
            .unwrap();

        let tecton = engine.lookup_symbol("Tecton").unwrap();
        let target = engine.lookup_symbol("InfraTarget").unwrap();
        let dim = engine.ops().dim();
        let byte_len = dim.binary_byte_len();

        // Maximally dissimilar vectors: all-ones vs all-zeros → similarity 0.0.
        engine.item_memory().insert(
            tecton,
            HyperVec::from_raw(vec![0xFFu8; byte_len], dim, Encoding::Bipolar),
        );
        engine.item_memory().insert(
            target,
            HyperVec::zero(dim, Encoding::Bipolar),
        );

        // Guard must block (Tecton is identity, similarity 0.0 < threshold 0.40).
        assert!(
            identity_vsa_similarity_guard(
                &engine,
                tecton,
                target,
                IDENTITY_VSA_GUARD_THRESHOLD
            ),
            "guard should block strongly dissimilar is-a for identity entity"
        );

        // Reverse: non-identity subject → guard is a no-op.
        assert!(
            !identity_vsa_similarity_guard(
                &engine,
                target,
                tecton,
                IDENTITY_VSA_GUARD_THRESHOLD
            ),
            "guard must not block when subject is not an identity entity"
        );

        // Identical vectors → similarity 1.0 → guard allows.
        engine.item_memory().insert(
            target,
            HyperVec::from_raw(vec![0xFFu8; byte_len], dim, Encoding::Bipolar),
        );
        assert!(
            !identity_vsa_similarity_guard(
                &engine,
                tecton,
                target,
                IDENTITY_VSA_GUARD_THRESHOLD
            ),
            "guard must allow when vectors are identical"
        );
    }

    #[test]
    fn entity_type_guard_blocks_contains_part_inverse() {
        let engine = test_engine();
        // Stor01 is-a DigitalArchitect; Container contains Stor01.
        // The `contains-part-inverse` builtin rule has a DigitalArchitect guard,
        // so "Stor01 part-of Container" must NOT be derived.
        engine
            .ingest_label_triples(&[
                (
                    "Stor01".into(),
                    "is-a".into(),
                    "DigitalArchitect".into(),
                    1.0,
                ),
                ("Container".into(), "contains".into(), "Stor01".into(), 1.0),
            ])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let _ = re.run(&engine).unwrap();

        let stor01 = engine.lookup_symbol("Stor01").unwrap();
        let container = engine.lookup_symbol("Container").unwrap();
        // If the rule was correctly blocked, "part-of" was never registered and
        // lookup_symbol returns Err — trivially no spurious triple exists.
        assert!(
            engine.lookup_symbol("part-of").map_or(true, |part_of| {
                !engine.has_triple(stor01, part_of, container)
            }),
            "contains-part-inverse must not fire for DigitalArchitect entities"
        );
    }

    #[test]
    fn entity_type_guard_does_not_block_non_guarded_entities() {
        let engine = test_engine();
        // Box contains Widget — neither is a DigitalArchitect,
        // so `Widget part-of Box` should be derived normally.
        engine
            .ingest_label_triples(&[("Box".into(), "contains".into(), "Widget".into(), 1.0)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let _ = re.run(&engine).unwrap();

        let widget = engine.lookup_symbol("Widget").unwrap();
        let box_sym = engine.lookup_symbol("Box").unwrap();
        let part_of = engine.lookup_symbol("part-of").unwrap();
        assert!(
            engine.has_triple(widget, part_of, box_sym),
            "contains-part-inverse should fire for non-guarded entities"
        );
    }

    #[test]
    fn min_antecedent_confidence_blocks_low_conf_source() {
        let engine = test_engine();
        // ServiceA contains ServiceB with only 0.5 confidence (regex extraction).
        // contains-part-inverse requires min_antecedent_confidence = 0.7,
        // so the part-of edge must NOT be derived.
        engine
            .ingest_label_triples(&[("ServiceA".into(), "contains".into(), "ServiceB".into(), 0.5)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let _ = re.run(&engine).unwrap();

        let service_b = engine.lookup_symbol("ServiceB").unwrap();
        let service_a = engine.lookup_symbol("ServiceA").unwrap();
        // If the rule was blocked, "part-of" may never have been registered.
        assert!(
            engine.lookup_symbol("part-of").map_or(true, |part_of| {
                !engine.has_triple(service_b, part_of, service_a)
            }),
            "low-confidence contains must not spawn part-of edges"
        );
    }

    #[test]
    fn min_antecedent_confidence_allows_high_conf_source() {
        let engine = test_engine();
        // Box contains Lid with 0.9 confidence — above the 0.7 threshold.
        engine
            .ingest_label_triples(&[("Box".into(), "contains".into(), "Lid".into(), 0.9)])
            .unwrap();

        let re = RuleEngine::new(RuleEngineConfig::default()).with_rules(RuleSet::builtin());
        let _ = re.run(&engine).unwrap();

        let lid = engine.lookup_symbol("Lid").unwrap();
        let box_sym = engine.lookup_symbol("Box").unwrap();
        let part_of = engine.lookup_symbol("part-of").unwrap();
        assert!(
            engine.has_triple(lid, part_of, box_sym),
            "high-confidence contains should produce part-of edge"
        );
    }

    #[test]
    fn purge_false_part_of_topology_removes_infra_edges() {
        let engine = test_engine();
        // MacMini is-a DigitalArchitect; a false part-of edge was inserted.
        engine
            .ingest_label_triples(&[
                (
                    "MacMini".into(),
                    "is-a".into(),
                    "DigitalArchitect".into(),
                    1.0,
                ),
                ("MacMini".into(), "part-of".into(), "DataCenter".into(), 1.0),
            ])
            .unwrap();

        let mac = engine.lookup_symbol("MacMini").unwrap();
        let dc = engine.lookup_symbol("DataCenter").unwrap();
        let part_of = engine.lookup_symbol("part-of").unwrap();
        assert!(
            engine.has_triple(mac, part_of, dc),
            "precondition: triple exists before purge"
        );

        let removed = RuleEngine::purge_false_part_of_topology(&engine, &["DigitalArchitect"]);
        assert_eq!(
            removed, 1,
            "exactly one false topology triple should be removed"
        );
        assert!(
            !engine.has_triple(mac, part_of, dc),
            "the false part-of triple should be gone after purge"
        );
    }

    #[test]
    fn purge_false_part_of_topology_leaves_legitimate_edges() {
        let engine = test_engine();
        // Wheel is-a PhysicalPart (not DigitalArchitect); Car has Wheel part-of it.
        // Purge with DigitalArchitect guard should leave this edge alone.
        engine
            .ingest_label_triples(&[
                ("Wheel".into(), "is-a".into(), "PhysicalPart".into(), 1.0),
                ("Wheel".into(), "part-of".into(), "Car".into(), 1.0),
            ])
            .unwrap();

        let wheel = engine.lookup_symbol("Wheel").unwrap();
        let car = engine.lookup_symbol("Car").unwrap();
        let part_of = engine.lookup_symbol("part-of").unwrap();

        let removed = RuleEngine::purge_false_part_of_topology(&engine, &["DigitalArchitect"]);
        assert_eq!(removed, 0, "non-guarded entities should not be purged");
        assert!(
            engine.has_triple(wheel, part_of, car),
            "legitimate part-of edge must survive purge"
        );
    }

    #[test]
    fn purge_returns_zero_when_no_part_of_predicate_exists() {
        let engine = test_engine();
        // No triples at all — purge should be a no-op and not panic.
        let removed = RuleEngine::purge_false_part_of_topology(&engine, &["DigitalArchitect"]);
        assert_eq!(removed, 0);
    }
}
