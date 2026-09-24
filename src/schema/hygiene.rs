//! Memory hygiene: the pass that makes an akh's memory improve itself
//! (Phase 53j, ADR 113 decision 6).
//!
//! Before 53j nothing ever retracted anything: every status update of a
//! state-like predicate added one more live value (and one more
//! `TemporalConflict` warning), and the same predicate accumulated under
//! several spellings. One pass ([`run_hygiene`]) does three things, in this
//! order, within a bounded number of graph mutations:
//!
//! 1. **Auto-merge exact spelling collisions.** Predicates whose
//!    [`PredicateKey`](super::canonical::PredicateKey)s are equal (e.g.
//!    `next_check` / `next-check` / `next check`) are rewritten onto one
//!    deterministic canonical spelling ([`super::canonical`]). Nothing else
//!    is merged automatically: declared aliases whose keys differ and
//!    ambiguous families are *reported*; merge them explicitly with
//!    [`merge_predicates`] (the `kg_merge_predicates` tool and the
//!    `_akh/memory/merge_predicates` ACP method).
//! 2. **Single-valued backlog.** For every predicate a schema pack declares
//!    `single_valued`, each `(subject, compartment)` keeps only its newest
//!    value ([`crate::engine::newest_first`]); older values are retracted
//!    through the TMS with `memory_hygiene` provenance naming their
//!    successor (and, for `temporal` predicates, when they stopped holding).
//! 3. **Contradiction repair** for declared functional predicates, per the
//!    akh's `[memory] contradiction_policy`: `warn` reports, `replace` keeps
//!    the newest value, `reject` keeps the oldest (the value a `reject`
//!    policy would have admitted).
//!
//! Every step is idempotent: a second pass over an unchanged graph performs
//! no mutation. A pass that hits its mutation bound sets
//! [`HygieneReport::truncated`] and the next pass continues.

use std::collections::{BTreeMap, HashMap};

use serde::Serialize;

use crate::engine::{ContradictionPolicy, Engine, newest_first};
use crate::graph::Triple;
use crate::provenance::HygieneAction;
use crate::symbol::SymbolId;

use super::canonical::{FamilyVerdict, PredicateFamily, collisions};
use super::cardinality::{declared_aliases, declared_cardinality, single_valued_predicates};
use super::{SchemaError, SchemaResult};

/// Default bound on graph mutations per hygiene pass (`[memory]
/// hygiene_batch`).
pub const DEFAULT_HYGIENE_BATCH: usize = 1_000;

/// How many undeclared multi-value predicates a report lists.
const UNRESOLVED_REPORT_LIMIT: usize = 20;

/// Persisted provenance tag of [`crate::provenance::DerivationKind::MemoryHygiene`].
const MEMORY_HYGIENE_TAG: u8 = 109;

/// Options for one hygiene pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HygieneOptions {
    /// Contradiction-repair policy (the akh's `[memory] contradiction_policy`).
    pub policy: ContradictionPolicy,
    /// Upper bound on graph mutations this pass (retractions + rewrites).
    pub max_mutations: usize,
    /// Count what would change without mutating anything.
    pub dry_run: bool,
}

impl HygieneOptions {
    /// Options using `engine`'s configured contradiction policy.
    pub fn for_engine(engine: &Engine, max_mutations: usize) -> Self {
        Self {
            policy: engine.phase9_config().contradiction_policy,
            max_mutations,
            dry_run: false,
        }
    }
}

/// The remaining mutation allowance of one pass.
#[derive(Debug, Clone, Copy)]
pub struct MutationBudget {
    remaining: usize,
    used: usize,
    exhausted: bool,
}

impl MutationBudget {
    /// A budget of `max` mutations.
    pub fn new(max: usize) -> Self {
        Self {
            remaining: max,
            used: 0,
            exhausted: false,
        }
    }

    /// An effectively unbounded budget (explicit operator merges).
    pub fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    /// Take one mutation; `false` (and remembered) once exhausted.
    fn take(&mut self) -> bool {
        if self.remaining == 0 {
            self.exhausted = true;
            return false;
        }
        self.remaining -= 1;
        self.used += 1;
        true
    }

    /// Mutations taken so far.
    pub fn used(&self) -> usize {
        self.used
    }

    /// Whether some work was left undone for lack of budget.
    pub fn exhausted(&self) -> bool {
        self.exhausted
    }
}

/// One predicate merge.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct MergeReport {
    /// The canonical spelling.
    pub canonical: String,
    /// The alias spellings merged into it.
    pub aliases: Vec<String>,
    /// Alias triples rewritten onto the canonical predicate (or, in a dry
    /// run, that would be).
    pub moved: usize,
}

/// A declared alias that still carries live triples. Not auto-merged when
/// its spelling does not collide structurally with the canonical one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AliasBacklog {
    /// The alias spelling.
    pub alias: String,
    /// Its declared canonical relation.
    pub canonical: String,
    /// Live triples still under the alias.
    pub live_triples: usize,
}

/// An undeclared predicate whose subjects hold several live values.
/// Reported so an operator can declare it `single_valued` if it is state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnresolvedPredicate {
    /// The predicate label.
    pub predicate: String,
    /// `(subject, compartment)` groups holding more than one value.
    pub subjects_with_multiple_values: usize,
    /// Most values any one subject holds.
    pub max_values: usize,
}

/// What one hygiene pass did (or, dry-run, would do).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HygieneReport {
    /// Policy used for contradiction repair.
    pub policy: &'static str,
    /// Whether nothing was mutated.
    pub dry_run: bool,
    /// Auto-merged spelling-collision families.
    pub merged: Vec<MergeReport>,
    /// Older values of single-valued predicates retracted.
    pub superseded: usize,
    /// Functional-predicate values retracted by contradiction repair.
    pub repaired: usize,
    /// Functional-predicate groups still holding several values (reported
    /// under `warn`, or left over by a truncated pass).
    pub functional_violations: usize,
    /// `memory_hygiene` provenance records written by this pass.
    pub provenance_records: usize,
    /// Graph mutations taken from the budget.
    pub mutations: usize,
    /// The mutation bound stopped the pass early; the next pass continues.
    pub truncated: bool,
    /// Collision families with several declared spellings (not merged).
    pub ambiguous_families: Vec<PredicateFamily>,
    /// Declared aliases still carrying triples (merge with
    /// `kg_merge_predicates`).
    pub pending_declared_aliases: Vec<AliasBacklog>,
    /// Undeclared predicates holding several values per subject, most
    /// affected first (at most 20).
    pub unresolved: Vec<UnresolvedPredicate>,
}

impl HygieneReport {
    /// Whether the pass changed nothing and found nothing left to change.
    pub fn is_noop(&self) -> bool {
        self.mutations == 0 && !self.truncated
    }

    /// Triples moved by every merge in this pass.
    pub fn merged_triples(&self) -> usize {
        self.merged.iter().map(|m| m.moved).sum()
    }
}

fn hygiene_err(message: impl std::fmt::Display) -> SchemaError {
    SchemaError::Hygiene {
        message: message.to_string(),
    }
}

fn merge_err(message: impl std::fmt::Display) -> SchemaError {
    SchemaError::Merge {
        message: message.to_string(),
    }
}

fn hygiene_record_count(engine: &Engine) -> usize {
    engine
        .provenance_by_kind_tags(&[MEMORY_HYGIENE_TAG])
        .map(|records| records.len())
        .unwrap_or(0)
}

/// Run one hygiene pass over `engine`.
pub fn run_hygiene(engine: &Engine, options: HygieneOptions) -> SchemaResult<HygieneReport> {
    let records_before = hygiene_record_count(engine);
    let mut budget = MutationBudget::new(options.max_mutations);

    // 1. Exact spelling collisions.
    let families = collisions(engine);
    let mut merged = Vec::new();
    let mut ambiguous_families = Vec::new();
    for family in families {
        match family.verdict {
            FamilyVerdict::Ambiguous => ambiguous_families.push(family),
            // Reported through `pending_declared_aliases` below.
            FamilyVerdict::DeclaredAliases => {}
            FamilyVerdict::Merge { .. } => {
                let (Some(canonical), aliases) =
                    (family.canonical(), family.aliases_with_triples())
                else {
                    continue;
                };
                if aliases.is_empty() {
                    continue;
                }
                let alias_ids: Vec<SymbolId> = aliases.iter().map(|a| a.symbol).collect();
                let report = merge_predicates(
                    engine,
                    canonical.symbol,
                    &alias_ids,
                    &mut budget,
                    options.dry_run,
                )?;
                if report.moved > 0 {
                    merged.push(report);
                }
            }
        }
    }

    // 2. Single-valued backlog.
    let superseded = supersede_backlog(engine, &mut budget, options.dry_run)?;

    // 3. Contradiction repair on functional predicates.
    let (repaired, functional_violations) =
        repair_functional(engine, options.policy, &mut budget, options.dry_run)?;

    // Reports.
    let pending_declared_aliases = declared_aliases(engine)
        .into_iter()
        .filter_map(|(alias, canonical)| {
            let live = engine.triples_with_predicate(alias).len();
            (live > 0).then(|| AliasBacklog {
                alias: engine.resolve_label(alias),
                canonical: engine.resolve_label(canonical),
                live_triples: live,
            })
        })
        .collect();
    let unresolved = unresolved_predicates(engine);

    Ok(HygieneReport {
        policy: options.policy.as_str(),
        dry_run: options.dry_run,
        merged,
        superseded,
        repaired,
        functional_violations,
        provenance_records: hygiene_record_count(engine).saturating_sub(records_before),
        mutations: budget.used(),
        truncated: budget.exhausted(),
        ambiguous_families,
        pending_declared_aliases,
        unresolved,
    })
}

/// Rewrite every live triple of `aliases` onto `canonical` (retract +
/// re-assert with `memory_hygiene` provenance; see
/// [`Engine::rewrite_predicate`]).
///
/// Triples move oldest first, so when `canonical` is single-valued the
/// newest value (across all spellings) ends up live and the older ones are
/// superseded with provenance. Idempotent: an alias with no live triples
/// moves nothing.
///
/// # Errors
///
/// [`SchemaError::Merge`] when `canonical` is among `aliases`, and
/// [`SchemaError::Hygiene`] when an engine write fails (triples moved
/// before the failure stay moved).
pub fn merge_predicates(
    engine: &Engine,
    canonical: SymbolId,
    aliases: &[SymbolId],
    budget: &mut MutationBudget,
    dry_run: bool,
) -> SchemaResult<MergeReport> {
    if aliases.contains(&canonical) {
        return Err(merge_err(format!(
            "the canonical predicate \"{}\" is also listed as an alias",
            engine.resolve_label(canonical)
        )));
    }
    let mut edges: Vec<Triple> = aliases
        .iter()
        .flat_map(|alias| engine.triples_with_predicate(*alias))
        .collect();
    // Oldest first; `newest_first` reversed keeps the tie-breaks consistent
    // with insert-time supersession.
    edges.sort_by(|a, b| newest_first(b, a));

    let mut moved = 0usize;
    for edge in &edges {
        if !budget.take() {
            break;
        }
        if dry_run {
            moved += 1;
            continue;
        }
        if engine
            .rewrite_predicate(edge, canonical)
            .map_err(hygiene_err)?
        {
            moved += 1;
        }
    }
    Ok(MergeReport {
        canonical: engine.resolve_label(canonical),
        aliases: aliases.iter().map(|a| engine.resolve_label(*a)).collect(),
        moved,
    })
}

/// Merge every declared alias (from installed schema packs) onto its
/// canonical relation, whether or not the spellings collide structurally.
/// The explicit operator path (`kg_merge_predicates mode=declared`).
pub fn merge_declared_aliases(
    engine: &Engine,
    budget: &mut MutationBudget,
    dry_run: bool,
) -> SchemaResult<Vec<MergeReport>> {
    let mut by_canonical: BTreeMap<SymbolId, Vec<SymbolId>> = BTreeMap::new();
    for (alias, canonical) in declared_aliases(engine) {
        by_canonical.entry(canonical).or_default().push(alias);
    }
    let mut reports = Vec::new();
    for (canonical, aliases) in by_canonical {
        let report = merge_predicates(engine, canonical, &aliases, budget, dry_run)?;
        if report.moved > 0 {
            reports.push(report);
        }
    }
    Ok(reports)
}

/// Merge every exact spelling-collision family onto its canonical spelling
/// (what the daemon pass does in step 1).
pub fn merge_collisions(
    engine: &Engine,
    budget: &mut MutationBudget,
    dry_run: bool,
) -> SchemaResult<Vec<MergeReport>> {
    let mut reports = Vec::new();
    for family in collisions(engine) {
        let Some(canonical) = family.canonical() else {
            continue;
        };
        let aliases: Vec<SymbolId> = family
            .aliases_with_triples()
            .iter()
            .map(|a| a.symbol)
            .collect();
        if aliases.is_empty() {
            continue;
        }
        let report = merge_predicates(engine, canonical.symbol, &aliases, budget, dry_run)?;
        if report.moved > 0 {
            reports.push(report);
        }
    }
    Ok(reports)
}

/// Default mutation bound of one explicit merge request.
pub const DEFAULT_MERGE_LIMIT: usize = 10_000;

/// Which predicates an explicit merge request rewrites.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize, Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum MergeSelection {
    /// These alias spellings onto this canonical spelling (labels or ids;
    /// every symbol must already exist).
    Explicit {
        canonical: String,
        aliases: Vec<String>,
    },
    /// Every alias declared by an installed schema pack onto its relation.
    Declared,
    /// Every exact spelling collision onto its deterministic canonical
    /// spelling (what the daemon pass does on its own).
    Collisions,
}

/// An explicit merge request (the `kg_merge_predicates` tool and the
/// `_akh/memory/merge_predicates` ACP method).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeRequest {
    /// What to merge.
    pub selection: MergeSelection,
    /// Count without mutating.
    pub dry_run: bool,
    /// Upper bound on rewritten triples.
    pub limit: usize,
}

/// The result of a [`MergeRequest`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MergeOutcome {
    /// Whether nothing was mutated.
    pub dry_run: bool,
    /// One entry per canonical predicate that received triples.
    pub merged: Vec<MergeReport>,
    /// Triples moved in total.
    pub moved: usize,
    /// `limit` stopped the request early; run it again to continue.
    pub truncated: bool,
}

/// Execute an explicit merge request.
///
/// # Errors
///
/// [`SchemaError::Merge`] for an unknown label, an empty alias list, or a
/// canonical predicate listed among its aliases; [`SchemaError::Hygiene`]
/// when an engine write fails.
pub fn execute_merge(engine: &Engine, request: &MergeRequest) -> SchemaResult<MergeOutcome> {
    let mut budget = MutationBudget::new(request.limit);
    let merged = match &request.selection {
        MergeSelection::Explicit { canonical, aliases } => {
            if aliases.is_empty() {
                return Err(merge_err(
                    "an explicit merge needs at least one alias to fold into the canonical predicate",
                ));
            }
            let resolve = |label: &str| {
                engine.resolve_symbol(label).map_err(|_| {
                    merge_err(format!(
                        "no symbol named \"{label}\"; nothing was rewritten. Check the spelling \
                         (labels are exact) or pass the numeric symbol id"
                    ))
                })
            };
            let canonical = resolve(canonical)?;
            let alias_ids = aliases
                .iter()
                .map(|a| resolve(a))
                .collect::<SchemaResult<Vec<_>>>()?;
            let report =
                merge_predicates(engine, canonical, &alias_ids, &mut budget, request.dry_run)?;
            if report.moved > 0 {
                vec![report]
            } else {
                Vec::new()
            }
        }
        MergeSelection::Declared => merge_declared_aliases(engine, &mut budget, request.dry_run)?,
        MergeSelection::Collisions => merge_collisions(engine, &mut budget, request.dry_run)?,
    };
    let moved = merged.iter().map(|m| m.moved).sum();
    Ok(MergeOutcome {
        dry_run: request.dry_run,
        merged,
        moved,
        truncated: budget.exhausted(),
    })
}

/// Group `triples` by `(subject, compartment)`, keeping groups with more
/// than one distinct object, each sorted newest first. Deterministic order.
fn multi_value_groups(triples: Vec<Triple>) -> Vec<Vec<Triple>> {
    let mut groups: BTreeMap<(SymbolId, Option<String>), Vec<Triple>> = BTreeMap::new();
    for triple in triples {
        groups
            .entry((triple.subject, triple.compartment_id.clone()))
            .or_default()
            .push(triple);
    }
    groups
        .into_values()
        .filter(|values| values.len() > 1)
        .map(|mut values| {
            values.sort_by(newest_first);
            values
        })
        .collect()
}

/// Retract every non-newest value of each declared single-valued predicate.
fn supersede_backlog(
    engine: &Engine,
    budget: &mut MutationBudget,
    dry_run: bool,
) -> SchemaResult<usize> {
    let mut superseded = 0usize;
    for (predicate, cardinality) in single_valued_predicates(engine) {
        for values in multi_value_groups(engine.triples_with_predicate(predicate)) {
            // values[0] is live; each older value was replaced by the next
            // newer one, which is when it stopped holding.
            for pair in values.windows(2) {
                let (successor, stale) = (&pair[0], &pair[1]);
                if !budget.take() {
                    return Ok(superseded);
                }
                if dry_run {
                    superseded += 1;
                    continue;
                }
                let valid_until = cardinality.is_temporal().then_some(successor.timestamp);
                if engine
                    .supersede_triple(
                        stale,
                        successor,
                        HygieneAction::SingleValuedSupersede,
                        valid_until,
                    )
                    .map_err(hygiene_err)?
                {
                    superseded += 1;
                }
            }
        }
    }
    Ok(superseded)
}

/// Repair functional-predicate contradictions per `policy`. Returns
/// `(repaired, groups still violating)`.
fn repair_functional(
    engine: &Engine,
    policy: ContradictionPolicy,
    budget: &mut MutationBudget,
    dry_run: bool,
) -> SchemaResult<(usize, usize)> {
    let functional = engine.functional_preds().sorted();
    let mut repaired = 0usize;
    let mut violations = 0usize;
    for predicate in functional {
        // Single-valued supersession already owns these.
        if declared_cardinality(engine, predicate).is_single_valued() {
            continue;
        }
        for mut values in multi_value_groups(engine.triples_with_predicate(predicate)) {
            let keep = match policy {
                ContradictionPolicy::Warn => {
                    violations += 1;
                    continue;
                }
                // Newest first: keep index 0.
                ContradictionPolicy::Replace => values.remove(0),
                // What `reject` would have admitted: the oldest value.
                ContradictionPolicy::Reject => values.pop().expect("group has two values"),
            };
            for stale in &values {
                if !budget.take() {
                    violations += 1;
                    return Ok((repaired, violations));
                }
                if dry_run {
                    repaired += 1;
                    continue;
                }
                if engine
                    .supersede_triple(stale, &keep, HygieneAction::ContradictionRepair, None)
                    .map_err(hygiene_err)?
                {
                    repaired += 1;
                }
            }
        }
    }
    Ok((repaired, violations))
}

/// Undeclared, non-multi-valued predicates whose subjects hold several live
/// values, most affected first.
fn unresolved_predicates(engine: &Engine) -> Vec<UnresolvedPredicate> {
    let mut per_group: HashMap<(SymbolId, SymbolId, Option<String>), usize> = HashMap::new();
    for triple in engine.all_triples() {
        *per_group
            .entry((triple.predicate, triple.subject, triple.compartment_id))
            .or_default() += 1;
    }
    let mut per_predicate: HashMap<SymbolId, (usize, usize)> = HashMap::new();
    for ((predicate, _, _), values) in per_group {
        if values > 1 {
            let entry = per_predicate.entry(predicate).or_default();
            entry.0 += 1;
            entry.1 = entry.1.max(values);
        }
    }
    let multi_valued = engine.multi_valued_preds();
    let functional = engine.functional_preds();
    let mut unresolved: Vec<UnresolvedPredicate> = per_predicate
        .into_iter()
        .filter(|(predicate, _)| {
            !multi_valued.is_multi_valued(*predicate)
                && !functional.is_functional(*predicate)
                && !declared_cardinality(engine, *predicate).is_single_valued()
        })
        .map(|(predicate, (groups, max))| UnresolvedPredicate {
            predicate: engine.resolve_label(predicate),
            subjects_with_multiple_values: groups,
            max_values: max,
        })
        .collect();
    unresolved.sort_by(|a, b| {
        b.subjects_with_multiple_values
            .cmp(&a.subjects_with_multiple_values)
            .then_with(|| a.predicate.cmp(&b.predicate))
    });
    unresolved.truncate(UNRESOLVED_REPORT_LIMIT);
    unresolved
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::EngineConfig;
    use crate::provenance::DerivationKind;
    use crate::schema::{compile, install_pack};

    const PACK: &str = r#"{
        "pack": {"name": "hygiene-test", "version": "0.1.0"},
        "relations": [
            {"name": "next_check", "cardinality": "single_valued", "temporal": true,
             "aliases": ["has_next_check"]},
            {"name": "status", "cardinality": "single_valued"}
        ]
    }"#;

    fn engine() -> (Engine, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(dir.path().join("kg")),
            ..Default::default()
        })
        .unwrap();
        (engine, dir)
    }

    fn put(engine: &Engine, s: &str, p: &str, o: &str, ts: u64) -> Triple {
        let triple = Triple {
            timestamp: ts,
            ..Triple::new(
                engine.resolve_or_create_entity(s).unwrap(),
                engine.resolve_or_create_relation(p).unwrap(),
                engine.resolve_or_create_entity(o).unwrap(),
            )
        };
        engine.add_triple(&triple).unwrap();
        triple
    }

    fn live_objects(engine: &Engine, s: &str, p: &str) -> Vec<String> {
        let s = engine.lookup_symbol(s).unwrap();
        let p = engine.lookup_symbol(p).unwrap();
        let mut objects: Vec<String> = engine
            .triples_from(s)
            .into_iter()
            .filter(|t| t.predicate == p)
            .map(|t| engine.resolve_label(t.object))
            .collect();
        objects.sort();
        objects
    }

    fn options() -> HygieneOptions {
        HygieneOptions {
            policy: ContradictionPolicy::Warn,
            max_mutations: DEFAULT_HYGIENE_BATCH,
            dry_run: false,
        }
    }

    /// The Roman shape: three spellings, several values per subject, written
    /// before any pack existed. One pass leaves one live value per subject
    /// under one spelling; a second pass is a no-op.
    #[test]
    fn backlog_pass_merges_supersedes_and_is_idempotent() {
        let (engine, _dir) = engine();
        put(&engine, "janus", "next_check", "a", 100);
        put(&engine, "janus", "next-check", "b", 200);
        put(&engine, "janus", "next check", "c", 300);
        put(&engine, "janus", "has_next_check", "d", 50);
        put(&engine, "vesta", "next-check", "x", 10);
        put(&engine, "vesta", "next check", "y", 20);
        put(&engine, "vesta", "status", "open", 1);
        put(&engine, "vesta", "status", "resolved", 2);

        install_pack(&engine, &compile(PACK).unwrap()).unwrap();

        let report = run_hygiene(&engine, options()).unwrap();
        assert!(!report.truncated);
        assert_eq!(report.merged.len(), 1, "{report:?}");
        assert_eq!(report.merged[0].canonical, "next_check");
        assert_eq!(report.merged_triples(), 4);
        assert_eq!(live_objects(&engine, "janus", "next_check"), vec!["c"]);
        assert_eq!(live_objects(&engine, "vesta", "next_check"), vec!["y"]);
        assert!(live_objects(&engine, "janus", "next-check").is_empty());
        assert_eq!(live_objects(&engine, "vesta", "status"), vec!["resolved"]);
        // The declared, non-colliding alias is reported, not merged.
        assert_eq!(report.pending_declared_aliases.len(), 1);
        assert_eq!(report.pending_declared_aliases[0].alias, "has_next_check");
        assert!(report.provenance_records > 0);

        let hygiene = engine
            .provenance_by_kind_tags(&[MEMORY_HYGIENE_TAG])
            .unwrap();
        assert!(hygiene.iter().any(|r| matches!(
            r.kind,
            DerivationKind::MemoryHygiene {
                action: HygieneAction::SingleValuedSupersede,
                valid_until: Some(300),
                ..
            }
        )));
        assert!(hygiene.iter().any(|r| matches!(
            r.kind,
            DerivationKind::MemoryHygiene {
                action: HygieneAction::PredicateMerge,
                ..
            }
        )));

        let again = run_hygiene(&engine, options()).unwrap();
        assert!(again.is_noop(), "second pass must be a no-op: {again:?}");
        assert_eq!(again.provenance_records, 0);

        // The explicit operator merge folds the declared alias in; since
        // `d` is older than `c`, it is superseded on arrival.
        let merged =
            merge_declared_aliases(&engine, &mut MutationBudget::unbounded(), false).unwrap();
        assert_eq!(merged.len(), 1);
        assert_eq!(live_objects(&engine, "janus", "next_check"), vec!["c"]);
        assert!(live_objects(&engine, "janus", "has_next_check").is_empty());
    }

    #[test]
    fn a_bounded_pass_truncates_and_the_next_one_finishes() {
        let (engine, _dir) = engine();
        install_pack(&engine, &compile(PACK).unwrap()).unwrap();
        // Written with supersession off: simulate the pre-pack backlog by
        // asserting under an undeclared predicate and merging later.
        for i in 0..5u64 {
            put(&engine, "janus", "next check", &format!("v{i}"), i);
        }
        let first = run_hygiene(
            &engine,
            HygieneOptions {
                max_mutations: 2,
                ..options()
            },
        )
        .unwrap();
        assert!(first.truncated);
        assert_eq!(first.mutations, 2);
        let rest = run_hygiene(&engine, options()).unwrap();
        assert!(!rest.truncated);
        assert_eq!(live_objects(&engine, "janus", "next_check"), vec!["v4"]);
        assert!(run_hygiene(&engine, options()).unwrap().is_noop());
    }

    #[test]
    fn dry_run_counts_without_mutating() {
        let (engine, _dir) = engine();
        put(&engine, "vesta", "status", "open", 1);
        put(&engine, "vesta", "status", "resolved", 2);
        install_pack(&engine, &compile(PACK).unwrap()).unwrap();
        let report = run_hygiene(
            &engine,
            HygieneOptions {
                dry_run: true,
                ..options()
            },
        )
        .unwrap();
        assert_eq!(report.superseded, 1);
        assert_eq!(
            live_objects(&engine, "vesta", "status"),
            vec!["open", "resolved"]
        );
    }

    #[test]
    fn functional_repair_follows_the_policy() {
        for (policy, kept) in [
            (ContradictionPolicy::Warn, vec!["new", "old"]),
            (ContradictionPolicy::Replace, vec!["new"]),
            (ContradictionPolicy::Reject, vec!["old"]),
        ] {
            let (engine, _dir) = engine();
            put(&engine, "alice", "spouse", "old", 1);
            put(&engine, "alice", "spouse", "new", 2);
            let spouse = engine.lookup_symbol("spouse").unwrap();
            engine.functional_preds_mut().declare_functional(spouse);
            let report = run_hygiene(
                &engine,
                HygieneOptions {
                    policy,
                    ..options()
                },
            )
            .unwrap();
            assert_eq!(live_objects(&engine, "alice", "spouse"), kept, "{policy:?}");
            match policy {
                ContradictionPolicy::Warn => {
                    assert_eq!((report.repaired, report.functional_violations), (0, 1));
                }
                _ => assert_eq!((report.repaired, report.functional_violations), (1, 0)),
            }
        }
    }

    #[test]
    fn merge_rejects_canonical_among_aliases() {
        let (engine, _dir) = engine();
        let p = engine.resolve_or_create_relation("state").unwrap();
        let err = merge_predicates(&engine, p, &[p], &mut MutationBudget::unbounded(), false)
            .unwrap_err();
        assert!(matches!(err, SchemaError::Merge { .. }));
    }

    #[test]
    fn undeclared_multi_value_predicates_are_reported() {
        let (engine, _dir) = engine();
        put(&engine, "wi", "observed", "a", 1);
        put(&engine, "wi", "observed", "b", 2);
        let report = run_hygiene(&engine, options()).unwrap();
        assert!(report.is_noop());
        assert_eq!(report.unresolved.len(), 1);
        assert_eq!(report.unresolved[0].predicate, "observed");
        assert_eq!(live_objects(&engine, "wi", "observed"), vec!["a", "b"]);
    }
}
