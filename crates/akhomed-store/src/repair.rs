//! Transactional repair of legacy symbol registries and allocator state.
//!
//! The engine owns duplicate detection and normalization semantics. This leaf
//! crate receives an already-validated loser→canonical plan and applies it to
//! every relational symbol reference in one transaction, including primary-key
//! collision handling, allocator reconciliation, and mutation-WAL audit rows.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use sea_orm::ActiveValue::{NotSet, Set};
use sea_orm::sea_query::OnConflict;
use sea_orm::{
    ColumnTrait, ConnectionTrait, DatabaseBackend, DatabaseConnection, EntityTrait,
    IntoActiveModel, QueryFilter, Statement, TransactionTrait,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::entity::{
    agenda, akh_state, episode, fact, goal, goal_dependency, goal_hierarchy, learned_method,
    learned_method_step, mutation_log, provenance, provenance_input, provenance_snapshot, symbol,
};
use crate::error::{StoreError, StoreResult};
use crate::mutations::MutationRow;

const ROWS_PER_CHUNK: usize = 6_000;

/// One deterministic loser→canonical rewrite selected by the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolMerge {
    pub canonical_id: i64,
    pub duplicate_id: i64,
}

/// Row counts changed or collapsed by a symbol repair.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolRepairCounts {
    pub symbols_deleted: usize,
    pub facts_rewritten: usize,
    pub facts_collapsed: usize,
    pub provenance_rewritten: usize,
    pub provenance_inputs_rewritten: usize,
    pub provenance_snapshots_rewritten: usize,
    pub agenda_rewritten: usize,
    pub agenda_collapsed: usize,
    pub goals_rewritten: usize,
    pub goals_collapsed: usize,
    pub goal_hierarchy_rewritten: usize,
    pub goal_hierarchy_collapsed: usize,
    pub goal_dependencies_rewritten: usize,
    pub goal_dependencies_collapsed: usize,
    pub learned_methods_rewritten: usize,
    pub learned_methods_collapsed: usize,
    pub learned_method_steps_rewritten: usize,
    pub learned_method_steps_collapsed: usize,
    pub episodes_rewritten: usize,
    pub episodes_collapsed: usize,
    pub wal_records: usize,
}

/// Allocator reconciliation observed and applied under the repair transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolAllocatorRepair {
    pub allocator_next_before: i64,
    pub max_surviving_symbol_id: Option<i64>,
    pub required_allocator_next: i64,
    pub allocator_next_after: i64,
    pub advanced: bool,
}

/// Complete result of the transactional symbol repair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolRepairOutcome {
    pub counts: SymbolRepairCounts,
    pub allocator: SymbolAllocatorRepair,
}

fn mapping_from(merges: &[SymbolMerge]) -> StoreResult<HashMap<i64, i64>> {
    let mut mapping = HashMap::with_capacity(merges.len());
    let mut canonicals = HashSet::with_capacity(merges.len());
    for merge in merges {
        if merge.canonical_id <= 0
            || merge.duplicate_id <= 0
            || merge.canonical_id >= merge.duplicate_id
        {
            return Err(StoreError::Conflict {
                message: format!(
                    "invalid symbol merge {} -> {}: canonical must be a positive, lower ID",
                    merge.duplicate_id, merge.canonical_id
                ),
            });
        }
        if mapping
            .insert(merge.duplicate_id, merge.canonical_id)
            .is_some()
        {
            return Err(StoreError::Conflict {
                message: format!("duplicate repair entry for symbol {}", merge.duplicate_id),
            });
        }
        canonicals.insert(merge.canonical_id);
    }
    if let Some(chained) = mapping.keys().find(|id| canonicals.contains(id)) {
        return Err(StoreError::Conflict {
            message: format!("repair plan contains chained canonical symbol {chained}"),
        });
    }
    Ok(mapping)
}

fn remap(id: i64, mapping: &HashMap<i64, i64>) -> i64 {
    mapping.get(&id).copied().unwrap_or(id)
}

fn rewrite_number(value: &mut serde_json::Value, mapping: &HashMap<i64, i64>) -> bool {
    let Some(id) = value.as_i64() else {
        return false;
    };
    let rewritten = remap(id, mapping);
    if rewritten == id {
        return false;
    }
    *value = serde_json::Value::from(rewritten);
    true
}

/// Rewrite symbol-shaped fields in legacy JSON payloads without touching
/// unrelated numeric counters/timestamps which may happen to equal a loser ID.
fn rewrite_symbol_json(value: &mut serde_json::Value, mapping: &HashMap<i64, i64>) -> bool {
    const SCALAR_FIELDS: &[&str] = &[
        "symbol_id",
        "episode_id",
        "derived_id",
        "derived_symbol",
        "subject",
        "predicate",
        "object",
        "from",
        "goal",
        "gap_entity",
        "reactivated_goal",
        "goal_id",
        "parent",
        "reformulated_from",
        "replacement",
        "primary",
        "source_note",
        "spawned_from",
        "input_symbol_id",
        "context",
        "ancestor",
        "from_context",
        "to_context",
        "specific",
        "general",
        "inverse",
        "winner",
        "loser",
        "existing_object",
        "incoming_object",
        "existential_relation",
        "bound_var",
        "skolem",
        "concrete_entity",
        "function",
        "source_goal",
        "concept",
        "a",
        "b",
        "c",
    ];
    const ARRAY_FIELDS: &[&str] = &[
        "sources",
        "inputs",
        "children",
        "blocked_by",
        "antecedents",
        "supporting",
        "learnings",
        "tags",
        "symbols",
    ];

    match value {
        serde_json::Value::Array(values) => {
            let mut changed = false;
            for value in values {
                changed |= rewrite_symbol_json(value, mapping);
            }
            changed
        }
        serde_json::Value::Object(object) => {
            let mut changed = false;
            for (key, child) in object {
                if SCALAR_FIELDS.contains(&key.as_str()) {
                    changed |= rewrite_number(child, mapping);
                } else if ARRAY_FIELDS.contains(&key.as_str()) {
                    if let serde_json::Value::Array(values) = child {
                        for value in values {
                            changed |= rewrite_number(value, mapping);
                            changed |= rewrite_symbol_json(value, mapping);
                        }
                    }
                } else {
                    changed |= rewrite_symbol_json(child, mapping);
                }
            }
            changed
        }
        _ => false,
    }
}

fn rewrite_symbol_array(value: &mut serde_json::Value, mapping: &HashMap<i64, i64>) -> bool {
    let serde_json::Value::Array(values) = value else {
        return rewrite_symbol_json(value, mapping);
    };
    let mut changed = false;
    for value in values {
        changed |= rewrite_number(value, mapping);
    }
    changed
}

fn rewrite_goal_status(status: &mut String, mapping: &HashMap<i64, i64>) -> bool {
    for prefix in ["reformulated:", "duplicate:"] {
        if let Some(raw) = status.strip_prefix(prefix)
            && let Ok(id) = raw.parse::<i64>()
        {
            let rewritten = remap(id, mapping);
            if rewritten != id {
                *status = format!("{prefix}{rewritten}");
                return true;
            }
        }
    }
    false
}

fn remap_and_collapse_facts(
    mut rows: Vec<fact::Model>,
    mapping: &HashMap<i64, i64>,
) -> (Vec<fact::Model>, usize, usize) {
    rows.sort_by_key(|row| (row.workspace_id, row.subject, row.predicate, row.object));
    let input_len = rows.len();
    let mut rewritten = 0;
    let mut by_key = BTreeMap::new();
    for mut row in rows {
        let before = (row.subject, row.predicate, row.object);
        row.subject = remap(row.subject, mapping);
        row.predicate = remap(row.predicate, mapping);
        row.object = remap(row.object, mapping);
        rewritten += usize::from(before != (row.subject, row.predicate, row.object));
        by_key
            .entry((row.workspace_id, row.subject, row.predicate, row.object))
            .and_modify(|existing: &mut fact::Model| {
                if row.confidence > existing.confidence {
                    *existing = row.clone();
                }
            })
            .or_insert(row);
    }
    let rows: Vec<_> = by_key.into_values().collect();
    let collapsed = input_len - rows.len();
    (rows, rewritten, collapsed)
}

fn remap_and_collapse_episodes(
    mut rows: Vec<episode::Model>,
    mapping: &HashMap<i64, i64>,
) -> (Vec<episode::Model>, usize, usize) {
    rows.sort_by_key(|row| row.episode_id);
    let input_len = rows.len();
    let mut rewritten = 0;
    let mut by_id = BTreeMap::new();
    for mut row in rows {
        let mut changed = false;
        let new_id = remap(row.episode_id, mapping);
        changed |= new_id != row.episode_id;
        row.episode_id = new_id;
        changed |= rewrite_symbol_json(&mut row.payload, mapping);
        rewritten += usize::from(changed);
        by_id.entry(row.episode_id).or_insert(row);
    }
    let rows: Vec<_> = by_id.into_values().collect();
    let collapsed = input_len - rows.len();
    (rows, rewritten, collapsed)
}

macro_rules! delete_akh_rows {
    ($entity:ident, $txn:expr, $akh_id:expr) => {
        $entity::Entity::delete_many()
            .filter($entity::Column::AkhId.eq($akh_id))
            .exec($txn)
            .await?
    };
}

macro_rules! insert_chunks {
    ($entity:ident, $txn:expr, $rows:expr, [$($column:expr),+ $(,)?]) => {{
        for chunk in $rows.chunks(if ($txn).get_database_backend() == sea_orm::DatabaseBackend::Sqlite { 500 } else { ROWS_PER_CHUNK }) {
            if !chunk.is_empty() {
                $entity::Entity::insert_many(
                    chunk.iter().cloned().map(IntoActiveModel::into_active_model),
                )
                .on_conflict(OnConflict::columns([$($column),+]).do_nothing().to_owned())
                .exec($txn)
                .await?;
            }
        }
    }};
}

/// Apply a validated duplicate-symbol plan atomically, reconcile the allocator
/// against the maximum surviving relational symbol, and append audit WAL rows
/// in the same transaction.
pub async fn apply_symbol_repair(
    db: &DatabaseConnection,
    akh_id: Uuid,
    workspace_id: Uuid,
    merges: &[SymbolMerge],
    audit_rows: &[MutationRow],
) -> StoreResult<SymbolRepairOutcome> {
    let mapping = mapping_from(merges)?;

    let txn = db.begin().await?;
    if txn.get_database_backend() == DatabaseBackend::Postgres {
        txn.query_one(Statement::from_sql_and_values(
            DatabaseBackend::Postgres,
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            [format!("snapshot:{workspace_id}").into()],
        ))
        .await?;
    }

    let symbols = symbol::Entity::find()
        .filter(symbol::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    if let Some(row) = symbols.iter().find(|row| row.id <= 0) {
        return Err(StoreError::Conflict {
            message: format!(
                "cannot repair symbol allocator for akh {akh_id}: invalid relational symbol ID {}",
                row.id
            ),
        });
    }
    let known_ids: HashSet<i64> = symbols.iter().map(|row| row.id).collect();
    for merge in merges {
        if !known_ids.contains(&merge.canonical_id) || !known_ids.contains(&merge.duplicate_id) {
            return Err(StoreError::Conflict {
                message: format!(
                    "symbol repair plan is stale: merge {} -> {} no longer exists",
                    merge.duplicate_id, merge.canonical_id
                ),
            });
        }
    }

    let max_surviving_symbol_id = symbols
        .iter()
        .filter(|row| !mapping.contains_key(&row.id))
        .map(|row| row.id)
        .max();
    let required_allocator_next = match max_surviving_symbol_id {
        Some(max_id) => max_id.checked_add(1).ok_or_else(|| StoreError::Conflict {
            message: format!(
                "cannot repair symbol allocator for akh {akh_id}: symbol ID space overflows after {max_id}"
            ),
        })?,
        None => 1,
    };
    let state = akh_state::Entity::find_by_id(akh_id)
        .one(&txn)
        .await?
        .ok_or_else(|| StoreError::Conflict {
            message: format!(
                "cannot repair symbol allocator for akh {akh_id}: akh_state is missing"
            ),
        })?;
    let allocator_next_before = state.allocator_next;
    let allocator_next_after = allocator_next_before.max(required_allocator_next);
    let allocator = SymbolAllocatorRepair {
        allocator_next_before,
        max_surviving_symbol_id,
        required_allocator_next,
        allocator_next_after,
        advanced: allocator_next_after > allocator_next_before,
    };

    let advance_allocator = |state: akh_state::Model| {
        let mut active = state.into_active_model();
        active.allocator_next = Set(allocator_next_after);
        active.updated_at = Set(chrono::Utc::now());
        active
    };

    if mapping.is_empty() {
        if allocator.advanced {
            akh_state::Entity::update(advance_allocator(state))
                .exec(&txn)
                .await?;
        }
        txn.commit().await?;
        return Ok(SymbolRepairOutcome {
            counts: SymbolRepairCounts::default(),
            allocator,
        });
    }

    let mut counts = SymbolRepairCounts::default();

    let facts = fact::Entity::find()
        // Wake binds facts through the workspace, not through the Akh which
        // originally asserted each row. Repair that same relational image so
        // shared/imported facts cannot retain a losing symbol ID.
        .filter(fact::Column::WorkspaceId.eq(workspace_id))
        .all(&txn)
        .await?;
    let (facts, facts_rewritten, facts_collapsed) = remap_and_collapse_facts(facts, &mapping);
    counts.facts_rewritten = facts_rewritten;
    counts.facts_collapsed = facts_collapsed;

    let mut provenance_rows = provenance::Entity::find()
        .filter(provenance::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    let provenance_ids: Vec<i64> = provenance_rows
        .iter()
        .map(|row| row.provenance_id)
        .collect();
    for row in &mut provenance_rows {
        let mut changed = false;
        let subject = remap(row.subject, &mapping);
        changed |= subject != row.subject;
        row.subject = subject;
        changed |= rewrite_symbol_json(&mut row.derivation, &mapping);
        counts.provenance_rewritten += usize::from(changed);
    }
    let mut provenance_inputs = if provenance_ids.is_empty() {
        Vec::new()
    } else {
        provenance_input::Entity::find()
            .filter(provenance_input::Column::ProvenanceId.is_in(provenance_ids.clone()))
            .all(&txn)
            .await?
    };
    for row in &mut provenance_inputs {
        let rewritten = remap(row.input_symbol_id, &mapping);
        counts.provenance_inputs_rewritten += usize::from(rewritten != row.input_symbol_id);
        row.input_symbol_id = rewritten;
    }

    let mut provenance_snapshots = provenance_snapshot::Entity::find()
        .filter(provenance_snapshot::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    for row in &mut provenance_snapshots {
        let mut changed = false;
        let derived = remap(row.derived_symbol, &mapping);
        changed |= derived != row.derived_symbol;
        row.derived_symbol = derived;
        changed |= rewrite_symbol_array(&mut row.sources, &mapping);
        changed |= rewrite_symbol_json(&mut row.derivation, &mapping);
        counts.provenance_snapshots_rewritten += usize::from(changed);
    }

    let mut agenda_rows = agenda::Entity::find()
        .filter(agenda::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    agenda_rows.sort_by_key(|row| row.project_id);
    let agenda_input_len = agenda_rows.len();
    let mut agenda_map = BTreeMap::new();
    for mut row in agenda_rows {
        let rewritten = remap(row.project_id, &mapping);
        counts.agenda_rewritten += usize::from(rewritten != row.project_id);
        row.project_id = rewritten;
        agenda_map.entry(row.project_id).or_insert(row);
    }
    let agenda_rows: Vec<_> = agenda_map.into_values().collect();
    counts.agenda_collapsed = agenda_input_len - agenda_rows.len();

    let mut goals = goal::Entity::find()
        .filter(goal::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    goals.sort_by_key(|row| row.id);
    let goals_input_len = goals.len();
    let mut goal_map = BTreeMap::new();
    for mut row in goals {
        let mut changed = false;
        let id = remap(row.id, &mapping);
        changed |= id != row.id;
        row.id = id;
        if let Some(parent) = row.parent {
            let rewritten = remap(parent, &mapping);
            changed |= rewritten != parent;
            row.parent = Some(rewritten);
        }
        changed |= rewrite_goal_status(&mut row.status, &mapping);
        if let Some(detail) = &mut row.detail {
            changed |= rewrite_symbol_json(detail, &mapping);
        }
        counts.goals_rewritten += usize::from(changed);
        goal_map.entry(row.id).or_insert(row);
    }
    let goals: Vec<_> = goal_map.into_values().collect();
    counts.goals_collapsed = goals_input_len - goals.len();

    let mut hierarchy = goal_hierarchy::Entity::find()
        .filter(goal_hierarchy::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    hierarchy.sort_by_key(|row| (row.parent_id, row.child_id, row.position));
    let hierarchy_input_len = hierarchy.len();
    let mut hierarchy_map = BTreeMap::new();
    for mut row in hierarchy {
        let before = (row.parent_id, row.child_id);
        row.parent_id = remap(row.parent_id, &mapping);
        row.child_id = remap(row.child_id, &mapping);
        counts.goal_hierarchy_rewritten += usize::from(before != (row.parent_id, row.child_id));
        hierarchy_map
            .entry((row.parent_id, row.child_id))
            .or_insert(row);
    }
    let hierarchy: Vec<_> = hierarchy_map.into_values().collect();
    counts.goal_hierarchy_collapsed = hierarchy_input_len - hierarchy.len();

    let mut dependencies = goal_dependency::Entity::find()
        .filter(goal_dependency::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    dependencies.sort_by_key(|row| (row.goal_id, row.blocking_goal_id));
    let dependencies_input_len = dependencies.len();
    let mut dependency_map = BTreeMap::new();
    for mut row in dependencies {
        let before = (row.goal_id, row.blocking_goal_id);
        row.goal_id = remap(row.goal_id, &mapping);
        row.blocking_goal_id = remap(row.blocking_goal_id, &mapping);
        counts.goal_dependencies_rewritten +=
            usize::from(before != (row.goal_id, row.blocking_goal_id));
        dependency_map
            .entry((row.goal_id, row.blocking_goal_id))
            .or_insert(row);
    }
    let dependencies: Vec<_> = dependency_map.into_values().collect();
    counts.goal_dependencies_collapsed = dependencies_input_len - dependencies.len();

    let mut methods = learned_method::Entity::find()
        .filter(learned_method::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    methods.sort_by_key(|row| row.id);
    let methods_input_len = methods.len();
    let mut method_map = BTreeMap::new();
    for mut row in methods {
        let rewritten = remap(row.id, &mapping);
        counts.learned_methods_rewritten += usize::from(rewritten != row.id);
        row.id = rewritten;
        method_map.entry(row.id).or_insert(row);
    }
    let methods: Vec<_> = method_map.into_values().collect();
    counts.learned_methods_collapsed = methods_input_len - methods.len();

    let mut steps = learned_method_step::Entity::find()
        .filter(learned_method_step::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    steps.sort_by_key(|row| (row.method_id, row.step_index));
    let steps_input_len = steps.len();
    let mut step_map = BTreeMap::new();
    for mut row in steps {
        let rewritten = remap(row.method_id, &mapping);
        counts.learned_method_steps_rewritten += usize::from(rewritten != row.method_id);
        row.method_id = rewritten;
        step_map
            .entry((row.method_id, row.step_index))
            .or_insert(row);
    }
    let steps: Vec<_> = step_map.into_values().collect();
    counts.learned_method_steps_collapsed = steps_input_len - steps.len();

    let episodes = episode::Entity::find()
        .filter(episode::Column::AkhId.eq(akh_id))
        .all(&txn)
        .await?;
    let (episodes, episodes_rewritten, episodes_collapsed) =
        remap_and_collapse_episodes(episodes, &mapping);
    counts.episodes_rewritten = episodes_rewritten;
    counts.episodes_collapsed = episodes_collapsed;

    // Once all replacement images are ready, replace each affected table while
    // holding the same transaction and snapshot publication lock.
    fact::Entity::delete_many()
        .filter(fact::Column::WorkspaceId.eq(workspace_id))
        .exec(&txn)
        .await?;
    insert_chunks!(
        fact,
        &txn,
        facts,
        [
            fact::Column::WorkspaceId,
            fact::Column::Subject,
            fact::Column::Predicate,
            fact::Column::Object,
        ]
    );

    delete_akh_rows!(provenance, &txn, akh_id);
    insert_chunks!(
        provenance,
        &txn,
        provenance_rows,
        [provenance::Column::ProvenanceId]
    );
    if !provenance_ids.is_empty() {
        provenance_input::Entity::delete_many()
            .filter(provenance_input::Column::ProvenanceId.is_in(provenance_ids))
            .exec(&txn)
            .await?;
    }
    insert_chunks!(
        provenance_input,
        &txn,
        provenance_inputs,
        [
            provenance_input::Column::ProvenanceId,
            provenance_input::Column::Position,
        ]
    );
    delete_akh_rows!(provenance_snapshot, &txn, akh_id);
    insert_chunks!(
        provenance_snapshot,
        &txn,
        provenance_snapshots,
        [
            provenance_snapshot::Column::AkhId,
            provenance_snapshot::Column::ProvenanceId,
        ]
    );
    delete_akh_rows!(agenda, &txn, akh_id);
    insert_chunks!(
        agenda,
        &txn,
        agenda_rows,
        [agenda::Column::AkhId, agenda::Column::ProjectId]
    );
    delete_akh_rows!(goal, &txn, akh_id);
    insert_chunks!(goal, &txn, goals, [goal::Column::AkhId, goal::Column::Id]);
    delete_akh_rows!(goal_hierarchy, &txn, akh_id);
    insert_chunks!(
        goal_hierarchy,
        &txn,
        hierarchy,
        [
            goal_hierarchy::Column::AkhId,
            goal_hierarchy::Column::ParentId,
            goal_hierarchy::Column::ChildId,
        ]
    );
    delete_akh_rows!(goal_dependency, &txn, akh_id);
    insert_chunks!(
        goal_dependency,
        &txn,
        dependencies,
        [
            goal_dependency::Column::AkhId,
            goal_dependency::Column::GoalId,
            goal_dependency::Column::BlockingGoalId,
        ]
    );
    delete_akh_rows!(learned_method, &txn, akh_id);
    insert_chunks!(
        learned_method,
        &txn,
        methods,
        [learned_method::Column::AkhId, learned_method::Column::Id,]
    );
    delete_akh_rows!(learned_method_step, &txn, akh_id);
    insert_chunks!(
        learned_method_step,
        &txn,
        steps,
        [
            learned_method_step::Column::AkhId,
            learned_method_step::Column::MethodId,
            learned_method_step::Column::StepIndex,
        ]
    );
    delete_akh_rows!(episode, &txn, akh_id);
    insert_chunks!(episode, &txn, episodes, [episode::Column::EpisodeId]);

    let losers: BTreeSet<i64> = mapping.keys().copied().collect();
    let deleted = symbol::Entity::delete_many()
        .filter(symbol::Column::AkhId.eq(akh_id))
        .filter(symbol::Column::Id.is_in(losers))
        .exec(&txn)
        .await?;
    counts.symbols_deleted = deleted.rows_affected as usize;

    let now = chrono::Utc::now();
    for row in audit_rows {
        mutation_log::Entity::insert(mutation_log::ActiveModel {
            seq: NotSet,
            akh_id: Set(akh_id),
            workspace_id: Set(workspace_id),
            kind: Set(row.kind.clone()),
            payload: Set(row.payload.clone()),
            applied_at: Set(now),
        })
        .exec(&txn)
        .await?;
    }
    counts.wal_records = audit_rows.len();

    if allocator.advanced {
        akh_state::Entity::update(advance_allocator(state))
            .exec(&txn)
            .await?;
    }

    txn.commit().await?;
    Ok(SymbolRepairOutcome { counts, allocator })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact_row(workspace_id: Uuid, akh_id: Uuid, subject: i64, confidence: f32) -> fact::Model {
        fact::Model {
            workspace_id,
            subject,
            predicate: 3,
            object: 4,
            akh_id,
            confidence,
            asserted_at: chrono::Utc::now(),
            retracted_at: None,
            provenance_id: None,
            compartment: None,
        }
    }

    #[test]
    fn mapping_rejects_chains_and_noncanonical_direction() {
        assert!(
            mapping_from(&[SymbolMerge {
                canonical_id: 2,
                duplicate_id: 1
            }])
            .is_err()
        );
        assert!(
            mapping_from(&[
                SymbolMerge {
                    canonical_id: 1,
                    duplicate_id: 2
                },
                SymbolMerge {
                    canonical_id: 2,
                    duplicate_id: 3
                },
            ])
            .is_err()
        );
    }

    #[test]
    fn json_rewrite_is_field_aware() {
        let mapping = HashMap::from([(7, 2), (8, 3)]);
        let mut value = serde_json::json!({
            "symbol_id": 7,
            "created_at": 7,
            "source": {"ContradictionDetected": {
                "existing": {"subject": 7, "predicate": 3, "object": 4, "timestamp": 7}
            }},
            "sources": [7, 8, 9],
        });
        assert!(rewrite_symbol_json(&mut value, &mapping));
        assert_eq!(value["symbol_id"], 2);
        assert_eq!(value["created_at"], 7);
        assert_eq!(
            value["source"]["ContradictionDetected"]["existing"]["subject"],
            2
        );
        assert_eq!(
            value["source"]["ContradictionDetected"]["existing"]["timestamp"],
            7
        );
        assert_eq!(value["sources"], serde_json::json!([2, 3, 9]));

        let mut root_sources = serde_json::json!([7, 8, 9]);
        assert!(rewrite_symbol_array(&mut root_sources, &mapping));
        assert_eq!(root_sources, serde_json::json!([2, 3, 9]));

        let mut root_objects = serde_json::json!([
            {"symbol_id": 7},
            {"symbol_id": 8},
        ]);
        assert!(rewrite_symbol_json(&mut root_objects, &mapping));
        assert_eq!(
            root_objects,
            serde_json::json!([
                {"symbol_id": 2},
                {"symbol_id": 3},
            ])
        );
    }

    #[test]
    fn fact_collision_preserves_the_maximum_confidence_row() {
        let workspace_id = Uuid::new_v4();
        let low_confidence_akh = Uuid::new_v4();
        let high_confidence_akh = Uuid::new_v4();
        let (rows, rewritten, collapsed) = remap_and_collapse_facts(
            vec![
                fact_row(workspace_id, low_confidence_akh, 2, 0.25),
                fact_row(workspace_id, high_confidence_akh, 7, 0.9),
            ],
            &HashMap::from([(7, 2)]),
        );

        assert_eq!(rewritten, 1);
        assert_eq!(collapsed, 1);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].confidence, 0.9);
        assert_eq!(rows[0].akh_id, high_confidence_akh);
    }

    #[test]
    fn episode_id_collision_keeps_the_existing_canonical_row() {
        let akh_id = Uuid::new_v4();
        let workspace_id = Uuid::new_v4();
        let now = chrono::Utc::now();
        let episode = |episode_id, payload| episode::Model {
            episode_id,
            akh_id,
            workspace_id,
            started_at: now,
            ended_at: None,
            payload,
        };
        let (rows, rewritten, collapsed) = remap_and_collapse_episodes(
            vec![
                episode(2, serde_json::json!({"marker": "canonical"})),
                episode(7, serde_json::json!({"marker": "duplicate"})),
                episode(9, serde_json::json!({"symbol_id": 7})),
            ],
            &HashMap::from([(7, 2)]),
        );

        assert_eq!(rewritten, 2);
        assert_eq!(collapsed, 1);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].episode_id, 2);
        assert_eq!(rows[0].payload["marker"], "canonical");
        assert_eq!(rows[1].payload["symbol_id"], 2);
        assert!(rows.iter().all(|row| row.episode_id != 7));
    }
}
