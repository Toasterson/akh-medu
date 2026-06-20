//! Phase 38c: snapshot the akh's relational state to Postgres, and wake a fresh
//! engine from it.
//!
//! The akh's database is mechanical, typed, relational data (no opaque blobs).
//! `write_snapshot` materialises the live KG + registry + cognitive state into
//! the `akhomed_*` tables (via `akhomed_store`); `wake` (38c.3) rebuilds an
//! identical engine on a fresh host.

use miette::Diagnostic;
use thiserror::Error;

use akhomed_store::DatabaseConnection;
use akhomed_store::chrono;
use akhomed_store::uuid::Uuid;
use akhomed_store::entity::{
    agenda, akh_state, competence_calibration, competence_category, competence_tool,
    curiosity_config, drive, effort_case, fact, failure_case, goal, learned_method, sleep_metric,
    symbol,
};

use crate::agent::Agent;
use crate::symbol::SymbolKind;

/// Errors from the snapshot/wake path.
#[derive(Debug, Error, Diagnostic)]
pub enum SnapshotError {
    /// A backing-store operation failed.
    #[error(transparent)]
    #[diagnostic(transparent)]
    Store(#[from] akhomed_store::StoreError),

    /// Flushing agent state to local storage failed.
    #[error("agent persist failed: {0}")]
    #[diagnostic(code(akh::snapshot::persist))]
    Agent(String),

    /// An engine operation failed.
    #[error("engine error: {0}")]
    #[diagnostic(code(akh::snapshot::engine))]
    Engine(String),
}

/// Convenience result alias.
pub type SnapshotResult<T> = std::result::Result<T, SnapshotError>;

/// Snapshot `agent`'s current state into Postgres under `(akh_id, workspace_id)`.
///
/// Call with the agent locked (a consistent point — no concurrent OODA
/// mutations). The watermark is the max `mutation_log.seq` at snapshot time;
/// wake replays rows newer than it.
pub async fn write_snapshot(
    agent: &Agent,
    akh_id: Uuid,
    workspace_id: Uuid,
    db: &DatabaseConnection,
) -> SnapshotResult<()> {
    let engine = agent.engine();

    // 1. Flush durable state to local redb (registry, allocator counter, KG snapshot).
    agent.persist().map_err(|e| SnapshotError::Agent(e.to_string()))?;

    // 2. Recovery boundary + allocator counter.
    let watermark = akhomed_store::snapshots::current_watermark(db, akh_id).await?;
    let allocator_next = read_allocator_next(engine);

    // 3. Symbols (registry → rows).
    let symbols: Vec<symbol::Model> = engine
        .dump_symbols()
        .into_iter()
        .map(|m| {
            let (kind, glyph) = symbol_kind_parts(&m.kind);
            symbol::Model {
                akh_id,
                id: m.id.get() as i64,
                kind,
                glyph_codepoint: glyph,
                label: m.label,
            }
        })
        .collect();
    akhomed_store::snapshots::replace_symbols(db, akh_id, symbols).await?;

    // 4. Facts (live KG → rows; retracted triples are absent).
    let facts: Vec<fact::Model> = engine
        .all_triples()
        .into_iter()
        .map(|t| fact::Model {
            workspace_id,
            subject: t.subject.get() as i64,
            predicate: t.predicate.get() as i64,
            object: t.object.get() as i64,
            akh_id,
            confidence: t.confidence,
            asserted_at: ts(t.timestamp),
            retracted_at: None,
            provenance_id: t.provenance_id.map(|p| p as i64),
            compartment: t.compartment_id.clone(),
        })
        .collect();
    akhomed_store::snapshots::replace_workspace_facts(db, workspace_id, facts).await?;

    // 5. Cognitive tables.
    let drives: Vec<drive::Model> = agent
        .drives
        .drives
        .iter()
        .enumerate()
        .map(|(i, d)| drive::Model {
            akh_id,
            kind: d.kind.label().to_owned(),
            strength: d.strength,
            last_computed: d.last_computed as i64,
            last_triple_count: d.last_snapshot.triple_count as i64,
            last_symbol_count: d.last_snapshot.symbol_count as i64,
            threshold: agent.drives.thresholds[i],
        })
        .collect();
    akhomed_store::snapshots::replace_drives(db, akh_id, drives).await?;

    let active = agent.agenda.active_project;
    let agenda_rows: Vec<agenda::Model> = agent
        .agenda
        .projects
        .iter()
        .map(|(id, prio)| agenda::Model {
            akh_id,
            project_id: id.get() as i64,
            priority: *prio as i16,
            is_active: Some(*id) == active,
        })
        .collect();
    akhomed_store::snapshots::replace_agenda(db, akh_id, agenda_rows).await?;

    let c = &agent.curiosity_config;
    let curiosity = vec![curiosity_config::Model {
        akh_id,
        max_targets: c.max_targets as i32,
        proximal_low: c.proximal_low,
        proximal_high: c.proximal_high,
        peak_fill_ratio: c.peak_fill_ratio,
        beyond_curiosity_factor: c.beyond_curiosity_factor,
        known_curiosity_factor: c.known_curiosity_factor,
    }];
    akhomed_store::snapshots::replace_curiosity(db, akh_id, curiosity).await?;

    let goals: Vec<goal::Model> = agent.goals.iter().map(|g| goal_to_model(akh_id, g)).collect();
    akhomed_store::snapshots::replace_goals(db, akh_id, goals).await?;

    let failures: Vec<failure_case::Model> = agent
        .failure_index
        .cases()
        .iter()
        .enumerate()
        .map(|(i, fc)| failure_case::Model {
            akh_id,
            seq: i as i32,
            goal_description: fc.goal_description.clone(),
            tool_name: fc.tool_name.clone(),
            error_message: fc.error_message.clone(),
            cycle: fc.cycle as i64,
            resolved: fc.resolved,
            resolution_tool: fc.resolution_tool.clone(),
            vector: fc.vector.clone(),
        })
        .collect();
    akhomed_store::snapshots::replace_failure_cases(db, akh_id, failures).await?;

    let efforts: Vec<effort_case::Model> = agent
        .effort_index
        .cases()
        .iter()
        .enumerate()
        .map(|(i, ec)| effort_case::Model {
            akh_id,
            seq: i as i32,
            description: ec.description.clone(),
            // Vec<u32> → bytes (bincode) — rebuilt on wake.
            goal_vector: bincode::serialize(&ec.goal_vector).unwrap_or_default(),
            cycles_consumed: ec.cycles_consumed as i32,
            succeeded: ec.succeeded,
            tool_usage: serde_json::to_value(&ec.tool_usage).unwrap_or_default(),
            initial_coverage: ec.initial_coverage as i32,
        })
        .collect();
    akhomed_store::snapshots::replace_effort_cases(db, akh_id, efforts).await?;

    let methods: Vec<learned_method::Model> = agent
        .method_index
        .methods()
        .iter()
        .map(|m| learned_method::Model {
            akh_id,
            id: m.id.get() as i64,
            name: m.name.clone(),
            precondition_pattern: m.precondition_pattern.clone(),
            precondition_vector: m.precondition_vector.clone(),
            confidence: m.confidence,
            usage_count: m.usage_count as i32,
            success_count: m.success_count as i32,
            failure_count: m.failure_count as i32,
            learned_at: m.learned_at as i64,
            last_used: m.last_used as i64,
            source_goal_description: m.source_goal_description.clone(),
            steps: serde_json::to_value(&m.steps).unwrap_or_default(),
        })
        .collect();
    akhomed_store::snapshots::replace_learned_methods(db, akh_id, methods).await?;

    // Competence — 3 tables.
    let cm = &agent.competence_model;
    let tools: Vec<competence_tool::Model> = cm
        .tools
        .iter()
        .map(|(tool, r)| competence_tool::Model {
            akh_id,
            tool: tool.clone(),
            successes: r.successes as i32,
            attempts: r.attempts as i32,
        })
        .collect();
    akhomed_store::snapshots::replace_competence_tools(db, akh_id, tools).await?;
    let cats: Vec<competence_category::Model> = cm
        .categories
        .iter()
        .map(|(category, r)| competence_category::Model {
            akh_id,
            category: category.clone(),
            completed: r.completed as i32,
            attempted: r.attempted as i32,
        })
        .collect();
    akhomed_store::snapshots::replace_competence_categories(db, akh_id, cats).await?;
    let calib: Vec<competence_calibration::Model> = cm
        .calibration_history
        .iter()
        .enumerate()
        .map(|(i, p)| competence_calibration::Model {
            akh_id,
            seq: i as i32,
            predicted: p.predicted,
            actual: p.actual,
        })
        .collect();
    akhomed_store::snapshots::replace_competence_calibration(db, akh_id, calib).await?;

    // Sleep metrics (time series).
    let sleep_rows: Vec<sleep_metric::Model> = agent
        .sleep_cycle
        .metrics_history
        .iter()
        .map(|(cycle, m)| sleep_metric::Model {
            akh_id,
            cycle: *cycle as i64,
            episodes_replayed: m.episodes_replayed as i32,
            duplicates_merged: m.duplicates_merged as i32,
            orphans_pruned: m.orphans_pruned as i32,
            dream_connections_found: m.dream_connections_found as i32,
            equivalences_discovered: m.equivalences_discovered as i32,
            duration_cycles: m.duration_cycles as i64,
        })
        .collect();
    akhomed_store::snapshots::replace_sleep_metrics(db, akh_id, sleep_rows).await?;

    // 6. The runtime-state row.
    let psyche = agent
        .psyche
        .as_ref()
        .and_then(|p| serde_json::to_value(p).ok());
    let state = akh_state::Model {
        akh_id,
        watermark,
        allocator_next,
        cycle_count: agent.cycle_count as i64,
        session_start_cycle: agent.session_start_cycle as i64,
        sleep_phase: format!("{:?}", agent.sleep_cycle.phase),
        last_sleep_cycle: agent.sleep_cycle.last_sleep_cycle as i64,
        psyche,
        extra: None,
        updated_at: chrono::Utc::now(),
    };
    akhomed_store::snapshots::upsert_akh_state(db, state).await?;

    Ok(())
}

// --- helpers ---

/// Read the symbol allocator counter from the engine's local store (written by
/// `Engine::persist`, which `agent.persist()` triggers).
fn read_allocator_next(engine: &crate::engine::Engine) -> i64 {
    engine
        .store()
        .get_meta(b"sym_allocator_next")
        .ok()
        .flatten()
        .and_then(|b| bincode::deserialize::<u64>(&b).ok())
        .unwrap_or(1) as i64
}

/// UNIX-seconds → `DateTimeUtc`, defaulting to now on overflow.
fn ts(secs: u64) -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::from_timestamp(secs as i64, 0).unwrap_or_else(chrono::Utc::now)
}

/// Split a `SymbolKind` into a text discriminant + optional glyph codepoint.
fn symbol_kind_parts(kind: &SymbolKind) -> (String, Option<i32>) {
    match kind {
        SymbolKind::Entity => ("Entity".into(), None),
        SymbolKind::Relation => ("Relation".into(), None),
        SymbolKind::Composite => ("Composite".into(), None),
        SymbolKind::Glyph { codepoint } => ("Glyph".into(), Some(*codepoint as i32)),
    }
}

/// Variant name of a serialised enum value (unit → the string; data-bearing →
/// the tag key).
fn json_variant(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(m) => m.keys().next().cloned(),
        _ => None,
    }
}

fn goal_to_model(akh_id: Uuid, g: &crate::agent::Goal) -> goal::Model {
    let status = serde_json::to_value(&g.status)
        .ok()
        .and_then(|v| json_variant(&v))
        .unwrap_or_else(|| format!("{:?}", g.status));
    let source = g
        .source
        .as_ref()
        .and_then(|s| serde_json::to_value(s).ok())
        .and_then(|v| json_variant(&v));
    let detail = serde_json::json!({
        "source": g.source,
        "priority_rationale": g.priority_rationale,
        "justification": g.justification,
        "reformulated_from": g.reformulated_from.map(|s| s.get()),
        "estimated_effort": g.estimated_effort,
    });
    goal::Model {
        akh_id,
        id: g.symbol_id.get() as i64,
        description: g.description.clone(),
        status,
        priority: g.priority as i16,
        success_criteria: g.success_criteria.clone(),
        parent: g.parent.map(|s| s.get() as i64),
        created_at: g.created_at as i64,
        cycles_worked: g.cycles_worked as i32,
        last_progress_cycle: g.last_progress_cycle as i64,
        source,
        children: serde_json::to_value(g.children.iter().map(|s| s.get()).collect::<Vec<_>>())
            .unwrap_or_default(),
        blocked_by: serde_json::to_value(g.blocked_by.iter().map(|s| s.get()).collect::<Vec<_>>())
            .unwrap_or_default(),
        detail: Some(detail),
    }
}
