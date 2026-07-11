//! Phase 38c: snapshot the akh's relational state to Postgres, and wake a fresh
//! engine from it.
//!
//! The akh's database is mechanical, typed, relational data (no opaque blobs).
//! `write_snapshot` materialises the live KG + registry + cognitive state into
//! the `akhomed_*` tables (via `akhomed_store`); `wake` (38c.3) rebuilds an
//! identical engine on a fresh host.

use std::path::Path;
use std::sync::Arc;

use miette::Diagnostic;
use thiserror::Error;

use akhomed_store::DatabaseConnection;
use akhomed_store::chrono;
use akhomed_store::uuid::Uuid;
use akhomed_store::entity::{
    agenda, akh_state, competence_calibration, competence_category, competence_tool,
    curiosity_config, drive, effort_case, fact, failure_case, goal, goal_dependency, goal_hierarchy,
    learned_method, learned_method_step, sleep_metric, symbol,
};

use crate::agent::Agent;
use crate::engine::{Engine, EngineConfig};
use crate::graph::Triple;
use crate::registry::SymbolRegistry;
use crate::store::TieredStore;
use crate::symbol::{SymbolId, SymbolKind, SymbolMeta};

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

    /// No snapshot exists for the requested Akh — nothing to wake from.
    #[error("no snapshot found for akh {0}")]
    #[diagnostic(
        code(akh::snapshot::not_found),
        help("take a snapshot first, or boot the Akh fresh instead of waking it")
    )]
    NotFound(Uuid),

    /// Opening the local redb store at the wake directory failed.
    #[error("failed to open local store at {path}: {source}")]
    #[diagnostic(code(akh::snapshot::store_open))]
    StoreOpen {
        /// The directory we tried to open.
        path: String,
        /// The underlying store error.
        #[source]
        source: crate::error::StoreError,
    },
}

/// Convenience result alias.
pub type SnapshotResult<T> = std::result::Result<T, SnapshotError>;

/// An owned, lock-free capture of an Akh's state, ready to persist.
///
/// Produced synchronously by [`capture_snapshot`] (holding the agent lock only
/// for the fast in-memory copy) and consumed by the async [`persist_snapshot`]
/// (the DB I/O, which must run *without* the lock — the daemon's agent lives
/// behind a `std::sync::Mutex` whose guard cannot cross an `.await`).
pub struct SnapshotData {
    akh_id: Uuid,
    workspace_id: Uuid,
    symbols: Vec<symbol::Model>,
    facts: Vec<fact::Model>,
    drives: Vec<drive::Model>,
    agenda: Vec<agenda::Model>,
    curiosity: Vec<curiosity_config::Model>,
    goals: Vec<goal::Model>,
    goal_hierarchy: Vec<goal_hierarchy::Model>,
    goal_dependencies: Vec<goal_dependency::Model>,
    failures: Vec<failure_case::Model>,
    efforts: Vec<effort_case::Model>,
    methods: Vec<learned_method::Model>,
    method_steps: Vec<learned_method_step::Model>,
    comp_tools: Vec<competence_tool::Model>,
    comp_cats: Vec<competence_category::Model>,
    comp_calib: Vec<competence_calibration::Model>,
    sleep_rows: Vec<sleep_metric::Model>,
    allocator_next: i64,
    cycle_count: i64,
    session_start_cycle: i64,
    sleep_phase: String,
    last_sleep_cycle: i64,
    psyche: Option<serde_json::Value>,
    extra: Option<serde_json::Value>,
}

/// Snapshot `agent`'s current state into Postgres under `(akh_id, workspace_id)`.
///
/// Convenience wrapper = [`capture_snapshot`] + [`persist_snapshot`]. Callers
/// that hold the agent behind a non-async mutex must instead call the two halves
/// separately: capture under the lock, drop the guard, then `await` the persist.
pub async fn write_snapshot(
    agent: &Agent,
    akh_id: Uuid,
    workspace_id: Uuid,
    db: &DatabaseConnection,
) -> SnapshotResult<()> {
    let data = capture_snapshot(agent, akh_id, workspace_id)?;
    persist_snapshot(data, db).await
}

/// Capture the agent's live state into owned rows (synchronous).
///
/// Flushes durable state to local redb, then copies the registry, KG, and
/// cognitive subsystems into entity `Model`s. Holds no DB handle and never
/// `await`s, so it is safe to call while holding a `std::sync::Mutex<Agent>`
/// guard; release the guard before [`persist_snapshot`].
pub fn capture_snapshot(
    agent: &Agent,
    akh_id: Uuid,
    workspace_id: Uuid,
) -> SnapshotResult<SnapshotData> {
    let engine = agent.engine();

    // Flush durable state to local redb (registry, allocator counter, KG snapshot).
    agent.persist().map_err(|e| SnapshotError::Agent(e.to_string()))?;

    let allocator_next = read_allocator_next(engine);

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

    // Live KG → rows; retracted triples are absent.
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

    let active = agent.agenda.active_project;
    let agenda: Vec<agenda::Model> = agent
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

    let goals: Vec<goal::Model> = agent.goals.iter().map(|g| goal_to_model(akh_id, g)).collect();
    // Normalised goal DAG: `children` → ordered hierarchy edges; `blocked_by` →
    // dependency edges.
    let goal_hierarchy: Vec<goal_hierarchy::Model> = agent
        .goals
        .iter()
        .flat_map(|g| {
            let parent_id = g.symbol_id.get() as i64;
            g.children.iter().enumerate().map(move |(pos, child)| goal_hierarchy::Model {
                akh_id,
                parent_id,
                child_id: child.get() as i64,
                position: pos as i32,
            })
        })
        .collect();
    let goal_dependencies: Vec<goal_dependency::Model> = agent
        .goals
        .iter()
        .flat_map(|g| {
            let goal_id = g.symbol_id.get() as i64;
            g.blocked_by.iter().map(move |b| goal_dependency::Model {
                akh_id,
                goal_id,
                blocking_goal_id: b.get() as i64,
            })
        })
        .collect();

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
        })
        .collect();
    // Normalised method steps: one ordered child row per `GeneralizedStep`.
    let method_steps: Vec<learned_method_step::Model> = agent
        .method_index
        .methods()
        .iter()
        .flat_map(|m| {
            let method_id = m.id.get() as i64;
            m.steps.iter().enumerate().map(move |(idx, step)| learned_method_step::Model {
                akh_id,
                method_id,
                step_index: idx as i32,
                tool: step.tool.clone(),
                arg_pattern: step.arg_pattern.clone(),
                expected_outcome: step.expected_outcome.clone(),
            })
        })
        .collect();

    // Competence — 3 tables.
    let cm = &agent.competence_model;
    let comp_tools: Vec<competence_tool::Model> = cm
        .tools
        .iter()
        .map(|(tool, r)| competence_tool::Model {
            akh_id,
            tool: tool.clone(),
            successes: r.successes as i32,
            attempts: r.attempts as i32,
        })
        .collect();
    let comp_cats: Vec<competence_category::Model> = cm
        .categories
        .iter()
        .map(|(category, r)| competence_category::Model {
            akh_id,
            category: category.clone(),
            completed: r.completed as i32,
            attempted: r.attempted as i32,
        })
        .collect();
    let comp_calib: Vec<competence_calibration::Model> = cm
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

    // Runtime-state row (sans watermark — that is read async in persist).
    let psyche = agent
        .psyche
        .as_ref()
        .and_then(|p| serde_json::to_value(p).ok());
    // `extra` carries the small, scalar-ish state that has no inspectable table
    // of its own but is cheap to round-trip: the active audience, the per-goal
    // improvement history, the HTN method-usage stats, the cross-session
    // continuity summary, and the person-memory index. Each is restored to its
    // `agent:*` meta key on wake.
    //
    // `person_memory` (Phase 25c `PersonMemoryIndex`) rides here as a JSON-safe
    // `PortablePersonMemory` projection — its `topic_index` tuple key can't be a
    // JSON object key, and a KG rebuild can't recover the topic→episode mapping,
    // so the flattened projection is the lossless path.
    let active_project_name = agent
        .agenda
        .active_project
        .and_then(|pid| agent.projects.iter().find(|p| p.id == pid))
        .map(|p| p.name.clone());
    let session_summary = crate::agent::memory::generate_session_summary(
        &agent.working_memory,
        &agent.goals,
        active_project_name.as_deref(),
        agent.session_start_cycle,
        agent.cycle_count,
    );
    let extra = serde_json::json!({
        "audience": agent.audience,
        "improvement_history": agent.improvement_history,
        "method_stats": agent.method_registry_htn.export_stats(),
        "session_summary": session_summary,
        "person_memory": agent.person_memory.to_portable(),
    });

    Ok(SnapshotData {
        akh_id,
        workspace_id,
        symbols,
        facts,
        drives,
        agenda,
        curiosity,
        goals,
        goal_hierarchy,
        goal_dependencies,
        failures,
        efforts,
        methods,
        method_steps,
        comp_tools,
        comp_cats,
        comp_calib,
        sleep_rows,
        allocator_next,
        cycle_count: agent.cycle_count as i64,
        session_start_cycle: agent.session_start_cycle as i64,
        sleep_phase: format!("{:?}", agent.sleep_cycle.phase),
        last_sleep_cycle: agent.sleep_cycle.last_sleep_cycle as i64,
        psyche,
        extra: Some(extra),
    })
}

/// Persist a previously-[captured](capture_snapshot) snapshot to Postgres
/// (async; holds no agent lock). Cognitive tables are full-replaced; `akh_state`
/// is upserted with the recovery watermark (`MAX(mutation_log.seq)` now).
pub async fn persist_snapshot(data: SnapshotData, db: &DatabaseConnection) -> SnapshotResult<()> {
    use akhomed_store::snapshots as ops;
    let SnapshotData {
        akh_id,
        workspace_id,
        symbols,
        facts,
        drives,
        agenda,
        curiosity,
        goals,
        goal_hierarchy,
        goal_dependencies,
        failures,
        efforts,
        methods,
        method_steps,
        comp_tools,
        comp_cats,
        comp_calib,
        sleep_rows,
        allocator_next,
        cycle_count,
        session_start_cycle,
        sleep_phase,
        last_sleep_cycle,
        psyche,
        extra,
    } = data;

    let watermark = ops::current_watermark(db, akh_id).await?;
    // ADR 091: `harvest_watermark` is excluded from `upsert_akh_state`'s
    // `update_columns`, so this value only matters the first time a row is
    // inserted for this Akh. Reading it back here (0 if no row exists yet)
    // keeps that first-insert value correct without ever clobbering a value
    // the dedicated `set_harvest_watermark` setter already wrote.
    let harvest_watermark = ops::get_harvest_watermark(db, akh_id).await?;

    ops::replace_symbols(db, akh_id, symbols).await?;
    ops::replace_workspace_facts(db, workspace_id, facts).await?;
    ops::replace_drives(db, akh_id, drives).await?;
    ops::replace_agenda(db, akh_id, agenda).await?;
    ops::replace_curiosity(db, akh_id, curiosity).await?;
    ops::replace_goals(db, akh_id, goals).await?;
    ops::replace_goal_hierarchy(db, akh_id, goal_hierarchy).await?;
    ops::replace_goal_dependencies(db, akh_id, goal_dependencies).await?;
    ops::replace_failure_cases(db, akh_id, failures).await?;
    ops::replace_effort_cases(db, akh_id, efforts).await?;
    ops::replace_learned_methods(db, akh_id, methods).await?;
    ops::replace_learned_method_steps(db, akh_id, method_steps).await?;
    ops::replace_competence_tools(db, akh_id, comp_tools).await?;
    ops::replace_competence_categories(db, akh_id, comp_cats).await?;
    ops::replace_competence_calibration(db, akh_id, comp_calib).await?;
    ops::replace_sleep_metrics(db, akh_id, sleep_rows).await?;

    let state = akh_state::Model {
        akh_id,
        watermark,
        allocator_next,
        cycle_count,
        session_start_cycle,
        sleep_phase,
        last_sleep_cycle,
        psyche,
        extra,
        harvest_watermark,
        updated_at: chrono::Utc::now(),
    };
    ops::upsert_akh_state(db, state).await?;

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
    // The flat columns above are for human/SQL inspection; `detail` carries the
    // full, lossless `Goal` (including data-bearing status variants like
    // `Failed { .. }` and nested optionals) so wake can reconstruct it exactly.
    let detail = serde_json::to_value(g).ok();
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
        detail,
    }
}

// ===========================================================================
// Wake (38c.3)
// ===========================================================================

/// Wake an Akh on a (possibly fresh) host from its Postgres snapshot.
///
/// Reconstructs the local redb meta the engine + agent expect from the
/// relational tables, opens a fresh [`Engine`] over `data_dir`, restores the KG
/// (+ item memory) from `facts`, then replays the mutation-log tail newer than
/// the snapshot watermark. The caller resumes the agent with
/// [`Agent::resume`](crate::agent::Agent::resume) and wires the WAL afterwards —
/// replay here runs **before** any sink is installed, so it is not re-logged.
///
/// A missing optional cognitive table degrades gracefully: its `agent:*` meta is
/// simply not written and `Agent::resume` falls back to defaults / KG rebuild.
/// The only hard requirement is the `akh_state` row ([`SnapshotError::NotFound`]
/// otherwise).
///
/// After the KG restore + WAL replay, also runs schema-pack lazy convergence
/// (ADR 081 decision 9 — [`crate::schema::registry_pg::converge_workspace`]):
/// installs every schema pack recorded active for `workspace_id`, idempotently.
pub async fn wake(
    data_dir: &Path,
    akh_id: Uuid,
    workspace_id: Uuid,
    db: &DatabaseConnection,
) -> SnapshotResult<Arc<Engine>> {
    use akhomed_store::snapshots as ops;

    // 1. The runtime-state row is mandatory — without it there is nothing to wake.
    let state = ops::load_akh_state(db, akh_id)
        .await?
        .ok_or(SnapshotError::NotFound(akh_id))?;

    // 2. Load the relational truth. Each is best-effort: a missing/empty table
    //    yields defaults downstream.
    let symbol_rows = ops::load_symbols(db, akh_id).await?;
    let fact_rows = ops::load_workspace_facts(db, workspace_id).await?;
    let drive_rows = ops::load_drives(db, akh_id).await?;
    let agenda_rows = ops::load_agenda(db, akh_id).await?;
    let curiosity_rows = ops::load_curiosity(db, akh_id).await?;
    let goal_rows = ops::load_goals(db, akh_id).await?;
    let hierarchy_rows = ops::load_goal_hierarchy(db, akh_id).await?;
    let dependency_rows = ops::load_goal_dependencies(db, akh_id).await?;
    let failure_rows = ops::load_failure_cases(db, akh_id).await?;
    let effort_rows = ops::load_effort_cases(db, akh_id).await?;
    let method_rows = ops::load_learned_methods(db, akh_id).await?;
    let method_step_rows = ops::load_learned_method_steps(db, akh_id).await?;
    let tool_rows = ops::load_competence_tools(db, akh_id).await?;
    let category_rows = ops::load_competence_categories(db, akh_id).await?;
    let calib_rows = ops::load_competence_calibration(db, akh_id).await?;
    let sleep_rows = ops::load_sleep_metrics(db, akh_id).await?;

    // 3. Reconstruct the local redb meta the engine + agent read on open. The
    //    store must be dropped before `Engine::new` reopens the same redb file.
    {
        let store = TieredStore::with_persistence(data_dir, "symbols").map_err(|source| {
            SnapshotError::StoreOpen {
                path: data_dir.display().to_string(),
                source,
            }
        })?;

        write_registry(&store, &symbol_rows);
        put_bincode(&store, b"sym_allocator_next", &(state.allocator_next as u64));
        put_bincode(&store, b"agent:cycle_count", &(state.cycle_count as u64));

        if let Some(d) = build_drive_system(&drive_rows) {
            put_bincode(&store, b"agent:drive_system", &d);
        }
        put_bincode(&store, b"agent:agenda", &build_agenda(&agenda_rows));
        if let Some(c) = build_curiosity(&curiosity_rows) {
            put_bincode(&store, b"agent:curiosity_config", &c);
        }
        let goals = build_goals(&goal_rows, &hierarchy_rows, &dependency_rows);
        if !goals.is_empty() {
            put_bincode(&store, b"agent:goals", &goals);
        }
        put_bincode(
            &store,
            b"agent:failure_cases",
            &build_failure_cases(&failure_rows),
        );
        if let Some(e) = build_effort_cases(&effort_rows) {
            put_bincode(&store, b"agent:effort_cases", &e);
        }
        if let Some(m) = build_learned_methods(&method_rows, &method_step_rows) {
            put_bincode(&store, b"agent:learned_methods", &m);
        }
        put_bincode(
            &store,
            b"agent:competence_model",
            &build_competence(&tool_rows, &category_rows, &calib_rows),
        );
        put_bincode(
            &store,
            b"agent:sleep_cycle",
            &build_sleep_cycle(&state, &sleep_rows),
        );

        // Psyche + the `extra` misc state.
        if let Some(psyche) = state.psyche.as_ref().and_then(|v| {
            serde_json::from_value::<crate::compartment::psyche::Psyche>(v.clone()).ok()
        }) {
            put_bincode(&store, b"agent:psyche", &psyche);
        }
        restore_extra(&store, state.extra.as_ref());
    }

    // 4. Open the engine over the reconstructed redb (registry + allocator come
    //    back here; the KG is still empty).
    let engine = Arc::new(
        Engine::new(EngineConfig {
            data_dir: Some(data_dir.to_path_buf()),
            ..Default::default()
        })
        .map_err(|e| SnapshotError::Engine(e.to_string()))?,
    );

    // 5. Restore the KG (+ item memory) from the relational facts.
    let triples: Vec<Triple> = fact_rows.iter().filter_map(fact_to_triple).collect();
    engine
        .restore_triples(&triples)
        .map_err(|e| SnapshotError::Engine(e.to_string()))?;

    // 6. Replay the WAL tail newer than the snapshot watermark. No sink is
    //    installed yet, so these mutations are applied but not re-logged.
    let tail = ops::load_mutation_tail(db, akh_id, state.watermark).await?;
    let mut replayed = 0usize;
    for m in &tail {
        if apply_mutation(&engine, m) {
            replayed += 1;
        }
    }
    tracing::info!(
        akh = %akh_id,
        triples = triples.len(),
        symbols = symbol_rows.len(),
        tail = tail.len(),
        replayed,
        "woke akh from postgres snapshot"
    );

    // 7. Lazy schema-pack convergence (ADR 081 decision 9): install every
    //    schema pack recorded as active for this workspace. Idempotent —
    //    reinstalling an already-installed pack@version is a no-op
    //    (`schema::install_pack`) — and always safe, because schema-pack
    //    types are immutable+additive. Best-effort: a failure here is logged
    //    and does not fail the wake (the akh still comes up; it just may be
    //    missing a pack's types until the next wake or an explicit install).
    let packs_converged =
        crate::schema::registry_pg::converge_workspace(&engine, db, workspace_id).await;
    if packs_converged > 0 {
        tracing::info!(
            akh = %akh_id,
            %workspace_id,
            packs_converged,
            "wake: converged active schema packs"
        );
    }

    // 8. Flush the reconstructed KG snapshot back to local redb so a plain
    //    restart (without Postgres) recovers too.
    engine
        .persist()
        .map_err(|e| SnapshotError::Engine(e.to_string()))?;

    Ok(engine)
}

// --- wake helpers ---

/// bincode-encode `value` and write it under `key` (best-effort: a failure to
/// encode or persist a single optional meta value is logged, not fatal).
fn put_bincode<T: serde::Serialize>(store: &TieredStore, key: &[u8], value: &T) {
    match bincode::serialize(value) {
        Ok(bytes) => {
            if let Err(e) = store.put_meta(key, &bytes) {
                tracing::warn!(key = %String::from_utf8_lossy(key), error = %e, "wake: meta write failed");
            }
        }
        Err(e) => {
            tracing::warn!(key = %String::from_utf8_lossy(key), error = %e, "wake: meta encode failed");
        }
    }
}

/// Rebuild the symbol registry from `symbols` rows and persist it as `sym_meta:`
/// entries — the format [`Engine::new`] reads on open.
fn write_registry(store: &TieredStore, rows: &[symbol::Model]) {
    let registry = SymbolRegistry::new();
    for row in rows {
        let Some(id) = SymbolId::new(row.id as u64) else {
            continue;
        };
        let kind = symbol_kind_from(&row.kind, row.glyph_codepoint);
        if let Err(e) = registry.register(SymbolMeta::new(id, kind, row.label.clone())) {
            tracing::warn!(label = %row.label, error = %e, "wake: skipping duplicate symbol");
        }
    }
    if let Err(e) = registry.persist(store) {
        tracing::warn!(error = %e, "wake: symbol registry persist failed");
    }
}

/// Inverse of [`symbol_kind_parts`].
fn symbol_kind_from(kind: &str, glyph: Option<i32>) -> SymbolKind {
    match kind {
        "Relation" => SymbolKind::Relation,
        "Composite" => SymbolKind::Composite,
        "Glyph" => SymbolKind::Glyph {
            codepoint: char::from_u32(glyph.unwrap_or(0) as u32).unwrap_or('\u{FFFD}'),
        },
        _ => SymbolKind::Entity,
    }
}

fn drive_kind_from(label: &str) -> Option<crate::agent::drives::DriveKind> {
    use crate::agent::drives::DriveKind;
    Some(match label {
        "curiosity" => DriveKind::Curiosity,
        "coherence" => DriveKind::Coherence,
        "completeness" => DriveKind::Completeness,
        "efficiency" => DriveKind::Efficiency,
        _ => return None,
    })
}

fn build_drive_system(rows: &[drive::Model]) -> Option<crate::agent::drives::DriveSystem> {
    use crate::agent::drives::{Drive, DriveSnapshot, DriveSystem};
    if rows.is_empty() {
        return None;
    }
    let mut system = DriveSystem::new();
    for row in rows {
        let Some(kind) = drive_kind_from(&row.kind) else {
            continue;
        };
        let idx = kind as usize;
        system.drives[idx] = Drive {
            kind,
            strength: row.strength,
            last_computed: row.last_computed as u64,
            last_snapshot: DriveSnapshot {
                triple_count: row.last_triple_count as usize,
                symbol_count: row.last_symbol_count as usize,
            },
        };
        system.thresholds[idx] = row.threshold;
    }
    Some(system)
}

fn build_agenda(rows: &[agenda::Model]) -> crate::agent::project::Agenda {
    let mut agenda = crate::agent::project::Agenda::default();
    for row in rows {
        let Some(id) = SymbolId::new(row.project_id as u64) else {
            continue;
        };
        agenda.projects.push((id, row.priority as u8));
        if row.is_active {
            agenda.active_project = Some(id);
        }
    }
    agenda
}

fn build_curiosity(
    rows: &[curiosity_config::Model],
) -> Option<crate::agent::curiosity::DirectedCuriosityConfig> {
    let row = rows.first()?;
    Some(crate::agent::curiosity::DirectedCuriosityConfig {
        max_targets: row.max_targets as usize,
        proximal_low: row.proximal_low,
        proximal_high: row.proximal_high,
        peak_fill_ratio: row.peak_fill_ratio,
        beyond_curiosity_factor: row.beyond_curiosity_factor,
        known_curiosity_factor: row.known_curiosity_factor,
    })
}

/// Reconstruct goals from the lossless `detail` JSON, then overlay the
/// normalised goal DAG (`children` from [`goal_hierarchy`], `blocked_by` from
/// [`goal_dependency`]) which is the authoritative source of those edges.
///
/// The flat columns are for inspection only and drop data-bearing status
/// variants, so the rich scalar/optional state still rides in `detail`; the two
/// list fields are reconstructed from the edge tables instead.
fn build_goals(
    rows: &[goal::Model],
    hierarchy: &[goal_hierarchy::Model],
    dependencies: &[goal_dependency::Model],
) -> Vec<crate::agent::Goal> {
    use std::collections::HashMap;

    // parent_id → [(position, child_id)] so children keep their live ordering.
    let mut children_by_parent: HashMap<i64, Vec<(i32, i64)>> = HashMap::new();
    for edge in hierarchy {
        children_by_parent
            .entry(edge.parent_id)
            .or_default()
            .push((edge.position, edge.child_id));
    }
    for v in children_by_parent.values_mut() {
        v.sort_by_key(|(pos, _)| *pos);
    }

    // goal_id → [blocking_goal_id].
    let mut blocked_by_goal: HashMap<i64, Vec<i64>> = HashMap::new();
    for edge in dependencies {
        blocked_by_goal
            .entry(edge.goal_id)
            .or_default()
            .push(edge.blocking_goal_id);
    }

    rows.iter()
        .filter_map(|row| {
            let mut goal = row
                .detail
                .as_ref()
                .and_then(|v| serde_json::from_value::<crate::agent::Goal>(v.clone()).ok())?;
            goal.children = children_by_parent
                .get(&row.id)
                .map(|v| v.iter().filter_map(|(_, c)| SymbolId::new(*c as u64)).collect())
                .unwrap_or_default();
            goal.blocked_by = blocked_by_goal
                .get(&row.id)
                .map(|v| v.iter().filter_map(|b| SymbolId::new(*b as u64)).collect())
                .unwrap_or_default();
            Some(goal)
        })
        .collect()
}

fn build_failure_cases(rows: &[failure_case::Model]) -> Vec<crate::agent::metacognition::FailureCase> {
    use crate::agent::metacognition::FailureCase;
    let mut rows: Vec<_> = rows.iter().collect();
    rows.sort_by_key(|r| r.seq);
    rows.into_iter()
        .map(|r| FailureCase {
            goal_description: r.goal_description.clone(),
            tool_name: r.tool_name.clone(),
            error_message: r.error_message.clone(),
            cycle: r.cycle as u64,
            resolved: r.resolved,
            resolution_tool: r.resolution_tool.clone(),
            vector: r.vector.clone(),
        })
        .collect()
}

fn build_effort_cases(rows: &[effort_case::Model]) -> Option<Vec<crate::agent::resource::EffortCase>> {
    use crate::agent::resource::EffortCase;
    if rows.is_empty() {
        return None;
    }
    let mut rows: Vec<_> = rows.iter().collect();
    rows.sort_by_key(|r| r.seq);
    Some(
        rows.into_iter()
            .map(|r| EffortCase {
                description: r.description.clone(),
                goal_vector: bincode::deserialize(&r.goal_vector).unwrap_or_default(),
                cycles_consumed: r.cycles_consumed as u32,
                succeeded: r.succeeded,
                tool_usage: serde_json::from_value(r.tool_usage.clone()).unwrap_or_default(),
                initial_coverage: r.initial_coverage as u32,
            })
            .collect(),
    )
}

fn build_learned_methods(
    rows: &[learned_method::Model],
    step_rows: &[learned_method_step::Model],
) -> Option<Vec<crate::agent::chunking::LearnedMethod>> {
    use crate::agent::chunking::{GeneralizedStep, LearnedMethod};
    use std::collections::HashMap;
    if rows.is_empty() {
        return None;
    }

    // method_id → ordered [GeneralizedStep], reconstructed from the child rows.
    let mut steps_by_method: HashMap<i64, Vec<(i32, GeneralizedStep)>> = HashMap::new();
    for s in step_rows {
        steps_by_method.entry(s.method_id).or_default().push((
            s.step_index,
            GeneralizedStep {
                tool: s.tool.clone(),
                arg_pattern: s.arg_pattern.clone(),
                expected_outcome: s.expected_outcome.clone(),
            },
        ));
    }
    for v in steps_by_method.values_mut() {
        v.sort_by_key(|(idx, _)| *idx);
    }

    Some(
        rows.iter()
            .filter_map(|r| {
                let id = SymbolId::new(r.id as u64)?;
                let steps = steps_by_method
                    .get(&r.id)
                    .map(|v| v.iter().map(|(_, s)| s.clone()).collect())
                    .unwrap_or_default();
                Some(LearnedMethod {
                    id,
                    name: r.name.clone(),
                    precondition_vector: r.precondition_vector.clone(),
                    precondition_pattern: r.precondition_pattern.clone(),
                    steps,
                    confidence: r.confidence,
                    usage_count: r.usage_count as u32,
                    success_count: r.success_count as u32,
                    failure_count: r.failure_count as u32,
                    learned_at: r.learned_at as u64,
                    last_used: r.last_used as u64,
                    source_goal_description: r.source_goal_description.clone(),
                })
            })
            .collect(),
    )
}

fn build_competence(
    tools: &[competence_tool::Model],
    categories: &[competence_category::Model],
    calib: &[competence_calibration::Model],
) -> crate::agent::metacognition::CompetenceModel {
    use crate::agent::metacognition::{CalibrationPoint, CategoryRecord, CompetenceModel, ToolRecord};
    let mut model = CompetenceModel::default();
    for t in tools {
        model.tools.insert(
            t.tool.clone(),
            ToolRecord {
                successes: t.successes as u32,
                attempts: t.attempts as u32,
            },
        );
    }
    for c in categories {
        model.categories.insert(
            c.category.clone(),
            CategoryRecord {
                completed: c.completed as u32,
                attempted: c.attempted as u32,
            },
        );
    }
    let mut calib: Vec<_> = calib.iter().collect();
    calib.sort_by_key(|c| c.seq);
    for c in calib {
        model.calibration_history.push(CalibrationPoint {
            predicted: c.predicted,
            actual: c.actual,
        });
    }
    model
}

fn build_sleep_cycle(
    state: &akh_state::Model,
    rows: &[sleep_metric::Model],
) -> crate::agent::sleep::SleepCycle {
    use crate::agent::sleep::{SleepConfig, SleepCycle, SleepMetrics};
    // SleepConfig is `[config]`-driven, not per-Akh state (see ADR 049 / plan);
    // start from defaults and overlay the persisted phase + metrics history.
    let mut cycle = SleepCycle::new(SleepConfig::default());
    cycle.phase = parse_consolidation_phase(&state.sleep_phase);
    cycle.last_sleep_cycle = state.last_sleep_cycle as u64;
    let mut rows: Vec<_> = rows.iter().collect();
    rows.sort_by_key(|r| r.cycle);
    cycle.metrics_history = rows
        .into_iter()
        .map(|r| {
            (
                r.cycle as u64,
                SleepMetrics {
                    episodes_replayed: r.episodes_replayed as usize,
                    duplicates_merged: r.duplicates_merged as usize,
                    orphans_pruned: r.orphans_pruned as usize,
                    dream_connections_found: r.dream_connections_found as usize,
                    equivalences_discovered: r.equivalences_discovered as usize,
                    duration_cycles: r.duration_cycles as u64,
                },
            )
        })
        .collect();
    cycle
}

/// Parse a `ConsolidationPhase` from its `Debug` rendering (how `write_snapshot`
/// stores it). Unknown / absent → `Wake`.
fn parse_consolidation_phase(s: &str) -> crate::agent::sleep::ConsolidationPhase {
    use crate::agent::sleep::ConsolidationPhase;
    match s {
        "Consolidation" => ConsolidationPhase::Consolidation,
        "Reorganization" => ConsolidationPhase::Reorganization,
        "DreamState" => ConsolidationPhase::DreamState,
        _ => ConsolidationPhase::Wake,
    }
}

/// Restore the `akh_state.extra` JSONB bag into its individual `agent:*` meta keys.
fn restore_extra(store: &TieredStore, extra: Option<&serde_json::Value>) {
    let Some(extra) = extra else {
        return;
    };
    if let Some(a) = extra.get("audience").and_then(|v| {
        serde_json::from_value::<crate::agent::priority_reasoning::Audience>(v.clone()).ok()
    }) {
        put_bincode(store, b"agent:audience", &a);
    }
    if let Some(h) = extra.get("improvement_history").and_then(|v| {
        serde_json::from_value::<crate::agent::resource::ImprovementHistory>(v.clone()).ok()
    }) {
        put_bincode(store, b"agent:improvement_history", &h);
    }
    if let Some(stats) = extra.get("method_stats").and_then(|v| {
        serde_json::from_value::<Vec<crate::agent::decomposition::MethodStats>>(v.clone()).ok()
    }) {
        put_bincode(store, b"agent:method_stats", &stats);
    }
    if let Some(summary) = extra.get("session_summary").and_then(|v| {
        serde_json::from_value::<crate::agent::memory::SessionSummary>(v.clone()).ok()
    }) {
        put_bincode(store, b"agent:session_summary", &summary);
    }
    if let Some(portable) = extra.get("person_memory").and_then(|v| {
        serde_json::from_value::<crate::agent::contact_memory::PortablePersonMemory>(v.clone()).ok()
    }) {
        let index = crate::agent::contact_memory::PersonMemoryIndex::from_portable(portable);
        put_bincode(store, b"agent:person_memory", &index);
    }
}

/// Convert a persisted `facts` row into a KG [`Triple`].
fn fact_to_triple(row: &fact::Model) -> Option<Triple> {
    let s = SymbolId::new(row.subject as u64)?;
    let p = SymbolId::new(row.predicate as u64)?;
    let o = SymbolId::new(row.object as u64)?;
    let mut t = Triple::new(s, p, o);
    t.confidence = row.confidence;
    t.timestamp = row.asserted_at.timestamp().max(0) as u64;
    t.provenance_id = row.provenance_id.map(|v| v as u64);
    t.compartment_id = row.compartment.clone();
    Some(t)
}

/// Apply one replayed WAL mutation. Returns whether it was applied (unknown
/// kinds and malformed payloads are skipped). Microtheory/compartment replay is
/// deferred (38d). `kind="turn"` rows (ADR 091) are explicitly skipped — raw
/// transcript-capture staging, never a KG mutation.
fn apply_mutation(engine: &Engine, m: &akhomed_store::TailMutation) -> bool {
    let ids = |p: &serde_json::Value| -> Option<(SymbolId, SymbolId, SymbolId)> {
        Some((
            SymbolId::new(p.get("subject")?.as_u64()?)?,
            SymbolId::new(p.get("predicate")?.as_u64()?)?,
            SymbolId::new(p.get("object")?.as_u64()?)?,
        ))
    };
    match m.kind.as_str() {
        "assert" => {
            let Some((s, p, o)) = ids(&m.payload) else {
                return false;
            };
            let mut t = Triple::new(s, p, o);
            if let Some(c) = m.payload.get("confidence").and_then(|v| v.as_f64()) {
                t.confidence = c as f32;
            }
            if let Some(comp) = m.payload.get("compartment").and_then(|v| v.as_str()) {
                t.compartment_id = Some(comp.to_string());
            }
            engine.add_triple(&t).is_ok()
        }
        "retract" => {
            let Some((s, p, o)) = ids(&m.payload) else {
                return false;
            };
            engine.remove_triple(s, p, o).is_ok()
        }
        // ADR 091 decision 3: `turn` rows are raw transcript-capture staging,
        // never KG mutations — the KG facts derived from a turn (if any) get
        // their own `assert` rows. Explicitly skip rather than falling through
        // to the catch-all, so the intent is visible at the call site. `false`
        // just means "not counted in `replayed`" here (see the caller at
        // `wake`, ~line 690-696) — it is NOT treated as an error or an unknown
        // kind; wake proceeds normally with `turn` rows present.
        "turn" => false,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::drives::DriveKind;
    use crate::agent::goal::{GoalSource, GoalStatus};

    // The reconstruction helpers (writer-side `goal_to_model` + wake-side
    // `build_*`) are the highest-risk paths — pure row<->struct mapping that the
    // live wake test exercises only with default state. These cover the rich
    // cases without needing Postgres.

    #[test]
    fn goal_detail_roundtrips_with_data_bearing_variants() {
        let goal = crate::agent::Goal {
            symbol_id: SymbolId::new(7).unwrap(),
            description: "learn pyramid construction".into(),
            status: GoalStatus::Failed {
                reason: "insufficient evidence".into(),
            },
            priority: 5,
            success_criteria: "knows the ramp theory".into(),
            parent: Some(SymbolId::new(3).unwrap()),
            children: vec![SymbolId::new(8).unwrap(), SymbolId::new(9).unwrap()],
            created_at: 100,
            cycles_worked: 3,
            last_progress_cycle: 120,
            source: Some(GoalSource::DriveExceeded {
                drive: "curiosity".into(),
                strength: 0.7,
            }),
            blocked_by: vec![SymbolId::new(4).unwrap()],
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
            estimated_effort: None,
        };
        // Writer: scalar/optional state → `detail`; `children`/`blocked_by` →
        // normalised edge rows. Reader: `build_goals` overlays the edges back on.
        let akh = Uuid::nil();
        let model = goal_to_model(akh, &goal);
        let parent_id = goal.symbol_id.get() as i64;
        let hierarchy: Vec<goal_hierarchy::Model> = goal
            .children
            .iter()
            .enumerate()
            .map(|(pos, c)| goal_hierarchy::Model {
                akh_id: akh,
                parent_id,
                child_id: c.get() as i64,
                position: pos as i32,
            })
            .collect();
        let dependencies: Vec<goal_dependency::Model> = goal
            .blocked_by
            .iter()
            .map(|b| goal_dependency::Model {
                akh_id: akh,
                goal_id: parent_id,
                blocking_goal_id: b.get() as i64,
            })
            .collect();
        let back = build_goals(std::slice::from_ref(&model), &hierarchy, &dependencies);
        assert_eq!(back.len(), 1, "one goal reconstructed");
        assert_eq!(
            serde_json::to_value(&back[0]).unwrap(),
            serde_json::to_value(&goal).unwrap(),
            "goal must round-trip losslessly: scalars via detail, edges via tables",
        );
        // The edge tables are the source of truth for the list fields.
        assert_eq!(
            back[0].children,
            vec![SymbolId::new(8).unwrap(), SymbolId::new(9).unwrap()],
            "children reconstructed in order from goal_hierarchy",
        );
        assert_eq!(
            back[0].blocked_by,
            vec![SymbolId::new(4).unwrap()],
            "blocked_by reconstructed from goal_dependencies",
        );
    }

    #[test]
    fn goal_hierarchy_children_keep_position_order() {
        // Edge rows arriving out of DB order must be sorted by `position`.
        let detail = serde_json::to_value(crate::agent::Goal {
            symbol_id: SymbolId::new(1).unwrap(),
            description: "root".into(),
            status: GoalStatus::Active,
            priority: 1,
            success_criteria: "".into(),
            parent: None,
            children: vec![],
            created_at: 0,
            cycles_worked: 0,
            last_progress_cycle: 0,
            source: None,
            blocked_by: vec![],
            priority_rationale: None,
            justification: None,
            reformulated_from: None,
            estimated_effort: None,
        })
        .unwrap();
        let row = goal::Model {
            akh_id: Uuid::nil(),
            id: 1,
            description: "root".into(),
            status: "Active".into(),
            priority: 1,
            success_criteria: "".into(),
            parent: None,
            created_at: 0,
            cycles_worked: 0,
            last_progress_cycle: 0,
            source: None,
            detail: Some(detail),
        };
        let hierarchy = vec![
            goal_hierarchy::Model { akh_id: Uuid::nil(), parent_id: 1, child_id: 30, position: 2 },
            goal_hierarchy::Model { akh_id: Uuid::nil(), parent_id: 1, child_id: 10, position: 0 },
            goal_hierarchy::Model { akh_id: Uuid::nil(), parent_id: 1, child_id: 20, position: 1 },
        ];
        let back = build_goals(std::slice::from_ref(&row), &hierarchy, &[]);
        assert_eq!(
            back[0].children,
            vec![
                SymbolId::new(10).unwrap(),
                SymbolId::new(20).unwrap(),
                SymbolId::new(30).unwrap(),
            ],
            "children must be ordered by position, not DB row order",
        );
    }

    #[test]
    fn learned_method_steps_roundtrip_via_child_rows() {
        use crate::agent::chunking::GeneralizedStep;
        let method = learned_method::Model {
            akh_id: Uuid::nil(),
            id: 42,
            name: "research-then-reason".into(),
            precondition_pattern: "has {Entity}".into(),
            precondition_vector: vec![0x01, 0x02],
            confidence: 0.8,
            usage_count: 5,
            success_count: 4,
            failure_count: 1,
            learned_at: 10,
            last_used: 20,
            source_goal_description: "learn pyramids".into(),
        };
        // Step rows in non-canonical order — must be re-ordered by step_index.
        let steps = vec![
            learned_method_step::Model {
                akh_id: Uuid::nil(),
                method_id: 42,
                step_index: 1,
                tool: "reason".into(),
                arg_pattern: "analyze results".into(),
                expected_outcome: "reason_result".into(),
            },
            learned_method_step::Model {
                akh_id: Uuid::nil(),
                method_id: 42,
                step_index: 0,
                tool: "kg_query".into(),
                arg_pattern: "find {Entity}".into(),
                expected_outcome: "kg_query_result".into(),
            },
        ];
        let back =
            build_learned_methods(std::slice::from_ref(&method), &steps).expect("methods present");
        assert_eq!(back.len(), 1, "one method reconstructed");
        let expected = vec![
            GeneralizedStep {
                tool: "kg_query".into(),
                arg_pattern: "find {Entity}".into(),
                expected_outcome: "kg_query_result".into(),
            },
            GeneralizedStep {
                tool: "reason".into(),
                arg_pattern: "analyze results".into(),
                expected_outcome: "reason_result".into(),
            },
        ];
        assert_eq!(
            serde_json::to_value(&back[0].steps).unwrap(),
            serde_json::to_value(&expected).unwrap(),
            "steps must be reconstructed in step_index order from the child rows",
        );
        assert_eq!(back[0].name, "research-then-reason");
        assert_eq!(back[0].usage_count, 5);
    }

    #[test]
    fn drive_system_restores_into_canonical_slots() {
        // Rows in non-canonical DB order; build_drive_system must place each by kind.
        let rows = vec![
            drive::Model {
                akh_id: Uuid::nil(),
                kind: "efficiency".into(),
                strength: 0.9,
                last_computed: 5,
                last_triple_count: 10,
                last_symbol_count: 4,
                threshold: 0.4,
            },
            drive::Model {
                akh_id: Uuid::nil(),
                kind: "curiosity".into(),
                strength: 0.1,
                last_computed: 6,
                last_triple_count: 11,
                last_symbol_count: 5,
                threshold: 0.6,
            },
        ];
        let sys = build_drive_system(&rows).expect("drives present");
        assert_eq!(sys.strength(DriveKind::Curiosity), 0.1);
        assert_eq!(sys.strength(DriveKind::Efficiency), 0.9);
        assert_eq!(sys.thresholds[DriveKind::Curiosity as usize], 0.6);
        assert_eq!(sys.thresholds[DriveKind::Efficiency as usize], 0.4);
        assert!(build_drive_system(&[]).is_none(), "no rows → None (default fallback)");
    }

    #[test]
    fn competence_restores_maps_and_sorted_calibration() {
        let tools = vec![competence_tool::Model {
            akh_id: Uuid::nil(),
            tool: "code_gen".into(),
            successes: 8,
            attempts: 10,
        }];
        let cats = vec![competence_category::Model {
            akh_id: Uuid::nil(),
            category: "research".into(),
            completed: 2,
            attempted: 3,
        }];
        // Out of order: seq 1 then 0 — must be sorted by seq on rebuild.
        let calib = vec![
            competence_calibration::Model {
                akh_id: Uuid::nil(),
                seq: 1,
                predicted: 0.5,
                actual: 1.0,
            },
            competence_calibration::Model {
                akh_id: Uuid::nil(),
                seq: 0,
                predicted: 0.3,
                actual: 0.0,
            },
        ];
        let cm = build_competence(&tools, &cats, &calib);
        assert_eq!(cm.tools["code_gen"].successes, 8);
        assert_eq!(cm.tools["code_gen"].attempts, 10);
        assert_eq!(cm.categories["research"].attempted, 3);
        assert_eq!(cm.calibration_history.len(), 2);
        assert_eq!(cm.calibration_history[0].predicted, 0.3, "sorted by seq");
        assert_eq!(cm.calibration_history[1].predicted, 0.5);
    }

    #[test]
    fn effort_case_goal_vector_bincode_roundtrips() {
        let rows = vec![effort_case::Model {
            akh_id: Uuid::nil(),
            seq: 0,
            description: "decompose".into(),
            goal_vector: bincode::serialize(&vec![1u32, 2, 3]).unwrap(),
            cycles_consumed: 4,
            succeeded: true,
            tool_usage: serde_json::json!({ "code_gen": 2, "search": 1 }),
            initial_coverage: 3,
        }];
        let cases = build_effort_cases(&rows).expect("effort cases present");
        assert_eq!(cases[0].goal_vector, vec![1u32, 2, 3]);
        assert_eq!(cases[0].tool_usage["code_gen"], 2);
        assert_eq!(cases[0].cycles_consumed, 4);
        assert!(cases[0].succeeded);
    }

    #[test]
    fn consolidation_phase_parses_debug_form() {
        use crate::agent::sleep::ConsolidationPhase;
        assert_eq!(parse_consolidation_phase("DreamState"), ConsolidationPhase::DreamState);
        assert_eq!(parse_consolidation_phase("Consolidation"), ConsolidationPhase::Consolidation);
        assert_eq!(parse_consolidation_phase("garbage"), ConsolidationPhase::Wake);
    }

    #[test]
    fn person_memory_restore_extra_writes_meta() {
        // Wake-side: an `extra` bag with a `person_memory` portable must land as
        // the `agent:person_memory` bincode meta that `Agent::resume` reads.
        let tmp = tempfile::tempdir().unwrap();
        let store = TieredStore::with_persistence(tmp.path(), "symbols").unwrap();

        let ep1 = SymbolId::new(11).unwrap();
        let ep2 = SymbolId::new(12).unwrap();
        let topic = SymbolId::new(20).unwrap();
        let portable = crate::agent::contact_memory::PortablePersonMemory {
            episodes_by_contact: [("alice".to_string(), vec![ep2, ep1])].into_iter().collect(),
            topic_index: vec![(("alice".to_string(), topic), vec![ep1])],
            max_per_contact: 100,
        };
        let extra = serde_json::json!({ "person_memory": portable });

        restore_extra(&store, Some(&extra));

        let bytes = store
            .get_meta(b"agent:person_memory")
            .unwrap()
            .expect("agent:person_memory meta written");
        let idx: crate::agent::contact_memory::PersonMemoryIndex =
            bincode::deserialize(&bytes).expect("deserialize as PersonMemoryIndex");
        assert_eq!(idx.recall_with_person("alice", 10), vec![ep2, ep1]);
        assert_eq!(idx.recall_with_person_about("alice", topic, 10), vec![ep1]);
    }

    #[test]
    fn symbol_kind_roundtrips() {
        for kind in [
            SymbolKind::Entity,
            SymbolKind::Relation,
            SymbolKind::Composite,
            SymbolKind::Glyph { codepoint: '𓂀' },
        ] {
            let (s, g) = symbol_kind_parts(&kind);
            assert_eq!(symbol_kind_from(&s, g), kind, "kind {kind:?} must round-trip");
        }
    }

    #[test]
    fn apply_mutation_skips_turn_rows_without_error() {
        // ADR 091 decision 3: `turn` rows are raw transcript staging, never
        // replayed into the KG. `apply_mutation` must return `false` (not
        // counted as "replayed") without the caller treating it as an error —
        // wake proceeds normally with `turn` rows present in the tail.
        let engine = Engine::new(EngineConfig::default()).unwrap();
        let m = akhomed_store::TailMutation {
            kind: "turn".to_string(),
            payload: serde_json::json!({
                "v": 1,
                "ts": "2026-07-10T00:00:00Z",
                "workspace": "test-akh",
                "model": null,
                "user_prompt": "hi",
                "agent_answer": "hello",
                "tools": [],
            }),
        };
        assert!(!apply_mutation(&engine, &m), "turn rows are not KG mutations");

        // An unrelated malformed/unknown kind also returns false — same
        // non-error signal — so the explicit "turn" arm is a documentation /
        // intent improvement, not a behavior change vs the prior catch-all.
        let unknown = akhomed_store::TailMutation {
            kind: "totally_unknown".to_string(),
            payload: serde_json::json!({}),
        };
        assert!(!apply_mutation(&engine, &unknown));
    }
}
