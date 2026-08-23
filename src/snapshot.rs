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
use akhomed_store::entity::{
    agenda, akh_state, competence_calibration, competence_category, competence_tool,
    curiosity_config, drive, effort_case, fact, failure_case, goal, goal_dependency,
    goal_hierarchy, learned_method, learned_method_step, provenance_snapshot, sleep_metric, symbol,
};
use akhomed_store::uuid::Uuid;

use crate::agent::Agent;
use crate::engine::{Engine, EngineConfig};
use crate::graph::Triple;
use crate::registry::SymbolRegistry;
use crate::store::TieredStore;
use crate::symbol::{SymbolId, SymbolKind, SymbolMeta};

const PROVENANCE_SNAPSHOT_VERSION: u64 = 1;

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
    /// Durable residency generation held by the capturing daemon.
    residency_epoch: i64,
    /// WAL sequence sampled before the in-memory capture began.  It is a safe
    /// lower bound: mutations racing with capture may be replayed twice, but a
    /// captured snapshot can never advance past a mutation it did not see.
    watermark: i64,
    symbols: Vec<symbol::Model>,
    facts: Vec<fact::Model>,
    provenance: Vec<provenance_snapshot::Model>,
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
    let watermark = akhomed_store::snapshots::current_watermark(db, akh_id).await?;
    let data = capture_snapshot_at_watermark(agent, akh_id, workspace_id, watermark)?;
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
    // A zero lower bound is deliberately conservative for callers that do not
    // have a database handle.  Operational callers use
    // `capture_snapshot_at_watermark`; replaying older rows is idempotent and
    // safer than sampling a watermark after capture.
    capture_snapshot_at_watermark(agent, akh_id, workspace_id, 0)
}

/// Capture state against a WAL lower bound sampled *before* acquiring the
/// agent guard and copying its state.
pub fn capture_snapshot_at_watermark(
    agent: &Agent,
    akh_id: Uuid,
    workspace_id: Uuid,
    watermark: i64,
) -> SnapshotResult<SnapshotData> {
    capture_snapshot_at_fence(agent, akh_id, workspace_id, watermark, 0)
}

/// Capture state at a WAL lower bound and a durable residency fencing epoch.
/// Production daemon callers must pass the epoch returned by
/// `StablePgSession::claim_akh_residency`; the legacy wrapper above is intentionally usable
/// only while the Akh remains at its unclaimed epoch zero.
pub fn capture_snapshot_at_fence(
    agent: &Agent,
    akh_id: Uuid,
    workspace_id: Uuid,
    watermark: i64,
    residency_epoch: i64,
) -> SnapshotResult<SnapshotData> {
    if watermark < 0 {
        return Err(SnapshotError::Agent(
            "snapshot WAL watermark cannot be negative".into(),
        ));
    }
    if residency_epoch < 0 {
        return Err(SnapshotError::Agent(
            "snapshot residency epoch cannot be negative".into(),
        ));
    }
    let engine = agent.engine();

    // Flush durable state to local redb (registry, allocator counter, KG snapshot).
    agent.persist().map_err(|e| SnapshotError::Agent(e.to_string()))?;

    let allocator_next = i64::try_from(engine.symbol_allocator_next()).map_err(|_| {
        SnapshotError::Agent(format!(
            "symbol allocator next ID {} exceeds the relational snapshot range",
            engine.symbol_allocator_next()
        ))
    })?;

    let symbol_metas = engine.dump_symbols();
    let symbols: Vec<symbol::Model> = symbol_metas
        .iter()
        .map(|m| {
            let (kind, glyph) = symbol_kind_parts(&m.kind);
            Ok(symbol::Model {
                akh_id,
                id: relational_symbol_id("symbol", m.id)?,
                kind,
                glyph_codepoint: glyph,
                label: m.label.clone(),
            })
        })
        .collect::<SnapshotResult<_>>()?;

    // Live KG → rows; retracted triples are absent.  Capture the graph
    // before the append-only ledger so any linked ID visible here is already
    // committed by producers that obey `add_triples_with_provenance`.
    let graph_triples = engine.all_triples();
    let facts = graph_triples
        .iter()
        .map(|t| {
            let provenance_id = t
                .provenance_id
                .map(i64::try_from)
                .transpose()
                .map_err(|_| {
                    SnapshotError::Agent(format!(
                        "fact provenance ID {} exceeds the relational snapshot range",
                        t.provenance_id.unwrap_or_default()
                    ))
            })?;
            Ok(fact::Model {
                workspace_id,
                subject: relational_symbol_id("fact subject", t.subject)?,
                predicate: relational_symbol_id("fact predicate", t.predicate)?,
                object: relational_symbol_id("fact object", t.object)?,
                akh_id,
                confidence: t.confidence,
                asserted_at: ts(t.timestamp),
                retracted_at: None,
                provenance_id,
                compartment: t.compartment_id.clone(),
            })
        })
        .collect::<SnapshotResult<Vec<_>>>()?;

    // The complete local ledger is paired transactionally with `facts` in
    // Postgres.  Keeping every record (not only currently linked ones)
    // preserves raw provenance queries and the exact next-ID watermark.
    let provenance_records = engine
        .all_provenance()
        .map_err(|e| SnapshotError::Agent(format!("capture provenance ledger: {e}")))?;
    let provenance = provenance_records
        .iter()
        .map(|record| {
            let id = record.id.ok_or_else(|| {
                SnapshotError::Agent("persisted provenance record has no ID".into())
            })?;
            Ok(provenance_snapshot::Model {
                akh_id,
                provenance_id: i64::try_from(id.get()).map_err(|_| {
                    SnapshotError::Agent(format!(
                        "provenance ID {} exceeds the relational snapshot range",
                        id.get()
                    ))
                })?,
                workspace_id,
                derived_symbol: i64::try_from(record.derived_id.get()).map_err(|_| {
                    SnapshotError::Agent(format!(
                        "derived symbol {} exceeds the relational snapshot range",
                        record.derived_id.get()
                    ))
                })?,
                sources: serde_json::to_value(&record.sources).map_err(|e| {
                    SnapshotError::Agent(format!("serialize provenance sources: {e}"))
                })?,
                derivation: serde_json::to_value(&record.kind).map_err(|e| {
                    SnapshotError::Agent(format!("serialize provenance derivation: {e}"))
                })?,
                confidence: record.confidence,
                depth: i64::try_from(record.depth)
                    .map_err(|_| SnapshotError::Agent("provenance depth exceeds i64".into()))?,
                created_at: ts(record.timestamp),
            })
        })
        .collect::<SnapshotResult<Vec<_>>>()?;
    // A complete snapshot is a closed manifest.  Validate it against the
    // exact registry captured above rather than relying on wake to discover a
    // missing nested derivation symbol after this corrupt image is durable.
    validate_symbol_snapshot(&symbol_metas, allocator_next)?;
    validate_fact_symbols(&facts, &symbol_metas)?;
    validate_provenance_symbols(&provenance_records, &symbol_metas)?;
    validate_fact_provenance_links(&facts, &provenance_records)?;

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
        .map(|(id, prio)| {
            Ok(agenda::Model {
                akh_id,
                project_id: relational_symbol_id("agenda project", *id)?,
                priority: *prio as i16,
                is_active: Some(*id) == active,
            })
        })
        .collect::<SnapshotResult<_>>()?;

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

    let goals: Vec<goal::Model> = agent
        .goals
        .iter()
        .map(|g| goal_to_model(akh_id, g))
        .collect::<SnapshotResult<_>>()?;
    // Normalised goal DAG: `children` → ordered hierarchy edges; `blocked_by` →
    // dependency edges.
    let mut goal_hierarchy = Vec::new();
    let mut goal_dependencies = Vec::new();
    for g in &agent.goals {
        let goal_id = relational_symbol_id("goal", g.symbol_id)?;
        for (position, child) in g.children.iter().enumerate() {
            goal_hierarchy.push(goal_hierarchy::Model {
                akh_id,
                parent_id: goal_id,
                child_id: relational_symbol_id("goal child", *child)?,
                position: i32::try_from(position).map_err(|_| {
                    SnapshotError::Agent("goal child position exceeds i32".into())
                })?,
            });
        }
        for blocker in &g.blocked_by {
            goal_dependencies.push(goal_dependency::Model {
                akh_id,
                goal_id,
                blocking_goal_id: relational_symbol_id("blocking goal", *blocker)?,
            });
        }
    }

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
        .map(|m| {
            Ok(learned_method::Model {
                akh_id,
                id: relational_symbol_id("learned method", m.id)?,
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
        })
        .collect::<SnapshotResult<_>>()?;
    // Normalised method steps: one ordered child row per `GeneralizedStep`.
    let mut method_steps = Vec::new();
    for method in agent.method_index.methods() {
        let method_id = relational_symbol_id("learned method step", method.id)?;
        for (index, step) in method.steps.iter().enumerate() {
            method_steps.push(learned_method_step::Model {
                akh_id,
                method_id,
                step_index: i32::try_from(index).map_err(|_| {
                    SnapshotError::Agent("learned method step index exceeds i32".into())
                })?,
                tool: step.tool.clone(),
                arg_pattern: step.arg_pattern.clone(),
                expected_outcome: step.expected_outcome.clone(),
            });
        }
    }

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
        "provenance_snapshot_version": PROVENANCE_SNAPSHOT_VERSION,
        "audience": agent.audience,
        "improvement_history": agent.improvement_history,
        "method_stats": agent.method_registry_htn.export_stats(),
        "session_summary": session_summary,
        "person_memory": agent.person_memory.to_portable(),
    });

    Ok(SnapshotData {
        akh_id,
        workspace_id,
        residency_epoch,
        watermark,
        symbols,
        facts,
        provenance,
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
/// (async; holds no agent lock). The complete relational image and its
/// recovery watermark are published in one serialized database transaction.
pub async fn persist_snapshot(data: SnapshotData, db: &DatabaseConnection) -> SnapshotResult<()> {
    use akhomed_store::snapshots as ops;
    let SnapshotData {
        akh_id,
        workspace_id,
        residency_epoch,
        watermark,
        symbols,
        facts,
        provenance,
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
        // The store reads any existing independently-advanced values while it
        // holds the publication lock. Zero is the correct first-row default.
        harvest_watermark: 0,
        librarian_watermark: 0,
        updated_at: chrono::Utc::now(),
    };
    ops::publish_snapshot(
        db,
        workspace_id,
        akh_id,
        ops::SnapshotPublication {
            residency_epoch,
            symbols,
            facts,
            provenance,
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
            competence_tools: comp_tools,
            competence_categories: comp_cats,
            competence_calibration: comp_calib,
            sleep_metrics: sleep_rows,
            state,
        },
    )
    .await?;

    Ok(())
}

// --- helpers ---

/// Checked conversion for any registry ID crossing into a signed relational
/// column. Silent `as` wrapping could persist a negative ID and make the next
/// wake either reject a valid snapshot or allocate over an existing symbol.
fn relational_symbol_id(field: &str, id: SymbolId) -> SnapshotResult<i64> {
    i64::try_from(id.get()).map_err(|_| {
        SnapshotError::Agent(format!(
            "{field} symbol ID {} exceeds the relational snapshot range",
            id.get()
        ))
    })
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

fn goal_to_model(akh_id: Uuid, g: &crate::agent::Goal) -> SnapshotResult<goal::Model> {
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
    Ok(goal::Model {
        akh_id,
        id: relational_symbol_id("goal", g.symbol_id)?,
        description: g.description.clone(),
        status,
        priority: g.priority as i16,
        success_criteria: g.success_criteria.clone(),
        parent: g
            .parent
            .map(|symbol| relational_symbol_id("goal parent", symbol))
            .transpose()?,
        created_at: g.created_at as i64,
        cycles_worked: g.cycles_worked as i32,
        last_progress_cycle: g.last_progress_cycle as i64,
        source,
        detail,
    })
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
    let mut fact_rows = ops::load_workspace_facts(db, workspace_id).await?;
    let provenance_rows = ops::load_provenance_snapshot(db, akh_id).await?;
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

    let mut provenance_records = provenance_rows
        .iter()
        .map(provenance_row_to_record)
        .collect::<SnapshotResult<Vec<_>>>()?;
    let provenance_snapshot_complete = state
        .extra
        .as_ref()
        .and_then(|extra| extra.get("provenance_snapshot_version"))
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|version| version >= PROVENANCE_SNAPSHOT_VERSION);

    let mut symbol_metas = symbol_rows_to_metas(&symbol_rows)?;
    let base_allocator_next =
        validate_symbol_snapshot(&symbol_metas, state.allocator_next)?;

    // Validate the base image before admitting any post-snapshot symbol or
    // provenance record.  Otherwise a dangling base fact can be
    // retroactively made to look valid when an unrelated tail record happens
    // to reuse the same numeric ID.
    validate_base_restore_image(
        &fact_rows,
        &provenance_records,
        &symbol_metas,
        provenance_snapshot_complete,
    )?;

    if !provenance_snapshot_complete {
        // A tail record can legitimately reuse an ID that a pre-v1 local-only
        // ledger once assigned. Clear unverifiable base links before merging
        // the tail, so that future record can never retroactively rebind an
        // older fact to unrelated provenance.
        let cleared =
            clear_unverifiable_legacy_provenance_links(&mut fact_rows, &provenance_records);
        if cleared > 0 {
            tracing::warn!(
                akh = %akh_id,
                %workspace_id,
                cleared,
                "wake: downgraded unverifiable pre-v1 provenance links to untracked"
            );
        }
    }
    let tail = ops::load_mutation_tail(db, akh_id, state.watermark).await?;
    let tail_allocator_next =
        prepare_tail_restore(&tail, &mut symbol_metas, &mut provenance_records)?;
    let restored_allocator_next = base_allocator_next.max(tail_allocator_next);
    validate_provenance_symbols(&provenance_records, &symbol_metas)?;
    validate_fact_provenance_links(&fact_rows, &provenance_records)?;
    validate_fact_symbols(&fact_rows, &symbol_metas)?;

    // 3. Reconstruct the local redb meta the engine + agent read on open. The
    //    store must be dropped before `Engine::new` reopens the same redb file.
    {
        let store = TieredStore::with_persistence(data_dir, "symbols").map_err(|source| {
            SnapshotError::StoreOpen {
                path: data_dir.display().to_string(),
                source,
            }
        })?;

        write_registry(&store, &symbol_metas)?;
        put_bincode_required(
            &store,
            b"sym_allocator_next",
            &restored_allocator_next,
        )?;
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
        Engine::new_for_restore(EngineConfig {
            data_dir: Some(data_dir.to_path_buf()),
            ..Default::default()
        })
        .map_err(|e| SnapshotError::Engine(e.to_string()))?,
    );

    // Replace the local redb ledger before publishing any restored fact.  This
    // both removes stale host-local records and restores the exact next ID, so
    // a dangling fact can never be rebound to an unrelated future record.
    engine
        .replace_all_provenance(&provenance_records)
        .map_err(|e| SnapshotError::Engine(format!("restore provenance ledger: {e}")))?;

    // 5. Restore the KG (+ item memory) from the relational facts.
    let triples = fact_rows
        .iter()
        .map(fact_to_triple)
        .collect::<SnapshotResult<Vec<_>>>()?;
    engine
        .restore_triples(&triples)
        .map_err(|e| SnapshotError::Engine(e.to_string()))?;

    // 6. Replay the WAL tail newer than the snapshot watermark. No sink is
    //    installed yet, so these mutations are applied but not re-logged.
    let mut replayed = 0usize;
    for m in &tail {
        if apply_mutation(&engine, m)? {
            replayed += 1;
        }
    }
    // Disk-backed compartments and skills may publish provenance-linked
    // triples.  Start them only after the restored ledger/allocator is live,
    // otherwise replacing the ledger above would immediately dangle their
    // freshly assigned IDs.
    engine.finish_deferred_startup_loads();
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

/// Encode and persist recovery-critical metadata. Unlike optional cognitive
/// state, failure here aborts wake so an engine with an incorrect allocator or
/// incomplete registry is never published.
fn put_bincode_required<T: serde::Serialize>(
    store: &TieredStore,
    key: &[u8],
    value: &T,
) -> SnapshotResult<()> {
    let bytes = bincode::serialize(value).map_err(|error| {
        SnapshotError::Agent(format!(
            "wake: encode required meta {}: {error}",
            String::from_utf8_lossy(key)
        ))
    })?;
    store.put_meta(key, &bytes).map_err(|error| {
        SnapshotError::Agent(format!(
            "wake: persist required meta {}: {error}",
            String::from_utf8_lossy(key)
        ))
    })?;
    Ok(())
}

/// Rebuild the symbol registry from `symbols` rows and persist it as `sym_meta:`
/// entries — the format [`Engine::new`] reads on open.
fn write_registry(store: &TieredStore, metas: &[SymbolMeta]) -> SnapshotResult<()> {
    let registry = SymbolRegistry::new();
    let mut ids = std::collections::HashSet::with_capacity(metas.len());
    for meta in metas {
        if !ids.insert(meta.id) {
            return Err(SnapshotError::Agent(format!(
                "wake: duplicate symbol ID {}",
                meta.id.get()
            )));
        }
        registry.register(meta.clone()).map_err(|error| {
            SnapshotError::Agent(format!(
                "wake: invalid symbol {} ({}): {error}",
                meta.id.get(),
                meta.label
            ))
        })?;
    }
    registry
        .replace_persisted(store)
        .map_err(|error| SnapshotError::Agent(format!("wake: replace symbol registry: {error}")))?;
    Ok(())
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

fn symbol_rows_to_metas(rows: &[symbol::Model]) -> SnapshotResult<Vec<SymbolMeta>> {
    rows.iter()
        .map(|row| {
            let raw_id = u64::try_from(row.id).map_err(|_| {
                SnapshotError::Agent(format!(
                    "wake: symbol {} has a non-positive ID {}",
                    row.label, row.id
                ))
            })?;
            let id = SymbolId::new(raw_id)
                .filter(|id| !is_reserved_symbol_id(id.get()))
                .ok_or_else(|| {
                    SnapshotError::Agent(format!(
                        "wake: symbol {} uses invalid/reserved ID {}",
                        row.label, row.id
                    ))
                })?;
            Ok(SymbolMeta {
                id,
                kind: symbol_kind_from(&row.kind, row.glyph_codepoint),
                label: row.label.clone(),
                // The legacy relational symbol table predates these fields.
                // Replay-complete WAL envelopes carry their exact values for
                // post-snapshot symbols.
                created_at: 0,
                source: None,
            })
        })
        .collect()
}

/// Validate the authoritative relational registry and its captured allocator.
/// Returns the exact captured next ID once it is known to be beyond every
/// restored symbol.
fn validate_symbol_snapshot(metas: &[SymbolMeta], allocator_next: i64) -> SnapshotResult<u64> {
    let allocator_next = u64::try_from(allocator_next).map_err(|_| {
        SnapshotError::Agent(format!(
            "wake: symbol allocator has invalid next ID {allocator_next}"
        ))
    })?;
    if allocator_next == 0 || is_reserved_symbol_id(allocator_next) {
        return Err(SnapshotError::Agent(format!(
            "wake: symbol allocator has invalid/reserved next ID {allocator_next}"
        )));
    }

    let mut ids = std::collections::HashSet::with_capacity(metas.len());
    let mut labels = std::collections::HashMap::with_capacity(metas.len());
    let mut max_id = 0u64;
    for meta in metas {
        let raw_id = meta.id.get();
        if is_reserved_symbol_id(raw_id) || !ids.insert(raw_id) {
            return Err(SnapshotError::Agent(format!(
                "wake: duplicate or reserved symbol ID {raw_id}"
            )));
        }
        let normalized = crate::registry::normalize_label_key(&meta.label);
        if let Some(previous) = labels.insert(normalized, raw_id)
            && previous != raw_id
        {
            return Err(SnapshotError::Agent(format!(
                "wake: duplicate symbol label {} for IDs {previous} and {raw_id}",
                meta.label
            )));
        }
        max_id = max_id.max(raw_id);
    }
    let minimum_next = max_id.checked_add(1).ok_or_else(|| {
        SnapshotError::Agent("wake: restored symbol ID space is exhausted".into())
    })?;
    if is_reserved_symbol_id(minimum_next) {
        return Err(SnapshotError::Agent(
            "wake: restored symbol ID space reaches a reserved protocol marker".into(),
        ));
    }
    if allocator_next < minimum_next {
        return Err(SnapshotError::Agent(format!(
            "wake: allocator next ID {allocator_next} collides with restored symbol {max_id}"
        )));
    }
    Ok(allocator_next)
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
                .map(|v| {
                    v.iter()
                        .filter_map(|(_, c)| SymbolId::new(*c as u64))
                        .collect()
                })
                .unwrap_or_default();
            goal.blocked_by = blocked_by_goal
                .get(&row.id)
                .map(|v| v.iter().filter_map(|b| SymbolId::new(*b as u64)).collect())
                .unwrap_or_default();
            Some(goal)
        })
        .collect()
}

fn build_failure_cases(
    rows: &[failure_case::Model],
) -> Vec<crate::agent::metacognition::FailureCase> {
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

fn build_effort_cases(
    rows: &[effort_case::Model],
) -> Option<Vec<crate::agent::resource::EffortCase>> {
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
    use crate::agent::metacognition::{
        CalibrationPoint, CategoryRecord, CompetenceModel, ToolRecord,
    };
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

/// Convert one portable relational row back into the exact local ledger
/// record.  Invalid IDs or malformed typed JSON fail wake rather than leaving
/// a dangling fact link behind.
fn provenance_row_to_record(
    row: &provenance_snapshot::Model,
) -> SnapshotResult<crate::provenance::ProvenanceRecord> {
    let raw_id = u64::try_from(row.provenance_id).map_err(|_| {
        SnapshotError::Engine(format!(
            "invalid restored provenance ID {}",
            row.provenance_id
        ))
    })?;
    let id = crate::provenance::ProvenanceId::new(raw_id)
        .ok_or_else(|| SnapshotError::Engine("restored provenance ID must be non-zero".into()))?;
    let derived_raw = u64::try_from(row.derived_symbol).map_err(|_| {
        SnapshotError::Engine(format!(
            "invalid restored derived symbol {}",
            row.derived_symbol
        ))
    })?;
    let derived_id = SymbolId::new(derived_raw)
        .ok_or_else(|| SnapshotError::Engine("restored derived symbol must be non-zero".into()))?;
    let sources = serde_json::from_value::<Vec<SymbolId>>(row.sources.clone())
        .map_err(|e| SnapshotError::Engine(format!("decode provenance {raw_id} sources: {e}")))?;
    let kind = serde_json::from_value::<crate::provenance::DerivationKind>(row.derivation.clone())
        .map_err(|e| {
            SnapshotError::Engine(format!("decode provenance {raw_id} derivation: {e}"))
        })?;
    let depth = usize::try_from(row.depth).map_err(|_| {
        SnapshotError::Engine(format!("invalid provenance {raw_id} depth {}", row.depth))
    })?;

    Ok(crate::provenance::ProvenanceRecord {
        id: Some(id),
        derived_id,
        sources,
        kind,
        confidence: row.confidence,
        depth,
        timestamp: row.created_at.timestamp().max(0) as u64,
    })
}

/// Ensure every fact-level provenance link resolves inside the same complete
/// ledger image.  Used on both sides of the relational snapshot boundary so a
/// corrupt source engine cannot publish a snapshot that wake would reject.
fn validate_fact_provenance_links(
    facts: &[fact::Model],
    records: &[crate::provenance::ProvenanceRecord],
) -> SnapshotResult<()> {
    let provenance_ids: std::collections::HashSet<u64> = records
        .iter()
        .filter_map(|record| record.id.map(|id| id.get()))
        .collect();
    for fact in facts {
        if let Some(raw_id) = fact.provenance_id {
            let id = u64::try_from(raw_id).map_err(|_| {
                SnapshotError::Engine(format!(
                    "fact ({}, {}, {}) has invalid provenance ID {raw_id}",
                    fact.subject, fact.predicate, fact.object
                ))
            })?;
            if id == 0 || !provenance_ids.contains(&id) {
                return Err(SnapshotError::Engine(format!(
                    "fact ({}, {}, {}) references missing provenance ID {raw_id}",
                    fact.subject, fact.predicate, fact.object
                )));
            }
        }
    }
    Ok(())
}

fn validate_base_restore_image(
    facts: &[fact::Model],
    records: &[crate::provenance::ProvenanceRecord],
    symbols: &[SymbolMeta],
    provenance_snapshot_complete: bool,
) -> SnapshotResult<()> {
    validate_fact_symbols(facts, symbols)?;
    validate_provenance_symbols(records, symbols)?;
    if provenance_snapshot_complete {
        validate_fact_provenance_links(facts, records)?;
    }
    Ok(())
}

/// Pre-v1 snapshots persisted numeric fact links but not the local ledger that
/// gave those IDs meaning.  Keep any link that a concurrently/backfilled
/// portable ledger proves, and clear only the unverifiable remainder so it can
/// never bind to an unrelated record after the allocator restarts.
fn clear_unverifiable_legacy_provenance_links(
    facts: &mut [fact::Model],
    records: &[crate::provenance::ProvenanceRecord],
) -> usize {
    let provenance_ids: std::collections::HashSet<u64> = records
        .iter()
        .filter_map(|record| record.id.map(|id| id.get()))
        .collect();
    let mut cleared = 0;
    for fact in facts {
        let verified = fact
            .provenance_id
            .and_then(|raw_id| u64::try_from(raw_id).ok())
            .is_some_and(|id| id != 0 && provenance_ids.contains(&id));
        if fact.provenance_id.is_some() && !verified {
            fact.provenance_id = None;
            cleared += 1;
        }
    }
    cleared
}

#[derive(serde::Deserialize)]
struct ReplayCompleteAssert {
    v: u64,
    subject: u64,
    predicate: u64,
    object: u64,
    confidence: f32,
    timestamp: u64,
    #[serde(default)]
    compartment: Option<String>,
    #[serde(default)]
    provenance_id: Option<u64>,
    #[serde(default)]
    provenance: Option<crate::provenance::ProvenanceRecord>,
    symbols: Vec<SymbolMeta>,
}

#[derive(serde::Deserialize)]
struct ReplayRetract {
    subject: u64,
    predicate: u64,
    object: u64,
}

#[derive(serde::Deserialize)]
struct ReplayCompleteReplace {
    v: u64,
    retracts: Vec<ReplayRetract>,
    assertion: serde_json::Value,
}

#[derive(serde::Deserialize)]
struct ReplayRetractEdge {
    subject: u64,
    predicate: u64,
    object: u64,
    confidence: f32,
    timestamp: u64,
    #[serde(default)]
    compartment: Option<String>,
    #[serde(default)]
    provenance_id: Option<u64>,
}

#[derive(serde::Deserialize)]
struct ReplayCompleteRetractBatch {
    v: u64,
    retracts: Vec<ReplayRetractEdge>,
    symbols: Vec<SymbolMeta>,
    #[serde(default)]
    provenance: Vec<crate::provenance::ProvenanceRecord>,
}

fn invalid_tail(message: impl Into<String>) -> SnapshotError {
    SnapshotError::Agent(format!("wake: invalid mutation tail: {}", message.into()))
}

fn validate_compartment_id(compartment_id: &str) -> SnapshotResult<()> {
    crate::graph::sparql::SparqlStore::compartment_to_iri(compartment_id)
        .map(|_| ())
        .map_err(|error| {
            SnapshotError::Agent(format!(
                "wake: invalid compartment {compartment_id:?}: {error}"
            ))
        })
}

fn is_reserved_symbol_id(raw: u64) -> bool {
    raw == crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL
        || raw == crate::provenance::RULE_APPLICATION_V2_SENTINEL
}

fn validate_rule_application_payload(
    record: &crate::provenance::ProvenanceRecord,
    conclusion: (SymbolId, SymbolId, SymbolId),
) -> SnapshotResult<()> {
    let crate::provenance::DerivationKind::RuleInference { antecedents, .. } = &record.kind
    else {
        return Ok(());
    };
    let Some(first) = antecedents.first() else {
        return Ok(());
    };
    if first.get() != crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL {
        return Ok(());
    }

    let is_v2 = antecedents.get(1).is_some_and(|marker| {
        marker.get() == crate::provenance::RULE_APPLICATION_V2_SENTINEL
    });
    if !is_v2 {
        if (antecedents.len() - 1) % 3 != 0 {
            return Err(invalid_tail(
                "versioned rule support payload is not a sequence of triples",
            ));
        }
        return Ok(());
    }
    if antecedents.len() < 5 || (antecedents.len() - 5) % 3 != 0 {
        return Err(invalid_tail(
            "V2 rule application payload has an invalid conclusion/support shape",
        ));
    }
    if (antecedents[2], antecedents[3], antecedents[4]) != conclusion {
        return Err(invalid_tail(format!(
            "V2 rule conclusion ({}, {}, {}) does not match assertion ({}, {}, {})",
            antecedents[2].get(),
            antecedents[3].get(),
            antecedents[4].get(),
            conclusion.0.get(),
            conclusion.1.get(),
            conclusion.2.get()
        )));
    }
    Ok(())
}

fn payload_triple_ids(
    payload: &serde_json::Value,
) -> SnapshotResult<(SymbolId, SymbolId, SymbolId)> {
    let parse = |field: &str| {
        let raw = payload
            .get(field)
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| invalid_tail(format!("missing/invalid {field}")))?;
        SymbolId::new(raw)
            .filter(|id| !is_reserved_symbol_id(id.get()))
            .ok_or_else(|| invalid_tail(format!("{field} uses zero/reserved symbol ID {raw}")))
    };
    Ok((parse("subject")?, parse("predicate")?, parse("object")?))
}

fn retract_triple_ids(retract: &ReplayRetract) -> SnapshotResult<(SymbolId, SymbolId, SymbolId)> {
    let parse = |field: &str, raw: u64| {
        SymbolId::new(raw)
            .filter(|id| !is_reserved_symbol_id(id.get()))
            .ok_or_else(|| {
                invalid_tail(format!(
                    "retract {field} uses zero/reserved symbol ID {raw}"
                ))
            })
    };
    Ok((
        parse("subject", retract.subject)?,
        parse("predicate", retract.predicate)?,
        parse("object", retract.object)?,
    ))
}

fn parse_replay_complete_replace(
    payload: &serde_json::Value,
) -> SnapshotResult<ReplayCompleteReplace> {
    let replacement: ReplayCompleteReplace = serde_json::from_value(payload.clone())
        .map_err(|error| invalid_tail(format!("replace-v1 decode failed: {error}")))?;
    if replacement.v != crate::mutation::REPLACE_MUTATION_VERSION {
        return Err(invalid_tail(format!(
            "unsupported replace mutation version {}",
            replacement.v
        )));
    }
    if replacement.retracts.is_empty() {
        return Err(invalid_tail("replace mutation has no displaced edge"));
    }
    for retract in &replacement.retracts {
        retract_triple_ids(retract)?;
    }
    // Decode the complete assertion before any retraction is applied. This is
    // the replay-side half of the compound transition's atomicity contract.
    parse_replay_complete_assert(&replacement.assertion)?;
    Ok(replacement)
}

fn parse_retract_edge(edge: &ReplayRetractEdge) -> SnapshotResult<Triple> {
    let ids = retract_triple_ids(&ReplayRetract {
        subject: edge.subject,
        predicate: edge.predicate,
        object: edge.object,
    })?;
    if !edge.confidence.is_finite() || !(0.0..=1.0).contains(&edge.confidence) {
        return Err(invalid_tail(format!(
            "retract batch confidence {} is outside [0,1]",
            edge.confidence
        )));
    }
    if let Some(compartment) = edge.compartment.as_deref() {
        validate_compartment_id(compartment)?;
    }
    if edge.provenance_id == Some(0) {
        return Err(invalid_tail("retract batch provenance ID must be non-zero"));
    }
    let mut triple = Triple::new(ids.0, ids.1, ids.2);
    triple.confidence = edge.confidence;
    triple.timestamp = edge.timestamp;
    triple.compartment_id = edge.compartment.clone();
    triple.provenance_id = edge.provenance_id;
    Ok(triple)
}

fn parse_replay_complete_retract_batch(
    payload: &serde_json::Value,
) -> SnapshotResult<(
    u64,
    Vec<Triple>,
    Vec<SymbolMeta>,
    Vec<crate::provenance::ProvenanceRecord>,
)> {
    let batch: ReplayCompleteRetractBatch = serde_json::from_value(payload.clone())
        .map_err(|error| invalid_tail(format!("retract-batch decode failed: {error}")))?;
    if !matches!(batch.v, 1 | crate::mutation::RETRACT_BATCH_MUTATION_VERSION) {
        return Err(invalid_tail(format!(
            "unsupported retract batch mutation version {}",
            batch.v
        )));
    }
    if batch.retracts.is_empty() {
        return Err(invalid_tail("retract batch has no removed edge"));
    }

    let mut metadata_ids = std::collections::HashMap::new();
    let mut metadata_labels = std::collections::HashMap::new();
    for meta in &batch.symbols {
        if is_reserved_symbol_id(meta.id.get())
            || metadata_ids.insert(meta.id.get(), meta).is_some()
        {
            return Err(invalid_tail(format!(
                "duplicate/reserved retract symbol metadata ID {}",
                meta.id.get()
            )));
        }
        let normalized = crate::registry::normalize_label_key(&meta.label);
        if let Some(previous) = metadata_labels.insert(normalized, meta.id.get())
            && previous != meta.id.get()
        {
            return Err(invalid_tail(format!(
                "retract symbol label {} is assigned to IDs {previous} and {}",
                meta.label,
                meta.id.get()
            )));
        }
    }

    let triples = batch
        .retracts
        .iter()
        .map(parse_retract_edge)
        .collect::<SnapshotResult<Vec<_>>>()?;
    let mut keys = std::collections::HashSet::new();
    for triple in &triples {
        if !keys.insert((triple.subject, triple.predicate, triple.object)) {
            return Err(invalid_tail("retract batch contains a duplicate edge"));
        }
        for id in [triple.subject, triple.predicate, triple.object] {
            if !metadata_ids.contains_key(&id.get()) {
                return Err(invalid_tail(format!(
                    "retract batch is missing full metadata for symbol {}",
                    id.get()
                )));
            }
        }
    }

    let referenced_provenance = triples
        .iter()
        .filter_map(|triple| triple.provenance_id)
        .collect::<std::collections::BTreeSet<_>>();
    let mut provenance_ids = std::collections::BTreeSet::new();
    for record in &batch.provenance {
        let id = record
            .id
            .ok_or_else(|| invalid_tail("retract batch provenance record has no ID"))?
            .get();
        if !provenance_ids.insert(id) {
            return Err(invalid_tail(format!(
                "retract batch contains duplicate provenance ID {id}"
            )));
        }
        if !record.confidence.is_finite() || !(0.0..=1.0).contains(&record.confidence) {
            return Err(invalid_tail(format!(
                "retract batch provenance {id} confidence {} is outside [0,1]",
                record.confidence
            )));
        }
        let mut required = vec![record.derived_id];
        required.extend(record.sources.iter().copied());
        required.extend(record.kind.referenced_symbol_ids());
        for symbol in required {
            if !metadata_ids.contains_key(&symbol.get()) {
                return Err(invalid_tail(format!(
                    "retract batch provenance {id} is missing metadata for symbol {}",
                    symbol.get()
                )));
            }
        }
    }
    if batch.v == 1 {
        if !batch.provenance.is_empty() {
            return Err(invalid_tail(
                "legacy retract batch must not carry a V2 provenance envelope",
            ));
        }
    } else if referenced_provenance != provenance_ids {
        return Err(invalid_tail(format!(
            "retract batch provenance envelope does not exactly close referenced IDs: referenced {referenced_provenance:?}, enclosed {provenance_ids:?}"
        )));
    }
    for triple in &triples {
        if let Some(raw_id) = triple.provenance_id
            && let Some(record) = batch
                .provenance
                .iter()
                .find(|record| record.id.is_some_and(|id| id.get() == raw_id))
        {
            validate_rule_application_payload(
                record,
                (triple.subject, triple.predicate, triple.object),
            )?;
        }
    }
    Ok((batch.v, triples, batch.symbols, batch.provenance))
}

fn parse_replay_complete_assert(
    payload: &serde_json::Value,
) -> SnapshotResult<(Triple, Vec<SymbolMeta>, Option<crate::provenance::ProvenanceRecord>)> {
    let envelope: ReplayCompleteAssert = serde_json::from_value(payload.clone())
        .map_err(|error| invalid_tail(format!("assert-v2 decode failed: {error}")))?;
    if envelope.v != crate::mutation::ASSERT_MUTATION_VERSION {
        return Err(invalid_tail(format!(
            "unsupported assert mutation version {}",
            envelope.v
        )));
    }
    if !envelope.confidence.is_finite() || !(0.0..=1.0).contains(&envelope.confidence) {
        return Err(invalid_tail(format!(
            "assert confidence {} is outside [0,1]",
            envelope.confidence
        )));
    }
    if let Some(compartment_id) = envelope.compartment.as_deref() {
        validate_compartment_id(compartment_id)
            .map_err(|error| invalid_tail(error.to_string()))?;
    }
    let subject = SymbolId::new(envelope.subject)
        .filter(|id| !is_reserved_symbol_id(id.get()))
        .ok_or_else(|| invalid_tail("assert subject uses zero/reserved symbol ID"))?;
    let predicate = SymbolId::new(envelope.predicate)
        .filter(|id| !is_reserved_symbol_id(id.get()))
        .ok_or_else(|| invalid_tail("assert predicate uses zero/reserved symbol ID"))?;
    let object = SymbolId::new(envelope.object)
        .filter(|id| !is_reserved_symbol_id(id.get()))
        .ok_or_else(|| invalid_tail("assert object uses zero/reserved symbol ID"))?;

    match (envelope.provenance_id, envelope.provenance.as_ref()) {
        (None, None) => {}
        (Some(raw_id), Some(record)) => {
            let id = crate::provenance::ProvenanceId::new(raw_id).ok_or_else(|| {
                invalid_tail("assert provenance ID must be non-zero")
            })?;
            if record.id != Some(id) {
                return Err(invalid_tail(format!(
                    "assert provenance ID {raw_id} does not match record {:?}",
                    record.id.map(crate::provenance::ProvenanceId::get)
                )));
            }
            if !record.confidence.is_finite() || !(0.0..=1.0).contains(&record.confidence) {
                return Err(invalid_tail(format!(
                    "provenance {raw_id} confidence {} is outside [0,1]",
                    record.confidence
                )));
            }
            validate_rule_application_payload(record, (subject, predicate, object))?;
        }
        _ => {
            return Err(invalid_tail(
                "assert provenance ID and full record must be present together",
            ));
        }
    }

    let mut metadata_ids = std::collections::HashMap::new();
    let mut metadata_labels = std::collections::HashMap::new();
    for meta in &envelope.symbols {
        if is_reserved_symbol_id(meta.id.get())
            || metadata_ids.insert(meta.id.get(), meta).is_some()
        {
            return Err(invalid_tail(format!(
                "duplicate/reserved symbol metadata ID {}",
                meta.id.get()
            )));
        }
        let normalized = crate::registry::normalize_label_key(&meta.label);
        if let Some(previous) = metadata_labels.insert(normalized, meta.id.get())
            && previous != meta.id.get()
        {
            return Err(invalid_tail(format!(
                "symbol label {} is assigned to IDs {previous} and {}",
                meta.label,
                meta.id.get()
            )));
        }
    }
    let mut required = vec![subject, predicate, object];
    if let Some(record) = &envelope.provenance {
        required.push(record.derived_id);
        required.extend(record.sources.iter().copied());
        required.extend(record.kind.referenced_symbol_ids());
    }
    required.sort_unstable();
    required.dedup();
    for id in required {
        if !metadata_ids.contains_key(&id.get()) {
            return Err(invalid_tail(format!(
                "assert is missing full metadata for symbol {}",
                id.get()
            )));
        }
    }

    let mut triple = Triple::new(subject, predicate, object);
    triple.confidence = envelope.confidence;
    triple.timestamp = envelope.timestamp;
    triple.compartment_id = envelope.compartment;
    triple.provenance_id = envelope.provenance_id;
    Ok((triple, envelope.symbols, envelope.provenance))
}

fn merge_tail_symbols(
    incoming: Vec<SymbolMeta>,
    symbols: &mut Vec<SymbolMeta>,
    symbol_positions: &mut std::collections::HashMap<u64, usize>,
    labels: &mut std::collections::HashMap<String, u64>,
    tail_symbol_ids: &mut std::collections::HashSet<u64>,
) -> SnapshotResult<()> {
    for meta in incoming {
        let normalized = crate::registry::normalize_label_key(&meta.label);
        if let Some(other_id) = labels.get(&normalized)
            && *other_id != meta.id.get()
        {
            return Err(invalid_tail(format!(
                "symbol label {} conflicts with restored ID {other_id}",
                meta.label
            )));
        }
        if let Some(position) = symbol_positions.get(&meta.id.get()).copied() {
            let existing = &symbols[position];
            if existing.kind != meta.kind || existing.label != meta.label {
                return Err(invalid_tail(format!(
                    "symbol ID {} conflicts with restored metadata",
                    meta.id.get()
                )));
            }
            if tail_symbol_ids.contains(&meta.id.get())
                && (existing.created_at != meta.created_at || existing.source != meta.source)
            {
                return Err(invalid_tail(format!(
                    "symbol ID {} has conflicting replay metadata",
                    meta.id.get()
                )));
            }
            // The WAL carries full metadata fields that the legacy relational
            // symbol row does not. Prefer it after identity/kind verification.
            symbols[position] = meta;
            tail_symbol_ids.insert(symbols[position].id.get());
        } else {
            let position = symbols.len();
            labels.insert(normalized, meta.id.get());
            symbol_positions.insert(meta.id.get(), position);
            tail_symbol_ids.insert(meta.id.get());
            symbols.push(meta);
        }
    }
    Ok(())
}

/// Merge every replay prerequisite into the cold-restore image before opening
/// the engine. Facts are still replayed in WAL order afterwards so retract/TMS
/// behavior stays unchanged.
fn prepare_tail_restore(
    tail: &[akhomed_store::TailMutation],
    symbols: &mut Vec<SymbolMeta>,
    provenance: &mut Vec<crate::provenance::ProvenanceRecord>,
) -> SnapshotResult<u64> {
    let mut symbol_positions = std::collections::HashMap::with_capacity(symbols.len());
    let mut labels = std::collections::HashMap::with_capacity(symbols.len());
    for (position, meta) in symbols.iter().enumerate() {
        symbol_positions.insert(meta.id.get(), position);
        labels.insert(crate::registry::normalize_label_key(&meta.label), meta.id.get());
    }
    let mut provenance_by_id = std::collections::HashMap::with_capacity(provenance.len());
    for (position, record) in provenance.iter().enumerate() {
        let id = record
            .id
            .ok_or_else(|| invalid_tail("snapshot provenance record is missing its ID"))?
            .get();
        if provenance_by_id.insert(id, position).is_some() {
            return Err(invalid_tail(format!("duplicate provenance ID {id}")));
        }
    }
    let mut tail_symbol_ids = std::collections::HashSet::new();

    for mutation in tail {
        let (tail_symbols, tail_records) = if mutation.kind == "retract_batch" {
            let (_, _, tail_symbols, tail_records) =
                parse_replay_complete_retract_batch(&mutation.payload)?;
            (tail_symbols, tail_records)
        } else {
            let assertion = match mutation.kind.as_str() {
                "assert" => std::borrow::Cow::Borrowed(&mutation.payload),
                "replace" => std::borrow::Cow::Owned(
                    parse_replay_complete_replace(&mutation.payload)?.assertion,
                ),
                _ => continue,
            };
            match assertion.get("v") {
                None => continue,
                Some(version)
                    if version.as_u64() == Some(crate::mutation::ASSERT_MUTATION_VERSION) => {}
                Some(version) => {
                    return Err(invalid_tail(format!(
                        "unsupported assert mutation version {version}"
                    )));
                }
            }
            let (_, tail_symbols, tail_record) = parse_replay_complete_assert(&assertion)?;
            (tail_symbols, tail_record.into_iter().collect())
        };
        merge_tail_symbols(
            tail_symbols,
            symbols,
            &mut symbol_positions,
            &mut labels,
            &mut tail_symbol_ids,
        )?;
        for record in tail_records {
            let id = record.id.expect("validated versioned provenance ID").get();
            if let Some(position) = provenance_by_id.get(&id).copied() {
                let existing = bincode::serialize(&provenance[position])
                    .map_err(|error| invalid_tail(format!("encode provenance {id}: {error}")))?;
                let incoming = bincode::serialize(&record)
                    .map_err(|error| invalid_tail(format!("encode provenance {id}: {error}")))?;
                if existing != incoming {
                    return Err(invalid_tail(format!(
                        "provenance ID {id} has conflicting records"
                    )));
                }
            } else {
                provenance_by_id.insert(id, provenance.len());
                provenance.push(record);
            }
        }
    }

    // Legacy assertions did not carry registry records. They remain replayable
    // only when every referenced symbol is already known from the snapshot or
    // from a later replay-complete envelope; otherwise waking would create an
    // unresolvable ID and risk allocator reuse.
    for mutation in tail {
        let referenced = match mutation.kind.as_str() {
            "assert" if mutation.payload.get("v").is_none() => {
                vec![payload_triple_ids(&mutation.payload)?]
            }
            "retract" => vec![payload_triple_ids(&mutation.payload)?],
            "replace" => parse_replay_complete_replace(&mutation.payload)?
                .retracts
                .iter()
                .map(retract_triple_ids)
                .collect::<SnapshotResult<Vec<_>>>()?,
            "retract_batch" => parse_replay_complete_retract_batch(&mutation.payload)?
                .1
                .into_iter()
                .map(|triple| (triple.subject, triple.predicate, triple.object))
                .collect(),
            _ => continue,
        };
        for (subject, predicate, object) in referenced {
            for id in [subject, predicate, object] {
                if !symbol_positions.contains_key(&id.get()) {
                    return Err(invalid_tail(format!(
                        "{} references unknown symbol {}",
                        mutation.kind,
                        id.get()
                    )));
                }
            }
        }
    }

    let max_id = symbols.iter().map(|meta| meta.id.get()).max().unwrap_or(0);
    let next = max_id.checked_add(1).ok_or_else(|| {
        invalid_tail("post-snapshot symbol ID space is exhausted")
    })?;
    if is_reserved_symbol_id(next) {
        return Err(invalid_tail(
            "post-snapshot symbol allocator would enter the reserved sentinel",
        ));
    }
    Ok(next.max(1))
}

fn validate_fact_symbols(facts: &[fact::Model], symbols: &[SymbolMeta]) -> SnapshotResult<()> {
    let ids: std::collections::HashSet<u64> =
        symbols.iter().map(|meta| meta.id.get()).collect();
    for fact in facts {
        if !fact.confidence.is_finite() || !(0.0..=1.0).contains(&fact.confidence) {
            return Err(SnapshotError::Agent(format!(
                "wake: fact ({}, {}, {}) confidence {} is outside [0,1]",
                fact.subject, fact.predicate, fact.object, fact.confidence
            )));
        }
        if let Some(compartment_id) = fact.compartment.as_deref() {
            validate_compartment_id(compartment_id)?;
        }
        for (field, raw) in [
            ("subject", fact.subject),
            ("predicate", fact.predicate),
            ("object", fact.object),
        ] {
            let id = u64::try_from(raw)
                .ok()
                .and_then(SymbolId::new)
                .filter(|id| !is_reserved_symbol_id(id.get()))
                .ok_or_else(|| {
                    SnapshotError::Agent(format!(
                        "wake: fact {field} has invalid symbol ID {raw}"
                    ))
                })?;
            if !ids.contains(&id.get()) {
                return Err(SnapshotError::Agent(format!(
                    "wake: fact {field} references missing symbol {}",
                    id.get()
                )));
            }
        }
    }
    Ok(())
}

fn validate_provenance_symbols(
    records: &[crate::provenance::ProvenanceRecord],
    symbols: &[SymbolMeta],
) -> SnapshotResult<()> {
    let ids: std::collections::HashSet<SymbolId> =
        symbols.iter().map(|meta| meta.id).collect();
    let mut provenance_ids = std::collections::HashSet::with_capacity(records.len());
    for record in records {
        let provenance_id = record
            .id
            .ok_or_else(|| SnapshotError::Agent("wake: provenance record has no ID".into()))?
            .get();
        if !provenance_ids.insert(provenance_id) {
            return Err(SnapshotError::Agent(format!(
                "wake: duplicate provenance ID {provenance_id}"
            )));
        }
        if !record.confidence.is_finite() || !(0.0..=1.0).contains(&record.confidence) {
            return Err(SnapshotError::Agent(format!(
                "wake: provenance {provenance_id} confidence {} is outside [0,1]",
                record.confidence
            )));
        }
        for (field, id) in std::iter::once(("derived", record.derived_id))
            .chain(record.sources.iter().copied().map(|id| ("source", id)))
            .chain(
                record
                    .kind
                    .referenced_symbol_ids()
                    .into_iter()
                    .map(|id| ("derivation", id)),
            )
        {
            if !ids.contains(&id) {
                return Err(SnapshotError::Agent(format!(
                    "wake: provenance {provenance_id} {field} references missing symbol {}",
                    id.get()
                )));
            }
        }
    }
    Ok(())
}

/// Convert a persisted `facts` row into a KG [`Triple`].
fn fact_to_triple(row: &fact::Model) -> SnapshotResult<Triple> {
    let convert = |field: &str, raw: i64| {
        u64::try_from(raw)
            .ok()
            .and_then(SymbolId::new)
            .filter(|id| !is_reserved_symbol_id(id.get()))
            .ok_or_else(|| {
                SnapshotError::Agent(format!(
                    "wake: fact {field} has invalid symbol ID {raw}"
                ))
            })
    };
    let s = convert("subject", row.subject)?;
    let p = convert("predicate", row.predicate)?;
    let o = convert("object", row.object)?;
    let mut t = Triple::new(s, p, o);
    t.confidence = row.confidence;
    t.timestamp = row.asserted_at.timestamp().max(0) as u64;
    t.provenance_id = row
        .provenance_id
        .map(|raw| {
            let id = u64::try_from(raw).map_err(|_| {
                SnapshotError::Agent(format!("wake: fact has invalid provenance ID {raw}"))
            })?;
            crate::provenance::ProvenanceId::new(id)
                .map(crate::provenance::ProvenanceId::get)
                .ok_or_else(|| {
                    SnapshotError::Agent(format!("wake: fact has invalid provenance ID {raw}"))
                })
        })
        .transpose()?;
    if let Some(compartment_id) = row.compartment.as_deref() {
        validate_compartment_id(compartment_id)?;
    }
    t.compartment_id = row.compartment.clone();
    Ok(t)
}

fn validate_replay_symbols_registered(
    engine: &Engine,
    triple: (SymbolId, SymbolId, SymbolId),
    operation: &str,
) -> SnapshotResult<()> {
    for id in [triple.0, triple.1, triple.2] {
        engine.get_symbol_meta(id).map_err(|error| {
            invalid_tail(format!(
                "{operation} references unrestored symbol {}: {error}",
                id.get()
            ))
        })?;
    }
    Ok(())
}

/// Fully validate one assertion against the already-prepared restore image,
/// without mutating the graph. Compound replacement replay calls this before
/// removing any displaced edge.
fn prepare_assert_for_replay(
    engine: &Engine,
    payload: &serde_json::Value,
) -> SnapshotResult<Triple> {
    let triple = if payload.get("v").is_some() {
        let (triple, _, record) = parse_replay_complete_assert(payload)?;
        if let Some(record) = record {
            let id = record.id.expect("validated provenance ID");
            let restored = engine.get_provenance(id).map_err(|error| {
                invalid_tail(format!(
                    "assert references unrestored provenance {}: {error}",
                    id.get()
                ))
            })?;
            let restored = bincode::serialize(&restored).map_err(|error| {
                invalid_tail(format!("encode restored provenance {}: {error}", id.get()))
            })?;
            let expected = bincode::serialize(&record).map_err(|error| {
                invalid_tail(format!("encode assertion provenance {}: {error}", id.get()))
            })?;
            if restored != expected {
                return Err(invalid_tail(format!(
                    "assert provenance {} changed before replay",
                    id.get()
                )));
            }
        }
        triple
    } else {
        let (subject, predicate, object) = payload_triple_ids(payload)?;
        let mut triple = Triple::new(subject, predicate, object);
        triple.timestamp = payload
            .get("timestamp")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        if let Some(confidence) = payload.get("confidence") {
            let confidence = confidence
                .as_f64()
                .ok_or_else(|| invalid_tail("legacy assert confidence is not numeric"))?
                as f32;
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                return Err(invalid_tail("legacy assert confidence is outside [0,1]"));
            }
            triple.confidence = confidence;
        }
        match payload.get("compartment") {
            None | Some(serde_json::Value::Null) => {}
            Some(serde_json::Value::String(value)) => {
                validate_compartment_id(value).map_err(|error| invalid_tail(error.to_string()))?;
                triple.compartment_id = Some(value.clone())
            }
            Some(_) => return Err(invalid_tail("legacy assert compartment is not text")),
        }
        triple
    };
    validate_replay_symbols_registered(
        engine,
        (triple.subject, triple.predicate, triple.object),
        "assert",
    )?;
    Ok(triple)
}

/// Apply one replayed WAL mutation. Returns whether it was applied. Unknown
/// future kinds are ignored, while malformed known KG mutations fail wake so
/// corruption cannot silently drop a fact. Microtheory/compartment replay is
/// deferred (38d). `kind="turn"` rows (ADR 091) are explicitly skipped — raw
/// transcript-capture staging, never a KG mutation.
fn apply_mutation(engine: &Engine, m: &akhomed_store::TailMutation) -> SnapshotResult<bool> {
    match m.kind.as_str() {
        "assert" => {
            let triple = prepare_assert_for_replay(engine, &m.payload)?;
            engine
                .replay_assert_exact(&triple)
                .map_err(|error| SnapshotError::Engine(error.to_string()))?;
            Ok(true)
        }
        "retract" => {
            let triple = payload_triple_ids(&m.payload)?;
            validate_replay_symbols_registered(engine, triple, "retract")?;
            engine
                .remove_triple(triple.0, triple.1, triple.2)
                .map_err(|error| SnapshotError::Engine(error.to_string()))?;
            Ok(true)
        }
        "retract_batch" => {
            // Decode and validate the complete batch before applying its first
            // edge. `prepare_tail_restore` has already merged the closed symbol
            // envelope into the registry, so every reference must resolve.
            let (_, retracts, _, _) = parse_replay_complete_retract_batch(&m.payload)?;
            for triple in &retracts {
                validate_replay_symbols_registered(
                    engine,
                    (triple.subject, triple.predicate, triple.object),
                    "retract batch",
                )?;
                let actual = engine.knowledge_graph().triple(
                    triple.subject,
                    triple.predicate,
                    triple.object,
                );
                // Legacy V1 batches did not enclose provenance. If the edge is
                // already absent, the exact retraction is idempotently safe and
                // does not need that record. A present edge still requires its
                // provenance and exact metadata before any removal begins.
                if actual.is_some()
                    && let Some(raw_id) = triple.provenance_id
                {
                    let id = crate::provenance::ProvenanceId::new(raw_id)
                        .ok_or_else(|| invalid_tail("retract batch provenance ID must be non-zero"))?;
                    engine.get_provenance(id).map_err(|error| {
                        invalid_tail(format!(
                            "retract batch references unavailable provenance ID {raw_id}: {error}"
                        ))
                    })?;
                }
                if let Some(actual) = actual
                    && actual != *triple
                {
                    return Err(invalid_tail(format!(
                        "retract batch edge ({}, {}, {}) metadata does not match replay state",
                        triple.subject.get(),
                        triple.predicate.get(),
                        triple.object.get()
                    )));
                }
            }
            for triple in &retracts {
                engine
                    .replay_retract_edge_exact(triple)
                    .map_err(|error| SnapshotError::Engine(error.to_string()))?;
            }
            Ok(true)
        }
        "replace" => {
            let replacement = parse_replay_complete_replace(&m.payload)?;
            let retracts = replacement
                .retracts
                .iter()
                .map(retract_triple_ids)
                .collect::<SnapshotResult<Vec<_>>>()?;
            for triple in &retracts {
                validate_replay_symbols_registered(engine, *triple, "replace retract")?;
            }
            let assertion = prepare_assert_for_replay(engine, &replacement.assertion)?;

            for (subject, predicate, object) in retracts {
                engine.replay_retract_exact(subject, predicate, object);
            }
            engine
                .replay_assert_exact(&assertion)
                .map_err(|error| SnapshotError::Engine(error.to_string()))?;
            Ok(true)
        }
        // ADR 091 decision 3: `turn` rows are raw transcript-capture staging,
        // never KG mutations — the KG facts derived from a turn (if any) get
        // their own `assert` rows. Explicitly skip rather than falling through
        // to the catch-all, so the intent is visible at the call site. `false`
        // just means "not counted in `replayed`" here (see the caller at
        // `wake`, ~line 690-696) — it is NOT treated as an error or an unknown
        // kind; wake proceeds normally with `turn` rows present.
        "turn" => Ok(false),
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::agent::drives::DriveKind;
    use crate::agent::goal::{GoalSource, GoalStatus};
    use crate::mutation::{MutationKind, MutationRecord, MutationSink};

    #[derive(Default)]
    struct RecordingSink(Mutex<Vec<MutationRecord>>);

    impl MutationSink for RecordingSink {
        fn record(&self, mutation: MutationRecord) {
            self.0.lock().unwrap().push(mutation);
        }
    }

    fn replay_symbol(id: u64, kind: SymbolKind, label: &str, created_at: u64) -> SymbolMeta {
        SymbolMeta {
            id: SymbolId::new(id).unwrap(),
            kind,
            label: label.into(),
            created_at,
            source: None,
        }
    }

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
        let model = goal_to_model(akh, &goal).unwrap();
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
            goal_hierarchy::Model {
                akh_id: Uuid::nil(),
                parent_id: 1,
                child_id: 30,
                position: 2,
            },
            goal_hierarchy::Model {
                akh_id: Uuid::nil(),
                parent_id: 1,
                child_id: 10,
                position: 0,
            },
            goal_hierarchy::Model {
                akh_id: Uuid::nil(),
                parent_id: 1,
                child_id: 20,
                position: 1,
            },
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
            assert_eq!(
                symbol_kind_from(&s, g),
                kind,
                "kind {kind:?} must round-trip"
            );
        }
    }

    #[test]
    fn provenance_snapshot_row_roundtrips_typed_record() {
        let sentinel = SymbolId::new(crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL).unwrap();
        let source = SymbolId::new(11).unwrap();
        let predicate = SymbolId::new(12).unwrap();
        let object = SymbolId::new(13).unwrap();
        let kind = crate::provenance::DerivationKind::RuleInference {
            rule_name: "roundtrip".into(),
            antecedents: vec![sentinel, source, predicate, object],
        };
        let row = provenance_snapshot::Model {
            akh_id: Uuid::nil(),
            provenance_id: 17,
            workspace_id: Uuid::nil(),
            derived_symbol: 19,
            sources: serde_json::to_value(vec![source, object]).unwrap(),
            derivation: serde_json::to_value(&kind).unwrap(),
            confidence: 0.82,
            depth: 3,
            created_at: chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
        };

        let restored = provenance_row_to_record(&row).unwrap();
        assert_eq!(restored.id.unwrap().get(), 17);
        assert_eq!(restored.derived_id.get(), 19);
        assert_eq!(restored.sources, vec![source, object]);
        assert_eq!(restored.kind, kind);
        assert_eq!(restored.confidence, 0.82);
        assert_eq!(restored.depth, 3);
        assert_eq!(restored.timestamp, 1_700_000_000);

        let mut malformed = row;
        malformed.sources = serde_json::json!({"not": "a symbol list"});
        assert!(provenance_row_to_record(&malformed).is_err());
    }

    #[test]
    fn snapshot_manifest_rejects_a_dangling_fact_provenance_link() {
        let dangling = fact::Model {
            workspace_id: Uuid::nil(),
            subject: 1,
            predicate: 2,
            object: 3,
            akh_id: Uuid::nil(),
            confidence: 1.0,
            asserted_at: chrono::DateTime::from_timestamp(1, 0).unwrap(),
            retracted_at: None,
            provenance_id: Some(42),
            compartment: None,
        };
        assert!(
            validate_fact_provenance_links(&[dangling], &[]).is_err(),
            "capture's manifest validation must reject a dangling provenance link"
        );
    }

    #[test]
    fn snapshot_manifest_rejects_unregistered_nested_provenance_symbols() {
        let derived = replay_symbol(30, SymbolKind::Entity, "Derived", 0);
        let predicate = replay_symbol(31, SymbolKind::Relation, "edge", 0);
        let missing = SymbolId::new(32).unwrap();
        let mut record = crate::provenance::ProvenanceRecord::new(
            derived.id,
            crate::provenance::DerivationKind::GraphEdge {
                from: missing,
                predicate: predicate.id,
            },
        );
        record.id = crate::provenance::ProvenanceId::new(1);
        assert!(
            validate_provenance_symbols(&[record], &[derived, predicate]).is_err(),
            "capture must reject a ledger whose typed derivation payload is outside the registry"
        );
    }

    #[test]
    fn deferred_restore_loads_compartments_after_the_ledger_watermark() {
        let root = tempfile::tempdir().unwrap();
        let pack_dir = root.path().join("compartments").join("restore-pack");
        std::fs::create_dir_all(&pack_dir).unwrap();
        std::fs::write(
            pack_dir.join("compartment.toml"),
            r#"id = "restore-pack"
name = "Restore Pack"
kind = "Workspace"
scope = "Private"
triples_file = "triples.json"
"#,
        )
        .unwrap();
        std::fs::write(
            pack_dir.join("triples.json"),
            r#"[{"subject":"DeferredSeed","predicate":"belongs-to","object":"RestorePack"}]"#,
        )
        .unwrap();

        // Production cold wake always publishes the validated relational
        // allocator before opening the restore-only engine.
        {
            let store = TieredStore::with_persistence(root.path(), "symbols").unwrap();
            put_bincode_required(&store, b"sym_allocator_next", &1_u64).unwrap();
        }

        let engine = Engine::new_for_restore(EngineConfig {
            data_dir: Some(root.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        assert!(engine.resolve_symbol("DeferredSeed").is_err());
        engine.replace_all_provenance(&[]).unwrap();
        engine.finish_deferred_startup_loads();

        let subject = engine.lookup_symbol("DeferredSeed").unwrap();
        let triple = engine.triples_from(subject).pop().unwrap();
        let provenance_id = triple.provenance_id.expect("tracked compartment edge");
        assert!(matches!(
            engine
                .get_provenance(crate::provenance::ProvenanceId::new(provenance_id).unwrap())
                .unwrap()
                .kind,
            crate::provenance::DerivationKind::CompartmentLoaded { .. }
        ));
    }

    #[test]
    fn cold_restore_ignores_a_stale_host_local_graph() {
        let root = tempfile::tempdir().unwrap();
        {
            let stale = Engine::new(EngineConfig {
                data_dir: Some(root.path().to_path_buf()),
                ..Default::default()
            })
            .unwrap();
            stale
                .ingest_label_triples(&[(
                    "StaleSubject".into(),
                    "stale-predicate".into(),
                    "StaleObject".into(),
                    1.0,
                )])
                .unwrap();
            stale.persist().unwrap();
        }

        let restoring = Engine::new_for_restore(EngineConfig {
            data_dir: Some(root.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        assert!(
            restoring.all_triples().is_empty(),
            "the relational wake snapshot, not a reused host cache, is authoritative"
        );
    }

    #[test]
    fn legacy_snapshot_clears_only_unverifiable_provenance_links() {
        let mut record = crate::provenance::ProvenanceRecord::new(
            SymbolId::new(3).unwrap(),
            crate::provenance::DerivationKind::Seed,
        );
        record.id = crate::provenance::ProvenanceId::new(17);
        let timestamp = chrono::DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let mut facts = vec![
            fact::Model {
                workspace_id: Uuid::nil(),
                subject: 1,
                predicate: 2,
                object: 3,
                akh_id: Uuid::nil(),
                confidence: 1.0,
                asserted_at: timestamp,
                retracted_at: None,
                provenance_id: Some(17),
                compartment: None,
            },
            fact::Model {
                workspace_id: Uuid::nil(),
                subject: 4,
                predicate: 2,
                object: 5,
                akh_id: Uuid::nil(),
                confidence: 1.0,
                asserted_at: timestamp,
                retracted_at: None,
                provenance_id: Some(99),
                compartment: None,
            },
        ];

        assert_eq!(
            clear_unverifiable_legacy_provenance_links(&mut facts, &[record]),
            1
        );
        assert_eq!(facts[0].provenance_id, Some(17));
        assert_eq!(facts[1].provenance_id, None);
    }

    #[test]
    fn replay_complete_tail_restores_fresh_symbols_provenance_and_exact_reassertion() {
        use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};

        let root = tempfile::tempdir().unwrap();
        let subject = replay_symbol(700, SymbolKind::Entity, "TailSubject", 101);
        let predicate = replay_symbol(701, SymbolKind::Relation, "tail-predicate", 102);
        let object = replay_symbol(702, SymbolKind::Entity, "TailObject", 103);
        let mut source = replay_symbol(703, SymbolKind::Entity, "TailSource", 104);
        source.source = Some(crate::symbol::SourceRef {
            document_id: "post-snapshot-document".into(),
            chunk_index: 4,
            byte_start: 8,
            byte_end: 21,
        });
        let symbols = vec![
            subject.clone(),
            predicate.clone(),
            object.clone(),
            source.clone(),
        ];

        let mut record = ProvenanceRecord::new(object.id, DerivationKind::Extracted)
            .with_sources(vec![source.id])
            .with_confidence(0.83)
            .with_depth(2);
        record.id = ProvenanceId::new(41);
        record.timestamp = 10;

        let tracked = akhomed_store::TailMutation {
            kind: "assert".into(),
            payload: serde_json::json!({
                "v": crate::mutation::ASSERT_MUTATION_VERSION,
                "subject": subject.id.get(),
                "predicate": predicate.id.get(),
                "object": object.id.get(),
                "confidence": 0.83,
                "timestamp": 10,
                "compartment": "tail-pack",
                "provenance_id": 41,
                "provenance": record,
                "symbols": symbols,
            }),
        };
        let untracked = akhomed_store::TailMutation {
            kind: "assert".into(),
            payload: serde_json::json!({
                "v": crate::mutation::ASSERT_MUTATION_VERSION,
                "subject": subject.id.get(),
                "predicate": predicate.id.get(),
                "object": object.id.get(),
                "confidence": 0.55,
                "timestamp": 20,
                "compartment": null,
                "provenance_id": null,
                "provenance": null,
                "symbols": [subject, predicate, object],
            }),
        };
        let tail = vec![tracked, untracked];
        let mut restored_symbols = Vec::new();
        let mut restored_provenance = Vec::new();
        let allocator_next =
            prepare_tail_restore(&tail, &mut restored_symbols, &mut restored_provenance).unwrap();
        assert_eq!(allocator_next, 704);
        assert_eq!(restored_provenance.len(), 1);

        {
            let store = TieredStore::with_persistence(root.path(), "symbols").unwrap();
            write_registry(&store, &restored_symbols).unwrap();
            put_bincode_required(&store, b"sym_allocator_next", &allocator_next).unwrap();
        }
        let engine = Engine::new_for_restore(EngineConfig {
            data_dir: Some(root.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        engine.replace_all_provenance(&restored_provenance).unwrap();
        for mutation in &tail {
            assert!(apply_mutation(&engine, mutation).unwrap());
        }

        let restored_source = engine.get_symbol_meta(source.id).unwrap();
        assert_eq!(restored_source.created_at, 104);
        assert_eq!(restored_source.source, source.source);
        assert_eq!(engine.get_provenance(ProvenanceId::new(41).unwrap()).unwrap().depth, 2);
        let edge = engine.triples_from(subject.id).pop().unwrap();
        assert_eq!(edge.confidence, 0.55);
        assert_eq!(edge.timestamp, 20);
        assert_eq!(edge.compartment_id, None);
        assert_eq!(edge.provenance_id, None, "the later untracked assertion is authoritative");

        let fresh = engine.create_symbol(SymbolKind::Entity, "AfterTail").unwrap();
        assert_eq!(fresh.id.get(), 704, "symbol allocation must advance beyond the WAL tail");
        let mut next_record = ProvenanceRecord::new(fresh.id, DerivationKind::Reasoned);
        assert_eq!(engine.store_provenance(&mut next_record).unwrap().get(), 42);
    }

    #[test]
    fn replay_complete_tail_requires_derivation_symbols_and_valid_compartments() {
        use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};

        let subject = replay_symbol(800, SymbolKind::Entity, "Subject", 1);
        let predicate = replay_symbol(801, SymbolKind::Relation, "predicate", 2);
        let object = replay_symbol(802, SymbolKind::Entity, "Object", 3);
        let missing_support = SymbolId::new(803).unwrap();
        let mut record = ProvenanceRecord::new(
            object.id,
            DerivationKind::GraphEdge {
                from: missing_support,
                predicate: predicate.id,
            },
        )
        .with_sources(vec![subject.id]);
        record.id = ProvenanceId::new(51);

        let missing_kind_symbol = akhomed_store::TailMutation {
            kind: "assert".into(),
            payload: serde_json::json!({
                "v": crate::mutation::ASSERT_MUTATION_VERSION,
                "subject": subject.id.get(),
                "predicate": predicate.id.get(),
                "object": object.id.get(),
                "confidence": 1.0,
                "timestamp": 1,
                "compartment": null,
                "provenance_id": 51,
                "provenance": record,
                "symbols": [subject.clone(), predicate.clone(), object.clone()],
            }),
        };
        assert!(
            prepare_tail_restore(
                &[missing_kind_symbol],
                &mut Vec::new(),
                &mut Vec::new()
            )
            .is_err(),
            "a provenance-kind symbol without metadata must fail closed"
        );

        let malformed_compartment = akhomed_store::TailMutation {
            kind: "assert".into(),
            payload: serde_json::json!({
                "v": crate::mutation::ASSERT_MUTATION_VERSION,
                "subject": subject.id.get(),
                "predicate": predicate.id.get(),
                "object": object.id.get(),
                "confidence": 1.0,
                "timestamp": 1,
                "compartment": "invalid compartment with spaces",
                "provenance_id": null,
                "provenance": null,
                "symbols": [subject, predicate, object],
            }),
        };
        assert!(
            prepare_tail_restore(
                &[malformed_compartment],
                &mut Vec::new(),
                &mut Vec::new()
            )
            .is_err(),
            "a malformed V2 compartment must return an error"
        );

        let malformed_fact = fact::Model {
            workspace_id: Uuid::nil(),
            subject: 800,
            predicate: 801,
            object: 802,
            akh_id: Uuid::nil(),
            confidence: 1.0,
            asserted_at: chrono::DateTime::from_timestamp(1, 0).unwrap(),
            retracted_at: None,
            provenance_id: None,
            compartment: Some("invalid compartment with spaces".into()),
        };
        assert!(
            fact_to_triple(&malformed_fact).is_err(),
            "a malformed relational compartment must return an error"
        );
    }

    #[test]
    fn complete_base_image_cannot_be_repaired_by_unrelated_tail_ids() {
        use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};

        let subject = replay_symbol(900, SymbolKind::Entity, "BaseSubject", 1);
        let predicate = replay_symbol(901, SymbolKind::Relation, "base-predicate", 1);
        let object = replay_symbol(902, SymbolKind::Entity, "BaseObject", 1);
        let tail_object = replay_symbol(903, SymbolKind::Entity, "TailObject", 2);
        let timestamp = chrono::DateTime::from_timestamp(10, 0).unwrap();
        let base_fact = fact::Model {
            workspace_id: Uuid::nil(),
            subject: subject.id.get() as i64,
            predicate: predicate.id.get() as i64,
            object: object.id.get() as i64,
            akh_id: Uuid::nil(),
            confidence: 1.0,
            asserted_at: timestamp,
            retracted_at: None,
            provenance_id: Some(61),
            compartment: None,
        };
        let mut symbols = vec![subject.clone(), predicate.clone(), object.clone()];
        let mut records = Vec::new();

        assert!(
            validate_base_restore_image(
                std::slice::from_ref(&base_fact),
                &records,
                &symbols,
                true,
            )
            .is_err(),
            "a complete base must reject its dangling link before tail merge"
        );

        let mut unrelated = ProvenanceRecord::new(tail_object.id, DerivationKind::Seed);
        unrelated.id = ProvenanceId::new(61);
        let tail = akhomed_store::TailMutation {
            kind: "assert".into(),
            payload: serde_json::json!({
                "v": crate::mutation::ASSERT_MUTATION_VERSION,
                "subject": subject.id.get(),
                "predicate": predicate.id.get(),
                "object": tail_object.id.get(),
                "confidence": 1.0,
                "timestamp": 11,
                "compartment": null,
                "provenance_id": 61,
                "provenance": unrelated,
                "symbols": [subject, predicate, tail_object],
            }),
        };
        prepare_tail_restore(&[tail], &mut symbols, &mut records).unwrap();
        assert!(
            validate_fact_provenance_links(&[base_fact], &records).is_ok(),
            "post-merge numeric validation alone would retroactively misbind the base fact"
        );
    }

    #[test]
    fn base_fact_symbols_are_closed_before_tail_metadata_is_admitted() {
        let missing_subject = replay_symbol(920, SymbolKind::Entity, "LateSubject", 2);
        let predicate = replay_symbol(921, SymbolKind::Relation, "base-predicate", 1);
        let object = replay_symbol(922, SymbolKind::Entity, "BaseObject", 1);
        let tail_object = replay_symbol(923, SymbolKind::Entity, "TailObject", 2);
        let base_fact = fact::Model {
            workspace_id: Uuid::nil(),
            subject: missing_subject.id.get() as i64,
            predicate: predicate.id.get() as i64,
            object: object.id.get() as i64,
            akh_id: Uuid::nil(),
            confidence: 1.0,
            asserted_at: chrono::DateTime::from_timestamp(10, 0).unwrap(),
            retracted_at: None,
            provenance_id: None,
            compartment: None,
        };
        let mut symbols = vec![predicate.clone(), object];
        assert!(
            validate_base_restore_image(
                std::slice::from_ref(&base_fact),
                &[],
                &symbols,
                true,
            )
            .is_err(),
            "tail metadata must not make an unknown base symbol valid"
        );

        let tail = akhomed_store::TailMutation {
            kind: "assert".into(),
            payload: serde_json::json!({
                "v": crate::mutation::ASSERT_MUTATION_VERSION,
                "subject": missing_subject.id.get(),
                "predicate": predicate.id.get(),
                "object": tail_object.id.get(),
                "confidence": 1.0,
                "timestamp": 11,
                "compartment": null,
                "provenance_id": null,
                "provenance": null,
                "symbols": [missing_subject, predicate, tail_object],
            }),
        };
        prepare_tail_restore(&[tail], &mut symbols, &mut Vec::new()).unwrap();
        assert!(validate_fact_symbols(&[base_fact], &symbols).is_ok());
    }

    #[test]
    fn replay_complete_v2_rule_record_must_bind_the_asserted_conclusion() {
        use crate::provenance::{DerivationKind, ProvenanceId, ProvenanceRecord};

        let subject = replay_symbol(940, SymbolKind::Entity, "ConclusionSubject", 1);
        let predicate = replay_symbol(941, SymbolKind::Relation, "concludes", 1);
        let object = replay_symbol(942, SymbolKind::Entity, "ConclusionObject", 1);
        let other = replay_symbol(943, SymbolKind::Entity, "OtherConclusion", 1);
        let support_subject = replay_symbol(944, SymbolKind::Entity, "SupportSubject", 1);
        let support_predicate = replay_symbol(945, SymbolKind::Relation, "supports", 1);
        let support_object = replay_symbol(946, SymbolKind::Entity, "SupportObject", 1);
        let mut record = ProvenanceRecord::new(
            object.id,
            DerivationKind::RuleInference {
                rule_name: "mismatched".into(),
                antecedents: vec![
                    SymbolId::new(crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL)
                        .unwrap(),
                    SymbolId::new(crate::provenance::RULE_APPLICATION_V2_SENTINEL).unwrap(),
                    subject.id,
                    predicate.id,
                    other.id,
                    support_subject.id,
                    support_predicate.id,
                    support_object.id,
                ],
            },
        );
        record.id = ProvenanceId::new(71);

        assert!(
            parse_replay_complete_assert(&serde_json::json!({
                "v": crate::mutation::ASSERT_MUTATION_VERSION,
                "subject": subject.id.get(),
                "predicate": predicate.id.get(),
                "object": object.id.get(),
                "confidence": 1.0,
                "timestamp": 1,
                "compartment": null,
                "provenance_id": 71,
                "provenance": record,
                "symbols": [
                    subject,
                    predicate,
                    object,
                    other,
                    support_subject,
                    support_predicate,
                    support_object,
                ],
            }))
            .is_err(),
            "a V2 rule record bound to another conclusion must fail closed"
        );
    }

    #[test]
    fn contradiction_replace_tail_survives_overflow_as_one_wake_transition() {
        let mut live = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let subject = live
            .create_symbol(SymbolKind::Entity, "replace-subject")
            .unwrap()
            .id;
        let predicate = live
            .create_symbol(SymbolKind::Relation, "replace-predicate")
            .unwrap()
            .id;
        let old_object = live
            .create_symbol(SymbolKind::Entity, "old-object")
            .unwrap()
            .id;
        let new_object = live
            .create_symbol(SymbolKind::Entity, "new-object")
            .unwrap()
            .id;
        live.functional_preds_mut().declare_functional(predicate);
        live.add_triple(&Triple::new(subject, predicate, old_object))
            .unwrap();

        // This is the captured base image. The Replace transition happens
        // after its watermark and must therefore be fully represented by the
        // tail: retract old, then assert new.
        let captured = live.all_triples();
        let mut phase9 = live.phase9_config().clone();
        phase9.contradiction_policy = crate::engine::ContradictionPolicy::Replace;
        live.set_phase9_config(phase9);
        let sink = Arc::new(RecordingSink::default());
        assert!(
            live.set_mutation_sink(Arc::clone(&sink) as Arc<dyn MutationSink>)
                .is_ok()
        );
        live.add_triple(&Triple::new(subject, predicate, new_object))
            .unwrap();

        let mutations = sink.0.lock().unwrap().clone();
        assert_eq!(mutations.len(), 1);
        assert_eq!(mutations[0].kind, MutationKind::Replace);

        // Force overflow with an unprotected row. The complete Replace entry
        // must remain intact rather than shedding only its displaced edge.
        let replacement_bytes = mutations[0].payload.to_string().len() + 64;
        let buffer = crate::wal::WalBuffer::new(replacement_bytes + 8);
        buffer.push(mutations[0].kind, mutations[0].payload.clone());
        buffer.push(
            MutationKind::Retract,
            serde_json::json!({"padding": "drop this unprotected overflow victim"}),
        );
        assert_eq!(buffer.dropped_on_overflow(), 1);
        let mutations = buffer.drain_records_for_test();
        assert_eq!(mutations.len(), 1);
        assert_eq!(mutations[0].kind, MutationKind::Replace);

        let restored = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            restored
                .create_symbol(SymbolKind::Entity, "replace-subject")
                .unwrap()
                .id,
            subject
        );
        assert_eq!(
            restored
                .create_symbol(SymbolKind::Relation, "replace-predicate")
                .unwrap()
                .id,
            predicate
        );
        assert_eq!(
            restored
                .create_symbol(SymbolKind::Entity, "old-object")
                .unwrap()
                .id,
            old_object
        );
        assert_eq!(
            restored
                .create_symbol(SymbolKind::Entity, "new-object")
                .unwrap()
                .id,
            new_object
        );
        restored.restore_triples(&captured).unwrap();
        for mutation in mutations {
            let tail = akhomed_store::TailMutation {
                kind: mutation.kind.as_str().into(),
                payload: mutation.payload,
            };
            assert!(apply_mutation(&restored, &tail).unwrap());
        }
        assert!(!restored.has_triple(subject, predicate, old_object));
        assert!(restored.has_triple(subject, predicate, new_object));
    }

    #[test]
    fn tms_cascade_retract_batch_replays_exactly_without_restored_tms() {
        let live = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let premise_subject = live
            .create_symbol(SymbolKind::Entity, "cascade-premise-subject")
            .unwrap()
            .id;
        let predicate = live
            .create_symbol(SymbolKind::Relation, "cascade-predicate")
            .unwrap()
            .id;
        let premise_object = live
            .create_symbol(SymbolKind::Entity, "cascade-premise-object")
            .unwrap()
            .id;
        let derived_subject = live
            .create_symbol(SymbolKind::Entity, "cascade-derived-subject")
            .unwrap()
            .id;
        let derived_object = live
            .create_symbol(SymbolKind::Entity, "cascade-derived-object")
            .unwrap()
            .id;
        let premise = Triple::new(premise_subject, predicate, premise_object);
        live.add_triple(&premise).unwrap();
        let derived = Triple::new(derived_subject, predicate, derived_object);
        live.add_triple_with_support(
            &derived,
            crate::tms::SupportSet::new(
                vec![premise_object],
                crate::provenance::DerivationKind::Seed,
                1.0,
            ),
        )
        .unwrap();
        let captured = live.all_triples();
        let wal = Arc::new(crate::wal::MutationLog::new_buffered_for_test(
            Uuid::from_u128(1),
            Uuid::from_u128(2),
            crate::wal::WalConfig::new(std::time::Duration::from_secs(1), 64, 1 << 20),
        ));
        assert!(live
            .set_mutation_sink(Arc::clone(&wal) as Arc<dyn MutationSink>)
            .is_ok());

        live.remove_triple(premise_subject, predicate, premise_object)
            .unwrap();
        assert!(live.all_triples().is_empty());
        let records = wal.drain_records_for_test();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].kind, MutationKind::RetractBatch);
        assert_eq!(records[0].payload["retracts"].as_array().unwrap().len(), 2);

        // A fresh replay engine deliberately has no TMS supports. The exact
        // compound tail must still remove both the primary and cascade edge.
        let replay = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        for (kind, label) in [
            (SymbolKind::Entity, "cascade-premise-subject"),
            (SymbolKind::Relation, "cascade-predicate"),
            (SymbolKind::Entity, "cascade-premise-object"),
            (SymbolKind::Entity, "cascade-derived-subject"),
            (SymbolKind::Entity, "cascade-derived-object"),
        ] {
            replay.create_symbol(kind, label).unwrap();
        }
        replay.restore_triples(&captured).unwrap();
        let tail = akhomed_store::TailMutation {
            kind: records[0].kind.as_str().to_string(),
            payload: records[0].payload.clone(),
        };
        assert!(apply_mutation(&replay, &tail).unwrap());
        assert!(replay.all_triples().is_empty());
    }

    #[test]
    fn retained_retract_batch_restores_provenance_after_older_assert_overflows() {
        use crate::provenance::{DerivationKind, ProvenanceRecord};

        let live_dir = tempfile::tempdir().unwrap();
        let live = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            data_dir: Some(live_dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        let subject = live
            .create_symbol(SymbolKind::Entity, "overflow-retract-subject")
            .unwrap()
            .id;
        let predicate = live
            .create_symbol(SymbolKind::Relation, "overflow-retract-predicate")
            .unwrap()
            .id;
        let object = live
            .create_symbol(SymbolKind::Entity, "overflow-retract-object")
            .unwrap()
            .id;
        let source = live
            .create_symbol(SymbolKind::Entity, "overflow-retract-source")
            .unwrap()
            .id;
        let mut provenance = ProvenanceRecord::new(object, DerivationKind::Extracted)
            .with_sources(vec![source])
            .with_confidence(0.84);
        let provenance_id = live.store_provenance(&mut provenance).unwrap();

        // Model a snapshot taken before these symbols, provenance, and edge.
        let mut restored_symbols = Vec::new();
        let mut restored_provenance = Vec::new();
        let wal = Arc::new(crate::wal::MutationLog::new_buffered_for_test(
            Uuid::from_u128(30),
            Uuid::from_u128(31),
            crate::wal::WalConfig::new(std::time::Duration::from_secs(1), 64, 1 << 20),
        ));
        assert!(live
            .set_mutation_sink(Arc::clone(&wal) as Arc<dyn MutationSink>)
            .is_ok());
        live.add_triple(
            &Triple::new(subject, predicate, object)
                .with_confidence(0.84)
                .with_provenance(provenance_id.get()),
        )
        .unwrap();
        live.remove_triple(subject, predicate, object).unwrap();

        let records = wal.drain_records_for_test();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].kind, MutationKind::Assert);
        assert_eq!(records[1].kind, MutationKind::RetractBatch);
        let batch_bytes = records[1].payload.to_string().len() + 64;
        let overflow = crate::wal::WalBuffer::new(batch_bytes + 8);
        for record in records {
            overflow.push(record.kind, record.payload);
        }
        assert_eq!(overflow.dropped_on_overflow(), 1);
        let retained = overflow.drain_records_for_test();
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].kind, MutationKind::RetractBatch);
        assert_eq!(
            retained[0].payload["v"],
            crate::mutation::RETRACT_BATCH_MUTATION_VERSION
        );
        assert_eq!(retained[0].payload["provenance"].as_array().unwrap().len(), 1);

        let tail = vec![akhomed_store::TailMutation {
            kind: retained[0].kind.as_str().into(),
            payload: retained[0].payload.clone(),
        }];
        let allocator_next = prepare_tail_restore(
            &tail,
            &mut restored_symbols,
            &mut restored_provenance,
        )
        .unwrap();
        assert_eq!(restored_symbols.len(), 4);
        assert_eq!(restored_provenance.len(), 1);

        let replay_dir = tempfile::tempdir().unwrap();
        {
            let store = TieredStore::with_persistence(replay_dir.path(), "symbols").unwrap();
            write_registry(&store, &restored_symbols).unwrap();
            put_bincode_required(&store, b"sym_allocator_next", &allocator_next).unwrap();
        }
        let replay = Engine::new_for_restore(EngineConfig {
            data_dir: Some(replay_dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        replay
            .replace_all_provenance(&restored_provenance)
            .unwrap();
        assert!(apply_mutation(&replay, &tail[0]).unwrap());
        assert!(replay.all_triples().is_empty());
        assert_eq!(
            replay.get_provenance(provenance_id).unwrap().sources,
            vec![source]
        );
    }

    #[test]
    fn duplicate_live_assertions_wake_to_exact_merged_metadata() {
        let live_dir = tempfile::tempdir().unwrap();
        let live = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            data_dir: Some(live_dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        let predicate = live
            .create_symbol(SymbolKind::Relation, "wake-duplicate-predicate")
            .unwrap()
            .id;
        let mut keys = Vec::new();
        for label in ["wake-lower", "wake-new-provenance", "wake-incoming-none"] {
            keys.push((
                live.create_symbol(SymbolKind::Entity, format!("{label}-subject"))
                    .unwrap()
                    .id,
                predicate,
                live.create_symbol(SymbolKind::Entity, format!("{label}-object"))
                    .unwrap()
                    .id,
            ));
        }
        live.add_triple(&Triple::new(keys[0].0, predicate, keys[0].2).with_confidence(0.9))
            .unwrap();
        live.add_triple(&Triple::new(keys[1].0, predicate, keys[1].2).with_confidence(0.8))
            .unwrap();
        let mut old_record = crate::provenance::ProvenanceRecord::new(
            keys[2].2,
            crate::provenance::DerivationKind::Seed,
        );
        let old_provenance = live.store_provenance(&mut old_record).unwrap();
        live.add_triple(
            &Triple::new(keys[2].0, predicate, keys[2].2)
                .with_confidence(0.7)
                .with_provenance(old_provenance.get()),
        )
        .unwrap();

        let captured_triples = live.all_triples();
        let mut restored_symbols = live.dump_symbols();
        let mut restored_provenance = live.all_provenance().unwrap();
        let wal = Arc::new(crate::wal::MutationLog::new_buffered_for_test(
            Uuid::from_u128(3),
            Uuid::from_u128(4),
            crate::wal::WalConfig::new(std::time::Duration::from_secs(1), 64, 1 << 20),
        ));
        assert!(live
            .set_mutation_sink(Arc::clone(&wal) as Arc<dyn MutationSink>)
            .is_ok());
        live.add_triple(&Triple::new(keys[0].0, predicate, keys[0].2).with_confidence(0.2))
            .unwrap();
        let mut new_record = crate::provenance::ProvenanceRecord::new(
            keys[1].2,
            crate::provenance::DerivationKind::Seed,
        );
        let new_provenance = live.store_provenance(&mut new_record).unwrap();
        live.add_triple(
            &Triple::new(keys[1].0, predicate, keys[1].2)
                .with_confidence(0.3)
                .with_provenance(new_provenance.get()),
        )
        .unwrap();
        live.add_triple(&Triple::new(keys[2].0, predicate, keys[2].2).with_confidence(0.1))
            .unwrap();

        // Exercise the production MutationLog sink and its bounded WalBuffer
        // before simulating the crash, then reconstruct the ordered relational
        // tail exactly as wake consumes it.
        let tail = wal
            .drain_records_for_test()
            .into_iter()
            .map(|record| akhomed_store::TailMutation {
                kind: record.kind.as_str().to_string(),
                payload: record.payload,
            })
            .collect::<Vec<_>>();
        let allocator_next = prepare_tail_restore(
            &tail,
            &mut restored_symbols,
            &mut restored_provenance,
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        {
            let store = TieredStore::with_persistence(dir.path(), "symbols").unwrap();
            write_registry(&store, &restored_symbols).unwrap();
            put_bincode_required(&store, b"sym_allocator_next", &allocator_next).unwrap();
        }
        let replay = Engine::new_for_restore(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            data_dir: Some(dir.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        replay
            .replace_all_provenance(&restored_provenance)
            .unwrap();
        replay.restore_triples(&captured_triples).unwrap();
        for mutation in &tail {
            assert!(apply_mutation(&replay, mutation).unwrap());
        }

        let by_key = |engine: &Engine| {
            let mut triples = engine.all_triples();
            triples.sort_by_key(|triple| {
                (triple.subject, triple.predicate, triple.object)
            });
            triples
        };
        assert_eq!(by_key(&live), by_key(&replay));
    }

    #[test]
    fn retract_batch_metadata_mismatch_is_atomic_and_fails_closed() {
        let engine = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let subject = engine
            .create_symbol(SymbolKind::Entity, "mismatch-subject")
            .unwrap()
            .id;
        let predicate = engine
            .create_symbol(SymbolKind::Relation, "mismatch-predicate")
            .unwrap()
            .id;
        let first_object = engine
            .create_symbol(SymbolKind::Entity, "mismatch-first")
            .unwrap()
            .id;
        let second_object = engine
            .create_symbol(SymbolKind::Entity, "mismatch-second")
            .unwrap()
            .id;
        let first = Triple::new(subject, predicate, first_object).with_confidence(0.8);
        let second = Triple::new(subject, predicate, second_object).with_confidence(0.7);
        engine.add_triple(&first).unwrap();
        engine.add_triple(&second).unwrap();
        let exact_first = engine
            .knowledge_graph()
            .triple(subject, predicate, first_object)
            .unwrap();
        let exact_second = engine
            .knowledge_graph()
            .triple(subject, predicate, second_object)
            .unwrap();
        let edge_json = |edge: &Triple, confidence: f32| {
            serde_json::json!({
                "subject": edge.subject.get(),
                "predicate": edge.predicate.get(),
                "object": edge.object.get(),
                "confidence": confidence,
                "timestamp": edge.timestamp,
                "compartment": edge.compartment_id.clone(),
                "provenance_id": edge.provenance_id,
            })
        };
        let mutation = akhomed_store::TailMutation {
            kind: "retract_batch".into(),
            payload: serde_json::json!({
                "v": crate::mutation::RETRACT_BATCH_MUTATION_VERSION,
                "retracts": [
                    edge_json(&exact_first, exact_first.confidence),
                    edge_json(&exact_second, 0.1),
                ],
                "symbols": engine.dump_symbols(),
            }),
        };

        assert!(apply_mutation(&engine, &mutation).is_err());
        assert!(engine.has_triple(subject, predicate, first_object));
        assert!(engine.has_triple(subject, predicate, second_object));
    }

    #[test]
    fn retract_batch_with_unknown_provenance_fails_before_removal() {
        let engine = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let subject = engine
            .create_symbol(SymbolKind::Entity, "unknown-provenance-subject")
            .unwrap()
            .id;
        let predicate = engine
            .create_symbol(SymbolKind::Relation, "unknown-provenance-predicate")
            .unwrap()
            .id;
        let object = engine
            .create_symbol(SymbolKind::Entity, "unknown-provenance-object")
            .unwrap()
            .id;
        let edge = Triple::new(subject, predicate, object);
        engine.add_triple(&edge).unwrap();
        let exact = engine
            .knowledge_graph()
            .triple(subject, predicate, object)
            .unwrap();
        let mutation = akhomed_store::TailMutation {
            kind: "retract_batch".into(),
            payload: serde_json::json!({
                "v": 1,
                "retracts": [{
                    "subject": exact.subject.get(),
                    "predicate": exact.predicate.get(),
                    "object": exact.object.get(),
                    "confidence": exact.confidence,
                    "timestamp": exact.timestamp,
                    "compartment": exact.compartment_id,
                    "provenance_id": 999,
                }],
                "symbols": engine.dump_symbols(),
            }),
        };

        let error = apply_mutation(&engine, &mutation).unwrap_err();
        assert!(error.to_string().contains("provenance ID 999"), "{error}");
        assert!(engine.has_triple(subject, predicate, object));
    }

    #[test]
    fn legacy_retract_batch_may_idempotently_skip_absent_edge_without_provenance() {
        let engine = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let subject = engine
            .create_symbol(SymbolKind::Entity, "legacy-absent-subject")
            .unwrap();
        let predicate = engine
            .create_symbol(SymbolKind::Relation, "legacy-absent-predicate")
            .unwrap();
        let object = engine
            .create_symbol(SymbolKind::Entity, "legacy-absent-object")
            .unwrap();
        let mutation = akhomed_store::TailMutation {
            kind: "retract_batch".into(),
            payload: serde_json::json!({
                "v": 1,
                "retracts": [{
                    "subject": subject.id.get(),
                    "predicate": predicate.id.get(),
                    "object": object.id.get(),
                    "confidence": 1.0,
                    "timestamp": 0,
                    "compartment": null,
                    "provenance_id": 999,
                }],
                "symbols": [subject, predicate, object],
            }),
        };

        assert!(apply_mutation(&engine, &mutation).unwrap());
        assert!(engine.all_triples().is_empty());
    }

    #[test]
    fn corrupt_tail_retract_with_unknown_symbol_fails_closed() {
        let engine = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let subject = engine
            .create_symbol(SymbolKind::Entity, "known-subject")
            .unwrap();
        let predicate = engine
            .create_symbol(SymbolKind::Relation, "known-predicate")
            .unwrap();
        let object = engine
            .create_symbol(SymbolKind::Entity, "known-object")
            .unwrap();
        let unknown = SymbolId::new(999_999).unwrap();
        engine
            .add_triple(&Triple::new(subject.id, predicate.id, object.id))
            .unwrap();

        let mutation = akhomed_store::TailMutation {
            kind: "retract".into(),
            payload: serde_json::json!({
                "subject": subject.id.get(),
                "predicate": predicate.id.get(),
                "object": unknown.get(),
            }),
        };
        assert!(apply_mutation(&engine, &mutation).is_err());
        assert!(engine.has_triple(subject.id, predicate.id, object.id));

        let mut symbols = vec![subject, predicate, object];
        let mut provenance = Vec::new();
        assert!(prepare_tail_restore(&[mutation], &mut symbols, &mut provenance).is_err());
    }

    #[test]
    fn relational_symbol_ids_reject_unsigned_overflow() {
        let too_large = SymbolId::new(i64::MAX as u64 + 1).unwrap();
        assert!(relational_symbol_id("test", too_large).is_err());
    }

    #[test]
    fn restore_validation_rejects_allocator_collisions_and_malformed_facts() {
        let meta = replay_symbol(9, SymbolKind::Entity, "Nine", 0);
        assert!(validate_symbol_snapshot(std::slice::from_ref(&meta), 9).is_err());
        assert!(validate_symbol_snapshot(std::slice::from_ref(&meta), 10).is_ok());

        let malformed = fact::Model {
            workspace_id: Uuid::nil(),
            subject: 0,
            predicate: 9,
            object: 9,
            akh_id: Uuid::nil(),
            confidence: 1.0,
            asserted_at: chrono::DateTime::from_timestamp(10, 0).unwrap(),
            retracted_at: None,
            provenance_id: None,
            compartment: None,
        };
        assert!(fact_to_triple(&malformed).is_err());
        assert!(validate_fact_symbols(&[malformed], &[meta]).is_err());

        let subject = replay_symbol(20, SymbolKind::Entity, "Subject", 0);
        let predicate = replay_symbol(21, SymbolKind::Relation, "predicate", 0);
        let object = replay_symbol(22, SymbolKind::Entity, "Object", 0);
        let missing_id_record = crate::provenance::ProvenanceRecord::new(
            object.id,
            crate::provenance::DerivationKind::Extracted,
        );
        assert!(
            parse_replay_complete_assert(&serde_json::json!({
                "v": crate::mutation::ASSERT_MUTATION_VERSION,
                "subject": subject.id.get(),
                "predicate": predicate.id.get(),
                "object": object.id.get(),
                "confidence": 1.0,
                "timestamp": 0,
                "provenance_id": 0,
                "provenance": missing_id_record,
                "symbols": [subject, predicate, object],
            }))
            .is_err(),
            "zero must not compare equal to a missing provenance ID"
        );
    }

    #[test]
    fn capture_uses_the_pre_capture_watermark() {
        let root = tempfile::tempdir().unwrap();
        let engine = Arc::new(
            Engine::new(EngineConfig {
                data_dir: Some(root.path().to_path_buf()),
                ..Default::default()
            })
            .unwrap(),
        );
        let agent = Agent::new(engine, crate::agent::AgentConfig::default()).unwrap();
        let data = capture_snapshot_at_watermark(&agent, Uuid::new_v4(), Uuid::new_v4(), 37)
            .unwrap();
        assert_eq!(data.watermark, 37);
        assert!(capture_snapshot_at_watermark(&agent, Uuid::nil(), Uuid::nil(), -1).is_err());
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
        assert!(
            !apply_mutation(&engine, &m).unwrap(),
            "turn rows are not KG mutations"
        );

        // An unrelated malformed/unknown kind also returns false — same
        // non-error signal — so the explicit "turn" arm is a documentation /
        // intent improvement, not a behavior change vs the prior catch-all.
        let unknown = akhomed_store::TailMutation {
            kind: "totally_unknown".to_string(),
            payload: serde_json::json!({}),
        };
        assert!(!apply_mutation(&engine, &unknown).unwrap());
    }
}
