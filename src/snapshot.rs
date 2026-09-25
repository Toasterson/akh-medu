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
const REACTOR_SNAPSHOT_VERSION: u64 = 2;
const REACTOR_SCHEDULE_PREFIX: &[u8] = b"reactor:schedule:";
const REACTOR_DJEHUTY_CURSOR_PREFIX: &[u8] = b"reactor:djehuty:cursor:";

/// Recovery-critical reactor metadata carried in the relational snapshot.
///
/// Values use exact-byte hex so redb JSON containing NUL separators remains
/// valid PostgreSQL JSONB. Wake decodes and semantically validates each value;
/// only the two explicitly allow-listed namespaces are ever restored.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct PortableReactorMetadata {
    key: String,
    /// Hex preserves the exact local JSON bytes while keeping control
    /// characters such as the reactor event-key NUL separator out of JSONB
    /// strings (PostgreSQL rejects `\\u0000` in `jsonb`).
    value_hex: String,
}

/// Version 1 embedded reactor values directly in JSON. It remains readable so
/// pods can roll forward from snapshots written before the JSONB/NUL boundary
/// was discovered; all new captures use the byte-exact v2 representation.
#[derive(Debug, Clone, serde::Deserialize)]
struct LegacyPortableReactorMetadata {
    key: String,
    value: serde_json::Value,
}

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
    agent
        .persist()
        .map_err(|e| SnapshotError::Agent(e.to_string()))?;

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

    // Goal symbols are relational primary keys. Autonomous proposal sources
    // can converge on the same semantic goal, so capture one canonical row
    // per symbol even if a legacy/imported in-memory list contains repeats.
    let unique_goals = unique_goals_by_symbol(&agent.goals);
    let goals: Vec<goal::Model> = unique_goals
        .iter()
        .map(|g| goal_to_model(akh_id, g))
        .collect::<SnapshotResult<_>>()?;
    // Normalised goal DAG: `children` → ordered hierarchy edges; `blocked_by` →
    // dependency edges.
    let mut goal_hierarchy = Vec::new();
    let mut goal_dependencies = Vec::new();
    for g in unique_goals {
        let goal_id = relational_symbol_id("goal", g.symbol_id)?;
        let children = unique_symbol_ids(&g.children);
        if children.len() != g.children.len() {
            tracing::warn!(
                goal_id = g.symbol_id.get(),
                discarded = g.children.len() - children.len(),
                "discarding repeated goal hierarchy edges before relational snapshot"
            );
        }
        for (position, child) in children.iter().enumerate() {
            goal_hierarchy.push(goal_hierarchy::Model {
                akh_id,
                parent_id: goal_id,
                child_id: relational_symbol_id("goal child", *child)?,
                position: i32::try_from(position)
                    .map_err(|_| SnapshotError::Agent("goal child position exceeds i32".into()))?,
            });
        }
        let blockers = unique_symbol_ids(&g.blocked_by);
        if blockers.len() != g.blocked_by.len() {
            tracing::warn!(
                goal_id = g.symbol_id.get(),
                discarded = g.blocked_by.len() - blockers.len(),
                "discarding repeated goal dependency edges before relational snapshot"
            );
        }
        for blocker in blockers {
            goal_dependencies.push(goal_dependency::Model {
                akh_id,
                goal_id,
                blocking_goal_id: relational_symbol_id("blocking goal", blocker)?,
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

    // Sleep metrics (time series). The table is keyed by (akh, cycle), but
    // several sleeps can land in one OODA cycle when the daemon sleeps and no
    // OODA cycle has run in between, so keep one row per cycle (the latest):
    // duplicate keys failed the whole snapshot (UNIQUE constraint, 2026-09-25).
    let latest_per_cycle: std::collections::BTreeMap<u64, _> = agent
        .sleep_cycle
        .metrics_history
        .iter()
        .map(|(cycle, m)| (*cycle, m))
        .collect();
    let sleep_rows: Vec<sleep_metric::Model> = latest_per_cycle
        .into_iter()
        .map(|(cycle, m)| sleep_metric::Model {
            akh_id,
            cycle: cycle as i64,
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
    let reactor_metadata = capture_reactor_metadata(engine)?;
    let budget_ledger = capture_budget_ledger(engine);
    let last_run = capture_last_run_log(engine);
    let provider_gate = capture_provider_gate(engine);
    let extra = serde_json::json!({
        "provenance_snapshot_version": PROVENANCE_SNAPSHOT_VERSION,
        "audience": agent.audience,
        "improvement_history": agent.improvement_history,
        "method_stats": agent.method_registry_htn.export_stats(),
        "session_summary": session_summary,
        "person_memory": agent.person_memory.to_portable(),
        "reactor": {
            "version": REACTOR_SNAPSHOT_VERSION,
            "metadata": reactor_metadata,
        },
        "autonomy": {
            "budget_ledger": budget_ledger,
            "last_run": last_run,
        },
        "provider": {
            "gate": provider_gate,
        },
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

fn unique_goals_by_symbol(goals: &[crate::agent::Goal]) -> Vec<&crate::agent::Goal> {
    // Agent goal lookups use `.iter().find(...)`, so the first occurrence is
    // already the live authority. Preserve that same deterministic policy at
    // the relational boundary and make divergent repeats observable.
    let mut positions = std::collections::HashMap::with_capacity(goals.len());
    let mut unique = Vec::with_capacity(goals.len());
    for candidate in goals {
        if let Some(&position) = positions.get(&candidate.symbol_id) {
            let canonical: &crate::agent::Goal = unique[position];
            let conflicting = match (
                serde_json::to_value(canonical),
                serde_json::to_value(candidate),
            ) {
                (Ok(canonical), Ok(candidate)) => canonical != candidate,
                _ => true,
            };
            tracing::warn!(
                goal_id = candidate.symbol_id.get(),
                conflicting,
                "discarding repeated goal symbol before relational snapshot; first live entry is authoritative"
            );
        } else {
            positions.insert(candidate.symbol_id, unique.len());
            unique.push(candidate);
        }
    }
    unique
}

fn unique_symbol_ids(ids: &[crate::symbol::SymbolId]) -> Vec<crate::symbol::SymbolId> {
    let mut seen = std::collections::HashSet::with_capacity(ids.len());
    ids.iter().copied().filter(|id| seen.insert(*id)).collect()
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
    let base_allocator_next = validate_symbol_snapshot(&symbol_metas, state.allocator_next)?;

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
    let mut tail = ops::load_mutation_tail(db, akh_id, state.watermark).await?;
    rewrite_tail_after_symbol_merges(&mut tail, &symbol_metas)?;
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
        put_bincode_required(&store, b"sym_allocator_next", &restored_allocator_next)?;
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
        if let Some(value) = state.psyche.as_ref() {
            let psyche: crate::compartment::psyche::Psyche = serde_json::from_value(value.clone())
                .map_err(|e| {
                    SnapshotError::Agent(format!("wake: invalid persisted psyche: {e}"))
                })?;
            let json = serde_json::to_vec(&psyche)
                .map_err(|e| SnapshotError::Agent(format!("wake: encode psyche JSON: {e}")))?;
            store
                .put_meta(b"agent:psyche_json_v1", &json)
                .map_err(|e| SnapshotError::Agent(format!("wake: persist psyche JSON: {e}")))?;
            put_bincode(&store, b"agent:psyche", &psyche);
        } else {
            for key in [
                b"agent:psyche_json_v1".as_slice(),
                b"agent:psyche".as_slice(),
            ] {
                store
                    .remove_meta(key)
                    .map_err(|e| SnapshotError::Agent(format!("wake: clear absent psyche: {e}")))?;
            }
        }
        restore_extra(&store, state.extra.as_ref());
        restore_reactor_metadata(&store, state.extra.as_ref())?;
        restore_budget_ledger(&store, state.extra.as_ref());
        restore_last_run_log(&store, state.extra.as_ref());
        restore_provider_gate(&store, state.extra.as_ref());
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
    let replayed = apply_mutation_tail(&engine, &tail)?;
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

/// One losing legacy symbol and the deterministic canonical row it will merge
/// into. Labels and kinds make dry-run output self-contained.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PlannedSymbolMerge {
    pub normalized_label: String,
    pub canonical_id: i64,
    pub canonical_label: String,
    pub canonical_kind: String,
    pub duplicate_id: i64,
    pub duplicate_label: String,
    pub duplicate_kind: String,
}

/// Dry-run/apply plan for WI-338/WI-359 symbol hydration repair.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SymbolRepairPlan {
    pub akh_id: Uuid,
    pub duplicate_groups: usize,
    pub extra_symbols: usize,
    pub merges: Vec<PlannedSymbolMerge>,
    pub allocator: akhomed_store::repair::SymbolAllocatorRepair,
}

/// Result returned to the daemon control plane.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SymbolRepairReport {
    pub dry_run: bool,
    pub applied: bool,
    pub plan: SymbolRepairPlan,
    pub counts: akhomed_store::repair::SymbolRepairCounts,
}

/// Detect duplicate labels and a stale relational allocator using the same
/// fail-closed constraints as wake. Lowest relational ID is always canonical.
pub async fn plan_duplicate_symbol_repair(
    db: &DatabaseConnection,
    akh_id: Uuid,
) -> SnapshotResult<SymbolRepairPlan> {
    let rows = akhomed_store::snapshots::load_symbols(db, akh_id).await?;
    let state = akhomed_store::snapshots::load_akh_state(db, akh_id)
        .await?
        .ok_or(SnapshotError::NotFound(akh_id))?;
    build_symbol_repair_plan(akh_id, rows, state.allocator_next)
}

fn build_symbol_repair_plan(
    akh_id: Uuid,
    mut rows: Vec<symbol::Model>,
    allocator_next: i64,
) -> SnapshotResult<SymbolRepairPlan> {
    for row in &rows {
        let raw_id = u64::try_from(row.id).map_err(|_| {
            SnapshotError::Agent(format!(
                "repair: symbol {} uses invalid ID {}",
                row.label, row.id
            ))
        })?;
        if raw_id == 0 || is_reserved_symbol_id(raw_id) {
            return Err(SnapshotError::Agent(format!(
                "repair: symbol {} uses invalid/reserved ID {}",
                row.label, row.id
            )));
        }
    }
    rows.sort_by_key(|row| row.id);
    let mut groups: std::collections::BTreeMap<String, Vec<symbol::Model>> =
        std::collections::BTreeMap::new();
    for row in &rows {
        groups
            .entry(crate::registry::normalize_label_key(&row.label))
            .or_default()
            .push(row.clone());
    }

    let duplicate_groups = groups.values().filter(|rows| rows.len() > 1).count();
    let mut merges = Vec::new();
    for (normalized_label, rows) in groups {
        let Some(canonical) = rows.first().filter(|_| rows.len() > 1) else {
            continue;
        };
        for duplicate in &rows[1..] {
            merges.push(PlannedSymbolMerge {
                normalized_label: normalized_label.clone(),
                canonical_id: canonical.id,
                canonical_label: canonical.label.clone(),
                canonical_kind: canonical.kind.clone(),
                duplicate_id: duplicate.id,
                duplicate_label: duplicate.label.clone(),
                duplicate_kind: duplicate.kind.clone(),
            });
        }
    }
    let losing_ids: std::collections::HashSet<i64> =
        merges.iter().map(|merge| merge.duplicate_id).collect();
    let max_surviving_symbol_id = rows
        .iter()
        .filter(|row| !losing_ids.contains(&row.id))
        .map(|row| row.id)
        .max();
    let allocator = plan_symbol_allocator_repair(allocator_next, max_surviving_symbol_id)?;
    Ok(SymbolRepairPlan {
        akh_id,
        duplicate_groups,
        extra_symbols: merges.len(),
        merges,
        allocator,
    })
}

fn required_allocator_next(max_surviving_symbol_id: Option<u64>) -> SnapshotResult<u64> {
    let required = max_surviving_symbol_id
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| SnapshotError::Agent("repair: symbol ID space is exhausted".into()))?;
    if is_reserved_symbol_id(required) {
        return Err(SnapshotError::Agent(
            "repair: symbol ID space reaches a reserved protocol marker".into(),
        ));
    }
    Ok(required)
}

fn plan_symbol_allocator_repair(
    allocator_next: i64,
    max_surviving_symbol_id: Option<i64>,
) -> SnapshotResult<akhomed_store::repair::SymbolAllocatorRepair> {
    let max_surviving = max_surviving_symbol_id
        .map(|id| {
            u64::try_from(id).map_err(|_| {
                SnapshotError::Agent(format!("repair: invalid surviving symbol ID {id}"))
            })
        })
        .transpose()?;
    let required = required_allocator_next(max_surviving)?;
    let current = u64::try_from(allocator_next).map_err(|_| {
        SnapshotError::Agent(format!(
            "repair: symbol allocator has invalid next ID {allocator_next}"
        ))
    })?;
    let planned = current.max(required);
    if is_reserved_symbol_id(planned) {
        return Err(SnapshotError::Agent(format!(
            "repair: symbol allocator has invalid/reserved next ID {planned}"
        )));
    }
    let required_allocator_next = i64::try_from(required).map_err(|_| {
        SnapshotError::Agent(format!(
            "repair: required allocator next ID {required} cannot be stored"
        ))
    })?;
    let allocator_next_after = i64::try_from(planned).map_err(|_| {
        SnapshotError::Agent(format!(
            "repair: allocator next ID {planned} cannot be stored"
        ))
    })?;
    Ok(akhomed_store::repair::SymbolAllocatorRepair {
        allocator_next_before: allocator_next,
        max_surviving_symbol_id,
        required_allocator_next,
        allocator_next_after,
        advanced: allocator_next_after > allocator_next,
    })
}

fn json_contains_symbol_id(value: &serde_json::Value, id: i64) -> bool {
    match value {
        serde_json::Value::Number(number) => number.as_i64() == Some(id),
        serde_json::Value::Array(values) => values
            .iter()
            .any(|value| json_contains_symbol_id(value, id)),
        serde_json::Value::Object(object) => object
            .values()
            .any(|value| json_contains_symbol_id(value, id)),
        _ => false,
    }
}

/// Apply a symbol repair plan transactionally. Every merge gets a `symbol_merge`
/// WAL audit record containing stable TripleIdentity changes and the provenance
/// records which referenced the losing ID; allocator-only plans remain valid.
pub async fn repair_duplicate_symbols(
    db: &DatabaseConnection,
    akh_id: Uuid,
    workspace_id: Uuid,
    dry_run: bool,
) -> SnapshotResult<SymbolRepairReport> {
    use akhomed_store::entity::{fact, provenance, provenance_input, provenance_snapshot};
    use akhomed_store::sea_orm::{ColumnTrait, EntityTrait, QueryFilter};

    let mut plan = plan_duplicate_symbol_repair(db, akh_id).await?;
    if dry_run {
        return Ok(SymbolRepairReport {
            dry_run,
            applied: false,
            plan,
            counts: akhomed_store::repair::SymbolRepairCounts::default(),
        });
    }

    if plan.merges.is_empty() {
        let outcome =
            akhomed_store::repair::apply_symbol_repair(db, akh_id, workspace_id, &[], &[]).await?;
        plan.allocator = outcome.allocator;
        return Ok(SymbolRepairReport {
            dry_run: false,
            applied: outcome.allocator.advanced,
            plan,
            counts: outcome.counts,
        });
    }

    let symbols = akhomed_store::snapshots::load_symbols(db, akh_id).await?;
    let symbol_labels: std::collections::HashMap<i64, String> = symbols
        .iter()
        .map(|row| (row.id, row.label.clone()))
        .collect();
    let final_ids: std::collections::HashMap<i64, i64> = plan
        .merges
        .iter()
        .map(|merge| (merge.duplicate_id, merge.canonical_id))
        .collect();
    let facts = fact::Entity::find()
        // Match wake/apply semantics: the durable graph image is selected by
        // workspace, regardless of which Akh originally asserted each fact.
        .filter(fact::Column::WorkspaceId.eq(workspace_id))
        .all(db)
        .await
        .map_err(akhomed_store::StoreError::from)?;
    let provenance_rows = provenance::Entity::find()
        .filter(provenance::Column::AkhId.eq(akh_id))
        .all(db)
        .await
        .map_err(akhomed_store::StoreError::from)?;
    let provenance_ids: Vec<i64> = provenance_rows
        .iter()
        .map(|row| row.provenance_id)
        .collect();
    let provenance_inputs = if provenance_ids.is_empty() {
        Vec::new()
    } else {
        provenance_input::Entity::find()
            .filter(provenance_input::Column::ProvenanceId.is_in(provenance_ids))
            .all(db)
            .await
            .map_err(akhomed_store::StoreError::from)?
    };
    let provenance_snapshots = provenance_snapshot::Entity::find()
        .filter(provenance_snapshot::Column::AkhId.eq(akh_id))
        .all(db)
        .await
        .map_err(akhomed_store::StoreError::from)?;

    let final_id = |id: i64| final_ids.get(&id).copied().unwrap_or(id);
    let label = |id: i64| {
        symbol_labels
            .get(&id)
            .map(String::as_str)
            .unwrap_or("<missing-symbol>")
    };
    let mut audits = Vec::with_capacity(plan.merges.len());
    for merge in &plan.merges {
        let mut triple_changes = Vec::new();
        let mut workspace_ids = std::collections::BTreeSet::new();
        for row in &facts {
            if ![row.subject, row.predicate, row.object].contains(&merge.duplicate_id) {
                continue;
            }
            workspace_ids.insert(row.workspace_id);
            let before = crate::graph::identity::TripleIdentity::compute(
                label(row.subject),
                label(row.predicate),
                label(row.object),
                row.compartment.as_deref(),
            );
            let after_ids = (
                final_id(row.subject),
                final_id(row.predicate),
                final_id(row.object),
            );
            let after = crate::graph::identity::TripleIdentity::compute(
                label(after_ids.0),
                label(after_ids.1),
                label(after_ids.2),
                row.compartment.as_deref(),
            );
            triple_changes.push(serde_json::json!({
                "before": [row.subject, row.predicate, row.object],
                "after": [after_ids.0, after_ids.1, after_ids.2],
                "before_identity": before.to_hex(),
                "after_identity": after.to_hex(),
                "compartment": row.compartment,
            }));
        }
        if workspace_ids.is_empty() {
            workspace_ids.insert(workspace_id);
        }

        let mut affected_provenance = std::collections::BTreeSet::new();
        for row in &provenance_rows {
            if row.subject == merge.duplicate_id
                || json_contains_symbol_id(&row.derivation, merge.duplicate_id)
            {
                affected_provenance.insert(format!("legacy:{}", row.provenance_id));
            }
        }
        for row in &provenance_inputs {
            if row.input_symbol_id == merge.duplicate_id {
                affected_provenance.insert(format!("legacy:{}", row.provenance_id));
            }
        }
        for row in &provenance_snapshots {
            if row.derived_symbol == merge.duplicate_id
                || json_contains_symbol_id(&row.sources, merge.duplicate_id)
                || json_contains_symbol_id(&row.derivation, merge.duplicate_id)
            {
                affected_provenance.insert(format!("snapshot:{}", row.provenance_id));
            }
        }

        audits.push(akhomed_store::MutationRow {
            kind: crate::mutation::MutationKind::SymbolMerge
                .as_str()
                .to_owned(),
            payload: serde_json::json!({
                "v": 1,
                "work_item": "WI-338",
                "akh_id": akh_id,
                "workspace_ids": workspace_ids,
                "normalized_label": merge.normalized_label,
                "canonical": {
                    "id": merge.canonical_id,
                    "label": merge.canonical_label,
                    "kind": merge.canonical_kind,
                },
                "duplicate": {
                    "id": merge.duplicate_id,
                    "label": merge.duplicate_label,
                    "kind": merge.duplicate_kind,
                },
                "triple_identity_changes": triple_changes,
                "provenance_records": affected_provenance,
            }),
        });
    }

    let store_merges: Vec<_> = plan
        .merges
        .iter()
        .map(|merge| akhomed_store::repair::SymbolMerge {
            canonical_id: merge.canonical_id,
            duplicate_id: merge.duplicate_id,
        })
        .collect();
    let outcome = akhomed_store::repair::apply_symbol_repair(
        db,
        akh_id,
        workspace_id,
        &store_merges,
        &audits,
    )
    .await?;
    plan.allocator = outcome.allocator;
    Ok(SymbolRepairReport {
        dry_run: false,
        applied: true,
        plan,
        counts: outcome.counts,
    })
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

/// Capture the autonomy budget ledger (`autonomy:budget:v1`, Phase 53b) so
/// a relocated or re-woken akh keeps today's spend instead of starting the
/// day with a fresh budget. `None` when the akh has no ledger yet; an
/// unreadable ledger is logged and left out (the daemon resets a bad ledger
/// the same way, which grants at most one extra day of budget).
fn capture_budget_ledger(engine: &Engine) -> Option<crate::agent::autonomy::BudgetLedger> {
    use crate::agent::autonomy::{BUDGET_LEDGER_KEY, BudgetLedger};
    match engine.store().get_meta(BUDGET_LEDGER_KEY) {
        Ok(Some(bytes)) => match serde_json::from_slice::<BudgetLedger>(&bytes) {
            Ok(ledger) => Some(ledger),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "snapshot: autonomy budget ledger is corrupt; not captured"
                );
                None
            }
        },
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "snapshot: autonomy budget ledger unreadable; not captured"
            );
            None
        }
    }
}

/// Restore the autonomy budget ledger captured by [`capture_budget_ledger`].
/// An older snapshot without one leaves the store untouched; a malformed one
/// is logged and skipped rather than failing the wake.
fn restore_budget_ledger(store: &TieredStore, extra: Option<&serde_json::Value>) {
    use crate::agent::autonomy::{BUDGET_LEDGER_KEY, BudgetLedger};
    let Some(value) = extra
        .and_then(|extra| extra.get("autonomy"))
        .and_then(|autonomy| autonomy.get("budget_ledger"))
        .filter(|value| !value.is_null())
    else {
        return;
    };
    let ledger = match serde_json::from_value::<BudgetLedger>(value.clone()) {
        Ok(ledger) => ledger,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "wake: autonomy budget ledger in the snapshot is malformed; today's budget starts empty"
            );
            return;
        }
    };
    let bytes = match serde_json::to_vec(&ledger) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(error = %error, "wake: autonomy budget ledger could not be encoded");
            return;
        }
    };
    if let Err(error) = store.put_meta(BUDGET_LEDGER_KEY, &bytes) {
        tracing::warn!(error = %error, "wake: autonomy budget ledger not restored");
    }
}

/// Capture the daemon last-run log (`daemon:last_run:v1`) so a relocated or
/// re-woken akh resumes its long-period tasks (the daily librarian) instead
/// of pushing each one a full period out. `None` when the akh has no log yet;
/// an unreadable log is logged and left out (each long-period task then
/// waits one period after the wake, never runs early).
fn capture_last_run_log(engine: &Engine) -> Option<crate::agent::autonomy::LastRunLog> {
    use crate::agent::autonomy::{LAST_RUN_LOG_KEY, LastRunLog};
    match engine.store().get_meta(LAST_RUN_LOG_KEY) {
        Ok(Some(bytes)) => match serde_json::from_slice::<LastRunLog>(&bytes) {
            Ok(log) => Some(log),
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    "snapshot: daemon last-run log is corrupt; not captured"
                );
                None
            }
        },
        Ok(None) => None,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "snapshot: daemon last-run log unreadable; not captured"
            );
            None
        }
    }
}

/// Restore the daemon last-run log captured by [`capture_last_run_log`]. An
/// older snapshot without one leaves the store untouched; a malformed one is
/// logged and skipped rather than failing the wake.
fn restore_last_run_log(store: &TieredStore, extra: Option<&serde_json::Value>) {
    use crate::agent::autonomy::{LAST_RUN_LOG_KEY, LastRunLog};
    let Some(value) = extra
        .and_then(|extra| extra.get("autonomy"))
        .and_then(|autonomy| autonomy.get("last_run"))
        .filter(|value| !value.is_null())
    else {
        return;
    };
    let log = match serde_json::from_value::<LastRunLog>(value.clone()) {
        Ok(log) => log,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "wake: daemon last-run log in the snapshot is malformed; long-period tasks wait one period"
            );
            return;
        }
    };
    let bytes = match serde_json::to_vec(&log) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(error = %error, "wake: daemon last-run log could not be encoded");
            return;
        }
    };
    if let Err(error) = store.put_meta(LAST_RUN_LOG_KEY, &bytes) {
        tracing::warn!(error = %error, "wake: daemon last-run log not restored");
    }
}

/// Capture the provider gate (`provider:gate:v1`, Phase 53g PR 4) so a
/// relocated or re-woken akh keeps backing off a rate-limited, exhausted or
/// credential-rejecting provider. `None` when the akh has no gate record; an
/// unreadable one is logged and left out (the gate then starts open, which
/// costs at most one failed model call per occurrence).
fn capture_provider_gate(
    engine: &Engine,
) -> Option<crate::agent::provider_failure::ProviderGateState> {
    match crate::agent::provider_failure::load_persisted(engine) {
        Ok(state) => state,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "snapshot: provider gate unreadable or corrupt; not captured"
            );
            None
        }
    }
}

/// Restore the provider gate captured by [`capture_provider_gate`]. An older
/// snapshot without one leaves the store untouched; a malformed one is
/// logged and skipped rather than failing the wake.
fn restore_provider_gate(store: &TieredStore, extra: Option<&serde_json::Value>) {
    use crate::agent::provider_failure::{PROVIDER_GATE_KEY, ProviderGateState};
    let Some(value) = extra
        .and_then(|extra| extra.get("provider"))
        .and_then(|provider| provider.get("gate"))
        .filter(|value| !value.is_null())
    else {
        return;
    };
    let state = match serde_json::from_value::<ProviderGateState>(value.clone()) {
        Ok(state) => state,
        Err(error) => {
            tracing::warn!(
                error = %error,
                "wake: provider gate in the snapshot is malformed; the gate starts open"
            );
            return;
        }
    };
    let bytes = match serde_json::to_vec(&state) {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!(error = %error, "wake: provider gate could not be encoded");
            return;
        }
    };
    if let Err(error) = store.put_meta(PROVIDER_GATE_KEY, &bytes) {
        tracing::warn!(error = %error, "wake: provider gate not restored");
    }
}

/// Capture schedules and board cursors from one durable-store read
/// transaction. Two independent scans could observe a cursor after ingress
/// advanced it but schedule state from before the same event was queued.
fn capture_reactor_metadata(engine: &Engine) -> SnapshotResult<Vec<PortableReactorMetadata>> {
    let entries = engine
        .store()
        .scan_prefix(b"reactor:")
        .map_err(|error| SnapshotError::Agent(format!("snapshot reactor metadata: {error}")))?;
    let mut captured = Vec::new();
    for (key, bytes) in entries {
        if bytes.is_empty()
            || !(key.starts_with(REACTOR_SCHEDULE_PREFIX)
                || key.starts_with(REACTOR_DJEHUTY_CURSOR_PREFIX))
        {
            continue;
        }
        let key = String::from_utf8(key).map_err(|error| {
            SnapshotError::Agent(format!(
                "snapshot reactor metadata key is not UTF-8: {error}"
            ))
        })?;
        let value = serde_json::from_slice(&bytes).map_err(|error| {
            SnapshotError::Agent(format!(
                "snapshot reactor metadata {key} is not valid JSON: {error}"
            ))
        })?;
        validate_reactor_metadata_entry(&key, &value)?;
        captured.push(PortableReactorMetadata {
            key,
            value_hex: encode_reactor_bytes(&bytes),
        });
    }
    captured.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(captured)
}

/// Restore the allow-listed namespaces atomically per prefix. A missing
/// reactor envelope from an older Postgres snapshot is authoritative empty:
/// production hydration quarantines host-local redb before this point.
fn restore_reactor_metadata(
    store: &TieredStore,
    extra: Option<&serde_json::Value>,
) -> SnapshotResult<()> {
    let entries = match extra.and_then(|value| value.get("reactor")) {
        Some(envelope) => {
            let version = envelope.get("version").and_then(serde_json::Value::as_u64);
            let metadata = envelope.get("metadata").cloned().ok_or_else(|| {
                SnapshotError::Agent("wake: reactor snapshot has no metadata array".into())
            })?;
            match version {
                Some(REACTOR_SNAPSHOT_VERSION) => serde_json::from_value::<
                    Vec<PortableReactorMetadata>,
                >(metadata)
                .map_err(|error| {
                    SnapshotError::Agent(format!("wake: decode reactor metadata: {error}"))
                })?,
                Some(1) => serde_json::from_value::<Vec<LegacyPortableReactorMetadata>>(metadata)
                    .map_err(|error| {
                        SnapshotError::Agent(format!(
                            "wake: decode legacy reactor metadata: {error}"
                        ))
                    })?
                    .into_iter()
                    .map(|entry| {
                        let bytes = serde_json::to_vec(&entry.value).map_err(|error| {
                            SnapshotError::Agent(format!(
                                "wake: encode legacy reactor metadata {}: {error}",
                                entry.key
                            ))
                        })?;
                        Ok(PortableReactorMetadata {
                            key: entry.key,
                            value_hex: encode_reactor_bytes(&bytes),
                        })
                    })
                    .collect::<SnapshotResult<Vec<_>>>()?,
                _ => {
                    return Err(SnapshotError::Agent(format!(
                        "wake: unsupported reactor snapshot version {version:?}"
                    )));
                }
            }
        }
        None => Vec::new(),
    };

    let mut seen = std::collections::HashSet::with_capacity(entries.len());
    let mut schedules = Vec::new();
    let mut cursors = Vec::new();
    for entry in entries {
        if !seen.insert(entry.key.clone()) {
            return Err(SnapshotError::Agent(format!(
                "wake: duplicate reactor metadata key {}",
                entry.key
            )));
        }
        let bytes = decode_reactor_bytes(&entry.value_hex).map_err(|error| {
            SnapshotError::Agent(format!(
                "wake: decode reactor metadata {}: {error}",
                entry.key
            ))
        })?;
        let value = serde_json::from_slice(&bytes).map_err(|error| {
            SnapshotError::Agent(format!(
                "wake: reactor metadata {} is not valid JSON: {error}",
                entry.key
            ))
        })?;
        validate_reactor_metadata_entry(&entry.key, &value)?;
        let key = entry.key.into_bytes();
        if key.starts_with(REACTOR_SCHEDULE_PREFIX) {
            schedules.push((key, bytes));
        } else {
            cursors.push((key, bytes));
        }
    }

    replace_reactor_namespace(store, REACTOR_SCHEDULE_PREFIX, &schedules)?;
    replace_reactor_namespace(store, REACTOR_DJEHUTY_CURSOR_PREFIX, &cursors)?;
    Ok(())
}

fn encode_reactor_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len().saturating_mul(2));
    for byte in bytes {
        encoded.push(char::from(HEX[usize::from(byte >> 4)]));
        encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    encoded
}

fn decode_reactor_bytes(encoded: &str) -> Result<Vec<u8>, &'static str> {
    if !encoded.len().is_multiple_of(2) {
        return Err("hex value has odd length");
    }
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = decode_hex_nibble(pair[0]).ok_or("hex value contains a non-hex digit")?;
            let low = decode_hex_nibble(pair[1]).ok_or("hex value contains a non-hex digit")?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn decode_hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn replace_reactor_namespace(
    store: &TieredStore,
    prefix: &[u8],
    entries: &[(Vec<u8>, Vec<u8>)],
) -> SnapshotResult<()> {
    let borrowed = entries
        .iter()
        .map(|(key, value)| (key.as_slice(), value.as_slice()))
        .collect::<Vec<_>>();
    store
        .replace_meta_prefix(prefix, &borrowed)
        .map_err(|error| {
            SnapshotError::Agent(format!(
                "wake: replace reactor metadata {}: {error}",
                String::from_utf8_lossy(prefix)
            ))
        })
}

fn validate_reactor_metadata_entry(key: &str, value: &serde_json::Value) -> SnapshotResult<()> {
    if key.chars().any(char::is_control) {
        return Err(SnapshotError::Agent(
            "reactor snapshot metadata key contains a control character".into(),
        ));
    }
    if let Some(schedule_id) = key.strip_prefix("reactor:schedule:") {
        if schedule_id.is_empty() {
            return Err(SnapshotError::Agent(
                "reactor snapshot contains an empty schedule id".into(),
            ));
        }
        #[cfg(feature = "daemon")]
        let payload_id = {
            let schedule =
                serde_json::from_value::<crate::agent::reactor::ReactorSchedule>(value.clone())
                    .map_err(|error| {
                        SnapshotError::Agent(format!(
                            "reactor snapshot schedule {schedule_id} is invalid: {error}"
                        ))
                    })?;
            schedule.validate().map_err(|error| {
                SnapshotError::Agent(format!(
                    "reactor snapshot schedule {schedule_id} is invalid: {error}"
                ))
            })?;
            schedule.id
        };
        #[cfg(not(feature = "daemon"))]
        let payload_id = value
            .get("id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                SnapshotError::Agent(format!(
                    "reactor snapshot schedule {schedule_id} has no string id"
                ))
            })?
            .to_owned();
        if payload_id != schedule_id {
            return Err(SnapshotError::Agent(format!(
                "reactor snapshot schedule key {schedule_id} does not match payload id {payload_id}"
            )));
        }
        return Ok(());
    }

    if let Some(cursor_key) = key.strip_prefix("reactor:djehuty:cursor:") {
        let Some((thread_id, schedule_id)) = cursor_key.split_once(':') else {
            return Err(SnapshotError::Agent(format!(
                "reactor snapshot cursor key {key} is malformed"
            )));
        };
        if uuid::Uuid::parse_str(thread_id).is_err()
            || schedule_id.is_empty()
            || schedule_id.chars().any(char::is_control)
        {
            return Err(SnapshotError::Agent(format!(
                "reactor snapshot cursor key {key} is malformed"
            )));
        }
        let Some(message_id) = value
            .get("last_message_id")
            .and_then(serde_json::Value::as_str)
        else {
            return Err(SnapshotError::Agent(format!(
                "reactor snapshot cursor {key} has no last_message_id"
            )));
        };
        uuid::Uuid::parse_str(message_id).map_err(|error| {
            SnapshotError::Agent(format!(
                "reactor snapshot cursor {key} has invalid message id: {error}"
            ))
        })?;
        return Ok(());
    }

    Err(SnapshotError::Agent(format!(
        "reactor snapshot metadata key {key} is outside the allow-list"
    )))
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
    let crate::provenance::DerivationKind::RuleInference { antecedents, .. } = &record.kind else {
        return Ok(());
    };
    let Some(first) = antecedents.first() else {
        return Ok(());
    };
    if first.get() != crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL {
        return Ok(());
    }

    let is_v2 = antecedents
        .get(1)
        .is_some_and(|marker| marker.get() == crate::provenance::RULE_APPLICATION_V2_SENTINEL);
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
) -> SnapshotResult<(
    Triple,
    Vec<SymbolMeta>,
    Option<crate::provenance::ProvenanceRecord>,
)> {
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
        validate_compartment_id(compartment_id).map_err(|error| invalid_tail(error.to_string()))?;
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
            let id = crate::provenance::ProvenanceId::new(raw_id)
                .ok_or_else(|| invalid_tail("assert provenance ID must be non-zero"))?;
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

fn remap_tail_number(
    value: &mut serde_json::Value,
    mapping: &std::collections::HashMap<u64, u64>,
) -> bool {
    let Some(id) = value.as_u64() else {
        return false;
    };
    let Some(canonical) = mapping.get(&id).copied() else {
        return false;
    };
    *value = serde_json::Value::from(canonical);
    true
}

/// Rewrite replayable symbol references through any WI-338 merge audit in the
/// same post-watermark tail. The audit is appended by the repair transaction,
/// after the historical records it repairs, so collecting the complete map
/// first makes the very first repaired wake safe before a new snapshot exists.
fn rewrite_tail_symbol_value(
    value: &mut serde_json::Value,
    mapping: &std::collections::HashMap<u64, u64>,
    canonical_symbols: &std::collections::HashMap<u64, SymbolMeta>,
) -> SnapshotResult<bool> {
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
        "tool",
        "source",
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
    ];

    match value {
        serde_json::Value::Array(values) => {
            let mut changed = false;
            for value in values {
                changed |= rewrite_tail_symbol_value(value, mapping, canonical_symbols)?;
            }
            Ok(changed)
        }
        serde_json::Value::Object(object) => {
            let mut changed = false;
            for (key, child) in object {
                if key == "symbols" {
                    let serde_json::Value::Array(symbols) = child else {
                        return Err(invalid_tail("symbol metadata envelope is not an array"));
                    };
                    let mut seen = std::collections::HashSet::with_capacity(symbols.len());
                    let mut rewritten = Vec::with_capacity(symbols.len());
                    for mut symbol in std::mem::take(symbols) {
                        let id = symbol
                            .get("id")
                            .and_then(serde_json::Value::as_u64)
                            .ok_or_else(|| invalid_tail("symbol metadata has no valid ID"))?;
                        let final_id = mapping.get(&id).copied().unwrap_or(id);
                        if let Some(canonical) = canonical_symbols.get(&final_id) {
                            symbol = serde_json::to_value(canonical).map_err(|error| {
                                invalid_tail(format!(
                                    "could not encode canonical symbol {final_id}: {error}"
                                ))
                            })?;
                            changed |= final_id != id;
                        } else if final_id != id {
                            return Err(invalid_tail(format!(
                                "symbol merge maps {id} to missing canonical {final_id}"
                            )));
                        }
                        if seen.insert(final_id) {
                            rewritten.push(symbol);
                        } else {
                            changed = true;
                        }
                    }
                    *symbols = rewritten;
                } else if SCALAR_FIELDS.contains(&key.as_str()) {
                    changed |= remap_tail_number(child, mapping);
                    changed |= rewrite_tail_symbol_value(child, mapping, canonical_symbols)?;
                } else if ARRAY_FIELDS.contains(&key.as_str()) {
                    if let serde_json::Value::Array(values) = child {
                        for value in values {
                            changed |= remap_tail_number(value, mapping);
                            changed |=
                                rewrite_tail_symbol_value(value, mapping, canonical_symbols)?;
                        }
                    }
                } else {
                    changed |= rewrite_tail_symbol_value(child, mapping, canonical_symbols)?;
                }
            }
            Ok(changed)
        }
        _ => Ok(false),
    }
}

fn collect_tail_canonical_symbols(
    value: &serde_json::Value,
    base_symbols: &std::collections::HashMap<u64, SymbolMeta>,
    tail_symbols: &mut std::collections::HashMap<u64, SymbolMeta>,
) -> SnapshotResult<()> {
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                collect_tail_canonical_symbols(value, base_symbols, tail_symbols)?;
            }
        }
        serde_json::Value::Object(object) => {
            for (key, child) in object {
                if key != "symbols" {
                    collect_tail_canonical_symbols(child, base_symbols, tail_symbols)?;
                    continue;
                }
                let serde_json::Value::Array(symbols) = child else {
                    return Err(invalid_tail("symbol metadata envelope is not an array"));
                };
                for symbol in symbols {
                    let Some(id) = symbol.get("id").and_then(serde_json::Value::as_u64) else {
                        return Err(invalid_tail("symbol metadata has no valid ID"));
                    };
                    let Some(base) = base_symbols.get(&id) else {
                        continue;
                    };
                    let meta: SymbolMeta =
                        serde_json::from_value(symbol.clone()).map_err(|error| {
                            invalid_tail(format!("could not decode canonical symbol {id}: {error}"))
                        })?;
                    if meta.kind != base.kind || meta.label != base.label {
                        return Err(invalid_tail(format!(
                            "canonical symbol {id} conflicts with repaired snapshot metadata"
                        )));
                    }
                    if let Some(existing) = tail_symbols.get(&id) {
                        if existing.kind != meta.kind
                            || existing.label != meta.label
                            || existing.created_at != meta.created_at
                            || existing.source != meta.source
                        {
                            return Err(invalid_tail(format!(
                                "canonical symbol {id} has conflicting replay metadata"
                            )));
                        }
                    } else {
                        tail_symbols.insert(id, meta);
                    }
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn rewrite_tail_after_symbol_merges(
    tail: &mut [akhomed_store::TailMutation],
    base_symbols: &[SymbolMeta],
) -> SnapshotResult<()> {
    let mut mapping = std::collections::HashMap::new();
    for mutation in tail.iter().filter(|row| row.kind == "symbol_merge") {
        let parse_id = |side: &str| {
            mutation
                .payload
                .get(side)
                .and_then(|value| value.get("id"))
                .and_then(serde_json::Value::as_u64)
                .filter(|id| *id > 0 && !is_reserved_symbol_id(*id))
                .ok_or_else(|| invalid_tail(format!("symbol merge has invalid {side} ID")))
        };
        let canonical = parse_id("canonical")?;
        let duplicate = parse_id("duplicate")?;
        if canonical >= duplicate {
            return Err(invalid_tail(format!(
                "symbol merge {duplicate} -> {canonical} is not canonical"
            )));
        }
        if mapping.insert(duplicate, canonical).is_some() {
            return Err(invalid_tail(format!(
                "symbol merge repeats duplicate ID {duplicate}"
            )));
        }
    }
    if mapping.is_empty() {
        return Ok(());
    }
    if let Some(chained) = mapping.values().find(|id| mapping.contains_key(id)) {
        return Err(invalid_tail(format!(
            "symbol merge contains chained canonical ID {chained}"
        )));
    }

    let mut canonical_symbols: std::collections::HashMap<u64, SymbolMeta> = base_symbols
        .iter()
        .filter(|meta| mapping.values().any(|id| *id == meta.id.get()))
        .map(|meta| (meta.id.get(), meta.clone()))
        .collect();
    for canonical in mapping.values() {
        if !canonical_symbols.contains_key(canonical) {
            return Err(invalid_tail(format!(
                "symbol merge references missing canonical symbol {canonical}"
            )));
        }
    }

    // Relational symbol rows do not retain `created_at` or `source`. If the
    // tail contains richer metadata for a canonical symbol, use one consistent
    // copy for both that symbol and every loser rewritten to it. Otherwise an
    // earlier loser envelope followed by a canonical envelope could restore
    // the same ID with conflicting metadata on the first repaired wake.
    let mut tail_canonical_symbols = std::collections::HashMap::new();
    for mutation in tail.iter().filter(|row| {
        matches!(
            row.kind.as_str(),
            "assert" | "retract" | "replace" | "retract_batch"
        )
    }) {
        collect_tail_canonical_symbols(
            &mutation.payload,
            &canonical_symbols,
            &mut tail_canonical_symbols,
        )?;
    }
    canonical_symbols.extend(tail_canonical_symbols);

    for mutation in tail {
        if matches!(
            mutation.kind.as_str(),
            "assert" | "retract" | "replace" | "retract_batch"
        ) {
            rewrite_tail_symbol_value(&mut mutation.payload, &mapping, &canonical_symbols)?;
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
        labels.insert(
            crate::registry::normalize_label_key(&meta.label),
            meta.id.get(),
        );
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
    let next = max_id
        .checked_add(1)
        .ok_or_else(|| invalid_tail("post-snapshot symbol ID space is exhausted"))?;
    if is_reserved_symbol_id(next) {
        return Err(invalid_tail(
            "post-snapshot symbol allocator would enter the reserved sentinel",
        ));
    }
    Ok(next.max(1))
}

fn validate_fact_symbols(facts: &[fact::Model], symbols: &[SymbolMeta]) -> SnapshotResult<()> {
    let ids: std::collections::HashSet<u64> = symbols.iter().map(|meta| meta.id.get()).collect();
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
                    SnapshotError::Agent(format!("wake: fact {field} has invalid symbol ID {raw}"))
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
    let ids: std::collections::HashSet<SymbolId> = symbols.iter().map(|meta| meta.id).collect();
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
                SnapshotError::Agent(format!("wake: fact {field} has invalid symbol ID {raw}"))
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

type ReplayEdge = (SymbolId, SymbolId, SymbolId);

enum FutureEdgeTouch {
    Assert(Triple),
    NonAssert,
}

#[derive(Default)]
struct FutureTouchIndex {
    by_edge: std::collections::HashMap<ReplayEdge, Vec<(usize, FutureEdgeTouch)>>,
}

impl FutureTouchIndex {
    fn push(&mut self, edge: ReplayEdge, index: usize, touch: FutureEdgeTouch) {
        self.by_edge.entry(edge).or_default().push((index, touch));
    }

    /// Return the first transition for `edge` after `index`. The vectors are
    /// populated in WAL order, so lookup is logarithmic rather than rescanning
    /// the remaining tail once for every edge in a retract batch.
    fn first_after(&self, edge: ReplayEdge, index: usize) -> Option<&FutureEdgeTouch> {
        let touches = self.by_edge.get(&edge)?;
        let position = touches.partition_point(|(touch_index, _)| *touch_index <= index);
        touches.get(position).map(|(_, touch)| touch)
    }
}

/// Parse the ordered tail once and index every edge transition. This both
/// bounds overlap lookup to O(log n) per mismatched edge and validates future
/// known mutations before replay changes the graph.
fn build_future_touch_index(
    engine: &Engine,
    tail: &[akhomed_store::TailMutation],
) -> SnapshotResult<FutureTouchIndex> {
    let mut touches = FutureTouchIndex::default();
    for (index, mutation) in tail.iter().enumerate() {
        match mutation.kind.as_str() {
            "assert" => {
                let triple = prepare_assert_for_replay(engine, &mutation.payload)?;
                touches.push(
                    (triple.subject, triple.predicate, triple.object),
                    index,
                    FutureEdgeTouch::Assert(triple),
                );
            }
            "retract" => touches.push(
                payload_triple_ids(&mutation.payload)?,
                index,
                FutureEdgeTouch::NonAssert,
            ),
            "retract_batch" => {
                for triple in parse_replay_complete_retract_batch(&mutation.payload)?.1 {
                    touches.push(
                        (triple.subject, triple.predicate, triple.object),
                        index,
                        FutureEdgeTouch::NonAssert,
                    );
                }
            }
            "replace" => {
                let replacement = parse_replay_complete_replace(&mutation.payload)?;
                for retract in &replacement.retracts {
                    touches.push(
                        retract_triple_ids(retract)?,
                        index,
                        FutureEdgeTouch::NonAssert,
                    );
                }
                // Compound replacement semantics are deliberately not
                // weakened, even if its assertion is byte-equivalent.
                touches.push(
                    payload_triple_ids(&replacement.assertion)?,
                    index,
                    FutureEdgeTouch::NonAssert,
                );
            }
            _ => {}
        }
    }
    Ok(touches)
}

/// Decide whether a mismatched retract is a safe snapshot/WAL overlap. The
/// first later mutation touching the same edge must be an exact assertion of
/// the edge already present in the snapshot; any other transition keeps the
/// fail-closed mismatch behavior.
fn later_assertion_matches_snapshot_edge(
    actual: &Triple,
    index: usize,
    future_touches: &FutureTouchIndex,
) -> bool {
    matches!(
        future_touches.first_after(
            (actual.subject, actual.predicate, actual.object),
            index,
        ),
        Some(FutureEdgeTouch::Assert(assertion)) if assertion == actual
    )
}

/// Apply one replayed WAL mutation with its ordered future visible for the
/// narrow snapshot-overlap reconciliation above. Unknown future kinds are
/// ignored, while malformed known KG mutations fail wake so corruption cannot
/// silently drop a fact. Microtheory/compartment replay is deferred (38d).
fn apply_mutation_with_future(
    engine: &Engine,
    m: &akhomed_store::TailMutation,
    index: usize,
    future_touches: &FutureTouchIndex,
) -> SnapshotResult<bool> {
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
            // Legacy single-edge record: its writer (pre-`retract_batch`
            // `remove_triple`) logged only the primary edge and relied on
            // replay re-deriving the unconditional object-symbol cascade.
            // Replay must reproduce that writer's semantics, so this is the
            // one place that keeps `RetractionScope::Symbol`.
            engine
                .retract_object_symbol(triple.0, triple.1, triple.2)
                .map_err(|error| SnapshotError::Engine(error.to_string()))?;
            Ok(true)
        }
        "retract_batch" => {
            // Decode and validate the complete batch before applying its first
            // edge. `prepare_tail_restore` has already merged the closed symbol
            // envelope into the registry, so every reference must resolve.
            let (_, retracts, _, _) = parse_replay_complete_retract_batch(&m.payload)?;
            let mut snapshot_overlap = std::collections::HashSet::new();
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
                    let id = crate::provenance::ProvenanceId::new(raw_id).ok_or_else(|| {
                        invalid_tail("retract batch provenance ID must be non-zero")
                    })?;
                    engine.get_provenance(id).map_err(|error| {
                        invalid_tail(format!(
                            "retract batch references unavailable provenance ID {raw_id}: {error}"
                        ))
                    })?;
                }
                if let Some(actual) = actual
                    && actual != *triple
                {
                    if later_assertion_matches_snapshot_edge(&actual, index, future_touches) {
                        snapshot_overlap.insert((triple.subject, triple.predicate, triple.object));
                    } else {
                        return Err(invalid_tail(format!(
                            "retract batch edge ({}, {}, {}) metadata does not match replay state",
                            triple.subject.get(),
                            triple.predicate.get(),
                            triple.object.get()
                        )));
                    }
                }
            }
            for triple in &retracts {
                if snapshot_overlap.contains(&(triple.subject, triple.predicate, triple.object)) {
                    continue;
                }
                engine
                    .replay_retract_edge_exact(triple)
                    .map_err(|error| SnapshotError::Engine(error.to_string()))?;
            }
            for (subject, predicate, object) in snapshot_overlap {
                tracing::info!(
                    subject = subject.get(),
                    predicate = predicate.get(),
                    object = object.get(),
                    "wake: reconciled snapshot/WAL overlap before matching assertion"
                );
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
        "turn" | "symbol_merge" => Ok(false),
        _ => Ok(false),
    }
}

/// Strict single-record replay used by focused tests and callers without an
/// ordered tail. No future means metadata mismatches still fail closed.
#[cfg(test)]
fn apply_mutation(engine: &Engine, m: &akhomed_store::TailMutation) -> SnapshotResult<bool> {
    apply_mutation_with_future(engine, m, 0, &FutureTouchIndex::default())
}

/// Replay one ordered tail, exposing only later records to each transition.
fn apply_mutation_tail(
    engine: &Engine,
    tail: &[akhomed_store::TailMutation],
) -> SnapshotResult<usize> {
    let future_touches = build_future_touch_index(engine, tail)?;
    let mut replayed = 0;
    for (index, mutation) in tail.iter().enumerate() {
        if apply_mutation_with_future(engine, mutation, index, &future_touches)? {
            replayed += 1;
        }
    }
    Ok(replayed)
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

    #[test]
    fn duplicate_symbol_repair_plan_uses_wake_normalization_and_lowest_id() {
        let akh_id = Uuid::nil();
        let row = |id, label: &str| symbol::Model {
            akh_id,
            id,
            kind: "Entity".into(),
            glyph_codepoint: None,
            label: label.into(),
        };
        let plan = build_symbol_repair_plan(
            akh_id,
            vec![
                row(30, "O’Reilly"),
                row(12, "O'Reilly"),
                row(7, "OReilly"),
                row(40, "distinct"),
            ],
            41,
        )
        .unwrap();
        assert_eq!(plan.duplicate_groups, 1);
        assert_eq!(plan.extra_symbols, 2);
        assert_eq!(
            plan.merges
                .iter()
                .map(|merge| (merge.duplicate_id, merge.canonical_id))
                .collect::<Vec<_>>(),
            vec![(12, 7), (30, 7)]
        );
        assert_eq!(plan.allocator.max_surviving_symbol_id, Some(40));
        assert_eq!(plan.allocator.allocator_next_after, 41);
        assert!(!plan.allocator.advanced);
    }

    #[test]
    fn symbol_repair_plan_advances_stale_allocator_without_duplicate_merges() {
        let akh_id = Uuid::nil();
        let row = |id, label: &str| symbol::Model {
            akh_id,
            id,
            kind: "Entity".into(),
            glyph_codepoint: None,
            label: label.into(),
        };
        let plan = build_symbol_repair_plan(
            akh_id,
            vec![row(7, "existing"), row(6_558, "restored max")],
            2_826,
        )
        .unwrap();

        assert_eq!(plan.duplicate_groups, 0);
        assert!(plan.merges.is_empty());
        assert_eq!(plan.allocator.max_surviving_symbol_id, Some(6_558));
        assert_eq!(plan.allocator.required_allocator_next, 6_559);
        assert_eq!(plan.allocator.allocator_next_before, 2_826);
        assert_eq!(plan.allocator.allocator_next_after, 6_559);
        assert!(plan.allocator.advanced);

        let second = plan_symbol_allocator_repair(6_559, Some(6_558)).unwrap();
        assert_eq!(second.allocator_next_after, 6_559);
        assert!(!second.advanced);
    }

    #[test]
    fn allocator_repair_rejects_exhaustion_and_reserved_markers() {
        let overflow = required_allocator_next(Some(u64::MAX)).unwrap_err();
        assert!(overflow.to_string().contains("exhausted"), "{overflow}");

        let reserved = required_allocator_next(Some(u64::MAX - 2)).unwrap_err();
        assert!(reserved.to_string().contains("reserved"), "{reserved}");

        let unstoreable = plan_symbol_allocator_repair(1, Some(i64::MAX)).unwrap_err();
        assert!(
            unstoreable.to_string().contains("cannot be stored"),
            "{unstoreable}"
        );
    }

    #[test]
    fn symbol_merge_audit_rewrites_earlier_replay_complete_tail() {
        let canonical = replay_symbol(7, SymbolKind::Entity, "OReilly", 0);
        let tail_canonical = replay_symbol(7, SymbolKind::Entity, "OReilly", 11);
        let duplicate = replay_symbol(12, SymbolKind::Entity, "O’Reilly", 5);
        let predicate = replay_symbol(20, SymbolKind::Relation, "relation", 0);
        let object = replay_symbol(21, SymbolKind::Entity, "object", 0);
        let mut tail = vec![
            akhomed_store::TailMutation {
                kind: "assert".into(),
                payload: serde_json::json!({
                    "v": crate::mutation::ASSERT_MUTATION_VERSION,
                    "subject": duplicate.id.get(),
                    "predicate": predicate.id.get(),
                    "object": object.id.get(),
                    "confidence": 1.0,
                    "timestamp": 5,
                    "compartment": null,
                    "provenance_id": null,
                    "provenance": null,
                    "symbols": [duplicate, predicate.clone(), object.clone()],
                }),
            },
            akhomed_store::TailMutation {
                kind: "assert".into(),
                payload: serde_json::json!({
                    "v": crate::mutation::ASSERT_MUTATION_VERSION,
                    "subject": tail_canonical.id.get(),
                    "predicate": predicate.id.get(),
                    "object": object.id.get(),
                    "confidence": 1.0,
                    "timestamp": 11,
                    "compartment": null,
                    "provenance_id": null,
                    "provenance": null,
                    "symbols": [tail_canonical, predicate.clone(), object.clone()],
                }),
            },
            akhomed_store::TailMutation {
                kind: "symbol_merge".into(),
                payload: serde_json::json!({
                    "v": 1,
                    "canonical": {"id": canonical.id.get()},
                    "duplicate": {"id": 12},
                }),
            },
        ];
        let mut symbols = vec![canonical.clone(), predicate, object];

        rewrite_tail_after_symbol_merges(&mut tail, &symbols).unwrap();
        assert_eq!(tail[0].payload["subject"], canonical.id.get());
        assert_eq!(tail[0].payload["symbols"][0]["id"], canonical.id.get());
        assert_eq!(tail[0].payload["symbols"][0]["label"], canonical.label);
        assert_eq!(tail[0].payload["symbols"][0]["created_at"], 11);
        assert_eq!(tail[0].payload["symbols"][0], tail[1].payload["symbols"][0]);
        prepare_tail_restore(&tail, &mut symbols, &mut Vec::new()).unwrap();
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

        let mut duplicate = goal.clone();
        duplicate.status = GoalStatus::Dormant;
        duplicate.children.push(SymbolId::new(10).unwrap());
        let repeated = [goal, duplicate];
        let unique = unique_goals_by_symbol(&repeated);
        assert_eq!(
            unique.len(),
            1,
            "relational goal primary keys must be unique"
        );
        assert_eq!(unique[0].symbol_id, SymbolId::new(7).unwrap());
        assert!(matches!(unique[0].status, GoalStatus::Failed { .. }));
        assert_eq!(
            unique[0].children,
            vec![SymbolId::new(8).unwrap(), SymbolId::new(9).unwrap()],
            "first live entry is the deterministic authority"
        );

        let repeated_edges = [
            SymbolId::new(8).unwrap(),
            SymbolId::new(8).unwrap(),
            SymbolId::new(9).unwrap(),
            SymbolId::new(8).unwrap(),
        ];
        assert_eq!(
            unique_symbol_ids(&repeated_edges),
            vec![SymbolId::new(8).unwrap(), SymbolId::new(9).unwrap()],
            "relational edges retain first-seen order with unique primary keys"
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
        assert!(
            build_drive_system(&[]).is_none(),
            "no rows → None (default fallback)"
        );
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
        assert_eq!(
            parse_consolidation_phase("DreamState"),
            ConsolidationPhase::DreamState
        );
        assert_eq!(
            parse_consolidation_phase("Consolidation"),
            ConsolidationPhase::Consolidation
        );
        assert_eq!(
            parse_consolidation_phase("garbage"),
            ConsolidationPhase::Wake
        );
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
            episodes_by_contact: [("alice".to_string(), vec![ep2, ep1])]
                .into_iter()
                .collect(),
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

    #[cfg(feature = "daemon")]
    #[test]
    fn reactor_metadata_roundtrips_and_replaces_stale_namespaces() {
        use crate::agent::reactor::{ReactorAction, ReactorEvent, ReactorSchedule, ScheduleRule};

        let source_root = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(source_root.path().to_path_buf()),
            ..Default::default()
        })
        .unwrap();
        let mut schedule = ReactorSchedule::new(
            "board-watch",
            "Board watch",
            ScheduleRule::Event {
                source: "djehuty:thread:01a04e2c-29ff-7230-967d-2534f39af690".into(),
            },
            ReactorAction::DjehutyReply {
                thread_id: "01a04e2c-29ff-7230-967d-2534f39af690".into(),
                prompt_prefix: "read only".into(),
                agent_id: Some("thoth".into()),
                allowed_message_types: vec!["board/note@v1".into()],
                max_prompt_chars: 12_000,
                max_reply_chars: 4_000,
                min_reply_interval_seconds: 30,
            },
            10,
        )
        .unwrap();
        schedule.state.pending_events.push_back(ReactorEvent {
            source: "djehuty:thread:01a04e2c-29ff-7230-967d-2534f39af690".into(),
            id: "01a04e4c-a665-7ca0-b551-b07851328fdf".into(),
            observed_at: 42,
            payload: "{\"message_type\":\"board/note@v1\"}".into(),
        });
        schedule
            .state
            .processed_event_keys
            .push_back("djehuty:thread:01a04e2c-29ff-7230-967d-2534f39af690\0processed".into());
        schedule.state.lease_until = Some(77);
        schedule.state.retry_at = Some(88);
        schedule.state.attempt = 2;
        let schedule_value = serde_json::to_value(&schedule).unwrap();
        let schedule_bytes = serde_json::to_vec(&schedule_value).unwrap();
        engine
            .store()
            .put_meta(b"reactor:schedule:board-watch", &schedule_bytes)
            .unwrap();
        let cursor_value = serde_json::json!({
            "last_message_id": "01a04e49-77ef-7421-bd09-0250e72a97f6"
        });
        let cursor_bytes = serde_json::to_vec(&cursor_value).unwrap();
        engine
            .store()
            .put_meta(
                b"reactor:djehuty:cursor:01a04e2c-29ff-7230-967d-2534f39af690:board-watch",
                &cursor_bytes,
            )
            .unwrap();

        let captured = capture_reactor_metadata(&engine).unwrap();
        assert_eq!(captured.len(), 2);
        let extra = serde_json::json!({
            "reactor": {
                "version": REACTOR_SNAPSHOT_VERSION,
                "metadata": captured,
            }
        });
        assert!(
            !serde_json::to_string(&extra).unwrap().contains("\\u0000"),
            "portable reactor metadata must be accepted by PostgreSQL jsonb"
        );

        let restored_root = tempfile::tempdir().unwrap();
        let restored = TieredStore::with_persistence(restored_root.path(), "symbols").unwrap();
        restored
            .put_meta(b"reactor:schedule:stale", br#"{"id":"stale"}"#)
            .unwrap();
        restored
            .put_meta(
                b"reactor:djehuty:cursor:01a04e2c-29ff-7230-967d-2534f39af690:stale",
                br#"{"last_message_id":"01a04e49-77ef-7421-bd09-0250e72a97f6"}"#,
            )
            .unwrap();
        restored.put_meta(b"unrelated", b"preserve").unwrap();

        restore_reactor_metadata(&restored, Some(&extra)).unwrap();

        let restored_schedule = restored
            .get_meta(b"reactor:schedule:board-watch")
            .unwrap()
            .unwrap();
        assert_eq!(restored_schedule, schedule_bytes);
        let restored_cursor = restored
            .get_meta(b"reactor:djehuty:cursor:01a04e2c-29ff-7230-967d-2534f39af690:board-watch")
            .unwrap()
            .unwrap();
        assert_eq!(restored_cursor, cursor_bytes);
        assert!(
            restored
                .get_meta(b"reactor:schedule:stale")
                .unwrap()
                .is_none()
        );
        assert_eq!(
            restored.get_meta(b"unrelated").unwrap().as_deref(),
            Some(b"preserve".as_slice())
        );
    }

    #[test]
    fn budget_ledger_roundtrips_through_snapshot_extra() {
        use crate::agent::autonomy::{BUDGET_LEDGER_KEY, BudgetLedger, UtcDay};
        let dir = tempfile::tempdir().unwrap();
        let engine = Engine::new(EngineConfig {
            data_dir: Some(dir.path().to_path_buf()),
            ..EngineConfig::default()
        })
        .unwrap();
        assert_eq!(capture_budget_ledger(&engine), None, "no ledger yet");
        let ledger = BudgetLedger {
            day: UtcDay::from_unix_secs(1_790_251_200),
            tokens_used: 12_345,
            turns_used: 7,
        };
        engine
            .store()
            .put_meta(BUDGET_LEDGER_KEY, &serde_json::to_vec(&ledger).unwrap())
            .unwrap();
        let captured = capture_budget_ledger(&engine).expect("captured");
        assert_eq!(captured, ledger);

        let extra = serde_json::json!({ "autonomy": { "budget_ledger": captured } });
        let restored_dir = tempfile::tempdir().unwrap();
        let store = TieredStore::with_persistence(restored_dir.path(), "snapshot-ledger").unwrap();
        restore_budget_ledger(&store, Some(&extra));
        let bytes = store
            .get_meta(BUDGET_LEDGER_KEY)
            .unwrap()
            .expect("restored");
        assert_eq!(
            serde_json::from_slice::<BudgetLedger>(&bytes).unwrap(),
            ledger
        );

        // Older snapshots (no envelope, or a null ledger) leave the store alone;
        // a malformed ledger is skipped, not fatal.
        restore_budget_ledger(&store, Some(&serde_json::json!({})));
        restore_budget_ledger(
            &store,
            Some(&serde_json::json!({ "autonomy": { "budget_ledger": null } })),
        );
        restore_budget_ledger(
            &store,
            Some(&serde_json::json!({ "autonomy": { "budget_ledger": "bogus" } })),
        );
        let bytes = store.get_meta(BUDGET_LEDGER_KEY).unwrap().expect("kept");
        assert_eq!(
            serde_json::from_slice::<BudgetLedger>(&bytes).unwrap(),
            ledger
        );
    }

    #[test]
    fn provider_gate_roundtrips_through_snapshot_extra() {
        use crate::agent::autonomy::UnixSecs;
        use crate::agent::provider_failure::{
            CooldownLadder, CoolingCause, PROVIDER_GATE_KEY, ProviderFailureClass, ProviderGate,
            ProviderGateState,
        };
        let dir = tempfile::tempdir().unwrap();
        let engine = Arc::new(
            Engine::new(EngineConfig {
                data_dir: Some(dir.path().to_path_buf()),
                ..EngineConfig::default()
            })
            .unwrap(),
        );
        assert_eq!(capture_provider_gate(&engine), None, "no gate yet");
        let now = UnixSecs::from_secs(1_790_251_200);
        let gate = ProviderGate::open(&engine, CooldownLadder::default()).unwrap();
        gate.record_failure(ProviderFailureClass::Quota, now)
            .unwrap();
        let captured = capture_provider_gate(&engine).expect("captured");
        assert!(matches!(
            captured,
            ProviderGateState::Cooling {
                cause: CoolingCause::Quota,
                step: 0,
                ..
            }
        ));

        let extra = serde_json::json!({ "provider": { "gate": captured } });
        let restored_dir = tempfile::tempdir().unwrap();
        let store =
            TieredStore::with_persistence(restored_dir.path(), "snapshot-provider-gate").unwrap();
        restore_provider_gate(&store, Some(&extra));
        let bytes = store
            .get_meta(PROVIDER_GATE_KEY)
            .unwrap()
            .expect("restored");
        assert_eq!(
            serde_json::from_slice::<ProviderGateState>(&bytes).unwrap(),
            captured
        );

        // Older snapshots (no envelope, or a null gate) leave the store alone;
        // a malformed gate is skipped, not fatal.
        restore_provider_gate(&store, Some(&serde_json::json!({})));
        restore_provider_gate(
            &store,
            Some(&serde_json::json!({ "provider": { "gate": null } })),
        );
        restore_provider_gate(
            &store,
            Some(&serde_json::json!({ "provider": { "gate": "bogus" } })),
        );
        let bytes = store.get_meta(PROVIDER_GATE_KEY).unwrap().expect("kept");
        assert_eq!(
            serde_json::from_slice::<ProviderGateState>(&bytes).unwrap(),
            captured
        );
    }

    #[test]
    fn last_run_log_roundtrips_through_snapshot_extra() {
        use crate::agent::autonomy::{
            DaemonTask, LAST_RUN_LOG_KEY, LastRun, LastRunCause, LastRunLog, LastRunStore, UnixSecs,
        };
        let dir = tempfile::tempdir().unwrap();
        let engine = Arc::new(
            Engine::new(EngineConfig {
                data_dir: Some(dir.path().to_path_buf()),
                ..EngineConfig::default()
            })
            .unwrap(),
        );
        assert_eq!(capture_last_run_log(&engine), None, "no log yet");
        let ran = LastRun {
            at: UnixSecs::from_secs(1_790_251_200),
            cause: LastRunCause::Ran,
        };
        LastRunStore::open(&engine)
            .unwrap()
            .record(DaemonTask::Librarian, ran)
            .unwrap();
        let captured = capture_last_run_log(&engine).expect("captured");
        assert_eq!(captured.get(DaemonTask::Librarian), Some(ran));

        let extra = serde_json::json!({ "autonomy": { "last_run": captured } });
        let restored_dir = tempfile::tempdir().unwrap();
        let store =
            TieredStore::with_persistence(restored_dir.path(), "snapshot-last-run").unwrap();
        restore_last_run_log(&store, Some(&extra));
        let bytes = store.get_meta(LAST_RUN_LOG_KEY).unwrap().expect("restored");
        let restored: LastRunLog = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(restored, captured);

        // Older snapshots (no envelope, or a null log) leave the store alone;
        // a malformed log is skipped, not fatal.
        restore_last_run_log(&store, Some(&serde_json::json!({})));
        restore_last_run_log(
            &store,
            Some(&serde_json::json!({ "autonomy": { "last_run": null } })),
        );
        restore_last_run_log(
            &store,
            Some(&serde_json::json!({ "autonomy": { "last_run": "bogus" } })),
        );
        let bytes = store.get_meta(LAST_RUN_LOG_KEY).unwrap().expect("kept");
        assert_eq!(
            serde_json::from_slice::<LastRunLog>(&bytes).unwrap(),
            captured
        );
    }

    #[test]
    fn reactor_metadata_restore_rejects_keys_outside_allow_list() {
        let tmp = tempfile::tempdir().unwrap();
        let store = TieredStore::with_persistence(tmp.path(), "symbols").unwrap();
        let value_hex = encode_reactor_bytes(br#"{"shadow":"overwrite"}"#);
        let extra = serde_json::json!({
            "reactor": {
                "version": REACTOR_SNAPSHOT_VERSION,
                "metadata": [{
                    "key": "agent:psyche",
                    "value_hex": value_hex
                }]
            }
        });
        let error = restore_reactor_metadata(&store, Some(&extra)).unwrap_err();
        assert!(error.to_string().contains("outside the allow-list"));
    }

    #[test]
    fn reactor_metadata_restore_accepts_legacy_v1_envelope() {
        let tmp = tempfile::tempdir().unwrap();
        let store = TieredStore::with_persistence(tmp.path(), "symbols").unwrap();
        let key = "reactor:djehuty:cursor:01a04e2c-29ff-7230-967d-2534f39af690:legacy";
        let extra = serde_json::json!({
            "reactor": {
                "version": 1,
                "metadata": [{
                    "key": key,
                    "value": {
                        "last_message_id": "01a04e49-77ef-7421-bd09-0250e72a97f6"
                    }
                }]
            }
        });

        restore_reactor_metadata(&store, Some(&extra)).unwrap();

        let restored = store.get_meta(key.as_bytes()).unwrap().unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&restored).unwrap(),
            serde_json::json!({
                "last_message_id": "01a04e49-77ef-7421-bd09-0250e72a97f6"
            })
        );
    }

    #[test]
    fn reactor_metadata_restore_rejects_malformed_hex() {
        let tmp = tempfile::tempdir().unwrap();
        let store = TieredStore::with_persistence(tmp.path(), "symbols").unwrap();
        let extra = serde_json::json!({
            "reactor": {
                "version": REACTOR_SNAPSHOT_VERSION,
                "metadata": [{
                    "key": "reactor:djehuty:cursor:01a04e2c-29ff-7230-967d-2534f39af690:bad",
                    "value_hex": "0g"
                }]
            }
        });

        let error = restore_reactor_metadata(&store, Some(&extra)).unwrap_err();
        assert!(error.to_string().contains("non-hex digit"));
    }

    #[test]
    fn reactor_metadata_rejects_control_characters_in_keys() {
        let value = serde_json::json!({
            "last_message_id": "01a04e49-77ef-7421-bd09-0250e72a97f6"
        });
        let error = validate_reactor_metadata_entry(
            "reactor:djehuty:cursor:01a04e2c-29ff-7230-967d-2534f39af690:bad\0id",
            &value,
        )
        .unwrap_err();
        assert!(error.to_string().contains("control character"));
    }

    #[cfg(feature = "daemon")]
    #[test]
    fn reactor_metadata_restore_rejects_semantically_invalid_schedule() {
        use crate::agent::reactor::{ReactorAction, ReactorSchedule, ScheduleRule};

        let schedule = ReactorSchedule::new(
            "invalid-interval",
            "Invalid interval",
            ScheduleRule::Interval {
                every_seconds: 1,
                anchor: 0,
            },
            ReactorAction::LlmTurn {
                prompt: "observe".into(),
                agent_id: None,
                allow_tools: false,
                require_mutation: false,
                exit_condition: String::new(),
                report_thread: None,
            },
            0,
        )
        .unwrap();
        let mut value = serde_json::to_value(schedule).unwrap();
        value["rule"]["every_seconds"] = serde_json::json!(0);
        let error = validate_reactor_metadata_entry("reactor:schedule:invalid-interval", &value)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("interval must be at least one second")
        );
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
        assert_eq!(
            engine
                .get_provenance(ProvenanceId::new(41).unwrap())
                .unwrap()
                .depth,
            2
        );
        let edge = engine.triples_from(subject.id).pop().unwrap();
        assert_eq!(edge.confidence, 0.55);
        assert_eq!(edge.timestamp, 20);
        assert_eq!(edge.compartment_id, None);
        assert_eq!(
            edge.provenance_id, None,
            "the later untracked assertion is authoritative"
        );

        let fresh = engine
            .create_symbol(SymbolKind::Entity, "AfterTail")
            .unwrap();
        assert_eq!(
            fresh.id.get(),
            704,
            "symbol allocation must advance beyond the WAL tail"
        );
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
            prepare_tail_restore(&[missing_kind_symbol], &mut Vec::new(), &mut Vec::new()).is_err(),
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
            prepare_tail_restore(&[malformed_compartment], &mut Vec::new(), &mut Vec::new())
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
            validate_base_restore_image(std::slice::from_ref(&base_fact), &[], &symbols, true,)
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
                    SymbolId::new(crate::provenance::RULE_SUPPORT_TRIPLES_V1_SENTINEL).unwrap(),
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
        assert!(
            live.set_mutation_sink(Arc::clone(&wal) as Arc<dyn MutationSink>)
                .is_ok()
        );

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
        assert!(
            live.set_mutation_sink(Arc::clone(&wal) as Arc<dyn MutationSink>)
                .is_ok()
        );
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
        assert_eq!(
            retained[0].payload["provenance"].as_array().unwrap().len(),
            1
        );

        let tail = vec![akhomed_store::TailMutation {
            kind: retained[0].kind.as_str().into(),
            payload: retained[0].payload.clone(),
        }];
        let allocator_next =
            prepare_tail_restore(&tail, &mut restored_symbols, &mut restored_provenance).unwrap();
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
        replay.replace_all_provenance(&restored_provenance).unwrap();
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
        assert!(
            live.set_mutation_sink(Arc::clone(&wal) as Arc<dyn MutationSink>)
                .is_ok()
        );
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
        let allocator_next =
            prepare_tail_restore(&tail, &mut restored_symbols, &mut restored_provenance).unwrap();

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
        replay.replace_all_provenance(&restored_provenance).unwrap();
        replay.restore_triples(&captured_triples).unwrap();
        for mutation in &tail {
            assert!(apply_mutation(&replay, mutation).unwrap());
        }

        let by_key = |engine: &Engine| {
            let mut triples = engine.all_triples();
            triples.sort_by_key(|triple| (triple.subject, triple.predicate, triple.object));
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
    fn snapshot_overlap_skips_stale_retract_when_later_assert_matches_image() {
        let live = Engine::new(EngineConfig {
            dimension: crate::vsa::Dimension::TEST,
            ..Default::default()
        })
        .unwrap();
        let subject = live
            .create_symbol(SymbolKind::Entity, "overlap-subject")
            .unwrap()
            .id;
        let predicate = live
            .create_symbol(SymbolKind::Relation, "overlap-predicate")
            .unwrap()
            .id;
        let object = live
            .create_symbol(SymbolKind::Entity, "overlap-object")
            .unwrap()
            .id;
        let mut old = Triple::new(subject, predicate, object);
        old.timestamp = 10;
        live.add_triple(&old).unwrap();

        let sink = Arc::new(RecordingSink::default());
        assert!(
            live.set_mutation_sink(Arc::clone(&sink) as Arc<dyn MutationSink>)
                .is_ok()
        );
        live.remove_triple(subject, predicate, object).unwrap();
        let mut newer = Triple::new(subject, predicate, object);
        newer.timestamp = 20;
        live.add_triple(&newer).unwrap();
        let captured = live.all_triples();
        let mut tail = sink
            .0
            .lock()
            .unwrap()
            .iter()
            .map(|record| akhomed_store::TailMutation {
                kind: record.kind.as_str().into(),
                payload: record.payload.clone(),
            })
            .collect::<Vec<_>>();
        assert_eq!(tail.len(), 2);
        assert_eq!(tail[0].kind, "retract_batch");
        assert_eq!(tail[1].kind, "assert");
        tail.insert(
            1,
            akhomed_store::TailMutation {
                kind: "turn".into(),
                payload: serde_json::json!({"unrelated": true}),
            },
        );

        let restored_replay = || {
            let replay = Engine::new(EngineConfig {
                dimension: crate::vsa::Dimension::TEST,
                ..Default::default()
            })
            .unwrap();
            for (kind, label) in [
                (SymbolKind::Entity, "overlap-subject"),
                (SymbolKind::Relation, "overlap-predicate"),
                (SymbolKind::Entity, "overlap-object"),
            ] {
                replay.create_symbol(kind, label).unwrap();
            }
            replay.restore_triples(&captured).unwrap();
            replay
        };

        let replay = restored_replay();
        let replayed = apply_mutation_tail(&replay, &tail).unwrap();
        assert_eq!(replayed, 2);
        assert_eq!(replay.all_triples(), captured);

        let mut non_matching_tail = tail.clone();
        non_matching_tail[2].payload["timestamp"] = serde_json::json!(21);
        let rejected = restored_replay();
        assert!(apply_mutation_tail(&rejected, &non_matching_tail).is_err());
        assert_eq!(rejected.all_triples(), captured);

        let direct_retract = akhomed_store::TailMutation {
            kind: "retract".into(),
            payload: serde_json::json!({
                "subject": subject.get(),
                "predicate": predicate.get(),
                "object": object.get(),
            }),
        };
        let repeated_batch = tail[0].clone();
        let replacement = akhomed_store::TailMutation {
            kind: "replace".into(),
            payload: serde_json::json!({
                "v": crate::mutation::REPLACE_MUTATION_VERSION,
                "retracts": [{
                    "subject": subject.get(),
                    "predicate": predicate.get(),
                    "object": object.get(),
                }],
                "assertion": tail[2].payload.clone(),
            }),
        };
        for (name, first_same_edge_touch) in [
            ("retract", direct_retract),
            ("retract_batch", repeated_batch),
            ("replace", replacement),
        ] {
            let rejected = restored_replay();
            let guarded_tail = vec![tail[0].clone(), first_same_edge_touch, tail[2].clone()];
            assert!(
                apply_mutation_tail(&rejected, &guarded_tail).is_err(),
                "first same-edge {name} must fail closed"
            );
            assert_eq!(rejected.all_triples(), captured, "{name} changed graph");
        }
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
        let data =
            capture_snapshot_at_watermark(&agent, Uuid::new_v4(), Uuid::new_v4(), 37).unwrap();
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
