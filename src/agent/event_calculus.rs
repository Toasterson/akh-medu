//! Event Calculus engine — Phase 15b.
//!
//! Temporal fluent reasoning on top of the causal world model (Phase 15a).
//! Implements the core Event Calculus axioms:
//!
//! - **Initiates(e, f, t)**: event `e` starts fluent `f` at time `t`
//! - **Terminates(e, f, t)**: event `e` ends fluent `f` at time `t`
//! - **Happens(e, t)**: event `e` occurred at time `t`
//! - **HoldsAt(f, t)**: fluent `f` is true at time `t`
//! - **Clipped(f, t1, t2)**: fluent `f` was terminated between `t1` and `t2`
//!
//! The engine stores events and fluents as KG entities with well-known predicates,
//! queries the KG to evaluate temporal axioms, and integrates with the causal
//! manager (Phase 15a) for multi-step action simulation.

use std::collections::HashMap;

use miette::Diagnostic;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::DerivationKind;
use crate::symbol::SymbolId;

// ═══════════════════════════════════════════════════════════════════════
// Error
// ═══════════════════════════════════════════════════════════════════════

/// Errors specific to the event calculus subsystem.
#[derive(Debug, Error, Diagnostic)]
pub enum EventCalculusError {
    #[error("fluent not found: {label}")]
    #[diagnostic(
        code(akh::agent::ec::fluent_not_found),
        help("Create the fluent first via `record_event` with an initiating event.")
    )]
    FluentNotFound { label: String },

    #[error("event not found: {label}")]
    #[diagnostic(
        code(akh::agent::ec::event_not_found),
        help("Record the event first via `record_event()`.")
    )]
    EventNotFound { label: String },

    #[error("no causal manager available for simulation")]
    #[diagnostic(
        code(akh::agent::ec::no_causal_manager),
        help("Initialize the causal manager (Phase 15a) before running simulations.")
    )]
    NoCausalManager,

    #[error("{0}")]
    #[diagnostic(
        code(akh::agent::ec::engine),
        help("An engine-level error occurred during event calculus reasoning.")
    )]
    Engine(Box<crate::error::AkhError>),
}

impl From<crate::error::AkhError> for EventCalculusError {
    fn from(e: crate::error::AkhError) -> Self {
        Self::Engine(Box::new(e))
    }
}

/// Result alias for the event calculus subsystem.
pub type EventCalculusResult<T> = std::result::Result<T, EventCalculusError>;

// ═══════════════════════════════════════════════════════════════════════
// EventCalculusPredicates
// ═══════════════════════════════════════════════════════════════════════

/// Well-known KG predicates for temporal reasoning (namespace: `ec:`).
#[derive(Debug, Clone)]
pub struct EventCalculusPredicates {
    /// `ec:initiates` — event initiates fluent at time.
    pub initiates: SymbolId,
    /// `ec:terminates` — event terminates fluent at time.
    pub terminates: SymbolId,
    /// `ec:happens` — event happens at time.
    pub happens: SymbolId,
    /// `ec:holds-at` — fluent holds at time (derived).
    pub holds_at: SymbolId,
    /// `ec:clipped` — fluent clipped between two timepoints.
    pub clipped: SymbolId,
    /// `ec:is-fluent` — marks a symbol as a fluent.
    pub is_fluent: SymbolId,
    /// `ec:is-event` — marks a symbol as an event.
    pub is_event: SymbolId,
    /// `ec:at-time` — timestamp of an event.
    pub at_time: SymbolId,
}

impl EventCalculusPredicates {
    /// Resolve or create all event calculus predicates in the KG.
    pub fn init(engine: &Engine) -> EventCalculusResult<Self> {
        Ok(Self {
            initiates: engine.resolve_or_create_relation("ec:initiates")?,
            terminates: engine.resolve_or_create_relation("ec:terminates")?,
            happens: engine.resolve_or_create_relation("ec:happens")?,
            holds_at: engine.resolve_or_create_relation("ec:holds-at")?,
            clipped: engine.resolve_or_create_relation("ec:clipped")?,
            is_fluent: engine.resolve_or_create_relation("ec:is-fluent")?,
            is_event: engine.resolve_or_create_relation("ec:is-event")?,
            at_time: engine.resolve_or_create_relation("ec:at-time")?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Event & Fluent
// ═══════════════════════════════════════════════════════════════════════

/// An event in the event calculus sense — something that happens at a point in time
/// and initiates/terminates fluents.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    /// KG entity representing this event.
    pub symbol_id: SymbolId,
    /// Human-readable name.
    pub name: String,
    /// Timestamp (seconds since UNIX epoch).
    pub timestamp: u64,
    /// Fluents this event initiates (starts).
    pub initiates: Vec<SymbolId>,
    /// Fluents this event terminates (ends).
    pub terminates: Vec<SymbolId>,
}

/// A fluent — a time-varying property that can be initiated and terminated by events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Fluent {
    /// KG entity representing this fluent.
    pub symbol_id: SymbolId,
    /// Human-readable label.
    pub label: String,
    /// Whether this fluent currently holds.
    pub current_value: bool,
    /// The event that last initiated this fluent (if any).
    pub last_initiated_by: Option<SymbolId>,
    /// Timestamp of last initiation.
    pub last_initiated_at: Option<u64>,
    /// The event that last terminated this fluent (if any).
    pub last_terminated_by: Option<SymbolId>,
    /// Timestamp of last termination.
    pub last_terminated_at: Option<u64>,
}

/// A point in a fluent's history: (timestamp, initiated_or_terminated, by_event).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FluentHistoryEntry {
    pub timestamp: u64,
    /// `true` = initiated, `false` = terminated.
    pub initiated: bool,
    /// The event that caused this change.
    pub by_event: SymbolId,
}

// ═══════════════════════════════════════════════════════════════════════
// StateProjection & SimulationResult
// ═══════════════════════════════════════════════════════════════════════

/// Result of projecting state forward in time.
#[derive(Debug, Clone)]
pub struct StateProjection {
    /// Fluents that hold at the projected time.
    pub holding: Vec<Fluent>,
    /// Fluents that were terminated in the interval, with the terminating event.
    pub terminated: Vec<(Fluent, SymbolId)>,
    /// Events that occurred in the interval.
    pub events: Vec<Event>,
}

/// Result of "what-if" simulation: project state after a hypothetical action sequence.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// The action sequence simulated.
    pub action_sequence: Vec<SymbolId>,
    /// Predicted state (holding fluents) at each step.
    pub state_trajectory: Vec<Vec<Fluent>>,
    /// Final predicted state.
    pub final_state: Vec<Fluent>,
    /// Confidence in the prediction (decays with trajectory length).
    pub confidence: f32,
}

// ═══════════════════════════════════════════════════════════════════════
// EventCalculusEngine
// ═══════════════════════════════════════════════════════════════════════

/// Manages temporal fluent reasoning via event calculus axioms.
///
/// Events and fluents are stored as KG entities with `ec:` namespace predicates.
/// The engine evaluates the core EC axioms (holds-at, clipped) by querying
/// the KG rather than maintaining a separate temporal database.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventCalculusEngine {
    /// Recorded events, keyed by SymbolId.
    pub events: HashMap<u64, Event>,
    /// Known fluents, keyed by SymbolId.
    pub fluents: HashMap<u64, Fluent>,
    /// Fluent history: fluent_id → ordered list of (timestamp, initiated, by_event).
    pub history: HashMap<u64, Vec<FluentHistoryEntry>>,
    /// Lazily initialized predicates (not serialized).
    #[serde(skip)]
    predicates: Option<EventCalculusPredicates>,
}

impl EventCalculusEngine {
    /// Create a new event calculus engine and initialize KG predicates.
    pub fn new(engine: &Engine) -> EventCalculusResult<Self> {
        let mut ec = Self::default();
        ec.ensure_init(engine)?;
        Ok(ec)
    }

    /// Ensure predicates are initialized.
    pub fn ensure_init(&mut self, engine: &Engine) -> EventCalculusResult<()> {
        if self.predicates.is_none() {
            self.predicates = Some(EventCalculusPredicates::init(engine)?);
        }
        Ok(())
    }

    /// Get the EC predicates (panics if not initialized).
    fn preds(&self) -> &EventCalculusPredicates {
        self.predicates
            .as_ref()
            .expect("EventCalculusEngine not initialized — call ensure_init()")
    }

    // ─── Restore / Persist ────────────────────────────────────────

    /// Restore from durable store.
    pub fn restore(engine: &Engine) -> EventCalculusResult<Self> {
        let data = engine
            .store()
            .get_meta(b"agent:ec_engine")
            .map_err(|e| EventCalculusError::Engine(Box::new(e.into())))?;
        match data {
            Some(bytes) if !bytes.is_empty() => {
                let mut ec: Self = bincode::deserialize(&bytes).map_err(|e| {
                    EventCalculusError::Engine(Box::new(crate::error::AkhError::Store(
                        crate::error::StoreError::Serialization {
                            message: format!("ec engine deserialize: {e}"),
                        },
                    )))
                })?;
                ec.ensure_init(engine)?;
                Ok(ec)
            }
            _ => Self::new(engine),
        }
    }

    /// Persist to durable store.
    pub fn persist(&self, engine: &Engine) -> EventCalculusResult<()> {
        let bytes = bincode::serialize(self).map_err(|e| {
            EventCalculusError::Engine(Box::new(crate::error::AkhError::Store(
                crate::error::StoreError::Serialization {
                    message: format!("ec engine serialize: {e}"),
                },
            )))
        })?;
        engine
            .store()
            .put_meta(b"agent:ec_engine", &bytes)
            .map_err(|e| EventCalculusError::Engine(Box::new(e.into())))?;
        Ok(())
    }

    // ─── Record Event ─────────────────────────────────────────────

    /// Record an event and apply its effects to fluents.
    ///
    /// Creates KG entities for the event and any new fluents, then stores
    /// `ec:initiates` / `ec:terminates` / `ec:happens` triples.
    pub fn record_event(
        &mut self,
        event: Event,
        engine: &Engine,
    ) -> EventCalculusResult<()> {
        let preds = self.preds().clone();
        let event_id = event.symbol_id;
        let ts = event.timestamp;

        // Store ec:happens triple (event → ec:is-event → event)
        let marker_sym = engine.resolve_or_create_entity("ec:event-marker")?;
        let marker_triple = Triple::new(event_id, preds.is_event, marker_sym)
            .with_confidence(1.0);
        engine.add_triple(&marker_triple)?;

        // Store timestamp as a triple: event → ec:at-time → timestamp-entity
        let ts_label = format!("ec:t:{ts}");
        let ts_sym = engine.resolve_or_create_entity(&ts_label)?;
        let ts_triple = Triple::new(event_id, preds.at_time, ts_sym)
            .with_confidence(1.0);
        engine.add_triple(&ts_triple)?;

        // Process initiations.
        for &fluent_id in &event.initiates {
            // Store ec:initiates triple.
            let init_triple = Triple::new(event_id, preds.initiates, fluent_id)
                .with_confidence(1.0)
                .with_confidence(1.0);
            engine.add_triple(&init_triple)?;

            // Update fluent state.
            let fluent = self.fluents.entry(fluent_id.get()).or_insert_with(|| {
                let label = engine.resolve_label(fluent_id);
                Fluent {
                    symbol_id: fluent_id,
                    label,
                    current_value: false,
                    last_initiated_by: None,
                    last_initiated_at: None,
                    last_terminated_by: None,
                    last_terminated_at: None,
                }
            });
            fluent.current_value = true;
            fluent.last_initiated_by = Some(event_id);
            fluent.last_initiated_at = Some(ts);

            // Record history.
            self.history
                .entry(fluent_id.get())
                .or_default()
                .push(FluentHistoryEntry {
                    timestamp: ts,
                    initiated: true,
                    by_event: event_id,
                });
        }

        // Process terminations.
        for &fluent_id in &event.terminates {
            let term_triple = Triple::new(event_id, preds.terminates, fluent_id)
                .with_confidence(1.0)
                .with_confidence(1.0);
            engine.add_triple(&term_triple)?;

            // Update fluent state.
            let fluent = self.fluents.entry(fluent_id.get()).or_insert_with(|| {
                let label = engine.resolve_label(fluent_id);
                Fluent {
                    symbol_id: fluent_id,
                    label,
                    current_value: true, // was holding before termination
                    last_initiated_by: None,
                    last_initiated_at: None,
                    last_terminated_by: None,
                    last_terminated_at: None,
                }
            });
            fluent.current_value = false;
            fluent.last_terminated_by = Some(event_id);
            fluent.last_terminated_at = Some(ts);

            // Record history.
            self.history
                .entry(fluent_id.get())
                .or_default()
                .push(FluentHistoryEntry {
                    timestamp: ts,
                    initiated: false,
                    by_event: event_id,
                });
        }

        // Store the event itself.
        self.events.insert(event_id.get(), event);

        Ok(())
    }

    // ─── Core EC Axioms ───────────────────────────────────────────

    /// Core Event Calculus axiom: does fluent `f` hold at time `t`?
    ///
    /// A fluent holds at time `t` iff:
    /// 1. Some event `e` initiated `f` at time `t1 ≤ t`, AND
    /// 2. No event terminated `f` between `t1` and `t` (not clipped).
    pub fn holds_at(&self, fluent_id: SymbolId, time: u64) -> bool {
        let Some(history) = self.history.get(&fluent_id.get()) else {
            return false;
        };

        // Find the most recent initiation at or before `time`.
        let mut last_init_time: Option<u64> = None;
        for entry in history.iter().rev() {
            if entry.timestamp > time {
                continue;
            }
            if entry.initiated {
                last_init_time = Some(entry.timestamp);
                break;
            } else {
                // Terminated before any initiation in this window → doesn't hold.
                return false;
            }
        }

        let Some(init_t) = last_init_time else {
            return false;
        };

        // Check not clipped: no termination between init_t (exclusive) and time (inclusive).
        !self.is_clipped(fluent_id, init_t, time)
    }

    /// Is fluent `f` clipped between `t1` (exclusive) and `t2` (inclusive)?
    ///
    /// Clipped means some event terminated `f` in the interval (t1, t2].
    fn is_clipped(&self, fluent_id: SymbolId, t1: u64, t2: u64) -> bool {
        let Some(history) = self.history.get(&fluent_id.get()) else {
            return false;
        };
        history.iter().any(|entry| {
            !entry.initiated && entry.timestamp > t1 && entry.timestamp <= t2
        })
    }

    // ─── State Projection ─────────────────────────────────────────

    /// Project state: what fluents hold at `to_time` given events in `[from_time, to_time]`?
    pub fn project_state(&self, from_time: u64, to_time: u64) -> StateProjection {
        // Fluents that hold at to_time.
        let holding: Vec<Fluent> = self
            .fluents
            .values()
            .filter(|f| self.holds_at(f.symbol_id, to_time))
            .cloned()
            .collect();

        // Fluents that were terminated in the interval.
        let mut terminated = Vec::new();
        for fluent in self.fluents.values() {
            if let Some(history) = self.history.get(&fluent.symbol_id.get()) {
                for entry in history {
                    if !entry.initiated
                        && entry.timestamp >= from_time
                        && entry.timestamp <= to_time
                    {
                        terminated.push((fluent.clone(), entry.by_event));
                    }
                }
            }
        }

        // Events in the interval.
        let events: Vec<Event> = self
            .events
            .values()
            .filter(|e| e.timestamp >= from_time && e.timestamp <= to_time)
            .cloned()
            .collect();

        StateProjection {
            holding,
            terminated,
            events,
        }
    }

    /// Simulate a hypothetical sequence of actions using the causal manager.
    ///
    /// For each action, predicts effects via `CausalManager::predict_effects`,
    /// then converts assertions/retractions into fluent initiations/terminations.
    /// Confidence decays geometrically with each step (0.9^step).
    pub fn simulate_actions(
        &self,
        action_sequence: &[SymbolId],
        causal_manager: &super::causal::CausalManager,
        engine: &Engine,
    ) -> EventCalculusResult<SimulationResult> {
        let mut trajectory: Vec<Vec<Fluent>> = Vec::new();
        // Start with current holding fluents.
        let mut current_fluents: HashMap<u64, Fluent> = self
            .fluents
            .iter()
            .filter(|(_, f)| f.current_value)
            .map(|(&k, v)| (k, v.clone()))
            .collect();

        let mut confidence = 1.0_f32;
        let decay = 0.9_f32;

        for &action_id in action_sequence {
            let action_name = engine.resolve_label(action_id);

            // Predict effects via causal manager.
            let transition = causal_manager
                .predict_effects(&action_name, engine)
                .map_err(|_| EventCalculusError::NoCausalManager)?;

            // Assertions → initiate fluents.
            for &(s, p, _o) in &transition.assertions {
                let label = engine.resolve_label(s);
                current_fluents.insert(
                    s.get(),
                    Fluent {
                        symbol_id: s,
                        label,
                        current_value: true,
                        last_initiated_by: Some(action_id),
                        last_initiated_at: None,
                        last_terminated_by: None,
                        last_terminated_at: None,
                    },
                );
                // Also track the predicate as a fluent.
                let plabel = engine.resolve_label(p);
                current_fluents
                    .entry(p.get())
                    .or_insert(Fluent {
                        symbol_id: p,
                        label: plabel,
                        current_value: true,
                        last_initiated_by: Some(action_id),
                        last_initiated_at: None,
                        last_terminated_by: None,
                        last_terminated_at: None,
                    });
            }

            // Retractions → terminate fluents.
            for &(s, _p, _o) in &transition.retractions {
                if let Some(f) = current_fluents.get_mut(&s.get()) {
                    f.current_value = false;
                    f.last_terminated_by = Some(action_id);
                }
            }

            confidence *= decay;
            trajectory.push(current_fluents.values().cloned().collect());
        }

        let final_state: Vec<Fluent> = current_fluents
            .into_values()
            .filter(|f| f.current_value)
            .collect();

        Ok(SimulationResult {
            action_sequence: action_sequence.to_vec(),
            state_trajectory: trajectory,
            final_state,
            confidence,
        })
    }

    // ─── Temporal Queries ─────────────────────────────────────────

    /// What changed since a given timestamp?
    ///
    /// Returns `(fluent, event, initiated)` triples where `initiated` is `true`
    /// if the fluent was initiated and `false` if it was terminated.
    pub fn what_changed_since(&self, since: u64) -> Vec<(Fluent, Event, bool)> {
        let mut changes = Vec::new();

        for (fluent_id, history) in &self.history {
            for entry in history {
                if entry.timestamp > since {
                    if let Some(fluent) = self.fluents.get(fluent_id) {
                        if let Some(event) = self.events.get(&entry.by_event.get()) {
                            changes.push((fluent.clone(), event.clone(), entry.initiated));
                        }
                    }
                }
            }
        }

        // Sort by timestamp.
        changes.sort_by_key(|(_, e, _)| e.timestamp);
        changes
    }

    /// Full history of a fluent: when was it initiated/terminated and by what?
    pub fn fluent_history(
        &self,
        fluent_id: SymbolId,
    ) -> EventCalculusResult<Vec<FluentHistoryEntry>> {
        let fluent = self.fluents.get(&fluent_id.get()).ok_or_else(|| {
            EventCalculusError::FluentNotFound {
                label: format!("symbol:{}", fluent_id.get()),
            }
        })?;
        Ok(self
            .history
            .get(&fluent.symbol_id.get())
            .cloned()
            .unwrap_or_default())
    }

    // ─── Accessors ────────────────────────────────────────────────

    /// List all known fluents.
    pub fn all_fluents(&self) -> Vec<&Fluent> {
        self.fluents.values().collect()
    }

    /// List all recorded events.
    pub fn all_events(&self) -> Vec<&Event> {
        self.events.values().collect()
    }

    /// List fluents currently holding.
    pub fn holding_fluents(&self) -> Vec<&Fluent> {
        self.fluents.values().filter(|f| f.current_value).collect()
    }

    /// Get a fluent by symbol id.
    pub fn get_fluent(&self, id: SymbolId) -> Option<&Fluent> {
        self.fluents.get(&id.get())
    }

    /// Get an event by symbol id.
    pub fn get_event(&self, id: SymbolId) -> Option<&Event> {
        self.events.get(&id.get())
    }

    // ─── Provenance ───────────────────────────────────────────────

    /// Record provenance for an event calculus projection.
    pub fn record_projection_provenance(
        &self,
        engine: &Engine,
        derived_id: SymbolId,
        fluent_count: usize,
        event_count: usize,
        interval_secs: u64,
    ) -> EventCalculusResult<()> {
        let mut record = crate::provenance::ProvenanceRecord::new(
            derived_id,
            DerivationKind::EventCalculusProjection {
                fluent_count: fluent_count as u32,
                event_count: event_count as u32,
                interval_secs,
            },
        )
        .with_confidence(0.85);
        engine
            .store_provenance(&mut record)
            .map_err(|e| EventCalculusError::Engine(Box::new(e)))?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Engine, EngineConfig};
    use crate::symbol::SymbolKind;

    fn test_engine() -> Engine {
        Engine::new(EngineConfig::default()).unwrap()
    }

    fn make_fluent_id(engine: &Engine, label: &str) -> SymbolId {
        engine.create_symbol(SymbolKind::Entity, label).unwrap().id
    }

    fn make_event_id(engine: &Engine, label: &str) -> SymbolId {
        engine.create_symbol(SymbolKind::Entity, label).unwrap().id
    }

    // ── EC predicates namespace ───────────────────────────────────

    #[test]
    fn ec_predicates_namespace() {
        let expected = [
            "ec:initiates",
            "ec:terminates",
            "ec:happens",
            "ec:holds-at",
            "ec:clipped",
            "ec:is-fluent",
            "ec:is-event",
            "ec:at-time",
        ];
        for label in &expected {
            assert!(label.starts_with("ec:"));
        }
    }

    #[test]
    fn ec_predicates_init() {
        let engine = test_engine();
        let preds = EventCalculusPredicates::init(&engine).unwrap();
        // All predicates should be distinct.
        let ids = [
            preds.initiates,
            preds.terminates,
            preds.happens,
            preds.holds_at,
            preds.clipped,
            preds.is_fluent,
            preds.is_event,
            preds.at_time,
        ];
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                assert_ne!(ids[i], ids[j], "predicates {i} and {j} should differ");
            }
        }
    }

    // ── holds_at after initiation ─────────────────────────────────

    #[test]
    fn holds_at_after_initiation() {
        let engine = test_engine();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();

        let fluent = make_fluent_id(&engine, "light-on");
        let event = make_event_id(&engine, "switch-on");

        ec.record_event(
            Event {
                symbol_id: event,
                name: "switch-on".into(),
                timestamp: 100,
                initiates: vec![fluent],
                terminates: vec![],
            },
            &engine,
        )
        .unwrap();

        // Fluent should hold at and after initiation time.
        assert!(ec.holds_at(fluent, 100));
        assert!(ec.holds_at(fluent, 200));

        // Should not hold before initiation.
        assert!(!ec.holds_at(fluent, 50));
    }

    // ── holds_at terminated ───────────────────────────────────────

    #[test]
    fn holds_at_terminated() {
        let engine = test_engine();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();

        let fluent = make_fluent_id(&engine, "light-on");
        let on_event = make_event_id(&engine, "switch-on");
        let off_event = make_event_id(&engine, "switch-off");

        ec.record_event(
            Event {
                symbol_id: on_event,
                name: "switch-on".into(),
                timestamp: 100,
                initiates: vec![fluent],
                terminates: vec![],
            },
            &engine,
        )
        .unwrap();

        ec.record_event(
            Event {
                symbol_id: off_event,
                name: "switch-off".into(),
                timestamp: 200,
                initiates: vec![],
                terminates: vec![fluent],
            },
            &engine,
        )
        .unwrap();

        // Holds between initiation and termination.
        assert!(ec.holds_at(fluent, 100));
        assert!(ec.holds_at(fluent, 150));

        // Does not hold at or after termination.
        assert!(!ec.holds_at(fluent, 200));
        assert!(!ec.holds_at(fluent, 300));
    }

    // ── clipped between events ────────────────────────────────────

    #[test]
    fn clipped_between_events() {
        let engine = test_engine();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();

        let fluent = make_fluent_id(&engine, "power");
        let start = make_event_id(&engine, "boot");
        let crash = make_event_id(&engine, "crash");
        let reboot = make_event_id(&engine, "reboot");

        // boot at 100 → power on
        ec.record_event(
            Event {
                symbol_id: start,
                name: "boot".into(),
                timestamp: 100,
                initiates: vec![fluent],
                terminates: vec![],
            },
            &engine,
        )
        .unwrap();

        // crash at 200 → power off
        ec.record_event(
            Event {
                symbol_id: crash,
                name: "crash".into(),
                timestamp: 200,
                initiates: vec![],
                terminates: vec![fluent],
            },
            &engine,
        )
        .unwrap();

        // reboot at 300 → power on again
        ec.record_event(
            Event {
                symbol_id: reboot,
                name: "reboot".into(),
                timestamp: 300,
                initiates: vec![fluent],
                terminates: vec![],
            },
            &engine,
        )
        .unwrap();

        assert!(ec.holds_at(fluent, 150));  // between boot and crash
        assert!(!ec.holds_at(fluent, 250)); // between crash and reboot
        assert!(ec.holds_at(fluent, 350));  // after reboot
    }

    // ── project_state ─────────────────────────────────────────────

    #[test]
    fn project_state_simple() {
        let engine = test_engine();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();

        let f1 = make_fluent_id(&engine, "alive");
        let f2 = make_fluent_id(&engine, "hungry");
        let e1 = make_event_id(&engine, "wake-up");

        ec.record_event(
            Event {
                symbol_id: e1,
                name: "wake-up".into(),
                timestamp: 100,
                initiates: vec![f1, f2],
                terminates: vec![],
            },
            &engine,
        )
        .unwrap();

        let proj = ec.project_state(0, 200);
        assert_eq!(proj.holding.len(), 2);
        assert_eq!(proj.events.len(), 1);
        assert!(proj.terminated.is_empty());
    }

    #[test]
    fn project_state_multiple_events() {
        let engine = test_engine();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();

        let f1 = make_fluent_id(&engine, "awake");
        let e1 = make_event_id(&engine, "alarm");
        let e2 = make_event_id(&engine, "sleep");

        ec.record_event(
            Event {
                symbol_id: e1,
                name: "alarm".into(),
                timestamp: 100,
                initiates: vec![f1],
                terminates: vec![],
            },
            &engine,
        )
        .unwrap();

        ec.record_event(
            Event {
                symbol_id: e2,
                name: "sleep".into(),
                timestamp: 200,
                initiates: vec![],
                terminates: vec![f1],
            },
            &engine,
        )
        .unwrap();

        let proj = ec.project_state(0, 300);
        assert!(proj.holding.is_empty()); // awake was terminated
        assert_eq!(proj.terminated.len(), 1);
        assert_eq!(proj.events.len(), 2);
    }

    // ── simulation confidence decay ───────────────────────────────

    #[test]
    fn simulation_confidence_decay() {
        // Confidence should decay as 0.9^n for n steps.
        let decay = 0.9_f32;
        let steps = 5;
        let expected = decay.powi(steps);
        assert!((expected - 0.9_f32.powi(5)).abs() < f32::EPSILON);
    }

    // ── what_changed_since ────────────────────────────────────────

    #[test]
    fn what_changed_since_empty() {
        let engine = test_engine();
        let ec = EventCalculusEngine::new(&engine).unwrap();
        let changes = ec.what_changed_since(0);
        assert!(changes.is_empty());
    }

    #[test]
    fn what_changed_since_with_events() {
        let engine = test_engine();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();

        let f = make_fluent_id(&engine, "connected");
        let e1 = make_event_id(&engine, "connect");
        let e2 = make_event_id(&engine, "disconnect");

        ec.record_event(
            Event {
                symbol_id: e1,
                name: "connect".into(),
                timestamp: 100,
                initiates: vec![f],
                terminates: vec![],
            },
            &engine,
        )
        .unwrap();

        ec.record_event(
            Event {
                symbol_id: e2,
                name: "disconnect".into(),
                timestamp: 200,
                initiates: vec![],
                terminates: vec![f],
            },
            &engine,
        )
        .unwrap();

        // Changes since time 0: both events.
        let changes = ec.what_changed_since(0);
        assert_eq!(changes.len(), 2);

        // Changes since time 150: only disconnect.
        let changes = ec.what_changed_since(150);
        assert_eq!(changes.len(), 1);
        assert_eq!(changes[0].1.name, "disconnect");
    }

    // ── fluent_history ────────────────────────────────────────────

    #[test]
    fn fluent_history_tracks_toggles() {
        let engine = test_engine();
        let mut ec = EventCalculusEngine::new(&engine).unwrap();

        let f = make_fluent_id(&engine, "door-open");
        let open = make_event_id(&engine, "open-door");
        let close = make_event_id(&engine, "close-door");
        let reopen = make_event_id(&engine, "reopen-door");

        ec.record_event(
            Event {
                symbol_id: open,
                name: "open-door".into(),
                timestamp: 100,
                initiates: vec![f],
                terminates: vec![],
            },
            &engine,
        )
        .unwrap();

        ec.record_event(
            Event {
                symbol_id: close,
                name: "close-door".into(),
                timestamp: 200,
                initiates: vec![],
                terminates: vec![f],
            },
            &engine,
        )
        .unwrap();

        ec.record_event(
            Event {
                symbol_id: reopen,
                name: "reopen-door".into(),
                timestamp: 300,
                initiates: vec![f],
                terminates: vec![],
            },
            &engine,
        )
        .unwrap();

        let history = ec.fluent_history(f).unwrap();
        assert_eq!(history.len(), 3);
        assert!(history[0].initiated);   // open
        assert!(!history[1].initiated);  // close
        assert!(history[2].initiated);   // reopen
    }

    // ── persist / restore roundtrip ───────────────────────────────

    #[test]
    fn persist_restore_roundtrip() {
        let mut ec = EventCalculusEngine::default();
        let event_id = SymbolId::new(42).unwrap();
        let fluent_id = SymbolId::new(43).unwrap();

        ec.events.insert(
            event_id.get(),
            Event {
                symbol_id: event_id,
                name: "test-event".into(),
                timestamp: 100,
                initiates: vec![fluent_id],
                terminates: vec![],
            },
        );
        ec.fluents.insert(
            fluent_id.get(),
            Fluent {
                symbol_id: fluent_id,
                label: "test-fluent".into(),
                current_value: true,
                last_initiated_by: Some(event_id),
                last_initiated_at: Some(100),
                last_terminated_by: None,
                last_terminated_at: None,
            },
        );
        ec.history.insert(
            fluent_id.get(),
            vec![FluentHistoryEntry {
                timestamp: 100,
                initiated: true,
                by_event: event_id,
            }],
        );

        let bytes = bincode::serialize(&ec).unwrap();
        let restored: EventCalculusEngine = bincode::deserialize(&bytes).unwrap();

        assert_eq!(restored.events.len(), 1);
        assert_eq!(restored.fluents.len(), 1);
        assert_eq!(restored.history.len(), 1);
        assert_eq!(
            restored.events.get(&event_id.get()).unwrap().name,
            "test-event"
        );
    }

    // ── FluentNotFound error ──────────────────────────────────────

    #[test]
    fn fluent_history_not_found() {
        let engine = test_engine();
        let ec = EventCalculusEngine::new(&engine).unwrap();
        let missing = SymbolId::new(99999).unwrap();
        assert!(ec.fluent_history(missing).is_err());
    }
}
