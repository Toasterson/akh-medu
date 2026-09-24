//! Mutation hooks for the write-ahead log (Phase 38b).
//!
//! The [`Engine`](crate::engine::Engine) is synchronous and the OODA loop must
//! never block on durability. So the engine only knows about a tiny
//! [`MutationSink`] trait: after a successful assert/retract it hands the sink a
//! [`MutationRecord`] and moves on. The actual WAL — buffering, batching, and
//! the async flush to Postgres — lives behind that trait in
//! [`crate::wal`](crate::wal) (compiled only under the `postgres` feature).
//!
//! Keeping the trait here, in the core, means the engine carries **no**
//! dependency on SeaORM/sqlx: the lean `akh` client builds without any of it.

/// Version of the replay-complete assertion payload written to the mutation
/// WAL. Version 1 was the original unmarked `{subject,predicate,object,...}`
/// shape and remains readable when all referenced symbols predate a snapshot.
pub const ASSERT_MUTATION_VERSION: u64 = 2;

/// Version of the compound contradiction-replacement payload. A replacement
/// is one logical transition: every displaced edge and the replay-complete new
/// assertion must survive (or be dropped) together in the bounded WAL.
pub const REPLACE_MUTATION_VERSION: u64 = 1;

/// Version of the replay-complete, overflow-protected TMS cascade retraction.
/// V2 closes every referenced provenance record so the batch can survive even
/// when its older assertion is dropped by bounded-WAL overflow.
pub const RETRACT_BATCH_MUTATION_VERSION: u64 = 2;

/// A single knowledge mutation captured at the engine boundary.
#[derive(Debug, Clone)]
pub struct MutationRecord {
    /// What kind of mutation this is.
    pub kind: MutationKind,
    /// Kind-specific payload, serialised to the `mutation_log.payload` JSONB
    /// column verbatim. For asserts/retracts this carries the symbol ids.
    pub payload: serde_json::Value,
}

impl MutationRecord {
    /// Construct a record from a kind and payload.
    pub fn new(kind: MutationKind, payload: serde_json::Value) -> Self {
        Self { kind, payload }
    }
}

/// Taxonomy of mutations recorded in `akhomed.mutation_log`.
///
/// `as_str` is the exact string stored in the `kind` column; replay (38c)
/// dispatches on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationKind {
    /// A triple was asserted.
    Assert,
    /// A triple was retracted.
    Retract,
    /// One primary retraction and every concrete edge removed by its TMS
    /// cascade. Kept as one protected record so fresh-host replay does not
    /// depend on an in-memory TMS that is absent from the relational snapshot.
    RetractBatch,
    /// One or more conflicting triples were retracted and a replacement was
    /// asserted as one indivisible replay/overflow unit.
    Replace,
    /// A microtheory was activated in the workspace.
    MicrotheoryLoad,
    /// A microtheory was deactivated.
    MicrotheoryUnload,
    /// A compartment was created.
    CompartmentCreate,
    /// A compartment was destroyed.
    CompartmentDestroy,
    /// A completed agentic conversation turn (ADR 091: transcript→episodic
    /// capture). Raw staging, not a KG mutation — never replayed through
    /// `Engine::add_triple` on wake (see the `"turn"` skip arm in the main
    /// crate's `snapshot.rs::apply_mutation`); the KG facts derived from a
    /// turn (if any) get their own `assert` rows. A dropped turn is silent
    /// memory loss, so it is protected on overflow just like `Assert`.
    Turn,
    /// Auditable, non-replayable merge of one legacy duplicate symbol into
    /// its canonical ID (WI-338). The payload carries before/after
    /// TripleIdentity values and affected provenance IDs; the repaired
    /// relational snapshot is authoritative.
    SymbolMerge,
}

impl MutationKind {
    /// The canonical `mutation_log.kind` string for this variant.
    pub fn as_str(self) -> &'static str {
        match self {
            MutationKind::Assert => "assert",
            MutationKind::Retract => "retract",
            MutationKind::RetractBatch => "retract_batch",
            MutationKind::Replace => "replace",
            MutationKind::MicrotheoryLoad => "microtheory_load",
            MutationKind::MicrotheoryUnload => "microtheory_unload",
            MutationKind::CompartmentCreate => "compartment_create",
            MutationKind::CompartmentDestroy => "compartment_destroy",
            MutationKind::Turn => "turn",
            MutationKind::SymbolMerge => "symbol_merge",
        }
    }

    /// Whether this kind carries knowledge that should be preserved over the
    /// buffer-overflow drop policy. Asserts, compound replacements, and turns are protected;
    /// everything else is reconstructable or less critical and is dropped
    /// first under pressure. A dropped turn is silent memory loss (ADR 091),
    /// same severity class as a dropped assert.
    pub fn is_protected_on_overflow(self) -> bool {
        matches!(
            self,
            MutationKind::Assert
                | MutationKind::RetractBatch
                | MutationKind::Replace
                | MutationKind::Turn
                | MutationKind::SymbolMerge
        )
    }
}

/// Monotonic counts of fact changes seen at the engine's mutation boundary
/// (Phase 53b). They are bumped for every recorded mutation, whether or not a
/// WAL sink is installed, so local-only engines count too.
///
/// The daemon snapshots these before and after each background task to
/// report a per-task [`FactTotals`] delta. The counters are engine-wide, so a
/// live turn that mutates the KG while a task runs is attributed to that
/// task as well.
#[derive(Debug, Default)]
pub struct FactCounters {
    asserted: std::sync::atomic::AtomicU64,
    retracted: std::sync::atomic::AtomicU64,
}

/// A point-in-time reading of [`FactCounters`], or the difference between
/// two readings.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FactTotals {
    /// Triples asserted, including confidence updates and the new edge of a
    /// contradiction replacement.
    pub asserted: u64,
    /// Triples retracted, including edges displaced by a replacement and
    /// every edge of a TMS cascade.
    pub retracted: u64,
}

impl FactTotals {
    /// `self - earlier`, saturating at zero.
    pub fn since(self, earlier: FactTotals) -> FactTotals {
        FactTotals {
            asserted: self.asserted.saturating_sub(earlier.asserted),
            retracted: self.retracted.saturating_sub(earlier.retracted),
        }
    }

    /// Whether nothing changed.
    pub fn is_empty(&self) -> bool {
        self.asserted == 0 && self.retracted == 0
    }
}

impl FactCounters {
    /// Count one mutation. The payload is only read for compound kinds, to
    /// learn how many edges they retract.
    pub fn observe(&self, kind: MutationKind, payload: &serde_json::Value) {
        use std::sync::atomic::Ordering::Relaxed;
        let retracts_in = |payload: &serde_json::Value| {
            payload
                .get("retracts")
                .and_then(serde_json::Value::as_array)
                .map_or(1, |edges| edges.len() as u64)
        };
        match kind {
            MutationKind::Assert => {
                self.asserted.fetch_add(1, Relaxed);
            }
            MutationKind::Replace => {
                self.asserted.fetch_add(1, Relaxed);
                self.retracted.fetch_add(retracts_in(payload), Relaxed);
            }
            MutationKind::Retract => {
                self.retracted.fetch_add(1, Relaxed);
            }
            MutationKind::RetractBatch => {
                self.retracted.fetch_add(retracts_in(payload), Relaxed);
            }
            MutationKind::MicrotheoryLoad
            | MutationKind::MicrotheoryUnload
            | MutationKind::CompartmentCreate
            | MutationKind::CompartmentDestroy
            | MutationKind::Turn
            | MutationKind::SymbolMerge => {}
        }
    }

    /// The current totals.
    pub fn totals(&self) -> FactTotals {
        use std::sync::atomic::Ordering::Relaxed;
        FactTotals {
            asserted: self.asserted.load(Relaxed),
            retracted: self.retracted.load(Relaxed),
        }
    }
}

/// Receives mutations as they happen at the engine boundary.
///
/// Implementations **must not block** and must not surface errors to the
/// caller: durability is best-effort from the engine's point of view. A WAL
/// that cannot reach Postgres buffers in memory and retries; it never stalls
/// the OODA loop.
pub trait MutationSink: Send + Sync {
    /// Record a mutation. Returns immediately.
    fn record(&self, mutation: MutationRecord);
}

/// Convenience: a sink that counts records, for tests and as a no-Postgres stub.
#[cfg(test)]
#[derive(Default)]
pub struct CountingSink {
    pub count: std::sync::atomic::AtomicUsize,
}

#[cfg(test)]
impl MutationSink for CountingSink {
    fn record(&self, _mutation: MutationRecord) {
        self.count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Pages a captured turn's body back out of the WAL at recall time (ADR 091
/// decision 5: "a minted episode is a lightweight index; the body stays in
/// the WAL"). Mirrors [`MutationSink`]'s shape — a tiny synchronous seam the
/// engine can call from the recall path without knowing about SeaORM or
/// tokio. The daemon binary installs the real, Postgres-backed implementation
/// (bridging its async DB hop with `block_on`/`block_in_place`); in-memory or
/// local-only engines simply never install one, and [`crate::engine::Engine::turn_body`]
/// returns `None`.
pub trait TurnBodyReader: Send + Sync {
    /// Look up the raw turn body for a given WAL `seq`. Returns `None` if the
    /// row does not exist, is not a `kind="turn"` row, or the lookup failed
    /// (best-effort: recall must fall back to the existing summary/label path,
    /// never error, on a missing body).
    fn turn_body(&self, seq: i64) -> Option<String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn kind_strings_are_stable() {
        assert_eq!(MutationKind::Assert.as_str(), "assert");
        assert_eq!(MutationKind::Retract.as_str(), "retract");
        assert_eq!(MutationKind::RetractBatch.as_str(), "retract_batch");
        assert_eq!(MutationKind::Replace.as_str(), "replace");
        assert_eq!(MutationKind::MicrotheoryLoad.as_str(), "microtheory_load");
        assert_eq!(
            MutationKind::CompartmentDestroy.as_str(),
            "compartment_destroy"
        );
        assert_eq!(MutationKind::Turn.as_str(), "turn");
        assert_eq!(MutationKind::SymbolMerge.as_str(), "symbol_merge");
    }

    #[test]
    fn assert_and_turn_are_protected() {
        assert!(MutationKind::Assert.is_protected_on_overflow());
        assert!(MutationKind::RetractBatch.is_protected_on_overflow());
        assert!(MutationKind::Replace.is_protected_on_overflow());
        assert!(MutationKind::Turn.is_protected_on_overflow());
        assert!(!MutationKind::Retract.is_protected_on_overflow());
        assert!(!MutationKind::MicrotheoryLoad.is_protected_on_overflow());
    }

    #[test]
    fn turn_kind_round_trips_through_as_str() {
        // `as_str` is the exact string stored in `mutation_log.kind`; the
        // sleep-cycle harvest phase and wake-replay skip both match on it
        // literally ("turn"), so a rename here is a breaking schema change.
        assert_eq!(MutationKind::Turn.as_str(), "turn");
    }

    #[test]
    fn sink_receives_records() {
        let sink = CountingSink::default();
        sink.record(MutationRecord::new(
            MutationKind::Assert,
            serde_json::json!({}),
        ));
        sink.record(MutationRecord::new(
            MutationKind::Retract,
            serde_json::json!({}),
        ));
        assert_eq!(sink.count.load(Ordering::Relaxed), 2);
    }
}
