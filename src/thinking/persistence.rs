//! Persistence seam for thinking traces.
//!
//! Phase 41d (Postgres `akhomed.thinking_traces` via the Phase 38 WAL) is
//! **deferred**; this module defines the seam so the loop already flushes
//! every step through it, plus an in-memory implementation used today.
//!
//! // Phase 41d: implement `ThoughtPersistence` over the Phase 38 WAL
//! // (`akhomed.thinking_traces` + `akhomed.thinking_steps`, schema in
//! // docs/ai/plans/2026-04-27-phase41-thinking-engine.md) and add
//! // resume-from-snapshot on Akh wake. No Postgres code belongs here yet.

use std::collections::HashMap;
use std::sync::Mutex;

use super::error::{ThinkingError, ThinkingResult};
use super::state::{Conclusion, HaltReason, ThoughtId, ThoughtOrigin, ThoughtStep};

/// Where thinking traces go, one step at a time.
///
/// Implementations must be cheap per call: the loop flushes after **every**
/// primitive step.
pub trait ThoughtPersistence: Send + Sync {
    /// A deliberation opened.
    fn begin(&self, id: ThoughtId, origin: &ThoughtOrigin) -> ThinkingResult<()>;

    /// One step was emitted (0-based `step_idx` within the thought).
    fn flush_step(&self, id: ThoughtId, step_idx: u32, step: &ThoughtStep) -> ThinkingResult<()>;

    /// The deliberation concluded.
    fn conclude(&self, id: ThoughtId, conclusion: &Conclusion) -> ThinkingResult<()>;
}

/// A persisted trace (in-memory backend).
#[derive(Debug, Clone)]
pub struct PersistedTrace {
    pub origin: ThoughtOrigin,
    pub steps: Vec<ThoughtStep>,
    pub halt_reason: Option<HaltReason>,
}

/// In-memory trace store — the Phase 41a..41c default.
#[derive(Debug, Default)]
pub struct InMemoryThoughtPersistence {
    traces: Mutex<HashMap<u64, PersistedTrace>>,
}

impl InMemoryThoughtPersistence {
    pub fn new() -> Self {
        Self::default()
    }

    /// Read back a persisted trace (tests, anytime introspection).
    pub fn trace(&self, id: ThoughtId) -> Option<PersistedTrace> {
        self.traces.lock().ok()?.get(&id.0).cloned()
    }

    /// Number of traces held.
    pub fn len(&self) -> usize {
        self.traces.lock().map(|t| t.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ThoughtPersistence for InMemoryThoughtPersistence {
    fn begin(&self, id: ThoughtId, origin: &ThoughtOrigin) -> ThinkingResult<()> {
        let mut traces = self.traces.lock().map_err(|e| ThinkingError::Persistence {
            message: format!("trace store poisoned: {e}"),
        })?;
        traces.insert(
            id.0,
            PersistedTrace {
                origin: origin.clone(),
                steps: Vec::new(),
                halt_reason: None,
            },
        );
        Ok(())
    }

    fn flush_step(&self, id: ThoughtId, step_idx: u32, step: &ThoughtStep) -> ThinkingResult<()> {
        let mut traces = self.traces.lock().map_err(|e| ThinkingError::Persistence {
            message: format!("trace store poisoned: {e}"),
        })?;
        let trace = traces.get_mut(&id.0).ok_or(ThinkingError::Persistence {
            message: format!("flush_step for unknown thought {id}"),
        })?;
        debug_assert_eq!(trace.steps.len() as u32, step_idx, "step indices contiguous");
        trace.steps.push(step.clone());
        Ok(())
    }

    fn conclude(&self, id: ThoughtId, conclusion: &Conclusion) -> ThinkingResult<()> {
        let mut traces = self.traces.lock().map_err(|e| ThinkingError::Persistence {
            message: format!("trace store poisoned: {e}"),
        })?;
        let trace = traces.get_mut(&id.0).ok_or(ThinkingError::Persistence {
            message: format!("conclude for unknown thought {id}"),
        })?;
        trace.halt_reason = Some(conclusion.halt_reason.clone());
        Ok(())
    }
}

/// Persistence that drops everything (benchmarks measuring pure loop cost).
#[derive(Debug, Default, Clone, Copy)]
pub struct NullThoughtPersistence;

impl ThoughtPersistence for NullThoughtPersistence {
    fn begin(&self, _id: ThoughtId, _origin: &ThoughtOrigin) -> ThinkingResult<()> {
        Ok(())
    }
    fn flush_step(&self, _id: ThoughtId, _idx: u32, _step: &ThoughtStep) -> ThinkingResult<()> {
        Ok(())
    }
    fn conclude(&self, _id: ThoughtId, _conclusion: &Conclusion) -> ThinkingResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolId;

    fn origin() -> ThoughtOrigin {
        ThoughtOrigin {
            goal: SymbolId::new(7).unwrap(),
            parent: None,
            started_at: 0,
        }
    }

    #[test]
    fn in_memory_records_steps_in_order() {
        let p = InMemoryThoughtPersistence::new();
        let id = ThoughtId::next();
        p.begin(id, &origin()).unwrap();
        p.flush_step(
            id,
            0,
            &ThoughtStep::SpreadingActivation {
                frontier_size: 1,
                new_concepts: vec![],
                novelty: 0.0,
            },
        )
        .unwrap();
        p.flush_step(
            id,
            1,
            &ThoughtStep::SuperpositionMerge {
                hypotheses_in: 2,
                hypotheses_out: 1,
                mean_similarity: 0.9,
            },
        )
        .unwrap();
        let trace = p.trace(id).unwrap();
        assert_eq!(trace.steps.len(), 2);
        assert_eq!(trace.steps[0].kind(), "spreading-activation");
        assert!(trace.halt_reason.is_none());
    }

    #[test]
    fn flush_to_unknown_thought_errors() {
        let p = InMemoryThoughtPersistence::new();
        let err = p.flush_step(
            ThoughtId(999_999),
            0,
            &ThoughtStep::SpreadingActivation {
                frontier_size: 0,
                new_concepts: vec![],
                novelty: 0.0,
            },
        );
        assert!(matches!(err, Err(ThinkingError::Persistence { .. })));
    }
}
