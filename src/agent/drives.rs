//! Drive system: CLARION-inspired motivational drives for autonomous goal generation.
//!
//! Four drives observe the agent's knowledge state and working memory to produce
//! strength signals [0.0, 1.0]. When a drive exceeds its threshold, the goal
//! generation pipeline can create new goals to address the underlying need.

use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::provenance::DerivationKind;
use crate::symbol::SymbolId;

use super::memory::{WorkingMemory, WorkingMemoryKind};

// ---------------------------------------------------------------------------
// Drive kinds
// ---------------------------------------------------------------------------

/// The four motivational drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DriveKind {
    /// Desire to acquire new knowledge. High when KG growth stagnates.
    Curiosity,
    /// Desire for consistent knowledge. High when contradictions accumulate.
    Coherence,
    /// Desire for thorough knowledge. High when many gaps exist.
    Completeness,
    /// Desire for effective action. High when tools fail frequently.
    Efficiency,
}

impl DriveKind {
    /// All four drives in canonical order.
    pub const ALL: [DriveKind; 4] = [
        DriveKind::Curiosity,
        DriveKind::Coherence,
        DriveKind::Completeness,
        DriveKind::Efficiency,
    ];

    /// Human-readable label for this drive.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Curiosity => "curiosity",
            Self::Coherence => "coherence",
            Self::Completeness => "completeness",
            Self::Efficiency => "efficiency",
        }
    }
}

impl std::fmt::Display for DriveKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

// ---------------------------------------------------------------------------
// Drive state
// ---------------------------------------------------------------------------

/// Snapshot of engine state used to compute drive deltas.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DriveSnapshot {
    pub triple_count: usize,
    pub symbol_count: usize,
}

/// A single motivational drive with its current strength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Drive {
    pub kind: DriveKind,
    /// Current strength in [0.0, 1.0].
    pub strength: f32,
    /// Cycle number when this drive was last computed.
    pub last_computed: u64,
    /// State snapshot from last computation (for delta-based drives).
    pub last_snapshot: DriveSnapshot,
}

impl Drive {
    fn new(kind: DriveKind) -> Self {
        Self {
            kind,
            strength: 0.0,
            last_computed: 0,
            last_snapshot: DriveSnapshot::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Drive system
// ---------------------------------------------------------------------------

/// The four-drive motivational system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriveSystem {
    /// The four drives in canonical order (Curiosity, Coherence, Completeness, Efficiency).
    pub drives: [Drive; 4],
    /// Per-drive activation thresholds.
    pub thresholds: [f32; 4],
}

/// Default thresholds: [Curiosity=0.6, Coherence=0.3, Completeness=0.5, Efficiency=0.5].
pub const DEFAULT_THRESHOLDS: [f32; 4] = [0.6, 0.3, 0.5, 0.5];

impl DriveSystem {
    /// Create a new drive system with default thresholds.
    pub fn new() -> Self {
        Self::with_thresholds(DEFAULT_THRESHOLDS)
    }

    /// Create a new drive system with custom thresholds.
    pub fn with_thresholds(thresholds: [f32; 4]) -> Self {
        Self {
            drives: [
                Drive::new(DriveKind::Curiosity),
                Drive::new(DriveKind::Coherence),
                Drive::new(DriveKind::Completeness),
                Drive::new(DriveKind::Efficiency),
            ],
            thresholds,
        }
    }

    /// Get a drive's current strength by kind.
    pub fn strength(&self, kind: DriveKind) -> f32 {
        self.drives[kind as usize].strength
    }

    /// Return drives whose strength exceeds their threshold.
    pub fn exceeded_drives(&self) -> Vec<&Drive> {
        self.drives
            .iter()
            .zip(self.thresholds.iter())
            .filter(|(drive, threshold)| drive.strength > **threshold)
            .map(|(drive, _)| drive)
            .collect()
    }

    /// Recompute all four drive strengths from current engine and working memory state.
    pub fn update(
        &mut self,
        engine: &Engine,
        working_memory: &WorkingMemory,
        goal_symbols: &[SymbolId],
        cycle: u64,
    ) {
        // Curiosity: measures KG growth stagnation.
        let info = engine.info();
        self.drives[0].strength =
            compute_curiosity(info.triple_count, &self.drives[0].last_snapshot);
        self.drives[0].last_snapshot = DriveSnapshot {
            triple_count: info.triple_count,
            symbol_count: info.symbol_count,
        };
        self.drives[0].last_computed = cycle;

        // Coherence: measures contradiction accumulation.
        self.drives[1].strength = compute_coherence(engine, info.triple_count);
        self.drives[1].last_computed = cycle;

        // Completeness: measures knowledge coverage gaps.
        self.drives[2].strength = compute_completeness(engine, goal_symbols);
        self.drives[2].last_computed = cycle;

        // Efficiency: measures tool failure rate.
        self.drives[3].strength = compute_efficiency(working_memory);
        self.drives[3].last_computed = cycle;
    }
}

impl Default for DriveSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Private strength computation
// ---------------------------------------------------------------------------

/// Curiosity: `1.0 - growth_rate` where growth_rate is the fraction of new
/// triples since the last snapshot. High when KG growth stagnates.
fn compute_curiosity(current_triples: usize, snapshot: &DriveSnapshot) -> f32 {
    if snapshot.triple_count == 0 {
        // First measurement — no data to compare; assume moderate curiosity.
        return 0.5;
    }
    let growth = current_triples.saturating_sub(snapshot.triple_count) as f32;
    let growth_rate = growth / snapshot.triple_count.max(1) as f32;
    (1.0 - growth_rate).clamp(0.0, 1.0)
}

/// Coherence: fraction of triples with contradiction provenance.
/// High when many contradictions have been detected.
fn compute_coherence(engine: &Engine, total_triples: usize) -> f32 {
    if total_triples == 0 {
        return 0.0;
    }

    // Count ContradictionDetected provenance records.
    let contradiction_count = engine
        .provenance_by_kind(&DerivationKind::ContradictionDetected {
            kind: String::new(),
            existing_object: SymbolId::new(1).unwrap(),
            incoming_object: SymbolId::new(1).unwrap(),
        })
        .map(|records| records.len())
        .unwrap_or(0);

    (contradiction_count as f32 / total_triples.max(1) as f32).clamp(0.0, 1.0)
}

/// Completeness: `1.0 - coverage_score` from gap analysis.
/// High when many gaps exist (low coverage).
fn compute_completeness(engine: &Engine, goal_symbols: &[SymbolId]) -> f32 {
    let config = crate::autonomous::gap::GapAnalysisConfig::default();
    match engine.analyze_gaps(goal_symbols, config) {
        Ok(result) => (1.0 - result.coverage_score).clamp(0.0, 1.0),
        Err(_) => 0.0, // Can't analyze → no signal.
    }
}

/// Efficiency: fraction of tool results that contain error/failure indicators.
/// High when tools fail frequently.
fn compute_efficiency(working_memory: &WorkingMemory) -> f32 {
    let tool_results = working_memory.by_kind(WorkingMemoryKind::ToolResult);
    let total = tool_results.len();
    if total == 0 {
        return 0.0;
    }

    let failures = tool_results
        .iter()
        .filter(|e| {
            let lower = e.content.to_lowercase();
            lower.contains("error") || lower.contains("failed")
        })
        .count();

    (failures as f32 / total as f32).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::WorkingMemoryEntry;

    #[test]
    fn drive_kind_labels() {
        assert_eq!(DriveKind::Curiosity.label(), "curiosity");
        assert_eq!(DriveKind::Coherence.label(), "coherence");
        assert_eq!(DriveKind::Completeness.label(), "completeness");
        assert_eq!(DriveKind::Efficiency.label(), "efficiency");
    }

    #[test]
    fn curiosity_high_when_stagnant() {
        let snapshot = DriveSnapshot {
            triple_count: 100,
            symbol_count: 50,
        };
        // No new triples → growth_rate = 0 → curiosity = 1.0.
        let strength = compute_curiosity(100, &snapshot);
        assert!((strength - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn curiosity_low_when_growing() {
        let snapshot = DriveSnapshot {
            triple_count: 100,
            symbol_count: 50,
        };
        // Doubled triples → growth_rate = 1.0 → curiosity = 0.0.
        let strength = compute_curiosity(200, &snapshot);
        assert!((strength - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn curiosity_first_measurement() {
        let snapshot = DriveSnapshot::default();
        let strength = compute_curiosity(0, &snapshot);
        assert!((strength - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn efficiency_zero_with_no_results() {
        let wm = WorkingMemory::new(20);
        let strength = compute_efficiency(&wm);
        assert!((strength - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn efficiency_tracks_failures() {
        let mut wm = WorkingMemory::new(20);
        wm.push(WorkingMemoryEntry {
            id: 0,
            content: "Tool result (kg_query):\nFound 3 triples".into(),
            symbols: vec![],
            kind: WorkingMemoryKind::ToolResult,
            timestamp: 0,
            relevance: 0.6,
            source_cycle: 1,
            reference_count: 0,
            access_timestamps: Vec::new(),
        })
        .unwrap();
        wm.push(WorkingMemoryEntry {
            id: 0,
            content: "Tool result (reason):\nerror: no rules match".into(),
            symbols: vec![],
            kind: WorkingMemoryKind::ToolResult,
            timestamp: 0,
            relevance: 0.6,
            source_cycle: 2,
            reference_count: 0,
            access_timestamps: Vec::new(),
        })
        .unwrap();
        // 1 failure out of 2 → 0.5
        let strength = compute_efficiency(&wm);
        assert!((strength - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn exceeded_drives_filters_correctly() {
        let mut system = DriveSystem::with_thresholds([0.5, 0.5, 0.5, 0.5]);
        system.drives[0].strength = 0.8; // Curiosity exceeds 0.5.
        system.drives[1].strength = 0.3; // Coherence below 0.5.
        system.drives[2].strength = 0.6; // Completeness exceeds 0.5.
        system.drives[3].strength = 0.4; // Efficiency below 0.5.

        let exceeded = system.exceeded_drives();
        assert_eq!(exceeded.len(), 2);
        assert_eq!(exceeded[0].kind, DriveKind::Curiosity);
        assert_eq!(exceeded[1].kind, DriveKind::Completeness);
    }

    #[test]
    fn drive_system_default_thresholds() {
        let system = DriveSystem::new();
        assert_eq!(system.thresholds, DEFAULT_THRESHOLDS);
        for drive in &system.drives {
            assert!((drive.strength - 0.0).abs() < f32::EPSILON);
        }
    }
}
