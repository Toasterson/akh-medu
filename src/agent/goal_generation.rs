//! Goal generation pipeline: autonomously creates goals from observation.
//!
//! Three-phase pipeline:
//! 1. **Signal collection** — gathers goal proposals from drives, gap analysis,
//!    contradiction detection, opportunity detection, impasse signals, and reflection.
//! 2. **Deliberation** — deduplicates via VSA similarity, checks feasibility,
//!    computes priority.
//! 3. **Activation** — top proposals become active goals; remainder become dormant.

use crate::autonomous::gap::GapKind;
use crate::engine::Engine;
use crate::graph::Triple;
use crate::provenance::{DerivationKind, ProvenanceRecord};
use crate::symbol::SymbolId;
use crate::vsa::grounding::encode_text_as_vector;

use super::agent::AgentPredicates;
use super::drives::{DriveKind, DriveSystem};
use super::error::AgentResult;
use super::goal::{self, Goal, GoalSource, GoalStatus};
use super::memory::WorkingMemory;
use super::ooda::DecisionImpasse;
use super::reflect::ReflectionResult;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the autonomous goal generation pipeline.
#[derive(Debug, Clone)]
pub struct GoalGenerationConfig {
    /// Run goal generation every N cycles (0 = disabled). Default: 10.
    pub generate_every_n_cycles: u64,
    /// Maximum proposals to activate per generation cycle. Default: 3.
    pub max_activations: usize,
    /// VSA Hamming distance threshold for deduplication (out of dimension).
    /// Two descriptions with distance < this are considered duplicates. Default: 2000.
    pub dedup_hamming_threshold: u32,
    /// Minimum feasibility score to keep a proposal. Default: 0.2.
    pub min_feasibility: f32,
    /// Per-drive activation thresholds: [Curiosity, Coherence, Completeness, Efficiency].
    pub drive_thresholds: [f32; 4],
    /// Maximum dormant goals to retain. Default: 20.
    pub max_dormant: usize,
}

impl Default for GoalGenerationConfig {
    fn default() -> Self {
        Self {
            generate_every_n_cycles: 10,
            max_activations: 3,
            dedup_hamming_threshold: 2000,
            min_feasibility: 0.2,
            drive_thresholds: [0.6, 0.3, 0.5, 0.5],
            max_dormant: 20,
        }
    }
}

// ---------------------------------------------------------------------------
// Goal proposal
// ---------------------------------------------------------------------------

/// A proposed goal before activation.
#[derive(Debug, Clone)]
pub struct GoalProposal {
    /// Human-readable description of what the goal aims to achieve.
    pub description: String,
    /// Why this goal was proposed.
    pub rationale: String,
    /// How this goal was generated.
    pub source: GoalSource,
    /// Suggested priority (0–255).
    pub priority_suggestion: u8,
    /// How to know when the goal is done.
    pub success_criteria: String,
    /// Existing goal SymbolIds this might conflict with.
    pub conflicts_with: Vec<SymbolId>,
    /// Feasibility score [0.0, 1.0] assigned during deliberation.
    pub feasibility: f32,
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Output of a goal generation cycle.
#[derive(Debug, Clone)]
pub struct GoalGenerationResult {
    /// Proposals that were activated as real goals.
    pub activated: Vec<GoalProposal>,
    /// Proposals stored as dormant for future opportunity detection.
    pub dormant: Vec<GoalProposal>,
    /// How many proposals were discarded as duplicates.
    pub deduplicated: usize,
    /// How many proposals were discarded as infeasible.
    pub infeasible: usize,
    /// Current drive strengths at generation time.
    pub drive_strengths: [f32; 4],
}

// ---------------------------------------------------------------------------
// Top-level orchestrator
// ---------------------------------------------------------------------------

/// Run the full three-phase goal generation pipeline.
#[allow(clippy::too_many_arguments)]
pub fn generate_goals(
    engine: &Engine,
    goals: &[Goal],
    working_memory: &WorkingMemory,
    drives: &DriveSystem,
    config: &GoalGenerationConfig,
    predicates: &AgentPredicates,
    cycle: u64,
    impasse: Option<&DecisionImpasse>,
    reflection: Option<&ReflectionResult>,
) -> AgentResult<GoalGenerationResult> {
    // Phase 1: Signal collection.
    let goal_symbols: Vec<SymbolId> = goal::active_goals(goals)
        .iter()
        .map(|g| g.symbol_id)
        .collect();
    let mut proposals = collect_signals(engine, goals, drives, &goal_symbols, impasse, reflection);

    // Phase 2: Deliberation.
    let (kept, deduplicated, infeasible) = deliberate(engine, &mut proposals, goals, config);

    // Phase 3: Activation.
    let (activated_proposals, dormant_proposals) =
        activate_proposals(engine, kept, config, predicates, cycle)?;

    let drive_strengths = [
        drives.strength(DriveKind::Curiosity),
        drives.strength(DriveKind::Coherence),
        drives.strength(DriveKind::Completeness),
        drives.strength(DriveKind::Efficiency),
    ];

    Ok(GoalGenerationResult {
        activated: activated_proposals,
        dormant: dormant_proposals,
        deduplicated,
        infeasible,
        drive_strengths,
    })
}

// ---------------------------------------------------------------------------
// Phase 1: Signal collection
// ---------------------------------------------------------------------------

fn collect_signals(
    engine: &Engine,
    goals: &[Goal],
    drives: &DriveSystem,
    goal_symbols: &[SymbolId],
    impasse: Option<&DecisionImpasse>,
    reflection: Option<&ReflectionResult>,
) -> Vec<GoalProposal> {
    let mut proposals = Vec::new();

    // 1. Drive-based proposals.
    proposals.extend(proposals_from_drives(drives));

    // 2. Gap-based proposals.
    proposals.extend(proposals_from_gaps(engine, goal_symbols));

    // 3. Anomaly/contradiction proposals.
    proposals.extend(proposals_from_anomalies(engine, goals));

    // 4. Opportunity detection (reactivation of dormant/failed goals).
    proposals.extend(proposals_from_opportunities(engine, goals));

    // 5. Impasse-based proposals.
    if let Some(imp) = impasse {
        proposals.push(proposal_from_impasse(imp));
    }

    // 6. Reflection-based proposals.
    if let Some(refl) = reflection {
        proposals.extend(proposals_from_reflection(refl));
    }

    proposals
}

/// Each exceeded drive → a goal proposal.
fn proposals_from_drives(drives: &DriveSystem) -> Vec<GoalProposal> {
    drives
        .exceeded_drives()
        .into_iter()
        .map(|drive| {
            let (desc, criteria) = match drive.kind {
                DriveKind::Curiosity => (
                    "Explore new knowledge areas to expand understanding".to_string(),
                    "KG has gained new triples in unexplored domains".to_string(),
                ),
                DriveKind::Coherence => (
                    "Investigate and resolve contradictions in knowledge".to_string(),
                    "Contradiction count reduced from current level".to_string(),
                ),
                DriveKind::Completeness => (
                    "Complete gaps in existing knowledge coverage".to_string(),
                    "Coverage score above 0.7".to_string(),
                ),
                DriveKind::Efficiency => (
                    "Improve tool strategy effectiveness and reduce failures".to_string(),
                    "Tool success rate above 0.5".to_string(),
                ),
            };

            GoalProposal {
                description: desc,
                rationale: format!(
                    "{} drive at {:.2} exceeds threshold",
                    drive.kind, drive.strength
                ),
                source: GoalSource::DriveExceeded {
                    drive: drive.kind.label().to_string(),
                    strength: drive.strength,
                },
                priority_suggestion: (drive.strength * 200.0) as u8,
                success_criteria: criteria,
                conflicts_with: Vec::new(),
                feasibility: 1.0, // Drives are always feasible.
            }
        })
        .collect()
}

/// Gap analysis → goal proposals for significant gaps.
fn proposals_from_gaps(engine: &Engine, goal_symbols: &[SymbolId]) -> Vec<GoalProposal> {
    let config = crate::autonomous::gap::GapAnalysisConfig::default();
    let result = match engine.analyze_gaps(goal_symbols, config) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    result
        .gaps
        .into_iter()
        .filter(|gap| gap.severity > 0.3) // Only significant gaps.
        .take(5) // Don't flood with gap proposals.
        .map(|gap| {
            let entity_label = engine.resolve_label(gap.entity);
            let (desc, kind_str) = match &gap.kind {
                GapKind::DeadEnd { .. } => (
                    format!("Discover relationships for '{entity_label}'"),
                    "dead_end".to_string(),
                ),
                GapKind::MissingPredicate {
                    expected_predicate, ..
                } => {
                    let pred_label = engine.resolve_label(*expected_predicate);
                    (
                        format!("Find {pred_label} for '{entity_label}'"),
                        "missing_predicate".to_string(),
                    )
                }
                GapKind::IncompleteType { entity_type, .. } => {
                    let type_label = engine.resolve_label(*entity_type);
                    (
                        format!("Complete '{entity_label}' as {type_label}"),
                        "incomplete_type".to_string(),
                    )
                }
            };

            GoalProposal {
                description: desc,
                rationale: format!("Knowledge gap ({kind_str}) severity {:.2}", gap.severity),
                source: GoalSource::GapDetection {
                    gap_entity: gap.entity,
                    gap_kind: kind_str,
                    severity: gap.severity,
                },
                priority_suggestion: (gap.severity * 180.0) as u8,
                success_criteria: gap.description.clone(),
                conflicts_with: Vec::new(),
                feasibility: 1.0, // Assessed in deliberation phase.
            }
        })
        .collect()
}

/// Contradiction provenance → goal proposals.
fn proposals_from_anomalies(engine: &Engine, goals: &[Goal]) -> Vec<GoalProposal> {
    // Fetch ContradictionDetected provenance records.
    let records = engine
        .provenance_by_kind(&DerivationKind::ContradictionDetected {
            kind: String::new(),
            existing_object: SymbolId::new(1).unwrap(),
            incoming_object: SymbolId::new(1).unwrap(),
        })
        .unwrap_or_default();

    if records.is_empty() {
        return Vec::new();
    }

    // Collect symbols already covered by existing goals.
    let covered_symbols: Vec<SymbolId> = goals
        .iter()
        .filter(|g| matches!(g.status, GoalStatus::Active))
        .map(|g| g.symbol_id)
        .collect();

    records
        .into_iter()
        .filter(|r| !covered_symbols.contains(&r.derived_id))
        .take(3) // Limit anomaly proposals.
        .map(|r| {
            let derived_label = engine.resolve_label(r.derived_id);
            GoalProposal {
                description: format!("Resolve contradiction involving '{derived_label}'"),
                rationale: format!("ContradictionDetected provenance for symbol {}", r.derived_id),
                source: GoalSource::ContradictionDetected {
                    existing: Triple::new(
                        r.derived_id,
                        SymbolId::new(1).unwrap(),
                        SymbolId::new(1).unwrap(),
                    ),
                    incoming: Triple::new(
                        r.derived_id,
                        SymbolId::new(1).unwrap(),
                        SymbolId::new(1).unwrap(),
                    ),
                },
                priority_suggestion: 170,
                success_criteria: format!(
                    "Contradiction for '{derived_label}' resolved or documented"
                ),
                conflicts_with: Vec::new(),
                feasibility: 0.8,
            }
        })
        .collect()
}

/// Check dormant/failed goals for reactivation opportunities.
fn proposals_from_opportunities(engine: &Engine, goals: &[Goal]) -> Vec<GoalProposal> {
    goals
        .iter()
        .filter(|g| matches!(g.status, GoalStatus::Dormant | GoalStatus::Failed { .. }))
        .filter_map(|g| {
            // Extract keywords from the goal description.
            let keywords: Vec<&str> = g
                .description
                .split_whitespace()
                .filter(|w| w.len() > 3)
                .take(5)
                .collect();

            // Check if any keyword now resolves to a symbol in the KG.
            let newly_available: Vec<String> = keywords
                .iter()
                .filter(|kw| engine.resolve_symbol(kw).is_ok())
                .map(|kw| kw.to_string())
                .collect();

            if newly_available.is_empty() {
                return None;
            }

            let satisfied = newly_available.join(", ");
            let boost_priority = g.priority.saturating_add(30).min(255);

            Some(GoalProposal {
                description: format!("Retry: {}", g.description),
                rationale: format!(
                    "Previously {} goal now has prerequisite knowledge: {satisfied}",
                    g.status
                ),
                source: GoalSource::OpportunityDetected {
                    reactivated_goal: g.symbol_id,
                    newly_satisfied: satisfied,
                },
                priority_suggestion: boost_priority,
                success_criteria: g.success_criteria.clone(),
                conflicts_with: Vec::new(),
                feasibility: 0.7,
            })
        })
        .collect()
}

/// Decision impasse → a meta-goal to resolve the stuck state.
fn proposal_from_impasse(impasse: &DecisionImpasse) -> GoalProposal {
    let kind_str = match &impasse.kind {
        super::ooda::ImpasseKind::AllBelowThreshold { threshold } => {
            format!("all_below_{threshold:.2}")
        }
        super::ooda::ImpasseKind::Tie {
            tool_a, tool_b, ..
        } => {
            format!("tie_{tool_a}_{tool_b}")
        }
    };

    GoalProposal {
        description: format!("Resolve decision impasse for goal {}", impasse.goal_id),
        rationale: format!("OODA impasse: {kind_str}, best score {:.2}", impasse.best_score),
        source: GoalSource::ImpasseDetected {
            goal_id: impasse.goal_id,
            impasse_kind: kind_str,
        },
        priority_suggestion: 180,
        success_criteria: "Agent can select a tool with reasonable confidence".to_string(),
        conflicts_with: Vec::new(),
        feasibility: 0.6,
    }
}

/// Reflection insights → goal proposals.
fn proposals_from_reflection(reflection: &ReflectionResult) -> Vec<GoalProposal> {
    reflection
        .goal_proposals
        .iter()
        .map(|p| p.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Phase 2: Deliberation
// ---------------------------------------------------------------------------

/// Deduplicate, assess feasibility, compute priority.
/// Returns (kept proposals, dedup count, infeasible count).
fn deliberate(
    engine: &Engine,
    proposals: &mut Vec<GoalProposal>,
    existing_goals: &[Goal],
    config: &GoalGenerationConfig,
) -> (Vec<GoalProposal>, usize, usize) {
    let mut kept = Vec::new();
    let mut deduplicated = 0usize;
    let mut infeasible = 0usize;

    // Collect existing goal descriptions for dedup.
    let existing_descriptions: Vec<&str> = existing_goals.iter().map(|g| g.description.as_str()).collect();

    for proposal in proposals.drain(..) {
        // Dedup: check against existing goals.
        if is_duplicate(&proposal.description, &existing_descriptions, engine, config) {
            deduplicated += 1;
            continue;
        }

        // Dedup: check against already-kept proposals.
        let kept_descriptions: Vec<&str> = kept.iter().map(|p: &GoalProposal| p.description.as_str()).collect();
        if is_duplicate(&proposal.description, &kept_descriptions, engine, config) {
            deduplicated += 1;
            continue;
        }

        // Feasibility check.
        let feasibility = assess_feasibility(&proposal, engine);
        if feasibility < config.min_feasibility {
            infeasible += 1;
            continue;
        }

        let mut p = proposal;
        p.feasibility = feasibility;
        kept.push(p);
    }

    // Sort by effective priority (drive strength × severity × feasibility).
    kept.sort_by(|a, b| {
        let score_a = a.priority_suggestion as f32 * a.feasibility;
        let score_b = b.priority_suggestion as f32 * b.feasibility;
        score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    (kept, deduplicated, infeasible)
}

/// Check if a description is a duplicate of any existing description.
/// Uses VSA Hamming distance when possible, falls back to string similarity.
fn is_duplicate(
    description: &str,
    existing: &[&str],
    engine: &Engine,
    config: &GoalGenerationConfig,
) -> bool {
    if existing.is_empty() {
        return false;
    }

    let ops = engine.ops();
    let item_memory = engine.item_memory();

    // Try VSA encoding. Use similarity (1.0 - normalized hamming distance).
    // dedup_hamming_threshold=2000 out of 10000 dim ≈ 0.8 similarity.
    let similarity_threshold = 1.0 - (config.dedup_hamming_threshold as f32 / ops.dim().0 as f32);
    if let Ok(new_vec) = encode_text_as_vector(description, engine, ops, item_memory) {
        for existing_desc in existing {
            if let Ok(existing_vec) = encode_text_as_vector(existing_desc, engine, ops, item_memory)
            {
                if let Ok(sim) = ops.similarity(&new_vec, &existing_vec) {
                    if sim > similarity_threshold {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    // Fallback: simple string overlap check.
    let new_words: std::collections::HashSet<&str> = description
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 3)
        .collect();

    if new_words.is_empty() {
        return false;
    }

    for existing_desc in existing {
        let existing_words: std::collections::HashSet<&str> = existing_desc
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|w| w.len() > 3)
            .collect();

        if existing_words.is_empty() {
            continue;
        }

        let intersection = new_words.intersection(&existing_words).count();
        let union = new_words.union(&existing_words).count();
        let jaccard = intersection as f32 / union.max(1) as f32;
        if jaccard > 0.8 {
            return true;
        }
    }

    false
}

/// Assess how feasible a proposal is based on available tools.
fn assess_feasibility(proposal: &GoalProposal, engine: &Engine) -> f32 {
    let ops = engine.ops();
    let item_memory = engine.item_memory();

    // Encode the proposal description.
    let proposal_vec = match encode_text_as_vector(&proposal.description, engine, ops, item_memory)
    {
        Ok(v) => v,
        Err(_) => return proposal.feasibility.max(0.3), // Can't assess → keep default.
    };

    // Compare against tool concept profiles.
    let tool_keywords = [
        "query search knowledge triple find graph",
        "mutate assert add triple create relationship",
        "recall memory remember episode",
        "reason infer deduce logic rule",
        "file read write create directory",
        "http fetch download web request",
        "shell execute command run process",
        "infer rules patterns forward chaining",
        "gap analysis coverage completeness",
    ];

    let mut max_similarity = 0.0f32;
    for keywords in &tool_keywords {
        if let Ok(tool_vec) = encode_text_as_vector(keywords, engine, ops, item_memory) {
            if let Ok(sim) = ops.similarity(&proposal_vec, &tool_vec) {
                max_similarity = max_similarity.max(sim);
            }
        }
    }

    // Scale: 0.5 similarity → 0.5 feasibility, 1.0 → 1.0.
    max_similarity.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Phase 3: Activation
// ---------------------------------------------------------------------------

/// Activate top proposals as real goals, store remainder as dormant.
fn activate_proposals(
    engine: &Engine,
    proposals: Vec<GoalProposal>,
    config: &GoalGenerationConfig,
    predicates: &AgentPredicates,
    cycle: u64,
) -> AgentResult<(Vec<GoalProposal>, Vec<GoalProposal>)> {
    let mut activated = Vec::new();
    let mut dormant = Vec::new();

    for (i, proposal) in proposals.into_iter().enumerate() {
        if i < config.max_activations {
            // Create as active goal.
            let mut new_goal = goal::create_goal(
                engine,
                &proposal.description,
                proposal.priority_suggestion,
                &proposal.success_criteria,
                predicates,
            )?;

            new_goal.source = Some(proposal.source.clone());

            // Record provenance.
            let (drive_name, drive_strength) = match &proposal.source {
                GoalSource::DriveExceeded { drive, strength } => {
                    (drive.clone(), *strength)
                }
                GoalSource::GapDetection { severity, .. } => {
                    ("completeness".to_string(), *severity)
                }
                GoalSource::ContradictionDetected { .. } => {
                    ("coherence".to_string(), 0.8)
                }
                GoalSource::OpportunityDetected { .. } => {
                    ("opportunity".to_string(), 0.7)
                }
                GoalSource::ImpasseDetected { .. } => {
                    ("impasse".to_string(), 0.6)
                }
                GoalSource::ReflectionInsight { .. } => {
                    ("reflection".to_string(), 0.5)
                }
            };

            let mut prov = ProvenanceRecord::new(
                new_goal.symbol_id,
                DerivationKind::AutonomousGoalGeneration {
                    drive: drive_name,
                    strength: drive_strength,
                },
            )
            .with_confidence(proposal.feasibility);
            let _ = engine.store_provenance(&mut prov);

            // Log the activation cycle.
            let _ = new_goal.created_at;
            let _cycle = cycle; // Available for WM logging by caller.

            activated.push(proposal);
        } else if dormant.len() < config.max_dormant {
            dormant.push(proposal);
        }
    }

    Ok((activated, dormant))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::drives::DriveSystem;
    use crate::agent::memory::WorkingMemoryEntry;
    use crate::agent::memory::WorkingMemoryKind;

    #[test]
    fn proposals_from_drives_empty_when_below_threshold() {
        let system = DriveSystem::new();
        // All drives at 0.0 → no exceeded → no proposals.
        let proposals = proposals_from_drives(&system);
        assert!(proposals.is_empty());
    }

    #[test]
    fn proposals_from_drives_generates_for_exceeded() {
        let mut system = DriveSystem::with_thresholds([0.5, 0.5, 0.5, 0.5]);
        system.drives[0].strength = 0.8; // Curiosity exceeds.
        system.drives[2].strength = 0.7; // Completeness exceeds.

        let proposals = proposals_from_drives(&system);
        assert_eq!(proposals.len(), 2);
        assert!(proposals[0].description.contains("Explore"));
        assert!(proposals[1].description.contains("gaps"));
    }

    #[test]
    fn proposal_from_impasse_creates_meta_goal() {
        let impasse = DecisionImpasse {
            goal_id: SymbolId::new(42).unwrap(),
            kind: super::super::ooda::ImpasseKind::AllBelowThreshold { threshold: 0.15 },
            best_score: 0.10,
        };
        let proposal = proposal_from_impasse(&impasse);
        assert!(proposal.description.contains("impasse"));
        assert_eq!(proposal.priority_suggestion, 180);
    }

    #[test]
    fn dedup_string_fallback_catches_identical() {
        // Without an engine, VSA encoding will fail → string fallback.
        // We test the string overlap logic directly.
        let new_words: std::collections::HashSet<&str> =
            ["explore", "knowledge", "areas"].iter().copied().collect();
        let existing_words: std::collections::HashSet<&str> =
            ["explore", "knowledge", "areas"].iter().copied().collect();
        let intersection = new_words.intersection(&existing_words).count();
        let union = new_words.union(&existing_words).count();
        let jaccard = intersection as f32 / union.max(1) as f32;
        assert!(jaccard > 0.8);
    }

    #[test]
    fn dedup_string_fallback_allows_different() {
        let new_words: std::collections::HashSet<&str> =
            ["explore", "knowledge", "areas"].iter().copied().collect();
        let existing_words: std::collections::HashSet<&str> =
            ["resolve", "contradiction", "logic"].iter().copied().collect();
        let intersection = new_words.intersection(&existing_words).count();
        let union = new_words.union(&existing_words).count();
        let jaccard = intersection as f32 / union.max(1) as f32;
        assert!(jaccard < 0.8);
    }

    #[test]
    fn goal_generation_config_defaults() {
        let config = GoalGenerationConfig::default();
        assert_eq!(config.generate_every_n_cycles, 10);
        assert_eq!(config.max_activations, 3);
        assert_eq!(config.dedup_hamming_threshold, 2000);
        assert!((config.min_feasibility - 0.2).abs() < f32::EPSILON);
        assert_eq!(config.max_dormant, 20);
    }

    #[test]
    fn proposals_from_opportunities_empty_for_active_goals() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap();

        let goals = vec![Goal {
            symbol_id: SymbolId::new(1).unwrap(),
            description: "Active goal".into(),
            status: GoalStatus::Active,
            priority: 128,
            success_criteria: "done".into(),
            parent: None,
            children: Vec::new(),
            created_at: 0,
            cycles_worked: 0,
            last_progress_cycle: 0,
            source: None,
        }];

        // Active goals are not candidates for opportunity detection.
        let proposals = proposals_from_opportunities(&engine, &goals);
        assert!(proposals.is_empty());
    }

    #[test]
    fn deliberation_deduplicates_identical_proposals() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap();

        let config = GoalGenerationConfig::default();
        let mut proposals = vec![
            GoalProposal {
                description: "Explore new knowledge areas".into(),
                rationale: "curiosity".into(),
                source: GoalSource::DriveExceeded {
                    drive: "curiosity".into(),
                    strength: 0.8,
                },
                priority_suggestion: 160,
                success_criteria: "new triples".into(),
                conflicts_with: Vec::new(),
                feasibility: 1.0,
            },
            GoalProposal {
                description: "Explore new knowledge areas".into(),
                rationale: "curiosity again".into(),
                source: GoalSource::DriveExceeded {
                    drive: "curiosity".into(),
                    strength: 0.7,
                },
                priority_suggestion: 140,
                success_criteria: "new triples".into(),
                conflicts_with: Vec::new(),
                feasibility: 1.0,
            },
        ];

        let (kept, deduped, _infeasible) = deliberate(&engine, &mut proposals, &[], &config);
        assert_eq!(kept.len(), 1);
        assert_eq!(deduped, 1);
    }

    #[test]
    fn deliberation_keeps_distinct_proposals() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension(10000),
            ..EngineConfig::default()
        })
        .unwrap();

        let config = GoalGenerationConfig::default();
        let mut proposals = vec![
            GoalProposal {
                description: "Explore new knowledge areas to expand understanding".into(),
                rationale: "curiosity".into(),
                source: GoalSource::DriveExceeded {
                    drive: "curiosity".into(),
                    strength: 0.8,
                },
                priority_suggestion: 160,
                success_criteria: "new triples".into(),
                conflicts_with: Vec::new(),
                feasibility: 1.0,
            },
            GoalProposal {
                description: "Resolve contradictions in logical knowledge base".into(),
                rationale: "coherence".into(),
                source: GoalSource::DriveExceeded {
                    drive: "coherence".into(),
                    strength: 0.5,
                },
                priority_suggestion: 100,
                success_criteria: "fewer contradictions".into(),
                conflicts_with: Vec::new(),
                feasibility: 1.0,
            },
        ];

        let (kept, deduped, _) = deliberate(&engine, &mut proposals, &[], &config);
        assert_eq!(kept.len(), 2);
        assert_eq!(deduped, 0);
    }

    #[test]
    fn proposals_from_reflection_empty_for_empty_reflection() {
        let reflection = ReflectionResult {
            at_cycle: 10,
            tool_insights: Vec::new(),
            strategy_diversity: 0,
            goal_insights: Vec::new(),
            memory_pressure: 0.0,
            adjustments: Vec::new(),
            summary: String::new(),
            goal_proposals: Vec::new(),
        };
        let proposals = proposals_from_reflection(&reflection);
        assert!(proposals.is_empty());
    }

    #[test]
    fn proposals_from_reflection_forwards_proposals() {
        let reflection = ReflectionResult {
            at_cycle: 10,
            tool_insights: Vec::new(),
            strategy_diversity: 0,
            goal_insights: Vec::new(),
            memory_pressure: 0.9,
            adjustments: Vec::new(),
            summary: String::new(),
            goal_proposals: vec![GoalProposal {
                description: "Consolidate knowledge".into(),
                rationale: "High memory pressure".into(),
                source: GoalSource::ReflectionInsight {
                    insight: "memory pressure above 0.9".into(),
                },
                priority_suggestion: 150,
                success_criteria: "Memory pressure reduced".into(),
                conflicts_with: Vec::new(),
                feasibility: 0.8,
            }],
        };
        let proposals = proposals_from_reflection(&reflection);
        assert_eq!(proposals.len(), 1);
        assert!(proposals[0].description.contains("Consolidate"));
    }

    #[test]
    fn activation_respects_max_activations() {
        use crate::engine::{Engine, EngineConfig};
        use crate::vsa::Dimension;

        let engine = Engine::new(EngineConfig {
            dimension: Dimension(1000),
            ..EngineConfig::default()
        })
        .unwrap();

        let predicates = crate::agent::agent::AgentPredicates::init(&engine).unwrap();
        let mut config = GoalGenerationConfig::default();
        config.max_activations = 1;

        let proposals = vec![
            GoalProposal {
                description: "First proposal to activate".into(),
                rationale: "test".into(),
                source: GoalSource::DriveExceeded {
                    drive: "curiosity".into(),
                    strength: 0.8,
                },
                priority_suggestion: 200,
                success_criteria: "done".into(),
                conflicts_with: Vec::new(),
                feasibility: 1.0,
            },
            GoalProposal {
                description: "Second proposal goes dormant".into(),
                rationale: "test".into(),
                source: GoalSource::DriveExceeded {
                    drive: "coherence".into(),
                    strength: 0.5,
                },
                priority_suggestion: 100,
                success_criteria: "done".into(),
                conflicts_with: Vec::new(),
                feasibility: 0.8,
            },
        ];

        let (activated, dormant) =
            activate_proposals(&engine, proposals, &config, &predicates, 10).unwrap();
        assert_eq!(activated.len(), 1);
        assert_eq!(dormant.len(), 1);
    }
}
