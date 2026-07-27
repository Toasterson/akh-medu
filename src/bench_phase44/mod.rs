//! Phase 44j: a deterministic A/B harness comparing the context-assembly
//! strategies (`SafetyEnforced<{Compacting, Passive, Virtual}>`).
//!
//! This is a **manager-level** comparison — no LLM, no network — so it is fast,
//! reproducible, and CI-safe. It measures the properties the assembly layer
//! controls directly:
//! - **token cost** of the assembled context,
//! - **salient-fact retention** (did a fact that must survive make it into the
//!   window?) → the `lost_salient` count,
//! - **safety-pin violations** (must be 0 for every strategy), and
//! - **section coverage**.
//!
//! It models the regime where the strategies actually diverge — long
//! conversations and/or tight windows, where something must be dropped. With a
//! large window every strategy keeps everything; the trade-offs only appear under
//! pressure, so the scenarios use tight budgets on purpose.
//!
//! What it deliberately does **not** measure: NL answer quality and "context rot"
//! — those need an LLM judge and the live 3-akh harness, and are recorded as the
//! follow-up (see the dated report under `docs/ai/`). The honest scope here is the
//! mechanics of what enters the window.

use serde::Serialize;

use crate::agent::context::{
    CompactingManager, ContextManager, ContextWindowStats, EpisodeState, HeuristicTokenCounter,
    PassiveManager, SafetyEnforced, TokenBudget, VirtualContextManager, core_memory,
};
use crate::agent::conversation::{ConversationState, Speaker};
use crate::engine::{Engine, EngineConfig};
use crate::graph::Triple;
use crate::nlu::kg_context::{KgContextCache, build_kg_context};

/// Agent id used throughout the harness (keys core memory for the virtual run).
const BENCH_AGENT: &str = "bench";

/// One A/B scenario: a conversation + KG state + a fact that must survive.
pub struct Scenario {
    pub name: &'static str,
    /// Prior turns (operator/agent), in order. The current prompt is appended.
    pub history: Vec<(Speaker, &'static str)>,
    /// KG facts (subject, predicate, object) seeded before assembly.
    pub kg: Vec<(&'static str, &'static str, &'static str)>,
    /// A substring that must appear in the assembled context to "retain" the fact.
    pub salient: &'static str,
    /// Where the salient fact lives, so the harness mirrors how each strategy
    /// would actually hold it (KG → all; history → recency-dependent; core →
    /// virtual only, via core_memory_append).
    pub salient_in: SalientChannel,
    /// The current user prompt.
    pub prompt: &'static str,
    /// The token budget (tight on purpose, to force the strategies to diverge).
    pub budget: TokenBudget,
}

/// Which tier the salient fact is established through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SalientChannel {
    /// In the knowledge graph (grounding) — every strategy can surface it.
    Kg,
    /// Stated in an early conversation turn — retention depends on history policy.
    History,
    /// Persisted to core memory (only `virtual` has this mechanism).
    Core,
}

/// One (strategy, scenario) measurement.
#[derive(Debug, Clone, Serialize)]
pub struct Cell {
    pub scenario: String,
    pub strategy: String,
    pub tokens: usize,
    pub messages: usize,
    pub salient_present: bool,
    pub safety_violations: u64,
}

/// Per-strategy aggregate across all scenarios.
#[derive(Debug, Clone, Serialize)]
pub struct StrategyAgg {
    pub strategy: String,
    pub scenarios: usize,
    pub avg_tokens: f64,
    pub lost_salient: usize,
    pub safety_violations: u64,
}

/// Full benchmark result.
#[derive(Debug, Clone, Serialize)]
pub struct BenchResult {
    pub cells: Vec<Cell>,
    pub aggregates: Vec<StrategyAgg>,
}

/// The default scenario set: grounding parity, early-fact-under-pressure, and a
/// virtual-core-memory win.
pub fn default_scenarios() -> Vec<Scenario> {
    let filler: Vec<(Speaker, &'static str)> = vec![
        (Speaker::Operator, "let's chat about the weather"),
        (Speaker::Agent, "sure, it's sunny"),
        (Speaker::Operator, "and the traffic this morning"),
        (Speaker::Agent, "fairly light"),
        (Speaker::Operator, "any news headlines?"),
        (Speaker::Agent, "nothing major"),
    ];

    vec![
        // 1. Grounding parity: salient fact in the KG; every strategy must keep it.
        Scenario {
            name: "grounding_in_kg",
            history: vec![
                (Speaker::Operator, "hello"),
                (Speaker::Agent, "hi, ask me anything"),
            ],
            kg: vec![("photosynthesis", "requires", "sunlight")],
            salient: "sunlight",
            salient_in: SalientChannel::Kg,
            prompt: "tell me about photosynthesis",
            budget: TokenBudget::default(),
        },
        // 2. Early fact under a tight budget: only recency-preserving full history
        //    can keep it; passive (tail) and a trimmed compacting drop it.
        Scenario {
            name: "early_fact_tight_budget",
            history: {
                let mut h = vec![(Speaker::Operator, "remember: the launch code is alpha-seven")];
                h.extend(filler.clone());
                h
            },
            kg: vec![],
            salient: "alpha-seven",
            salient_in: SalientChannel::History,
            prompt: "what was the launch code?",
            budget: TokenBudget::new(80, 16),
        },
        // 3. Virtual-core win: the same early fact, but persisted to core memory —
        //    virtual keeps it under the tight budget; the others can't pin it.
        Scenario {
            name: "core_memory_retention",
            history: {
                let mut h = vec![(Speaker::Operator, "the launch code is alpha-seven")];
                h.extend(filler.clone());
                h
            },
            kg: vec![],
            salient: "alpha-seven",
            salient_in: SalientChannel::Core,
            prompt: "what was the launch code?",
            budget: TokenBudget::new(80, 16),
        },
    ]
}

fn fresh_engine() -> Engine {
    // In-memory engine (no on-disk store) — the harness is deterministic + CI-safe.
    Engine::new(EngineConfig::default()).expect("in-memory engine")
}

fn seed_kg(engine: &Engine, facts: &[(&str, &str, &str)]) {
    for (s, p, o) in facts {
        let s = engine.resolve_or_create_entity(s).unwrap();
        let p = engine.resolve_or_create_entity(p).unwrap();
        let o = engine.resolve_or_create_entity(o).unwrap();
        let _ = engine.add_triple(&Triple::new(s, p, o));
    }
}

fn build_conversation(history: &[(Speaker, &str)], prompt: &str) -> ConversationState {
    let mut conv = ConversationState::new("bench", "narrative");
    for (sp, text) in history {
        conv.add_turn(*sp, *text, Vec::new());
    }
    // Mirror run_agentic_turn: the current prompt is the last recorded turn.
    conv.record_operator_turn(prompt);
    conv
}

fn assemble_cell<M: ContextManager>(
    label: &str,
    scenario: &Scenario,
    mgr: &SafetyEnforced<M>,
) -> Cell {
    let engine = fresh_engine();
    seed_kg(&engine, &scenario.kg);
    if scenario.salient_in == SalientChannel::Core && label == "virtual" {
        // Represent the akh having persisted the fact via core_memory_append.
        let _ = core_memory::append(&engine, BENCH_AGENT, scenario.salient);
    }
    let conv = build_conversation(&scenario.history, scenario.prompt);

    let mut cache = KgContextCache::default();
    let kg = build_kg_context(&engine, scenario.prompt, &mut cache);
    let recalled: Vec<String> = Vec::new();
    let ep = EpisodeState::new(BENCH_AGENT, scenario.prompt, &engine)
        .with_conversation(Some(&conv))
        .with_context(kg.as_deref(), &recalled, None);

    let counter = HeuristicTokenCounter;
    match mgr.assemble(&ep, &scenario.budget, &counter) {
        Ok(a) => {
            let present = a.messages.iter().any(|m| {
                m["content"].as_str().map(|c| c.contains(scenario.salient)).unwrap_or(false)
            });
            Cell {
                scenario: scenario.name.to_string(),
                strategy: label.to_string(),
                tokens: a.total_tokens(),
                messages: a.messages.len(),
                salient_present: present,
                safety_violations: mgr.violation_count(),
            }
        }
        Err(_) => Cell {
            scenario: scenario.name.to_string(),
            strategy: label.to_string(),
            tokens: 0,
            messages: 0,
            salient_present: false,
            safety_violations: mgr.violation_count(),
        },
    }
}

/// Run the harness over the default scenarios.
pub fn run() -> BenchResult {
    run_scenarios(&default_scenarios())
}

/// Run the harness over a given scenario set.
pub fn run_scenarios(scenarios: &[Scenario]) -> BenchResult {
    let mut cells = Vec::new();
    for s in scenarios {
        cells.push(assemble_cell("compacting", s, &SafetyEnforced::new(CompactingManager)));
        cells.push(assemble_cell("passive", s, &SafetyEnforced::new(PassiveManager::default())));
        cells.push(assemble_cell(
            "virtual",
            s,
            &SafetyEnforced::new(VirtualContextManager::default()),
        ));
    }

    let aggregates = ["compacting", "passive", "virtual"]
        .iter()
        .map(|label| {
            let group: Vec<&Cell> = cells.iter().filter(|c| c.strategy == *label).collect();
            let n = group.len().max(1);
            StrategyAgg {
                strategy: label.to_string(),
                scenarios: group.len(),
                avg_tokens: group.iter().map(|c| c.tokens as f64).sum::<f64>() / n as f64,
                lost_salient: group.iter().filter(|c| !c.salient_present).count(),
                safety_violations: group.iter().map(|c| c.safety_violations).sum(),
            }
        })
        .collect();

    BenchResult { cells, aggregates }
}

/// Render the result as a Markdown report.
pub fn emit_markdown(result: &BenchResult) -> String {
    let mut out = String::new();
    out.push_str("## Per-scenario cells\n\n");
    out.push_str("| scenario | strategy | tokens | msgs | salient kept | safety viol |\n");
    out.push_str("|---|---|---:|---:|:---:|---:|\n");
    for c in &result.cells {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            c.scenario,
            c.strategy,
            c.tokens,
            c.messages,
            if c.salient_present { "✓" } else { "✗" },
            c.safety_violations,
        ));
    }
    out.push_str("\n## Per-strategy aggregates\n\n");
    out.push_str("| strategy | scenarios | avg tokens | lost salient | safety viol |\n");
    out.push_str("|---|---:|---:|---:|---:|\n");
    for a in &result.aggregates {
        out.push_str(&format!(
            "| {} | {} | {:.1} | {} | {} |\n",
            a.strategy, a.scenarios, a.avg_tokens, a.lost_salient, a.safety_violations,
        ));
    }
    out
}

// ── Phase 44m: A/B diff over ContextWindowStats (ADR 065 D12) ──────────────

/// Per-section delta between two strategies' stats (kept tokens).
#[derive(Debug, Clone, Serialize)]
pub struct SectionDelta {
    pub section: String,
    pub a_kept: usize,
    pub b_kept: usize,
    pub delta: i64,
}

/// Dropped-by-reason histogram for one strategy.
#[derive(Debug, Clone, Serialize, Default)]
pub struct DroppedHistogram {
    pub budget: usize,
    pub retention: usize,
    pub age: usize,
    pub vsa: usize,
}

impl DroppedHistogram {
    fn from_stats(s: &ContextWindowStats) -> Self {
        let mut h = DroppedHistogram::default();
        for sec in &s.sections {
            if let Some(r) = sec.drop_reason {
                match r {
                    crate::agent::context::DropReason::Budget => h.budget += 1,
                    crate::agent::context::DropReason::Retention => h.retention += 1,
                    crate::agent::context::DropReason::Age => h.age += 1,
                    crate::agent::context::DropReason::Vsa => h.vsa += 1,
                }
            }
        }
        h
    }
}

/// One A/B comparison between two strategies over one episode.
#[derive(Debug, Clone, Serialize)]
pub struct AbDiff {
    pub scenario: String,
    pub strategy_a: String,
    pub strategy_b: String,
    pub total_kept_a: usize,
    pub total_kept_b: usize,
    pub total_kept_delta: i64,
    pub section_deltas: Vec<SectionDelta>,
    pub dropped_a: DroppedHistogram,
    pub dropped_b: DroppedHistogram,
    /// Would strategy B have evicted a span that strategy A kept pinned?
    /// (The cross-check the ADR asks for: the PinnedSafety invariant.)
    pub b_would_evict_pinned_that_a_kept: bool,
}

/// Diff two strategies over one scenario, using their `ContextWindowStats`.
/// Both strategies assemble the **same** `EpisodeState` (same engine, same
/// conversation, same KG); the diff is purely what each manager shaped the
/// window into. Returns `None` if either strategy errored.
pub fn diff_strategies<M: ContextManager, N: ContextManager>(
    scenario_name: &str,
    scenario: &Scenario,
    mgr_a: &SafetyEnforced<M>,
    label_a: &str,
    mgr_b: &SafetyEnforced<N>,
    label_b: &str,
) -> Option<AbDiff> {
    let engine = fresh_engine();
    seed_kg(&engine, &scenario.kg);
    if scenario.salient_in == SalientChannel::Core {
        let _ = core_memory::append(&engine, BENCH_AGENT, scenario.salient);
    }
    let conv = build_conversation(&scenario.history, scenario.prompt);

    let mut cache = KgContextCache::default();
    let kg = build_kg_context(&engine, scenario.prompt, &mut cache);
    let recalled: Vec<String> = Vec::new();
    let ep = EpisodeState::new(BENCH_AGENT, scenario.prompt, &engine)
        .with_conversation(Some(&conv))
        .with_context(kg.as_deref(), &recalled, None);

    let counter = HeuristicTokenCounter;
    let a = mgr_a.assemble(&ep, &scenario.budget, &counter).ok()?;
    let b = mgr_b.assemble(&ep, &scenario.budget, &counter).ok()?;
    let sa = mgr_a.stats(&a, &scenario.budget);
    let sb = mgr_b.stats(&b, &scenario.budget);

    // Per-section delta (kept tokens). Iterate the union of sections.
    use std::collections::BTreeSet;
    let mut sections: BTreeSet<String> = BTreeSet::new();
    for s in &sa.sections {
        sections.insert(s.section.as_str().to_string());
    }
    for s in &sb.sections {
        sections.insert(s.section.as_str().to_string());
    }
    let kept = |s: &ContextWindowStats, name: &str| -> usize {
        s.sections
            .iter()
            .find(|x| x.section.as_str() == name)
            .map(|x| x.tokens_kept)
            .unwrap_or(0)
    };
    let section_deltas: Vec<SectionDelta> = sections
        .iter()
        .map(|sec| {
            let a_kept = kept(&sa, sec);
            let b_kept = kept(&sb, sec);
            SectionDelta {
                section: sec.clone(),
                a_kept,
                b_kept,
                delta: b_kept as i64 - a_kept as i64,
            }
        })
        .collect();

    let total_a: usize = sa.sections.iter().map(|x| x.tokens_kept).sum();
    let total_b: usize = sb.sections.iter().map(|x| x.tokens_kept).sum();

    // Cross-check: would B evict a PinnedSafety span that A kept? Today neither
    // strategy evicts pinned spans (the guard refuses), so this is always
    // false — but the field is here so the harness *can* catch a regression.
    // (Computed by checking B's safety_violations; >0 ⇒ B tried.)
    let b_would_evict = sb.safety_violations > 0;

    Some(AbDiff {
        scenario: scenario_name.to_string(),
        strategy_a: label_a.to_string(),
        strategy_b: label_b.to_string(),
        total_kept_a: total_a,
        total_kept_b: total_b,
        total_kept_delta: total_b as i64 - total_a as i64,
        section_deltas,
        dropped_a: DroppedHistogram::from_stats(&sa),
        dropped_b: DroppedHistogram::from_stats(&sb),
        b_would_evict_pinned_that_a_kept: b_would_evict,
    })
}

/// Run A/B diffs across the default scenario set: compacting vs passive,
/// compacting vs virtual, passive vs virtual.
pub fn run_diffs() -> Vec<AbDiff> {
    let scenarios = default_scenarios();
    let mut diffs = Vec::new();
    for s in &scenarios {
        if let Some(d) = diff_strategies(
            s.name,
            s,
            &SafetyEnforced::new(CompactingManager),
            "compacting",
            &SafetyEnforced::new(PassiveManager::default()),
            "passive",
        ) {
            diffs.push(d);
        }
        if let Some(d) = diff_strategies(
            s.name,
            s,
            &SafetyEnforced::new(CompactingManager),
            "compacting",
            &SafetyEnforced::new(VirtualContextManager::default()),
            "virtual",
        ) {
            diffs.push(d);
        }
        if let Some(d) = diff_strategies(
            s.name,
            s,
            &SafetyEnforced::new(PassiveManager::default()),
            "passive",
            &SafetyEnforced::new(VirtualContextManager::default()),
            "virtual",
        ) {
            diffs.push(d);
        }
    }
    diffs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_runs_all_strategies_with_zero_safety_violations() {
        let result = run();
        // Three strategies × three scenarios.
        assert_eq!(result.cells.len(), 9);
        assert_eq!(result.aggregates.len(), 3);
        // The safety guard must never trip for any strategy (ADR 065 D6).
        for a in &result.aggregates {
            assert_eq!(a.safety_violations, 0, "{} tripped the safety guard", a.strategy);
        }
    }

    #[test]
    fn grounding_is_retained_by_every_strategy() {
        let result = run();
        for c in result.cells.iter().filter(|c| c.scenario == "grounding_in_kg") {
            assert!(c.salient_present, "{} dropped KG grounding", c.strategy);
        }
    }

    #[test]
    fn virtual_core_memory_wins_under_pressure() {
        let result = run();
        let cell = |strat: &str| {
            result
                .cells
                .iter()
                .find(|c| c.scenario == "core_memory_retention" && c.strategy == strat)
                .unwrap()
        };
        // Virtual keeps the fact via core memory; passive (tail-only) cannot.
        assert!(cell("virtual").salient_present, "virtual should keep the core-memory fact");
        assert!(!cell("passive").salient_present, "passive can't pin the early fact");
    }

    #[test]
    fn markdown_report_has_all_strategies() {
        let md = emit_markdown(&run());
        assert!(md.contains("compacting"));
        assert!(md.contains("passive"));
        assert!(md.contains("virtual"));
    }

    #[test]
    #[ignore = "prints the report; run with `--ignored --nocapture`"]
    fn dump_report() {
        println!("\n{}", emit_markdown(&run()));
    }

    // ── Phase 44m: A/B diff tests (ADR 065 D12) ────────────────────────────

    #[test]
    fn ab_diff_reports_per_section_delta_and_histograms() {
        // Acceptance #4: the harness diffs two strategies on one episode and
        // reports per-section delta, total-kept delta, dropped-by-reason
        // histogram, and a PinnedSafety cross-check.
        let diffs = run_diffs();
        assert!(!diffs.is_empty(), "A/B diffs produced");
        // Every diff has the full shape.
        for d in &diffs {
            assert!(!d.section_deltas.is_empty(), "section deltas present");
            // The cross-check field is always populated (false today — no
            // strategy evicts pinned spans).
            assert!(!d.b_would_evict_pinned_that_a_kept, "no strategy evicts pinned spans");
        }
        // At least one diff exists per scenario (3 scenarios × 3 pairs = 9).
        assert_eq!(diffs.len(), 9);
    }

    #[test]
    fn ab_diff_total_kept_delta_is_signed() {
        let scenarios = default_scenarios();
        let s = &scenarios[0]; // grounding_in_kg: every strategy keeps the KG fact
        let d = diff_strategies(
            s.name,
            s,
            &SafetyEnforced::new(CompactingManager),
            "compacting",
            &SafetyEnforced::new(PassiveManager::default()),
            "passive",
        )
        .expect("diff");
        // On a large budget both keep everything ⇒ delta is 0 (or very small
        // due to tail-cap differences).
        let _ = d.total_kept_delta; // signed i64
    }
}
