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
    CompactingManager, ContextManager, EpisodeState, HeuristicTokenCounter, PassiveManager,
    SafetyEnforced, TokenBudget, VirtualContextManager, core_memory,
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
}
