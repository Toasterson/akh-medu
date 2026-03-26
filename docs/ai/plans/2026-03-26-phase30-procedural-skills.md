# Phase 30 — Procedural Skills

> Date: 2026-03-26
> Status: Planned
> Phase: 30
> Depends on: Phase 26b (Candle LLM), Phase 15a (causal schemas), Phase 11 (HTN)
> Enhances: Phase 29b (skill exposure), Phase 31 (skill intelligence)
> ADR: [040-procedural-skills](../decisions/040-procedural-skills.md)

## Motivation

Akh-medu skills currently carry facts and inference rules but no procedural
knowledge — "how to do things." A human expects a "skill" to mean capability:
"I know how to review code", not just "I know facts about code review."

The engine already has procedural representation mechanisms (HTN, causal schemas,
MCTS, event calculus) but they're disconnected from the skill system. And the
formal syntax for egg rules is a barrier for most users.

This phase bridges both gaps: extend skills with procedural knowledge, and use
the LLM to make skill authoring accessible.

## Sub-phases

### 30a — Extended Skill Format (~400 lines)

Add `actions.json` and `plans.json` to the skill directory structure.

**Extend SkillManifest**:
```rust
pub struct SkillManifest {
    // ... existing fields ...
    pub actions_file: Option<String>,    // Path to actions.json (default: "actions.json")
    pub plans_file: Option<String>,      // Path to plans.json (default: "plans.json")
    pub procedures_file: Option<String>, // Path to procedures.md (source of truth)
}
```

**ActionSchema** (reuse existing `src/agent/causal.rs` types):
```rust
// actions.json is deserialized into Vec<SkillAction>
pub struct SkillAction {
    pub action: String,              // Action name (becomes a SymbolId)
    pub description: String,         // Human-readable purpose
    pub preconditions: Vec<String>,  // Fluent names that must hold
    pub effects: SkillEffects,
    pub estimated_duration: Option<String>, // "fast", "medium", "slow"
}

pub struct SkillEffects {
    pub initiates: Vec<String>,      // Fluents this action starts
    pub terminates: Vec<String>,     // Fluents this action ends
}
```

**PlanTemplate** (maps to HTN decomposition):
```rust
// plans.json is deserialized into Vec<SkillPlan>
pub struct SkillPlan {
    pub goal: String,                // Goal name (becomes a SymbolId)
    pub method: PlanMethod,          // How steps are organized
    pub steps: Vec<String>,          // Action names (references to actions.json)
    pub preconditions: Vec<String>,  // When this plan is applicable
    pub alternatives: Vec<String>,   // Other plan names that achieve the same goal
}

pub enum PlanMethod {
    Sequential,                      // Steps must execute in order
    Parallel,                        // Steps can execute concurrently
    Conditional(Vec<ConditionalStep>), // Steps depend on runtime conditions
}

pub struct ConditionalStep {
    pub condition: String,           // Fluent to check
    pub then_step: String,           // Action if true
    pub else_step: Option<String>,   // Action if false
}
```

**Modify**: `src/skills/mod.rs`, `src/skills/manager.rs`

### 30b — Activation Wiring (~500 lines)

Wire skill activation to the procedural reasoning systems.

**On skill activate (Hot)**:
```rust
// In SkillManager::activate() — after triples + rules:

// 1. Load action schemas → CausalManager
if let Some(actions_path) = manifest.actions_file {
    let actions: Vec<SkillAction> = load_json(&skill_dir.join(actions_path))?;
    for action in &actions {
        let schema = action.to_causal_schema(engine)?; // resolve labels to SymbolIds
        engine.causal_manager().register_action_schema(schema)?;
    }
}

// 2. Load plan templates → HTN decomposer + MCTS seed
if let Some(plans_path) = manifest.plans_file {
    let plans: Vec<SkillPlan> = load_json(&skill_dir.join(plans_path))?;
    for plan in &plans {
        let htn_rule = plan.to_htn_rule(engine)?;
        engine.htn_decomposer().register_rule(htn_rule)?;

        // Seed MCTS with known-good plans for warm-start
        let mcts_seed = plan.to_mcts_seed(engine)?;
        engine.mcts_planner().add_seed(mcts_seed)?;
    }
}
```

**On skill deactivate (Warm)**:
- Deregister action schemas from CausalManager (by skill_id tag)
- Remove HTN rules from decomposer (by skill_id tag)
- Remove MCTS seeds (by skill_id tag)

**Provenance**: All skill-originated actions/plans get
`DerivationKind::SkillProcedure { skill_id, source_file }` provenance.

**Modify**: `src/skills/manager.rs`, `src/agent/causal.rs`, `src/agent/ooda.rs`

### 30c — LLM-Assisted Skill Authoring (~800 lines)

**New module**: `src/skills/author.rs`

The authoring pipeline takes natural language and produces a complete skill:

```rust
pub struct SkillAuthor {
    llm: Arc<CandleBackend>,     // Phase 26b
    grammar: Arc<ConcreteGrammar>,
    engine: Arc<Engine>,
}

pub struct AuthoredSkill {
    pub manifest: SkillManifest,
    pub triples: Vec<LabelTriple>,
    pub actions: Vec<SkillAction>,
    pub plans: Vec<SkillPlan>,
    pub rules: Vec<String>,          // egg rule strings
    pub source_text: String,         // Original natural language
}

impl SkillAuthor {
    /// Generate a complete skill from natural language description
    pub fn author_from_description(
        &self,
        name: &str,
        description: &str,
        domains: &[String],
    ) -> SkillResult<AuthoredSkill>;

    /// Re-generate structured files from updated procedures.md
    pub fn regenerate_from_procedures(
        &self,
        skill_id: &str,
        procedures_text: &str,
    ) -> SkillResult<AuthoredSkill>;
}
```

**LLM prompt strategy** (structured extraction):

The LLM receives a system prompt with:
1. The skill format specification (JSON schemas for actions/plans)
2. Examples of well-formed skills
3. The user's natural language description

Output is constrained to produce valid JSON sections via the LogitsProcessor
(Phase 26b) or brace-depth tracking.

**Multi-pass extraction**:
1. **Concept pass**: Extract domain entities and relationships → triples
2. **Action pass**: Extract steps with preconditions/effects → actions
3. **Plan pass**: Extract goal decomposition and ordering → plans
4. **Rule pass**: Extract heuristics and inference patterns → rules
5. **Validation pass**: Check internal consistency (all action refs in plans
   exist, all preconditions reference known fluents)

Each pass uses a focused prompt to avoid overwhelming the small LLM.
For Qwen2.5-1.5B, this means 5 short inference calls rather than 1 long one.

**NLU integration**:
- New dialogue act: `SkillTeach { name, description }`
- Grammar patterns: "teach me to...", "learn how to...", "I want to show you how..."
- ChatProcessor routes to SkillAuthor when detected

### 30d — Rule Refinement Loop (~600 lines)

**New module**: `src/skills/refine.rs`

When the agent executes a skill's procedures and encounters problems:

```rust
pub struct SkillRefiner {
    llm: Arc<CandleBackend>,
    engine: Arc<Engine>,
}

pub struct RefinementSuggestion {
    pub skill_id: String,
    pub kind: RefinementKind,
    pub description: String,         // Human-readable explanation
    pub confidence: f32,
    pub source_failure: ProvenanceId, // What provoked this suggestion
}

pub enum RefinementKind {
    NewRule { lhs: String, rhs: String },
    ModifiedPrecondition { action: String, add: Vec<String>, remove: Vec<String> },
    AlternativePlan { goal: String, new_method: SkillPlan },
    NewAction { action: SkillAction },
    TripleMissing { triple: LabelTriple },
}
```

**Failure analysis pipeline**:
1. Agent executes plan from skill, action fails (precondition unmet or
   unexpected outcome)
2. Engine records failure with full provenance chain
3. Refiner queries: "What state was expected? What state was actual?
   What inference path was taken?"
4. LLM receives the failure context + skill's procedures.md
5. LLM generates RefinementSuggestions
6. Suggestions stored as pending (KG triples with `skill:pending_refinement`
   predicate)
7. User reviews via MCP/CLI/dashboard: approve, reject, or modify

**Idle task integration**:
- New daemon idle task: `refine_skills` (runs every 30 min)
- Scans recent failures involving skill procedures
- Batches failure contexts for efficient LLM processing
- Accumulates suggestions without blocking

### 30e — Skill Quality Metrics (~300 lines)

**New module**: `src/skills/quality.rs`

Track per-skill effectiveness:

```rust
pub struct SkillQuality {
    pub skill_id: String,
    pub success_rate: f32,           // Plans completed / plans attempted
    pub failure_modes: Vec<FailureMode>,
    pub rule_hit_rates: HashMap<String, f32>, // Rule name → % of inferences it participated in
    pub triple_coverage: f32,        // % of triples ever queried
    pub last_used: Option<u64>,      // Timestamp
    pub total_uses: u32,
    pub refinements_applied: u32,
    pub refinements_pending: u32,
}

pub struct FailureMode {
    pub action: String,
    pub failure_type: String,        // "precondition_unmet", "unexpected_effect", "timeout"
    pub count: u32,
    pub last_occurrence: u64,
}
```

**Storage**: Quality metrics stored as KG triples with `skill:quality_*`
predicates. Updated after each plan execution involving the skill.

**Staleness detection**: If a skill hasn't been used in N days and has low
coverage, suggest deactivation to free memory budget.

**MCP tools**: `skill_quality(skill_id)` returns the metrics.
**Dashboard**: Quality panel in Phase 29b skill browser.

## Estimated Effort

| Sub-phase | Lines | Complexity | Dependencies |
|---|---|---|---|
| 30a — Extended format | ~400 | Low | None (schema only) |
| 30b — Activation wiring | ~500 | Medium | Phase 15a, 11, 16 |
| 30c — LLM authoring | ~800 | High | Phase 26b (Candle) |
| 30d — Rule refinement | ~600 | High | Phase 26b, 30b |
| 30e — Quality metrics | ~300 | Low | 30b |
| **Total** | **~2,600** | | |

## New Dependencies

None — uses existing Candle (Phase 26b), serde for JSON, and existing engine
subsystems (CausalManager, HTN, MCTS).

## Feature Flags

```toml
[features]
skill-author = ["candle-backend"]  # LLM-assisted authoring (30c, 30d)
# 30a, 30b, 30e have no additional feature requirements
```

## Example: Teaching a Skill via Natural Language

```
User: "I want to teach you how to triage a bug report"

Agent: "I'll create a skill for that. Please describe the procedure."

User: "First categorize the bug: crash, incorrect behavior, or cosmetic.
       Then assess severity based on impact — crashes are always critical.
       For incorrect behavior, check if there's a workaround.
       If no workaround, mark as high. If workaround exists, mark as medium.
       Cosmetic issues are always low priority.
       Finally, assign to the right team based on the component."

Agent generates skill:
  skills/bug-triage/
  ├── skill.json          { name: "Bug Triage", domains: ["software", "qa"] }
  ├── triples.json        [ "crash is-a bug_category", "severity:critical is-a severity", ... ]
  ├── actions.json        [ { action: "categorize_bug", ... }, { action: "assess_severity", ... } ]
  ├── plans.json          [ { goal: "triage_bug", method: "sequential", steps: [...] } ]
  ├── rules.txt           (is_crash ?bug) => (severity ?bug critical)
  │                        (no_workaround ?bug) (incorrect_behavior ?bug) => (severity ?bug high)
  └── procedures.md       (original user text)

Agent: "I've created the 'Bug Triage' skill with 8 domain concepts,
        3 action schemas, 1 plan template, and 4 inference rules.
        Want me to activate it?"
```

## Relationship to Other Phases

- **Phase 26b** (Candle): Provides the LLM for authoring and refinement
- **Phase 27** (Burn training): Could train the LLM to be better at skill extraction
  over time (the skills it generates become training data)
- **Phase 28** (`n akh` router): Routes queries that match a skill's domain to the
  appropriate reasoning path
- **Phase 29b** (skill exposure): The user-facing controls for browsing, activating,
  and managing skills
- **Phase 31** (skill intelligence): Automatic skill discovery, dependency resolution,
  and ecosystem management
