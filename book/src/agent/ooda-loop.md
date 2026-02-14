# OODA Loop

The akh-medu agent operates on a continuous Observe-Orient-Decide-Act (OODA)
loop. Each cycle gathers state, builds context, selects a tool, executes it,
and evaluates progress toward goals.

## Cycle Structure

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Observe │ --> │  Orient │ --> │  Decide │ --> │   Act   │
│         │     │         │     │         │     │         │
│ Goals   │     │ KG ctx  │     │ Tool    │     │ Execute │
│ WM      │     │ Infer   │     │ select  │     │ Eval    │
│ Episodes│     │ Episodes│     │ Score   │     │ Progress│
└─────────┘     └─────────┘     └─────────┘     └─────────┘
      ^                                               │
      └───────────── next cycle ──────────────────────┘
```

## Phase Details

### Observe

Gathers the current state of the world:

- Active goals and their status (Active, Suspended, Completed, Failed)
- Working memory size and recent entries
- Recalled episodic memories relevant to current goals

The observation is recorded in working memory as a `WorkingMemoryKind::Observation`.

### Orient

Builds context from available knowledge:

- Collects adjacent KG triples for each active goal's symbols
- Runs spreading-activation inference from goal-related symbols
- Incorporates knowledge from recalled episodic entries
- Measures memory pressure (ratio of WM entries to capacity)

### Decide

Selects the best tool for the current situation using utility-based scoring:

```
total_score = base_score - recency_penalty + novelty_bonus
            + episodic_bonus + pressure_bonus + archetype_bonus
```

| Factor | Range | Purpose |
|--------|-------|---------|
| `base_score` | 0.0-1.0 | State-dependent score (e.g., kg_query scores high when KG has few triples) |
| `recency_penalty` | -0.4 to 0.0 | Penalizes recently used tools (-0.4 last cycle, -0.2 two ago, -0.1 three ago) |
| `novelty_bonus` | 0.0 or +0.15 | Rewards tools never used on this goal |
| `episodic_bonus` | 0.0 or +0.2 | Rewards tools mentioned in recalled episodes |
| `pressure_bonus` | 0.0 or +0.2 | Boosts consolidation tools when memory is nearly full |
| `archetype_bonus` | -0.07 to +0.07 | [Psyche](psyche.md) archetype weight bias |

The decision includes a score breakdown string for transparency:
```
[score=0.85: base=0.80 recency=-0.00 novelty=+0.15 episodic=+0.00 pressure=+0.00 archetype=+0.030]
```

#### VSA-Based Tool Selection

Core tools (`kg_query`, `kg_mutate`, `memory_recall`, `reason`,
`similarity_search`) are scored with hand-tuned state-dependent rules.
All remaining tools -- external, library, and advanced -- use **VSA semantic
scoring**: the goal description is encoded as a hypervector and compared
against each tool's semantic profile via cosine-like similarity.

Each tool carries a keyword array (17-25 words) that defines its semantic
profile. A **synonym expansion** pass widens these at build time using a
static lookup table (~20 root-word entries), so natural sentences like
"What did that paper say about gravity?" activate `library_search` even
when few tokens overlap with the base keywords.

Tools with lower risk use lower activation thresholds:

| Tool | Threshold | Multiplier | Danger |
|------|-----------|------------|--------|
| `library_search` | 0.50 | 0.80 | Safe |
| `content_ingest` | 0.50 | 0.75 | Cautious |
| `file_io` | 0.55 | 0.75 | Cautious/Danger |
| `http_fetch` | 0.55 | 0.75 | Cautious |
| `shell_exec` | 0.55 | 0.75 | Danger |
| `user_interact` | 0.55 | 0.75 | Safe |

The `infer_rules` and `gap_analysis` tools use adaptive scoring with floors
and context boosts instead of hard thresholds, ensuring they remain available
as fallback reasoning strategies.

### Act

Executes the chosen tool and evaluates the outcome:

1. **Shadow check**: If a [psyche](psyche.md) is configured, veto/bias
   patterns are checked before execution.
2. **Tool execution**: The tool runs against the engine, producing a `ToolOutput`.
3. **Goal evaluation**: Success criteria are checked against KG state:
   - Keywords from criteria are matched against non-metadata triples.
   - Tool output content is also checked for criteria matches.
4. **Progress assessment**: Goal is marked Completed, Failed, or Advanced.
5. **Provenance**: An `AgentDecision` provenance record is stored.

## Goal Management

Goals are the agent's driving objectives. Each goal has:

```rust
Goal {
    symbol_id: SymbolId,       // KG entity for this goal
    description: String,       // What to achieve
    success_criteria: String,  // Evaluated against KG state
    priority: u8,              // 0-255 (higher = more important)
    status: GoalStatus,        // Active, Suspended, Completed, Failed
    stall_threshold: usize,    // Cycles without progress before decomposition
}
```

### Stall Detection

The agent tracks per-goal progress:
- `cycles_worked`: Total OODA cycles spent on this goal.
- `last_progress_cycle`: Cycle number when last progress was made.

If `cycles_worked - last_progress_cycle >= stall_threshold`, the goal is
considered stalled and decomposition fires automatically.

### Goal Decomposition

Stalled goals are split into sub-goals:
- Criteria are split on commas and "and" conjunctions.
- The parent goal is suspended.
- Child goals are created with Active status and linked via
  `agent:parent_goal` / `agent:child_goal` predicates.

## Working Memory

Working memory is the agent's ephemeral scratch space:

```rust
WorkingMemoryEntry {
    id: u64,
    content: String,                   // Text representation
    symbols: Vec<SymbolId>,            // Linked entities
    kind: WorkingMemoryKind,           // Observation, Inference, Decision, Action, Reflection
    timestamp: u64,
    relevance: f32,                    // 0.0-1.0
    source_cycle: u64,
    reference_count: u64,              // Incremented when consulted
}
```

Capacity is configurable (default: 100 entries). When full, the oldest
low-relevance entries are evicted.

## Episodic Memory

High-relevance working memory entries are consolidated into episodic
memories -- persistent long-term records:

```rust
EpisodicEntry {
    timestamp: u64,
    goal: SymbolId,
    summary: String,
    learnings: Vec<SymbolId>,          // Symbols found relevant
    derivation_kind: DerivationKind,   // AgentConsolidation
}
```

Consolidation fires automatically when memory pressure exceeds 0.8, or
manually via `agent consolidate`.

## Session Persistence

The agent's full state (working memory, cycle count, goals, plans, psyche)
is serialized to the durable store on exit and restored on resume:

```bash
# Persists automatically on exit
akh-medu agent run --goals "..." --max-cycles 20

# Resume where you left off
akh-medu agent resume --max-cycles 50
```

## CLI Commands

```bash
# Single cycle
akh-medu agent cycle --goal "Find mammals"

# Multi-cycle
akh-medu agent run --goals "Discover planet properties" --max-cycles 20

# Fresh start (ignores persisted session)
akh-medu agent run --goals "..." --max-cycles 10 --fresh

# Interactive REPL
akh-medu agent repl

# Resume persisted session
akh-medu agent resume

# Trigger consolidation
akh-medu agent consolidate

# Recall episodic memories
akh-medu agent recall --query "mammals" --top-k 5
```

## Configuration

```rust
AgentConfig {
    working_memory_capacity: 100,     // Max WM entries
    consolidation: ConsolidationConfig::default(),
    max_cycles: 1000,                 // Safety limit
    auto_consolidate: true,           // Auto-fire when pressure > 0.8
    reflection: ReflectionConfig::default(),
    max_backtrack_attempts: 3,        // Plan retries before giving up
}
```
