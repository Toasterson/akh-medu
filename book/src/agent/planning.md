# Planning and Reflection

The agent generates multi-step plans for each goal and periodically reflects
on its progress, adjusting priorities and strategies.

## Planning

Before the OODA loop begins work on a goal, the agent generates a `Plan` --
an ordered sequence of tool calls designed to achieve the goal's success
criteria.

### Plan Structure

```rust
Plan {
    goal_id: SymbolId,
    steps: Vec<PlanStep>,
    status: PlanStatus,       // Active, Completed, Failed, Superseded
    attempt: u32,             // Incremented on backtrack
    strategy: String,         // Summary of the approach
}

PlanStep {
    tool_name: String,
    tool_input: ToolInput,
    rationale: String,
    status: StepStatus,       // Pending, Active, Completed, Failed, Skipped
    index: usize,
}
```

### Strategy Selection

The planner analyzes the goal description and success criteria using
VSA-based semantic analysis. It measures interference between the goal
text and five strategy patterns:

| Strategy | Keywords | When Selected |
|----------|----------|---------------|
| **Knowledge** | find, query, search, discover, explore, list, identify | Goal is about finding information |
| **Reasoning** | reason, infer, deduce, classify, analyze, why | Goal requires logical derivation |
| **Creation** | create, add, build, connect, link, store, write | Goal is about constructing knowledge |
| **External** | file, http, command, shell, fetch, download | Goal involves external data |
| **Similarity** | similar, like, related, compare, cluster | Goal is about finding relationships |

### Alternating Strategies

To avoid getting stuck in local optima, the planner alternates between
two meta-strategies based on the attempt number:

- **Even attempts (0, 2, 4...)**: Explore-first -- knowledge gathering
  tools, then reasoning tools.
- **Odd attempts (1, 3, 5...)**: Reason-first -- reasoning tools, then
  knowledge tools.

### Backtracking

When a plan step fails:

1. The plan is marked as `PlanStatus::Failed`.
2. A new plan is generated with `attempt += 1`.
3. The alternating strategy ensures a different approach.
4. After `max_backtrack_attempts` (default: 3), the goal fails.

### CLI

```bash
# Generate and display a plan for the current goal
akh-medu agent plan

# In the REPL
> p
> plan
```

## Reflection

After every N cycles (configurable), the agent pauses to reflect on its
performance. Reflection also runs automatically on a periodic schedule
during [background learning](autonomy.md) (every 5 minutes in TUI idle
mode, every 3 minutes in daemon mode).

### What Reflection Examines

1. **Goal progress**: Which goals advanced since the last reflection?
2. **Stagnation**: Which goals have been stuck for many cycles?
3. **Tool effectiveness**: Which tools produced useful results?
4. **Memory pressure**: Is working memory getting full?

### Adjustments

Reflection produces a list of `Adjustment` actions:

| Adjustment | Trigger | Effect |
|------------|---------|--------|
| Boost priority | Goal made progress | Priority increased to keep momentum |
| Demote priority | Goal stagnant | Priority decreased to try other goals |
| Suggest decomposition | Goal stalled beyond threshold | Recommends splitting the goal |
| Trigger consolidation | Memory pressure high | Saves episodic memories, frees WM |

### Psyche Evolution

If a [psyche](psyche.md) is configured, reflection also triggers
`Psyche::evolve()`:

- Archetype weights adjust based on tool success rates.
- Shadow encounter counter grows individuation level.
- Dominant archetype is recalculated.

### Configuration

```rust
ReflectionConfig {
    interval: usize,            // Reflect every N cycles (default: 5)
    min_cycles_worked: usize,   // Minimum work before reflecting
}
```

### CLI

```bash
# Trigger reflection manually
akh-medu agent reflect

# In the REPL
> r
> reflect
```

## Plan-Reflection Interaction

Plans and reflection work together:

1. A plan is generated for a goal.
2. The OODA loop executes plan steps cycle-by-cycle.
3. If a step fails, backtracking generates an alternative plan.
4. During periodic reflection, the agent assesses whether the current
   plan strategy is working.
5. Reflection may boost or demote the goal, triggering plan regeneration
   on the next cycle.
