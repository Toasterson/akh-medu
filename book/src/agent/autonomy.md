# Autonomous Background Learning

The agent can learn, reflect, and improve continuously without user
prompting. Two mechanisms provide autonomy at different scales:

- **TUI idle learning**: Lightweight tasks run during interactive idle time.
- **Daemon mode**: A long-running background process for headless operation.

Both use the same underlying agent operations (consolidation, reflection,
equivalence learning, rule inference) -- the difference is how and when
they are scheduled.

## How It Works

```
                     User-Triggered           Background
                     ─────────────           ──────────
OODA cycles          agent run / repl        daemon idle cycles
Reflection           agent reflect           scheduled (3-5 min)
Consolidation        agent consolidate       on WM pressure check
Equivalence learn    equivalences learn      scheduled (5-10 min)
Rule inference       agent infer             scheduled (10-15 min)
Schema discovery     agent schema            scheduled (30 min)
Gap analysis         agent gaps              scheduled (15 min)
Session persist      automatic on exit       scheduled (60s)
```

Without background learning, every operation requires explicit user
commands. With it, the agent's knowledge base improves over time --
discovering new equivalences, deriving triples from inference rules,
consolidating memories, and adjusting goal priorities -- all while idle.

## TUI Idle Learning

When using the TUI (`akh-medu agent chat` or `akh-medu agent repl`), the
event loop polls for keyboard input every 100ms. When no key is pressed,
the idle scheduler checks if any background task is overdue and runs
exactly one -- the most overdue task -- to avoid blocking the interface.

### Default Intervals

| Task | Interval | What It Does |
|------|----------|--------------|
| Memory consolidation | 2 min | Checks WM pressure; if high, persists important entries to episodic memory |
| Reflection | 5 min | Reviews tool effectiveness and goal progress; auto-applies safe priority adjustments |
| Equivalence learning | 10 min | Discovers cross-lingual and structural equivalences from KG + VSA similarity |
| Rule inference | 15 min | Runs forward-chaining rules to derive new triples |

### Behavior

- Only one task runs per idle tick (at most every 100ms).
- Tasks that produce no change emit a short status message but don't
  clutter the screen.
- The scheduler resets all timers on startup -- nothing fires immediately.
- Consolidation only runs when working memory pressure exceeds the
  configured threshold (default: 80% of capacity).

### Custom Intervals

The `IdleScheduler` accepts custom intervals via `IdleScheduler::with_intervals()`:

```rust
let scheduler = IdleScheduler::with_intervals(
    Duration::from_secs(300),   // equivalence: 5 min
    Duration::from_secs(120),   // reflection: 2 min
    Duration::from_secs(60),    // consolidation: 1 min
    Duration::from_secs(600),   // rules: 10 min
);
```

### TUI Status Messages

When a background task completes, the TUI displays an `[idle:*]` message:

```
[idle:consolidation] consolidated 12 entries, 3 episodes created
[idle:reflection] 2 insights, 1 adjustments applied
[idle:equivalence] 5 new equivalences discovered
[idle:rules] 8 new triples from 3 iterations
[idle:consolidation] WM pressure low (23/100), skipped
```

## Daemon Mode

For headless, long-running operation, the daemon provides a dedicated
tokio event loop that schedules all background tasks at configurable
intervals. This is useful for:

- Overnight knowledge base enrichment after bulk ingest
- Continuous learning in production deployments
- Running the full autonomous reasoning pipeline without a terminal

### Requirements

The daemon requires the `daemon` feature (or `server`, which implies it):

```bash
cargo build --features daemon
```

### Usage

```bash
# Start with defaults
akh-medu agent daemon

# Custom intervals
akh-medu agent daemon --equiv-interval 600 --reflect-interval 300

# Fresh start with cycle limit
akh-medu agent daemon --fresh --max-cycles 100

# Tune persistence frequency
akh-medu agent daemon --persist-interval 120
```

### Default Intervals

| Task | Interval | Description |
|------|----------|-------------|
| Equivalence learning | 5 min | Cross-lingual and VSA-based equivalence discovery |
| Reflection | 3 min | Strategy review with auto-applied priority adjustments |
| Consolidation check | 1 min | Persists episodic memories when WM pressure is high |
| Schema discovery | 30 min | Discovers type patterns and relation hierarchies from KG structure |
| Rule inference | 10 min | Forward-chaining derivation of new triples |
| Gap analysis | 15 min | Identifies knowledge gaps around active goals |
| Session persist | 60s | Serializes full agent state to durable store |
| Idle OODA cycle | 30s | Runs one OODA cycle if active goals exist |

### Lifecycle

```
  start
    │
    ▼
 ┌────────────────────────────────────────────┐
 │  tokio::select! on 8 interval timers       │
 │  + ctrl_c signal                           │
 │                                            │
 │  Each tick runs the corresponding sync     │
 │  agent method and emits a status message   │
 │  to the configured MessageSink.            │
 └─────────────┬──────────────────────────────┘
               │
        Ctrl+C or max_cycles
               │
               ▼
         persist session
               │
               ▼
            exit
```

The daemon:
1. Resumes a persisted session if one exists (unless `--fresh`).
2. Enters the tokio select loop.
3. On each timer tick, runs the corresponding synchronous agent method.
4. Emits `[daemon:*]` messages via the agent's `MessageSink`.
5. On Ctrl+C (or `max_cycles` reached), persists the session and exits.

### Error Handling

Background task failures are non-fatal. Errors are logged via `tracing::warn!`
and the daemon continues operating. This means a transient failure in one
subsystem (e.g., a SPARQL serialization error) does not bring down the
entire process.

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-cycles <N>` | Maximum OODA cycles before shutdown (0 = unlimited) | `0` |
| `--fresh` | Ignore persisted session, start clean | off |
| `--equiv-interval <SECS>` | Equivalence learning interval | `300` |
| `--reflect-interval <SECS>` | Reflection interval | `180` |
| `--rules-interval <SECS>` | Rule inference interval | `600` |
| `--persist-interval <SECS>` | Session persist interval | `60` |

### DaemonConfig

```rust
DaemonConfig {
    equivalence_interval: Duration::from_secs(300),
    reflection_interval: Duration::from_secs(180),
    consolidation_interval: Duration::from_secs(60),
    schema_discovery_interval: Duration::from_secs(1800),
    rule_inference_interval: Duration::from_secs(600),
    gap_analysis_interval: Duration::from_secs(900),
    persist_interval: Duration::from_secs(60),
    idle_cycle_interval: Duration::from_secs(30),
    max_cycles: 0,
}
```

## Comparing the Two Modes

| Aspect | TUI Idle | Daemon |
|--------|----------|--------|
| Runtime | No extra dependency | Requires `tokio` (`daemon` feature) |
| Scheduling | Time-based, one task per 100ms tick | tokio intervals, parallel timers |
| Tasks | 4 (consolidation, reflection, equivalence, rules) | 8 (adds schema, gaps, OODA cycles, persist) |
| User interaction | Full TUI available | Headless, message sink only |
| Best for | Interactive sessions | Overnight / CI / production |
| Concurrency | Single-threaded, interleaved with TUI | Single-threaded async (agent is sync) |

## What Runs Automatically

### Equivalence Learning

Discovers cross-lingual and structural equivalences by:
1. Analyzing KG structure for entities that share predicate neighborhoods.
2. Comparing VSA hypervectors for high-similarity entity pairs.
3. Finding library-context matches from ingested documents.

New equivalences are persisted to the durable store and available to the
entity resolver for all future parsing and preprocessing.

### Reflection

Reviews [tool effectiveness and goal progress](planning.md#reflection),
then auto-applies safe adjustments:
- **Priority boosts** for goals making progress.
- **Priority demotions** for stagnant goals.

Destructive adjustments (goal abandonment, new goal creation) require
explicit user confirmation and are not auto-applied.

### Consolidation

When working memory pressure exceeds the configured threshold:
1. Entries are scored by relevance, reference count, and recency.
2. High-value entries are persisted as [episodic memories](ooda-loop.md#episodic-memory).
3. Low-value entries are evicted to free capacity.

### Rule Inference

Runs the [forward-chaining rule engine](../concepts/reasoning.md) with
built-in ontological rules. Derived triples are committed to the KG with
provenance records tracking the specific rule that produced them.

### Schema Discovery (daemon only)

Analyzes the KG for recurring structural patterns: entity types,
predicate co-occurrence clusters, and relation hierarchies. Results
inform future inference and help identify ontological structure.

### Gap Analysis (daemon only)

For each active goal, identifies knowledge gaps: entities with few
connections, predicates used only once, and missing symmetric relations.
Gaps are reported as `[daemon:gaps]` messages.

### Idle OODA Cycles (daemon only)

If active goals exist, the daemon runs one [OODA cycle](ooda-loop.md)
per interval. This allows goals to make progress without user input --
the agent selects tools, executes them, evaluates progress, and
potentially completes goals entirely on its own.

## Session Continuity

Both idle learning and daemon mode integrate with the existing
[session persistence](ooda-loop.md#session-persistence) system. The
agent's full state -- working memory, cycle count, goals, plans, psyche,
and learned equivalences -- is serialized to the durable store and
restored on resume. This means:

- A daemon session can be stopped with Ctrl+C and resumed later.
- TUI idle learning accumulates across multiple interactive sessions.
- Knowledge discovered in daemon mode is available in subsequent TUI sessions.

```bash
# Start daemon, let it run overnight
akh-medu agent daemon --max-cycles 500

# Next morning, resume in TUI
akh-medu agent chat
# All daemon-discovered knowledge is available
```
