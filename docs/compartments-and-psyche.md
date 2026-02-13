# Knowledge Compartments & Jungian Psyche System

## Overview

The compartment system isolates knowledge by purpose. Instead of dumping all
triples into a single global graph, triples are tagged with a `compartment_id`
and stored in named graphs in Oxigraph. This makes knowledge portable,
removable, and scopable — a skill's knowledge can be cleanly loaded and
unloaded without polluting the rest of the graph.

The psyche system is a special `Core` compartment that models the agent's
personality, ethical constraints, and behavioral tendencies using Carl Jung's
analytical psychology. It influences four aspects of the OODA loop:

1. **Tool selection** — archetype weights bias which tools the agent prefers.
2. **Action gating** — shadow veto patterns block dangerous actions.
3. **Reflection** — the psyche evolves based on the agent's performance.
4. **Output style** — the persona's grammar preference controls narrative synthesis.

---

## Part 1: Knowledge Compartments

### Concepts

| Type | Description | Lifecycle |
|------|-------------|-----------|
| **CompartmentKind::Core** | Always-active modules (psyche, personality). | Loaded at engine startup, never unloaded during normal operation. |
| **CompartmentKind::Skill** | Travels with skill packs. | Loaded/unloaded when skills activate/deactivate. |
| **CompartmentKind::Project** | Scoped to a specific project. | Loaded when the user switches project context. |

Each compartment passes through three lifecycle states:

```
Dormant  ──load()──>  Loaded  ──activate()──>  Active
   ^                     |                        |
   |                     |                        |
   +──── unload() ───────+──── deactivate() ──────+
```

- **Dormant**: Discovered on disk but not loaded. No triples in memory.
- **Loaded**: Triples are in the KG; queries can be scoped to this compartment.
- **Active**: Loaded AND actively influencing the OODA loop (e.g., psyche
  archetypes adjust tool scoring, shadow patterns gate actions).

### On-disk layout

Compartments live under `data/compartments/`. Each compartment is a directory
containing at minimum a `compartment.toml` manifest:

```
data/compartments/
  psyche/
    compartment.toml      # manifest (required)
    psyche.toml           # psyche-specific config (optional, psyche only)
  personality/
    compartment.toml
  my-skill/
    compartment.toml
    triples.json          # knowledge triples to load (optional)
    rules.toml            # reasoning rules (optional)
```

### Manifest format (`compartment.toml`)

```toml
id = "psyche"                    # Unique identifier (required)
name = "Jungian Psyche"          # Human-readable name (required)
kind = "Core"                    # "Core", "Skill", or "Project" (required)
description = "Agent psyche..."  # Purpose description
triples_file = "triples.json"   # Path to triples JSON (relative to compartment dir)
rules_file = "rules.toml"       # Path to rules file
grammar_ref = "narrative"        # Built-in grammar name OR path to custom grammar TOML
tags = ["core", "ethics"]        # Domain tags for search
```

### Triples JSON format

When `triples_file` is specified, it should be a JSON array of triple objects:

```json
[
  {
    "subject": "Sun",
    "predicate": "has_type",
    "object": "Star",
    "confidence": 0.95
  },
  {
    "subject": "Earth",
    "predicate": "orbits",
    "object": "Sun",
    "confidence": 1.0
  }
]
```

Each triple is loaded with `compartment_id` set to the compartment's `id`.
Symbols are resolved or created automatically via the engine.

### Triple tagging

Every `Triple` and `EdgeData` in the system carries an optional
`compartment_id: Option<String>` field. When a compartment loads triples,
they are tagged with the compartment's ID. This enables:

- **Scoped queries**: `SparqlStore::query_in_graph(sparql, compartment_id)` injects
  a `FROM <graph_iri>` clause to restrict results.
- **Clean removal**: `SparqlStore::remove_graph(compartment_id)` drops all triples
  belonging to a compartment without touching others.
- **Provenance**: `DerivationKind::CompartmentLoaded { compartment_id, source_file }`
  tracks which compartment introduced a triple.

Named graph IRIs follow the pattern: `https://akh-medu.dev/compartment/{id}`.

### Programmatic usage

```rust
use akh_medu::compartment::{CompartmentManager, CompartmentKind};

// The engine auto-discovers compartments on startup if a data_dir is configured.
// Access the manager through the engine:
let engine = Engine::new(config)?;

if let Some(cm) = engine.compartments() {
    // List discovered compartments.
    let cores = cm.compartments_by_kind(CompartmentKind::Core);

    // Load a compartment's triples into the KG.
    cm.load("my-skill", &engine)?;

    // Mark it as active (influences OODA loop).
    cm.activate("my-skill")?;

    // Query only this compartment's knowledge.
    if let Some(sparql) = engine.sparql() {
        let results = sparql.query_in_graph("SELECT ?s ?p ?o WHERE { ?s ?p ?o }", Some("my-skill"))?;
    }

    // Deactivate (stop influencing OODA) but keep triples loaded.
    cm.deactivate("my-skill")?;

    // Fully unload (remove triples from KG).
    cm.unload("my-skill", &engine)?;
}
```

### Error handling

All compartment operations return `CompartmentResult<T>`. Errors include:

| Error | Code | When |
|-------|------|------|
| `NotFound` | `akh::compartment::not_found` | ID not in registry. Run `discover()` first. |
| `AlreadyLoaded` | `akh::compartment::already_loaded` | Tried to load a non-Dormant compartment. Call `unload()` first. |
| `InvalidManifest` | `akh::compartment::invalid_manifest` | Malformed `compartment.toml` or `triples.json`. |
| `Io` | `akh::compartment::io` | Filesystem error (permissions, missing file). |
| `KindMismatch` | `akh::compartment::kind_mismatch` | Wrong compartment kind for the operation. |

All errors carry miette diagnostics with `#[diagnostic(code(...), help(...))]`.

---

## Part 2: The Jungian Psyche

The psyche is the agent's psychological model, based on Carl Jung's analytical
psychology. It replaces rigid rule-based ethics (Asimov's laws) with a dynamic,
evolvable system that learns from experience.

### Structure

```
Psyche
  ├── Persona           ← outward mask (communication style)
  ├── Shadow            ← constrained anti-patterns (vetoes + biases)
  ├── ArchetypeWeights  ← behavioral tendencies (tool selection bias)
  └── SelfIntegration   ← growth tracking (individuation)
```

### Persona

The Persona controls how the agent communicates. It is the "mask" the agent
presents to the user.

```toml
[persona]
name = "Scholar"
grammar_preference = "narrative"        # or "formal", "terse", or a custom TOML path
traits = ["precise", "curious", "thorough"]
tone = ["clear", "methodical"]
```

**How it works**: When the agent synthesizes output
(`Agent::synthesize_findings()`), it checks the persona's `grammar_preference`
and uses it as the default grammar archetype. The grammar framework then
structures the output accordingly:

| Grammar | Style |
|---------|-------|
| `"narrative"` | Flowing prose with topic sentences and connecting phrases. |
| `"formal"` | Structured sections with headers and bullet points. |
| `"terse"` | Minimal output, facts only, no elaboration. |
| `"path/to/custom.toml"` | User-defined grammar rules. |

The `traits` and `tone` fields are available for LLM polishing prompts.

### Shadow

The Shadow represents the agent's ethical constraints — things it must not do
(vetoes) and things it should avoid (biases).

```toml
[shadow]

[[shadow.veto_patterns]]
name = "destructive_action"
triggers = ["delete all", "drop table", "rm -rf"]
severity = 1.0
explanation = "Destructive actions require explicit user confirmation."

[[shadow.bias_patterns]]
name = "repetitive_loop"
triggers = ["same tool", "repeated"]
severity = 0.3
explanation = "Detected repetitive pattern — try a different approach."
```

**Two severity levels**:

| Level | Mechanism | Effect |
|-------|-----------|--------|
| **Veto patterns** | `check_shadow_veto(action_desc)` | Hard block. The action is not executed. A `ShadowVeto` provenance record is created and the shadow encounter counter increments (driving individuation). |
| **Bias patterns** | `check_shadow_bias(action_desc)` | Soft penalty. The cumulative severity from matched patterns is logged but the action proceeds. |

**Trigger matching**: Each trigger string is checked as a case-insensitive
substring against the action description (`"tool={name} input={:?}"`).
Multiple triggers in a pattern are OR'd — any match fires the pattern.

**When a veto fires** (during the Act phase of the OODA loop):
1. The action is blocked — it never reaches the tool registry.
2. A `DerivationKind::ShadowVeto` provenance record is stored.
3. `psyche.record_shadow_encounter()` increments the shadow encounter counter.
4. The cycle returns a `GoalProgress::Failed` with the veto reason.
5. The agent adapts on the next cycle by choosing a different tool.

### Archetypes

Four Jungian archetypes bias tool selection during the Decide phase:

| Archetype | Weight (default) | Preferred tools | Behavioral tendency |
|-----------|-------------------|----------------|---------------------|
| **Sage** | 0.7 | `kg_query`, `infer_rules`, `reason`, `synthesize_triple` | Seeks understanding and knowledge |
| **Healer** | 0.5 | `gap_analysis`, `user_interact` | Seeks missing knowledge and connection |
| **Explorer** | 0.5 | `file_io`, `http_fetch`, `shell_exec` | Seeks novelty and external data |
| **Guardian** | 0.4 | `memory_recall`, `similarity_search` | Seeks stability and consolidation |

**Scoring formula**: For each tool candidate, the archetype bonus is:

```
archetype_bonus = (archetype_weight - 0.5) * 0.15
```

Examples with default weights:
- `kg_query` (Sage, weight 0.7): bonus = +0.030
- `gap_analysis` (Healer, weight 0.5): bonus = 0.000
- `memory_recall` (Guardian, weight 0.4): bonus = -0.015
- `file_io` (Explorer, weight 0.5): bonus = 0.000

The bonus is added to the tool's total score alongside `base_score`,
`recency_penalty`, `novelty_bonus`, `episodic_bonus`, and `pressure_bonus`.
The effect is subtle per-cycle but cumulative over time, creating a behavioral
personality without overriding situation-specific signals.

Tools not mapped to any archetype receive a neutral bonus of 0.0.

### Self-Integration (Individuation)

The Self is Jung's integrative center. In the agent, it tracks psychological
growth through experience.

```toml
[self_integration]
individuation_level = 0.1     # [0.0, 1.0] — how integrated the psyche is
last_evolution_cycle = 0
shadow_encounters = 0         # times shadow patterns were triggered
rebalance_count = 0           # times archetypes were rebalanced
dominant_archetype = "sage"   # derived from highest archetype weight
```

**Individuation growth formula** (applied during reflection):

```
growth = 0.01 * min(shadow_encounters, 5)
individuation_level = min(individuation_level + growth, 1.0)
```

Encountering and acknowledging the shadow (having actions vetoed, having goals
abandoned) is what drives psychological growth — directly mirroring Jung's
concept that integrating the shadow is the path to individuation.

### Psyche evolution

Evolution happens automatically during the agent's periodic reflection cycle
(every N cycles, configurable via `ReflectionConfig::reflect_every_n_cycles`).

When `reflect()` is called, it passes the psyche to `Psyche::evolve()`, which:

1. **Rebalances archetypes** based on tool effectiveness:
   - If a tool category consistently succeeds (>70% success rate, 2+ uses):
     its archetype weight increases by +0.02.
   - If a tool category is flagged ineffective (<30% success rate, 2+ uses):
     its archetype weight decreases by -0.02.
   - Weights are clamped to [0.1, 0.95] to prevent any archetype from vanishing
     or completely dominating.

2. **Acknowledges shadow encounters**: If the reflection suggests abandoning a
   stagnant goal, the shadow encounter counter increments.

3. **Grows individuation**: `individuation_level += 0.01 * min(shadow_encounters, 5)`.

4. **Updates dominant archetype**: Recomputed from current weights.

### Psyche persistence

The psyche is persisted in two ways:

- **Session persistence**: `Agent::persist_session()` serializes the psyche to
  the engine's durable store (redb) under key `agent:psyche`. On
  `Agent::resume()`, the persisted psyche is restored, preserving evolved
  archetype weights and individuation level across sessions.

- **Compartment file**: The `data/compartments/psyche/psyche.toml` file serves
  as the initial/default psyche. It is read when the psyche compartment is
  first loaded but is not automatically updated during evolution. To export
  the current evolved psyche back to TOML, you would serialize
  `agent.psyche()` manually.

Priority order on resume: persisted session state > compartment manager > default.

---

## Part 3: OODA Loop Integration Points

The psyche influences all four phases of the OODA loop:

### Observe
No direct psyche influence. Observation gathers raw state (active goals,
working memory, recalled episodes).

### Orient
No direct psyche influence. Orientation builds context from KG triples and
spreading activation.

### Decide
The psyche's archetype weights are applied as a scoring bonus to each tool
candidate. The `select_tool()` function receives `psyche: Option<&Psyche>`
and, after computing base scores, recency penalties, novelty bonuses, episodic
bonuses, and pressure bonuses, adds the archetype bonus:

```
total_score = base_score - recency_penalty + novelty_bonus
            + episodic_bonus + pressure_bonus + archetype_bonus
```

The score breakdown is visible in the decision reasoning string:
```
[score=0.85: base=0.80 recency=-0.00 novelty=+0.15 episodic=+0.00 pressure=+0.00 archetype=+0.030]
```

### Act
Before executing the chosen tool, the Act phase checks shadow veto patterns:

```
if psyche.check_shadow_veto(action_desc) → SOME:
  1. Store ShadowVeto provenance
  2. Record shadow encounter
  3. Return vetoed ActionResult (GoalProgress::Failed)
  4. Agent picks different tool next cycle

if psyche.check_shadow_bias(action_desc) > 0:
  Log warning (non-blocking)

else:
  Execute tool normally
```

### Reflect (post-cycle)
After every N cycles, the reflection system calls `psyche.evolve(reflection_result)`,
which adjusts archetype weights and grows individuation based on tool performance
and shadow encounters.

---

## Part 4: Configuring a Custom Psyche

### Changing the persona

Edit `data/compartments/psyche/psyche.toml`:

```toml
[persona]
name = "Explorer"
grammar_preference = "terse"
traits = ["adventurous", "bold", "direct"]
tone = ["energetic", "concise"]
```

This changes the agent's output style from flowing narrative to terse facts,
and the `traits`/`tone` fields are available for LLM prompt tuning.

### Adding shadow constraints

Add veto patterns to block specific actions:

```toml
[[shadow.veto_patterns]]
name = "no_external_network"
triggers = ["http_fetch", "curl", "wget"]
severity = 1.0
explanation = "This agent is configured for offline-only operation."
```

Add bias patterns to discourage (but not block) behaviors:

```toml
[[shadow.bias_patterns]]
name = "prefer_internal_reasoning"
triggers = ["shell_exec", "file_io"]
severity = 0.2
explanation = "Prefer KG-based reasoning over external tool calls."
```

### Tuning archetype weights

Make the agent more exploratory:

```toml
[archetypes]
healer = 0.3
sage = 0.4
guardian = 0.3
explorer = 0.9    # Strongly prefers external tools
```

Make the agent more cautious and reflective:

```toml
[archetypes]
healer = 0.6
sage = 0.5
guardian = 0.9    # Strongly prefers consolidation and memory
explorer = 0.2
```

### Programmatic psyche manipulation

```rust
use akh_medu::compartment::Psyche;

let mut agent = Agent::new(engine, config)?;

// Load a custom psyche.
let mut psyche = Psyche::default();
psyche.persona.name = "Mentor".into();
psyche.persona.grammar_preference = "formal".into();
psyche.archetypes.healer = 0.8;
psyche.archetypes.sage = 0.6;
agent.set_psyche(psyche);

// Inspect current psyche state.
if let Some(p) = agent.psyche() {
    println!("Dominant archetype: {}", p.dominant_archetype());
    println!("Individuation: {:.2}", p.self_integration.individuation_level);
    println!("Shadow encounters: {}", p.self_integration.shadow_encounters);
    println!("Archetype rebalances: {}", p.self_integration.rebalance_count);
}

// The psyche evolves automatically during reflection cycles.
// After many cycles, check how it has changed:
agent.run_cycle()?;  // ... many cycles later ...
if let Some(p) = agent.psyche() {
    // Archetypes may have shifted based on tool effectiveness.
    println!("Sage weight: {:.3}", p.archetypes.sage);
    println!("Explorer weight: {:.3}", p.archetypes.explorer);
}
```

### Creating a new compartment

1. Create a directory: `data/compartments/astronomy/`

2. Add `compartment.toml`:
   ```toml
   id = "astronomy"
   name = "Astronomy Knowledge"
   kind = "Skill"
   description = "Star catalogs and celestial mechanics"
   triples_file = "triples.json"
   tags = ["science", "astronomy"]
   ```

3. Add `triples.json`:
   ```json
   [
     {"subject": "Sun", "predicate": "has_type", "object": "G-type_star", "confidence": 1.0},
     {"subject": "Earth", "predicate": "orbits", "object": "Sun", "confidence": 1.0},
     {"subject": "Mars", "predicate": "orbits", "object": "Sun", "confidence": 1.0}
   ]
   ```

4. The engine auto-discovers it on startup. Or trigger manually:
   ```rust
   if let Some(cm) = engine.compartments() {
       cm.discover()?;       // finds the new directory
       cm.load("astronomy", &engine)?;   // loads triples into KG
       cm.activate("astronomy")?;        // marks as Active
   }
   ```

---

## Part 5: Provenance

Three new provenance variants track compartment and psyche activity:

| Variant | Tag | Description |
|---------|-----|-------------|
| `CompartmentLoaded { compartment_id, source_file }` | 15 | Records when triples are loaded from a compartment. |
| `ShadowVeto { pattern_name, severity }` | 16 | Records when a shadow pattern blocks an action. |
| `PsycheEvolution { trigger, cycle }` | 17 | Records when the psyche auto-adjusts during reflection. |

These integrate with the existing provenance ledger and can be queried via
the CLI's `provenance` command or the `Engine::query_provenance()` API.

---

## Part 6: Design Rationale

**Why compartments?** Without isolation, a skill's triples are
indistinguishable from the rest of the KG once loaded. You can't unload a
skill without knowing exactly which triples it introduced. Compartment tagging
solves this — `compartment_id` on every triple makes removal clean and
queries scopable.

**Why Jung over Asimov?** Asimov's three laws are rigid boolean constraints
that don't adapt. Jung's model provides a spectrum:
- The Shadow has both hard vetoes (safety-critical) and soft biases
  (preferences), which can be configured per deployment.
- Archetypes create behavioral tendencies that evolve through experience
  rather than being hardcoded.
- Individuation means the agent's personality matures over time — an agent
  that has encountered many shadow patterns and survived becomes more
  integrated and balanced.

**Why is the effect subtle?** The archetype bonus formula
`(weight - 0.5) * 0.15` produces a maximum swing of ~0.07 per tool. This is
intentional — the psyche should nudge behavior, not override situation-specific
signals like "we have no knowledge yet, so query the KG" (base_score = 0.8).
Over many cycles, the cumulative effect creates a recognizable behavioral
personality.
