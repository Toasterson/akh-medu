# Jungian Psyche

The psyche system models the agent's personality, ethical constraints, and
behavioral tendencies using Carl Jung's analytical psychology. It replaces
rigid rule-based ethics (Asimov's laws) with a dynamic, evolvable system
that learns from experience.

The psyche is a special `Core` compartment that influences four aspects of
the OODA loop:

1. **Tool selection** -- archetype weights bias which tools the agent prefers.
2. **Action gating** -- shadow veto patterns block dangerous actions.
3. **Reflection** -- the psyche evolves based on the agent's performance.
4. **Output style** -- the persona's grammar preference controls narrative synthesis.

## Structure

```
Psyche
  ├── Persona           <- outward mask (communication style)
  ├── Shadow            <- constrained anti-patterns (vetoes + biases)
  ├── ArchetypeWeights  <- behavioral tendencies (tool selection bias)
  └── SelfIntegration   <- growth tracking (individuation)
```

## Persona

The Persona controls how the agent communicates. It is the "mask" the agent
presents to the user.

```toml
[persona]
name = "Scholar"
grammar_preference = "narrative"        # or "formal", "terse", or a custom TOML path
traits = ["precise", "curious", "thorough"]
tone = ["clear", "methodical"]
```

When the agent synthesizes output (`Agent::synthesize_findings()`), it checks
the persona's `grammar_preference` and uses it as the default grammar
archetype. The grammar framework then structures the output accordingly:

| Grammar | Style |
|---------|-------|
| `"narrative"` | Flowing prose with topic sentences and connecting phrases. |
| `"formal"` | Structured sections with headers and bullet points. |
| `"terse"` | Minimal output, facts only, no elaboration. |
| `"path/to/custom.toml"` | User-defined grammar rules. |

## Shadow

The Shadow represents the agent's ethical constraints -- things it must not
do (vetoes) and things it should avoid (biases).

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
explanation = "Detected repetitive pattern - try a different approach."
```

### Two severity levels

| Level | Mechanism | Effect |
|-------|-----------|--------|
| **Veto patterns** | `check_shadow_veto(action_desc)` | Hard block. The action is not executed. A `ShadowVeto` provenance record is created and the shadow encounter counter increments (driving individuation). |
| **Bias patterns** | `check_shadow_bias(action_desc)` | Soft penalty. The cumulative severity from matched patterns is logged but the action proceeds. |

**Trigger matching**: Each trigger string is checked as a case-insensitive
substring against the action description (`"tool={name} input={:?}"`).
Multiple triggers in a pattern are OR'd -- any match fires the pattern.

**When a veto fires** (during the Act phase):

1. The action is blocked -- it never reaches the tool registry.
2. A `DerivationKind::ShadowVeto` provenance record is stored.
3. `psyche.record_shadow_encounter()` increments the shadow encounter counter.
4. The cycle returns a `GoalProgress::Failed` with the veto reason.
5. The agent adapts on the next cycle by choosing a different tool.

## Archetypes

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

The bonus is added to the tool's total score alongside `base_score`,
`recency_penalty`, `novelty_bonus`, `episodic_bonus`, and `pressure_bonus`.
The effect is subtle per-cycle but cumulative over time.

## Self-Integration (Individuation)

The Self is Jung's integrative center. In the agent, it tracks psychological
growth through experience.

```toml
[self_integration]
individuation_level = 0.1     # [0.0, 1.0] - how integrated the psyche is
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

Encountering and acknowledging the shadow (having actions vetoed, having
goals abandoned) drives psychological growth -- mirroring Jung's concept
that integrating the shadow is the path to individuation.

## Psyche Evolution

Evolution happens automatically during the agent's periodic reflection cycle.
When `reflect()` is called, it passes the psyche to `Psyche::evolve()`, which:

1. **Rebalances archetypes** based on tool effectiveness:
   - Consistent success (>70% rate, 2+ uses): archetype weight +0.02
   - Flagged ineffective (<30% rate, 2+ uses): archetype weight -0.02
   - Weights clamped to [0.1, 0.95]
2. **Acknowledges shadow encounters**: Incrementing counter on goal abandonment.
3. **Grows individuation**: `individuation_level += 0.01 * min(shadow_encounters, 5)`
4. **Updates dominant archetype**: Recomputed from current weights.

## OODA Loop Integration

| Phase | Psyche Influence |
|-------|-----------------|
| **Observe** | None |
| **Orient** | None |
| **Decide** | Archetype weights added as scoring bonus to tool candidates |
| **Act** | Shadow veto/bias check before tool execution |
| **Reflect** | `Psyche::evolve()` adjusts weights and grows individuation |

The score breakdown is visible in the decision reasoning:
```
[score=0.85: base=0.80 recency=-0.00 novelty=+0.15 episodic=+0.00 pressure=+0.00 archetype=+0.030]
```

## Persistence

The psyche is persisted in two ways:

- **Session persistence**: `Agent::persist_session()` serializes the psyche to
  the durable store (redb). On `Agent::resume()`, the persisted psyche is
  restored, preserving evolved weights and individuation across sessions.
- **Compartment file**: `data/compartments/psyche/psyche.toml` is the initial
  default, read when the psyche compartment is first loaded.

Priority order on resume: persisted session > compartment manager > default.

## Configuration

### Changing the persona

```toml
[persona]
name = "Explorer"
grammar_preference = "terse"
traits = ["adventurous", "bold", "direct"]
tone = ["energetic", "concise"]
```

### Adding shadow constraints

```toml
[[shadow.veto_patterns]]
name = "no_external_network"
triggers = ["http_fetch", "curl", "wget"]
severity = 1.0
explanation = "This agent is configured for offline-only operation."

[[shadow.bias_patterns]]
name = "prefer_internal_reasoning"
triggers = ["shell_exec", "file_io"]
severity = 0.2
explanation = "Prefer KG-based reasoning over external tool calls."
```

### Tuning archetype weights

```toml
# More exploratory
[archetypes]
healer = 0.3
sage = 0.4
guardian = 0.3
explorer = 0.9

# More cautious
[archetypes]
healer = 0.6
sage = 0.5
guardian = 0.9
explorer = 0.2
```

### Programmatic manipulation

```rust
use akh_medu::compartment::Psyche;

let mut agent = Agent::new(engine, config)?;

let mut psyche = Psyche::default();
psyche.persona.name = "Mentor".into();
psyche.persona.grammar_preference = "formal".into();
psyche.archetypes.healer = 0.8;
agent.set_psyche(psyche);

if let Some(p) = agent.psyche() {
    println!("Dominant archetype: {}", p.dominant_archetype());
    println!("Individuation: {:.2}", p.self_integration.individuation_level);
}
```

## Design Rationale

**Why Jung over Asimov?** Asimov's three laws are rigid boolean constraints
that don't adapt. Jung's model provides a spectrum:
- The Shadow has both hard vetoes and soft biases, configurable per deployment.
- Archetypes create behavioral tendencies that evolve through experience.
- Individuation means the agent's personality matures over time.

**Why is the effect subtle?** The archetype bonus formula
`(weight - 0.5) * 0.15` produces a maximum swing of ~0.07 per tool. The
psyche should nudge behavior, not override situation-specific signals. Over
many cycles, the cumulative effect creates a recognizable personality.
