# ADR 040 — Procedural Skills: From "What Is" to "How To Do"

> Date: 2026-03-26
> Status: Proposed
> Phase: 30
> Depends on: Phase 26b (Candle LLM backend), Phase 15a (causal action schemas),
>             Phase 11 (HTN decomposition), Phase 16 (MCTS planning)
> Enhances: Phase 29b (skill exposure), Phase 27 (live training)

## Context

Akh-medu skills currently carry two types of knowledge:

1. **Declarative** — triples in `triples.json` ("Pluto is-a DwarfPlanet")
2. **Inferential** — egg rewrite rules in `rules.txt` (`(causes ?x ?y) => (enables ?x ?y)`)

This is powerful but incomplete. A human "skill" also includes **procedural
knowledge** — knowing *how to do things*: "To debug Rust: first read the error,
then trace the stack, then check the types..."

Meanwhile, the engine already HAS procedural representation mechanisms that are
disconnected from the skill system:

| System | Represents | Phase | In Skills? |
|---|---|---|---|
| HTN decomposition | Goal → ordered substeps | 11 | No |
| Causal action schemas | Action + preconditions + effects | 15a | No |
| MCTS plan templates | Scored multi-step strategies | 16 | No |
| Event calculus | "Doing X initiates Y, terminates Z" | 15b | No |

Furthermore, egg rewrite rules require formal syntax expertise that most users
lack. The barrier to creating useful skills is too high.

## Decision

### 1. Extend the skill format with procedural knowledge files

Add two new optional files to the skill directory:

```
skills/{skill-id}/
├── skill.json         # Manifest (existing)
├── triples.json       # Declarative facts (existing)
├── rules.txt          # Inference rules (existing)
├── actions.json       # NEW: Causal action schemas
├── plans.json         # NEW: HTN decomposition templates
└── procedures.md      # NEW: Natural language source (for LLM re-generation)
```

**actions.json** — Action schemas with preconditions and effects:
```json
[
  {
    "action": "read_entire_file",
    "preconditions": ["file_exists", "file_accessible"],
    "effects": {
      "initiates": ["context_understood"],
      "terminates": ["context_unknown"]
    },
    "description": "Read the entire file to understand its structure"
  }
]
```

**plans.json** — HTN decomposition templates:
```json
[
  {
    "goal": "review_code",
    "method": "sequential",
    "steps": ["read_entire_file", "identify_bugs", "assess_severity",
              "suggest_fixes", "comment_style"],
    "preconditions": ["has_file_to_review"],
    "quality_gate": "all_steps_completed"
  }
]
```

**procedures.md** — Natural language source of truth:
```markdown
# Code Review Procedure

To review code:
1. First read the entire file to understand context
2. Check for common bugs (null derefs, off-by-one, resource leaks)
3. If you find a bug, note the line and severity
4. Suggest a fix with explanation
5. Only then comment on style issues

Important: severity assessment before style — never block on formatting.
```

### 2. Wire skill activation to CausalManager + HTN + MCTS

When a skill transitions to Hot:
- `triples.json` → KG (existing)
- `rules.txt` → e-graph rule store (existing)
- `actions.json` → CausalManager.register_action_schema() (new wiring)
- `plans.json` → HTN decomposition templates + MCTS seed plans (new wiring)

When deactivated:
- Action schemas deregistered from CausalManager
- HTN templates removed
- (Triples persist in KG as before — deactivate ≠ forget)

### 3. LLM-assisted skill authoring via Candle

The Candle LLM (Phase 26b) translates natural language procedures into the
structured skill format. Three authoring tiers:

**Tier 1 — Casual user** (natural language only):
```
User: "I want to teach you how to review code. First read the file..."
  → LLM extracts domain concepts → triples.json
  → LLM extracts steps + ordering → plans.json
  → LLM extracts preconditions/effects → actions.json
  → LLM generates inference heuristics → rules.txt
  → Original text preserved → procedures.md
```

**Tier 2 — Intermediate user** (edit generated files):
- LLM generates initial skill from natural language
- User edits JSON/rules to refine behavior
- Re-generation from procedures.md preserves manual edits via merge

**Tier 3 — Expert user** (write directly):
- Author egg rules, action schemas, HTN plans by hand
- Full control, no LLM involvement
- Existing workflow, preserved as-is

### 4. LLM-assisted rule refinement loop

When the agent uses a skill's procedures and fails:

1. Engine logs failure with provenance (which action failed, what precondition
   wasn't met, which rule didn't fire)
2. LLM analyzes the failure trace
3. LLM suggests:
   - New rules to handle the edge case
   - Modified action preconditions
   - Additional HTN decomposition paths (alternative methods)
4. Suggestions stored as pending updates (not auto-applied)
5. User reviews and approves via MCP/CLI/dashboard
6. Skill files updated, re-activated

This is distinct from Phase 27 (neural weight training) — here we're refining
**symbolic** procedural knowledge, not neural parameters.

### 5. Skill quality metrics

Track per-skill:
- **Success rate**: % of plans using this skill's procedures that completed
- **Failure modes**: which actions fail most often, which preconditions are unmet
- **Rule hit rate**: which inference rules actually fire during reasoning
- **Coverage**: what fraction of the skill's triples are ever queried
- **Staleness**: time since last successful use

Store as KG triples with `skill:quality_*` predicates. Surface in Phase 29b
dashboard.

## Rejected Alternatives

### Procedural knowledge as triples only
Attempted to encode "step 1 comes before step 2" as triples with ordering
predicates. Too verbose, hard to author, and loses the structured plan semantics
that HTN/MCTS need.

### Separate "procedure" entity type
Considered adding a new SymbolKind for procedures. Rejected because action
schemas and HTN plans are already well-defined in the engine — we should reuse
them, not reinvent.

### Auto-apply LLM rule suggestions
Too risky. LLM-generated rules can be subtly wrong. The human-in-the-loop
review step is essential for maintaining the engine's correctness guarantees.

## Consequences

- SkillManifest extended with `actions_file` and `plans_file` optional fields
- SkillManager.activate() wires to CausalManager + HTN + MCTS
- New module: `src/skills/author.rs` — LLM-assisted skill generation
- New module: `src/skills/refine.rs` — failure analysis + rule suggestion
- New module: `src/skills/quality.rs` — skill metrics tracking
- `procedures.md` becomes the user-friendly entry point for skill creation
- Phase 29b exposure gains "create skill from description" NLU intent
- Feature-gated: `skill-author` requires `candle-backend` (Phase 26b)

## References

- HTN planning: Phase 11, `src/agent/` (task decomposition)
- Causal action schemas: Phase 15a, `src/agent/causal.rs`
- MCTS planning: Phase 16, `src/agent/mcts.rs`
- Candle LLM backend: Phase 26b, ADR 037
- Skill exposure: Phase 29b, ADR 039
