# ADR 041 — Skill Intelligence: Self-Organizing Skill Ecosystem

> Date: 2026-03-26
> Status: Proposed
> Phase: 31
> Depends on: Phase 30 (procedural skills), Phase 26d (neural bridge),
>             Phase 28 (n akh router)
> Enhances: Phase 27 (training data from skills), Phase 29b (skill exposure)

## Context

Phase 30 makes individual skills richer (procedural knowledge) and easier to
create (LLM-assisted authoring). Phase 31 addresses the ecosystem level:
how skills relate to each other, how they're discovered, how the engine
self-organizes its skill inventory.

Currently:
- Skills are manually created and manually activated
- No dependency relationships between skills
- No way to discover that existing reasoning patterns should be packaged as skills
- No sharing mechanism beyond copying directories
- No awareness of which skills are relevant to a given query
- No cross-skill inference beyond shared KG triples

## Decision

### 1. Skill discovery from reasoning

The agent already discovers facts during background learning (spreading
activation, sleep/dream phase, continuous learning). When it discovers a
cluster of related facts + inference patterns in a domain, it should
recognize this as a **latent skill** and offer to package it.

**Trigger**: During background learning, if the consolidation phase identifies
N+ related triples (connected subgraph) with consistent provenance and high
usage, flag as skill candidate.

**Process**:
1. Extract the connected subgraph of triples
2. Identify inference rules that fired within this subgraph
3. Identify action schemas that were used in this domain
4. LLM generates a skill name + description from the content
5. Store as `SkillCandidate` in KG (pending user approval)
6. User reviews and promotes to full skill

### 2. Skill dependency graph

Extend `SkillManifest` with:
```rust
pub depends: Vec<String>,           // Skill IDs that must be active first
pub conflicts: Vec<String>,         // Skill IDs that cannot be co-active
pub provides: Vec<String>,          // Capability tags this skill offers
pub requires: Vec<String>,          // Capability tags this skill needs
```

**Activation**: Topological sort on dependency graph. Auto-activate
prerequisites. Reject activation if conflicts are detected.

**Capability matching**: `provides`/`requires` enable loose coupling —
a skill says "I need spatial_reasoning" without naming a specific provider.
Multiple skills can provide the same capability (MoE selection).

### 3. Neural skill embeddings

Using Phase 26d's NeuralVsaBridge, create a **prototype HyperVec per skill**:

- Bundle all of a skill's triple vectors into a single skill prototype
- Store in ItemMemory with a well-known `skill:prototype:{skill_id}` symbol
- Enable: "Which skill is most relevant to this query?" via similarity search
- Enable: "Which skills overlap?" via pairwise similarity
- Enable: "Which skill should I activate for this domain?" via n akh routing

This is cheap — one extra HyperVec per skill, computed at activation time.

### 4. Skill-aware routing (n akh integration)

Extend Phase 28's `n akh` router with skill awareness:

- Before routing a query, check similarity against skill prototypes
- If a relevant skill is Cold/Warm, auto-activate it (lazy loading)
- If multiple skills match, activate the one with highest quality score (Phase 30e)
- After query, if skill was auto-activated and isn't used again within
  cooldown period, auto-deactivate (memory conservation)

This transforms skills from static, manually-managed knowledge packs into
**demand-loaded expertise modules**.

### 5. Dynamic skill synthesis

When the agent identifies a knowledge gap (directed curiosity, Phase 11j)
and can't find an existing skill:

1. Agent recognizes gap: "I don't know how to reason about X"
2. Agent triggers bootstrap-style ingestion for domain X (web search, API)
3. Ingested knowledge is consolidated
4. LLM-assisted authoring (Phase 30c) packages the result as a new skill
5. New skill is activated and its quality tracked

This closes the autonomy loop: the agent teaches itself by creating skills.

### 6. Skill sharing protocol

Skills are portable directories. A sharing protocol enables:

**Export**: `akh skill export {name}` → produces a `.akhskill` archive
(tar.gz of the skill directory + integrity checksum)

**Import**: `akh skill import {path}` → unpacks, validates manifest,
installs + activates

**Registry** (future): A simple HTTP index where skills can be discovered
by domain tag, quality score, and compatibility version. Not a marketplace —
a FLOSS package registry like crates.io for skills.

**Recipe integration**: Community recipes (Phase 14i) already reference
skillpacks by name. The sharing protocol provides the transport layer.

### 7. Skill-to-skill inference

Currently, triples from different skills coexist in the shared KG, enabling
implicit cross-skill inference via spreading activation. Make this explicit:

**Bridge rules**: When two skills are co-active and share overlapping concepts
(detected via neural prototype similarity > threshold), generate bridge rules
that connect their ontologies.

Example: "astronomy" skill has "star is-a celestial_body", "mythology" skill
has "star is-a symbol". Bridge rule: `(is_a ?x star) => (ambiguous ?x star)`
to flag the polysemy.

**Cross-skill provenance**: When an inference chain crosses skill boundaries,
provenance records both source skills. Enables: "This conclusion combines
knowledge from astronomy and physics skills."

### 8. Skill as curriculum

Extend skills with a learning progression:

```rust
pub struct SkillCurriculum {
    pub concepts: Vec<CurriculumConcept>,
    pub prerequisite_order: Vec<Vec<String>>,  // Topological layers
    pub bloom_levels: HashMap<String, BloomLevel>, // Concept → target mastery
    pub assessment_queries: Vec<String>,        // Questions to test understanding
}
```

This connects to the bootstrap pipeline (Phase 14): when awakening in a new
domain, the skill's curriculum provides the learning path.

## Consequences

- SkillManifest extended with dependency/capability fields
- SkillManager gains auto-activation, dependency resolution, conflict detection
- New module: `src/skills/discover.rs` — latent skill detection
- New module: `src/skills/synthesis.rs` — dynamic skill creation from gap
- New module: `src/skills/share.rs` — export/import protocol
- New module: `src/skills/bridge.rs` — cross-skill inference rules
- Neural skill prototypes computed at activation (uses Phase 26d bridge)
- n akh router extended with skill-aware dispatch (uses Phase 28)
- Background idle task: skill candidate detection + quality monitoring

## References

- Phase 30: Procedural skills (format + authoring)
- Phase 26d: Neural→VSA bridge (skill embeddings)
- Phase 28: n akh semantic router (skill-aware routing)
- Phase 11j: Directed curiosity (knowledge gap detection)
- Phase 14i: Community recipes (skill sharing)
- Phase 29b: Skill exposure (user-facing controls)
