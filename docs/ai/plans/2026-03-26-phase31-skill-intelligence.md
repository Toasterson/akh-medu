# Phase 31 — Skill Intelligence

> Date: 2026-03-26
> Status: Planned
> Phase: 31
> Depends on: Phase 30 (procedural skills), Phase 26d (neural bridge),
>             Phase 28 (n akh router)
> ADR: [041-skill-intelligence](../decisions/041-skill-intelligence.md)

## Motivation

Phase 30 makes skills richer and easier to create. Phase 31 makes the skill
ecosystem self-organizing: skills discover themselves, load on demand, resolve
dependencies, share across instances, and bridge across domains.

The end state: the agent is a learning system that acquires, organizes, and
deploys domain expertise autonomously — asking the user only for approval at
key decision points.

## Sub-phases

### 31a — Skill Dependency Graph (~400 lines)

**Extend SkillManifest**:
```rust
pub depends: Vec<String>,        // Must be active before this skill
pub conflicts: Vec<String>,      // Cannot be co-active
pub provides: Vec<String>,       // Capability tags offered
pub requires: Vec<String>,       // Capability tags needed
```

**Activation logic**:
1. Build dependency DAG (petgraph, reuse existing graph infrastructure)
2. Topological sort → activation order
3. Auto-activate prerequisites (recursive)
4. Reject if circular dependency or unresolvable conflict
5. Capability matching: find providers for `requires` tags

**Deactivation**: Reverse topological order. Warn if dependents are still active.

### 31b — Neural Skill Embeddings (~300 lines)

Compute a prototype HyperVec per skill at activation time:

```rust
fn compute_skill_prototype(engine: &Engine, skill_id: &str) -> AkhResult<HyperVec> {
    let triples = engine.triples_of_skill(skill_id)?;
    let vectors: Vec<HyperVec> = triples.iter()
        .flat_map(|t| [
            engine.item_memory().get_or_create(engine.ops(), t.subject),
            engine.item_memory().get_or_create(engine.ops(), t.predicate),
            engine.item_memory().get_or_create(engine.ops(), t.object),
        ])
        .collect();
    Ok(engine.ops().bundle(&vectors))
}
```

Store prototype in ItemMemory with symbol `skill:proto:{skill_id}`.

**Use cases**:
- Query→skill matching via similarity search
- Skill overlap detection (pairwise similarity matrix)
- n akh routing target selection

### 31c — Skill-Aware Routing (~300 lines)

Extend Phase 28's `n akh` router:

```rust
// In NakhRouter::route():
// After feature extraction, before bandit selection:

let query_vec = encode_label(ops, query_text);
let skill_matches = item_memory.search_prefix("skill:proto:", &query_vec, k=3)?;

for matched_skill in skill_matches {
    if matched_skill.similarity > ACTIVATION_THRESHOLD {
        let skill_id = matched_skill.symbol.strip_prefix("skill:proto:");
        if skill_manager.state(skill_id) != SkillState::Hot {
            skill_manager.activate(skill_id)?;  // Lazy loading
        }
    }
}
```

**Auto-deactivation**: Track last-used timestamp per skill. Idle task checks
for skills not used within cooldown period → deactivate to free memory.

### 31d — Skill Discovery from Reasoning (~500 lines)

**New module**: `src/skills/discover.rs`

During background learning (sleep/consolidation), detect latent skills:

```rust
pub struct SkillCandidate {
    pub suggested_name: String,
    pub domain_tags: Vec<String>,
    pub core_triples: Vec<Triple>,          // Connected subgraph
    pub related_rules: Vec<String>,         // Rules that fired within this subgraph
    pub related_actions: Vec<String>,       // Actions used in this domain
    pub discovery_provenance: ProvenanceId,
    pub confidence: f32,
}
```

**Detection algorithm**:
1. During consolidation, identify clusters of frequently co-activated symbols
   (spreading activation patterns over last N cycles)
2. For each cluster, extract the induced subgraph from KG
3. Filter: must have ≥ threshold triples, ≥ 2 distinct predicates,
   coherent domain (prototype similarity within cluster > 0.6)
4. Check: is this already covered by an existing skill? (compare prototype
   against all skill prototypes)
5. If novel: LLM generates name + description, store as SkillCandidate
6. Present to user for approval

**Idle task**: `discover_skills` runs during dream phase (Phase 11i),
piggybacks on the consolidation cycle.

### 31e — Dynamic Skill Synthesis (~500 lines)

**New module**: `src/skills/synthesis.rs`

When the agent identifies a knowledge gap it cannot fill from existing skills:

```rust
pub async fn synthesize_skill(
    engine: &Engine,
    gap: &KnowledgeGap,    // From directed curiosity (Phase 11j)
    author: &SkillAuthor,   // From Phase 30c
) -> SkillResult<SkillCandidate> {
    // 1. Use bootstrap ingestion to gather knowledge about the gap domain
    let sources = discover_resources(engine, &gap.domain, &gap.concepts)?;
    let ingested = ingest_resources(engine, &sources)?;

    // 2. LLM-assisted authoring from ingested content
    let skill = author.author_from_triples(
        &gap.suggested_name,
        &ingested.triples,
        &gap.domain_tags,
    )?;

    // 3. Return as candidate (user approval before activation)
    Ok(SkillCandidate::from_authored(skill, gap.provenance_id))
}
```

**Integration with directed curiosity** (Phase 11j):
- Curiosity identifies information gaps
- If gap maps to a potential skill domain, trigger synthesis
- Synthesis uses existing bootstrap pipeline (web search, API, ingestion)
- Result is packaged as a skill via Phase 30c authoring

### 31f — Skill Sharing Protocol (~400 lines)

**New module**: `src/skills/share.rs`

**Export**:
```rust
pub fn export_skill(skill_dir: &Path, output: &Path) -> SkillResult<PathBuf> {
    // 1. Validate skill integrity (manifest + all referenced files exist)
    // 2. Compute content checksum (SHA-256)
    // 3. Create .akhskill archive (tar.gz)
    // 4. Include checksum in archive metadata
}
```

**Import**:
```rust
pub fn import_skill(archive: &Path, skills_dir: &Path) -> SkillResult<String> {
    // 1. Verify checksum
    // 2. Extract to temporary directory
    // 3. Validate manifest schema
    // 4. Check for conflicts with existing skills
    // 5. Move to skills_dir/{skill_id}/
    // 6. Return skill_id for activation
}
```

**CLI**: `akh skill export {name} -o path.akhskill`
        `akh skill import path.akhskill`

**Recipe integration**: Recipes can reference remote skill URLs:
```toml
[resources]
skillpacks = ["local:astronomy", "https://skills.example.com/physics-v2.akhskill"]
```

### 31g — Cross-Skill Inference (~400 lines)

**New module**: `src/skills/bridge.rs`

When multiple skills are co-active, detect and handle concept overlap:

**Overlap detection** (at activation time):
1. Compute pairwise similarity of skill prototypes
2. If similarity > OVERLAP_THRESHOLD (e.g., 0.55): flag as overlapping
3. Find shared symbols (symbols present in both skills' triple sets)

**Bridge rule generation**:
- For shared symbols with different predicate contexts: generate disambiguation
  rules (e.g., "star" in astronomy vs. "star" in mythology)
- For complementary contexts: generate integration rules
  (e.g., physics velocity concept bridges to sports velocity concept)
- LLM assists: given two skill descriptions + shared concepts, generate
  appropriate bridge rules
- Bridge rules tagged with `DerivationKind::SkillBridge { source_skills }`

**Cross-skill provenance**: When an inference chain traverses triples from
multiple skills, provenance records all contributing skills.

### 31h — Skill Curriculum (~300 lines)

Extend skills with a learning progression for bootstrapping:

```rust
pub struct SkillCurriculum {
    pub concepts: Vec<CurriculumConcept>,
    pub layers: Vec<Vec<String>>,        // Topological layers (learn in order)
    pub assessments: Vec<Assessment>,
}

pub struct CurriculumConcept {
    pub name: String,
    pub bloom_target: BloomLevel,        // Remember, Understand, Apply, etc.
    pub prerequisites: Vec<String>,
}

pub struct Assessment {
    pub query: String,                   // Question to test understanding
    pub expected_concepts: Vec<String>,  // Concepts the answer should reference
    pub bloom_level: BloomLevel,
}
```

**Integration with bootstrap** (Phase 14):
- When awakening with a skill-backed domain, use the skill's curriculum
  to order concept acquisition
- Assessment queries used by competence assessment (Phase 14, `competence.rs`)
  to measure progress

**LLM-assisted curriculum generation** (during Phase 30c authoring):
- When authoring a skill, LLM also generates a curriculum
- Curriculum stored as `curriculum.json` in skill directory

## Estimated Effort

| Sub-phase | Lines | Complexity | Dependencies |
|---|---|---|---|
| 31a — Dependency graph | ~400 | Medium | None (schema) |
| 31b — Neural embeddings | ~300 | Low | Phase 26d |
| 31c — Skill-aware routing | ~300 | Medium | Phase 28, 31b |
| 31d — Discovery from reasoning | ~500 | High | Phase 30c, 31b |
| 31e — Dynamic synthesis | ~500 | High | Phase 30c, 11j |
| 31f — Sharing protocol | ~400 | Low | None |
| 31g — Cross-skill inference | ~400 | High | Phase 26b, 31b |
| 31h — Skill curriculum | ~300 | Medium | Phase 14 |
| **Total** | **~3,100** | | |

## Priority Order

1. **31a — Dependencies** (foundational, enables safe multi-skill activation)
2. **31b — Neural embeddings** (enables 31c, 31d, 31g)
3. **31c — Skill-aware routing** (biggest UX impact: automatic expertise loading)
4. **31f — Sharing protocol** (enables community ecosystem)
5. **31d — Discovery** (autonomy: agent finds its own skills)
6. **31e — Synthesis** (autonomy: agent creates skills from gaps)
7. **31g — Cross-skill inference** (quality: multi-domain reasoning)
8. **31h — Curriculum** (bootstrap enhancement)
