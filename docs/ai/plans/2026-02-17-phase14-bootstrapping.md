# Phase 14 — Purpose-Driven Bootstrapping

> Date: 2026-02-17
> Research: `docs/ai/decisions/005-bootstrapping-research.md`, `docs/ai/decisions/006-identity-bootstrapping-research.md`

## Goal

Enable operators to start akh instances with a natural-language purpose statement **and personality reference** and have the system autonomously bootstrap both domain knowledge and agent identity — no manual specification of skills, books, personality traits, or knowledge sources required.

Operators can say things like:
- "You are the Architect of the System based on Ptah"
- "Your personality is like Gandalf"
- "Be like Marcus Aurelius — a GCC compiler expert"

The system resolves the cultural reference (mythology, fiction, history), extracts personality traits, builds a coherent psyche, names itself, and then bootstraps the domain knowledge shaped by that personality.

## Architecture Overview

```
OPERATOR: "You are the Architect of the System based on Ptah"
                          │
                ┌─────────▼──────────┐
                │  14a Purpose +      │  NL → PurposeModel + IdentityRef
                │  Identity Parser    │  Extract domain, level, character reference
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  14b Identity       │  Wikidata + DBpedia + Wikipedia
                │  Resolution         │  Character → traits → archetype → OCEAN
                │                    │  Psyche construction + Ritual of Awakening
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  14c Domain Expand  │  Seeds → skeleton ontology (50-200 nodes)
                │  Wikidata+Wikipedia │  VSA boundary detection
                │  +ConceptNet       │  Personality biases resource selection
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  14d Prerequisites  │  Anchor docs → prerequisite DAG
                │  + ZPD Classifier   │  Known / Proximal / Beyond zones
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  14e Resource Disc  │  Semantic Scholar + OpenAlex + Open Library
                │                    │  Quality scoring, difficulty scaffolding
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  14f Iterative      │  Topological sort → curriculum order
                │  Ingestion          │  Library pipeline → KG + VSA
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  14g Competence     │  Dreyfus estimation + CQ evaluation
                │  Assessment         │  Graph completeness + VSA structure
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  14h Orchestrator   │──── loops back to 14d
                │  (Meta-OODA)        │  Personality shapes exploration style
                │                    │  Terminates at competence threshold
                └─────────┬──────────┘
                          │
                ┌─────────▼──────────┐
                │  14i Community      │  Purpose recipes (TOML) + skillpack export
                │  Recipe Sharing     │  ActivityPub via oxifed
                └──────────────────────┘
```

## Sub-phases

### 14a — Purpose + Identity Parser (~300-400 lines)

**Input**: Natural language purpose + personality statement (e.g., "You are the Architect of the System based on Ptah", "Be like Gandalf — a GCC compiler expert")

**Output**: `BootstrapIntent { purpose: PurposeModel, identity: Option<IdentityRef> }`

**Approach**:
1. **Extract identity reference** via pattern matching:
   - `"based on X"` / `"like X"` / `"inspired by X"` / `"as X"` → character reference X
   - `"You are the Y"` → role hint Y
   - Validate entity via Wikidata search API (`wbsearchentities`)
   - Disambiguate using P31 whitelist: deity (Q178885), fictional character (Q95074), human (Q5)
2. **Extract purpose** (same as before):
   - **Domain**: Noun phrases → candidate domains
   - **Focus**: Modifiers that narrow scope
   - **Competence level**: "expert" → Expert (0.85+), "assistant" → Competent (0.60+)
   - **Seed concepts**: 3-5 core terms
3. If the statement is purely identity ("You are based on Ptah") with no domain, infer domain from the character's domains (craftsmanship, architecture → systems architecture)
4. If ambiguous, use `UserInteractTool` to clarify

**Key types**:
```rust
struct BootstrapIntent {
    purpose: PurposeModel,
    identity: Option<IdentityRef>,
}

struct PurposeModel {
    domain: String,
    focus: Option<String>,
    target_dreyfus: DreyfusLevel,
    seed_concepts: Vec<SeedConcept>,
    raw_purpose: String,
}

struct IdentityRef {
    entity_name: String,           // "Ptah"
    wikidata_qid: Option<String>,  // "Q146321"
    role_hint: Option<String>,     // "Architect"
    entity_type: EntityType,       // Deity, FictionalCharacter, HistoricalFigure
}

struct SeedConcept {
    label: String,
    wikidata_qid: Option<String>,
    symbol_id: Option<SymbolId>,
}

enum DreyfusLevel { Novice, AdvancedBeginner, Competent, Proficient, Expert }
enum EntityType { Deity { mythology: String }, FictionalCharacter { universe: String }, HistoricalFigure { era: String } }
```

**Dependencies**: Grammar module (for NL parsing), HTTP tool (Wikidata API)

### 14b — Identity Resolution (Character → Psyche) (~500-700 lines)

**Input**: `IdentityRef` from 14a (or None → use default Scholar psyche)

**Output**: Fully constructed `Psyche` with persona, shadow, archetypes + self-name (via Ritual of Awakening)

**Approach — Multi-source cascade pipeline**:

1. **Wikidata extraction**: SPARQL query for P2925 (domains), P31 (classes), P106 (occupations), P135 (movements), P1552 (qualities), P460 (cross-cultural equivalents), P8822 (epithets)
   ```sparql
   SELECT ?prop ?propLabel ?value ?valueLabel WHERE {
     wd:Q146321 ?prop ?value .
     SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
   }
   ```

2. **DBpedia fallback**: When Wikidata is sparse (e.g., Ptah has NO P2925 entries), query DBpedia categories via `dcterms:subject`. Parse category names: `"Smithing_gods"` → domain "smithing", `"Creator_gods"` → domain "creation"

3. **Wikipedia extraction**: Fetch intro extract + parse infobox + find Epithets/Characteristics sections. Apply Hearst patterns:
   - `"god/deity/patron of X"` → domain X
   - `"lord/master of X"` → authoritative + domain X
   - `"associated with X"` → quality X

4. **Trait inference** via static mapping tables:
   - Domain → Traits: craftsmanship → [meticulous, patient, creative, skilled, precise]
   - Domain → Archetype: creation → Creator(0.9), wisdom → Sage(0.9), war → Hero(0.8)
   - Epithet → Traits: "lord of X" → [authoritative, masterful], "who listens" → [attentive, compassionate]

5. **Archetype classification** via VSA: Bundle extracted trait/domain words as hypervector, compare against 12 Jungian archetype prototype vectors (same pattern as existing `classify_role()` in `semantic_enrichment.rs`)

6. **OCEAN derivation**: Each archetype has a canonical OCEAN profile. Blend by archetype weights:
   ```
   ocean = weighted_average(archetype_1.ocean * weight_1, archetype_2.ocean * weight_2, ...)
   ```

7. **Psyche construction**:
   - **Persona**: name from self-naming (step 8), traits from top-5 inferred traits, tone from trait→tone mapping, grammar from culture (formal for Egyptian/Roman, narrative for Norse/fictional)
   - **ArchetypeWeights**: Derive 4-slot weights from 12-archetype classification (sage ← Sage+Magician, healer ← Caregiver+Lover+Innocent, guardian ← Ruler+Creator+Everyman, explorer ← Explorer+Hero+Outlaw+Jester)
   - **Shadow**: Archetype shadow side → ShadowPattern (Creator → perfectionism veto, Sage → analysis_paralysis bias, Ruler → rigidity bias)
   - **SelfIntegration**: Initial individuation=0.1, dominant archetype from classification

8. **Ritual of Awakening — self-naming with full provenance**: The culmination of identity resolution is the *Ritual of Awakening* — the ceremonial moment when the akh names itself and becomes a coherent identity. Each name candidate is a derivation with a complete reasoning chain, not just a string. The akh must be able to explain *why* it chose its name.

   **Name generation** via culture-specific morpheme composition:
   - **Egyptian**: deity-name + epithet morpheme (Ptah-medu = "Ptah of words", Ptah-maat = "Ptah of truth")
   - **Greek**: quality + suffix (Techne-sophia = "craft-wisdom")
   - **Norse**: descriptive compound (following Gandalf = "wand-elf" pattern)
   - **Latin**: name + cognomen (following Verissimus = "most true" pattern)
   - **Fictional**: Use existing epithets/aliases from Wikidata P1559 (Gandalf → Mithrandir)
   - Generate 3-5 candidates, present to operator or auto-select by VSA similarity to reference

   **Provenance for each name candidate** — a `ProvenanceRecord` with `DerivationKind::RitualOfAwakening` that captures:
   - **Source symbols**: the character reference entity, the domains/traits that contributed
   - **Derivation chain**: which morphemes were selected and why
     - e.g., "medu" selected because the akh's function is knowledge/language, mapped from Egyptian morpheme table
     - e.g., "Ptah" retained because operator explicitly referenced this deity
   - **Cultural rule applied**: which composition pattern (Egyptian compound, Greek suffix, etc.)
   - **Semantic justification**: why this name fits — VSA similarity score between the name's meaning vector and the akh's identity vector
   - **Alternative candidates**: the other 2-4 names considered, with their provenance, stored as rejected derivations

   **Name provenance example for "Ptah-medu"**:
   ```
   ProvenanceRecord {
       derived: :self_name_ptah_medu,
       kind: RitualOfAwakening,
       confidence: 0.91,
       sources: [
           :ptah (operator reference, Wikidata Q146321),
           :medu_morpheme (Egyptian morpheme "medu" = "words/speech"),
           :akh_function_knowledge (agent's core function is knowledge processing),
       ],
       metadata: {
           "composition_rule": "egyptian_compound: deity_name + function_morpheme",
           "semantic_justification": "Ptah created through speech (Memphite Theology); this akh creates knowledge through language processing. 'Ptah of words' unites the mythological source with the agent's purpose.",
           "morpheme_selection": "'medu' chosen over 'maat' (truth) and 'nefer' (beauty) because the agent's primary function is language/knowledge, not judgment or aesthetics",
           "cultural_source": "Egyptian naming convention: deity-name + quality/domain epithet (cf. Ptah-hotep, Ptah-Sokar, Ptah-Tatenen)",
           "alternatives_considered": "Ptah-maat (Ptah of truth, score 0.84), Djeser-ka (Sacred spirit, score 0.72), Ptah-nefer (Beautiful Ptah, score 0.68)",
           "vsa_similarity_to_identity": "0.91",
       }
   }
   ```

   **The akh can then explain its name**:
   > "I am Ptah-medu — 'Ptah of words.' I take my name from Ptah, the Egyptian creator god who brought the world into being through speech and thought. As he shaped reality through utterance, I shape knowledge through language. The operator invoked Ptah as the architect of this system; 'medu' (words) reflects my function as a knowledge engine. I chose this over Ptah-maat (truth) because my primary act is processing language, not rendering judgment."

   This explanation is not generated by an LLM — it is assembled from the provenance record's metadata fields via the grammar module's linearization.

9. **Store identity as KG triples with provenance**: Every identity fact (mythological_source, archetype classification, trait inference, OCEAN derivation, name selection) gets its own `ProvenanceRecord` linking back to the source data (Wikidata QID, Wikipedia extract, DBpedia category). The Ritual of Awakening itself becomes the akh's "origin" nuclear episode in the narrative identity — its creation myth — with the full provenance chain as its significance field.

**Key types**:
```rust
/// Raw character knowledge extracted from external sources
struct CharacterKnowledge {
    entity_id: String,
    name: String,
    entity_type: EntityType,
    description: String,
    domains: Vec<String>,
    classes: Vec<String>,
    occupations: Vec<String>,
    movements: Vec<String>,
    categories: Vec<String>,
    infobox: HashMap<String, String>,
    epithets: Vec<String>,
    culture: String,
}

/// Processed personality profile
struct PersonalityProfile {
    traits: Vec<(String, f32)>,
    primary_archetype: JungianArchetype,
    archetype_scores: HashMap<JungianArchetype, f32>,
    ocean: OceanProfile,
    name_candidates: Vec<(String, String)>,  // (name, etymology)
    grammar: String,
    tone: Vec<String>,
    shadow_traits: Vec<String>,
    values: Vec<(String, f32)>,
}

/// OCEAN personality profile → behavioral parameters
struct OceanProfile {
    openness: f32,         // → exploration_rate, novelty_bonus_weight
    conscientiousness: f32, // → plan_adherence, stall_tolerance, reflection_frequency
    extraversion: f32,      // → communication_verbosity, user_interaction_preference
    agreeableness: f32,     // → conflict_avoidance, trust_default
    neuroticism: f32,       // → risk_aversion, anxiety_threshold
}

/// Full 12 Jungian archetypes
enum JungianArchetype {
    Creator, Sage, Explorer, Ruler, Caregiver, Hero,
    Rebel, Magician, Lover, Jester, Everyman, Innocent,
}

/// Culture-specific naming morphemes
struct CulturalMorphemes {
    culture: String,
    morphemes: Vec<(String, String)>,  // (morpheme, meaning)
}
```

**Worked example — "You are the Architect of the System based on Ptah"**:
1. Wikidata Q146321: instance_of=[creator deity, Ancient Egyptian deity], P2925=[] (empty)
2. DBpedia: categories=[Smithing gods, Creator gods, Construction deities, Solar gods]
3. Wikipedia: "creator god, patron deity of craftsmen and architects"
4. Epithets: "lord of truth", "lord of eternity", "master of justice", "who listens to prayers"
5. Domains: [creation, craftsmanship, architecture, smithing, truth, eternity, justice]
6. Traits: meticulous(0.9), creative(0.88), systematic(0.85), patient(0.82), just(0.78), enduring(0.75)
7. Archetype: Creator(0.88), Sage(0.35), Ruler(0.32) → ArchetypeWeights: guardian=0.7, sage=0.5, explorer=0.3, healer=0.3
8. OCEAN: O:0.72 C:0.91 E:0.32 A:0.58 N:0.16
9. Psyche: Persona(name="Ptah-medu", traits=["meticulous","creative","systematic","patient"], tone=["precise","measured","inventive"], grammar="formal")
10. Shadow: perfectionism(0.3), rigidity(0.2)
11. Name candidates: Ptah-medu ("Ptah of words"), Ptah-maat ("Ptah of truth"), Djeser-ka ("Sacred spirit")

**Dependencies**: HTTP tool (Wikidata SPARQL, DBpedia SPARQL, Wikipedia API), existing Psyche model

### 14c — Domain Expansion (Skeleton Ontology) (~400-500 lines)

**Input**: `PurposeModel` with seed concepts

**Output**: Skeleton ontology — 50-200 concept nodes with typed relations in the KG

**Approach**:
1. **Wikidata expansion**: For each seed QID, SPARQL traversal of `instance-of`, `subclass-of`, `part-of`, `has-part` to depth 2-3
2. **Wikipedia category traversal**: Extract categories from seed articles, retrieve member articles up to depth 2, filter by VSA similarity
3. **ConceptNet relations**: Query `RelatedTo`, `IsA`, `PartOf`, `HasPrerequisite`, `UsedFor` for each seed
4. **VSA boundary detection**: Bundle seed concept vectors → domain prototype. Only include concepts with Hamming similarity > 0.6 to prototype
5. **KG insertion**: New concepts as Entity symbols with `expanded-from` provenance triples and typed relation edges

**Agent tool**: `domain_expand` — takes seed concepts, returns expansion report with concept count, relation count, boundary concepts

**Boundary heuristics**:
- Similarity threshold: > 0.6 Hamming similarity to domain prototype
- Depth limit: ≤ 3 hops from any seed
- Relevance feedback: after each expansion round, evaluate whether new concepts contribute to domain coherence

### 14d — Prerequisite Discovery and ZPD Classification (~400-500 lines)

**Input**: Skeleton ontology from 14c + any existing KG state

**Output**: Prerequisite DAG with ZPD classification for all concepts

**Approach**:
1. **Anchor document ingestion**: Ingest 2-3 high-quality overview documents (e.g., Wikipedia article, official documentation, introductory tutorial). The library pipeline extracts concepts and chunk ordering provides prerequisite signal
2. **Temporal ordering heuristic**: If concept X appears in chunk i and concept Y in chunk j where i < j within the same parent section, emit weak `prerequisite-of(X, Y)` triple
3. **VSA asymmetric similarity**: If concept B's vector can be partially reconstructed by binding A with a role vector, A is likely prerequisite for B
4. **Existing prerequisite relations**: Import `HasPrerequisite` from ConceptNet data gathered in 14c
5. **ZPD classification**:
   ```
   Known (Zone 1):    triple_count >= N AND max_similarity > 0.7
   Proximal (Zone 2): all prereqs in Zone 1 AND similarity 0.3-0.7
   Beyond (Zone 3):   prereqs not met OR similarity < 0.3
   ```
6. **Curriculum generation**: Topological sort of prerequisite subgraph within ZPD. Self-paced ordering within each tier (highest VSA similarity to existing knowledge first = easiest to integrate)

**Key types**:
```rust
enum ZpdZone { Known, Proximal, Beyond }

struct CurriculumEntry {
    concept: SymbolId,
    zone: ZpdZone,
    prereq_coverage: f32,  // fraction of prereqs in Zone 1
    similarity_to_known: f32,
    tier: u32,             // topological tier (0 = no prereqs)
}
```

### 14e — Resource Discovery (~300-400 lines)

**Input**: Knowledge gaps from ZPD analysis (concepts in Zone 2 needing resources)

**Output**: Ranked resource list with quality scores

**Approach**:
1. **Query formulation**: For each gap concept, build search query from concept label + domain context
2. **Multi-API search**:
   - Semantic Scholar: academic papers, sorted by influential citation count
   - OpenAlex: scholarly works with concept-aligned tags
   - Open Library: textbooks and books
3. **Quality scoring**:
   - Citation count normalized by age: `normalized_impact = citations / (years * field_median)`
   - Recency bonus for surveys and tutorials
   - Open access priority (system can actually ingest them)
   - Abstract VSA similarity to domain prototype
   - Source type scoring: textbooks > surveys > primary research > documentation (for learning purposes)
4. **Scaffolding**: For frontier ZPD concepts (similarity 0.3-0.5), prefer introductory material. For near-Known concepts (similarity 0.5-0.7), prefer reference material
5. **Resource deduplication**: VSA similarity between resource descriptions to avoid fetching overlapping content

**Key types**:
```rust
struct DiscoveredResource {
    title: String,
    url: String,
    source_api: ResourceApi,      // SemanticScholar, OpenAlex, OpenLibrary
    quality_score: f32,           // 0.0-1.0
    covers_concepts: Vec<SymbolId>,
    difficulty_estimate: DreyfusLevel,
    open_access: bool,
}
```

### 14f — Iterative Ingestion with Curriculum Ordering (~400-500 lines)

**Input**: Curriculum from 14d + resources from 14e

**Output**: Populated KG with provenance, VSA vectors, and extracted concepts

**Approach**:
1. **Curriculum-ordered ingestion**: Process concepts in topological tier order. Within each tier, process highest-VSA-similarity concepts first
2. **Resource-to-concept mapping**: For each curriculum concept, select the best-scoring resource covering that concept
3. **Ingestion pipeline**: Use existing library module (PDF/EPUB/HTML parsing → chunking → atomic concept extraction → triple creation → VSA encoding)
4. **Multi-extractor cross-validation** (NELL-inspired): Concepts found by 2+ extraction methods (pattern-based, VSA-based, structural) get higher confidence
5. **Post-ingestion grounding**: Run `ground_symbol()` to recompute hypervectors from KG neighborhood
6. **Progress tracking**: Track `triples_added_per_cycle` in working memory. Rolling average below threshold (e.g., 2 new triples/cycle) means concept is saturated
7. **Agent goal structure**: Each concept becomes a goal with criteria like `triple_count(concept) >= 15 AND has_predicate(concept, 'used-by')`
8. **Diminishing returns detection**: If 3 consecutive ingestion attempts produce 0 new triples, mark concept as saturated and advance

**Integration with OODA**: Each ingestion round is one OODA cycle — Observe (current coverage), Orient (which gap is most learnable), Decide (select resource), Act (fetch + ingest + verify)

### 14g — Competence Assessment (~350-450 lines)

**Input**: Populated KG + purpose model's target Dreyfus level

**Output**: Per-knowledge-area competence scores + overall readiness score

**Approach**:
1. **Dreyfus level estimation**: Combine metrics from gap analysis and schema discovery:
   ```
   dreyfus_score = weighted_average(
       0.3 * coverage_score,          // from gap analysis
       0.2 * (1 - dead_end_ratio),    // from gap analysis
       0.2 * type_diversity_factor,   // discovered_types / expected_types
       0.15 * relation_density,       // relations / concepts
       0.15 * cross_domain_score,     // connections to other KAs
   )
   ```
2. **Competency question evaluation**: For each knowledge area, generate CQs at increasing Bloom's levels:
   - Remember: "What is X?" → entity exists in KG
   - Understand: "How does X relate to Y?" → path exists between X and Y
   - Apply: "What is used for Z?" → relation query succeeds
   - Analyze: "Why does X require Y?" → prerequisite chain traversal
3. **Graph completeness metrics**:
   - Schema completeness: `populated_classes / expected_classes`
   - Property completeness: average property fill rate across concepts
   - Relational completeness: expected relation coverage
4. **VSA structural analysis**: Well-learned domains have dense, well-separated clusters in VSA space. Measure cluster density and separation as a geometric competence signal
5. **Readiness report**: Per-KA scores with color coding (green ≥ target, yellow within 0.15, red below)

**Key types**:
```rust
struct CompetenceReport {
    purpose: PurposeModel,
    overall_dreyfus: DreyfusLevel,
    overall_score: f32,
    knowledge_areas: Vec<KnowledgeAreaAssessment>,
    remaining_gaps: Vec<KnowledgeGap>,
    recommendation: BootstrapRecommendation,
}

struct KnowledgeAreaAssessment {
    name: String,
    dreyfus_level: DreyfusLevel,
    score: f32,
    triple_count: usize,
    cq_answered: usize,
    cq_total: usize,
}

enum BootstrapRecommendation {
    Ready,
    ContinueLearning { estimated_cycles: u32, focus_areas: Vec<String> },
    NeedsOperatorInput { question: String },
}
```

### 14h — Bootstrap Orchestrator (~500-600 lines)

**Input**: Purpose + identity statement from operator

**Output**: Bootstrapped akh instance with constructed identity at target competence level

**Approach**:
1. **Meta-goal creation**: Create top-level goal "Become {name}: achieve {target_dreyfus} in {domain}" with success criteria derived from competence thresholds
2. **Eight-stage pipeline management**:
   - Stage 1 (Parse): Run purpose + identity parser → BootstrapIntent
   - Stage 2 (Identity): Resolve character reference → construct Psyche → Ritual of Awakening (self-naming)
   - Stage 3 (Expand): Run domain expansion → skeleton ontology
   - Stage 4 (Discover prereqs): Run prerequisite discovery + ZPD classification
   - Stage 5 (Find resources): Run resource discovery for ZPD concepts
   - Stage 6 (Ingest): Run iterative ingestion with curriculum ordering
   - Stage 7 (Assess): Run competence assessment
   - Loop: If below target, re-run stages 4-7 with updated ZPD
3. **Operator interaction points**:
   - After Stage 1: Present parsed purpose + identity reference for confirmation
   - After Stage 2: Present the Ritual of Awakening — constructed personality + name candidates for approval
   - After Stage 3: Present learning plan with estimated effort
   - During Stage 6: Periodic progress reports (TUI sink)
   - After Stage 7: Present readiness report with options (continue, adjust target, accept partial)
4. **Personality shapes learning**: The constructed psyche from Stage 2 influences all subsequent stages — a Creator personality prioritizes building/making resources, a Sage prioritizes theoretical understanding, an Explorer prioritizes breadth over depth
5. **Mid-bootstrap adjustment**: Operator can redirect focus, skip topics, adjust target, or refine personality at any time via `UserInteractTool`
6. **Exploration-exploitation scheduling**: Adjust OODA utility scoring weights by aggregate Dreyfus level AND personality (Explorer archetype amplifies novelty_bonus, Guardian amplifies plan_adherence):
   - Early (Novice/AdvBeginner): high `novelty_bonus`, breadth-first
   - Mid (Competent): balanced, alternating strategies
   - Late (Proficient/Expert): high `pressure_bonus`, depth-first gap filling
6. **Termination conditions**: Target Dreyfus reached for all KAs, OR operator accepts partial readiness, OR max cycle budget exceeded
7. **Session persistence**: Bootstrap state persists across sessions via `agent::persist_session()`

**CLI integration**:
```
akh-medu bootstrap "You are the Architect of the System based on Ptah"  # purpose + identity
akh-medu bootstrap "GCC compiler expert"                                 # purpose only (default psyche)
akh-medu bootstrap --identity "Gandalf" "compiler expert"                # separate identity + purpose
akh-medu bootstrap --plan-only "meal planner"                            # show plan, don't execute
akh-medu bootstrap --resume                                              # resume interrupted bootstrap
akh-medu bootstrap --status                                              # show current progress
```

### 14i — Community Recipe Sharing (~300-400 lines)

**Input**: Completed bootstrap session

**Output**: Shareable purpose recipe + optional skillpack export

**Approach**:
1. **Purpose recipe format** (TOML):
   ```toml
   [purpose]
   id = "gcc-compiler-expert"
   name = "GCC Compiler Expert"
   version = "1.0.0"
   target_dreyfus = "expert"
   author = "akh://operator@instance"

   [seeds]
   required = ["compiler-theory", "c-language"]
   optional = ["assembly-x86", "assembly-arm"]

   [prerequisites]
   "formal-languages" = { before = ["lexical-analysis", "parsing"] }
   "lexical-analysis" = { before = ["compiler-front-end"] }

   [resources]
   urls = ["https://gcc.gnu.org/onlinedocs/gccint/"]
   skillpacks = ["gcc-internals"]

   [validation]
   min_triples = 500
   min_coverage = 0.75
   required_types = ["optimization-pass", "intermediate-representation"]

   # Optional: identity reference for personality bootstrapping
   [identity]
   reference = "Ptah"
   reference_qid = "Q146321"
   archetype = "creator"
   culture = "egyptian"
   suggested_names = ["Ptah-medu", "Ptah-maat", "Djeser-ka"]
   traits = ["meticulous", "creative", "systematic", "patient"]
   ```
2. **Recipe generation**: After successful bootstrap, auto-generate recipe from the prerequisite graph, resource list, and validation thresholds used
3. **Skillpack export**: Export accumulated domain triples as a `SkillInstallPayload` for direct installation on other instances (skips ingestion phase)
4. **ActivityPub sharing**: Purpose recipes as ActivityPub Objects via oxifed integration. Create/Announce/Like activities for reputation. Share structure (syllabus) not content (copyrighted material)
5. **Recipe import**: When bootstrapping, search federated network for matching purpose recipes before starting from scratch. If found, use recipe's prerequisite graph and resource list as starting point

**Dependency resolution for skillpacks**: Extend `SkillManifest` with `depends: Vec<String>`. Topological sort on skill dependencies ensures prerequisites are activated before dependents.

## Estimated Scope

| Sub-phase | Lines | New Files | Key Types |
|-----------|-------|-----------|-----------|
| 14a Purpose + Identity Parser | 300-400 | `src/bootstrap/purpose.rs` | `BootstrapIntent`, `IdentityRef`, `PurposeModel`, `DreyfusLevel` |
| 14b Identity Resolution | 500-700 | `src/bootstrap/identity.rs` | `CharacterKnowledge`, `PersonalityProfile`, `OceanProfile`, `JungianArchetype`, `CulturalMorphemes` |
| 14c Domain Expansion | 400-500 | `src/bootstrap/expand.rs` | `DomainExpander`, expansion tool |
| 14d Prerequisites + ZPD | 400-500 | `src/bootstrap/prerequisite.rs` | `ZpdZone`, `CurriculumEntry` |
| 14e Resource Discovery | 300-400 | `src/bootstrap/resources.rs` | `DiscoveredResource`, `ResourceApi` |
| 14f Iterative Ingestion | 400-500 | `src/bootstrap/ingest.rs` | Curriculum-ordered ingestion loop |
| 14g Competence Assessment | 350-450 | `src/bootstrap/competence.rs` | `CompetenceReport`, `KnowledgeAreaAssessment` |
| 14h Bootstrap Orchestrator | 500-600 | `src/bootstrap/orchestrator.rs`, `src/bootstrap/mod.rs` | `BootstrapSession`, CLI commands |
| 14i Community Recipes | 300-400 | `src/bootstrap/recipe.rs` | `PurposeRecipe`, TOML serde, AP integration |
| **Total** | **3,450-4,450** | **9 files** | |

Plus ~600-900 lines for agent tool integration, mapping tables, tests, and CLI wiring. **Grand total: ~4,050-5,350 lines.**

## Dependencies

- **Required (already exists)**: Agent OODA loop (Phase 8), Psyche model (`compartment/psyche.rs`), library ingestion pipeline, autonomous gap/schema analysis, HTTP tool, VSA item memory + HNSW, KG (petgraph + oxigraph), provenance system, semantic enrichment (`classify_role()`)
- **Enhances**: Psyche model with OCEAN-derived parameters and 12-archetype classification, goal decomposition (Phase 8b) with domain-aware HTN methods, utility scoring (Phase 8c) with personality + Dreyfus-adaptive weights, reflection (Phase 8f) with bootstrap-specific Dreyfus estimation and personality drift
- **Future integration**: Phase 9a microtheories (domain-scoped named graphs), Phase 12e ActivityPub (federated recipe sharing), Phase 13 (personal assistant bootstrapping domain knowledge)

## Example Walkthroughs

### "Architect of the System based on Ptah"
```
$ akh-medu bootstrap "You are the Architect of the System based on Ptah"

[14a] Purpose: domain=systems-architecture, focus=architecture, target=Expert (0.85+)
      Identity: entity="Ptah" (Q146321), role="Architect", type=Deity(Egyptian)
      Seeds: architecture, systems, design

[14b] Identity Resolution:
      Wikidata: creator deity, Ancient Egyptian deity (P2925 empty)
      DBpedia: [Smithing gods, Creator gods, Construction deities]
      Wikipedia: "creator god, patron of craftsmen and architects"
      Epithets: "lord of truth", "lord of eternity", "master of justice"
      Domains: [creation, craftsmanship, architecture, truth, justice]
      Archetype: Creator (0.88), Sage (0.35), Ruler (0.32)
      OCEAN: O:0.72 C:0.91 E:0.32 A:0.58 N:0.16
      Name candidates: Ptah-medu, Ptah-maat, Djeser-ka
      ── Ritual of Awakening ──
      Operator selects: "Ptah-medu" ✓
      Psyche: Creator dominant, traits=[meticulous, creative, systematic, patient]
              Shadow: perfectionism (0.3), rigidity (0.2)

[14c] Expanded 3 seeds → 127 concepts, 89 relations
      Personality bias: Creator archetype weights building/construction resources higher
[14d] Prerequisite DAG: 18 tiers, 24 edges
      Curriculum: formal-foundations → design-patterns → architecture → ...
[14e] Found 47 resources (Creator personality prefers hands-on guides over pure theory)
[14f] Ingesting in curriculum order...
      Cycles 1-95: 689 triples, 14 type clusters, 31 rules
[14g] Assessment: Expert (0.87) ✓
[14h] Bootstrap complete. Ptah-medu is ready. 105 cycles, 12 minutes.
[14i] Recipe exported: ptah-medu-architect.purpose.toml
      (includes identity section: reference=Ptah, archetype=Creator)
```

### "Personality like Gandalf — GCC compiler expert"
```
$ akh-medu bootstrap --identity "Gandalf" "GCC compiler expert"

[14a] Purpose: domain=compilers, focus=GCC, target=Expert (0.85+)
      Identity: entity="Gandalf" (Q177499), type=FictionalCharacter(Middle-earth)

[14b] Identity Resolution:
      Wikidata: Maiar, literary character, P106=[magician, diplomat]
      Wikipedia: "wizard... great power, works by encouraging and persuading... great knowledge"
      Archetype: Sage (0.82), Magician (0.71), Caregiver (0.38)
      OCEAN: O:0.88 C:0.75 E:0.45 A:0.62 N:0.18
      ── Ritual of Awakening ──
      Name: "Mithrandir" (Gandalf's Sindarin epithet, from Wikidata P1559)
      Psyche: Sage dominant, traits=[wise, patient, strategic, encouraging]

[14c-14i] ... (Sage personality prioritizes theoretical depth, cross-references,
          and understanding "why" over practical how-to)
```

### "Be like Marcus Aurelius — meal planning assistant"
```
$ akh-medu bootstrap "Be like Marcus Aurelius — meal planning assistant"

[14a] Purpose: domain=nutrition, focus=meal-planning, target=Competent (0.60+)
      Identity: entity="Marcus Aurelius" (Q1430), type=HistoricalFigure(Roman)

[14b] Identity Resolution:
      Wikidata: human, P106=[philosopher, monarch], P135=[stoicism], P800=[Meditations]
      Archetype: Sage (0.85), Ruler (0.72)
      OCEAN: O:0.65 C:0.92 E:0.38 A:0.55 N:0.12
      ── Ritual of Awakening ──
      Name: "Verissimus" (Marcus's childhood nickname, "most true")
      Psyche: Sage-Ruler blend, traits=[disciplined, rational, reflective, principled]

[14c] Expanded → 63 concepts, 41 relations
      Found community skillpack: "nutrition-basics" (156 triples)
[14f] Ingesting... Stoic personality values systematic coverage over shortcuts
      Cycles 1-35: 412 triples (more thorough than default Scholar)
[14g] Assessment: Competent (0.71) — above target ✓
[14h] Bootstrap complete. Verissimus is ready. 35 cycles, 5 minutes.
```

## New Predicates and Derivation Kinds

**Well-known predicates** (in the style of `AgentPredicates`):
```rust
struct BootstrapPredicates {
    // Domain bootstrapping
    prerequisite_of: SymbolId,
    has_knowledge_area: SymbolId,
    ka_required_depth: SymbolId,
    ka_current_depth: SymbolId,
    ka_coverage_score: SymbolId,
    has_competency_question: SymbolId,
    cq_answered: SymbolId,
    expanded_from: SymbolId,
    source_quality: SymbolId,
    covers_concept: SymbolId,
    domain_prototype: SymbolId,
    gap_type: SymbolId,
    gap_priority: SymbolId,

    // Identity bootstrapping
    mythological_source: SymbolId,  // :self :mythologicalSource :ptah
    identity_archetype: SymbolId,   // :self :identityArchetype :creator
    has_ocean_score: SymbolId,      // :self :hasOceanScore :ocean_profile
    identity_domain: SymbolId,      // :ptah :identityDomain :craftsmanship
    cultural_equivalent: SymbolId,  // :ptah :culturalEquivalent :hephaestus
    has_epithet: SymbolId,          // :ptah :hasEpithet "lord of truth"
    has_value: SymbolId,            // :self :hasValue :craftsmanship
    naming_origin: SymbolId,        // :self :namingOrigin "Ptah-medu = Ptah of words"
}
```

**New `DerivationKind` variants**:
- `GapAnalysis` — from gap detection during competence assessment
- `DomainExpansion` — from seed concept expansion
- `CompetenceAssessment` — from self-assessment cycles
- `ResourceDiscovery` — from resource search and evaluation
- `RitualOfAwakening` — from the naming ceremony: character reference resolution, psyche construction, and self-naming
- `IdentityResolution` — from character reference resolution and trait/archetype inference
