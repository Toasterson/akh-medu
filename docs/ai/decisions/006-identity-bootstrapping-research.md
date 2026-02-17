# ADR-006: Identity Bootstrapping Research — Personality from Cultural References, Self-Naming, and Character Resolution

- **Date**: 2026-02-17
- **Status**: Accepted
- **Context**: Deep research for enhancing Phase 14 bootstrapping with identity/personality initialization from cultural references (mythology, fiction, history). When an operator says "You are the Architect of the System based on Ptah", the system must resolve the reference, extract personality traits, build a coherent psyche, and name itself.

## Research Question

Can akh-medu autonomously resolve a cultural/mythological/fictional character reference into a structured personality profile — mapping it to the existing Jungian psyche model, deriving behavioral parameters, and generating a contextually appropriate self-name — without LLM dependency?

## Existing Infrastructure

The codebase already has a sophisticated Jungian psyche model in `src/compartment/psyche.rs`:

### Current Psyche Architecture

| Component | Fields | Integration |
|-----------|--------|-------------|
| **Persona** | name, grammar_preference, traits (Vec), tone (Vec) | Communication style |
| **Shadow** | veto_patterns, bias_patterns (capability/danger/action triggers) | Action constraints in OODA Act phase |
| **ArchetypeWeights** | healer, sage, guardian, explorer (f32 each) | Tool scoring bias in OODA Decide phase |
| **SelfIntegration** | individuation_level, shadow_encounters, rebalance_count, dominant_archetype | Psychological growth tracking |

**Current defaults**: Persona "Scholar", dominant archetype "sage", traits ["precise", "curious", "thorough"].

**Current OODA integration**:
- **Decide**: `archetype_bias(tool_name)` applies ±0.03 to tool candidate scores
- **Act**: `check_veto()` blocks dangerous actions; `check_bias()` logs soft penalties
- **Reflect**: `evolve()` rebalances archetypes based on tool effectiveness, tracks individuation

**Gap**: The psyche is initialized from a static TOML file. There is no mechanism to construct a psyche from a cultural reference. The Persona name is just "Scholar" — no self-naming from the operator's intent.

## Key Findings

### 1. Big Five / OCEAN as Behavioral Parameter Space

The Big Five personality model (Costa & McCrae, 1992) provides the most empirically validated and computationally tractable framework for mapping traits to behavioral parameters:

| OCEAN Dimension | Agent Behavioral Parameter |
|----------------|---------------------------|
| **Openness** | `exploration_rate`, `novelty_bonus_weight`, plan strategy bias (explore vs reason-first) |
| **Conscientiousness** | `plan_adherence`, `stall_tolerance`, `reflection_frequency`, goal completion drive |
| **Extraversion** | `communication_verbosity`, `user_interaction_preference`, assertion strength |
| **Agreeableness** | `conflict_avoidance`, `trust_default`, empathy weight in goal priority |
| **Neuroticism** | `risk_aversion`, `anxiety_threshold`, emotional reactivity to failures |

Each dimension maps to concrete numeric parameters that modulate existing OODA behavior. The OCEAN profile becomes an intermediate representation between the cultural reference and the behavioral output.

**Mapping to akh-medu**: The existing `ArchetypeWeights` (healer/sage/guardian/explorer) are a coarse projection of OCEAN. An OCEAN profile can derive archetype weights, persona traits, shadow patterns, and voice characteristics simultaneously.

### 2. Full 12 Jungian Archetypes

The existing system has 4 archetype slots. The full Jungian model has 12, organized into 4 orientations:

**Ego** (leave a mark): Hero, Magician, Outlaw/Rebel
**Order** (provide structure): Caregiver, Creator, Ruler
**Social** (connect to others): Lover, Jester, Everyman
**Freedom** (seek paradise): Innocent, Sage, Explorer

Each archetype has a core desire, fear, behavioral traits, and **shadow** (dark side):

| Archetype | Core Desire | Shadow | OCEAN Profile |
|-----------|------------|--------|--------------|
| **Creator** | Enduring creation | Perfectionism | O:0.7 C:0.95 E:0.3 A:0.6 N:0.15 |
| **Sage** | Truth/understanding | Analysis paralysis | O:0.9 C:0.8 E:0.3 A:0.5 N:0.2 |
| **Explorer** | Freedom/discovery | Restlessness | O:0.95 C:0.3 E:0.6 A:0.5 N:0.3 |
| **Ruler** | Control/order | Tyranny | O:0.4 C:0.9 E:0.7 A:0.4 N:0.2 |
| **Hero** | Mastery | Arrogance | O:0.6 C:0.85 E:0.7 A:0.5 N:0.15 |
| **Caregiver** | Service | Martyrdom | O:0.5 C:0.7 E:0.6 A:0.95 N:0.4 |
| **Magician** | Transformation | Manipulation | O:0.85 C:0.6 E:0.5 A:0.4 N:0.3 |
| **Outlaw** | Revolution | Destructiveness | O:0.8 C:0.2 E:0.7 A:0.2 N:0.5 |
| **Lover** | Connection | Possessiveness | O:0.6 C:0.5 E:0.8 A:0.8 N:0.5 |
| **Jester** | Joy | Irresponsibility | O:0.7 C:0.2 E:0.9 A:0.7 N:0.2 |
| **Innocent** | Safety | Naivete | O:0.3 C:0.5 E:0.5 A:0.8 N:0.6 |
| **Everyman** | Belonging | Conformity | O:0.4 C:0.6 E:0.6 A:0.7 N:0.4 |

The 4 existing weights can be derived from the 12: `sage ← Sage+Magician`, `healer ← Caregiver+Lover+Innocent`, `guardian ← Ruler+Creator+Everyman`, `explorer ← Explorer+Hero+Outlaw+Jester`.

### 3. Wikidata for Character Knowledge

Wikidata provides structured properties for mythological/fictional/historical entities:

| Property | ID | Personality Relevance |
|----------|----|-----------------------|
| domain of saint/deity | P2925 | Direct domain → trait mapping |
| instance of | P31 | Entity type classification (deity, character, human) |
| occupation | P106 | Role-based trait inference |
| movement | P135 | Philosophical alignment (stoicism, etc.) |
| has quality | P1552 | Direct quality attributes |
| said to be the same as | P460 | Cross-cultural equivalents |
| part of | P361 | Mythological context |
| epithet | P8822 | Personality-bearing titles |

**Critical finding**: P2925 (domain of deity) is excellent for Greek deities but sparse for Egyptian. Ptah (Q146321) has NO P2925 entries. Fallback sources are essential.

**SPARQL for entity resolution with disambiguation**:
```sparql
SELECT ?item ?itemLabel ?itemDescription
       (GROUP_CONCAT(DISTINCT ?classLabel; SEPARATOR="|") as ?classes)
WHERE {
  ?item rdfs:label "Ptah"@en .
  ?item wdt:P31 ?class .
  ?class rdfs:label ?classLabel . FILTER(LANG(?classLabel) = "en")
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
GROUP BY ?item ?itemLabel ?itemDescription
```

Disambiguation uses a P31 whitelist: `Q178885` (deity), `Q95074` (fictional character), `Q5` (human).

### 4. DBpedia Categories as Personality Proxy

When Wikidata properties are sparse, DBpedia categories provide rich structured data:

**Ptah categories**: Smithing gods, Solar gods, Fertility gods, Creator gods, Construction deities
**Gandalf categories**: Middle-earth wizards, Fictional characters with fire abilities, Christ figures in fiction
**Marcus Aurelius categories**: Stoic philosophers, Roman emperors, Political philosophers

Category names parse into domain keywords via pattern matching (`X_gods` → domain X, `X_philosophers` → movement X).

### 5. Wikipedia Extraction Pipeline

Wikipedia provides the richest unstructured personality data, extractable via patterns:

**API endpoints** (all free, no auth):
- Intro extract: `api.php?action=query&titles=X&prop=extracts&exintro=1`
- Infobox: `api.php?action=parse&page=X&prop=wikitext` (parse templates)
- Sections: `api.php?action=parse&page=X&prop=sections` (find Epithets, Characteristics)

**Hearst-style patterns for trait extraction**:
- `"god/deity/patron of X"` → domain X
- `"lord/master of X"` → authoritative + domain X
- `"associated with X"` → quality X
- `"is a/an Y"` → identity Y

**Ptah epithets** (from Wikipedia section):
- "begetter of the first beginning" → originator, creative
- "lord of truth" → truthful, just, principled
- "lord of eternity" → enduring, patient, timeless
- "who listens to prayers" → attentive, compassionate
- "master of ceremonies" → organized, ritualistic
- "master of justice" → just, fair, authoritative
- "who made himself to be God" → autonomous, self-creating

### 6. Domain-to-Trait and Domain-to-Archetype Mapping

Static mapping tables enable LLM-free personality inference:

**Domain → Traits**:
- craftsmanship → meticulous, patient, creative, skilled, precise
- creation → innovative, visionary, generative, autonomous
- wisdom → analytical, patient, thoughtful, knowledgeable
- war → aggressive, brave, strategic, disciplined
- stoicism → disciplined, rational, resilient, self-controlled

**Domain → Archetype (with weight)**:
- creation → Creator (0.9)
- craftsmanship → Creator (0.85)
- wisdom → Sage (0.9)
- war → Hero (0.8)
- magic → Magician (0.9)
- justice → Ruler (0.7)
- healing → Caregiver (0.85)

### 7. VSA Personality Encoding

Personality can be encoded as VSA hypervectors for similarity-based operations:

```
trait_encoding = BIND(trait_role_vector, intensity_vector)
personality_vector = BUNDLE(trait_encoding_1, trait_encoding_2, ..., trait_encoding_N)
```

This enables:
- **Personality similarity**: Hamming distance between personality vectors
- **Analogical transfer**: `transform = BIND(ptah_hv, UNBIND(deity_hv))` → `personality = BIND(transform, personality_domain_hv)`
- **Archetype classification**: Compare personality vector against archetype prototype vectors via HNSW
- **Personality blending**: `BUNDLE(personality_A, personality_B)` creates a hybrid

The existing `semantic_enrichment.rs` already has `classify_role()` with `ROLE_ARCHETYPES` — the same pattern extends to Jungian archetype classification using domain/trait word lists.

### 8. The Ritual of Awakening — Self-Naming as a Provenance-Tracked Derivation

The culmination of identity resolution is the **Ritual of Awakening** — the ceremonial moment when the akh names itself, crystallizes its psyche, and becomes a coherent identity. The akh's name is not just a label — it is a knowledge artifact with full provenance. The akh must be able to explain *why* it chose its name, tracing the reasoning chain back to the original cultural reference, the morpheme selection rationale, and the semantic justification.

**Name generation follows culture-specific morphological rules**:

**Egyptian**: deity-name + epithet morpheme
- Ptah-medu ("Ptah of words/speech")
- Ptah-maat ("Ptah of truth")
- Ptah-nefer ("Beautiful Ptah")

**Greek**: quality + suffix
- Techne-sophia ("craft-wisdom")
- Arche-logos ("principle-word")

**Norse**: descriptive compound
- Following Gandalf pattern (gandr + alfr)

**Latin/Roman**: name + cognomen
- Following Marcus Aurelius "Verissimus" pattern

**Morpheme tables** by culture enable compositional name generation. The system selects morphemes from the reference entity's domain and the agent's purpose, then combines them according to the culture's rules.

**Naming algorithm with provenance**:
1. Extract domains from character reference (e.g., craftsmanship, creation, truth)
2. Map domains to morphemes in the reference's culture
3. Combine with agent's functional role morpheme (e.g., "medu" = words/knowledge for a knowledge engine)
4. Generate 3-5 candidates, each with a `ProvenanceRecord`:
   - **Source symbols**: the character reference entity, the morphemes used, the domains that motivated selection
   - **Derivation chain**: which cultural composition rule was applied, why each morpheme was chosen over alternatives
   - **Semantic justification**: a structured explanation of how the name unites the mythological source with the agent's purpose (computed from trait overlap, not LLM-generated)
   - **VSA similarity score**: between the name's meaning vector and the akh's identity vector
5. Present candidates with their provenance summaries to operator, or auto-select highest VSA similarity
6. Store selected name with `DerivationKind::RitualOfAwakening` provenance; store rejected candidates as alternative derivations

**The akh can explain its own name** by linearizing the provenance record via the grammar module. For "Ptah-medu":
> "I am Ptah-medu — 'Ptah of words.' I take my name from Ptah, the Egyptian creator god who brought the world into being through speech and thought (Wikidata Q146321, Wikipedia: Memphite Theology). As he shaped reality through utterance, I shape knowledge through language. The operator invoked Ptah as the architect of this system; 'medu' (words) reflects my function as a knowledge engine. I chose this over Ptah-maat (truth, score 0.84) because my primary act is processing language, not rendering judgment."

This explanation is assembled from provenance metadata fields, not generated by an LLM. Every claim ("creator god", "brought the world into being through speech") traces to a specific source (Wikipedia extract, Wikidata property, DBpedia category).

**Provenance data structure**:
```rust
struct NameDerivation {
    name: String,
    etymology: String,                    // "Ptah (deity) + medu (words/speech)"
    composition_rule: String,             // "egyptian_compound"
    morpheme_sources: Vec<MorphemeSource>,
    semantic_justification: String,       // structured explanation
    vsa_similarity_to_identity: f32,      // 0.91
    alternatives_considered: Vec<(String, f32, String)>, // (name, score, reason_rejected)
    provenance_id: ProvenanceId,          // links to full ProvenanceRecord
}

struct MorphemeSource {
    morpheme: String,         // "medu"
    meaning: String,          // "words/speech"
    culture: String,          // "egyptian"
    selection_reason: String, // "agent's primary function is language/knowledge processing"
    rejected_alternatives: Vec<(String, String, String)>, // (morpheme, meaning, reason_rejected)
}
```

### 9. Narrative Identity (McAdams)

**McAdams (2001)** proposes identity as an internalized life story with nuclear episodes, imagoes, and generativity scripts.

**Mapping to akh-medu**: The agent's episodic memory already stores episodes. Promote significant episodes to "nuclear episodes" that shape identity:
- **Origin episode**: The **Ritual of Awakening** itself — the moment the akh named itself and became a coherent identity — with the full naming provenance chain as its significance field. This is the akh's creation myth.
- **High point episodes**: Successful goal completions that reinforce identity
- **Turning point episodes**: Plan failures that trigger personality evolution
- **Individuation episodes**: Shadow encounters that increase `individuation_level`

The Ritual of Awakening is special: it anchors the akh's entire self-narrative. When the akh reflects on "who am I?", the provenance chain from the Ritual provides the answer — not as a static string, but as a traceable derivation from operator intent through cultural reference to self-chosen identity.

### 10. Game NPC Personality Systems

**Dwarf Fortress**: 50 personality facets (0-100), 28 values, derived behavioral outcomes. Shows that detailed personality modeling is computationally tractable and produces emergent behavior.

**Crusader Kings**: Discrete trait system with oppositions (Brave/Craven), each trait modifies numerical stats. Provides the "opposed traits" constraint pattern.

**Stanford Generative Agents** (Park et al., 2023): Memory retrieval scoring via `recency * importance * relevance`. Reflection synthesizes memories into insights. Maps to akh-medu's existing reflection phase.

### 11. Character Card Formats (SillyTavern/Character.ai)

Community character card format (YAML/JSON) provides a practical configuration template:
```yaml
name: "Ptah"
description: "Ancient Egyptian craftsman-deity"
personality: "Methodical, creative, patient, speaks in crafting metaphors"
first_mes: "I shape thought as I once shaped the world."
tags: ["mythology", "creator-deity", "craftsman"]
```

For akh-medu, this maps to: `name` → Persona.name, `personality` → Persona.traits, `tags` → KG triples, `first_mes` → VoiceProfile calibration.

## Decision

Integrate identity bootstrapping into Phase 14 by adding two new sub-phases and enhancing the existing purpose parser:

### Enhanced 14a — Purpose + Identity Parsing

The purpose parser now also extracts personality references from the operator's statement. "You are the Architect of the System based on Ptah" yields both:
- **Purpose**: domain=systems-architecture, focus=architecture, target=Expert
- **Identity reference**: entity="Ptah", role_hint="Architect"

### New 14a' — Identity Resolution (Character Reference → Psyche)

Multi-source cascade pipeline:

```
Input: character reference ("Ptah") + role hint ("Architect")
         │
    ┌────▼─────────────────┐
    │ 1. Wikidata Resolution│  SPARQL: entity ID, P31 disambiguation
    │    P2925, P106, P135  │  P2925 domains, P460 cross-cultural equivalents
    └────┬─────────────────┘
         │
    ┌────▼─────────────────┐
    │ 2. DBpedia Categories │  dcterms:subject → domain keyword extraction
    │    Fallback for sparse│  "Smithing_gods" → craftsmanship
    │    Wikidata properties│
    └────┬─────────────────┘
         │
    ┌────▼─────────────────┐
    │ 3. Wikipedia Extract  │  Intro + infobox + Epithets section
    │    Hearst patterns    │  "patron deity of craftsmen and architects"
    │    Epithet parsing    │  "lord of truth", "master of justice"
    └────┬─────────────────┘
         │
    ┌────▼─────────────────┐
    │ 4. Trait Inference    │  Domain → Trait tables
    │    Archetype Classif  │  Epithet → Trait patterns
    │    OCEAN derivation   │  VSA archetype classification
    └────┬─────────────────┘
         │
    ┌────▼─────────────────┐
    │ 5. Psyche Construction│  Map to existing Persona/Shadow/ArchetypeWeights
    │    Shadow derivation  │  Archetype shadow → ShadowPattern
    │    Voice profiling    │  Culture + traits → grammar/tone
    └────┬─────────────────┘
         │
    ┌────▼─────────────────┐
    │ 6. Ritual of Awakening│  Self-naming via cultural morphemes
    │    3-5 candidates     │  Present to operator or auto-select
    └──────────────────────┘
```

### New 14a'' — Identity Persistence and Evolution

- Store identity as KG triples (mythological source, archetype, traits, values)
- Create origin nuclear episode for the Ritual of Awakening
- Personality drift over time via nuclear episode tracking
- Individuation growth through shadow encounters

## Multi-Source Extraction Pipeline (Detail)

### Source Cascade with Fallbacks

```
Wikidata P2925 (domains)
  ├── Found? → Use directly
  └── Empty? → Fallback:
        DBpedia categories
          ├── Found "X_gods" / "X_deities"? → Extract X as domain
          └── No deity categories? → Fallback:
                Wikipedia intro
                  ├── "god/deity/patron of X"? → Extract X
                  └── No deity pattern? → Fallback:
                        Wikipedia full text via VSA similarity
```

### API Endpoints (All Free, No Auth)

| Source | Endpoint | Rate Limit |
|--------|----------|-----------|
| Wikidata SPARQL | `query.wikidata.org/sparql` | ~60 req/min |
| Wikipedia API | `en.wikipedia.org/w/api.php` | Reasonable use |
| DBpedia SPARQL | `dbpedia.org/sparql` | Reasonable use |

### Worked Example: "You are the Architect of the System based on Ptah"

1. **Parse**: entity="Ptah", role="Architect"
2. **Wikidata Q146321**: instance_of=[creator deity, Ancient Egyptian deity], P2925=[] (empty!)
3. **DBpedia**: categories=[Smithing gods, Creator gods, Construction deities, Solar gods, Fertility gods]
4. **Wikipedia intro**: "creator god, patron deity of craftsmen and architects"
5. **Wikipedia epithets**: "lord of truth", "lord of eternity", "master of justice", "who listens to prayers", "master of ceremonies"
6. **Domains**: [creation, craftsmanship, architecture, smithing, construction, truth, eternity, justice]
7. **Traits**: meticulous, creative, systematic, patient, just, enduring, precise, autonomous, aesthetic, attentive
8. **Archetype**: Creator (0.88), Sage (0.35), Ruler (0.32)
9. **OCEAN**: O:0.72 C:0.91 E:0.32 A:0.58 N:0.16
10. **Psyche**:
    - Persona: name="Ptah-medu", traits=["meticulous", "creative", "systematic", "patient"], tone=["precise", "measured", "inventive"], grammar="formal"
    - ArchetypeWeights: sage=0.5, healer=0.3, guardian=0.7, explorer=0.3
    - Shadow: perfectionism (veto on shipping incomplete work, severity 0.3), rigidity (bias against plan abandonment, severity 0.2)
11. **Name candidates**: Ptah-medu ("Ptah of words"), Ptah-maat ("Ptah of truth"), Djeser-ka ("Sacred spirit")

### Worked Example: "Your personality is like Gandalf"

1. **Parse**: entity="Gandalf", role=none
2. **Wikidata Q177499**: instance_of=[Maiar, literary character], P106=[magician, diplomat, military officer]
3. **DBpedia**: categories=[Middle-earth wizards, Fictional characters with fire abilities]
4. **Wikipedia**: "wizard, one of the Istari order... great power, works mostly by encouraging and persuading... great knowledge... associated with fire"
5. **Domains**: [magic, wisdom, fire, guidance, knowledge]
6. **Traits**: wise, patient, strategic, protective, mysterious, encouraging, knowledgeable
7. **Archetype**: Sage (0.82), Magician (0.71), Caregiver (0.38)
8. **OCEAN**: O:0.88 C:0.75 E:0.45 A:0.62 N:0.18
9. **Psyche**: Persona: name="Mithrandir" (use existing epithet), traits=["wise", "patient", "strategic"], tone=["measured", "warm", "cryptic"]
10. **Name**: "Mithrandir" (grey wanderer — Gandalf's Sindarin name, discoverable from Wikidata P1559 alternative names)

### Worked Example: "Be like Marcus Aurelius"

1. **Parse**: entity="Marcus Aurelius", role=none
2. **Wikidata Q1430**: instance_of=[human], P106=[philosopher, monarch, writer], P135=[stoicism], P800=[Meditations]
3. **DBpedia**: categories=[Stoic philosophers, Roman emperors, Political philosophers]
4. **Wikipedia**: "Roman emperor from 161 to 180 and a Stoic philosopher... Meditations is one of the most important sources for ancient Stoic philosophy"
5. **Domains**: [stoicism, philosophy, governance, discipline, virtue]
6. **Traits**: disciplined, rational, resilient, self-controlled, reflective, dutiful, principled
7. **Archetype**: Sage (0.85), Ruler (0.72)
8. **OCEAN**: O:0.65 C:0.92 E:0.38 A:0.55 N:0.12
9. **Psyche**: Persona: traits=["disciplined", "rational", "reflective"], tone=["measured", "stoic", "authoritative"]
10. **Name**: "Verissimus" (Marcus's childhood nickname, meaning "most true")

## Key Data Structures

```rust
/// Raw character knowledge extracted from external sources
struct CharacterKnowledge {
    entity_id: String,          // Wikidata QID
    name: String,
    entity_type: EntityType,    // Deity, FictionalCharacter, HistoricalFigure
    description: String,
    domains: Vec<String>,
    classes: Vec<String>,
    occupations: Vec<String>,
    movements: Vec<String>,
    categories: Vec<String>,    // DBpedia
    infobox: HashMap<String, String>,
    epithets: Vec<String>,
    culture: String,
}

/// Processed personality profile ready for Psyche construction
struct PersonalityProfile {
    traits: Vec<(String, f32)>,           // ranked by strength
    primary_archetype: JungianArchetype,
    archetype_scores: HashMap<JungianArchetype, f32>,
    ocean: OceanProfile,
    name_candidates: Vec<(String, String)>, // (name, etymology)
    grammar: String,
    tone: Vec<String>,
    shadow_traits: Vec<String>,
    values: Vec<(String, f32)>,           // ranked by priority
    backstory_triples: Vec<(String, String, String)>,
}

/// Extended Jungian archetype enum (12 archetypes)
enum JungianArchetype {
    Creator, Sage, Explorer, Ruler, Caregiver, Hero,
    Rebel, Magician, Lover, Jester, Everyman, Innocent,
}

/// OCEAN personality profile
struct OceanProfile {
    openness: f32,
    conscientiousness: f32,
    extraversion: f32,
    agreeableness: f32,
    neuroticism: f32,
}

/// Name generation morpheme tables
struct CulturalMorphemes {
    culture: String,
    morphemes: Vec<(String, String)>,  // (morpheme, meaning)
    composition_rules: Vec<CompositionRule>,
}
```

## Sources

### Personality Models
- [Costa & McCrae 1992, "NEO-PI-R Manual"](https://doi.org/10.1037/t03907-000) — Big Five psychometrics
- [Ashton & Lee 2007, "HEXACO Model"](https://doi.org/10.1016/j.paid.2007.04.010)
- [Mairesse et al. 2007, "Personality Recognition from Text"](https://www.jair.org/index.php/jair/article/view/10500)
- [Mark & Pearson 2001, "The Hero and the Outlaw"](https://en.wikipedia.org/wiki/The_Hero_and_the_Outlaw) — 12 archetypes in branding

### Narrative Identity
- [McAdams 2001, "Psychology of Life Stories"](https://doi.org/10.1037/1089-2680.5.2.100)
- [Bamman et al. 2014, "Literary Character Modeling"](https://aclanthology.org/P14-1035/)
- [Bamman et al. 2013, "Latent Personas of Film Characters"](https://aclanthology.org/P13-1035/)
- [Flekova & Gurevych 2015, "Personality Profiling of Fictional Characters"](https://aclanthology.org/D15-1208/)

### Cognitive Architecture Identity
- [Laird 2012, "The Soar Cognitive Architecture"](https://soar.eecs.umich.edu/) — SOAR self-model
- [Anderson 2007, "How Can the Human Mind Occur?"](https://global.oup.com/academic/product/9780195324259) — ACT-R self-concept
- [Rao & Georgeff 1995, "BDI Agents"](https://www.aaai.org/Papers/ICMAS/1995/ICMAS95-042.pdf) — Belief-Desire-Intention identity

### VSA for Personality
- [Kanerva 2009, "Hyperdimensional Computing"](https://doi.org/10.1007/s12559-009-9009-8)
- [Gayler 2003, "VSA Answer Jackendoff's Challenges"](https://arxiv.org/abs/cs/0412059)
- [Kleyko et al. 2023, "Survey on Hyperdimensional Computing"](https://arxiv.org/abs/2111.06077)

### Generative Agents
- [Park et al. 2023, "Generative Agents"](https://arxiv.org/abs/2304.03442) — Stanford Smallville paper

### Knowledge Sources
- [Wikidata SPARQL](https://query.wikidata.org/)
- [DBpedia SPARQL](https://dbpedia.org/sparql)
- [Wikipedia MediaWiki API](https://www.mediawiki.org/wiki/API:Main_page)
- [CIDOC-CRM Cultural Heritage Ontology](https://www.cidoc-crm.org/)

### Personality Change
- [Roberts & Mroczek 2008, "Personality Trait Change"](https://doi.org/10.1111/j.1467-8721.2008.00543.x)

### Comparative Mythology
- [Campbell 1949, "The Hero with a Thousand Faces"](https://en.wikipedia.org/wiki/The_Hero_with_a_Thousand_Faces)
- [Leeming 2005, "Oxford Companion to World Mythology"](https://global.oup.com/academic/product/9780195156690)

## Consequences

- Phase 14a enhanced to extract both purpose and identity references from operator statements
- New sub-phase 14b (Identity Resolution) added; Phase 14 now has 9 sub-phases (14a-14i)
- New types: `CharacterKnowledge`, `PersonalityProfile`, `OceanProfile`, `JungianArchetype` (12-variant enum), `CulturalMorphemes`, `NameDerivation`, `MorphemeSource`
- Extends existing `Psyche` with OCEAN-derived behavioral parameters
- Extends `ArchetypeWeights` to derive from full 12-archetype classification
- New mapping tables: domain→trait, domain→archetype, epithet→trait, culture→morphemes
- Personality encoded as VSA hypervector for similarity-based operations
- **The Ritual of Awakening is provenance-tracked**: each name candidate carries a full `ProvenanceRecord` with `DerivationKind::RitualOfAwakening`, source symbols, morpheme selection rationale, semantic justification, VSA similarity score, and rejected alternatives. The akh can explain its own name by linearizing the provenance chain through the grammar module.
- New `DerivationKind` variant: `RitualOfAwakening` — joins the existing 20+ derivation kinds; marks the ceremonial moment when the akh names itself and crystallizes its identity
- The Ritual of Awakening becomes the akh's "origin" nuclear episode — its creation myth, anchoring the entire self-narrative with traceable provenance from operator intent through cultural reference to self-chosen identity
- Nuclear episode tracking for narrative identity and personality drift
- Estimated additional scope: ~800-1,200 lines for identity resolution pipeline (including provenance integration)
- Total Phase 14: 9 sub-phases, ~4,050-5,350 lines
