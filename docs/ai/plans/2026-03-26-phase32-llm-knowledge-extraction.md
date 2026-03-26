# Phase 32 — Knowledge Extraction for Bootstrap

> Date: 2026-03-26 (revised 2026-03-27)
> Status: Planned
> Phase: 32
> Depends on: Phase 26b (Candle backend), Phase 35 (T5 training infrastructure)
> Optional enhancement: Phase 26d (neural bridge, for hallucination detection)
> Enhances: Phase 14 (bootstrap), Phase 30 (skill authoring), Phase 31e (skill synthesis)
> ADR: [042-llm-knowledge-extraction](../decisions/042-llm-knowledge-extraction.md)

## Motivation

The bootstrap pipeline extracts ~1 triple per 28 sentences from web sources.
A purpose-built T5 Knowledge model, trained offline on Wikidata + ConceptNet
(Phase 35), can generate structured triples about any concept from its learned
knowledge patterns — yielding 10-100x more triples per domain.

This model is part of the unified T5 stack (no Qwen dependency):

| Model | Task | Size (GGUF) | Resident? |
|---|---|---|---|
| T5 NLU | text → AbsTree | ~140-250 MB | Always |
| T5 NLG | triples → text | ~140-250 MB | Always |
| **T5 Knowledge** | **concept → triples** | **~140-250 MB** | **Always** |
| DistilBERT NER | entity extraction | ~130 MB | Always |
| **Total** | | **~550-880 MB** | |

All three T5 models use the same Candle `quantized_t5` loader. No Qwen, no
on-demand loading/unloading, no 1.1 GB memory spikes.

## Sub-phases

### 32a — LLM Elicitation Engine (~600 lines)

**New module**: `src/bootstrap/llm_elicit.rs`

Core extraction engine that prompts the LLM with focused queries and parses
structured triple output.

```rust
pub struct KnowledgeElicitor {
    model: Arc<T5Model>,              // T5 Knowledge model (Candle quantized_t5)
    tokenizer: Arc<Tokenizer>,
    config: ElicitConfig,
}

pub struct ElicitConfig {
    pub max_triples_per_pass: usize,  // Cap per prompt (default: 30)
    pub passes: Vec<ElicitPass>,      // Which passes to run
    pub self_consistency_rounds: u8,  // Re-ask count for reliability (default: 2)
    pub min_agreement: f32,           // Fraction of rounds a triple must appear in (default: 0.5)
}

pub enum ElicitPass {
    Taxonomy,         // is-a, subclass-of, part-of
    Properties,       // has-property, characterized-by, measured-in
    Relationships,    // causes, enables, prevents, depends-on
    Procedures,       // action schemas (for Phase 30)
    Temporal,         // event sequences, before/after
}

pub struct ElicitResult {
    pub triples: Vec<ElicitedTriple>,
    pub pass: ElicitPass,
    pub model_id: String,
    pub prompt_hash: u64,
}

pub struct ElicitedTriple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub raw_confidence: f32,         // From self-consistency
    pub agreement_count: u8,         // How many rounds produced this triple
}
```

**Prompt templates per pass**:

```rust
fn taxonomy_prompt(domain: &str, concepts: &[String]) -> String {
    format!(
        "List the main categories and subcategories of {domain}.\n\
         Format each fact as a JSON object: {{\"s\": \"subject\", \"p\": \"predicate\", \"o\": \"object\"}}\n\
         Use predicates: is-a, subclass-of, part-of, instance-of.\n\
         Output a JSON array. Known concepts: {}\n",
        concepts.join(", ")
    )
}

fn properties_prompt(concept: &str, domain: &str) -> String {
    format!(
        "List all known properties and attributes of '{concept}' in the domain of {domain}.\n\
         Format: JSON array of {{\"s\": \"{concept}\", \"p\": \"predicate\", \"o\": \"value\"}}.\n\
         Use predicates: has-property, characterized-by, measured-in, typical-value, range.\n"
    )
}

fn relationships_prompt(concepts: &[String], domain: &str) -> String {
    format!(
        "How do these concepts relate to each other in {domain}?\n\
         Concepts: {}\n\
         Format: JSON array of {{\"s\": \"concept1\", \"p\": \"relation\", \"o\": \"concept2\"}}.\n\
         Use predicates: causes, enables, prevents, depends-on, precedes, co-occurs-with.\n",
        concepts.join(", ")
    )
}
```

**Self-consistency**: Run each pass N times with slight prompt variations
(reorder concept list, rephrase instruction). Keep triples that appear in
≥ min_agreement fraction of rounds.

**Output constraint**: JSON array via LogitsProcessor or brace-depth tracking
(Phase 26b). Small LLMs sometimes produce malformed JSON — parser is lenient
(accepts trailing commas, partial arrays, recovers partial output).

### 32b — Validation Pipeline (~500 lines)

**New module**: `src/bootstrap/validate_elicit.rs`

Multi-layer validation for LLM-generated triples:

```rust
pub struct ElicitValidator {
    engine: Arc<Engine>,
    bridge: Option<Arc<NeuralVsaBridge>>,  // Phase 26d, optional
}

pub struct ValidationResult {
    pub accepted: Vec<ValidatedTriple>,
    pub rejected: Vec<(ElicitedTriple, RejectionReason)>,
    pub flagged: Vec<(ElicitedTriple, FlagReason)>,  // For human review
}

pub struct ValidatedTriple {
    pub triple: ElicitedTriple,
    pub final_confidence: f32,
    pub validation_notes: Vec<String>,
}

pub enum RejectionReason {
    ContradictExisting { conflicting_triple: Triple },
    ArityViolation { constraint: String },
    SelfReferential,
    Duplicate { existing_triple: Triple },
}

pub enum FlagReason {
    NeuralBridgeIncoherent { similarity: f32 },
    LowSelfConsistency { agreement: f32 },
    NoWebCorroboration,
    NovelPredicate { predicate: String },
}
```

**Validation layers** (in order):

1. **Syntactic**: Well-formed triple (non-empty subject/predicate/object)
2. **Deduplication**: Check against existing KG triples (exact + fuzzy via VSA)
3. **Arity constraints** (Phase 9j): Does this predicate accept these argument types?
4. **Contradiction detection** (Phase 9l): Does this conflict with existing knowledge?
5. **Neural bridge coherence** (Phase 26d, if available):
   ```rust
   // Run concept through LLM → hidden state → HyperVec
   let concept_vec = bridge.encode(llm, concept_text)?;
   let nearest = item_memory.search(&concept_vec, k=5)?;
   // If nearest neighbors are all from the same domain → coherent
   // If isolated (low similarity to everything) → possibly hallucinated
   if nearest[0].similarity < COHERENCE_THRESHOLD {
       flag(NeuralBridgeIncoherent { similarity });
   }
   ```
6. **Self-consistency score**: Map agreement_count / rounds → confidence factor

**Confidence assignment**:
```rust
fn compute_confidence(triple: &ElicitedTriple, validation: &ValidationContext) -> f32 {
    let base = 0.55;  // LLM-only baseline
    let mut conf = base;

    // Self-consistency bonus
    let agreement = triple.agreement_count as f32 / validation.total_rounds as f32;
    conf += (agreement - 0.5) * 0.2;  // +0.1 for perfect agreement

    // Neural bridge coherence bonus
    if let Some(coherence) = validation.neural_coherence {
        conf += (coherence - 0.5) * 0.15;
    }

    // Web cross-validation bonus (applied later in 32c)
    // Applied in merge step

    conf.clamp(0.3, 0.95)
}
```

### 32c — Bootstrap Integration (~500 lines)

Wire elicitation into the existing bootstrap pipeline stages.

**Modify**: `src/bootstrap/expand.rs` (Stage 4 — Domain Expansion)

```rust
// After Wikidata + Wikipedia + ConceptNet queries:
if candle_available {
    let elicitor = LlmElicitor::new(candle_backend, elicit_config);

    // Concept elicitation: expand the skeleton ontology
    let taxonomy = elicitor.elicit(
        &[ElicitPass::Taxonomy],
        &domain,
        &seed_concepts,
    )?;

    let validated = validator.validate(&taxonomy)?;

    // Merge: LLM concepts that match web concepts get boosted
    for triple in validated.accepted {
        if web_concepts.contains_label(&triple.subject) ||
           web_concepts.contains_label(&triple.object) {
            triple.final_confidence += 0.2;  // Cross-validated
        }
        skeleton.add_concept(triple)?;
    }
}
```

**Modify**: `src/bootstrap/ingest.rs` (Stage 7 — Curriculum Ingestion)

```rust
// For each concept in the curriculum:
if candle_available {
    // LLM fact elicitation per concept
    let facts = elicitor.elicit(
        &[ElicitPass::Properties, ElicitPass::Relationships],
        &domain,
        &[concept.name.clone()],
    )?;

    let validated = validator.validate(&facts)?;

    // Merge with web-extracted triples
    for triple in validated.accepted {
        let web_match = web_triples.find_similar(&triple);
        if let Some(wt) = web_match {
            // Agreement: boost to high confidence
            triple.final_confidence = (triple.final_confidence + wt.confidence) / 2.0 + 0.15;
        }
        engine.add_triple(&triple.to_engine_triple(engine)?)?;
    }
}
```

**Parallel execution**: LLM elicitation and web API calls run concurrently
(LLM on CPU/Metal, web on async I/O). No serialization needed.

### 32d — Provenance & Retraction (~300 lines)

New provenance kind for LLM-elicited knowledge:

```rust
// In DerivationKind enum:
LlmElicitation {
    model_id: String,          // "qwen2.5-1.5b-q4km" or "qwen2.5-7b-q4km"
    pass: String,              // "taxonomy", "properties", "relationships"
    prompt_hash: u64,          // Reproducibility
    self_consistency: f32,     // Agreement score
    neural_coherence: Option<f32>, // Bridge validation score
    web_corroborated: bool,    // Was this confirmed by web source?
}
```

**Confidence decay**: LLM-only triples (not web-corroborated) decay via
temporal projection (Phase 9k). Rate: -0.05 per month. If confidence drops
below 0.3, triple is flagged for review or auto-retracted.

**Bulk retraction**: If a model is later found to be unreliable for a domain,
all triples with `LlmElicitation { model_id }` provenance can be queried and
retracted via TMS cascade (Phase 9c).

### 32e — Elicitation Scheduling (~200 lines)

Wire knowledge elicitation into the daemon's background learning cycle.

**No on-demand loading needed**: The T5 Knowledge model is always resident
(~140-250 MB), sharing the Candle runtime with the NLU and NLG models.
Elicitation runs at any time without memory spikes.

**New idle task**: `elicit_knowledge`
- Trigger: When the agent identifies a knowledge gap (directed curiosity,
  Phase 11j) or when a domain has low triple density
- Schedule: During sleep/consolidation phase, after standard background learning
- Rate limit: Max N elicitation sessions per hour (configurable, default: 2)
- No memory budget concern — model is always loaded

**Integration with continuous learning** (Phase 11):
```
Directed curiosity identifies gap → "I know little about thermodynamics"
    │
    ▼
Elicitation task queued
    │
    ▼
T5 Knowledge generates triples about thermodynamics (passes 1-3)
    │
    ▼
Validation pipeline filters hallucinations
    │
    ▼
Accepted triples ingested → KG enriched
    │
    ▼
Next curiosity cycle finds fewer gaps → satisfaction signal
```

## Estimated Effort

| Sub-phase | Lines | Complexity | Dependencies |
|---|---|---|---|
| 32a — Elicitation engine | ~600 | Medium | Phase 26b (Candle), Phase 35 (T5 Knowledge model) |
| 32b — Validation pipeline | ~500 | Medium | Phase 9j/9l, optionally 26d |
| 32c — Bootstrap integration | ~500 | Medium | 32a, 32b |
| 32d — Provenance & retraction | ~300 | Low | Phase 9c/9k |
| 32e — Scheduling | ~200 | Low | Phase 11j |
| **Total** | **~2,100** | | |

## Feature Flags

```toml
[features]
llm-bootstrap = ["candle-backend"]  # LLM knowledge elicitation
```

Falls back gracefully: if `candle-backend` not available, bootstrap uses
existing web-only pipeline. If `neural-bridge` not available, validation
skips coherence check.

## Expected Impact

**Current bootstrap** (web-only, conservative extraction):
- Domain expansion: ~50-100 concepts from Wikidata
- Ingestion: ~1 triple per 28 sentences from web text
- Total: ~200-500 triples per domain after full bootstrap

**With LLM elicitation** (5 passes × 2 self-consistency rounds):
- Taxonomy pass: ~30-50 triples
- Properties pass: ~5-10 triples × 50 concepts = ~250-500 triples
- Relationships pass: ~30-50 triples
- Web-extracted: ~200-500 triples (unchanged)
- Total: ~500-1,100 triples per domain, 2-5x improvement
- Cross-validated (LLM + web agreement): ~30-40% of triples at high confidence

**Bootstrapping time**: LLM elicitation runs in parallel with web APIs.
5 passes × 30 triples × ~3 sec/pass = ~15 seconds of LLM time per domain.
Negligible compared to web API latency.

## Relationship to Other Phases

- **Phase 26b** (Candle): Provides the `quantized_t5` inference runtime
- **Phase 26d** (neural bridge): Provides hallucination detection via coherence check
- **Phase 27** (Burn training): Elicited triples become training data — the T5
  Knowledge model improves at generating triples over time (self-improving)
- **Phase 28** (n akh router): Router can trigger elicitation when a query hits
  a domain with low KG density
- **Phase 30c** (skill authoring): Skill content generation reuses the elicitation
  engine for domain facts
- **Phase 31e** (skill synthesis): Dynamic skill creation triggers elicitation
  for the gap domain
- **Phase 35** (T5 training infrastructure): Produces the T5 Knowledge model
  from Wikidata + ConceptNet. This phase provides the model; Phase 32 uses it.
