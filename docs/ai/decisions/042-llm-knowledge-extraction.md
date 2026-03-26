# ADR 042 — LLM Knowledge Extraction for Bootstrap

> Date: 2026-03-26
> Status: Proposed
> Phase: 32
> Depends on: Phase 26b (Candle LLM backend), Phase 26d (neural bridge, for validation)
> Enhances: Phase 14 (bootstrap pipeline), Phase 30 (skill authoring), Phase 31e (skill synthesis)

## Context

The bootstrap pipeline currently acquires knowledge from:
1. **Web APIs** (Wikidata, Wikipedia, ConceptNet) — ~50 API calls, rate-limited
2. **Rule-based extraction** from fetched text — extracts ~1 triple per 28 sentences
3. **Optional LLM extraction** from text — prompt LLM to find triples in a passage

This is conservative by design — prioritizes precision over recall. But it means
bootstrapping a new domain yields sparse knowledge that takes many cycles to fill.

Meanwhile, the local LLM (Qwen2.5-1.5B, or 7B with PolarQuant) already "knows"
vast amounts from its pretraining data. This knowledge is locked in the weights
and only accessible via text generation. We fetch a Wikipedia article, pass it
to the LLM for extraction, and get back a handful of triples — when the LLM
could directly generate hundreds of triples about a domain from memory alone.

The known limitation from project memory confirms this:
> "Rule-based text ingest is conservative — extracted 1 triple from 28 sentences;
> Claude-as-LLM + assert_batch is more effective for knowledge population"

## Decision

### Use the local LLM as a primary knowledge source during bootstrapping

Instead of treating the LLM as a text-extraction tool (give it text, extract
triples), treat it as a **knowledge oracle** (ask it what it knows, generate
triples directly from its weights).

**Pipeline**:
```
Bootstrap: "Learn about stellar evolution"
    │
    ▼
1. LLM knowledge elicitation
   "List all facts about stellar evolution as (subject, predicate, object) triples"
   → LLM generates 50-200 triples from pretraining knowledge
    │
    ▼
2. Symbolic validation
   - Arity constraints (Phase 9j)
   - Contradiction detection (Phase 9l)
   - Duplicate detection (existing KG triples)
    │
    ▼
3. Neural bridge verification (Phase 26d)
   - Run concepts through LLM → hidden states → HyperVec
   - Check: do vectors cluster near existing domain concepts?
   - Divergent vectors → possible hallucination → lower confidence
    │
    ▼
4. Confidence assignment
   - LLM-only triples: 0.5-0.6 (moderate, unverified)
   - LLM + web API agreement: 0.8-0.9 (cross-validated)
   - LLM + neural bridge coherent: +0.1 bonus
   - LLM contradicts existing KG: reject or flag for review
    │
    ▼
5. KG ingestion with provenance
   - DerivationKind::LlmElicitation { model, prompt_hash, confidence }
   - Source tracking enables later verification or retraction
    │
    ▼
6. Web API cross-reference (existing pipeline)
   - Run in parallel with LLM elicitation
   - Triples confirmed by both sources get boosted confidence
   - Web-only triples still ingested at original confidence
```

### Integration with existing bootstrap stages

The bootstrap pipeline (Phase 14) has 8 stages. LLM elicitation slots in as
an enhancement to **Stage 4 (Domain Expansion)** and **Stage 7 (Ingestion)**:

**Stage 4 — Domain Expansion** (currently: Wikidata + Wikipedia + ConceptNet):
- Add: LLM concept elicitation ("What are the key concepts in domain X?")
- LLM generates a concept list with brief descriptions
- Each concept becomes a candidate node in the skeleton ontology
- Cross-reference with Wikidata results (agreement → keep, novel → lower confidence)

**Stage 7 — Curriculum Ingestion** (currently: fetch + extract from resources):
- Add: LLM fact elicitation per concept ("Tell me everything about concept Y")
- LLM generates triples about each concept from the curriculum
- Validate and ingest alongside web-extracted triples
- Compare: LLM-generated vs web-extracted for the same concept → boost on agreement

### Multi-pass elicitation strategy

Small LLMs (1.5B-7B) can't dump their entire knowledge about a domain in one
generation. Use focused, iterative prompts:

**Pass 1 — Taxonomy**: "List the main categories and subcategories of {domain}"
→ `is-a`, `subclass-of`, `part-of` triples

**Pass 2 — Properties**: For each key concept: "What are the properties of {concept}?"
→ `has-property`, `characterized-by`, `measured-in` triples

**Pass 3 — Relationships**: "How do these concepts relate to each other: {concept_list}"
→ `causes`, `enables`, `prevents`, `depends-on` triples

**Pass 4 — Procedures** (if skill authoring, Phase 30): "How does one {do_thing}?"
→ action schemas, plan templates

**Pass 5 — Temporal**: "What sequence of events occurs in {process}?"
→ event calculus entries, temporal ordering

Each pass uses a focused prompt with JSON output constraint (LogitsProcessor
or brace-depth tracking from Phase 26b).

### Hallucination mitigation

LLMs hallucinate. For a symbolic engine where incorrect triples poison reasoning,
this is critical. Multi-layer defense:

1. **Symbolic validation**: Arity constraints, contradiction detection against
   existing KG (these are already implemented in Phases 9j, 9l)

2. **Neural bridge coherence** (Phase 26d): Run extracted concepts through the
   LLM hidden state → HyperVec pipeline. Concepts that cluster near known
   domain symbols are likely real. Isolated vectors may be hallucinated.

3. **Self-consistency**: Ask the same question with different prompt phrasings.
   Triples that appear across multiple phrasings are more reliable.

4. **Web cross-validation**: Triples confirmed by Wikidata/Wikipedia get
   confidence boost. Triples only from LLM stay at moderate confidence.

5. **Provenance tracking**: Every LLM-elicited triple carries
   `DerivationKind::LlmElicitation` provenance. If later proven wrong,
   all downstream inferences can be retracted via TMS (Phase 9c).

6. **Confidence decay**: LLM-only triples that are never confirmed by external
   sources decay in confidence over time (temporal projection, Phase 9k).

## Rejected Alternatives

### Use cloud LLM for extraction (higher quality)
Violates FLOSS sovereignty principle. Also expensive at scale. The local LLM
is sufficient when combined with validation layers.

### Replace web APIs entirely with LLM
Web APIs provide independently verifiable facts. LLM knowledge is unverifiable
from a single source. Cross-validation between both is stronger than either alone.

### Auto-ingest without validation
Too risky. LLM hallucinations would propagate through the KG. The validation
pipeline + moderate confidence assignment ensures bad triples are caught.

## Consequences

- New module: `src/bootstrap/llm_elicit.rs` — structured knowledge extraction
- Extended: `src/bootstrap/expand.rs` — LLM concept elicitation in domain expansion
- Extended: `src/bootstrap/ingest.rs` — LLM fact elicitation in curriculum ingestion
- New `DerivationKind::LlmElicitation` variant in provenance
- Feature-gated: `llm-bootstrap` requires `candle-backend` (Phase 26b)
- Optional neural bridge validation requires Phase 26d (graceful degradation without it)
- Confidence assignment policy configurable in `BootstrapConfig`
- Phase 30c (skill authoring) can reuse the elicitation pipeline for skill content

## References

- Bootstrap pipeline: Phase 14, `src/bootstrap/`
- Contradiction detection: Phase 9l, `src/graph/`
- Arity constraints: Phase 9j, `src/graph/`
- TMS retraction: Phase 9c, `src/tms.rs`
- Temporal confidence decay: Phase 9k, `src/temporal.rs`
- Neural bridge: Phase 26d, ADR 037
- Candle backend: Phase 26b, ADR 037
