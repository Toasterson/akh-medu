# ADR 045 — T5 Training Infrastructure

> Date: 2026-03-27
> Status: Proposed
> Phase: 35
> Depends on: None (offline, external to akh-medu runtime)
> Produces models for: Phase 32 (knowledge elicitation), Phase 33 (NLU + NLG)
> Refined by: Phase 27 (in-situ Burn training loop)

## Context

The akh-medu architecture uses three purpose-built T5 models instead of a
single general-purpose LLM (Qwen). All three must be fine-tuned offline before
shipping, then refined in-situ by Phase 27's Burn training loop.

| Model | Task | Training Source |
|---|---|---|
| T5 NLU | text → AbsTree JSON | Synthetic corpus (Qwen distillation) |
| T5 NLG | triples → natural text | WebNLG benchmark |
| T5 Knowledge | concept → triples | Wikidata + ConceptNet |

This ADR covers the offline training infrastructure that produces these models.

## Decision

### Build a standalone training harness (external to akh-medu)

The training pipeline is a **separate Rust project** (or Python script for
prototyping) that:
1. Downloads and processes training data (Wikidata, ConceptNet, WebNLG)
2. Fine-tunes T5-Base using Burn (or PyTorch for initial prototyping)
3. Evaluates on held-out test sets
4. Quantizes to GGUF via Candle tensor-tools
5. Produces model artifacts for `akh setup models`

This is NOT part of the akh-medu binary. It runs once per release (or on
a schedule for nightly builds).

### Data sources

**Wikidata** (CC0, no restrictions):
- Weekly JSON dumps (~100 GB uncompressed)
- Filter: entities with ≥5 sitelinks (ensures notability) → ~2-5M entities
- Each entity has structured statements (property-value pairs)
- Directly maps to (subject, predicate, object) triples
- Predicates normalized to akh-medu conventions (e.g., P31 → "is-a", P279 → "subclass-of")

**ConceptNet** (CC-BY-SA 4.0):
- ~34M assertions across multiple languages
- Relations: IsA, PartOf, HasA, UsedFor, CapableOf, Causes, HasProperty, etc.
- Complementary to Wikidata: more common-sense, less encyclopedic
- Already structured as (start, relation, end) with weights
- Multi-lingual: assertions in EN, RU, FR, ES, and more

**WebNLG** (CC-BY-NC 4.0):
- Standard benchmark for RDF-to-text generation
- ~45K (triple set, reference text) pairs across 15 domains
- Includes RDF triples from DBpedia
- Multi-lingual editions (2023 challenge): EN, RU, + under-resourced languages

**Synthetic AbsTree corpus** (generated):
- Run Qwen2.5-1.5B on diverse inputs (or use Claude for higher quality)
- Collect (input_text, AbsTree_json) pairs
- Hand-curate edge cases for all 12 AbsTree variants
- Include multilingual examples (EN, RU, FR, ES, AR)

### Three training pipelines

Each produces one model. Can run independently.

## Consequences

- New repository or workspace: `akh-medu-models/` (training code + configs)
- Model artifacts published to HuggingFace or bundled with releases
- `akh setup models` downloads the latest trained models
- Phase 27 Burn loop refines shipped models in-situ per user
- Training infrastructure reusable for future model additions

## References

- Wikidata dumps: https://dumps.wikimedia.org/wikidatawiki/entities/
- ConceptNet: https://conceptnet.io/ (API + data downloads)
- WebNLG: https://synalp.gitlabpages.inria.fr/webnlg-challenge/
- Candle tensor-tools: quantize safetensors → GGUF
- Burn framework: https://burn.dev/
