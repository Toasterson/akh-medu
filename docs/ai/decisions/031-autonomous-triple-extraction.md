# ADR 031: Autonomous Triple Extraction via LLM-Backed Pipeline

**Status:** Accepted
**Date:** 2026-03-19
**Deciders:** toasty + Claude

## Context

The rule-based text ingest (`TextIngestTool`) extracted 1 triple from 28 sentences during testing. Claude-as-LLM + `assert_batch` is effective but requires an interactive session. The akh needs to populate its own knowledge graph autonomously — running 24/7 in daemon mode without human intervention.

The existing infrastructure provides:
- Curiosity drives that detect knowledge gaps and generate learning goals
- Content fetching (HTTP, Wikipedia, ConceptNet) already in the OODA loop
- A local Qwen2.5-1.5B LLM already loaded for NLU boundary translation
- Cheap external APIs (OpenRouter/DeepSeek at ~$0.14/Mtok) as optional fallback

What's missing: a structured triple extraction step between "fetch text" and "store in KG."

## Decision

### 1. New `src/extraction/` module

A top-level module (peer to `src/nlu/`, `src/library/`) containing:

- **`TripleExtractor` trait** with `extract(&self, text: &str) -> ExtractionResult<Vec<RawTriple>>`
- **`LocalLlmBackend`** — shares `Arc<LlmTranslator>` from NLU pipeline (no second model load)
- **`ExternalApiBackend`** — OpenAI-compatible `/v1/chat/completions` via `ureq` (synchronous, matches OODA loop)
- **`RegexFallback`** — reuses existing patterns from `text_ingest.rs`
- **`ExtractionOrchestrator`** — three-tier fallback: Local LLM -> External API -> Regex

### 2. Independent-acting architecture

The extraction system is NOT a passive library called by the continuous learning loop. It is an **autonomous actor** with its own daemon task:

```
run_knowledge_extraction:
  1. Query curiosity targets (ZPD-proximal concepts with high information gap)
  2. For each target, check if content has been fetched but not extracted
  3. Run extraction on unprocessed content
  4. Assert triples with LlmTripleExtraction provenance
  5. Run competence assessment to verify improvement
  6. If score improved: commit. If not: retract low-confidence triples.
```

This runs on its own timer (default: 45 min) independently from the OODA cycle, reflection, and gap analysis tasks. The akh decides what to extract based on its own curiosity, not external instruction.

### 3. Three-tier fallback with budget control

| Tier | Backend | Cost | Quality | Availability |
|------|---------|------|---------|-------------|
| 1 | Local Qwen2.5-1.5B | $0 | Good for simple text | Always (with `nlu-llm` feature) |
| 2 | External API (OpenRouter/DeepSeek) | ~$0.14/Mtok | Better for complex text | When configured |
| 3 | Regex patterns | $0 | Basic (is-a, part-of) | Always |

Budget: `max_extraction_api_calls` per cycle (default: 10) keeps monthly cost under $2 even with daily extraction cycles.

### 4. Config

```toml
[extraction]
enabled = true
prefer_local = true
max_chunk_chars = 2000
min_confidence = 0.5
max_triples_per_chunk = 20

[extraction.external_api]
base_url = "https://openrouter.ai/api/v1"
# api_key via AKH_EXTRACTION_API_KEY env var
model = "deepseek/deepseek-chat"
max_tokens = 512
temperature = 0.1
timeout_secs = 30
```

### 5. Provenance

New `DerivationKind::LlmTripleExtraction` variant tracks:
- Which backend produced the triple
- Source text hash (for deduplication)
- Model identifier
- Extraction confidence

This enables the TMS to distinguish LLM-extracted knowledge from seed-pack or human-asserted knowledge, and apply appropriate retraction policies.

## Consequences

**Positive:**
- The akh can learn autonomously 24/7 without Claude or any human attached
- Three-tier fallback means it always works, even offline with no API key
- Shared LLM model (no additional RAM)
- Full provenance tracking enables quality-based retraction

**Negative:**
- Local Qwen2.5-1.5B extraction quality is lower than Claude/GPT-4
- External API adds a (optional) cloud dependency
- Extraction prompt tuning will need iteration

**Risks:**
- LLM hallucination → garbage triples. Mitigated by: min_confidence threshold, competence re-assessment after extraction, retraction of non-improving triples.
- API cost runaway. Mitigated by: per-cycle call budget, prefer-local default.

## Alternatives Considered

1. **Enhance regex NER only** — rejected: ceiling is too low (1/28 sentences)
2. **Always require Claude** — rejected: no independence, requires paid session
3. **Fine-tune Qwen on triple extraction** — future option, not MVP
4. **Use ONNX NER model for extraction** — the existing Tier 2 NER handles entity recognition but not relation extraction; would need a separate relation extraction model
