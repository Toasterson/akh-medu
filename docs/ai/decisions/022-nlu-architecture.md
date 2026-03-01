# ADR-022: Four-Tier Hybrid NLU Architecture

> Date: 2026-02-24
> Status: Accepted
> Relates to: Phase 14j-14m, Release Alpha

## Context

The existing GF-inspired grammar framework (`src/grammar/`) handles declarative triples
("Dogs are mammals"), questions ("What is a dog?"), commands, and goals in 5 languages.
However, it cannot parse negation, quantifiers, comparatives, conditionals, temporal
expressions, modals, relative clauses, or complex sentence structures.

For Release Alpha, akh-medu needs a substantially richer NLU to serve as the language
boundary between human operators and the internal VSA/symbolic representation.

### Constraints

- **Must run on Mac Mini M2** (8-16 GB RAM total, NLU gets ~2-4 GB)
- **Must be FLOSS** — no proprietary models, no API dependencies, no cloud calls
- **Must support 5 languages**: EN, RU, FR, ES, AR
- **Must integrate with existing GF-inspired grammar** (AbsTree, abstract/concrete split)
- **Internal representation is VSA** — NLU translates to AbsTree, not to embeddings
- **Latency**: interactive, <500ms for 90% of input
- **Sovereignty**: no dependence on external services at runtime

## Decision

Adopt a **four-tier cascading hybrid** architecture:

1. **Tier 1 (Rule Parser)**: Extend existing recursive descent parser with negation,
   quantifiers, comparatives, conditionals, temporal, and modal patterns. Zero ML,
   zero dependencies, sub-microsecond. Handles ~70% of input.

2. **Tier 2 (Micro-ML)**: DistilBERT multilingual NER via ONNX Runtime (`ort` crate).
   ~130 MB, ~5ms. Entity recognition for names, dates, locations. Feature-gated.

3. **Tier 3 (Small LLM)**: Qwen2.5-1.5B-Instruct (Q4_K_M) via `llama-cpp-2`.
   ~1.1 GB, ~800ms. GBNF-constrained decoding ensures valid AbsTree JSON output.
   Fallback for complex/novel constructions. Feature-gated.

4. **Tier 4 (VSA Parse Ranker)**: Encode successful parses as hypervectors in exemplar
   memory. Rank ambiguous candidates by similarity. Self-improving. Zero extra memory.

## Alternatives Considered

### A. Grammatical Framework (GF) via FFI

The real GF Haskell library with Resource Grammar Library (40+ languages).

**Rejected because**:
- C runtime (`libpgf`) dormant since 2015, only tested on Linux x86
- `gf-core` C runtime tightly coupled to Haskell/GHC build system
- Arabic support partial in RGL
- GF grammars handle controlled language, not free-form NLU
- Maintenance burden of unmaintained C library on ARM is unacceptable
- Our existing GF-inspired design already captures the key abstract/concrete split

### B. Snips NLU-RS

Rust-native intent classification + slot filling.

**Rejected because**:
- Unmaintained since Sonos acquisition (2020)
- Training requires Python (inference-only in Rust)
- No multilingual support out of the box
- English-centric built-in entity types

### C. Full rust-bert / tch-rs

Transformer pipelines via libtorch or ONNX.

**Rejected because**:
- libtorch backend is ~2 GB library alone
- Full pipelines use 500-800 MB per model
- Overkill for structured extraction — using `ort` directly (Tier 2) is lighter

### D. spaCy / Stanza via Python FFI

Industry-standard NLP via PyO3.

**Rejected because**:
- Requires embedding Python runtime (~50 MB overhead + complexity)
- No Rust-native option
- Python FFI negates Rust's safety and deployment advantages

### E. CCG / SEMPRE-style Semantic Parsing

Combinatory Categorial Grammar for syntax-semantics.

**Rejected because**:
- SEMPRE is Java, unmaintained since ~2017
- CCG parsers require large lexicon induction, computationally expensive
- No Rust implementation exists
- Our AbsTree category system already captures the key type-theoretic principles

### F. Pure LLM (no rule parser)

Route everything through a small LLM.

**Rejected because**:
- 800ms latency for every input is unacceptable
- 1.5 GB RAM always resident
- Cannot leverage system's own knowledge for disambiguation
- Loses the self-improving property of the VSA parse ranker
- Unnecessary for the 70% of input the rule parser handles trivially

### G. Large LLM (7B+) via llama.cpp

Llama 3.1 8B, Mistral 7B, etc.

**Rejected because**:
- ~6 GB RAM — consumes too much of the M2 budget
- Only marginally better than Qwen2.5-1.5B for structured extraction
- With GBNF constraint, the smaller model is sufficient for valid output

## Consequences

### Positive

- **Graceful degradation**: Each tier is optional. Tier 1+4 work with zero ML deps.
  Adding `nlu-ml` enables Tier 2. Adding `nlu-llm` enables Tier 3. Full stack is `nlu-full`.
- **Fast path dominates**: 70% of input handled in <1ms with zero ML overhead.
- **Self-improving**: VSA parse ranker gets better as the knowledge graph grows.
- **FLOSS throughout**: ort (MIT/Apache), llama-cpp-2 (MIT), Qwen2.5 (Apache 2.0).
- **Sovereignty**: Everything runs locally, no cloud calls, no token costs.
- **Multilingual**: All tiers support EN/RU/FR/ES/AR.

### Negative

- **Three optional crates**: `ort`, `tokenizers`, `llama-cpp-2` add build complexity
  and platform-specific concerns (ONNX Runtime C++ lib, llama.cpp C build).
- **Model management**: Users must download ~1.2 GB of model files. Not bundled.
- **Tier 3 latency**: ~800ms is at the edge of interactive. Acceptable as fallback only.
- **Fine-tuning pipeline**: Training Qwen2.5 on akh-medu parse pairs requires separate
  tooling (Python + LoRA), not integrated into the main Rust binary.

### Risks

- **ONNX Runtime on ARM**: Well-supported for Apple Silicon but may need build flags.
  Fallback: skip Tier 2 if ONNX build fails, Tier 3 still works.
- **llama.cpp Metal support**: Works on M-series but is an active development target.
  May need periodic dependency updates.
- **Qwen2.5 quality at 1.5B**: May struggle with very complex constructions. Mitigation:
  GBNF constraint ensures valid output; self-training improves over time; upgrade path
  to 3B+ exists if hardware budget allows.

## Model Selection Rationale

Qwen2.5-1.5B-Instruct was selected over alternatives for these specific reasons:

| Criterion | Qwen2.5-1.5B | Phi-3-mini (3.8B) | TinyLlama (1.1B) | Mistral 7B |
|-----------|-------------|-------------------|-------------------|------------|
| RAM (Q4) | 1.5 GB | 2.8 GB | 1.0 GB | 5 GB |
| Multilingual | Excellent (all 5 langs) | Moderate | Poor (EN-centric) | Good |
| Structured output | Excellent | Excellent | Adequate | Excellent |
| License | Apache 2.0 | MIT | Apache 2.0 | Apache 2.0 |
| Fits budget | Yes | Tight | Yes | No |

Qwen2.5-1.5B is the best balance of multilingual quality, structured output capability,
and memory footprint for our specific 5-language AbsTree extraction task.
