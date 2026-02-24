# Phase 14j-14m — Natural Language Understanding Extension

> Date: 2026-02-24

- **Status**: Planned
- **Phase**: 14j-14m (extends Phase 14 bootstrapping)
- **Depends on**: Phase 14i (bootstrap orchestrator complete)
- **Required by**: Release Alpha
- **Research**: `docs/ai/decisions/022-nlu-architecture.md`

## Goal

Extend the existing GF-inspired grammar framework into a full four-tier hybrid NLU
pipeline capable of parsing arbitrary natural language into `AbsTree` representations.
The system must run entirely on a Mac Mini M2 (16 GB RAM), use only FLOSS components,
and require no cloud API calls. Human language enters and exits at the boundary; all
internal reasoning remains in VSA space.

The NLU does NOT reason — the symbolic engine handles reasoning. The NLU only translates
between human language and the internal `AbsTree` representation.

## Architecture Overview

```
Input utterance
     │
     ▼
┌─────────────────────────────────────────────┐
│ Tier 1: Extended Rule Parser (14j)          │  0 MB, <1ms
│ Negation, quantifiers, comparatives,        │  Handles ~70% of input
│ conditionals, modals, temporal (EN+multi)   │
└──────────────┬──────────────────────────────┘
               │ if Freeform or low confidence
               ▼
┌─────────────────────────────────────────────┐
│ Tier 2: Micro-ML NER + Intent (14k)        │  ~130 MB, ~5ms
│ DistilBERT multilingual NER via ONNX        │  Handles ~20% more
│ Entity boundaries, intent classification    │
└──────────────┬──────────────────────────────┘
               │ if still ambiguous
               ▼
┌─────────────────────────────────────────────┐
│ Tier 3: Small LLM Translator (14l)         │  ~1.1 GB, ~800ms
│ Qwen2.5-1.5B via llama-cpp-2               │  Handles remaining ~10%
│ GBNF-constrained AbsTree JSON output        │  Fallback only
└──────────────┬──────────────────────────────┘
               │ always
               ▼
┌─────────────────────────────────────────────┐
│ Tier 4: VSA Parse Ranker (14m)             │  0 MB extra, <1ms
│ Exemplar memory of successful parses        │  Self-improving over time
│ Uses existing HNSW + ItemMemory             │
└─────────────────────────────────────────────┘
```

## Memory Budget

| Component | Memory |
|-----------|--------|
| Tier 1: Rule parser | ~0 MB (code only) |
| Tier 2: DistilBERT NER (ONNX, quantized) | ~130 MB |
| Tier 3: Qwen2.5-1.5B (Q4_K_M GGUF) | ~1,100 MB |
| Tier 4: VSA exemplar memory | ~50 MB (shared with existing ItemMemory) |
| **Total NLU** | **~1,280 MB** |

Leaves ~6-8 GB on a 16 GB M2 for the engine, KG, storage tiers, and OS.

## Sub-phases

### 14j — Extended Rule Parser (~1,500 lines, ~3 days)

Extend the existing recursive descent parser in `src/grammar/parser.rs` with new
pattern recognizers. Zero dependencies, zero ML, sub-microsecond latency.

**New AbsTree variants**:
```rust
// Add to AbsTree enum in src/grammar/abs.rs:
Negation { inner: Box<AbsTree> },
Quantified { quantifier: Quantifier, scope: Box<AbsTree> },
Comparison {
    entity_a: Box<AbsTree>,
    entity_b: Box<AbsTree>,
    property: String,
    ordering: CompareOrd,
},
Conditional { condition: Box<AbsTree>, consequent: Box<AbsTree> },
Temporal { time_expr: TemporalExpr, inner: Box<AbsTree> },
RelativeClause { head: Box<AbsTree>, clause: Box<AbsTree> },
Modal { modality: Modality, inner: Box<AbsTree> },
```

**Supporting types**:
```rust
enum Quantifier { Universal, Existential, Most, None, Specific(u32) }
enum CompareOrd { GreaterThan, LessThan, Equal }
enum Modality { Want, Can, Should, Must, May }
enum TemporalExpr {
    Absolute(u64),           // Unix timestamp
    Relative(i64),           // Seconds from now
    Named(String),           // "tomorrow", "next week"
    Recurring(String),       // "every Monday"
}
```

**Pattern implementations**:

| Pattern | Trigger words (EN) | AbsTree output | LOC |
|---------|-------------------|----------------|-----|
| Negation | not, no, never, isn't, aren't, doesn't, don't | `Negation { inner }` | ~200 |
| Quantifiers | all, every, some, most, no, any | `Quantified { quantifier, scope }` | ~300 |
| Comparatives | bigger than, more X than, less X than, as X as | `Comparison` | ~250 |
| Conditionals | if, when, whenever, unless → split at comma/then | `Conditional` | ~200 |
| Temporal | tomorrow, yesterday, next week, in N days/hours | `Temporal` | ~300 |
| Modals | want to, can, should, must, may, need to | `Modal` | ~150 |

**Multilingual lexicon extensions** (all 5 languages):
- Negation: EN(not/no/never) RU(не/нет/никогда) FR(ne...pas/non/jamais) ES(no/nunca/jamás) AR(لا/ليس/لن)
- Quantifiers: EN(all/every/some) RU(все/каждый/некоторые) FR(tous/chaque/quelques) ES(todos/cada/algunos) AR(كل/بعض/أي)
- Modals: EN(want/can/should) RU(хотеть/мочь/должен) FR(vouloir/pouvoir/devoir) ES(querer/poder/deber) AR(يريد/يستطيع/يجب)
- Comparatives: EN(more/less/bigger) RU(больше/меньше) FR(plus/moins) ES(más/menos) AR(أكثر/أقل)
- Conditionals: EN(if/when/unless) RU(если/когда) FR(si/quand) ES(si/cuando) AR(إذا/عندما)

**Temporal crate**: Add `chrono-english` or `temps` crate for English temporal expression
parsing. Build rule-based temporal lexicons for RU/FR/ES/AR (temporal expressions are
finite and regular).

**VSA encoding**: Each new AbsTree variant gets a `to_vsa()` implementation using
role-filler binding — consistent with existing encoding pattern.

**Concrete grammar linearization**: Each new variant gets render methods in all 4
archetypes (formal, terse, narrative, rust-gen where applicable).

**Tests**: ~40 unit tests covering each pattern in all 5 languages, edge cases
(double negation, nested quantifiers, temporal + conditional combos).

### 14k — Micro-ML NER & Intent Classification (~800 lines, ~3 days)

Add a lightweight ML layer for entity recognition and intent classification using
ONNX Runtime. Invoked only when the rule parser returns Freeform or low confidence.

**Dependencies**:
```toml
[dependencies]
ort = { version = "2", optional = true }        # ONNX Runtime
tokenizers = { version = "0.21", optional = true } # HuggingFace tokenizer

[features]
nlu-ml = ["ort", "tokenizers"]
```

**NER model**: `Davlan/distilbert-base-multilingual-cased-ner-hrl`
- Covers EN, RU, FR, ES, AR (and 30+ more languages)
- ~130 MB ONNX quantized
- ~5ms inference per sentence on ARM
- Outputs BIO tags for PER, LOC, ORG, DATE entities

**Architecture**:
```rust
struct MicroMlLayer {
    ner_session: Option<ort::Session>,       // ONNX NER model
    tokenizer: Option<tokenizers::Tokenizer>, // WordPiece tokenizer
    intent_patterns: IntentPatterns,          // Rule-based intent (no model)
}

impl MicroMlLayer {
    /// Run NER on input, return entity spans with types
    fn extract_entities(&self, text: &str) -> Vec<EntitySpan>;

    /// Classify intent from entity spans + keyword patterns
    fn classify_intent(&self, text: &str, entities: &[EntitySpan]) -> IntentClass;

    /// Augment a Freeform parse with NER + intent info
    fn augment_parse(&self, text: &str, tokens: &[Token]) -> AugmentedParse;
}

struct EntitySpan {
    text: String,
    entity_type: NerEntityType,  // Person, Location, Organization, Date
    start: usize,
    end: usize,
    confidence: f32,
}

enum IntentClass {
    Assertion,   // "X is Y"
    Query,       // "What is X?"
    Command,     // "Do X"
    Goal,        // "Find X", "Explore Y"
    Temporal,    // Contains date/time references
    Social,      // Mentions people/organizations
    Unknown,
}
```

**Integration with parser pipeline**:
```
parser.parse(input)
  → if AbsTree::Freeform AND nlu-ml feature enabled:
      ml_layer.augment_parse(input, tokens)
        → NER entities fed back into entity resolution
        → Intent classification guides AbsTree construction
        → Attempt re-parse with augmented token stream
```

**Model management**:
- Models stored in `$AKH_DATA_DIR/models/` (not bundled in binary)
- `akh init --with-models` downloads ONNX models on first run
- Graceful degradation: if models not found, skip Tier 2 (fall through to Tier 3 or Freeform)

**Feature-gated**: Entire Tier 2 behind `--features nlu-ml`. Core parser works without it.

**Tests**: ~20 unit tests with mock ONNX session, entity span extraction, intent
classification accuracy on test sentences.

### 14l — Small LLM Translator with Constrained Decoding (~1,000 lines, ~5 days)

Integrate a quantized small language model as the final fallback for complex natural
language that the rule parser and micro-ML cannot handle. The model translates NL to
AbsTree JSON using GBNF-constrained decoding — it can ONLY output valid structures.

**Dependencies**:
```toml
[dependencies]
llama-cpp-2 = { version = "0.1", optional = true }  # llama.cpp bindings

[features]
nlu-llm = ["llama-cpp-2"]
```

**Model**: Qwen2.5-1.5B-Instruct (Q4_K_M quantization)
- Apache 2.0 license (fully FLOSS)
- ~1.1 GB GGUF file
- ~1.5 GB RAM at runtime
- ~40-80 tokens/second on M2 (Apple Silicon Metal via llama.cpp)
- Native multilingual: EN, RU, FR, ES, AR all supported in training data

**GBNF grammar** (constrains model output to valid AbsTree JSON):
```
root   ::= object
object ::= "{" ws pair ws "}"
pair   ::= key ws ":" ws value

key    ::= "\"Entity\"" | "\"Relation\"" | "\"Triple\"" | "\"Query\"" |
           "\"Command\"" | "\"Goal\"" | "\"Negation\"" | "\"Quantified\"" |
           "\"Comparison\"" | "\"Conditional\"" | "\"Temporal\"" |
           "\"Modal\"" | "\"Conjunction\"" | "\"RelativeClause\"" |
           "\"Similarity\""

value  ::= string | object | array | number
string ::= "\"" [^"\\]* "\""
array  ::= "[" ws (value ws ("," ws value ws)*)? "]"
number ::= [0-9]+ ("." [0-9]+)?
ws     ::= [ \t\n]*
```

(Full grammar would enumerate all AbsTree variant shapes precisely.)

**Architecture**:
```rust
struct LlmTranslator {
    model: Option<LlamaModel>,
    grammar: String,            // GBNF grammar text
    system_prompt: String,      // Instructions for NL→AbsTree translation
    max_tokens: u32,            // Cap output length
}

impl LlmTranslator {
    /// Translate natural language to AbsTree via constrained LLM generation
    fn translate(&self, input: &str) -> NluResult<AbsTree>;

    /// Generate AbsTree JSON with GBNF constraint
    fn generate_constrained(&self, prompt: &str) -> NluResult<String>;

    /// Deserialize JSON into AbsTree
    fn parse_abstree_json(json: &str) -> NluResult<AbsTree>;
}
```

**System prompt** (baked into the translator, not user-visible):
```
You are a natural language parser. Convert the input sentence into a structured
JSON representation. Output ONLY valid JSON matching the AbsTree schema.

Examples:
Input: "Dogs are not cats"
Output: {"Negation": {"inner": {"Triple": {"subject": {"Entity": "dog"}, "relation": "is-a", "object": {"Entity": "cat"}}}}}

Input: "If it rains tomorrow, cancel the meeting"
Output: {"Conditional": {"condition": {"Temporal": {"time_expr": {"Named": "tomorrow"}, "inner": {"Entity": "rain"}}}, "consequent": {"Command": {"verb": "cancel", "args": ["meeting"]}}}}
```

**Integration**:
```
parser.parse(input)
  → Freeform
    → ml_layer.augment_parse(input)
      → still ambiguous
        → llm_translator.translate(input)
          → valid AbsTree (guaranteed by GBNF constraint)
```

**Self-training pipeline** (Phase 14m integration):
- Every successful LLM translation is stored as a (input, AbsTree) pair
- These pairs become training data for fine-tuning the model on akh-medu's domain
- Over time, the model learns the system's specific patterns and entities
- Fine-tuning via LoRA on the base model (separate tooling, not in the main binary)

**Model management**:
- GGUF model stored in `$AKH_DATA_DIR/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- `akh init --with-llm` downloads the model (~1.1 GB)
- Graceful degradation: if model not found, Tier 3 is skipped (Freeform persists)
- Warm-up: model loaded lazily on first Tier 3 invocation, stays resident

**Feature-gated**: Behind `--features nlu-llm`. Core parser + micro-ML work without it.

**Tests**: ~15 unit tests with mock model, GBNF grammar validation, JSON→AbsTree
deserialization, edge cases (empty input, very long input, non-target language).

### 14m — VSA Parse Ranker & Self-Improving Loop (~600 lines, ~3 days)

Build a parse disambiguation system using the existing VSA infrastructure. Every
successful parse is encoded as a hypervector and stored in an exemplar memory.
Future ambiguous parses are ranked by similarity to known-good parses.

**Architecture**:
```rust
struct ParseRanker {
    exemplar_memory: ItemMemory,   // HNSW index of successful parse vectors
    exemplar_count: usize,
    min_similarity: f32,           // Threshold for "confident" match
}

impl ParseRanker {
    /// Record a successful parse as an exemplar
    fn record_success(&mut self, ops: &VsaOps, tree: &AbsTree, input: &str);

    /// Rank candidate parses by similarity to exemplar memory
    fn rank_candidates(
        &self,
        ops: &VsaOps,
        candidates: &[AbsTree],
    ) -> Vec<(AbsTree, f32)>;

    /// Check if input pattern is similar to known successful parses
    fn has_similar_exemplar(&self, ops: &VsaOps, input_vec: &HyperVec) -> Option<f32>;
}
```

**Encoding strategy**: Use existing `AbsTree::to_vsa()` which encodes the parse tree
structure as a compositional hypervector with role-filler binding. The encoding captures:
- Node types (Entity, Relation, Triple, Negation, etc.)
- Structural depth and branching
- Semantic content (entity/relation labels via symbol registry lookup)

**Self-improvement loop**:
1. Input parsed successfully by Tier 1/2/3 → `record_success()`
2. Next similar input → Tier 4 recognizes pattern → guides Tier 1 disambiguation
3. Over time, Tier 1 handles more cases as exemplar memory grows
4. Eventually, Tier 3 (expensive LLM) is rarely needed

**Persistence**: Exemplar memory persisted via `put_meta`/`get_meta` on the durable
store, same pattern as spam classifier and triage prototypes.

**Integration with parser pipeline**:
```
Final step (always runs):
  candidates = [tier1_result, tier2_result, tier3_result].filter(|c| c.is_some())
  if candidates.len() > 1:
      ranked = parse_ranker.rank_candidates(candidates)
      return ranked[0]  // highest similarity to exemplar memory
  else:
      result = candidates[0]
      parse_ranker.record_success(result)
      return result
```

**DerivationKind**: `NluParsed` (new tag) — records which tier produced the final
parse, the confidence, and the exemplar similarity score.

**Tests**: ~15 unit tests covering exemplar recording, ranking, persistence roundtrip,
self-improvement over repeated similar inputs.

## Files to Create/Modify

| File | Change |
|------|--------|
| `src/grammar/abs.rs` | Add 7 new AbsTree variants + supporting enums |
| `src/grammar/parser.rs` | Extend priority cascade with 6 new pattern recognizers |
| `src/grammar/lexer.rs` | Add negation/quantifier/modal/comparative/conditional/temporal words to all 5 lexicons |
| `src/grammar/formal.rs` | Linearize new variants in formal archetype |
| `src/grammar/terse.rs` | Linearize new variants in terse archetype |
| `src/grammar/narrative.rs` | Linearize new variants in narrative archetype |
| `src/grammar/concrete.rs` | Update ConcreteGrammar trait if needed |
| `src/nlu/mod.rs` | NEW — NLU pipeline orchestrator (tier routing) |
| `src/nlu/error.rs` | NEW — NluError miette diagnostic enum |
| `src/nlu/micro_ml.rs` | NEW — ONNX NER + intent classification |
| `src/nlu/llm_translator.rs` | NEW — llama-cpp-2 integration + GBNF grammar |
| `src/nlu/parse_ranker.rs` | NEW — VSA exemplar memory + ranking |
| `src/nlu/abstree_gbnf.rs` | NEW — GBNF grammar definition for AbsTree JSON |
| `src/main.rs` | Add NLU-related CLI commands, wire features |
| `src/engine.rs` | Integrate NLU pipeline into query/chat paths |
| `Cargo.toml` | Add `ort`, `tokenizers`, `llama-cpp-2`, `chrono-english` (feature-gated) |
| `src/provenance.rs` | Add `DerivationKind::NluParsed` variant |
| `src/agent/conversation.rs` | Route through NLU pipeline instead of raw parser |

## New Feature Flags

```toml
[features]
default = []
nlu-ml = ["ort", "tokenizers"]              # Tier 2: ONNX NER
nlu-llm = ["llama-cpp-2"]                   # Tier 3: Small LLM
nlu-full = ["nlu-ml", "nlu-llm"]            # Both ML tiers
```

Tier 1 (rule parser extensions) and Tier 4 (VSA ranker) are always available — no
feature gate, no extra dependencies.

## Estimated Scope

| Sub-phase | Lines | New Files | Dependencies | Days |
|-----------|-------|-----------|-------------|------|
| 14j Rule Parser Extensions | ~1,500 | 0 (extends existing) | `chrono-english` | 3 |
| 14k Micro-ML NER + Intent | ~800 | 3 (`nlu/`) | `ort`, `tokenizers` | 3 |
| 14l LLM Translator | ~1,000 | 2 (`nlu/`) | `llama-cpp-2` | 5 |
| 14m VSA Parse Ranker | ~600 | 1 (`nlu/`) | none | 3 |
| **Total** | **~3,900** | **6 new files** | **3 optional crates** | **~14 days** |

## Success Criteria

1. "Dogs are NOT cats" → `Negation { Triple(dog, is-a, cat) }` (Tier 1)
2. "All dogs are mammals" → `Quantified { Universal, Triple(dog, is-a, mammal) }` (Tier 1)
3. "If it rains, cancel the meeting" → `Conditional { ... }` (Tier 1)
4. "I want to schedule a meeting tomorrow" → `Modal { Want, Temporal { ... } }` (Tier 1)
5. "The dogs that bark loudly are not all mammals" → correct nested AbsTree (Tier 3 fallback)
6. Multilingual: same patterns work in all 5 languages (EN, RU, FR, ES, AR)
7. Latency: <1ms for 70% of input, <10ms for 90%, <1.5s for 100%
8. Memory: NLU stack fits in <1.5 GB total
9. Self-improvement: repeated similar inputs increasingly handled by Tier 1 (not Tier 3)
10. Feature-gated: `cargo build` (no features) builds Tier 1+4 only, no ML deps
11. `cargo test` passes all existing + 90 new NLU tests

## Non-Goals

- Full syntactic parsing (we don't need parse trees, we need AbsTree)
- Training custom models from scratch (use pre-trained, fine-tune at most)
- Real-time speech recognition (text input only)
- 100% accuracy on adversarial input (graceful Freeform fallback is acceptable)
- Competing with frontier LLMs on open-ended NLU (we only need structured extraction)
