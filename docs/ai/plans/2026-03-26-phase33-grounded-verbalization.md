# Phase 33 — Grounded Verbalization

> Date: 2026-03-26
> Status: Planned
> Phase: 33
> Depends on: Phase 26b (Candle backend)
> Enhances: Phase 34 (communicative intelligence)
> ADR: [043-grounded-verbalization](../decisions/043-grounded-verbalization.md)

## Motivation

The verbalization pipeline produces robotic output via grammar templates.
Previous attempts to use small LLMs for "enrichment" failed (passthrough or
hallucination). The solution is data-to-text NLG with purpose-built models
(T5/mT5) and constrained decoding — a fundamentally different task framing.

## Sub-phases

### 33a — Grammar Aggregation Rules (~500 lines)

Improve the existing grammar framework with rule-based sentence combining.
No model needed — pure AbsTree transformations before linearization.

**New module**: `src/grammar/aggregate.rs`

**Aggregation rules**:

```rust
pub fn aggregate_triples(triples: &[AbsTree], ctx: &GrammarContext) -> Vec<AbsTree> {
    let grouped = group_by_subject(triples);
    let mut result = Vec::new();

    for (subject, group) in grouped {
        let aggregated = match group.len() {
            1 => group[0].clone(),
            _ => apply_rules(&subject, &group, ctx),
        };
        result.push(aggregated);
    }

    apply_discourse_ordering(&mut result, ctx);
    apply_pronoun_substitution(&mut result);
    result
}
```

**Rule implementations**:

| Rule | Before | After |
|---|---|---|
| **Subject merge** | "X is-a Y. X has Z." | "X, a Y, has Z." |
| **Predicate enumeration** | "X has A. X has B. X has C." | "X has A, B, and C." |
| **Relative clause** | "X is-a Y. X is in Z." | "X, which is in Z, is a Y." |
| **Pronoun substitution** | "X is-a Y. X has Z." | "X is a Y. It has Z." |
| **Discourse ordering** | Random triple order | is-a first, then properties, then relations |
| **Connective insertion** | No transitions | "Additionally", "In particular", based on predicate type |

**Per-language rules**: Each language has its own aggregation patterns.
English: relative "which/that". Russian: relative "который". Arabic: different
clause joining patterns. Store as language-specific rule sets in the Lexicon.

### 33b — T5/mT5 NLG Model Integration (~600 lines)

**New module**: `src/nlu/t5_nlg.rs`

Load T5-Base or mT5-Small via Candle's `quantized_t5` for data-to-text generation.

```rust
pub struct T5NlgBackend {
    model: T5Model,              // Candle quantized T5
    tokenizer: Tokenizer,
    device: Device,
    config: T5NlgConfig,
}

pub struct T5NlgConfig {
    pub max_output_tokens: usize,      // Default: 128
    pub num_beams: usize,              // Beam search width (default: 4)
    pub enable_constraints: bool,      // NeuroLogic decoding (default: true)
    pub language: Language,            // For mT5 language prefix
}

pub struct NlgInput {
    pub triples: Vec<LabelTriple>,     // Structured input
    pub style_prefix: Option<String>,  // From psyche (Phase 34)
}

pub struct NlgOutput {
    pub text: String,
    pub entities_covered: Vec<String>, // Entity audit result
    pub entities_missing: Vec<String>, // Entities that should appear but don't
    pub faithful: bool,                // All input entities present, no extras
}
```

**Triple linearization for T5 input**:
```rust
fn linearize_triples_for_t5(triples: &[LabelTriple]) -> String {
    // WebNLG format: "subject | predicate | object"
    // Multiple triples separated by " <sep> "
    triples.iter()
        .map(|t| format!("{} | {} | {}", t.subject, t.predicate, t.object))
        .collect::<Vec<_>>()
        .join(" <sep> ")
}
```

This input format matches what T5 was trained on for WebNLG — critical for
good results without fine-tuning.

**Model loading**:
- Candle `quantized_t5` loads GGUF files natively
- Metal acceleration on M2 via `Device::metal_if_available()`
- Models stored in `data/models/t5-nlg.gguf` (downloaded via `akh setup models`)

**Multilingual (mT5)**:
- Language prefix: `"generate {lang}: "` prepended to input
- Single model handles all 5 target languages
- Language detected from engine's active grammar or explicitly specified

### 33c — NeuroLogic Constrained Decoding (~400 lines)

**New module**: `src/nlu/neurologic.rs`

Implement NeuroLogic-style constraints as a Candle LogitsProcessor:

```rust
pub struct NeurologicConstraints {
    /// Entity tokens that MUST appear in the output
    required_entities: Vec<Vec<u32>>,  // Token ID sequences per entity
    /// Entity tokens that MUST NOT appear in the output
    forbidden_entities: Vec<Vec<u32>>,
    /// Track which required entities have been satisfied
    satisfied: Vec<bool>,
    /// Boost factor for unsatisfied entity tokens (increases as generation progresses)
    boost_schedule: BoostSchedule,
}

pub enum BoostSchedule {
    /// Linear increase: boost = base + (step/max_steps) * scale
    Linear { base: f32, scale: f32 },
    /// Exponential: boost aggressively near end of generation
    Exponential { base: f32, growth: f32 },
}

impl NeurologicConstraints {
    pub fn from_triples(
        triples: &[LabelTriple],
        tokenizer: &Tokenizer,
        engine: &Engine,
    ) -> Self {
        // Extract all entity names from triples (subjects + objects)
        let required: Vec<String> = triples.iter()
            .flat_map(|t| [&t.subject, &t.object])
            .collect::<HashSet<_>>()
            .into_iter()
            .cloned()
            .collect();

        // Tokenize each entity name
        let required_entities = required.iter()
            .map(|e| tokenizer.encode(e).ids)
            .collect();

        // Forbidden: all entity names in the engine's symbol registry
        // that are NOT in the input triples
        // (This is the hallucination firewall)
        let input_entities: HashSet<&str> = required.iter().map(|s| s.as_str()).collect();
        let forbidden = engine.symbol_registry()
            .all_labels()
            .filter(|label| !input_entities.contains(label.as_str()))
            .map(|e| tokenizer.encode(&e).ids)
            .collect();

        Self {
            required_entities,
            forbidden_entities: forbidden,
            satisfied: vec![false; required.len()],
            boost_schedule: BoostSchedule::Linear { base: 2.0, scale: 5.0 },
        }
    }

    /// Apply constraints to logits at each decoding step
    pub fn process_logits(&mut self, logits: &mut Tensor, step: usize, max_steps: usize) {
        // Boost unsatisfied required entities
        for (i, entity_tokens) in self.required_entities.iter().enumerate() {
            if !self.satisfied[i] {
                let boost = self.boost_schedule.compute(step, max_steps);
                // Boost first token of entity sequence
                logits.add_scalar_at(entity_tokens[0], boost);
            }
        }

        // Suppress forbidden entities
        for entity_tokens in &self.forbidden_entities {
            logits.add_scalar_at(entity_tokens[0], -100.0); // Effectively zero probability
        }
    }

    /// Check if a generated token satisfies any pending entity constraint
    pub fn update_satisfied(&mut self, generated_token: u32) { ... }
}
```

**Post-generation entity audit**:
```rust
pub fn audit_faithfulness(output: &str, input_triples: &[LabelTriple]) -> FaithfulnessAudit {
    let input_entities: HashSet<String> = /* extract from triples */;
    let output_entities: HashSet<String> = /* NER on output text */;

    FaithfulnessAudit {
        covered: input_entities.intersection(&output_entities),
        missing: input_entities.difference(&output_entities),
        hallucinated: output_entities.difference(&input_entities),
        faithful: hallucinated.is_empty(),
    }
}
```

If audit fails (hallucinated entities detected), fall back to grammar
aggregation (33a). Never serve unfaithful output.

### 33d — Verbalization Router (~300 lines)

Extend `n akh` (Phase 28) with output-side routing:

```rust
pub enum VerbalizationTier {
    /// 1-2 triples: grammar template (existing, improved)
    Template,
    /// 3-5 triples: grammar + aggregation rules (33a)
    Aggregated,
    /// 6+ triples: T5/mT5 data-to-text with constraints (33b+33c)
    NeuralNlg,
}

pub fn select_verbalization_tier(
    triple_count: usize,
    t5_available: bool,
) -> VerbalizationTier {
    match (triple_count, t5_available) {
        (0..=2, _) => VerbalizationTier::Template,
        (3..=5, _) => VerbalizationTier::Aggregated,
        (6.., true) => VerbalizationTier::NeuralNlg,
        (6.., false) => VerbalizationTier::Aggregated, // Graceful fallback
    }
}
```

### 33e — T5 for NLU (Qwen replacement path) (~400 lines)

After Phase 27 fine-tunes T5 on AbsTree extraction, replace Qwen for NLU:

**New module**: `src/nlu/t5_nlu.rs`

```rust
pub struct T5NluBackend {
    model: T5Model,               // Same Candle quantized T5
    tokenizer: Tokenizer,
    device: Device,
}

impl T5NluBackend {
    /// Parse natural language into AbsTree JSON
    pub fn parse(&self, input: &str) -> NluResult<AbsTree> {
        let prompt = format!("parse to abstree: {}", input);
        let output = self.model.generate(&prompt, max_tokens=256)?;
        serde_json::from_str(&output)?
    }
}
```

**Training data** (Phase 27):
- Input: natural language sentences from conversation history
- Target: AbsTree JSON produced by current NLU pipeline (Tier 1-3)
- This is **distillation**: T5 learns to do what the cascade does, in one step

**Migration path**:
1. Fine-tune T5 on accumulated NLU examples (Phase 27 Burn loop)
2. Evaluate: T5 parse accuracy vs current cascade
3. If competitive (>90% agreement): replace Qwen as NLU Tier 3
4. Qwen transitions to on-demand bootstrap oracle only
5. ~1.3 GB RAM freed during normal operation

## Estimated Effort

| Sub-phase | Lines | Complexity | Dependencies |
|---|---|---|---|
| 33a — Grammar aggregation | ~500 | Medium | None |
| 33b — T5/mT5 NLG backend | ~600 | High | Phase 26b (Candle) |
| 33c — NeuroLogic constraints | ~400 | Medium | 33b |
| 33d — Verbalization router | ~300 | Low | Phase 28 (n akh) |
| 33e — T5 NLU (Qwen replacement) | ~400 | High | Phase 27 (Burn training) |
| **Total** | **~2,200** | | |

## New Dependencies

```toml
[features]
nlu-t5 = ["candle-backend"]  # T5/mT5 for NLG + eventual NLU
```

## Model Files

| Model | Purpose | Size (GGUF Q4) | Languages |
|---|---|---|---|
| T5-Base (WebNLG fine-tuned) | English NLG | ~70 MB | EN |
| mT5-Small (WebNLG fine-tuned) | Multilingual NLG | ~100 MB | EN, RU, FR, ES, AR |
| T5-Base (AbsTree fine-tuned) | NLU (Phase 33e) | ~70 MB | EN |

Downloaded via `akh setup models`. Graceful degradation if absent.
