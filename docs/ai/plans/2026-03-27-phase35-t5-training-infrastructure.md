# Phase 35 — T5 Training Infrastructure

> Date: 2026-03-27
> Status: Planned
> Phase: 35
> Depends on: None (offline, external to akh-medu runtime)
> Produces: T5 NLU, T5 NLG, T5 Knowledge model artifacts
> ADR: [045-t5-training-infrastructure](../decisions/045-t5-training-infrastructure.md)

## Motivation

Akh-medu ships three purpose-built T5 models. Each requires a fine-tuning
pipeline with specific training data. This phase builds the offline training
infrastructure that produces these models before each release.

## Architecture

```
akh-medu-models/              ← Separate project (not in akh-medu binary)
├── data/
│   ├── wikidata/             ← Downloaded + filtered Wikidata dump
│   ├── conceptnet/           ← Downloaded ConceptNet assertions
│   ├── webnlg/               ← WebNLG benchmark data
│   └── abstree/              ← Synthetic AbsTree corpus
├── pipelines/
│   ├── knowledge/            ← T5 Knowledge training pipeline
│   ├── nlg/                  ← T5 NLG training pipeline
│   ├── nlu/                  ← T5 NLU training pipeline
│   └── shared/               ← Common data loading, evaluation, export
├── configs/                  ← Training hyperparameters per model
├── eval/                     ← Evaluation scripts + test sets
└── export/                   ← GGUF quantization + packaging
```

## Sub-phases

### 35a — Data Acquisition & Processing (~800 lines)

**Wikidata pipeline**:
```rust
pub struct WikidataProcessor {
    pub min_sitelinks: usize,          // Notability filter (default: 5)
    pub max_entities: usize,           // Cap (default: 5_000_000)
    pub predicate_map: HashMap<String, String>,  // P31 → "is-a", P279 → "subclass-of"
}
```

Steps:
1. Download latest `wikidata-YYYYMMDD-all.json.bz2` (or use streaming API)
2. Filter entities: ≥ min_sitelinks, has English label
3. Extract statements → (subject_label, predicate_label, object_label) triples
4. Normalize predicates via predicate_map (Wikidata PIDs → akh-medu conventions)
5. Group by entity → training samples: `{concept, domain_tags, triples[]}`
6. Output: JSONL file, one entity per line

**Predicate normalization** (Wikidata PID → akh-medu):

| Wikidata PID | Akh-medu Predicate | Example |
|---|---|---|
| P31 | is-a | "Einstein is-a Physicist" |
| P279 | subclass-of | "Mammal subclass-of Vertebrate" |
| P361 | part-of | "Engine part-of Car" |
| P527 | has-part | "Car has-part Engine" |
| P1376 | capital-of | "Berlin capital-of Germany" |
| P19 | born-in | "Einstein born-in Ulm" |
| P69 | educated-at | "Einstein educated-at ETH_Zurich" |
| P106 | occupation | "Einstein occupation Physicist" |
| P1441 | appears-in | entity appears in creative work |
| ... | ... | ~100 most common predicates mapped |

**ConceptNet pipeline**:
```rust
pub struct ConceptNetProcessor {
    pub min_weight: f32,               // Assertion confidence filter (default: 1.0)
    pub languages: Vec<String>,        // ["en", "ru", "fr", "es", "ar"]
    pub relation_map: HashMap<String, String>,  // IsA → "is-a", PartOf → "part-of"
}
```

Steps:
1. Download ConceptNet CSV dump (~2 GB compressed)
2. Filter by language + min_weight
3. Normalize relations via relation_map
4. Deduplicate against Wikidata triples (same subject+predicate+object → keep higher weight)
5. Output: JSONL, same format as Wikidata output

**ConceptNet relation mapping**:

| ConceptNet Relation | Akh-medu Predicate | Type |
|---|---|---|
| IsA | is-a | Taxonomy |
| PartOf | part-of | Meronymy |
| HasA | has | Possession |
| UsedFor | used-for | Function |
| CapableOf | capable-of | Capability |
| Causes | causes | Causal |
| HasProperty | has-property | Attribution |
| MadeOf | made-of | Composition |
| ReceivesAction | receives-action | Passive |
| AtLocation | located-at | Spatial |
| Desires | desires | Intentional |
| CreatedBy | created-by | Origin |

**Merge strategy**: Wikidata provides encyclopedic facts (born-in, capital-of).
ConceptNet provides common-sense relations (used-for, capable-of, causes).
Together they cover both dimensions. The merged dataset annotates each triple
with its source for provenance.

**WebNLG data**: Already structured as (triple_set, reference_text) pairs.
Download from the WebNLG challenge repository. No processing needed beyond
format conversion to our JSONL.

**AbsTree corpus generation**:
1. Write ~200 hand-curated examples covering all 12 AbsTree variants
2. Generate ~2000-5000 synthetic examples by running Qwen on diverse inputs
   (this is a ONE-TIME use of Qwen — for creating training data, not for production)
3. Validate: each generated AbsTree must parse back to valid JSON
4. Split: 80% train, 10% validation, 10% test

### 35b — T5 Knowledge Training Pipeline (~600 lines)

Fine-tune T5-Base to generate triples from concept names.

**Training format**:
```
Input:  "concept: Albert Einstein | domain: physics, biography"
Output: '[{"s":"Einstein","p":"is-a","o":"Physicist"},
          {"s":"Einstein","p":"born-in","o":"Ulm"},
          {"s":"Einstein","p":"developed","o":"General Relativity"}]'
```

**Training configuration**:
```rust
pub struct KnowledgeTrainConfig {
    pub base_model: String,          // "google-t5/t5-base"
    pub learning_rate: f32,          // 1e-4
    pub batch_size: usize,           // 16
    pub epochs: usize,               // 5-10
    pub max_input_length: usize,     // 64 tokens
    pub max_output_length: usize,    // 256 tokens
    pub warmup_steps: usize,         // 500
    pub weight_decay: f32,           // 0.01
}
```

**Evaluation metrics**:
- **Triple F1**: precision/recall of generated triples vs ground truth
- **Predicate accuracy**: does the model use correct predicates?
- **Novel entity detection**: does the model generate entities not in the input?
  (hallucination metric — lower is better)
- **Domain coverage**: per-domain F1 (physics, biology, geography, etc.)

**Target**: >70% Triple F1 on held-out test set, <5% hallucinated entities.

### 35c — T5 NLG Training Pipeline (~400 lines)

Fine-tune T5-Base on WebNLG for data-to-text generation.

**Training format** (standard WebNLG):
```
Input:  "Albert_Einstein | birthPlace | Ulm <sep> Albert_Einstein | field | Physics"
Output: "Albert Einstein was born in Ulm and worked in the field of physics."
```

**Note**: `webnlg/en-t5base` already exists pre-fine-tuned on HuggingFace.
This pipeline allows us to:
- Retrain from scratch if needed (reproducibility)
- Add akh-medu-specific predicate examples to the training data
- Train mT5-Small for multilingual NLG
- Combine WebNLG with ConceptNet verbalization examples

**Evaluation**: BLEU, METEOR, BERTScore on WebNLG test set.
**Target**: >52 BLEU (matching published T5-Base WebNLG results).

### 35d — T5 NLU Training Pipeline (~400 lines)

Fine-tune T5-Base on AbsTree extraction (distillation from Qwen).

**Training format**:
```
Input:  "Dogs are domesticated mammals related to wolves"
Output: '{"variant":"Triple","subject":"Dog","predicate":"is-a","object":"Mammal"}'
```

**Note**: The synthetic corpus (35a) provides the initial dataset. After
deployment, Phase 27f collects real user conversation pairs that feed back
into this pipeline for future retraining.

**Evaluation**: Exact match on AbsTree structure (variant + key fields).
**Target**: >85% exact match on held-out test set, >90% variant-correct.

### 35e — GGUF Export & Packaging (~300 lines)

Convert trained models to GGUF for Candle `quantized_t5`.

```bash
# For each trained model:
# 1. Export to safetensors (Burn native or PyTorch conversion)
# 2. Quantize to GGUF via Candle tensor-tools
cargo run --release --example tensor-tools -- quantize \
  --quantization q4k \
  model.safetensors \
  --out-file model-q4k.gguf

# 3. Package for distribution
# Output files:
#   t5-nlu-abstree.gguf      (~140-250 MB)
#   t5-nlg-webnlg.gguf       (~140-250 MB)
#   t5-kg-wikidata.gguf       (~140-250 MB)
#   tokenizer.json            (~2 MB, shared)
```

**Quantization levels**:
- Q4_K_M: smallest, ~60-65% of Q6_K size, slight quality loss
- Q6_K: default, good balance (~250 MB per model)
- Ship Q6_K as default, Q4_K_M as "lite" option for constrained devices

**Automation**: CI/CD script that runs the full pipeline:
1. Download latest data sources (Wikidata, ConceptNet, WebNLG)
2. Process data (35a)
3. Train all three models (35b, 35c, 35d)
4. Evaluate (must meet quality targets)
5. Export GGUF (35e)
6. Upload to model hosting (HuggingFace or self-hosted)
7. Update `akh setup models` manifest

### 35f — Per-Release Retraining (~200 lines)

Automation to retrain models with:
- Latest Wikidata dump (new entities, updated facts)
- Accumulated user feedback from Phase 27f (anonymized, opt-in)
- Improved AbsTree corpus (from production NLU successes)

**Schedule**: Monthly or per minor release.
**Versioning**: Models tagged with training date + data snapshot version.
**Rollback**: `akh setup models --version YYYY-MM-DD` to pin a specific version.

## Estimated Effort

| Sub-phase | Lines | Complexity | Dependencies |
|---|---|---|---|
| 35a — Data acquisition + processing | ~800 | Medium | None (downloads) |
| 35b — T5 Knowledge training | ~600 | High | 35a, Burn or PyTorch |
| 35c — T5 NLG training | ~400 | Medium | 35a (WebNLG) |
| 35d — T5 NLU training | ~400 | Medium | 35a (AbsTree corpus) |
| 35e — GGUF export + packaging | ~300 | Low | Candle tensor-tools |
| 35f — Per-release automation | ~200 | Low | All above |
| **Total** | **~2,700** | | |

## Implementation Strategy

**Phase 1 (prototype)**: Use PyTorch + HuggingFace Transformers for initial
training. Fast iteration, proven tooling, many examples available. Produces
safetensors that Candle can quantize to GGUF.

**Phase 2 (production)**: Port training to Burn for a fully Rust-native
pipeline. Burn's `Learner` API + Wgpu backend can train T5-Base on M2.
This aligns with Phase 27's Burn infrastructure.

PyTorch-first is pragmatic: get models shipping quickly, migrate training
to Burn once the in-situ training loop (Phase 27) is built and proven.

## Model Stack Summary

| Model | Trained on | Size (Q6_K) | Ships with release |
|---|---|---|---|
| `t5-nlu-abstree.gguf` | Synthetic AbsTree corpus | ~250 MB | Yes |
| `t5-nlg-webnlg.gguf` | WebNLG + ConceptNet verbalization | ~250 MB | Yes |
| `t5-kg-wikidata.gguf` | Wikidata + ConceptNet triples | ~250 MB | Yes |
| `tokenizer.json` | T5 tokenizer (shared) | ~2 MB | Yes |
| **Total shipped** | | **~752 MB** | |

Compare: Qwen2.5-1.5B alone = 1.1 GB, mediocre at all three tasks.
Three purpose-built experts < one large generalist.

## Relationship to Other Phases

- **Phase 26b** (Candle): Provides the `quantized_t5` runtime that loads these models
- **Phase 27** (Burn in-situ): Refines the shipped models per-user over time
- **Phase 27f** (data collection): Production data feeds back into 35f retraining
- **Phase 32** (knowledge extraction): Uses the T5 Knowledge model
- **Phase 33** (verbalization): Uses the T5 NLG + NLU models
