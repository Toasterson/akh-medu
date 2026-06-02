# ADR 043 — Grounded Verbalization & T5 Migration

> Date: 2026-03-26
> Status: Proposed
> Phase: 33
> Depends on: Phase 26b (Candle backend)
> Enhances: Phase 34 (communicative intelligence), Phase 14m (NLU)

## Context

The current verbalization pipeline uses grammar templates with ~15 predicate
mappings. Output feels robotic: "Dogs are mammals. Dogs have fur. Dogs are
similar to wolves." The LLM (Qwen2.5-1.5B) is used only for short dialogue
acts, not knowledge verbalization.

A previous attempt to use the small LLM for "enrichment" (rewriting grammar
output to be more natural) failed: the model either passed through the text
unchanged or hallucinated new facts. This is a fundamental limitation of
framing the task as "rewrite prose."

Research shows that **data-to-text NLG** (generate text directly from structured
triples) is a solved problem for small models. T5-Base (220M params) achieves
55+ BLEU on the WebNLG benchmark — outperforming general-purpose 1.5B models at
this specific task while being 7x smaller.

## Decision

### 1. Replace "enrichment" with data-to-text NLG

The task is NOT "rewrite this grammar output." The task IS "generate natural
prose from these RDF triples." Different framing, different model, different
results.

**Previous approach (failed)**:
```
Grammar: "Dogs are mammals. Dogs have fur." → LLM: "rewrite" → hallucination
```

**New approach**:
```
Triples: [(Dog, is-a, Mammal), (Dog, has, Fur)] → T5 NLG: "Dogs are furry mammals."
```

NeuroLogic constrained decoding ensures the T5 model physically cannot generate
entities not present in the input triples. Hallucination is structurally prevented.

### 2. T5/mT5 as the NLG model (not Qwen)

| | Qwen2.5-1.5B | T5-Base (WebNLG) | mT5-Small |
|---|---|---|---|
| Purpose | General chat | Data-to-text NLG | Multilingual data-to-text |
| Params | 1.5B | 220M | 300M |
| GGUF Q4 size | ~1.1 GB | ~70 MB | ~100 MB |
| WebNLG BLEU | Not trained | 55.2 | ~50 (with fine-tuning) |
| Languages | Many | English | 101 (all 5 targets) |
| Faithfulness | Hallucination-prone | High with constraints | High with constraints |

T5 is purpose-built for this. Qwen is not.

### 3. Qwen role change: always-loaded → on-demand bootstrap oracle

With T5 handling both NLG (this phase) and eventually NLU (post-Phase 27
fine-tuning), Qwen's role shrinks to:

- **Phase 32**: Knowledge elicitation during bootstrap (needs world knowledge)
- **Transition period**: NLU Tier 3 until T5 is fine-tuned for AbsTree extraction

Qwen is loaded on-demand during bootstrap, not kept resident in memory.

**Memory budget change**:
- Current: Qwen always loaded = ~1.5 GB RAM
- New: T5/mT5 resident (~100-200 MB) + Qwen on-demand (~1.1 GB during bootstrap)
- Normal operation saves ~1.3 GB RAM

### 4. Tiered verbalization (not all responses need a model)

| Response complexity | Strategy | Model | Latency |
|---|---|---|---|
| 1-2 triples | Grammar template (improved) | None | <1ms |
| 3-5 triples | Grammar + aggregation rules | None | <1ms |
| 6+ triples | T5/mT5 data-to-text + constraints | ~70-100MB | ~100-200ms |

The n akh router (Phase 28) decides which tier to use.

### 5. NeuroLogic constrained decoding prevents hallucination

Implement as a Rust-native LogitsProcessor in Candle's generation loop:

1. Before decoding: extract all entity tokens from input triples
2. Maintain checklist of unsatisfied entity constraints
3. At each step: boost logits for tokens beginning unsatisfied entities
4. Suppress logits for tokens starting entities NOT in input
5. Post-generation: entity audit (reject if output mentions unknown entities)

This makes faithfulness a **structural guarantee**, not a hope.

## Consequences

- New feature: `nlu-t5` alongside existing `nlu-llm` (Qwen)
- T5/mT5 loaded via Candle's `quantized_t5` module (pure Rust GGUF)
- Grammar framework improved with aggregation rules (no model dependency)
- Qwen transitions from always-loaded to on-demand
- Phase 27 fine-tuning targets T5 for both NLG and eventually NLU
- NeuroLogic LogitsProcessor reusable across all Candle models
- n akh router (Phase 28) gains output routing (not just input routing)

## References

- Kale & Rastogi 2020: Text-to-Text Pre-Training for Data-to-Text Tasks
- NeuroLogic Decoding: https://arxiv.org/abs/2010.12884
- Candle quantized_t5: https://docs.rs/candle-transformers/latest/candle_transformers/models/quantized_t5
- WebNLG challenge: https://synalp.gitlabpages.inria.fr/webnlg-challenge/
- ADR 037: LLM-VSA Deep Integration (Candle backend)
- ADR 038: n akh Semantic Router
