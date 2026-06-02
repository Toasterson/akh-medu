# ADR 047 — Live KG→LLM Training

> Date: 2026-04-09
> Status: Accepted
> Phase: 27
> Supersedes: None

## Context

Phase 26 established the read path (LLM hidden states → VSA HyperVecs). The
engine can now "see" what the LLM is thinking via the neural bridge. But the
LLM itself doesn't learn from the knowledge it processes — every session
starts from the same base GGUF weights.

We want the local LLM to improve over time by learning from verified KG content,
creating a self-improving loop: user knowledge → KG → training pairs → fine-tuned
model → better parsing → more knowledge.

## Decision

Implement a two-layer training system:

1. **Data collection layer** (`training-data` feature, no extra deps): logs every
   successful NLU parse, every grammar linearization, and every validated
   knowledge extraction as training pairs in redb. Runs from day one.

2. **Training layer** (`burn-training` feature): uses the Burn framework for
   LoRA fine-tuning of the local Qwen2.5 model. Runs in the background during
   idle periods, only after sufficient data has accumulated.

### Why Burn, not Candle for training

Candle is inference-first. It has basic autograd but lacks:
- LoRA adapter composition
- Gradient checkpointing (essential for M2 memory budget)
- Learning rate schedulers
- Built-in training metrics and checkpointing

Burn provides all of these with a clean Rust API. The dual-framework approach
(Candle for inference, Burn for training) is clean: both use safetensors for
weight exchange.

### Why LoRA, not full fine-tuning

- Memory: full fine-tuning of 1.5B params requires ~12 GB. LoRA rank-8 adds
  only ~12 MB of trainable parameters.
- Speed: LoRA converges faster on small domain-specific datasets.
- Safety: base model weights are never modified. Bad training → just delete
  the adapter file.
- Composability: future per-domain adapters (one per workspace).

### Cold-start thresholds

Training is deferred until:
- NLU pairs ≥ 500 (ensures enough variety for robust parsing)
- NLG pairs ≥ 200 (ensures enough linearization examples)

These thresholds are conservative. The system operates perfectly well with
just the base model — training is an optimization, not a requirement.

## Alternatives Considered

### 1. Only use Phase 35 offline training

Phase 35 trains on Wikidata/ConceptNet — generic knowledge. Phase 27 trains
on the user's actual domain. Both are needed.

### 2. Use Candle for everything (including training)

Candle's training support is basic. We'd have to implement LoRA, gradient
checkpointing, and LR scheduling ourselves. Burn provides these out of the box.

### 3. Export data and train externally (Python)

Defeats the sovereignty principle. The training loop must run locally without
cloud dependencies.

## Consequences

- Burn dependency adds ~500ms compile time and ~2 MB binary size (ndarray backend)
- Feature-gated: users who don't want training can skip it entirely
- Data collection runs unconditionally (zero overhead — just redb writes)
- Training is CPU-only (ndarray), no GPU requirement
- LoRA weights are per-workspace (different domains get different adapters)
