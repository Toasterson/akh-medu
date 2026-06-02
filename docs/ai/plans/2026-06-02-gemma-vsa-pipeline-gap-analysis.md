# Gap Analysis — Controlled Knowledge Development & Inference with Gemma as Driver and Knowledge → VSA Bridge

- **Status:** Assessment / planning
- **Date:** 2026-06-02
- **Author:** integration pass (`goal/integrate-merge-main`)
- **Scope:** What is missing to run a *controlled* knowledge-development + inference loop where **Gemma (Gemma 3 / "Gemma 4"-class)** is the LLM driver and where knowledge "jumps into VSA space" from the LLM's calculation/embedding space.
- **Code state basis:** the integrated tree after merging 48 of the 50 open PRs (see `docs/ai/2026-06-02-pr-integration-report.md`).

## TL;DR

Today the pipeline is **LLM-as-symbolic-translator** (text ⇄ triples/symbols) plus **structural VSA grounding** (random base vectors + neighborhood/role-filler binding). The continuous path the goal asks for — *Gemma's hidden/embedding space → VSA hypervector space* — is **scaffolded but not operational**, and **Gemma itself is not supported** (the stack is hardwired to Qwen2.5-1.5B). The five gaps below are ordered by dependency.

---

## Current pipeline (what exists and works)

```
            ┌─────────────── boundary (human language) ───────────────┐
 text ─▶ NLU tiers (rule → ONNX NER → Qwen2.5-1.5B → VSA ranker)        │
            │                                                          │
            ▼                                                          ▼
   triple/intent extraction                              synthesize.rs (template/grammar NL)
   (src/extraction, knowledge_elicit)                    + CandleFormatter (DORMANT)
            │                                                          ▲
            ▼                                                          │
   engine.add_triple + store_provenance                    NarrativeSummary / PipelineOutput
            │                                                          │
            ▼                                                          │
   KG (petgraph + oxigraph SPARQL)  ──▶ ground_symbol (structural VSA) ─┘
            │
            ▼
   OODA loop: VsaNative decide (vsa_decide) → act → vsa_rl reward (symbolic predicted-outcome)
```

- **LLM boundary translator:** Qwen2.5-1.5B-Instruct, two interchangeable backends — Candle GGUF (`nlu-candle`, default) and llama-cpp-2 (`nlu-llm`). `src/nlu/candle_backend.rs`, `src/nlu/llm_translator.rs`, `src/nlu/mod.rs`.
- **Controlled ingest:** `src/extraction/` (text → Qwen → `RawTriple` → `add_triple` + `DerivationKind::LlmTripleExtraction`) and `src/bootstrap/knowledge_elicit.rs` (Phase 32, `DerivationKind::KnowledgeElicitation`), wired into the daemon (`run_knowledge_extraction`, `run_knowledge_elicitation_task`).
- **VSA grounding:** `src/vsa/item_memory.rs` (random seeded hypervec per symbol) + `src/vsa/grounding.rs` (`ground_symbol`: bundle neighbors + role-aware `bind(predicate, object)`), bipolar bit-packed 10K-bit (`src/vsa/mod.rs`).
- **Neural→VSA bridge primitives:** `src/vsa/neural_bridge.rs` — `NeuralVsaBridge` (`encode(hidden_state) = binarize(W·h + b)`), `LinearProjection`, `BridgeTrainer` (real SGD/BCE), reverse decoder. These compile and are unit-tested.
- **Candle NL generation:** `src/agent/output/candle_format.rs` — `CandleFormatter` turns VSA/KG `NarrativeSummary`/`PipelineOutput` into prose via Qwen, with template fallback.
- **OODA VSA-native loop:** `src/agent/vsa_decide.rs` (action selection), `src/agent/vsa_rl.rs` (`encode_predicted_outcome`, predicted-vs-actual cosine reward), wired in `src/agent/ooda.rs`.

---

## Gap 1 — Gemma is not a supported driver (hardwired to Qwen2.5-1.5B)

**Observed:** repo-wide there is **zero** `gemma`/`Gemma` support. The Candle backend type is hardwired: `src/nlu/candle_backend.rs:22 use candle_transformers::models::quantized_qwen2::ModelWeights;`. The model file is a hardcoded constant `qwen2.5-1.5b-instruct-q4_k_m.gguf` under `models/llm/` resolved relative to `data_dir` (`candle_backend.rs:74-76, :190`) with **no env override**. The chat template and EOS tokens are Qwen-specific (`<|im_start|>`/`<|im_end|>`, EOS 151643/151645, `candle_backend.rs:~531`). llama-cpp-2 path is likewise documented as Qwen (`llm_translator.rs:3`).

**Missing / work required:**
1. A model-backend abstraction (`LlmBackend` trait) selecting the model architecture + chat template + special tokens at runtime. Phase 39a (`claude/wi-019dc9c9-phase-39a-llmbackend-trait-candlelocal-i`, unmerged) appears to start this — adopt or finish it.
2. A Gemma model type for Candle. `candle-transformers` ships Gemma/Gemma2/Gemma3 (`models::gemma`, `models::gemma2`, `models::gemma3`) and quantized variants; wire a `quantized_gemma`/`gemma3` `ModelWeights` path. **Fork `candle-transformers` if the desired quantized Gemma 3 variant is not exposed** (forking deps is acceptable per project direction).
3. Gemma chat template (`<start_of_turn>user … <end_of_turn>\n<start_of_turn>model`) and Gemma EOS/`<end_of_turn>` token id handling.
4. Config: an env var / `DaemonConfig`/`ChatProcessor` field for model path + architecture, replacing the hardcoded constant. Ship/document the Gemma GGUF under `models/llm/`.

**Effort:** medium. Mechanical once the `LlmBackend` trait lands; the risk is the quantized Gemma 3 kernel availability in candle (may require the fork).

## Gap 2 — No operational "knowledge jumps into VSA space from the LLM calculation space"

This is the **core** of the goal and the largest gap. The projection machinery exists but is fed garbage and discarded.

**Observed:**
- `NeuralVsaBridge` (`src/vsa/neural_bridge.rs`) is real, but the daemon trainer `run_bridge_training` (`src/agent/daemon.rs:~990`) (a) uses `hidden_dim = 128` (real Qwen hidden is 1536; Gemma 3 is 1152/2304/3584 depending on size), (b) **constructs a fresh random bridge every cycle and drops it** — weights are never persisted or used for inference, and (c) trains via `Session::train_from_grounded_symbols` → `trainer.rs proxy_hidden_state(sym_id)`, an **xorshift hash of the symbol ID**, explicitly *"a placeholder for real LLM hidden states"* (`src/training/trainer.rs:~233`).
- `CandleBackend::extract_hidden_state` (`src/nlu/candle_backend.rs:~275`) has **no call sites** and returns **last-token logits (~vocab dim), not hidden states**, because *"`quantized_qwen2::ModelWeights` does not expose per-layer hidden states … would require forking candle-transformers"*.
- `encode_predicted_outcome` (`src/agent/vsa_rl.rs:~83`) is purely symbolic (item-memory + role-filler bind), no LLM.

**Missing / work required (the real bridge):**
1. **Real hidden-state extraction** from the driver model. Requires a forked/extended Candle model that returns the final-layer hidden state (or a chosen layer) for a token span — not logits. This is the dependency fork the goal anticipates. For Gemma 3, expose `hidden_states` from the decoder before the LM head.
2. **Correct, configurable `hidden_dim`** matching the chosen Gemma variant; thread it from the backend, not a `128` literal.
3. **Persist + load the trained bridge** (`LinearProjection` weights) in the TieredStore / `data_dir`, and **use it at inference** (`encode(hidden_state) → HyperVec`) so a concept's LLM embedding actually produces its VSA vector. Today nothing calls `encode` on real embeddings outside tests.
4. **A training signal** that aligns the projection: pairs of (LLM hidden state for a concept's text, target VSA vector from `ground_symbol`). Replace `proxy_hidden_state` with cached real hidden states captured during ingest.
5. **Bipolar/bit-width considerations:** the binarize-at-0 step (`neural_bridge.rs binarize_to_hypervec`) collapses continuous semantics into the 10K-bit Hamming space whose analogy is degenerate (see Gap 5). Consider keeping a continuous "shadow" embedding alongside the binary VSA vector, or higher-dim / bipolar-with-magnitude encoding for the bridged vectors specifically.

**Effort:** large. This is the headline feature. Phase 26 (LLM-VSA Deep Integration, ADR 037) and Phase 27 (Live KG→LLM Training) are the planned homes; neither is operational.

## Gap 3 — Candle NL generation is built but not wired into the output path

**Observed:** `CandleFormatter` (`src/agent/output/candle_format.rs`) is exported (`src/agent/output/mod.rs`, `src/agent/mod.rs`) but has **no call site** outside its module/tests. Live agent responses still go through `src/agent/synthesize.rs` (`synthesize` / `synthesize_with_grammar`) — template/grammar rendering.

**Missing / work required:** route `ask`/`chat` synthesis through `CandleFormatter` when a model is loaded (with the existing template fallback when absent). This is the "Gemma as driver for OUTPUT" half — once Gemma is the backend (Gulf 1), wiring this gives Gemma-driven verbalization of VSA/KG state. Small change; gated behind `candle-backend`/model availability.

## Gap 4 — Controlled knowledge ingest is symbolic extraction, not embedding transfer; dedicated model stubbed

**Observed:** ingest is text → Qwen → JSON triples → `add_triple` + provenance (good: provenance is written for both extraction paths). But `knowledge_elicit.rs` targets a **T5-Knowledge** model (`model_id: "t5-knowledge-q6k"`, `models/llm/t5-knowledge-q6k.gguf`) that **does not ship** and is "currently stubbed to the available Candle backend"; Phase 35 (T5 training infra) is planned-only.

**Missing / work required for *controlled* development:**
1. A **confidence/validation gate** on elicited triples before they enter the KG (the `ElicitValidator` exists — ensure it is enforced and tuned; hallucination detection per Phase 32 plan).
2. A **review/quarantine compartment** for LLM-proposed triples so knowledge development is *controlled* (human/agent approval) rather than auto-committed — this also addresses the contamination issues that motivated many of the merged PRs (self-referential triples, is-a hairballs, persona bleed).
3. Decide Gemma-vs-T5 for elicitation: either ship the T5-Knowledge GGUF (Phase 35) or formally make Gemma the elicitation backend and delete the T5 stub path to avoid silent degradation.

## Gap 5 — VSA grounding is structural, not semantic (binary-VSA analogy degeneracy stands)

**Observed:** `ground_symbol` gives *relational* structure (neighborhood bundle + role-filler bind) over random base vectors in bipolar 10K-bit space. The documented degeneracy remains: bind/unbind in 10K-bit Hamming space yields near-equidistant vectors, so analogy/semantic-similarity is weak. This is *why* a learned LLM→VSA projection (Gap 2) matters — it would inject continuous semantics the structural grounding cannot provide.

**Missing / work required:** either (a) higher-dimensional or magnitude-bearing encoding for bridged vectors, or (b) keep continuous LLM embeddings as a parallel semantic index (e.g. pgvector/HNSW already present) and reserve binary VSA for compositional/relational ops — explicitly splitting "semantic similarity" (LLM/embedding) from "relational binding" (VSA).

---

## Recommended sequence

1. **Gap 1** — land `LlmBackend` trait (Phase 39a) + Gemma 3 Candle backend (fork candle-transformers if needed) + configurable model path. *Unblocks everything.*
2. **Gap 3** — wire `CandleFormatter` into `ask`/`chat` output (cheap win; gives Gemma-driven verbalization immediately).
3. **Gap 2** — real hidden-state extraction (forked model) → persisted, inference-time `NeuralVsaBridge` with correct dims and a real training signal. *The headline "knowledge → VSA from Gemma calculation space" capability.*
4. **Gap 4** — controlled-development gate (validation + quarantine compartment) for LLM-proposed knowledge.
5. **Gap 5** — split semantic (embedding/pgvector) vs relational (binary VSA) indexes; revisit encoding for bridged vectors.

## Cross-references

- Phase 26 — LLM-VSA Deep Integration: `docs/ai/plans/2026-03-26-phase26-llm-vsa-deep-integration.md`, ADR 037.
- Phase 27 — Live KG→LLM Training (depends on 26).
- Phase 32 — LLM Knowledge Extraction: `docs/ai/decisions/042-llm-knowledge-extraction.md` (merged as PR #118).
- Phase 35 — T5 Training Infrastructure: `docs/ai/plans/2026-03-27-phase35-t5-training-infrastructure.md`.
- Phase 39a — `LlmBackend` trait / CandleLocal (unmerged branch `claude/wi-019dc9c9-phase-39a-llmbackend-trait-candlelocal-i`).
- ADR 022 — NLU architecture (Qwen as boundary translator).
