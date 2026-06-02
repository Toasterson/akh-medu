# ADR-030: LLM Integration Fixes — GBNF Enforcement, Observability & Setup

**Date**: 2026-03-14
**Status**: Accepted

## Context

The four-tier NLU pipeline (Phase 14j-14m) was structurally complete but the LLM translator (Tier 3) was non-functional. Testing revealed that `LlmTranslator::translate()` implemented manual greedy argmax over raw logits without applying the GBNF grammar constraint. The grammar file `abstree.gbnf` was loaded via `include_str!` but never passed to `LlamaSampler::grammar()`. This caused unconstrained LLM output that failed JSON deserialization, silently disabling Tier 3 and causing complex queries to fall through to expensive OODA escalation.

Additionally, zero tracing existed in the NLU module, model loading failures were swallowed silently, and no installation routine existed for models or the ONNX Runtime.

## Decisions

### 1. Replace greedy sampling with grammar-constrained LlamaSampler

The translate() method now chains `LlamaSampler::grammar(model, ABSTREE_GBNF, "root")` with `LlamaSampler::greedy()` via `LlamaSampler::chain_simple()`. This guarantees all generated tokens conform to the GBNF grammar, producing only valid AbsTree JSON.

Added `NluError::GrammarInitFailed` for grammar initialization errors.

### 2. Add tracing throughout the NLU pipeline

All tier attempts, successes, and failures now emit `tracing::debug!`/`tracing::info!`/`tracing::warn!` events with structured fields (tier number, confidence, error). Model loading in `load_models()` now uses `load()` directly instead of `try_load()` to capture and log the actual error.

### 3. Surface model status to users

Added `NluTierStatus` struct with `Display` impl and `NluPipeline::tier_status()` method. `ChatProcessor::new()` logs tier status at initialization and warns when the LLM tier is missing.

### 4. Shared model directory with XDG fallback

`load_models()` now searches the workspace-local path first, then falls back to the shared XDG models directory (`$XDG_DATA_HOME/akh-medu/models/`). Added `AkhPaths::models_dir()`.

### 5. `akh setup` CLI subcommand

New `akh setup models|onnx-runtime|check` commands handle downloading NER (~130MB), LLM (~1.1GB), and ONNX Runtime binaries with progress reporting. Uses `ureq` for HTTP, `flate2`+`tar` for archive extraction.

## Alternatives Considered

- **Bundling models in the binary**: Rejected — models total ~1.3GB, impractical for binary distribution.
- **Using `try_load()` with warning messages**: Rejected — `try_load()` swallows errors via `.ok()`, making debugging impossible.
- **Temperature sampling instead of greedy**: Deferred — greedy is deterministic and sufficient for structured output; temperature can be added later if needed.

## Consequences

- Tier 3 (LLM) will now produce valid AbsTree JSON when the model is loaded
- NLU failures are visible via `RUST_LOG=akh=debug`
- Users can run `akh setup check` to diagnose missing models
- New dependencies: `flate2` and `tar` (for ONNX Runtime archive extraction)
