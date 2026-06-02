# Plan: NLU & Chat Output Quality Improvements

**Date**: 2026-03-02
**Status**: Proposed
**Context**: After enabling the full NLU pipeline (ONNX NER + Qwen2.5 LLM) via MCP chat, several quality issues surfaced during testing.

## Issues Found

### 1. Complex queries fall through to OODA goal escalation

**Symptom**: "What does akh-medu use for reasoning?" couldn't be parsed by any NLU tier. The rule parser only handles simple `What is X` / `Tell me about X` patterns. The ONNX NER extracts entities but can't form a query structure. The LLM translator should handle this but may not be activating.

**Root cause**: Needs investigation — the LLM tier (Qwen2.5-1.5B) should be able to translate complex natural language queries into AbsTree structures. Either the model isn't loading, the prompt template isn't matching, or the output parsing fails silently.

**Impact**: Any query beyond simple subject lookup escalates to a multi-cycle OODA investigation, which is slow, noisy, and often unhelpful.

**Fix direction**:
- Verify LLM tier is actually loading and being called (add tracing)
- Check `LlmTranslator::try_load` return value and error logging
- Test the Qwen2.5 prompt template with example queries
- Consider expanding Tier 1 rule parser patterns for common question forms

### 2. Stale goals accumulate from bootstrap ingestion

**Symptom**: The agent has dozens of "Learn more about..." goals from the bootstrap text ingestion that were never completed. These pollute the OODA loop's context when it tries to investigate new queries.

**Root cause**: `ingest_text` extracts entities and creates investigation goals for unknown concepts. During bulk ingestion (awaken bootstrap), this creates many goals that never get resolved.

**Fix direction**:
- Add a post-bootstrap goal cleanup step that marks investigation goals as completed or suspended
- Or: don't create investigation goals during bootstrap ingestion (add a `bulk_mode` flag)
- Or: add goal garbage collection — auto-suspend goals older than N cycles with no progress

### 3. Chat output is triple-dump, not natural prose

**Symptom**: "What is Rust?" returns `Rust is-a Systems Programming Language. Rust created-by Graydon Hoare. Rust has-feature Ownership System.` — raw SPO lines instead of natural language paragraphs.

**Root cause**: The grammar linearizer (`abs_to_summary`) fails for most predicates because grammar rules only cover a subset. Falls back to `synthesize_from_triples` → `render_template` which outputs `subject predicate object` lines.

**Fix direction**:
- Add grammar rules for common predicates: `is-a`, `created-by`, `has-feature`, `uses`, `written-in`, `part-of`
- Or: add a prose-assembly pass in `synthesize_from_triples` that groups related facts into sentences (e.g., "Rust is a systems programming language created by Graydon Hoare. It features an Ownership System, Algebraic Data Types, and...")
- Consider using the LLM tier for response generation (translate structured triples → natural prose)

### 4. Similarity search returns near-identical scores for unrelated concepts

**Symptom**: Searching for "classify" returns "why", "connect", "Create", "add" all at 0.9999 similarity. These are not semantically related.

**Root cause**: VSA binary hypervectors at 10K dimensions may have insufficient discrimination for this many symbols (3500+). Or the HNSW index parameters need tuning. Or the symbols weren't encoded with sufficient semantic differentiation.

**Fix direction**:
- Check if entity VSA encodings actually differ (dump raw vectors for a few symbols)
- Tune HNSW ef_search / ef_construction parameters
- May need to increase dimension or use different encoding strategy for lexical items vs concepts

### 5. `synthesize_from_triples` truncates with "...and N more relation(s)"

**Symptom**: Response ends with "...and 12 more relation(s)" — useful facts are hidden.

**Root cause**: Truncation limit in the synthesis path, probably a hardcoded cap.

**Fix direction**:
- Find and increase the truncation limit, or make it configurable
- Better: rank triples by relevance/importance before truncating

## Already Fixed (2026-03-02)

- **Metadata filter missing bootstrap prefixes**: Added `expand:`, `ingest:`, `xval:`, `wd:`, `resource:`, `prereq:`, `assess:` to `is_metadata_label()` in `src/agent/synthesize.rs`
- **ONNX Runtime version mismatch**: Upgraded from 1.21.0 to 1.23.2 (ort 2.0.0-rc.11 requires >= 1.23.x)
- **Slow graceful shutdown**: Added `TimeoutStopSec=10` to systemd service (SSE connections block drain)

## Fixed (2026-03-14, ADR-030)

- **Issue 1 — LLM tier not activating**: Root cause was missing GBNF grammar enforcement in `translate()`. Replaced manual greedy sampling with `LlamaSampler::grammar()` + `LlamaSampler::greedy()` chain. Added tracing to all NLU tiers.
- **Issue 6 — No ONNX Runtime setup**: Added `akh setup onnx-runtime` command that detects platform, downloads, and installs the correct ONNX Runtime version.
- **Issue 7 — No model download**: Added `akh setup models` command for NER (~130MB) and LLM (~1.1GB) downloads with progress. Added `akh setup check` for status verification. Models stored in shared XDG dir with workspace fallback.

### 6. No installation routine for ONNX Runtime

**Symptom**: Setting up the NLU requires manually downloading the correct ONNX Runtime version, placing it in `~/.local/lib/onnxruntime/`, and configuring `ORT_DYLIB_PATH`. This is error-prone and undocumented.

**Root cause**: The `ort` crate's `load-dynamic` feature requires a system-level shared library that isn't bundled.

**Fix direction**:
- Add an `akh setup` (or `akhomed setup`) CLI subcommand that:
  - Detects OS + arch (linux-x64, linux-aarch64, macos-x64, macos-aarch64)
  - Downloads the matching ONNX Runtime release from GitHub (version pinned to match `ort` crate requirement, currently >= 1.23.x)
  - Installs to `~/.local/lib/onnxruntime/` (Linux) or `~/Library/Frameworks/` (macOS)
  - Verifies the library loads correctly
  - Prints the env var to set (or writes it to akh-medu config)
- Update the systemd service template to reference the installed path
- Add a `--check` flag that validates the current installation
- Document in the book (`docs/book/`) and `contrib/systemd/`

### 7. No automatic model download for Qwen2.5 and NER

**Symptom**: The NLU pipeline silently degrades when model files are missing — no user-visible error or download prompt. Users must manually download ~1.2GB of model files and place them in the correct workspace subdirectory.

**Root cause**: `MicroMlLayer::try_load` and `LlmTranslator::try_load` return `None` on missing files. No download mechanism exists.

**Prerequisite**: Fix Issue 1 first (verify LLM tier actually works when models are present).

**Fix direction**:
- Add model download to the `akh setup` subcommand (or a separate `akh models download` command):
  - NER: `Xenova/distilbert-base-multilingual-cased-ner-hrl` (ONNX quantized, ~130MB)
  - LLM: `Qwen/Qwen2.5-1.5B-Instruct-GGUF` q4_k_m variant (~1.1GB)
- Download to a shared location (`~/.local/share/akh-medu/models/`) and symlink or configure per-workspace
- Show progress bar during download (indicatif or similar)
- On first `chat` call, if models are missing, return a helpful error message instead of silent degradation:
  `"NLU models not installed. Run 'akh setup models' to download them."`
- Pin model versions in a manifest file so upgrades are tracked

## Priority Order

1. Issue 1 (LLM tier not activating) — highest impact, unlocks complex queries
2. Issue 6 (ONNX Runtime setup) — prerequisite for anyone else to use NLU
3. Issue 7 (model download) — depends on Issue 1 being confirmed working
4. Issue 3 (prose output) — user experience
5. Issue 2 (stale goals) — agent quality
6. Issue 5 (truncation) — easy fix
7. Issue 4 (similarity scores) — deeper investigation needed
