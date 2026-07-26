# Plan: End-to-End Chat & Inference with a Native Candle Conversational Boundary

- **Date:** 2026-06-02
- **Status:** In progress
- **ADR:** [052](../decisions/067-candle-conversational-boundary.md)

## Goal

Make akh-medu chat + inference work end-to-end so it feels like a real conversation,
while every bit of reasoning and state stays in the engine (VSA/KG). Exercise all route
paths, including a real LLM (Gemma 3 / Qwen) running through the **native Candle**
backend. Polish the TUI so reasoning is decluttered (collapsed) and answers are clear.

## Workstreams

### 1. Conversational output boundary (Candle)
- Add `generate_dialogue` + `verbalize` to `CandleBackend` (it already had `translate`
  and `generate`). `verbalize` is **grounded**: constrained to KG-derived facts.
- Dual model family: load Qwen2.5 (`quantized_qwen2`) or Gemma 3 (`quantized_gemma3`),
  detected from GGUF `general.architecture`; family drives chat template + EOS.
- Auto-discover the `.gguf` + `tokenizer.json` under `…/models/llm/` (`AKH_LLM_MODEL`
  selects which model).
- Wire `NluPipeline::{has_verbalizer, verbalize, generate_dialogue}` to Candle.

### 2. Grounded verbalization in the chat path
- `ChatProcessor::maybe_verbalize` rephrases the grounded grammar prose into chat-like
  text when a verbalizer is present, else emits the grammar output. `AKH_LLM_VERBALIZE=0`
  opts out.

### 3. Fix "teach then ask" (direct AbsTree assertion)
- `ChatProcessor::assert_abstree_triples` writes a parsed assertable tree
  (`Triple`/`Conjunction`/modifiers) straight to the KG via `ingest_extracted`, instead
  of re-running the lossy regex ingester. Regex remains the fallback.

### 4. TUI declutter
- `MessageRole` classifies each message. The transcript collapses contiguous
  reasoning/tool/audit runs into one dim summary line; Ctrl-R / `/reasoning` toggles.
  User vs. persona turns are visually distinct; final answers stay prominent.

### 5. Ollama removed
- Ollama feature/module/wiring deleted — Candle is the only LLM backend (the user needs
  hidden states, which llama.cpp/Ollama can't expose).

### 6. Hidden-state fork (DONE)
- Vendored + patched candle-transformers `quantized_qwen2` / `quantized_gemma3` into
  `src/nlu/vendored/` to expose the post-norm last-position hidden state
  (`hidden_states()` / `forward_with_hidden()`). `extract_hidden_state` now returns the
  true representation (Qwen 1536-d / Gemma 2560-d), not the logits proxy — unblocking the
  Neural→VSA bridge. Generation unchanged (`forward = output(hidden_states())`).

## Verification
- Unit tests for the backends and message roles.
- `examples/candle_route.rs` / `candle_hidden.rs` exercise translate/dialogue/verbalize
  and hidden-state extraction against real models.
- `akh chat --headless` round-trips "teach a fact → ask about it → grounded answer".

## Yes/no (polar) question handler — DONE
`ChatProcessor::try_answer_polar` intercepts "Is/Are/Does X … ?" before NLU, reads the
proposition, and checks the KG **read-only** (bounded transitive BFS, singular/plural
bridging) — answering a direct, human **"Yes, X are Y." / "No…" / "Not that I know of…"**.
This fixes the old behaviour where yes/no questions fell to the ~1.5-min LLM tier and got
**mis-stored as facts**, decomposing into provisional triples that polluted the KG.
Instant, no LLM, no writes. Examples: "Are dogs mammals?" → *Yes*; "Are dogs animals?" →
*Yes* (transitive dogs→mammals→animals); "Are dogs reptiles?" → *Not that I know of*.

## Syllogism + wh-transitivity — DONE
- **`try_quantified`** now parses its scope into a real triple (via `try_fact`), so
  "All men are mortal" stores as `(men, is-a, mortal)` instead of the literal subject
  "All men" the regex ingester produced. Quantifier stripped.
- **`try_locative`** (new Tier-1 parser) handles "X is in/at/on Y" and "X is part of Y"
  → `Triple(X, in, Y)` — fast, and keeps the preposition as the relation (rather than
  dropping it to "X is Y" via the slow LLM tier).
- **`is_transitive_pred`** follows copula/locative predicates ("is"/"are"/"in"/… exact +
  "is-a"/"part-of"/… substring) so chains connect; `inflect_variants` bridges
  singular/plural including irregulars (man↔men) at every hop.
- A new wh-relational handler (`try_answer_wh_relational`) answers "What is X in?" /
  "Where is X?" by collecting transitively-reachable objects ("Paris is in France and
  Europe.").

## Reasoning benchmark (`scripts/reason_bench.sh`)
Teaches facts in a **clean** workspace per case (deletes the workspace first — `--fresh`
clears session/goals but not the KG), asks a question, checks the answer against the
*correct* token (grammar answers via `AKH_LLM_VERBALIZE=0`). **7/7** (2026-06-02):
recall, type-query, property, transitive-yesno (dogs→animals), **syllogism**
(Socrates→man→men→mortal), **transitivity-wh** (Paris→France→Europe), negation.
Remaining: storing explicit negations (so "Are dogs cats?" can give a definite "No"
rather than "Not that I know of"); broader NLU coverage of statement forms.

## Follow-ups
- **AMD GPU (next):** ROCm/HIP Candle backend — wgpu/Vulkan can't run GGML-quantized ops
  and the user wants quantized Gemma, so ROCm is the path (port cudarc+nvcc kernels to
  HIP; needs ROCm dev toolkit installed). Large, dedicated effort.
- Wire `extract_hidden_state` into the live NLU path (Neural→VSA concept grounding).
- Dated events: route `Temporal`/event sentences from free-text chat into the event
  calculus (`record_event`) — currently "X happened yesterday" doesn't round-trip.
- Stream tokens to the TUI for live typing feel.
