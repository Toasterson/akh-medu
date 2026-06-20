# ADR 067: Native Candle LLM as the Conversational Boundary

- **Status:** Accepted
- **Date:** 2026-06-02
- **Anima WI:** (chat end-to-end goal)
- **Supersedes / relates to:** [ADR 022 (NLU architecture)](022-nlu-architecture.md), [ADR 037 (LLM-VSA deep integration)](037-llm-vsa-deep-integration.md)

## Context

The TUI chat needed to "feel like writing to a person" while keeping **all reasoning
and state in VSA/KG space**. Two gaps blocked a real end-to-end experience:

1. **No conversational output boundary.** Knowledge answers were linearized only by
   the deterministic grammar (faithful but stilted). Dialogue acts had a Qwen path
   via `llama-cpp-2`, but knowledge answers never used an LLM, so replies read like
   database dumps rather than chat.
2. **Teaching was lossy.** When the user stated a fact, the NLU produced a perfectly
   good `AbsTree::Triple`, but the assert path threw it away and re-ran a conservative
   regex ingester (`TextIngestTool`) that extracted ~0 triples. So "teach then ask"
   never round-tripped.

A short-lived spike routed these through **Ollama** (external daemon) and proved the
shape works — including Gemma 3 on the AMD GPU via ROCm. We then chose **Candle**
(native, in-process, pure-Rust) and **removed Ollama entirely**, for two reasons:
(1) **sovereignty** — no external model daemon, the loader is compiled into `akh-medu`
itself; (2) **hidden states** — Candle gives access to intermediate tensors for the
Neural→VSA bridge (Phase 26), which `llama.cpp`/Ollama cannot expose. The internal
states are the decisive reason it must be Candle, not a C++/HTTP backend.

## Decision

Use **Candle** (`candle-backend` feature) as the conversational boundary, exposing three
boundary operations on `CandleBackend`, all of which keep the KG as the source of truth:

- **`translate`** — NL → `AbsTree` (NLU Tier 3, input boundary). Existing.
- **`generate_dialogue`** — persona-framed greeting/farewell/ack (output boundary). New.
- **`verbalize`** — rephrase **KG-derived facts** into a fluent, chat-like answer,
  *constrained to those facts* (grounded generation, output boundary). New.

Supporting changes:

1. **Dual model family.** `CandleBackend` now loads either Qwen2.5 (`quantized_qwen2`)
   or Gemma 2/3 (`quantized_gemma3`), detected from GGUF `general.architecture`. A
   `ModelFamily` drives chat templating (`<|im_start|>` vs `<start_of_turn>`) and EOS
   tokens (`<|im_end|>` vs `<end_of_turn>`). Model files are auto-discovered under
   `…/models/llm/` (`AKH_LLM_MODEL` hint selects which `.gguf`).
2. **Grounded verbalization in chat.** `ChatProcessor::maybe_verbalize` rephrases the
   grounded grammar prose into conversational text when a verbalizer is loaded; it
   falls back to the grammar output otherwise. Opt out per request with
   `AKH_LLM_VERBALIZE=0`.
3. **Direct AbsTree assertion.** `ChatProcessor::assert_abstree_triples` writes the
   parsed `Triple`/`Conjunction`/modifier tree straight to the KG (reusing
   `text_ingest::ingest_extracted`), so a fact the NLU understood actually lands —
   fixing "teach then ask". Regex ingest remains the fallback.
4. **TUI declutter.** Messages carry a `MessageRole`; the transcript collapses runs of
   reasoning/tool/audit "thinking" into one dim summary line (Ctrl-R / `/reasoning`
   toggles), and labels user vs. persona turns. Final answers stay prominent.

Ollama was **removed** — Candle is the only conversational/translation backend.

## GPU note (important, unresolved)

Candle's GPU backends are **CUDA and Metal only** (`Device` is `Cpu | Cuda | Metal` in
candle-core 0.10; confirmed in source). It has **no AMD ROCm/Vulkan backend**, so on AMD
hardware (this dev box is an RX 7900 XTX / gfx1100) Candle runs Gemma/Qwen on **CPU**
(~13 s/answer Qwen-1.5B, ~33 s Gemma-3-4B in release).

Getting Candle onto the AMD GPU is a **fork-scale effort** with no one-shot path:
- **Vulkan via wgpu** — Vulkan is available on this box (RADV + Mesa 23.2). candle has no
  mainline wgpu backend (issue #344 open since 2023); third-party wgpu forks exist but
  are immature and do **not** support GGML-quantized ops, so they'd require a
  non-quantized (f16) model and a new `BackendStorage`/`BackendDevice` impl. Large.
- **ROCm/HIP** — ROCm runtime works here (Ollama used it), but candle's CUDA backend is
  `cudarc` + nvcc-compiled kernels; porting to HIP needs the ROCm dev toolkit (not
  installed) plus a kernel/runtime port. Largest.
- **ZLUDA** (CUDA-on-AMD shim) — could in principle run candle's `cuda` backend
  unmodified, but needs the CUDA toolkit to build candle-kernels and ZLUDA's PTX
  coverage is uncertain on RDNA3.

Tracked as a dedicated follow-up; the wgpu/Vulkan backend (f16 model) is the
recommended path. Until then the conversational boundary runs on CPU.

## Hidden states (the reason it must be Candle) — DONE

`extract_hidden_state` now returns the **true post-norm last-position hidden state**
(Qwen2.5-1.5B → 1536-dim, Gemma-3-4B → 2560-dim), not the old vocab-sized logits proxy.

Upstream candle-transformers keeps `ModelWeights::{norm, output, layers}` private, so the
hidden state couldn't be reached. We **vendored + patched** the two quantized loaders into
`src/nlu/vendored/{quantized_qwen2,quantized_gemma3}.rs` (transformer math byte-for-byte
upstream; only imports rewritten and a `hidden_states()` / `forward_with_hidden()` pair
added that returns the representation *before* the lm_head projection). `forward()` is
re-expressed as `output.forward(hidden_states(...))`, so generation is unchanged.

This aligns `extract_hidden_state` with what `NeuralVsaBridge::new(hidden_dim, …)` already
expected (1536), unblocking the Phase-26 Neural→VSA bridge. Verified: dims 1536 (Qwen) /
2560 (Gemma); verbalize still works through the vendored `forward()`.

**Next:** AMD GPU. wgpu/Vulkan can't run GGML-quantized ops, and the user wants quantized
Gemma — so the GPU path is a **ROCm/HIP** candle backend (port cudarc+nvcc kernels to HIP;
needs the ROCm dev toolkit). Tracked as the next dedicated effort.

## Consequences

- **Sovereign:** the conversational model is loaded in-process; no daemon to run.
- **Grounded:** verbalization is constrained to KG facts; reasoning stays symbolic.
- **Flexible:** Qwen (fast, 1.5B) or Gemma 3 (richer, 4B) via the same code path; the
  model file decides.
- **Honest about hardware:** GPU only on CUDA/Metal; AMD users either accept CPU or use
  the Ollama feature.
- **Model files required:** a `.gguf` + matching HF `tokenizer.json` must be present;
  the Ollama-exported Gemma 3 GGUF is multimodal and lacks the text RMS-epsilon key
  candle expects, so a text-only Gemma 3 GGUF (e.g. `unsloth/gemma-3-4b-it-GGUF`) is
  used instead.
