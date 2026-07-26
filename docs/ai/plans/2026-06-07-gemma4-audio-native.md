# Plan: Gemma 4 12B audio-native, GPU-only boundary + benchmark self-improvement

- **Date:** 2026-06-07
- **Status:** In progress
- **ADR:** [054](../decisions/069-gemma4-audio-native.md) (builds on [052](../decisions/067-candle-conversational-boundary.md), [053](../decisions/068-rocm-candle-gpu-backend.md))

## Goal (verbatim)

> with gemma 12B on the GPU. Make it drive all akh-medu interaction (no other fallbacks)
> including listening to an audio file from the web, Let it reason how it can improve its
> benchmark results and improve them in a chosen benchmark.

Clarified: the model is **Gemma 4 12B** (released 2026-06-03, audio-native, encoder-free) — not
the Gemma 3 12B already on disk. See ADR 069.

## Phases (commit between each)

### P0 — Acquire + inspect (no code) ✅
- [x] Downloaded `gemma4-12b.gguf` (6.7 GB, arch `gemma4`), `gemma4-12b.mmproj.gguf` (168 MB),
      `gemma4-12b.tokenizer.json` (31 MB).
- [x] Inspected GGUF: arch `gemma4`, 48 blocks, hidden 3840, 16 heads. **Per-layer geometry** —
      sliding layers (head_dim 256, 8 KV heads, rope base 1e4, full 256-dim rotary); full-attn
      every-6th (head_dim 512, **1** KV head MQA, rope base 1e6, **partial** rotary via `rope_freqs`
      → only 128 dims rotate). Tensors per block: attn_q/k/v, attn_output, attn_q_norm/attn_k_norm
      (per-head, head_dim), attn_norm, post_attention_norm, ffn_gate/up/down, ffn_norm,
      post_ffw_norm, **layer_output_scale[1]**; globals token_embd (tied lm_head), output_norm,
      rope_freqs[256]. final_logit_softcapping 30. Exact forward math captured from llama.cpp
      reference (commit 379ac66): plain RMSNorm (no +1), q/k-norm before RoPE, bare V RMSNorm,
      **softmax scale 1.0** (no 1/√d), **GELU** gate (not SiLU), sandwich norms, embed ×√3840,
      logits `30·tanh(x/30)`.
- [x] mmproj audio projector = **`mm.a.input_projection.weight` [640→3840] f16** (single linear,
      no bias). Audio = PCM → 640-sample frames → this matmul → splice at `audio_token` (258881),
      delimited boa 256000 / eoa 258883.

### P1 — Gemma 4 text decoder in Candle (GPU) ✅
- [x] Vendored `src/nlu/vendored/quantized_gemma4.rs`: per-layer sliding(256-d,8kv,θ1e4) / full
      every-6th(512-d,1kv MQA,θ1e6, partial rotary 64 pairs via `rope_freqs`) attention; plain
      RMSNorm (no +1); per-head q/k-norm before RoPE; **bare V-norm + V tied to K** on full layers
      (they omit `attn_v`); attention softmax scale **1.0**; **GELU** MLP; per-block
      `layer_output_scale`; tied lm_head; final logit softcap 30. GPU path: q/k/v/o via
      `GpuAttnProj`, gate/up/down via three `GpuLinear` + Candle GELU (NOT the silu-fused `GpuMlp`),
      tied lm_head via `gpu_output`; native Q4_K (ADR 068).
- [x] `ModelFamily::Gemma4` (detect `gemma4` before generic `gemma`) + dispatch; **new Gemma 4 chat
      template `<|turn>role\n…<turn|>\n<|turn>model\n`** + EOS `<turn|>` (id 106); discovery excludes
      `.mmproj` sidecars + exact-stem match; tolerant signed-int GGUF metadata reads.
- [x] **Fixed degenerate output**: Gemma needs a leading **`<bos>`** (tokenizer doesn't auto-prepend)
      and the new turn template — without them greedy decoding collapsed. Also strip Gemma 4's
      `<|channel>thought…<channel|>` wrapper in `clean_generated_text`.
- [x] **Validated on GPU**: `gemma4-12b` loads (peak VRAM **7663 MB** int4, gpu_busy ~98%) and all
      three `candle_route` routes are coherent — translate → clean `Triple{dogs,are,mammals}`;
      dialogue → "Hello there, I am Akh…"; verbalize → grounded answer. Hidden-state path intact
      (forward_with_hidden wired). Default (non-rocm) + lib builds clean.

### P2 — Native audio
- [ ] Vendor the audio projector: load `gemma4-12b.mmproj.gguf`, expose
      `project_audio(pcm16k_mono: &[f32]) -> Tensor[n_frames, 3840]` (640-sample frames + projector
      linears). Unit-test the frame split + projection shapes.
- [ ] Embed-splice: build the token stream with `boa`/`audio_token×n`/`eoa`, embed tokens, overwrite
      audio placeholders with the projected frame embeddings, run the decoder.
- [ ] `extract_audio_understanding`/route through `CandleBackend` so the engine can take audio in.

### P3 — Audio from the web (end to end)
- [ ] WAV decode → 16 kHz mono f32 (`hound` + simple resampler). Extend HTTP fetch beyond 256 KB
      for media (separate bounded media-fetch, e.g. ≤ 25 MB).
- [ ] `examples/gemma4_audio.rs` (and/or an agent tool): given an audio URL, fetch → decode →
      `project_audio` → Gemma 4 on GPU answers a question about the clip. Validate on a real FLOSS
      WAV (e.g. a short public-domain speech clip).

### P4 — Strict single-model mode (no fallbacks)
- [ ] `AKH_LLM_STRICT=1`: model load + GPU mandatory (hard miette error, no CPU fallback); LLM
      verbalize + dialogue mandatory (no grammar/scripted fallback); disable `AKH_LLM_VERBALIZE=0`
      opt-out. Default model id → `gemma4-12b` when present.
- [ ] Validate: chat/route paths all driven by Gemma 4 on the GPU; a deliberately missing GPU/model
      errors with a clear diagnostic instead of silently degrading.

### P5 — Benchmark self-improvement (agentic)
- [ ] Pick a benchmark the agent can measure + move (reasoning 7/7 via `scripts/reason_bench.sh`,
      or the `CompetenceAssessor` score). Capture baseline.
- [ ] Have the akh-medu agent (driven by Gemma 4) **reason** about its weak cases and act
      (learning cycle / parser-exemplar / KG fill), then **re-measure** and show the score move.
      Record before→after with provenance (`DerivationKind::AgentDecision`).

## Verification of the end goal
- Gemma 4 12B answers chat **and** a question about a web audio clip, generated **on the GPU**.
- No fallback path is taken in strict mode (missing GPU/model → diagnostic, not silent CPU/grammar).
- A chosen benchmark's score is higher after the agent's self-directed improvement, with provenance.

## Status / log
- (P0 in progress) Architecture gathered from HF config + announcements (ADR 069). Download running.
