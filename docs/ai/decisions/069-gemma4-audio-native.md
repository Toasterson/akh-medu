# ADR 069: Gemma 4 12B as the audio-native, GPU-resident boundary model

- **Date:** 2026-06-07
- **Status:** Accepted
- **Supersedes/extends:** [ADR 067](067-candle-conversational-boundary.md) (Candle boundary), [ADR 068](068-rocm-candle-gpu-backend.md) (ROCm GPU backend)
- **Plan:** [2026-06-07-gemma4-audio-native.md](../plans/2026-06-07-gemma4-audio-native.md)

## Context

Gemma 4 12B was released **2026-06-03** — an *encoder-free unified multimodal* model
(text + image + **audio** + video, 256K context, Apache 2.0) that runs on a 16 GB laptop.
The user wants it to be the **sole** boundary model for akh-medu, GPU-resident, with no
fallback to other models, and to **listen to an audio file fetched from the web**.

akh-medu's hard constraint (ADR 067) is that the boundary LLM runs **in-process under Candle**
so hidden states stay extractable for the Neural→VSA bridge — this rules out llama.cpp/Ollama
even though they already run Gemma 4 audio. Candle has **no Gemma 4 support** (the model is days
old), so we vendor it, exactly as we vendored Qwen2/Gemma3 (ADR 067/053).

## Why this is feasible now (not a multi-week encoder port)

Gemma 4 12B is **encoder-free**: unlike the E2B/E4B variants (12 conformer layers), the 12B
projects **raw audio frames straight into the text embedding space** with a single linear map.

- Audio: 16 kHz mono → **40 ms = 640-sample frames** → linear projection → 3840-d embedding,
  one embedding per frame. No mel spectrogram, no conv frontend, no attention encoder.
- The frame embeddings are spliced into the token-embedding sequence at `audio_token_id`
  (258881) placeholders, delimited by `boa` (256000) / `eoa` (258883).

The **text decoder** is structurally close to Gemma 3 (which we already vendored and GPU-wired
under ROCm), so the GPU path (`GpuLinear`/`GpuMlp`/`GpuAttnProj`, ADR 068) transfers directly.
Known decoder deltas vs Gemma 3, to implement in `quantized_gemma4.rs`:
- **Partial rotary factor 0.25** — only 64 of the 256 head dims are rotated by RoPE.
- Per-layer RoPE: full-attention layers θ=1e6 ("proportional"), sliding layers θ=1e4.
- Sliding window 1024 with **full attention every 6th layer** (as Gemma 3).
- **final_logit_softcapping = 30.0**.
(The downloaded GGUF metadata is the source of truth and is reconciled against this before coding.)

## Decision

1. **Vendor `src/nlu/vendored/quantized_gemma4.rs`** — a Candle GGUF loader + decoder for the
   `gemma4_unified` text stack, adapted from `quantized_gemma3.rs`, with the four deltas above.
   Reuse the ROCm `GpuLinear`/`GpuMlp`/`GpuAttnProj`/`gpu_output` wiring (ADR 068) and the native
   Q4_K path so it loads the omni Q4_K_M GGUF resident on the GPU.
2. **Vendor the audio projector** — load `mmproj-F16.gguf`, expose
   `project_audio(pcm_16k_mono: &[f32]) -> Tensor[n_frames, 3840]` (frame-slice + the projector's
   linear layers). Splice frame embeddings into the embedded token stream at the audio placeholders.
3. **Audio-from-web path** — fetch an audio URL (extend the HTTP fetch past the 256 KB cap for
   media), decode WAV → 16 kHz mono f32 PCM (`hound` + a simple resampler), call `project_audio`,
   prepend a user turn carrying the audio, generate on the GPU. An example/tool exercises it.
4. **Strict single-model mode** (`AKH_LLM_STRICT=1`) — Gemma 4 12B on the GPU drives ALL
   interaction: model + GPU mandatory (hard error, no CPU fallback), LLM verbalize/dialogue
   mandatory (no grammar/scripted fallback), `AKH_LLM_VERBALIZE=0` opt-out disabled.
5. **Make Gemma 4 the default** model id (`gemma4-12b`) when present.

## Consequences

- **+** True native-audio understanding in-process, GPU-resident, FLOSS, hidden states intact.
- **+** One boundary model for every modality; the four-tier NLU keeps the rule parser as the fast
  structural path, but generation/understanding fall through to Gemma 4 only (strict mode).
- **−** We carry a vendored copy of a brand-new architecture; GGUF tensor naming follows llama.cpp's
  `gemma4` conversion (PR #24118) and must be tracked if it changes.
- **−** Audio decode starts WAV-only (`hound`); mp3/ogg deferred (symphonia) — logged, not silent.
- **Risk:** partial-rotary / softcapping / per-layer-RoPE must match the GGUF exactly; validated by
  a text-generation coherence check and (where possible) hidden-state parity vs the f32 path.

## Alternatives rejected

- **llama.cpp/Ollama for Gemma 4 audio** (works today) — breaks the in-process hidden-state
  requirement (ADR 067). Rejected as the engine path; may be used only as an external oracle to
  cross-check transcription/understanding during bring-up.
- **Whisper STT → Gemma text** — not native audio; loses prosody/non-speech audio and adds a second
  model. Rejected per the user's explicit "native audio" requirement.
- **Gemma 4 E2B/E4B** (also audio-native) — smaller, but use conformer encoders (more to port) and
  aren't the requested 12B.
