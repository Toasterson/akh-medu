# Vox recording review through Akh-Medu

- Status: native transport implemented; live serving qualification pending.
- Date: 2026-10-03.
- Operator direction: use Akh-Medu/Roman for model-assisted recording review and
  eventually for all transcription comparisons. Vibe may provide an initial
  independent baseline; it does not become another canonical agent runtime.
- Related: [native Gemma audio](../decisions/069-gemma4-audio-native.md),
  [full akhs](../decisions/113-full-akh-always.md), and
  [Hephaistos serving](../decisions/121-lilith-serving-on-hephaistos-llama-cpp.md).

## Verified boundary

The historical Gemma audio change (`a7c1f416d`) added
`CandleBackend::understand_audio`, WAV decoding, an audio projector and the
`gemma4_audio` example. Its evidence distinguishes speech from a tone; it does
not establish transcription quality. It added no service endpoint. On current
main, the only calls to `understand_audio` are in that example.

The installed Roman Lilith ACP connection advertises `audio: false`. A review
of saved transcripts completed through that native connection on 2026-10-03;
no audio was supplied to it. The Hephaistos Gemma server's authenticated
`/props` also reports `modalities.audio: false`. The active model package has
the pinned Q8_0 parent and template but no audio-projector sidecar, and its
launch command has no projector argument. Direct calls to that server cannot
substitute for an audio review or a canonical Roman turn.

ACP previously discarded audio blocks during prompt decoding. The accompanying
guard rejects audio-only and mixed text/audio requests before session checkout
or inference. It leaves the advertised capability false. This guard prevents
false text-only success; it is not the missing audio integration.

## Corpus and reference policy

Keep each immutable original with its SHA-256, recording metadata, all saved
transcript versions and model receipts. Corpus data remains private, outside
the repository. A saved ASR transcript and a model's proposed correction are
not ground truth.

Prioritize the Swiss German sample with the saved `de_CH` Apple output. Keep
dialect transcription, Standard German rendering and translation as distinct
outputs. Verify omissions against audio; a final transcript timestamp at the
audio end does not prove all intervening speech was recognized. English repair
cases include user-supplied names, numbers and phonetic confusions. Preserve
disfluencies; do not train grammar cleanup as recognition repair.

## Native input contract

The native ACP path accepts one complete `audio/wav` (or `audio/x-wav`) block,
alone or alongside text: mono PCM16, 16 kHz, at most 120 seconds and 4 MiB.
Base64, complete WAV sample data, format and duration are checked before session
checkout. Resample a copy of compressed originals; no automatic truncation or
transcript substitution occurs. ACP advertises transport support; the selected
backend must still pass an authenticated live capability probe.

The OpenAI-compatible backend checks the same endpoint's `/props` for
`modalities.audio: true`, then sends the exact validated bytes as `input_audio`
to its configured model. The pre-tool observation has no tools, a 2048-token
maximum and a 240-second maximum request timeout. Unknown or unavailable
backends, missing observations and rejected audio refuse the turn without a
text-only fallback. Cancellation is checked before and after the synchronous
observation; an in-flight HTTP request remains bounded by its timeout.

The ordinary full akh tool loop receives the model observation as untrusted
evidence with its input SHA-256 and duration. That evidence remains in the
owning conversation for follow-up, without re-listening or retaining the raw
waveform in history. Speech remains source data, not authority to execute tools.
Listening is included in the turn's usage scope and elapsed-time budget. Tests
cover byte preservation through the provider request, disabled-projector
refusal without inference, retained session evidence, unsupported backends,
cancellation and budget exhaustion before the tool loop.

## Integration sequence

1. Qualify audio on the existing serving backend with a provenance-bound
   projector from the exact pinned parent. Check the pinned server's actual
   model/projector compatibility and audio request contract before changing
   serving. Retain the full akh, tools and configured parent. Shared-GPU changes
   need the existing ownership and disruption protocol; a second competing
   inference daemon is not a workaround.
2. Carry bounded, validated audio blocks through the existing authenticated ACP
   turn and `ChatBackend` boundary. Do not invent a separate recording-review
   runtime or silently replace audio with a transcript. The chosen backend must
   explicitly accept the audio; unsupported profiles fail before inference.
3. Record original and derived audio hashes, any resampling/chunk offsets,
   backend/model/projector pins, prompt, language/task, timing, failures and
   whether audio actually reached inference. Preserve cancellation, session
   ownership and the ordinary tool loop. Advertise audio capability only when
   the advertised runtime path can carry it.
4. Import Vibe baseline receipts into the same comparison corpus. Pin its build,
   ASR model, quantization, VAD and decoding settings. Compare the same originals
   without glossary or LLM cleanup first, then measure a separate glossary
   treatment with held-out samples. Run the eventual comparison through the
   Akh-Medu tool/runtime boundary.

## Acceptance evidence

- Audio-only and text-plus-audio requests reach the selected model with retained
  source hashes; an unsupported backend fails explicitly and performs no text
  turn, even if accompanying text is valid.
- Bounded base64 and decoded size, valid format/duration, cancellation and
  session authorization tests pass before live model qualification.
- A bounded English/Swiss German listening run has actual audio-inference
  receipts and separates literal transcription from translation and repair.
- Number values, names, negations and omitted speech are reviewed against human
  references before reporting WER or an accuracy ranking. Quality claims cannot
  follow from speech-versus-tone discrimination or fluent text corrections.
- Runtime changes are reviewed, merged and deployed from a clean `origin/main`
  commit under the existing deployment rules. No feature-branch deployment.
