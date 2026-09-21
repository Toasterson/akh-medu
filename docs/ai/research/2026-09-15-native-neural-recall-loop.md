# Native tap → projection → retrieval → prompt loop

First end-to-end neural-recall slice on top of the same-pass taps and the
core span annex. Local only: no shared daemon, production database or remote
host was changed, and no production native daemon is installed.

## What the loop does per turn

1. **Capture.** On the first model iteration of a turn, `akhrtd` renders the
   prompt and asks the keyed completion to pool the post-block residual
   streams over the newly forwarded user-input span (five target blocks:
   5, 19, 33, 47, 61 on the 64-block model; `CapturePolicy::for_model`).
   The span is the trimmed accepted input, bounded to 8192 bytes, mapped onto
   exact token boundaries through tokenizer offsets. A span that is not
   token-aligned or already inside the reused prefix is an explicit miss.
2. **Project.** `nlu::span_projection` turns the five pooled f32 means into
   exactly 10000 bits: per layer L2-normalise, zero-pad 5120 → 8192, seeded
   randomized Hadamard rotation (distinct seed per layer), first 2000
   coefficients, sign. Policy and projector recipes are canonical strings
   whose SHA-256 values are the `policy_sha256` / `projector_sha256` pins; the
   model and tokenizer pins come from the pinned `NativeCandle` artifacts.
3. **Recall.** Still inside the compute lane, before decoding, `akhrtd` sends
   the vector to `akhsessd` (`POST /v1/runtime/recall`, job identity checked
   by `validate_runtime_call`). The session daemon owns the core principal and
   akh/session scope, checks the core's `neural_spans_v1` capability (cached
   60 s), and forwards a `SpanQuery`. A daemon or core without support, or a
   capture policy that forbids recall, yields `supported: false`; the runtime
   continues without recall. Recall failures never fail the turn.
4. **Materialize.** Hits whose text is already visible in the prompt before
   the user turn are skipped. Remaining hits (at most 4, 4096 bytes) become one
   `system` message placed directly before the user turn. The model is rewound
   to the pre-turn checkpoint and the re-rendered prompt is forwarded, reusing
   the unchanged history prefix. This is the honest memory-before-query layout,
   not a same-pass splice. The inserted message is persisted in the durable
   runtime conversation so later prefix reuse sees exactly what the model saw.
5. **Deliver.** After the reply parses, the capture (user source, byte range
   of the accepted input, pinned space, vector) goes to `akhsessd`
   (`POST /v1/runtime/spans`), validated against the turn's accepted input and
   retained per turn. `publish_completions` attaches it to the accepted turn
   only when the core advertises the annex; otherwise the legacy payload and
   its operation fingerprint are unchanged.

Only user-input spans are captured. Assistant output is decoded token by
token outside the prefill collector and stays uncaptured in this slice.

## Boundaries kept

- One cognitive core: recall is a bounded exchange to the existing annex; no
  second Engine or index in `akhrtd`. Authoritative history stays in the
  durable session loop and the runtime conversation record.
- Compatibility: `RuntimeJob`/`RuntimeEvent` are unchanged. The two new
  callbacks are discovered by probing; an older `akhsessd` answers HTTP 404
  and the runtime degrades to the pre-recall behaviour. `AcceptedTurn`
  carries `span_captures` only after capability discovery.
- Cached spans never replay: iterations after tool results find the user
  turn in the reused prefix and report a miss, so at most one capture and one
  recall exchange happen per turn.
- Pins are representation identity, not attestation. HIP and Vulkan share
  one space; cross-backend projected stability is still a separate gate.
- No calibration data or raw means are retained anywhere.

## Verification

- `nlu::span_projection`: 7 CPU tests (pins, layer selection, determinism,
  scale invariance, near/far Hamming ranking on synthetic input, rejection of
  degenerate/mismatched means, packing).
- `candle_backend::tests::capture`: 4 CPU tests on an eight-block fixture
  (pooled means match the known embeddings; cached spans and unaligned spans
  are misses; a reprompt rewinds to the pre-call state and the resident state
  then matches the rewritten prompt; recall failure discards the cache).
  All 20 Qwen3.5 model tests still pass with the fixture parameterised.
- `akh-session-types`: recall/delivery validation tests (2).
- `akh-session-services` (`candle-runtime`, CPU fixture whose reply is `ok`):
  a three-turn run proves capture delivery with the exact byte range and
  pins, the recall query reusing the same vector, dedupe of already-visible
  text, a recalled block persisted as a system message before the user turn,
  and graceful operation when the host lacks recall.
- `akhsessd` (`session`): recall rides the session scope to a fake core with
  authenticated job identity; deliveries are validated and attached to the
  accepted turn exactly once; an older core without `/v1/capabilities` gets
  the legacy payload and no query. The other 4 lifecycle tests pass.
- Regression: 17 store tests, 4 shared-type tests, 2 core HTTP tests,
  akh-runtime tests, `git diff --check`.

## Exact-model gate

See the ledger entry for the result of
`installed_artifact_span_projection_ranks_paraphrases_above_unrelated_text`
(pinned artifacts, HIP): determinism across fresh sessions, every paraphrase
pair closer than every unrelated pair, unrelated pairs not near-duplicates.
This is a same-pipeline similarity check on six sentences, not a retrieval
quality benchmark or grounded symbolic recall.

## Open items

- Exact chat-template placement of the recalled block (a mid-conversation
  system turn) needs a quality comparison against alternatives before
  acceptance; the current placement is the minimal honest layout.
- Assistant-output capture, projection calibration, held-out paraphrase and
  memory-answer fixtures, cross-backend projected stability, and the real
  ContextManager commit/materialization remain outstanding.
- Within-session recall only adds context once history is no longer fully
  visible (retired/replaced continuation, future compaction); today's full
  histories dedupe every hit.
