# Same-pass context signals: Fable assessment

The local tool-disabled `claude-fable-5` review completed successfully against
the [bounded source brief](../plans/2026-09-14-native-taps-fable-brief.md).
Raw response and model metadata: `/tmp/ptah-fable-taps-review.json`.
This is architectural advice, not numerical or retrieval-quality evidence.

Accepted for the next primitive:

- Capture residual streams after selected complete blocks, before the next
  block's normalization, inside the existing target forward.
- Pool only selected absolute token positions on the device; preserve layer
  identity until a deliberate projector combines them.
- Make untapped execution unchanged and test that tapping adds no target
  forward. Chunked pooling needs count-weighted accumulation, not an average
  of chunk means.
- A span already in the reused prefix is an explicit capture miss unless its
  signal was separately retained. Never secretly re-prefill to report a hit.
- Cancellation must not publish partial signals. Keep the first collector local
  to one controlled operation, returning a value only on success; no persistent
  collector or archival side effect is necessary for this primitive.

Integration decisions remain gated:

- Append-only recalled context after the query avoids reprocessing query tokens,
  but changes the existing manager's memory-before-query layout. It needs a real
  quality comparison and exact chat-template placement before acceptance.
- Rewind plus memory-before-query re-prefill is an honest alternative, not a
  same-pass optimization. Recurrent state forbids arbitrary prefix slicing.
- Do not instantiate a second Engine inside the native runtime. Inspect the
  session/cognitive-core contract before wiring retrieval or durable archival.
- Fable referred to checkpoint serialization and an existing native commit path;
  neither should be assumed. Checkpoints here are opaque in-process values, and
  the current native brokered loop does not invoke legacy ContextManager::commit.
- Low-rank dimensions, pooling quality and projection training are unverified.
  No random projector should be claimed to improve recall without evaluation.

Hidden taps, VSA materialization, commit integration and nonzero virtual-context
statistics remain unimplemented at this checkpoint.

## Verified existing integration contract

Subsequent source inspection found an existing durable completion route:
`akhsessd::publish_completions` sends completed turns to the configured core's
`/v1/turns/accepted`, records its receipt, and retries only undelivered turns.
The core's `accepted` handler fingerprints the request, rejects conflicting IDs,
respects transcript-capture policy, journals accepted transcripts and persists
the Engine. It does not currently call ContextManager::commit or accept hidden
signals. This is a candidate archival integration seam, not an absent channel.

Query context currently runs earlier: akhsessd calls `/v1/context` once before
durable dispatch and stores the response. It receives symbolic KG/recent episode
text from the core and binds the authoritative model. A neural query produced
inside akhrtd therefore cannot simply be added to that pre-dispatch request:
it needs an explicitly staged, scoped retrieval exchange or a separately
evaluated query-encoding path. Reusing the ordinary completion callback as if it
were pre-decode retrieval would be incorrect.

The shared protocol types deny unknown fields. Any future signal transport must
address capability/version compatibility, bounded payloads, model/projector
identity, session/turn ownership and durable acceptance together. No shared
protocol or core-service behavior was changed for the pooling primitive.
