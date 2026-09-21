# Native runtime: Fable architecture consultation

User priority: feature completeness before further performance optimization.

The local Claude CLI successfully returned a review using `claude-fable-5`
(`is_error=false`; response metadata names that model). Tools, MCP and session
persistence were disabled; safe mode excluded project hooks/customizations. The
review received the [written brief](../plans/2026-09-14-native-runtime-fable-brief.md),
not repository access. [Returned review](2026-09-14-native-runtime-fable-response.md)
is retained alongside this assessment; raw CLI metadata remains at
`/tmp/ptah-fable-architecture-review.json`.
This is design advice, not numerical verification or deployment approval.

## Accepted guidance

- Share immutable model weights; explicitly own attention KV, recurrent and
  convolution state, positions and masks per inference session.
- Make this the existing forward's implementation, not another execution stack.
- Keep checkpoints opaque, so future KV representations remain private.
- Start runtime continuation with one resident conversation and bounded state;
  re-prefill on conversation/identity mismatch rather than adding a GPU cache.
- Compare exact forwarded token IDs for prefix reuse. Sampling remains outside
  model forward, and sampled-but-not-forwarded tokens do not advance state.
- Recurrent state cannot be arbitrarily truncated like attention KV. Divergence
  needs an applicable full checkpoint or fresh prefill.
- Sequence session ownership/continuation, same-pass hidden taps, exact vision,
  then live KV compression and validated larger-context operation. Revisit
  kernel throughput only after feature gates are complete.

## Corrections and deliberate limits

- Fable suggested a mask-cache key omitting prefix length. Actual causal masks
  depend on query length AND full KV length; retain that key, session-local.
- Do not assume bit equality for different prefill chunk shapes or different
  backends: reduction order already differs. Interleaved versus serial execution
  with identical shapes must match exactly; broader comparisons retain established
  numerical gates rather than inventing a looser tolerance after failure.
- Compression does not authorize logical context beyond configured/model limits.
  Validate logical positions and physical allocation independently; the canary
  stays capped at 2048 until explicit long-context tests pass.
- No pretend image support or unused general backend trait. Future multimodal
  prefix identity must include processed image/projector identity and positional
  layout, not just equality of source-image hashes.
- Fable's first slice bundled several features. Split it further: first prove
  immutable weights and isolated state using the existing execution path; then
  add transactional extension/token accounting and daemon continuation.

## First implementation slice

Qwen's `InferenceSession` now contains `Arc<Weights>`, per-layer state, a private
mask cache and session-specific checkpoint identity. Both mixer forwards take
immutable weights plus explicitly mutable state. Native Candle dispatch uses
this type directly; `ModelWeights` is only a type alias for existing probes.

The CPU interleaving test first failed on unsupported session creation, then
passed after the split. Two sessions share weights, match isolated execution,
reject cross-session snapshots, and survive dropping the original session.
All nine prior CPU regressions also pass. Persistent native
conversation reuse, hidden taps, vision and live compression are not implemented
by this structural slice.

Hardware follow-up: the exact pinned model passed HIP session isolation in
104.08 seconds, with two sessions and 33 positions each. Full hidden/logit vectors
were bit-identical between serial and interleaved execution. HIP Q8 was enabled
and rejected recurrence fusion disabled. This is correctness evidence, not a
throughput measurement (a CPU runtime build overlapped the test). Vulkan follows
serially with its accepted default routes.

Vulkan follow-up PASSED in 118.11 seconds with the accepted defaults: the same
two-session/33-position test matched all hidden/logit bits. The combined native
HIP/Vulkan `akhrtd` release build also passed (3m02s). Ten CPU model tests passed
in total, and `git diff --check` passed. Existing daemon call behavior remains
fresh-prefill per call; no production installation or shared restart occurred.
