# Native runtime architecture review brief

Requested by the user: features first, performance later; ask Fable for design
hints. Review only the supplied material. Do not use tools or change anything.
Your answer is advisory, not test evidence. Call out assumptions and distinguish
necessary feature architecture from speculative infrastructure.

## Current implementation and requirements

Rust akhrtd owns Candle inference in process, replacing Ptah's external inference
runner. Exact Qwen3.8-27B Q4_K_M (GGUF qwen35), 48 gated-delta recurrent and 16
full-attention layers. HIP and Vulkan work on RX7900XTX; both passed numerical,
resident-idle and isolated daemon/tool/restart tests. HIP->Vulkan->HIP in one PID
also passed. All HIP compute is globally serialized on the default stream to
avoid a reproduced private-stream idle-power bug. Performance tuning is deferred.

The session daemon owns authoritative durable conversation and approval/tool
receipts. Native continuation pins include artifact hashes, device and limits.
akhrtd currently caches one mutable CandleBackend under a process-global mutex.
Weight identity excludes per-call limits and logical names. Model replacement
drops the previous backend and releases unused HIP allocator cache before loading
another model/device. No silent CPU fallback. Input+output reservation beyond a
2048-token physical cap fails rather than truncating durable history.

Every generate_completion tokenizes the entire rendered prompt, starts bounded
128-query prefill at zero, samples, and decodes. The last sampled token is not
necessarily forwarded. Cancellation is checked between chunks/steps. Prefill
errors clear partial model state. Replies are buffered so truncated/malformed XML
cannot authorize tools. Tools execute only through the canonical approval broker.

Qwen ModelWeights currently combines immutable weights with mutable mixer state.
Attention has optional key/value tensors; append uses Tensor::cat, both F32.
GatedDelta has convolution history, recurrent tensor and position. Existing
immutable checkpoints clone tensor handles for all mixer state and hold an Arc
owner identity; restore rejects another loaded model. Updates are out of place.
The current mask cache is model-held; weights and RoPE are resident. Checkpoints
are tested but not wired into native per-conversation calls.

Accepted ADR098 calls for a small owned InferenceSession interface: prefill,
extension/decode, checkpoint/rewind; all position-dependent state session-owned.
Required features still missing: session isolation with efficient continuation,
hidden taps/VSA signals from the existing forward (no second inference pass), exact
vision projector integration, live KV compression, controlled 8K-32K working
context, and eventual canonical homelab cutover. 100K is a later pressure ceiling.
No speculative decoding or new inference framework is wanted now.

## Please advise

1. Recommend the smallest useful model/session interface and explicit ownership
   of weights, recurrent/KV state, hidden taps, token position and sampling.
2. Explain a safe incremental migration from the current mutable model without
   hiding a second execution stack behind compatibility wrappers.
3. Decide whether bounded single-active-session state is a sound first slice
   before a multi-session GPU cache, and what correctness it must guarantee.
4. Detail exact-token prefix reuse, last-sampled-token handling, cancellation,
   divergent prefixes, checkpoint invalidation and model/device replacement.
5. Sequence hidden taps, compression and vision; identify architectural choices
   now that would make those features difficult later.
6. Propose a first bounded feature slice and acceptance tests. Keep the review
   under 1500 words; be concrete about tradeoffs, not a framework wish list.
