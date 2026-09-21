# Same-pass context signals: bounded architecture question

User wants features first, then performance. Review the following local-source
facts and propose a small, testable next slice (at most 700 words). No tools,
remote operations, code changes, or claims of benchmark evidence.

The native akhrtd now owns Candle Qwen35 with immutable Arc weights and private
InferenceSession state: attention KV, gated-delta/conv state, mask cache, and
forwarded-only token log. One resident conversation is cached. Prefix reuse
requires the entire saved log to match. Divergence re-prefills; no arbitrary KV
truncation. Opaque owner-checked checkpoints restore all mixer state and token
tracking. HIP/Vulkan exact-model gates pass; keyed HIP uses scoped f32 activation
arithmetic because mixed Q8 decode/f32 prefill failed recurrent-state parity.
GGUF Q4_K_M weights remain unchanged. Limits apply to full prompt+output reserve.

Next required feature: target-layer taps at zero-based blocks 5/19/33/47/61 from
the SAME existing forward, GPU pooling of a selected query/span, transferring
only a 5120-vector (or final 10K-bit HyperVec) rather than token activations.
Use a structured/low-rank projector, not a 205 MB dense bridge per session.
Query signals should rank VSA retrieval and materialize memory before decode;
completed-span signals should be archived only when the turn is committed.

Actual source seam: run_hidden_states(tokens,index_pos) loops all layers then
returns only the final position after final RMS norm; prefill chunks at 128.
resume_prompt returns last logits and reuse count. forward_token shares this
same path. Currently no taps. Public extract_hidden_state does a second prefill;
we must NOT call it and claim same-pass capture. Normal untapped generation must
remain unchanged. Capture must survive checkpoints/rewind correctly or be
explicitly transaction-scoped; cancellation must not publish partial signals.

Existing ContextManager is synchronous assemble(&self, EpisodeState,budget,
counter) and commit(&mut self,TurnOutcome). EpisodeState borrows Engine and has
already-retrieved text, not native hidden signals. VirtualContextManager renders
core memory near the stable head and dynamic context before the final user turn;
its VSA stats honestly remain zero today. TurnOutcome has counts/status only.
Native akhrtd's brokered loop currently has durable messages, host tool calls and
receipts, not a borrowed Engine or that legacy agent manager. Do not create a
second cognitive core or assume a mutable manager is already in the native loop.

Main design tension: query signals exist AFTER processing query tokens. Existing
retrieval placement before the query would change that processed prefix and
force re-prefill. How should the native path stage query prefill, bounded recall,
and response prefill without pretending duplicate work disappeared or changing
tool/safety authority? Compare append-only memory context versus explicit
checkpoint/re-prefill, separating the tap primitive from retrieval integration.

Recommend the smallest initial capture API and its tests (pooling/chunking,
session isolation, cache hits, cancellation, checkpoints, zero extra model
forward), then the integration seam. Flag unsupported assumptions. No need for
a general plugin framework or a new tensor backend trait.
