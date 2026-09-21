# Native same-pass span taps

First primitive following the [Fable assessment](2026-09-14-native-taps-fable-assessment.md).
It is not yet retrieval, VSA projection, archival, or a production feature toggle.

`InferenceSession::resume_prompt_with_taps` selects an absolute nonempty token
span and up to five distinct target blocks. It returns normal logits/reuse count
and optional device-resident per-layer means of post-block residual streams.
The existing bounded prefill loop performs the capture; there is no second target
forward or CPU activation readback in the implementation. Ordinary untapped
calls take the same path with no collector.

The operation-local collector sums selected token vectors in f32 on the device,
then divides by the total token count. It never averages chunk means equally.
Layer identity is retained for later deliberate projection. The target model's
five desired indices are 5, 19, 33, 47, and 61; smaller test models select their
own valid indices. This interface permits at most five pooled vectors, not an
unbounded hidden-state dump.

If any requested position is already in the reused prefix, capture returns None
while generation still reuses that prefix. It neither pretends a partial span is
complete nor silently replays cached tokens. No previously returned signal is
stored in the checkpoint; a result is an operation-local value, not proof that a
turn has been committed. Future archival must associate it with the durable turn
and reject cancelled/rewound work at that integration seam.

Controlled failure drops the collector and resets partial inference state.
Invalid span/layer requests reject before mutation. Existing owner-checked
checkpoints can still restore text/mixer state; subsequent newly forwarded
suffixes can be captured again.

## Tests

- Initial selected-span test failed on the unsupported method, then passed.
  The zero-mixer fixture has independently known residual values: two ones and
  one zero must mean 2/3 in every hidden dimension.
- Unequal prefill chunks test crosses position 128 and distinguishes correct
  token weighting (2/3) from erroneous chunk weighting (3/4). The initial fixture
  needed its GGUF context metadata extended so its RoPE table also covered 130
  tokens; changing only the session's numerical limit was insufficient.
- Cache-only and partially cached spans report misses; complete new suffixes
  capture after rewind. Sibling sessions remain independent.
- Invalid requests preserve valid state; cancellation after actual suffix
  capture returns no result, resets state, and permits clean recovery/rewind.
- All 18 CPU model regressions pass (the general harness additionally reports
  four path-conditional artifact tests as passing no-ops without model env).
- Exact-model HIP/Vulkan tap-state gates are built and await the serial GPU
  window after the continuation daemon canary. They require finite, nonzero
  5120-vectors at all five layers and bit-identical untapped logits plus 33
  subsequent hidden/logit positions. No hardware acceptance claimed yet.

No throughput, retrieval quality, live virtual-context statistics, projector
quality, or completion-span archival claim follows from these primitive tests.

Exact HIP tap-state gate PASSED in 102.30 seconds including pinned artifact
hash/load. All five pooled 5120-vectors were finite and nonzero; prompt logits
and all 33 subsequent hidden/logit positions matched the untapped session bit
for bit. Global Q8 remained enabled with the same scoped f32 policy as native
continuation, and recurrence fusion disabled. A CPU runtime build overlapped:
this is correctness evidence, not throughput evidence. Vulkan follows serially.

Exact Vulkan tap-state gate also PASSED (113.02 seconds): the same five pooled
vectors and bit-identical prompt/33-step hidden/logit comparison. Combined
HIP/Vulkan `akhrtd` plus the public session probe rebuilt successfully (3m08s).
The next isolated Vulkan-primary HTTP canary uses this updated binary; taps are
not yet enabled by the daemon, while its untapped continuation shares the
refactored forward path.

The updated Vulkan-primary four-daemon canary PASSED: authentication, real
write/approval/session restart, read/receipts, stable binding, ordinary-turn
prefix reuse (972 tokens), and runtime restart with exact artifact reload.
The cache-hit assertion examines only the ordinary-turn log interval, excluding
earlier tool turns. Test-owned processes and temporary state were cleaned up;
no production service was changed. This validates the rebuilt untapped runtime
path; tap-to-retrieval/archival integration remains outstanding.
