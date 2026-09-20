# Native exact-prefix continuation

Features-first continuation of the Fable-guided ownership split. Local only;
no production service installation, shared restarts or kernel tuning.

## Interface and invariants

`InferenceSession::resume_prompt` receives the complete authoritative token
sequence. `Some(TextState)` records only forwarded tokens and the corresponding
logits. Exact whole-prefix match permits bounded suffix prefill; identical input
returns saved logits. Divergence, shorter input or legacy tensor forwards cause
fresh prefill. An invalidated log is `None`, never an empty reusable prefix.
`forward_token` records a sampled token only after successful forward. Opaque
checkpoints restore token tracking, logits and mixer state together, and retain
the session-specific owner check.

`CandleBackend::generate_completion_for_session` keeps one active key. Switching
keys drops mutable inference state while sharing weights; unkeyed calls always
start fresh. The same sampling loop serves both entry points. The native daemon
key serializes canonical session ID, provider-conversation ID and runtime binding.
Full prompt plus output reservation is checked before reuse. All generation
errors, cancellation and length exhaustion discard the cache; native XML parsing
failure also discards it before releasing the compute lock. History/approval/tool
authority remains in the durable session loop, never in the cache.

## Follow-up Fable consultation

The requested `claude-fable-5` returned `is_error=false` through the local CLI,
with tools, MCP, customizations and persistence disabled. It reviewed the
[bounded brief](../plans/2026-09-14-native-continuation-fable-brief.md), not files or
credentials. Raw output is `/tmp/ptah-fable-continuation-review.json`.

Its principal concerns were checked against source:

- Invalidated tracking must force complete state reset. The implementation uses
  `Option<TextState>` and a position-zero prefill resets every mixer and mask.
- Partial mutation must not leave a valid log. Tensor forward invalidates tracking
  before layer mutation; tracked decode takes the log before computing. Controlled
  failures reset state. A panic in the native compute lane poisons its mutex and
  requires restart; it cannot resume a coherent-looking partial cache.
- Snapshots must not be forgeable by clients. They are private in-process Rust
  values, not serialized client blobs or RPC handles; Arc ownership rejects
  sibling/reloaded sessions. No new MAC/handle-table mechanism is needed here.
- Tokens, logits and mixer state must rewind together: the checkpoint carries all
  three. Native output reservation bounds the same full prompt used for suffix
  reuse; no extra forward occurs after consuming the final output token.

Keeping coherent caches after invalid XML/length exhaustion remains a possible
later optimization; conservative discard is deliberate for this first slice.

## Verification checkpoint

Fourteen CPU model tests pass, including extension, equal prompts, forwarded-only
decode accounting, snapshot rewind, divergence, legacy-call invalidation,
context rejection and cancellation after partial extension/decode. Initial
extension and tracked-decode tests failed on unsupported stubs before their
implementations. The exact-model continuation gate, public completion-interface
probe and daemon rebuild/integration checks are in progress. Do not treat this
checkpoint as deployment or numerical acceptance.

## Numerical gate: initial HIP failure

The exact-model cached-continuation versus fresh-prefill test FAILED with
`AKH_HIP_Q8_GEMV=1`, recurrence fusion off (103.58 s including artifact hash/load).
It reused 11 forwarded tokens, appended four prompt tokens, then compared 33
forced decode positions. Initial prompt-logit cosine was 0.999983553. Later
minimum hidden cosine was 0.614551019 and logit cosine 0.957467716; greedy IDs
still matched. This fails the existing 0.9999 gate and is not acceptable merely
because generated tokens agree. No production cutover occurred.

Controls in progress: Q8-off HIP, then Vulkan with accepted defaults. Ranked
hypotheses are activation-precision mismatch between decode and fresh prefill,
batch-shape rounding amplified by recurrence, and an append-position/mask defect.
The exact-model test is retained; the tiny CPU fixture alone did not reveal this
failure. Full artifact verification/load dominates this roughly 100-second loop.

Controls passed: Q8-off HIP (98.83 s) and default Vulkan (115.50 s). Prompt logits
and all 33 subsequent hidden/logit cosines round to 1.0; all greedy IDs match.
They are not bit-identical across differing prefill shapes. This isolates the
observed failure to the mixed Q8-decode/full-precision-prefill route, rather than
the prefix offset or mask accounting. No acceptance threshold was changed.

The fix scopes full-precision activation arithmetic to synchronous keyed Qwen
HIP completions. A thread-local RAII guard suppresses Q8 only inside that call;
it changes neither GGUF weight compression, environment variables nor streams.
Legacy/unkeyed experiments retain their existing Q8 policy. The scope test went
red on a no-op implementation, then passed nesting, cross-thread independence
and unwind restoration with process-wide Q8 enabled. Exact-model guarded replay,
public completion checks and four-daemon integration remain pending. Set
`AKH_QWEN35_RAW_Q8_CONTINUATION=1` only in the diagnostic test to reproduce the
rejected raw-Q8 combination; normal native calls do not use that escape hatch.

Guarded HIP replay passed (104.34 s) with global Q8 enabled: 11 reused tokens,
prompt and all 33 hidden/logit cosines rounding to 1.0 and all greedy IDs equal.
The combined HIP/Vulkan `akhrtd` and public probe build passed (4m48s).
The public HIP completion probe passed all six behavioral checks and observed
24 reused prompt tokens. After those checks, 60 resident-idle samples averaged
0.433% GPU busy and 11.9 W against an 11 W unloaded baseline: PASS.
Vulkan public-interface and updated four-daemon prefix-reuse checks follow.

The Vulkan public probe also passed all six checks with 24 reused prompt tokens.
Its 60 resident-idle samples averaged 0.717% busy and 12.917 W against a 10.667 W
baseline: PASS. Neither idle run overlapped compilation or another GPU test.
The updated isolated HIP -> Vulkan -> HIP daemon canary is now running.

The updated four-daemon canary PASSED: authentication, real model write held for
approval, session restart with pending approval, approved write/read and durable
receipts, provider binding, actual native prefix reuse (maximum 970 tokens),
HIP -> Vulkan -> HIP in one runtime PID, then runtime restart and exact artifact
reload with correct conversation continuation. The temporary test processes and
workspace were cleaned by the harness; shared services were not touched. This
run used the completed continuation build before the subsequent tap primitive.

A second Vulkan-primary canary with the rebuilt tap-capable source also PASSED
all normal tool/approval/receipt and restart checks, including 972 reused tokens
observed specifically during ordinary continuation turns. Same-pass capture
remains a separately tested model primitive, not enabled daemon retrieval.
