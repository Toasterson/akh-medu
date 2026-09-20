# Fable response — advisory, not verified implementation claims

Exact returned review of the supplied brief. See the companion architecture assessment for corrections and acceptance evidence.

# Native runtime architecture review

**Assumptions I'm making** (correct me where wrong): gated-delta recurrent state is constant-size per layer while attention KV grows with position; the rendered prompt for a conversation is deterministic given durable history; F32 mixer state plus globally serialized HIP makes bit-exact reproducibility a realistic target; "efficient continuation" means avoiding full re-prefill on ordinary turn-taking, not sub-token latency work.

## 1. Smallest interface and ownership

Three layers, not two:

- **`ModelArtifact`** (immutable, `Arc`-shared): weights, RoPE tables, tokenizer, later the vision projector. Identity = artifact hashes + device, as today. Zero mutable fields.
- **`SessionState`** (owned, mutable, one per active conversation): per-layer KV tensors, gated-delta convolution history + recurrent tensors, token position, the **token log** (exact token IDs forwarded, in order), a **pending token** slot (sampled but not yet forwarded), and tap output buffers.
- **Mask cache**: evict it from the model. It's shape/position-dependent, so either session-owned or a shared pure cache keyed by `(seq_len, device)`. Model-held mutable caches are the thing that blocks multi-session later.

Interface, close to ADR098:

```rust
trait InferenceSession {
    fn prefill(&mut self, tokens: &[Segment], taps: &TapSpec) -> Result<PrefillOut>;
    fn decode_step(&mut self, token: u32, taps: &TapSpec) -> Result<StepOut>; // logits + requested taps
    fn checkpoint(&self) -> Result<Checkpoint>;   // opaque blob
    fn restore(&mut self, cp: &Checkpoint) -> Result<()>;
    fn token_log(&self) -> &[Segment];
}
```

Two deliberate choices: **sampling lives outside the session** as a pure function on returned logits — the session never knows sampling policy, which makes the pending-token rule (below) and taps clean. And **checkpoints are opaque**, owned by the session type; nobody outside sees raw KV tensors. That single decision is what keeps live KV compression a non-breaking change later.

The ownership rule in one sentence: everything position-dependent is session-owned; everything else is `Arc<ModelArtifact>`. Keep the existing `Arc` owner-identity check on restore.

## 2. Migration without a hidden second stack

The anti-pattern to avoid is a `LegacyModel` wrapper that internally holds a session while the old mutable path stays alive. Instead, make the split the *only* path, in-place:

1. **Mechanical struct split**: `ModelWeights` → immutable `Weights` + `MixerState` holding exactly the fields that mutate today; forward becomes `fn(&Weights, &mut MixerState, ...)`. Rewrite `generate_completion` on top of it in the same change. The compiler enforces the split; existing numerical tests must pass unchanged.
2. **Wrap in `SessionState`** (MixerState + position + token log) and make `generate_completion` "create ephemeral session → prefill → decode loop." Behavior-identical to today (fresh state per call), verified against the same tests.
3. **Persist the session** across calls — that's the feature slice in §6.

Steps 1–2 are pure refactors with existing tests as the oracle. There is never a moment with two forward implementations.

## 3. Single-active-session is the right first slice

Yes. With a 2048-token physical cap, one GPU, and HIP globally serialized, a multi-session GPU cache buys literally nothing today — it's the definition of speculative infrastructure. One resident `SessionState`, keyed by `(conversation id, artifact identity)`; a request for a different conversation drops the resident one (re-prefill at 2048 tokens is cheap; features first).

Correctness it must guarantee:

- **Continuation only on exact match**: artifact identity matches (Arc + hashes + device) *and* the new rendered prompt's tokens exactly prefix-match the token log. Anything else → fresh prefill. Never "close enough."
- **No cross-conversation leakage**: key mismatch = discard, provable by test §6.4.
- **Errors invalidate**: any forward error poisons the resident session and its checkpoints (extends your existing prefill-error clearing).
- **Cancellation leaves defined state** (see §4).
- **Replacement drops session + checkpoints before releasing the allocator cache**, preserving your current ordering.
- **Bounded memory**: one session is O(2048 × 16 attention layers) KV plus constant recurrent state; assert the bound.

## 4. Prefix reuse mechanics

**Token log and compare.** The session records forwarded token IDs. New request: tokenize the full rendered prompt, longest-common-prefix against the log. Log is a full prefix and new tokens extend it → prefill only the suffix. Divergence mid-log → see below. The tokenizer boundary hazard (retokenizing prompt+reply concatenated can merge differently across the boundary) is caught by the exact compare — treat occasional full re-prefills as a counted metric, not a bug to eliminate.

**The recurrent-state constraint** — this is the load-bearing point. Attention KV can be truncated to any position by slicing; the 48 gated-delta layers *cannot* be rewound to an arbitrary position — recurrence only runs forward. So divergent-prefix handling is: restore the nearest checkpoint at or before the divergence point, or full re-prefill. That makes checkpoint *cadence* a policy decision: checkpoint at message boundaries (end of each user/assistant turn), because that's where real divergence happens (edits, regenerations) and it aligns with the daemon's durable structure.

**Pending token.** Invariant: session state reflects only *forwarded* tokens; the token log contains only forwarded tokens. The last sampled token sits in the pending slot and enters state only when forwarded as the next input. On continuation, if the new prompt's next token equals the pending token, it's just the first suffix token — no special case. Never advance state speculatively; this makes checkpoint semantics unambiguous.

**Cancellation.** Take a checkpoint at prefill-end; on cancel mid-decode, restore it. Since a cancelled partial reply isn't committed to durable history (and buffered replies already can't authorize tools), rolling back to prefill-end keeps the session consistent with durable state. Keeping partial decode state is an optimization to skip in slice 1.

**Invalidation.** All checkpoints die on: model/device replacement (Arc identity already enforces this), any forward error, and eviction under a bounded checkpoint count. HIP↔Vulkan switching invalidates checkpoints (device-resident tensors); do not build cross-device migration.

## 5. Sequencing taps, compression, vision — and what to lock in now

Order: **taps → vision → compression**.

- **Taps** are cheapest and satisfy the no-second-pass requirement naturally if `TapSpec` is a parameter of the forward from day one and returns tensors from the same pass. Bake in "logits-only" returns now and you re-thread every layer later. Requirements: zero cost when the spec is empty (no per-layer syncs/copies), and decide *now* whether taps capture pre- or post-mixer activations per layer — changing that later silently invalidates VSA consumers.
- **Vision** breaks the assumption that a prompt is a pure token sequence. That's why the token log should be **segment-structured now** (`Text(tokens) | Image(hash, projector_version)`), even with only text segments implemented. It's a few lines today; retrofitting it means rebuilding the prefix-reuse machinery when vision lands. Image prefix comparison = hash equality.
- **KV compression** last, because it weakens "exact continuation" (compressed KV is at best approximate) and changes checkpoint contents. It stays tractable if (a) checkpoints are opaque, (b) the interface distinguishes **logical position from physical KV length** from day one — your 8K–32K working context over a 2048 physical cap *requires* that distinction, and the gated-delta layers are what make it plausible while compression targets the 16 attention layers.

One current implementation detail that will hurt: **`Tensor::cat` per-token KV append** is O(n²) in copies and makes in-place eviction impossible. Don't build paged buffers now — just keep KV storage strictly private to the session so its representation can change without touching callers.

Explicitly speculative, avoid: multi-session GPU pools, paged-attention allocators, cross-device checkpoint migration, a backend abstraction for hypothetical future frameworks, streaming tap subscriptions.

## 6. First slice and acceptance tests

**Slice**: migration steps 1–2 plus one persistent session with exact-prefix continuation, message-boundary checkpoints, prefill-end checkpoint for cancellation, segment-structured token log, sampling extracted from forward, `TapSpec` present but only the empty spec implemented, and hit/miss metrics with re-prefill cause.

**Acceptance tests**:

1. **Equivalence**: continuation (prefill turn 1 → decode → forward reply + turn 2) produces bit-identical logits to a fresh full prefill at every shared position, on both HIP and Vulkan. If bit-exactness fails, document why and define a tolerance.
2. **Divergence**: edit an earlier turn → mismatch detected, full re-prefill, output equals fresh-session output.
3. **Cancellation**: cancel mid-decode → next request on the same conversation matches fresh-session output.
4. **Isolation**: interleave two conversations → each output identical to running it alone.
5. **Injected prefill error** → session and checkpoints invalidated, next call clean.
6. **Replacement**: swap model or device mid-conversation → continuation refused, memory released (extend the existing resident-idle test).
7. **Cap**: continuation exceeding 2048 fails without truncating durable history.
8. **Restart**: after daemon restart, replay from durable history with a fixed sampler seed reproduces the reply.

Tests 1–4 are the ones that certify the feature; 5–8 certify you haven't regressed invariants you already hold.

