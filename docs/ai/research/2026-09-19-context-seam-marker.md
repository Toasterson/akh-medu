# The ContextManager seam had a silent fallback, and the topology fork is real

Date: 2026-09-19.

## What the seam actually looks like

Tracing the path end to end rather than reasoning about it:

- `src/core_service.rs` `assemble_context` runs `SafetyEnforced(SelectingManager)`
  with a heuristic counter. Implemented, tested, and reached from `/v1/context`
  **only when the request carries messages**.
- `crates/akh-session-services/src/server/sessions.rs` builds that request with
  `messages: Vec::new()`, with a comment saying the core assembles once this
  carries the session's messages.
- `crates/akh-session-services/src/internal_native.rs` already has the receiving
  half: `native_window::core_budgeted(messages)` decides between
  `native_window::verify` - which checks the fit against the model's own
  tokenizer and **fails closed, never dropping** - and `native_window::select`,
  the local evictor. Its comment states the intent plainly: *"akhomed owns
  eviction once it assembles to budget; the runtime then only checks the fit."*

So both ends exist. What was missing is the join.

## The latent fallback

`context_budgeted` appeared in exactly one place in the tree: `native_window.rs`
and its own tests. **The core never emitted it.**

That matters beyond the missing wiring. Once the topology lands and messages
start flowing, the core would assemble, return its decision, and the runtime
would read `core_budgeted == false` and run its own selector over a list the
core had already decided - re-evicting on top of it. Two context policies
disagreeing, which is the exact thing this seam exists to prevent, and it would
have happened silently: the turn still succeeds, just with a different context
than the core chose.

`assemble_context` now marks every message it returns. The renderer reads only
`role` and `content` by key, so the marker cannot leak into the prompt text, and
a test asserts both fields survive.

### A marker echoed back in history

The hazard of putting the flag on messages is that a previous turn's assembled
output, fed back as history, would describe an assembly that did not happen for
the current turn. The core therefore strips any inbound marker before deciding
anything, so a surviving marker is always one this assembly set. A test drives a
stale marker in, forces an eviction, and asserts every surviving message is
marked by the current assembly.

Note the direction of the residual risk. If a stale marker ever did reach the
runtime on an unassembled list, `verify` fails the turn on overflow rather than
truncating - a false failure, not silent context loss. Fail-closed is the safe
direction here, which is why this is a hazard worth removing rather than a
correctness hole.

### One spelling, one place

The constant now lives in `akh-session-types` as `CONTEXT_BUDGETED` and both
crates use it. As two independent string literals, a change on one side would
not fail to compile - it would silently turn assembly off and restore exactly
the double-eviction described above.

## The topology fork is real

It is not resolvable by reading the code, and it is a genuine architectural
choice:

- `RuntimeJobRequest` carries `input`, `instructions` and `tools`. It has **no
  message array**. akhsessd holds turns, not a rendered conversation.
- The message array is built in `internal.rs`, in the agent loop, which has no
  core client.

So no party holds both the conversation and a line to the core. Either the agent
loop gains a core client, or the conversation moves into the session record.
Everything above is needed under both, which is why it was worth doing first.
