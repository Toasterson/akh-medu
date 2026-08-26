# Evidence-Gated Completion Controller

Date: 2026-08-23

Anima: existing “Lane: activate ptah on hephaistos…” work item; ID unavailable
because no Anima project tool is exposed in this session.
Seshat sync: not performed; no `seshat_ingest` tool is available on Hephaistos.
This file is the offline canonical paste source.

## Outcome

Make self-training and other tool workflows finish only when their caller's
observable post-conditions have passed. Put VSA spreading activation directly
into the LLM's pre-action context, while keeping compiler/tests or another
domain verifier as the proof boundary. Enable Ptah's existing MemGPT-style
virtual context manager so it can curate core, recall, and archival memory.

## Interface

A controlled request contains one fail-closed JSON directive:

```text
<akh:control>{"stages":[{"label":"mutation","any_of":[{"tool":"edit"},{"tool":"file_io","arguments":{"action":"write"}}]},{"label":"checked","any_of":[{"tool":"compile_feedback","arguments":{"command":"check"}}]},{"label":"tested","any_of":[{"tool":"compile_feedback","arguments":{"command":"test"}}]},{"label":"learned","any_of":[{"tool":"memory_write"}]}],"vsa_seeds":["rust:typestate"]}</akh:control>
```

The controller is domain-neutral. Each ordered stage is satisfied by one
successful call matching a typed tool predicate whose arguments contain an
optional exact subset. A later-stage call made early is not banked; re-running
an earlier stage invalidates all later evidence. All evidence is fresh because
controller state exists only for the current turn. Candidate narrative remains
provisional until acceptance; exhaustion and unavailable tools fail closed and
ACP reports `StopReason::Refusal` rather than a successful end turn.

## Slices

1. Parse and validate explicit contracts; malformed explicit control fails
   closed and ordinary prompts retain their existing behavior.
2. Pin evidence-producing tools into a retrieval-scoped catalog, observe real
   `ToolOutput.success`, and reject premature completion in both interactive
   and dispatch turns.
3. Resolve explicit and natural-language KG seeds, run bounded VSA spreading
   activation, and inject activations with provenance as advisory system
   context.
4. Set Ptah's per-akh `context_strategy = "virtual"`, document the Rust
   training profile, deploy the native systemd daemon, and run a real loop.

## Acceptance

- Unit tests prove ordered, successful, fresh evidence semantics and malformed
  directive handling.
- An agent-loop test proves an interactive model cannot finish early and that
  the host context reaches the model.
- Non-completion exits with an open contract return a typed refusal through
  ACP, so Anima cannot mistake an exhausted executor for completed work.
- A VSA test proves a namespaced Rust seed activates a related symbol with
  provenance.
- Ptah starts under systemd with the virtual context manager and its core-memory
  tools, then completes a compiler-verified training loop and writes a memory.
