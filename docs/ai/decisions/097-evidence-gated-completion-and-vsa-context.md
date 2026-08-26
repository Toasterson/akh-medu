# ADR 097: Evidence-Gated Completion and VSA Reasoning Context

Date: 2026-08-23
Status: Accepted

Anima: existing “Lane: activate ptah on hephaistos…” work item; ID unavailable
because no Anima project tool is exposed in this session.
Seshat sync: not performed; no `seshat_ingest` tool is available on Hephaistos.
This file is the offline canonical paste source.

## Context

The model-driven agent loop previously accepted any prose completion in an
interactive turn. Autonomous dispatch had a mutation check, but it did not
prove compilation, tests, or learning writeback. In a live Ptah exercise this
allowed the model to stop after an edit, leave integration tests failing, and
omit the promised compiler lesson. VSA tools worked when given exact
namespaced symbols, but VSA associations were not reliably present before the
model chose an action. Ptah also needs the already-built MemGPT-style
`VirtualContextManager` to curate its own long-lived context.

## Decision

Introduce an explicit `<akh:control>` JSON directive parsed at the agent-loop
boundary. It carries an ordered list of labelled evidence gates, where each
gate accepts a successful result from any named tool and may require an exact
subset of its arguments per alternative, plus optional VSA seed labels. The controller is
generic: Rust training is one profile, not a special case in the loop.

Controller state is per turn. Later evidence cannot satisfy an earlier gate,
and a new successful call matching an earlier gate invalidates all later
evidence. Required tool manifests are pinned after retrieval scoping. An
explicit malformed directive, unavailable required tool, or exhausted contract
fails closed without forced synthesis or a completed-turn record. Candidate
narrative is buffered until acceptance. Prompts without a directive keep their
previous completion behavior. The neural loop returns a typed refusal outcome,
which ACP maps to `StopReason::Refusal` for dispatch consumers such as Anima.

Before the first model action, resolve explicit labels, exact prompt keywords,
and namespaced suffixes against the existing symbol registry. Run bounded VSA
spreading activation and inject the top activations with provenance into a
system context block. The block says that it is associative evidence rather
than proof; the completion contract's verifier remains authoritative.

Configure Ptah with `context_strategy = "virtual"`. This uses the existing
core/recall/archival manager and exposes its self-edit and archival tools; the
completion controller remains independent of the selected context manager.

## Consequences

- Callers can define evidence-driven workflows without adding task-specific
  logic to the agent loop.
- Small models receive concrete feedback naming the next missing gate instead
  of being trusted to self-certify; rejected completion prose is never exposed
  as an answer.
- VSA associations influence action selection automatically when resolvable,
  but cannot override compiler or test evidence.
- Controlled prompts are a deliberate protocol. Callers that omit the
  directive retain ordinary chat semantics and receive no completion guarantee.
- Namespaced suffix resolution may scan the symbol registry once per relevant
  turn; output and seed counts are bounded.
