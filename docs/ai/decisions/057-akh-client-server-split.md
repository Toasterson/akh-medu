# ADR 057 — `akh` is a client of `akhomed`; retire the `client-only` feature in the ACP collapse

> Date: 2026-06-15
> Status: Accepted
> Phase: P3 (Pure-ACP Agentic Harness pivot); executed in the P3/P8 mode-collapse
> Supersedes: the build-model half of [ADR-023](023-client-only-mode.md) (client-only build mode)
> Depends on: ADR-053 (first-class AkhConfig)
> Plan: [`docs/ai/plans/2026-06-14-pivot-acp-agentic-harness.md`](../plans/2026-06-14-pivot-acp-agentic-harness.md)

## Context

In the pure-ACP pivot the unit of agency is an **akh** served by the daemon
**`akhomed`** (the ACP *agent*). The `akh` CLI and the TUI are ACP *clients*.

Today the build model contradicts that framing. ADR-023 introduced
`client-only` as an **opt-in** Cargo feature that *strips* the embedded engine
from `akh`, on the premise that `akh`'s **default** is to embed a full local
`Engine`/`Agent` (HNSW, redb, egg, petgraph, oxigraph, candle) and run
standalone, with a `Local`/`Remote` split inside `AkhClient`. That leaves three
problems:

1. **The default is backwards.** `akh` is inherently a *client* — it requires a
   running `akhomed` to do anything real. Embedding a full engine by default
   (and treating the lean client as the exception behind `client-only`) inverts
   the actual architecture.
2. **A divergent, half-maintained code path.** `client-only` gates ~73 `cfg`
   sites and a separate `run_client_only` dispatch that drifts from the normal
   dispatch — it was already non-exhaustive for `Commands::Cluster` and
   `SkillAction::{Export,Import}` (the `client-only` build did not compile),
   precisely because the parallel path is easy to forget when commands are added.
3. **It was missed in the pivot ADRs.** The plan says "the CLI becomes an ACP
   client," but never stated that this *is* the `client-only` model promoted to
   the default, nor that the embedded-engine local mode is on its way out.

## Decision

Affirm the architecture and record the migration; do **not** rip it out now.

- **`akh` is a client of `akhomed`.** It requires the server to run and may be
  **compiled independently** as a lean binary (no embedded engine). `akhomed`
  is the server (ACP agent). This is the intended steady state — the
  `client-only` behaviour is the *correct* behaviour, not an exceptional one.
- **The embedded-engine local mode and the `client-only` feature flag are
  transitional artifacts** to be removed in the pivot's **mode-collapse** (P3
  finish → P8), where the local, REST/WS, and (future) ACP client paths collapse
  into a **single ACP client path**. At that point:
  - `AkhClient::Local(Engine)` and the `#[cfg(not(feature = "client-only"))]`
    engine-embedding paths are deleted;
  - the parallel `run_client_only` dispatch is merged into the one client
    dispatch;
  - the `client-only` Cargo feature is dropped (its behaviour becomes the only
    behaviour);
  - `akh` talks to `akhomed` purely over ACP (`session/prompt` for chat, `_akh/*`
    for the data-plane), per the pivot plan.
- **Until the collapse, the tree must compile in every shipped configuration.**
  The minimal fix lands now: make `--features client-only` build again by giving
  `run_client_only` the two missing arms — `Skill {Export,Import}` are local
  filesystem operations (implemented locally, as `Scaffold` already is) and
  `Cluster` requires a local engine, so it returns a clear "not available in
  client-only mode" error. This is a stopgap; the collapse deletes the whole
  parallel path.

## Alternatives considered

### Remove `client-only` and the local mode now

**Why considered.** It is the stated end state; doing it now stops the drift.

**Why rejected (for now).** It is a ~73-`cfg`-site change that also deletes the
embedded-engine local mode and merges two dispatchers — the same surgery P3/P8
already schedule when local + REST + WS collapse to one ACP path. Doing it ahead
of that duplicates churn and risks conflicting with the in-flight transport work.
Sequencing it *with* the collapse is lower-risk and avoids touching the same
files twice.

### Leave the design undocumented and just patch the build

**Why rejected.** The inverted default + parallel dispatch is exactly the kind of
implicit assumption that produced the non-exhaustive-match bug. Recording the
model (this ADR) is what prevents the next omission.

### Keep `akh` engine-embedding as a supported standalone mode

**Why rejected.** It contradicts the pivot: a standalone `akh` with its own
engine is a second, un-fed akh, not a client of the operator's akh. Memory,
model pin, and identity live in `akhomed`; `akh` is a window onto it.

## Consequences

### Positive

- The architecture is now written down: `akh` = client (compilable
  independently), `akhomed` = server; the lean client is the *default* intent.
- The collapse has an explicit checklist (delete `Local`, merge dispatch, drop
  the feature) so it is not missed again.
- The tree compiles in all configurations in the meantime.

### Negative / risks

- The `client-only` parallel dispatch persists until the collapse, so new
  commands must still be added in both dispatchers until then (mitigated: this
  ADR + the build now failing loudly on omission via the exhaustive match).
- The default `akh` build still embeds the engine until the collapse, so the
  "lean by default" intent is documented but not yet realised.

## Implementation

- **Now (stopgap, this change):** add the missing `run_client_only` arms so
  `--features client-only` and `--features client-only,acp` compile —
  `Skill {Export,Import}` as local file ops, `Cluster` as a clear bail.
- **At the collapse (P3 finish / P8):** delete `AkhClient::Local` + the
  `not(client-only)` engine paths, merge `run_client_only` into a single client
  dispatch, drop the `client-only` feature, and route `akh` entirely over ACP.

## References

- [ADR-023](023-client-only-mode.md) — original client-only build mode (this ADR
  supersedes its build-model premise: the lean client becomes the default).
- [ADR-053](053-first-class-akh-config.md) — first-class `AkhConfig`.
- Plan: `docs/ai/plans/2026-06-14-pivot-acp-agentic-harness.md` (P3 client, P8
  retire REST/WS).
