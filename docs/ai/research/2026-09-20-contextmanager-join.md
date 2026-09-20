# ContextManager materialization: akhrtd defers per-turn assembly to the core

Date: 2026-09-20. Evidence:
[`contextmgr-20260920-hip-v1`](../../../experiments/native-recall/contextmgr-20260920-hip-v1).

## The join that was missing

Every piece of the ContextManager seam existed but nothing drove it: the core's
`SelectingManager` under `SafetyEnforced` assembled context (`src/core_service.rs`
`assemble_context`, behind `POST /v1/context`), the wire carried
`ContextRequest.messages` / `ContextResponse.assembled`, and the runtime's
`native_window` had a `verify` branch for an already-budgeted prompt. But **no
party held both the conversation and the policy**: history lived in akhrtd's
`Conversation`, and `assemble_context` ran only for a caller that supplied
messages, which no one did. The broker fetched instructions with an empty
message list; the runtime windowed locally with `select`.

The user chose "give akhrtd a core client" over "move the conversation into the
session record."

## What that choice cost, and how it was honoured

Reading the code surfaced a cost the decision did not name: core's `/v1/context`
is **per-akh and authorized** (`authorize(&p, &request.akh)`,
`engine(&request.akh)`), and `RuntimeJobRequest` carries no akh id, and akhrtd
holds no core credential. A literal socket from akhrtd to core would have given
the runtime daemon a new credential and the akh identity — a trust-boundary
change.

akhrtd already routes tools, recall and span capture through the session daemon,
which already holds the akh id, the `core_service` name and the core credential
(it half-makes this call at `sessions.rs`). So the join was built as: the runtime
initiates per-turn assembly **against its own held conversation**, over the
channel it already has, and the session daemon forwards to core with the akh and
auth it already owns. History stays in akhrtd; no new credential crosses a
boundary. That is the reading of "akhrtd has a core client" that does not widen
the trust model.

## The change

- `Host::assemble_context` (akh-runtime), defaulting to `Ok(None)` — the same
  optional-capability idiom as `recall`/`spans`. `None` means no core to defer
  to, and the runtime windows locally exactly as before.
- `RemoteHost` implements it by posting `RuntimeContextRequest` to the session
  daemon's new `POST /v1/runtime/context`, 404-soft like the other callbacks.
- The session daemon's `assemble_turn_context` forwards to the akh's core with
  the session's akh and tools, returning `None` when the session has no core.
- `run_loop` asks every native turn; a returned assembly (whose messages carry
  `CONTEXT_BUDGETED`) makes the native leg run `verify` instead of `select`.

No new config field: always-ask, fail-soft to local windowing. The blast radius
is one new route and one Host method.

## Measured, five-daemon canary on HIP

The canary passed with the eviction locus moved to the core:

| signal | value | meaning |
| --- | --- | --- |
| assembled (all 5 turns) | true | core assembled every turn |
| candidate messages | 1, 3, 5, 7, 9 | full durable history sent each turn |
| `final_core_dropped_messages` | **4** | the core evicted the aged-out fact turn |
| `runtime_dropped_messages` | **[0]** | the runtime verified, never dropped on top |
| `max_materialized_spans` | 1 | recall brought the fact back; answer `4317` |

All eight checks pass, including the two the canary gained: **history eviction
at the core** and **runtime defers eviction to the core**. The canary's old
assertion checked akhrtd's local drop count; that was coupled to the pre-join
topology and is replaced, not weakened — it now asserts eviction happened at the
core and that the runtime added no second policy on top. Recall correctness
(`4317`) is unchanged and still asserted.

## Faults on the way

- The first canary showed `budgeted_by_core=false`; I chased it through two
  instrument-rebuild-rerun cycles before realising the daemon log filter
  (`RUST_LOG=warn,...internal::native=debug`) suppressed my `info!` traces in
  the `internal` and `runtime` modules. The core-drop count is now logged
  through the `internal::native` target the canary already captures.
- Two runs OOM'd at model load (`hipMalloc rc=2`). The cause was a **separate
  user service** — `ollama-apertus` under `systemd --user` — holding 9.87 GiB.
  I misattributed it to my own leftover and killed it once; it respawned. The
  considerate fix was to wait for its model to idle-unload (27 min) rather than
  commandeer the shared GPU, then run once cleanly.

## Limits

Verified on the private five-daemon canary, one akh, a 1024-token window. It is
not a deployment or performance acceptance, and the homelab canary (item 9)
remains gated on reachability. The always-ask design adds one core round-trip
per native turn even when the assembly changes nothing; the broker makes a
similar per-turn call, so this matches existing cost.
