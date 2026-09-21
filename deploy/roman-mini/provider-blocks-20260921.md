# WI-569 escalation and provider availability investigation

Investigated on 2026-09-21 through Anima MCP, the GitHub API, the T3
orchestration snapshot, and Roman session-service read APIs. No tracker SQL,
forced gates, duplicate workers, or live ownership changes were used.

## Tracked mirror disposition

WI-569 is `01a0a661-a56b-7933-8697-66cf76611351` in OpenIndiana project
`019de483-cdd9-7f52-9f49-7db6177102b3`; the incident's `project=anima` identifies
the service to repair, not this item's owning project.

GitHub independently reports PR
<https://github.com/OpenIndiana/oi-userland/pull/27606> merged into `oi/hipster`
at `2026-09-16T12:10:57Z`, merge commit
`1c167a17f30a6a49982c9a8e17c387c0d8442b4f`. The Anima item already records
`source=github`, the canonical PR URL, `external_ref=PR #27606`, and
`external_state=merged`. It remains `open/working`, with no sessions, changes,
PRs, CI runs, exit-bar rows, or dependencies. Its pinned implementation phase
nevertheless requires a PR head for the normal close transition.

Current Anima source at `d9ad9fb3` has an explicit upstream-sync path:
`WorkItemLifecycleService::transition_with_external_state_sync` permits a
tracked upstream close without synthesizing a PR head. The gRPC update service
passes `external_state_sync` through, but `mcp/handlers.rs` hard-codes it to
false and the exposed MCP tool has no such parameter. Therefore the requested
close is unavailable under this incident's MCP-only access boundary. No fifth
identical close attempt was made. WI-565's prior closed state does not grant a
new close capability or justify bypassing gates.

Janus: treat WI-569 as closed upstream using its existing `external_state=merged`
provenance, while reporting Anima closure as blocked. Stop unchanged close
attempts until a supported MCP capability is available. Next owner: Anima API
maintainer, to expose a narrowly authorized, audited tracked-mirror disposition
path with provenance, ownership, dependency and implementation-gate safeguards.

## Provider evidence

The WI-569 escalation thread `janus-astra-920d359b486189d7b8d0bb63` is running
this investigation. Its current snapshot does not prove prior quota exhaustion.
An older escalation, `janus-astra-66bc694ffb4c6c46a5b1efda`, has an errored turn
and an explicit revoked-refresh-token error. Authentication repair is different
from waiting for a usage reset.

The latest failed sessions inspected for Janus, Minerva, and Vesta respectively
were `01a0c131-642e-7c83-a610-d0a37e37e6dd`,
`01a0c11d-3812-74c1-859a-347e0bdf7661`, and
`01a0c11f-0e19-7a32-b291-641f06e0bdcb`. Each recorded HTTP 429 after three
provider attempts. Their errors do not distinguish exhausted account quota
from rate limiting. All three managers have newer completed turns; Janus also
had a running turn during inspection. None was restarted or replaced.

## Prepared change and remaining gates

This branch adds explicit escalation outcome reporting and durable per-manager
provider cooldowns. It preserves queued events, current owners, and the existing
fresh-wake scheduler. Authentication repair is visible and requires an operator;
there is no silent provider/model substitution or automatic replay of agent
actions. Local Python regression tests cover restart, receipt identity, stale
failures, operator recovery, blocked backlog fairness, and status outages.

The branch is based on deployed Roman source branch
`fix/roman-wi-pagination-20260918` at `7fe1ee48`, not current Akh-Medu main.
Review/CI and integration of that prerequisite remain gates. The installed
observer additionally monitors `runtime-minerva` and `runtime-vesta`; preserve
that existing customization when applying the update. No deployment or runtime
recovery claim follows from local tests. Next owner: Roman/Akh-Medu reviewer
and operator for exact-head review, CI, dependency integration, and activation.
