# ADR 061 — Retire REST/WS/web-UI/inbound-MCP; go ACP-only (the P8 rip-out)

> Date: 2026-06-16
> Status: Implemented (P8.1–P8.8 landed 2026-06-17 on branch `claude/naughty-johnson-1add1a`; see commits `9e1c00e`…`8547069`)
> Phase: P8 (Pure-ACP Agentic Harness pivot — retire the legacy inbound surfaces)
> Depends on: ADR-059 (akh client / akhomed server), ADR-060 (tooling), the parked-data-plane decision (memory `project-acp-client-vs-agent-tools`)
> Plan: [`docs/ai/plans/2026-06-14-pivot-acp-agentic-harness.md`](../plans/2026-06-14-pivot-acp-agentic-harness.md)

## Context

akhomed serves four legacy inbound surfaces alongside ACP: a ~104-route REST
API, a WebSocket chat (`/ws`), a bundled HTMX/Alpine web UI, and an inbound MCP
*server* (`/mcp`). A parallel-reader survey of the codebase established:

- **104 REST routes** (~110 method+path ops), all consumed by `src/client.rs`
  (~94 methods) backing **~90 `akh` CLI commands**. The ACP `_akh/*` data-plane
  has only **11** methods — so **~84 CLI commands have no ACP equivalent** (the
  deliberately *parked* migration: cognitive ops are the agent LLM's tools, not
  client control-plane methods).
- **Anima is NOT an inbound consumer.** The only akh↔Anima coupling is
  *outbound* (`akh → Anima/Seshat /mcp`, `src/upstream_client.rs`), and Anima is
  expected to arrive on ACP. So deleting REST/MCP-server does not break Anima,
  and the chicken-and-egg invariant (akh must not depend on Anima) holds.
- **`src/mcp/mod.rs` is not pure MCP**: it also defines `PsycheSnapshotMap`,
  `DaemonPauseHandle`, `WorkspaceStats`, which `ServerState` references
  unconditionally — so the module can't be deleted until those move out, and
  `import_anima_project`/`link_seshat_doc` (outbound logic) must be lifted out.
- **Pre-existing K8s bugs**: liveness/readiness probes target `/status` (not a
  route — only `/health` exists); `AKH_HOME=/data` is read by nothing (the
  engine uses XDG vars, so the 30Gi PVC is disconnected from the data root);
  ACP-TCP defaults off (`127.0.0.1`, no port) and the chart exposes no ACP port.
- `ureq` is shared (REST client **and** outbound MCP / http_fetch / model
  downloads) → stays; `reqwest` (under `oxifed`) is unused.

## Decision

**Go fully ACP-only.** Operator decision (over the alternative of keeping REST
as the CLI backbone): build a minimal ACP control-plane, **drop the ~75
cognitive CLI commands** (they are agent tools, per the parking decision), and
delete the entire REST/WS/web-UI/inbound-MCP surface. Also fix the K8s bugs.

Executed as eight green-gated, additive-first increments:

1. **ACP control-plane** — a `ServerControl` trait (lib, dyn-compatible via a
   boxed-future `dispatch`) implemented over `ServerState` (akhomed), exposing
   `_akh/workspace/{list,create,delete}`, `_akh/daemon/{start,stop,status}`,
   `_akh/config/{get,set}` — the ~9 genuine client-control ops with no ACP home.
2. **ACP discovery** — `discover_server` verifies liveness via the ACP
   socket/TCP, not `GET /health`.
3. **`AkhClient::Acp` + CLI rewire** — control-plane commands over ACP; the ~75
   cognitive CLI commands are removed (not migrated).
4. **Delete REST** — router + handlers + WS chat + web UI + `src/client.rs`
   `Remote` role.
5. **Delete inbound MCP server** — relocate the 3 shared types + lift the
   anima/seshat outbound tools first; drop `rmcp`.
6. **Cargo deps/features** — drop `axum`, `tower-http`, `tokio-tungstenite`,
   `rmcp`; remove unused `reqwest`; flip `akhomed` required-features to `acp`.
7. **K8s/Helm** — probes → `tcpSocket` on ACP-TCP; `XDG_DATA_HOME=/data` (+
   siblings) so the PVC holds engine data; bind + expose ACP-TCP via a Service.
8. **Symbolic test hardening** — adversarial belief-revision/reasoning tests +
   VSA round-trips (parallel workflow).

## Consequences

- The CLI's surface shrinks to: chat (ACP), the `_akh/*` data-plane
  (status/symbols/triples/query/memory/infer/permission), and the new
  control-plane (workspace/daemon/config). The ~75 cognitive subcommands are
  gone from the CLI — those capabilities remain reachable by *prompting the
  akh* (the agent's own tools), which is the intended model.
- One transport, one discovery record, one listener set. axum/tower-http/
  tokio-tungstenite/rmcp drop out; binary shrinks; attack surface shrinks.
- **Risk**: largest, most destructive phase — sequenced additive-first
  (control-plane + discovery land before any deletion) so the tree stays green
  and the system keeps working at every commit. The `client-only` build is
  retired here (ADR-059).
- **Deferred within P8**: nothing structural — but the symbolic hardening (8)
  is independent and may trail the rip-out.

## Status

Decision recorded; increments execute on `claude/naughty-johnson-1add1a`. The
survey that grounds this ADR was workflow `w84iqkgxs` (6 readers, 104 routes +
~94 client methods + features/deps + Helm + Anima coupling mapped).
