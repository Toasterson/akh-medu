# ADR 071 — The `akh` CLI command tree

> Date: 2026-06-20
> Status: Accepted (implemented incrementally with Phase 38e/f/g)
> Related: ADR 061 (ACP-only — `akh` is a pure ACP client), ADR 059 (client/server
>          split), ADR 049 / Phase 38 plan (relocatable Akh), ADR 070 (backings)

## Context

After the ACP-only rip-out (ADR 061) `akh` is a **pure ACP client**: every
command discovers the running `akhomed` and sends one `_akh/<group>/<op>`
JSON-RPC request over the Unix socket (or TCP). All engine work happens in the
daemon. Phase 38 (relocatable Akh) adds operator verbs — sleep, relocate,
workspace membership, projection refresh, migrate — that need a *coherent*,
discoverable command tree rather than ad-hoc additions.

## Decision

One dispatch pattern, one transport. Every CLI command maps to exactly one
`_akh/<group>/<op>` method via the shared `acp_call(method, params)` helper
(`src/main.rs`). Methods are either **control-plane** (routed through
`ServerControl::dispatch` against `ServerState`; listed in `CONTROL_METHODS`,
`src/acp/control.rs`) or **data-plane** (`dispatch_akh` / `dispatch_akh_agent`,
`src/acp/mod.rs`). Every method is advertised in `_akh/meta/capabilities`.

### Command tree

```
akh -w <workspace> <command>

  workspace                         control-plane: workspaces.* + memberships
    list                            _akh/workspace/list
    create <name> [--role R]        _akh/workspace/create
    delete <name>                   _akh/workspace/delete
    info [name]                     _akh/workspace/info
    join <id>                       _akh/workspace/join          (38f)
    leave <id>                      _akh/workspace/leave         (38f)
    list-members <id>               _akh/workspace/list-members  (38f)

  akh                               relocation / lifecycle (control-plane)
    snapshot [ws]                   _akh/akh/snapshot            (38c)
    wake [ws]                       _akh/akh/wake                (38c)
    sleep [ws]                      _akh/akh/sleep               (38e)
    relocate [ws]                   _akh/akh/relocate            (38g)

  daemon  { start|stop|status }     _akh/daemon/*
  config  { get|set }               _akh/config/*
  projection refresh                _akh/projection/refresh      (38d, stub)
  migrate                           _akh/migrate                 (38a/g ops verb)

  acp <op>                          data-plane (status/symbols/triples/query/
                                    recall/infer/add-triple/permission-mode/…)
  chat [--headless]                 session/prompt (interactive)
  service { … }                     launchd lifecycle (macOS, local)
  setup   { … }                     model / ONNX-runtime download (local)
```

### Conventions

- **Workspace selection.** A global `-w/--workspace` (default `"default"`); ops
  that name a target take an optional positional that defaults to `-w`.
- **Control vs data.** Lifecycle/admin → control-plane (`ServerControl`);
  knowledge reads/writes → data-plane. New verbs register in `CONTROL_METHODS`
  and are surfaced in `_akh/meta/capabilities` so `akh acp capabilities` lists
  them.
- **Output.** Results are the method's JSON, pretty-printed (`print_acp`). Errors
  are miette diagnostics (no server reachable → actionable help).
- **Postgres-optional.** 38e/f/g verbs degrade clearly when `[postgres]` is
  unset: `sleep`/`relocate`/membership return a diagnostic telling the operator
  the backing store is required, rather than failing opaquely.
- **Safety.** `wake`/`relocate` are guarded by the `last_seen` heartbeat **and** a
  Postgres advisory lock (ADR 049 / 38g split-brain guard); a refused wake exits
  non-zero with a clear message.

## Consequences

- New Phase-38 operator actions slot into the existing tree without a second
  control channel or transport. `akh --help` is the discovery surface; the
  capability list is the machine-readable one.
- `projection refresh` ships as a **stub** (AGE/38d deferred) so the command and
  the design are complete; it returns an explicit "AGE projection deferred"
  message rather than 404-ing.
- `migrate` exposes the `akhomed-store` migration runner (which otherwise only
  runs on daemon start when `migrate_on_start`) as an explicit ops verb.

## Verification

- `akh --help` shows the new verbs; each dispatches to its `_akh/*` method.
- New methods appear in `_akh/meta/capabilities`.
- CLI parse tests (clap) for the new subcommands; ACP dispatch handled by the
  control-plane tests. Postgres-dependent behaviour is exercised when
  `AKH_TEST_PG_URL` is set; otherwise the verbs return the documented "needs
  Postgres" diagnostic.
