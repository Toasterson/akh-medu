---
label: wayfinder:map
created: 2026-09-09
status: charting
anima_plan_id: 01a0878b-aa4a-72d1-a8cf-7915bd279615
plan_creation: verified
---

# Map: interchangeable agent loops with Akh-owned services

The Wayfinder map **is an Anima Plan**. Each decision or research item is a
Work Item belonging to that Plan through `plan_id`; dependencies are native
Work Item edges. There is no separate map Work Item.

Canonical map: [Interchangeable agent loops with Akh-owned services](https://anima.wegmueller.it/plans/01a0878b-aa4a-72d1-a8cf-7915bd279615).
Its nine analysis Work Items have native Plan membership and 15 dependency
edges. [Readback](native-plan-readback.json) verifies draft state and zero
sessions. All ten obsolete records remain **cancelled**, with replacement links
and [retirement evidence](retirement-readback.json).

## Destination

Specify interchangeable Internal, Codex, Claude, Cursor and OpenCode loops
behind Akh-owned sessions, tools, memory, workspace and execution services.
Finish with reviewed contracts, provider feasibility evidence and a
dependency-ordered implementation handoff, including where alternative loop
strategies fit.

## Notes

- 2026-09-11 environment-onboarding clarification: a small per-user machine
  process adds a bootstrapped remote T3 environment through upstream pairing.
  The [connection-helper contract](assets/upstream-services-tool-bindings.md#machine-local-environment-connection-helper)
  covers readiness, device reachability, duplicate events and reconnection,
  keeping server lifecycle and session ownership with their existing daemons.
- 2026-09-11 latest clarification: external services use unmodified upstream
  versions. Our helpers manage launch, supported config/login and client
  connection flows. [Service profiles and tool bindings](assets/upstream-services-tool-bindings.md)
  replaces mandatory T3 source changes with supported-interface feasibility,
  and specifies semantic tool replacement plus enforced native-call restrictions.
  Model correction feedback helps in-session adaptation; it is not enforcement
  or persistent training. This remains local design work, with no tracker update.
- 2026-09-11 clarification: sessions remain central in `akhsessd`; selectable
  sandbox images co-locate `akhexecd` with the chosen T3, VS Code-compatible or
  Zed remote server. The [updated deployment plan](assets/k8s-acp-t3-deployment.md)
  removes custom remote T3 filesystem/Git/PTY adapters from the required scope.
  [Image profiles and Anima migration](assets/sandbox-images-anima.md) traces how
  Anima can retire its execution-routing layer while keeping work-item and
  review/result ownership. Runtime placement remains independent.
- 2026-09-11: the deployment target is Kubernetes at work/homelab, ACP over
  WebSocket, kagent, sandbox workspaces on Archibald and the Mac mini, and
  independently placed runtimes with host helpers. The earlier selection of
  upstream T3 with integration changes is now narrowed to unmodified upstream
  services and separate helpers/adapters. The local
  [researched deployment plan](assets/k8s-acp-t3-deployment.md) prioritizes
  ACP/WSS and records the T3 integration research, subsequently refined by the
  co-location clarification above. Released kagent BYO
  and development native harnesses have separate version/validation gates.
  AG-UI remains optional. This follow-on has not changed the native Anima Plan.
- 2026-09-10: the user expanded the client/process design to ACP plus AG-UI,
  with `akhsessd`, `akhrtd`, `akhsbxd`, `akhexecd` and a small `akheventd`.
  The local [daemon-boundary addendum](assets/daemon-boundaries.md) covers
  event/automation ownership, sandbox lifecycle and the Codex/Claude migration
  trace. These names/scope are user-selected; detailed policies are proposals.
  This addendum has not been synchronized into Anima's Plan or documents.
- The user has accepted provider runtimes owning their internal loops, with
  Akh owning the surrounding services and exposing its tools to frontier models.
  Supported subscription authentication is a primary requirement.
- This is a design map. A provider appearing in the proposed registry does not
  mean an adapter exists, all its native operations can be redirected, or every
  subscription authentication method is permitted.
- [Architecture draft](https://anima.wegmueller.it/documents/01a0878d-9529-7c82-bfbe-2ce4f1f8490b): component diagram,
  illustrative Rust trait, session lifecycle, tool broker, execution location,
  clients, memory/usage, alternative strategies and migration checks. It is
  discussion material; human decisions are not locked by writing a draft.
- Primary-source research is available for
  [Codex and Claude](https://anima.wegmueller.it/documents/01a0878d-9451-7182-b97c-bf3470f924fc) and
  [Cursor and OpenCode](https://anima.wegmueller.it/documents/01a0878d-94d0-7100-8258-812d3dba8eb3). Findings are attached to
  their research tickets; execution exclusivity still requires provider-specific
  tests. Both findings documents are ingested and linked. Anima blocks draft-to-done
  without promotion, so the planning items remain draft; the completed research
  must not be rerun merely to advance their lifecycle.
- [Tracker operations](tracker.md) records the Plan/Work Item mapping and
  the verified migration and dedicated integration setup. [tracker.json](tracker.json) is a local navigation
  index; it does not substitute for native `plan_id` membership.
- Apply Wayfinder, grilling and domain-modeling. Research uses independent
  primary-source research workers and isolated `research/*` branches. Skill
  sources for this session were found under
  `/Users/toasty/ws/Droneworld/.agents/skills/`.
- The user chose **one loop fixed per Akh** in
  [Session ownership and loop selection](tickets/session-ownership.md).
  All sessions inherit that implementation, with isolated session state.
  The remaining question is how an Akh changes its loop when sessions exist;
  requiring all sessions to close first is proposed, not accepted.
- Relevant existing boundaries: ADR 061 (ACP), ADR 070 (`ChatBackend`), ADR 080
  (lean client), ADR 082 (`Backend`). Read current code alongside these ADRs;
  their implementation-status notes can be older than the code.

## Decisions so far

No complete ticket resolved yet. The accepted loop-selection subdecision is
recorded on [Session ownership and loop selection](tickets/session-ownership.md);
that ticket remains open for the loop-change lifecycle rule.

## Not yet specified

- Provider research may expose a need to separate runtime placement, native
  tool substitution and provider extension isolation more finely.
- Composition may require new authority/budget rules beyond a simple loop
  adapter. Sharpen these after defining the minimal lifecycle contract.
- Persistence and recovery scope may expose a separate storage decision once
  the desired reconnect guarantees are known.

## Out of scope

- Implementing, installing, deploying or enabling the five adapters.
- Promoting analysis items to autonomous implementation dispatch.
- Subscription-token extraction, credential migration or silently changing
  accounts/billing modes.
- Implementing the daemon split, changing live execution backends or building
  a new graphical client. The user-requested ACP/AG-UI and process-boundary
  discussion is covered by the local addendum linked above.
- Promising provider strategy control, portable hidden model state or verified
  remote process termination without a corresponding provider/executor contract.
