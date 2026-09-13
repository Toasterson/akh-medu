# Wayfinder uses Anima Plans and Work Items

The **Wayfinder map is [the native Plan](https://anima.wegmueller.it/plans/01a0878b-aa4a-72d1-a8cf-7915bd279615)**.
Its nine research or decision items are native analysis Work Items, each with
that `plan_id`. All 15 dependency edges were recreated with the new IDs and
verified by exact-ID readback. Every member remains in draft, with zero sessions.

Anima is the primary tracker. Local Markdown and JSON preserve navigation and
evidence. All tracker operations use Anima MCP.

## Verified migration

The supported update schema does not expose `plan_id` reassignment, so the nine
orphan records were replaced with actual Plan members. The incorrect map
placeholder and all nine old records remain **cancelled**, with replacement links.
[Retirement readback](retirement-readback.json) verifies no open obsolete items.
The accepted one-loop-per-Akh decision, descriptions, research and dependency
structure were preserved. [tracker.json](tracker.json) maps old and new IDs.
[Native readback](native-plan-readback.json) proves membership and dependencies.

## Research and discussion assets

Both provider research assets and the architecture discussion draft are ingested
into Seshat through Anima MCP, backed by private Anima Document records. The two
research Documents are linked to their Work Items. IDs and metadata readback are
in [document-import-results.json](document-import-results.json).

The findings are complete as research; exclusive Akh/SSH execution remains an
explicitly unproven provider conformance requirement. Anima rejects a direct
`draft` → `done` transition and requires promotion to `ready`. The Plan remains
in planning, so no promotion was performed and its research items remain draft.
Do not rerun completed research just to move the tracker phase. Human decision
items remain unresolved except for the accepted one-loop-per-Akh subdecision.

## Dedicated Codex integration

Codex's `mcp_servers.anima` runs `/Users/toasty/.local/bin/anima-codex-mcp`.
The launcher reads only its dedicated `anima-codex-planning` Keychain credential
through macOS Security and starts pinned `mcp-remote@0.8.6`. Credential reads
fail explicitly if unavailable; no token is written to the launcher or Codex
configuration, and there is no fallback to another integration's credential.

The credential was issued via Anima Login and IssuePat using the user-supplied
**Anima Admin Account** item in 1Password. The bootstrap login session was logged
out. Its eight scopes cover project reads, Work Item reads/writes, thread
reads/posts, document reads/writes and `plan.write`; it omits `plan.promote`.
Credential expiry: 2027-09-09. Existing integrations were not rotated or modified.

A fresh MCP handshake through the configured launcher discovered 63 tools,
including `create_plan` and `anima_import`, and omitted Plan promotion. Native
Plan creation, membership and document ingestion also succeeded with this PAT.
The running conversation's initial tool list predates this credential; direct
MCP calls here use the same dedicated Keychain entry. Future MCP connections
load it automatically.

## Evidence and next step

The earlier [initial readback](tracker-readback.json),
[pre-retirement snapshot](pre-retirement-readback.json) and
[pre-retirement migration readback](plan-migration-readback.json) describe the
retired structure only. The native and retirement readbacks describe current
state. Research branches remain local and have not been published to Forgejo.

Continue the Wayfinder decision discussion at **Session ownership and loop
selection**: the user chose one fixed loop per Akh; whether all sessions must
close before changing that loop remains a proposal, not an accepted decision.
