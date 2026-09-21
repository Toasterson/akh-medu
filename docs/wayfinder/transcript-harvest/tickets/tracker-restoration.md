---
type: task
status: open
blocked-on: a human-supplied Anima PAT — cannot be done by an agent
blocked-by: []
---

## Question

Restore Anima as this map's tracker, or accept local markdown for the duration.

Anima is this repo's primary Wayfinder tracker — `docs/wayfinder/agent-loops/`
runs on a native Anima Plan with Work Items and native dependency edges. This map
does not, because from this machine:

- `/home/toasty/.local/bin/anima-pat get-auth-header` reports *"no PAT stored. Run
  `anima-pat init` first."*
- `https://anima.wegmueller.it/mcp` answers **401**, and the configured MCP server
  fails to connect.
- `http://seshat.lab.home/mcp` also answers **401** (relevant to
  [the Seshat projection](seshat-projection.md), not only to tracking).

The existing tracker doc describes a *macOS Keychain* launcher
(`anima-codex-mcp`); this is a Linux box, so the credential path is different and
may simply never have been set up here.

Work: run `anima-pat init` with a PAT of appropriate scope, confirm the MCP
handshake, then either migrate this map to a native Plan with its tickets as Work
Items and dependency edges, or record the decision to keep it in markdown. Note
the precedent that Anima's schema does not expose `plan_id` reassignment, so
placing items in the right Plan from the start matters.

Until this closes, the local markdown files in this directory are the tracker.

## Status 2026-09-19 — confirmed still blocked, and why

Re-verified this session; nothing has changed and **an agent cannot resolve this**:

- `/home/toasty/.local/bin/anima-pat get-auth-header` → *"no PAT stored. Run
  `anima-pat init` first."*
- `https://anima.wegmueller.it/mcp` → **401**; the configured MCP server fails to
  connect.
- `http://seshat.lab.home/mcp` → **401** (now moot for this map — Seshat was cut —
  but the same credential gap).

Issuing a PAT requires an authenticated human session against Anima. The map has
run to completion on the local-markdown tracker in the meantime, so this is a
**migration** task, not a blocker: 23 tickets with resolutions, a dependency
graph, and `CONTEXT.md` all exist as files.

If it is done, note the precedent from `docs/wayfinder/agent-loops/tracker.md`
that Anima's schema does not expose `plan_id` reassignment — items must be created
in the right Plan from the start. And note that the map's ticket bodies are long;
they are the deliverable, not scaffolding, so migration must carry them verbatim
rather than summarising.

**Recommendation:** leave it in markdown. The map is complete and its value now
lives in [the ADR and plan](#) generated from it, not in tracker ergonomics.
