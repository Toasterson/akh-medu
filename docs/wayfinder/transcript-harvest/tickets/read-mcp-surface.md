---
type: grilling
status: closed
resolved: 2026-09-19
blocked-by: [event-normal-form.md, store-schema.md]
---

## Question

What tools does the harvester's MCP server expose, and how do they stay inside an
agent's context budget?

**Scope grew on 2026-09-19.** [The revised read surface](read-surface.md)
collapsed to one store, so this server is now the *only* read surface and must
serve semantic search as well as exact addressing. See
[the embedding pipeline](embedding-pipeline.md) for what backs it.

The first consumer is a coding agent mid-session, which makes context discipline
a correctness property, not a nicety: a single Codex rollout in the local corpus
is 11,762 lines, and the whole corpus is 68 MB.

1. **The tool set.** Session listing (filtered by source, repo, branch, time),
   session metadata, `get_lines(session, from, to)`, grep, **and semantic search**
   — which is what actually makes "what did I do about X last week" answerable
   without knowing a session id. Decide whether exact and semantic are separate
   tools or one tool with a mode; a model choosing badly between two similar tools
   is a real failure mode, and having them in one server is the chance to avoid it.
2. **Bounded results, always.** Every tool needs a hard cap and a continuation
   token. Decide the caps by measuring against the real corpus, and decide what a
   truncated result says about what it omitted.
3. **Raw or rendered.** Reads return [the compact render](render-format.md) by
   default, with the verbatim row available on request — confirm, and define the
   escape hatch.
4. **Redaction at the read boundary**, per [the posture](secrets-posture.md).
5. **Splicing is a read concern.** [The derivation worker](derivation-worker.md)
   normalizes `started`/`result` into `lifecycle` events carrying parent and child
   identifiers, and deliberately stops there — derivation is per-record and
   order-free. Assembling the spliced reading order
   ([normal form](event-normal-form.md)) is therefore this ticket's work, and it
   is a cross-stream join.
6. **Result shape for semantic hits.** A hit must carry its
   `(source, session_id, seq)` so the agent can immediately widen to the
   surrounding lines. Within one store that is a join, not a cross-service
   handoff — make widening a single call, not a documented two-step.
6. **Deployment and auth.** Where it runs, how the three CLIs register it, and
   who may read whose sessions. Note the local precedent that an akh's MCP server
   already registers itself via `.mcp.json`.

## Resolution (2026-09-19)

### 1. Bounding — byte budget, per-record truncation, and kind-aware tiers

Every tool takes a **byte budget** with a sane default and fills it. A record
whose own text exceeds a per-record cap is **truncated in place** and carries a
pointer to fetch it whole. Bytes rather than records, because records span
811 B (median) to **2,287,629 B** (max) — 27 exceed 200 KB on their own, so a
record count is fine or catastrophic depending on what it happens to contain.

**Truncation is kind-aware, and this is the load-bearing part.** Measured byte
shares of the 158 MB corpus:

| Content | share | note |
|---|---|---|
| Codex `event_msg` | **37.7%** | already noise — not normalized |
| tool call + output | **29.3%** | process output |
| `compacted` | 7.7% | 129 records, **avg 94 KB** |
| `attachment` / `world_state` / `turn_context` | 8.1% | injected context |
| **prose** — `reply`, `prompt`, `thought`, `reasoning` | **14.2%** | the decisions and plans |

So the interesting content is roughly **one seventh** of the bytes. Tiers:

- **Tier 1 — never truncated below a generous cap:** `prompt`, `reply`,
  `thought`. These carry the decisions made and the plans formed, which is what a
  reader is nearly always after.
- **Tier 2 — truncated hard, head-and-tail with the middle elided:** `tool_call`
  arguments and `tool_result` output. Process output is the least interesting
  content per byte and the most voluminous; a head-and-tail keeps the command and
  its outcome while dropping the body.
- **Tier 3 — excluded from default reads entirely:** `telemetry`, compaction
  snapshots, and anything already classified noise. Reachable explicitly.

Truncation is always **visible** — a truncated record says so and says how much
was withheld. Silent elision would make the surface untrustworthy for exactly the
audits it exists to serve.

### 2. Tool set — five, sharply separated

`list_sessions` · `get_session` · `get_lines` · `search_transcripts` (semantic) ·
`grep_transcripts` (literal).

Distinct names for distinct jobs. Merging the two retrieval modes into one tool
with a flag reliably yields models that always take the default mode, and the
description then has to teach two retrieval models at once. Five also stays well
inside the roughly-a-dozen ceiling this project already measured on gemma4 before
tool sets degenerate.

Defaults, inherited rather than re-decided:

- **Rendered by default**, verbatim on request — [compact render](compact-render.md).
- **Spliced reading order by default** for `get_session`, stream order on request.
  The cross-stream join lives here, since
  [the derivation worker](derivation-worker.md) deliberately stops at per-record
  `lifecycle` events carrying parent and child ids.
- **Redaction applied at this boundary** — [secrets posture](secrets-posture.md).
  This is now the *only* outward boundary, since Seshat was cut.
- **Semantic hits carry `(source, session_id, stream_id, seq)`** so widening to
  surrounding lines is one further call, not a documented two-step.

### 3. Deployment and access

One server, in-cluster beside the database, connecting with a **read-only PG
role**. `Authorization: Bearer <shared secret>`, fail-closed, env or file —
copying `crates/seshat/src/auth.rs`, which already does constant-time comparison
and returns a proper challenge. The three CLIs register it over the LAN via
`.mcp.json`, for which this repo already has precedent.

**No per-caller scoping.** For a single operator every token would be granted
everything, so an authorization model would add maintenance without adding
defence. Stated plainly instead: **the bearer token is the protection**, and the
corpus is full of other services' credentials. That is an argument for guarding
the token carefully, not for pretending scoping would help.

## Correction to the session-4 census

[The normal form](event-normal-form.md)'s record census ranked types by **count**,
so Codex `compacted` never appeared — 129 records, but **12.1 MB at an average of
94 KB each**, the third-largest byte consumer in the corpus. A count-ordered
census systematically hides exactly the types that matter most to a byte budget.
Recorded as an addendum there.

## Correction 2026-09-19 — `thought` leaves tier 1

[The render prototype](render-format.md) found reasoning content is 0.06% of the
corpus: Codex reasoning is 100% encrypted, and 91.8% of Claude thinking blocks are
empty signatures. Reserving tier-1 budget for `thought` would spend it on opaque
markers.

`thought` moves to a **marker tier**: rendered as its bare `<seq> thought` (plus
`enc 1` where the payload is sealed), never expanded. Tier 1 is `prompt` and
`reply`. The 180 blocks that *do* carry text are still served in full when a
caller asks for that record explicitly.
