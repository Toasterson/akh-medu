---
type: grilling
status: closed
resolved: 2026-09-19
blocked-by: [session-identity.md]
skills: [grilling, domain-modeling]
---

## Question

What is the normalized cross-tool event — the schema that makes three
incompatible sources answerable by one query?

This is the domain-modeling core of the effort. It needs a glossary, not just a
table definition.

Starting material, measured from the local corpus on 2026-09-19:

- **Claude Code** line types: `user`, `assistant`, `attachment` (77),
  `queue-operation` (44), `atis-latch` (29), `last-prompt` (27). Message content
  is a block list: `text`, `thinking`, `tool_use`, `tool_result`.
- **Codex** types: `event_msg`, `response_item`, `token_usage_record`,
  `turn_context`, `world_state`, `inter_agent_communication_metadata`,
  `session_meta`. `event_msg/token_count` alone is 17,311 records — the single
  largest category and almost certainly noise.
- **OpenCode**: `event.type` values plus `message`/`part` JSON payloads.

Decisions this ticket owes:

1. **The event kinds.** Something like prompt / reply / thought / tool-call /
   tool-result / meta — but named and defined precisely, and defensible for all
   three sources rather than a Claude Code schema with the others bent into it.
2. **What is noise.** Which record types are dropped from the normalized layer
   (they stay verbatim in the store regardless). `token_count` is the obvious
   candidate; `atis-latch` and `queue-operation` need a determination.
3. **Codex duplication.** `event_msg` and `response_item` both carry agent
   messages (2,620 vs 290 observed). Which is canonical, and is the other dropped
   or retained as a distinct kind? Getting this wrong double-counts every reply.
4. **Thinking content.** Claude Code emits `thinking` blocks and Codex emits
   `reasoning` items (4,642 of them). Are these first-class events, and does the
   secrets posture treat them differently from output?
5. **Sub-agent and multi-agent records.** Claude `isSidechain`, Codex
   `sub_agent_activity` and `inter_agent_communication_metadata`. Does the normal
   form represent agent hierarchy, or flatten it?
6. **Unknown types.** What the normalizer does with a record type it has never
   seen — this is where [format stability](source-format-stability.md) lands.

7. **Dispatch key.** [The format research](source-format-stability.md)
   recommends normalizing on `(source, type_family, schema_marker)`, with unknown
   combinations stored and counted but never guessed at, and degraded-known ones
   tagged `degraded=true`. Adopt or revise that, and decide where the counters
   live — it implies columns in [the store schema](store-schema.md).
8. **The two undetermined types.** Claude Code's `atis-latch` (179) and Codex's
   `world_state` (324) could not be classified from data alone. Either determine
   them from upstream source or classify them explicitly as
   retained-but-not-normalized; do not silently drop them.

9. **Reading order across streams.** Handed over by
   [session identity](session-identity.md): a session's streams are *not* totally
   ordered. Claude's main and subagent streams run concurrently; Codex's rollouts
   are sequential resumes. Define what "replay this session in order" means —
   stream-ordered, timestamp-interleaved, or parent-call-site spliced (a subagent
   stream logically belongs *inside* the tool_use that spawned it, which the
   normal form may want to represent). This is the single most consequential
   remaining modelling question.
10. **Project vs worktree.** `-home-toasty-ws-agi-akh-medu` and
   `-home-toasty-ws-agi-akh-medu--claude-worktrees-cool-carson-b63ef4` are
   separate slugs over the *same* repository, and subagent streams carry their
   own worktree `cwd`. Decide what "project" means for grouping and filtering —
   repo, worktree, or both — since [the read surface](read-mcp-surface.md)
   promises listing by repo and branch. Does not affect identity.

A full per-source record census with counts already exists in
`assets/source-format-stability.md` — start from it rather than re-surveying.

Record the resulting vocabulary in a `CONTEXT.md` glossary as it is settled.

## Resolution (2026-09-19)

Vocabulary is recorded in [CONTEXT.md](../CONTEXT.md) — scoped to this effort, not
a repo-wide glossary.

### 1. Ten event kinds (closed set)

`prompt` · `reply` · `thought` · `tool_call` · `tool_result` · `context` ·
`session_meta` · `lifecycle` · `error` · `telemetry`

**Chosen against the standing recommendation** of a six-kind set with an open
`subkind`. The accepted trade-off: queries like "every subagent spawn" need no
subkind filter and the schema is self-describing — but every new upstream concept
now becomes a schema decision rather than a new string, and the set will feel
per-tool pressure. Mitigation: `kind` is closed, and anything that does not fit
becomes an **undetermined** record rather than an eleventh kind added in haste.

### 2. Source mapping

| Kind | Claude Code | Codex (`response_item` unless noted) | OpenCode |
|---|---|---|---|
| `prompt` | `user` w/ text or string content | `message` role=user | `message` role=user |
| `reply` | `assistant` → `text` block | `message` role=assistant, `agent_message` | assistant message parts |
| `thought` | `assistant` → `thinking` block | `reasoning` | — |
| `tool_call` | `assistant` → `tool_use` block | `custom_tool_call`, `function_call` | tool parts |
| `tool_result` | `user` → `tool_result` block | `custom_tool_call_output`, `function_call_output` | tool parts |
| `context` | `attachment` | `message` role=developer, `turn_context` | — |
| `session_meta` | `ai-title`, `mode` | `session_meta` (event) | `session.created/updated` |
| `lifecycle` | `started`, `result` | `event_msg`: `task_started`, `task_complete`, `sub_agent_activity` | session lifecycle events |
| `error` | `system` subtype `model_refusal_fallback`; `fallback` content block | error events | — |
| `telemetry` | — | `token_usage_record` | — |

### 3. `response_item` is canonical for Codex; `event_msg` is not its duplicate

Measured: only **22% of distinct `event_msg` agent texts appear verbatim in
`response_item`**, and **11 of 33 files contain no `event_msg/agent_message` at
all**. They are not two views of one thing. `response_item` is the complete,
role-typed model-I/O record and is canonical for `prompt`/`reply`/`thought`/
`tool_call`/`tool_result`. `event_msg` is mined **only** for what `response_item`
lacks — the `lifecycle` row of the table above. This closes the ticket's
double-counting risk: nothing is read from both.

### 4. Noise — retained verbatim, yields no normalized event

- Claude `queue-operation` (1,112) and `last-prompt` (784) — both restate the
  user's prompt text that `prompt` already carries. Pure duplication.
- Claude `atis-latch` (225) — an opaque repeated 16-hex token with no content;
  `{type, atis, sessionId}` and nothing else. Determined: noise.
- Codex streaming `token_count` (17,311) — per-delta counters superseded by
  `token_usage_record`, which is what `telemetry` normalizes.

"Noise" is a statement about derivation only. Every one of these stays in the
verbatim store, and re-derivation can promote any of them later.

### 5. Undetermined — counted, never guessed

- Codex `world_state` (324) — opaque payload; **remains undetermined**. It is
  stored, counted, and excluded from normalization until someone reads upstream
  source. It is explicitly *not* classified as noise: we know what noise is, and
  this is not it.
- Claude `atis-latch` was the research's other undetermined type and is now
  **resolved to noise** by inspection (content-free, constant within a session).

### 6. Dispatch and degradation

Adopted from [the format research](source-format-stability.md): normalize on
`(source, type_family, schema_marker)`; unknown combinations are stored and
counted but never guessed at; partially-recognised shapes produce a normalized
event carrying **`degraded = true`**. Unknown-type counters are per
`(source, type, schema_marker)` and live in the store —
[store schema](store-schema.md) owns their columns, and the drift alarm
thresholds the research specified fire off them.

### 7. Reading order — splice at the spawn point

A session's canonical linearization is **depth-first**: a subagent stream is
inserted into its parent at the `tool_call` that spawned it. Claude Code supplies
the **spawn link** via `started`/`result` records carrying `agentId`; `result`
additionally carries the subagent's returned report, which is why these
eight-occurrence types are normalized as `lifecycle` rather than dropped. Codex
resume-rollouts concatenate in stream order.

Stream order and wall-clock order remain available; reading order is the default
the read surface serves.

### 8. Repo and workspace are separate fields

`repo` is canonical identity resolved from the git remote or repo root; `workspace`
is the directory a stream actually ran in; `branch` rides alongside. Sessions group
by `repo` by default. This keeps worktree sessions
(`-home-toasty-ws-agi-akh-medu--claude-worktrees-…`) and subagent streams running
under `.claude/worktrees/` rolled up to the repo they belong to without losing
where they ran. Does not touch identity.

### 9. `thought` is first-class but off by default

Normalized and queryable, excluded from the default read projection — 6,668
records across the two sources (2,026 Claude `thinking`, 4,642 Codex `reasoning`).
Substantial and relevant to Phase-51 reasoning work, but also the least-guarded
text in a transcript, so it is opt-in and gets
[the redaction detector](redaction-detector.md)'s attention explicitly.

### 10. `ai-title` is session metadata, not an event

Claude's `ai-title` (513 records, repeated identically within a session) collapses
to a session-level attribute. [The read surface](read-mcp-surface.md) needs a
human-legible session title for listing, and re-emitting it as 513 events would be
noise wearing a useful hat.

## Residual risk

**The spawn link is only proven for Claude Code.** Codex has
`sub_agent_activity` (264) and `inter_agent_communication_metadata` (290), but
whether either carries a resolvable parent-call-site link is **unverified**.
OpenCode's local corpus is too small (18 events, one session) to show subagent
behaviour at all. So reading order is fully specified for Claude Code and
*assumed* elsewhere. If the link turns out to be absent, those sources fall back
to stream-order concatenation — a degradation, not a failure, but it should be
confirmed before [the read surface](read-mcp-surface.md) promises spliced reads.

## Addendum 2026-09-19 — `compacted`, and a flaw in this census

[The read surface](read-mcp-surface.md) measured the corpus by **bytes** and found
a type this ticket's census missed: Codex **`compacted`** — only 129 records, but
**12.1 MB at an average of 94 KB each**, the third-largest byte consumer after
`event_msg` and tool output.

It was missed because the census ranked types by **count**. A count-ordered
census systematically hides low-frequency, high-volume types — precisely the ones
that dominate a read budget. Any future census should be run both ways.

Classification: `compacted` records are material placed into the model's context
(a compaction snapshot of earlier turns), so they normalize as **`context`** with
a compaction subtype, and are **excluded from default reads and from embedding** —
they restate content the corpus already holds in its original form. Their precise
semantics are **unverified** from the data alone; if that matters later they
should be treated as [undetermined](../CONTEXT.md) and read from upstream source
rather than assumed.

## Correction 2026-09-19 — `thought` has almost no content

[The render prototype](render-format.md) measured what decision 9 assumed:

- **Codex `reasoning`: 4,642 records, 100% encrypted** (`summary: []` +
  `encrypted_content`). Zero readable text.
- **Claude `thinking`: 2,203 blocks, 180 (8.2%) carry text**; the rest are an
  empty field plus an opaque `signature`.
- **Total readable reasoning in the corpus: 0.11 MB — 0.06%.**

Decision 9 justified first-class `thought` by "6,668 records … substantial". That
counted records rather than content. The kind **stays** — that thinking occurred
at a point in the stream is real ordered information, and the opaque payload is
retained verbatim like everything else — but it must be treated as a **marker, not
a body**: no tier-1 budget, no embedding, and no expectation that a reader can see
what the model was thinking.
