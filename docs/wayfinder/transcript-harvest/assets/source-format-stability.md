# Source format stability across CLI versions (Claude Code, Codex, OpenCode)

Wayfinder RESEARCH ticket: "Source format stability across CLI versions."
Date: 2026-09-19.

**Legend**: `[MEASURED]` = produced by a script in `assets/scripts/` run against local
evidence, command/output given. `[WEB]` = claim sourced from a public URL, cited.
`[INFERENCE]` = my own reasoning connecting the above, not directly stated by either.

Local evidence used (as of 2026-09-19):
- Claude Code: `~/.claude/projects/**/*.jsonl` — 46 files, 23M, 18,290 lines total
  (ticket said 44 files/lines-not-given; the corpus grew between the ticket being
  filed and this survey running — see "corpus size note" below).
- Codex: `~/.codex/sessions/**/*.jsonl` — 33 files, 46M, 46,301 lines total.
- OpenCode: `~/.local/share/opencode/opencode.db` — SQLite, 262KB, single session.

Corpus size note `[MEASURED]`: `find ~/.claude/projects -name '*.jsonl' | wc -l` returned
**46**, not the 44 quoted in the ticket. Both counts are consistent with an
append-only, still-growing local corpus (this very research session is itself
adding lines to it) — not a discrepancy in the tool, just a moving target. All
numbers below are from the run captured in this document.

---

## 1. Version markers: availability and dispatch reliability

### Claude Code `[MEASURED]`

Script: `assets/scripts/survey_claude_code.py`.

```
# files found: 46
total lines: 18290, bad json lines: 0, lines missing 'version': 3061, lines missing 'type': 0

== version value counts ==
  '2.1.263': 4338
  '2.1.197': 4061
  '2.1.205': 3301
  '2.1.202': 2561
  '2.1.219': 602
  '2.1.209': 366

== distinct versions per file ==
files with a single version: 44
files with >1 version: 1   (one long-lived resumed session spans 5 versions)

== type x version matrix ==
  version '2.1.197'..'2.1.263': types=['assistant', 'attachment', 'system', 'user']  (identical set at every version)
```

**Findings:**
- 6 distinct `version` values across the corpus (`2.1.197` → `2.1.263`), i.e. real
  upgrades happened while these transcripts were being written.
- `version` is **not universal**: only 4 of 11 observed top-level `type` values ever
  carry a `version` field (`assistant`, `attachment`, `system`, `user` — the
  message-DAG record types). The other 7 types (`queue-operation`, `last-prompt`,
  `mode`, `ai-title`, `atis-latch`, `started`, `result`) never carry `version` at
  all — 3,061 of 18,290 lines (16.7%) have no version marker whatsoever.
- Within the four versioned types, the field set was **byte-for-byte identical**
  across all 6 versions observed (same key union per type at every version) — no
  detectable schema change in this corpus for this version range, only value churn.
- Reliability for dispatch: **good but partial**. `version` is a reliable minor/patch
  identifier when present, and one file recorded 5 versions in sequence (a session
  resumed across CLI upgrades), which is itself useful evidence: a harvester must
  expect `version` to change mid-file, not just per-file. But a `type`-first dispatch
  is mandatory regardless, because most of the harness/UI-state record types simply
  don't carry the field. Never key primary dispatch on `version` alone.

### Codex `[MEASURED]`

Scripts: `assets/scripts/survey_codex.py`, `assets/scripts/codex_version_diff.py`.

```
== cli_version values (from session_meta payload) ==
  '0.148.0': 39   (22 files)
  '0.153.4': 13   (11 files)

== payload types present in 0.148.0 but NOT in 0.153.4 (or vice versa) ==
only in 0.148.0: [('event_msg','agent_message'), ('event_msg','context_compacted'),
                  ('event_msg','mcp_tool_call_end'), ('event_msg','patch_apply_end'),
                  ('event_msg','sub_agent_activity'), ('event_msg','user_message'),
                  ('event_msg','web_search_end')]
only in 0.153.4: [('event_msg','item_completed'), ('token_usage_record', None)]
shared: 17
```
Confirmed by direct counts:
```
('event_msg','agent_message')     2620 @0.148.0 |    0 @0.153.4
('event_msg','item_completed')       0 @0.148.0 | 4488 @0.153.4
('token_usage_record', None)         0 @0.148.0 | 1968 @0.153.4
('event_msg','user_message')       261 @0.148.0 |    0 @0.153.4
('event_msg','patch_apply_end')   1391 @0.148.0 |    0 @0.153.4
```

**Findings:**
- `cli_version` lives **only** inside `payload.cli_version` of the `session_meta`
  record. No other record carries it. A parser must thread it forward from the
  session header to every subsequent line — it is not self-contained per-line.
- `session_meta` is not guaranteed to be line 1 only: 18 of 33 files re-emit a
  **second** `session_meta` later in the file (line 2, one file at lines 2 and 3).
  These carry `forked_from_id` / `parent_thread_id` and embed the **parent
  session's full header, including its complete system prompt**, inline. A
  harvester that assumes "one version marker, on line 1" will silently miss the
  fork boundary and misattribute the forked continuation to the wrong version if
  the parent used a different one.
- Between the two `cli_version`s actually present locally (0.148.0 and 0.153.4,
  which are not adjacent releases), the `event_msg` payload vocabulary changed
  **completely and disjointly**: `agent_message`, `user_message`,
  `patch_apply_end`, `context_compacted`, `sub_agent_activity`,
  `web_search_end`, `mcp_tool_call_end` (7 payload kinds, all present only in
  0.148.0) were replaced by a single `item_completed` payload kind, and an
  entirely new **top-level record type**, `token_usage_record`, was introduced
  wholesale in 0.153.4. This is not a cosmetic rename — the counts are exactly
  disjoint (0 ↔ N), consistent with a full event-model rewrite between these
  releases, not a gradual deprecation.
- Reliability for dispatch: the version field itself is reliable and stable in
  location (`session_meta.payload.cli_version`), but it dispatches nothing by
  itself — a harvester needs a `(cli_version, top-level type, payload.type)`
  triple to normalize correctly, and that mapping is empirically known to break
  entirely between two point releases seen in a single home directory.

### OpenCode `[MEASURED]`

Script: `assets/scripts/survey_opencode.py`.

```
== session table row (excerpt) ==
version = '1.18.18'   -- this is the OpenCode application version, one column,
                          stamped once per session row (not per event/message/part)
== event.type distribution ==
('message.part.updated.1', 7)
('session.updated.1', 5)
('message.updated.1', 5)
('session.created.1', 1)
```

**Findings:**
- There is no separate "app CLI version" column on `event`, `message`, or `part`
  rows — only `session.version` (one string per session, `1.18.18` in the single
  local session). This is a much weaker version signal than Codex's per-session
  header or Claude Code's per-line field: it tells you which OpenCode binary
  wrote the session, not which schema shape a given row uses.
- OpenCode compensates for this by baking a **schema version directly into the
  event `type` string**: `session.created.1`, `session.updated.1`,
  `message.updated.1`, `message.part.updated.1` all carry an explicit trailing
  `.1`. This is the most self-describing of the three version markers, in
  principle — a consumer doesn't need any cross-reference to a header record to
  know "this is shape 1 of `message.updated`"; if OpenCode ever ships
  `message.updated.2` alongside `.1`, existing `.1` handling keeps working
  untouched, and a `.2` handler can be added additively. This is corroborated by
  upstream source: the event log is Drizzle/SQLite-backed and the app is
  mid-transition from a "1.x" `session`/`message`/`part` shape to a "2.0"
  `session_v2`/`session_message` shape `[WEB]` (see §4) — i.e. the vendor itself
  needed a bigger version seam than the in-string `.N` suffix once, which is
  exactly the kind of drift a harvester must plan to survive.
- Local corpus is too thin (1 session, 18 events) to observe an actual `.2` in the
  wild; this finding is about the *mechanism*, verified structurally, not about
  observed version churn in this specific machine's data.

---

## 2. Full top-level record-type enumeration, counts, classification

### Claude Code `[MEASURED]` — 18,290 lines, 46 files, 11 types

| type | count | class | notes |
|---|---:|---|---|
| `assistant` | 8,599 | transcript-bearing | model turn: message content blocks (text/thinking/tool_use), `uuid`/`parentUuid` DAG edge |
| `user` | 4,776 | transcript-bearing | human/tool-result turn, same DAG fields |
| `attachment` | 1,694 | transcript-bearing | file/image/etc. attached to a turn; carries `rendered`/`renderedInHumanTurn` |
| `queue-operation` | 1,024 | harness noise | prompt-queue enqueue/dequeue bookkeeping, no `version`, no DAG link |
| `last-prompt` | 737 | harness noise (derived pointer) | caches the most recent prompt text + `leafUuid`; redundant with the DAG, used for UI resume-preview |
| `mode` | 600 | harness noise | permission-mode toggle log (`normal`/`plan`/etc.) |
| `ai-title` | 513 | harness noise (derived) | Haiku-generated session title, cosmetic |
| `atis-latch` | 179 | **undetermined from data alone** | only a 16-hex-char `atis` token + `sessionId`; no other field, no public doc found. Plausibly an internal feature-flag/telemetry latch ("ATIS" unexplained). Flagged, not guessed further. |
| `system` | 160 | transcript-bearing (meta) | hook output, compaction metadata, refusal/stop-reason bookkeeping — needed for faithful replay even though it's not a chat turn |
| `started` | 4 | harness noise | background-agent task-cache marker (`agentId`+content-hash `key`) |
| `result` | 4 | harness noise | background-agent task-cache result payload, keyed the same way as `started` |

Classification note: `started`/`result` are almost certainly the Task-tool
background-agent result cache (the same mechanism this very research task runs
under) — content-addressed by `key`, not part of the conversation DAG. Confirmed
by structure (`agentId`, `key`, `result.report`), not by any doc.

### Codex `[MEASURED]` — 46,301 lines, 33 files, 8 top-level types

| type | count | class | notes |
|---|---:|---|---|
| `event_msg` | 27,719 | mixed | envelope for many payload kinds (see below); UI/telemetry-heavy but includes real turn content |
| `response_item` | 15,409 | transcript-bearing | model/tool wire items: `reasoning` (4,642), `custom_tool_call`/`_output` (4,138/4,136), `message` (1,507), `function_call`/`_output` (348/348), `agent_message` (290) |
| `token_usage_record` | 1,968 | harness noise | 0.153.4-only, per-call token accounting |
| `turn_context` | 410 | harness noise | per-turn config snapshot (model, sandbox, cwd) |
| `world_state` | 324 | **undetermined from data alone** | opaque `payload.type=None` blob per file section; name suggests environment/tool-state snapshot but no field-level doc found publicly |
| `inter_agent_communication_metadata` | 290 | harness noise (multi-agent) | sub-agent/orchestration bookkeeping, `payload.type=None` |
| `compacted` | 129 | transcript-bearing (derived) | context-compaction summary that **replaces** prior history for replay purposes — losing this silently changes what "the transcript" means |
| `session_meta` | 52 | transcript-bearing (header) | session/fork header incl. `cli_version`; 33 primary + 19 mid-stream fork re-emissions |

`event_msg` payload breakdown (top ones, all `[MEASURED]`): `token_count`
(17,311, harness noise — per-token-delta UI ticker), `item_completed` (4,488,
transcript-bearing, 0.153.4-only), `agent_message` (2,620, transcript-bearing,
0.148.0-only), `patch_apply_end` (1,391, transcript-bearing — records an actual
file edit), `web_search_end` (310), `thread_settings_applied` (309, harness
noise), `task_started`/`task_complete` (293/263, harness noise),
`user_message` (261, transcript-bearing, 0.148.0-only), `context_compacted`
(115), `mcp_tool_call_end` (81), `turn_aborted` (13).

Classification note: unlike Claude Code, Codex's real transcript content is
**smeared across two top-level types** (`response_item` for the wire-format
model I/O, `event_msg` for a derived/human-readable narration of the same
turns) — `agent_message`/`user_message`/`item_completed` under `event_msg` look
like restatements of what's already in `response_item.message`. A harvester
should capture both verbatim (they're not proven redundant without deeper
tracing than this survey did) but should not assume normalizing one is
sufficient.

### OpenCode `[MEASURED]` — 18 events, 2 messages, 5 parts, 1 session (thin corpus)

| type | count | class | notes |
|---|---:|---|---|
| `message.part.updated.1` | 7 | transcript-bearing | incremental part (text/reasoning/step-start) append |
| `session.updated.1` | 5 | harness noise (derived) | session metadata/cost/token rollup, re-derivable from messages |
| `message.updated.1` | 5 | transcript-bearing | message envelope update (role, model, tokens, parentID) |
| `session.created.1` | 1 | transcript-bearing (header) | session genesis record |

No type in this thin sample was undetermined — the `.N` suffix and the JSON
shape inside `data` were both legible from the sample alone. This table is not
a complete enumeration of OpenCode's event vocabulary (only one real session
exists locally); treat it as a floor, not a ceiling — see §5 for how the
harvester should treat unseen `event.type` values it will inevitably meet in
the wild.

---

## 3. OpenCode `migration` table: what it implies, and event vs. message/part

### Migration cadence `[MEASURED]`

38 rows, `id` timestamp-prefixed `YYYYMMDDHHMMSS_slug`, earliest
`20260127222353_familiar_lady_ursula`, latest
`20260622202450_simplify_session_input`. That is **38 migrations in ~146 days**
(2026-01-27 → 2026-06-22) — one every **3.8 days** on average, with no gaps
longer than a couple of weeks. This is a schema under continuous, active churn,
not a stabilized format.

Classifying the 38 migration names by what they touch (by name, since I did not
get read access to the upstream migration SQL bodies in this pass — see
caveat below):
- **~24 touch `session`/`message`/`part`/`session_message`/`session_input`
  shape directly**: `add_project_commands`, `workspace`,
  `add_session_workspace_id`, `add_workspace_fields`, `move_org_to_state`,
  `session_message_cursor`, `workspace-name`, `add_icon_url_override`,
  `add_session_path`, `add_sync_owner`, `add_workspace_time`,
  `session_usage`, `data_migration_state`, `session-metadata`,
  `normalize_storage_paths`, `add_project_directories`,
  `session_message_projection_indexes`, `session_message_projection_order`,
  `session_input_inbox`, `event_sourced_session_input`,
  `add_session_context_snapshot`, `add_context_epoch_agent`,
  `project_dir_strategy`, `simplify_session_context_epoch`,
  **`reset_v2_session_state`**, `simplify_session_input`.
  The name `reset_v2_session_state` (2026-06-22, the *last* migration in the
  local table) is notable: "reset" implies a breaking restructure of session
  state, not an additive column.
- **Exactly 1 touches the `event` table's existence**: `20260323234822_events`
  (2026-03-23), which is presumably where the table was introduced. No later
  migration name in the local 38 mentions "event" again.
- The remainder are cross-cutting (`credential`, `add_icon_url_override`,
  `chief_energizer`/`blue_harpoon`/`wakeful_the_professor`/etc. — auto-generated
  slug names I can't classify from the name alone).

**Caveat `[INFERENCE labeled as such]`**: this classification is by migration
*name* only, since GitHub API path lookups for the migrations directory in
`anomalyco/opencode` (the confirmed upstream repo, verified live, see §4) did
not resolve within this research pass's budget. The row/column-count evidence
below is a stronger, direct-observation basis for the recommendation than the
name classification is.

### Is `event` more drift-resistant than `message`/`part`? `[MEASURED + INFERENCE]`

Yes, and the local DB's own row shapes prove why:

- `event` has a fixed, generic 5-column envelope: `id, aggregate_id, seq, type, data`
  (`[MEASURED]`, `pragma table_info(event)`). This envelope has not needed a
  schema change since its introduction migration — all subsequent schema drift
  is carried **inside** the opaque `data` JSON blob, keyed out by the
  self-versioned `type` string (`message.updated.1` etc., §1). A consumer that
  only ever does `SELECT id, aggregate_id, seq, type, data FROM event` cannot
  be broken by a `data`-shape change; it can only be caused to see a `type` it
  doesn't recognize yet, which is a normalizer-level, not a SQL-level, failure.
- `message`/`part`/`session` have wide, concrete column lists (`session` alone
  has 28 columns: `id, project_id, workspace_id, parent_id, slug, directory,
  path, title, version, share_url, summary_additions, ..., agent, model,
  time_created, time_updated, time_compacting, time_archived`) that were
  assembled incrementally by ~24 of the 38 migrations. A harvester issuing
  `SELECT *` (or naming columns) against these tables is exposed to `no such
  column` / column-order errors on every future migration that touches them —
  which, at the observed 1-per-3.8-days cadence against these specific tables,
  is a "when," not an "if." Public evidence independently confirms this
  exact failure mode has already happened to other harvesters (§4:
  `openchamber/openchamber#3713`, `anomalyco/opencode#13818`).
- Independent corroboration from the public repo (`[WEB]`, §4): OpenCode is
  already migrating *away* from `session`/`message`/`part` toward
  `session_v2`/`session_message`, i.e. the very tables this ticket's evidence
  block points at are the ones the vendor is deprecating, while the
  event-sourced log is what both the legacy and the new design are built on
  top of.

**Recommendation**: read `event` (id, aggregate_id, seq, type, data) as the
system of record, dispatch normalization on `type` (which already carries an
inline version suffix), and treat `message`/`part`/`session` as convenience
read-models to capture verbatim for reference but never as the thing the
normalizer's correctness depends on. If `event` selection ever needs to change
because OpenCode adds a 6th column, that's a strictly rarer and more visible
event (one CREATE/ALTER on a table we explicitly whitelist) than the
open-ended column churn already measured on `session`.

---

## 4. Vendor documentation / stability guarantees

### Claude Code — explicit, official, unambiguous `[WEB]`

`code.claude.com/docs/en/sessions` (fetched 2026-09-19), "Where transcripts are
stored" section:

> "Each line is a JSON object for a message, tool use, or metadata entry.
> **The entry format is internal to Claude Code and changes between versions,
> so scripts that parse these files directly can break on any release.** To
> build on session data, use `/export` or the script interfaces instead."

This is as clear as vendor guidance gets: the format is declared **internal
and explicitly unstable**, with an official recommendation to use `/export` or
the hook/status-line `transcript_path` + Agent SDK interfaces instead of
parsing JSONL directly. Our harvester is deliberately doing the thing this
paragraph warns against, for good reason (we need the raw, unredacted record,
not the export renderer's lossy prose) — but the vendor's own text is the
strongest evidence available that drift is a *when*, not an *if*, and that no
compatibility contract exists.
Source: <https://code.claude.com/docs/en/sessions>

Corroborating community evidence of concrete breakage already observed:
- `anthropics/claude-code#27361` (verified real, filed 2026-02-21, closed
  not-planned): JSONL never logs the final `message_stop` event, so
  `output_tokens`/`stop_reason` derived from the file undercount by ~1.9x —
  i.e. even a currently-working parser can be silently *wrong*, not just
  broken by an upgrade. <https://github.com/anthropics/claude-code/issues/27361>
- `anthropics/claude-code#20531` (verified real, 2026-01-24, closed): a
  background-agent's `TaskOutput` result embeds the full sidechain JSONL
  instead of just the final text — evidence that even Anthropic's own tooling
  around this format has shape surprises. <https://github.com/anthropics/claude-code/issues/20531>
- Community parsers exist precisely because there's no official schema:
  `simonw/claude-code-transcripts` (verified real, 1,691 stars) and
  `daaain/claude-code-log` (verified real, 1,225 stars) both reverse-engineer
  the format from sample files; `daaain/claude-code-log`'s README does not
  document any version-drift handling strategy, consistent with "read what you
  see, adapt when it breaks."
  <https://github.com/simonw/claude-code-transcripts>,
  <https://github.com/daaain/claude-code-log>

### Codex — no explicit stability statement found; de facto unstable `[WEB]`

I found no page in OpenAI's Codex docs or the `openai/codex` repo that
explicitly labels the rollout JSONL format "internal" or "unstable" the way
Claude Code's docs do. The only authoritative schema definition is the Rust
source itself — `RolloutItem`/`RolloutLine` in
`codex-rs/protocol/src/protocol.rs` — which is an implementation detail of the
CLI's own crate, not a published/versioned wire-format spec.
(<https://deepwiki.com/openai/codex/3.5.2-rollout-persistence-and-replay>,
third-party summary of the source; I did not get direct byte-range access to
the Rust source itself in this pass, so treat the line numbers as
secondhand.)

Public evidence of concrete, repeated breakage (`[WEB]`, all verified real via
`gh api`):
- `openai/codex#45150`, **open**, filed 2026-09-13, `codex-cli 0.154.0`:
  "codex cli corrupts rollout jsonl during upgrade."
  <https://github.com/openai/codex/issues/45150>
- `openai/codex#24948`, **open**, filed 2026-05-28, 31 comments, still open as
  of 2026-09-04: rollout files grow to 700MB–2GB because `compacted` records
  re-embed full `replacement_history` snapshots and raw tool output is stored
  unbounded; one machine had 91GB across 184 files >100MB.
  <https://github.com/openai/codex/issues/24948>
- `openai/codex#41806`: `~/.codex` grew 28.6GB in one month from the same
  `replacement_history` re-embedding pattern; archived sessions never cleaned
  up.
- `openai/codex#21660`: rollout JSONL files are created world-readable (mode
  0644) on Unix — a data-handling/security-relevant format detail, not just a
  schema one.
- Compression is itself a format-adjacent breaking change a harvester must
  handle: rollouts older than 7 days are transparently rewritten from
  `.jsonl` to `.jsonl.zst` (community report,
  `vshulcz/deja-vu#3640`, and corroborated by a general web summary of a mid-2026
  Codex change) — a harvester globbing for `*.jsonl` only will silently stop
  seeing files older than a week.
- A third-party reverse-engineering write-up exists
  (`dev.to/milkoor/reverse-engineering-codex-cli-rollout-traces-3b9b`)
  precisely because there is no first-party schema doc to read instead.

Verdict: **undocumented, and empirically it breaks — treat as unstable**,
same practical posture as Claude Code, just without the courtesy of an
explicit warning label.

### OpenCode — no explicit stability statement found; actively mid-migration `[WEB]`

I found no OpenCode/`anomalyco` doc page declaring `opencode.db`'s schema
public or stable. Structural/community evidence points the other way:
- The upstream repo (verified real: `anomalyco/opencode`, formerly
  `sst/opencode`) uses Drizzle ORM with generated migrations
  (`schema.ts`/`schema.gen.ts`/`migration.gen.ts`) — confirming the local DB's
  38-row `migration` table maps to a real, actively-generated migration
  pipeline, not a one-off local artifact.
- `kenn-io/agentsview#1642` (verified real, closed 2026-09-06): "OpenCode 2.0
  parser fails because of new event-sourced SQLite schema" — a harvester
  written against the "1.x" `session`/`message`/`part` shape (exactly what our
  local DB has) broke against a newer `session_v2`/`session_message` shape.
  <https://github.com/kenn-io/agentsview/issues/1642>
- `Dicklesworthstone/cross_agent_session_resumer#27` (verified real, closed
  2026-09-04): the inverse failure — a parser built for the "2.0" shape found
  "0 sessions scanned" against the "1.x" shape.
  <https://github.com/Dicklesworthstone/cross_agent_session_resumer/issues/27>
- `anomalyco/opencode#13654` (verified real, closed): "JSON→SQLite session
  migration silently skips for incremental upgrades" — the vendor's *own*
  migration path has a silent-data-loss failure mode, which is exactly the
  class of bug this ticket wants the harvester to be defensive against.
  <https://github.com/anomalyco/opencode/issues/13654>
- `anomalyco/opencode#13818` (verified real, closed): "No message parts after
  sqlite migration" — direct evidence of the `part` table specifically losing
  data across a schema migration.
  <https://github.com/anomalyco/opencode/issues/13818>
- `openchamber/openchamber#3713` (verified real via search, not independently
  re-fetched): "SQLiteError: no such column: project_id" after an update —
  the exact `SELECT`-level failure mode predicted in §3 for naive readers of
  `session`/`message`.

Verdict: **undocumented, and there is direct third-party evidence of two
different parsers each breaking against the schema version the other one
expects, simultaneously, in the wild, within the last two weeks of this
research date.** This is the least stable of the three formats by observed
breakage rate, and it is breakage at the SQL-schema level (missing
table/column), which is strictly harder to degrade gracefully from than a
missing JSON key.

---

## 5. Recommended detection + degradation strategy

Design goal restated: the system-of-record keeps the source byte-exact even
when upstream drifts; the normalizer degrades **visibly**, never silently.

### 5.1 Capture layer (per source) — never parses for meaning, only for framing

- **Claude Code**: read each `.jsonl` line as an opaque JSON value. The only
  structural fact the capture layer may rely on is "one JSON object per
  non-empty line" (measured: 0 malformed lines across 18,290 in this corpus,
  but do not assume this holds forever — wrap `json.loads`/`serde_json` per
  line in a try/catch and store genuinely malformed lines as raw bytes with an
  `unparseable_line` flag rather than dropping them, which is the failure mode
  Anthropic's own parser reportedly has, per the "drops malformed lines with no
  counter" community report in §4's search).
- **Codex**: same per-line JSON framing, plus: (a) track `cli_version` as a
  **running value carried from the most recent `session_meta` seen in the
  stream**, not a one-time read of line 1 — forks re-emit `session_meta`
  mid-file (§1); (b) glob for both `*.jsonl` and `*.jsonl.zst` and transparently
  decompress the latter, or the harvester will silently stop seeing anything
  older than 7 days (§4).
- **OpenCode**: capture layer reads the `event` table only —
  `SELECT id, aggregate_id, seq, type, data FROM event ORDER BY aggregate_id,
  seq` — never `message`/`part`/`session` for the system-of-record copy (§3).
  Store `type` and `data` verbatim; `data`'s internal JSON shape is exactly as
  opaque to the capture layer as a Claude Code/Codex line. Also capture
  `session`/`message`/`part` rows as a **secondary, best-effort** verbatim
  snapshot (they're useful for cross-checking and for sources that haven't
  been event-sourced yet), but never let their absence or a
  `no such column`/`no such table` error on them abort capture of `event`.

Every capture, regardless of source, is a single atomic write of the
raw bytes/row plus: source name, source-reported version (or "unknown" if
absent for that record), capture timestamp, and a stable record identity
(Claude Code `uuid`; Codex `(session file, ordinal)`; OpenCode
`(aggregate_id, seq)`). This is what makes verbatim capture keep working
independent of anything below.

### 5.2 Normalizer dispatch — version-and-type keyed, closed-world with an escape hatch

- Build a dispatch table keyed by `(source, type_family, schema_marker)` where
  `type_family` is the top-level `type` (Claude Code, Codex) or the
  pre-suffix event name (OpenCode: `message.updated` from `message.updated.1`),
  and `schema_marker` is the most specific version signal actually available
  for that record (Claude Code: per-line `version` when present, else "no
  marker"; Codex: the threaded `cli_version`; OpenCode: the trailing `.N` on
  the event type, which is the record's *own* schema version, not the app
  version).
- A record whose `(source, type_family, schema_marker)` combination is **not**
  in the dispatch table is an **unknown record**: store it (already done by
  capture), increment a per-source, per-`type_family` `unknown_count` counter,
  and do **not** attempt to normalize it into the projection. Do not guess a
  best-effort shape from partial field matches — that is exactly the "silent
  corruption" failure mode this ticket is trying to avoid (Claude Code's own
  undercounted-tokens bug in §4 is a cautionary example of a normalizer
  quietly producing a wrong-but-plausible number instead of failing loud).
- A record whose `type_family` is known but whose `schema_marker` is new
  (e.g. Claude Code ships `2.1.300` with the same `type`s but an added field;
  OpenCode ships `message.updated.2`) is a **degraded-known** record: attempt
  normalization using the nearest known-lower schema_marker's rules (fields
  the old rule doesn't know about are ignored, which is safe by construction
  since JSON/JSON-in-SQLite are additive-friendly formats in practice, per the
  fact that Claude Code's 6 observed versions kept identical key sets), but
  tag the projection row `degraded=true` and increment a separate
  `degraded_count` counter. This is what lets the normalizer "degrade
  visibly" instead of either crashing or silently mis-projecting.

### 5.3 Drift alarm

- Emit one gauge per `(source, type_family)`: `unknown_rate = unknown_count /
  total_count` over a rolling window (e.g. last 24h or last N captured
  records, whichever the harvester's existing telemetry favors).
- Alarm thresholds, informed by the measured baseline rates in this corpus so
  the alarm isn't tripped by expected noise:
  - Claude Code: the harness-noise types (`queue-operation`, `mode`,
    `ai-title`, `atis-latch`, `started`, `result`, `last-prompt`) are expected
    to have **zero** unknown rate once modeled (their shape was constant
    across all 6 versions observed) — any nonzero unknown rate on these is
    suspicious immediately. The transcript-bearing types
    (`assistant`/`user`/`attachment`/`system`) should alarm at a low absolute
    threshold, e.g. >1% unknown over a rolling 500-record window, since these
    are exactly the records whose loss the ticket cares most about.
  - Codex: alarm on **any** new top-level `type` appearing at all (there were
    only 8 in this whole corpus; a 9th appearing is a schema event worth a
    human look, not a statistical threshold) — and separately alarm on `%
    unknown` within `event_msg`/`response_item` payload types crossing 2%,
    since that's where the measured 0.148.0→0.153.4 rewrite actually landed.
  - OpenCode: alarm on **any** `event.type` whose numeric suffix exceeds the
    highest one the normalizer has rules for (i.e. seeing `message.updated.2`
    when only `.1` is known) — this is a clean, binary, self-describing signal
    unique to this source (§1) and should be treated as a same-day "go add a
    `.2` rule" ticket rather than a percentage threshold. Separately, alarm
    hard (page, not just log) on any `no such column`/`no such table` error
    against the best-effort `session`/`message`/`part` snapshot — per §4 that
    is the observed real-world failure mode for this source, and it must
    never be allowed to touch the `event`-table capture path.
- Alarms report counts, not just rates, and always include a sample of the raw
  captured (still-byte-exact) unknown records so a human fixing the
  normalizer has real examples immediately, without needing to go re-query
  production.

### 5.4 What NOT to do

- Do not use `version`/`cli_version` as the sole dispatch key anywhere — for
  Claude Code most record types don't carry it at all (§1), and for Codex it
  must be threaded, not read locally (§1).
  For OpenCode, do not use `session.version` (app version) as a stand-in for
  per-event schema version; use the in-string `.N` suffix instead (§1).
- Do not glob `*.jsonl` only for Codex; the vendor already rewrites aged files
  to `.jsonl.zst` (§4).
- Do not build the OpenCode harvester against `session`/`message`/`part`
  columns as the source of truth; they are a churning read-model the vendor
  itself is migrating away from (§3, §4), and naive column access is the one
  failure mode with public evidence of happening to two different named
  projects within the same two-week window as this research.
