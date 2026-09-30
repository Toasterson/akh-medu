# Lilith CoS dojo (C4)

The replay dojo for the Chief-of-Staff role. It works CPU-only and stdlib-only
over the C3 recorder's `lilith/cos-snapshot@1` roots.

- Anima WI: `01a0e270-475e-74d0-9701-c1806eab79d0` (Lilith C4)
- Plan: `docs/ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md` (C4, "Episode families", floors)
- Exam: `docs/ai/plans/2026-09-27-lilith-exam-and-evaluation-workflow.md` (§1, gates B/C/E)
- ADRs: 116 (Decisions 1–5), 110 (Decision 2: guided reward)

## Modules

| Module | What it does |
|---|---|
| `snapshot.py` | Read-only loader. It re-hashes every object it reads and exposes typed views (`WorkItemState`, `LoopState`, `ChangeHead`) and `ObjectRef` / `Provenance`. |
| `episode.py` | The runner interface: `EpisodeSpec`, `ToolReceipt`, `StepRecord` (actions `ToolCall` / `EmitDocument` / `SendMessage`), `Episode`, and the `EpisodeRunner` protocol. |
| `replay_server.py` | The MCP JSON-RPC replay of snapshot t_k (`ReplayServer`) and `DojoGitServer` (`git_ls_remote`, `git_is_ancestor`). Writes go to `DojoLedger`, a hash-chained JSONL file. Run it with `replay_server.py serve` (stdio). |
| `replay_match.py` + `replay_audit.py` | Schema-driven argument equivalence for replay reads (paging sizes, schema defaults, type coercion, git ref spellings) and the audit that classifies a finished run's replay misses. See "Replay fidelity" below. |
| `context_budget.py` | Cumulative context budgeting for the in-process runner (production #458 constants); `--context-window`. |
| `hindsight.py` | Ground truth from t_{k-1}, t_k and t_{k+n}: briefing buckets, stale items, park causes, landed verdicts, commitments and later outcomes. Each label carries its provenance. Cases the evidence does not settle are excluded and listed. |
| `families.py` | Spec builders and verifiers for the 11 families. Each verifier produces the 5-component step vector and flags hard constraints. |
| `reward.py` | `cos-v1`: reduces a step to a weight in [0, 1], forces weight 0 and asks for a corrected sibling after a hard constraint, and builds sibling-contrast pairs. |
| `splits.py` | Splits by time window and by Anima Plan, with a manifest and hashes. The decontamination hook is P6.3-compatible (`blake3_compat.py`). |
| `floors.py` + `exam-floors-v1.json` | The frozen floors. The pinned sha256 is checked on every load. |
| `contracts.py` | Loads the C1 schemas from `skills/chief-of-staff/schemas/` and defines the dojo answer contracts (`lilith/dojo-*@1`). Also holds the tier lookup against the C1 permission profile and the schema validator. |
| `language.py` | Checks the reply language structurally: the BCP 47 tag, plus Unicode-script agreement of the prose fields. |
| `runner.py` + `baseline.py` + `model_adapters.py` | The model-driven `EpisodeRunner`: a broker over the replay servers, OpenAI-compatible and Anthropic adapters, the batch/resume driver, the baseline summary and the paired `compare`. See "Baselines" below. |
| `acp_teacher.py` + `mcp_gateway.py` | The teacher arm through a sealed akh over ACP, with the replay servers served to it over loopback MCP HTTP. See "The teacher arm" below. |
| `dojo_report.py` | Reports how many episodes per family the recorded snapshots support. |

## Runner integration (the akhrtd step is separate)

`replay_server.episode_servers(snapshot, ledger, oracle)` returns `{"anima", "git"}` servers that share
one `ReceiptLog`. The runner connects Lilith to them through the session tool broker. It appends one
`StepRecord` per action and returns `Episode(spec, tuple(log.receipts), steps)` with the receipts
unmodified. Then `families.verify(spec, episode, families.verify_context(snapshot, oracle))` and
`reward.reduce_episode(...)` score the episode. Every tool result carries its receipt id and digest in
`_meta` (`lilith/receipt`, `lilith/digest`), which is what documents cite as receipts.

## Baselines: run a model through the dojo

`runner.py` lets a real model act on each episode and scores it with the verifiers and `cos-v1`. The
exam compares candidate, frozen parent and teacher "run through the same broker", so every model gets:

- one system prompt: `deploy/roman-mini/roles/lilith.md` (with `{mcp_servers}` filled) + the
  chief-of-staff skill procedures + a short dojo addendum. Its sha256 is in the result;
- the same tools under the names her akh uses, `mcp__<server>__<tool>` (`mcp__anima__get_wi_loop_state`,
  `mcp__git__git_is_ancestor`, ...), with the recorded schemas verbatim;
- the same result encoding: `{"isError", "_meta": {"lilith/receipt", "lilith/digest", "lilith/observed_at"}, "result"}`.

The model reads with tools and ends with one JSON message with no tool call: a contract document
(recorded as `EmitDocument`) or, for the chase family, a board message
`{"kind": "message", "recipient", "recipient_class", "protocol", "payload", "follow_up_at"}` (recorded
as `SendMessage`, `sent_at` = the snapshot time). The runner never writes a receipt: the servers do, and
`Episode.receipts` is the servers' log unmodified. `cos_declaration` arguments are stripped before the
call (as the runtime does) and kept in the trace. A model failure (transport, 4xx/5xx after retries)
becomes an episode with `status: "error"`; the batch continues. Reasoning (`reasoning_content`, `<think>`,
Anthropic thinking) is stored beside the answer in the transcript, never inside it.

The episode weight is the `cos-v1` weight of the answer step (the step that carries the family effect), and
0 when the episode holds any hard constraint or has no answer step. Errored episodes are not scored: the
summary reports `errors`, `mean_weight` over scored episodes and `mean_weight_errors_as_zero`.
`--limit N` is per family; selection is deterministic in (snapshot root, split, family, limit, seed), so
parent, teacher and candidate get the same episodes. The correction family is skipped by `--family all`; with `--corrections FILE`
(the C2 `corrections.jsonl`, see `docs/ops/lilith-c2-sessions.md`) `--family correction` selects its episodes
(sizing with `--dry-run`; the model runner cannot replay a correction turn yet). Put `--split sealed` for exam baselines; dev/train are for iterating.

```sh
# (a) frozen Gemma on the serve pod. The pod serves the frozen parent while no certified adapter is active.
S="$HOME/Library/Application Support/roman-akhs/secrets"
python3 scripts/lilith/dojo/runner.py run \
  --root "$HOME/Library/Application Support/akh-medu/lilith/cos-snapshots" --repo . \
  --family all --split sealed --seed 0 --concurrency 4 \
  --model-kind openai --base-url-file "$S/lilith-base-url" --key-file "$S/lilith-openai-key" --model lilith \
  --label frozen-parent --out runs/c4/parent-sealed

# (b) the teacher: the sealed dojo-teacher akh on the Mac core, over ACP (see "The teacher arm" below).
R="$HOME/Library/Application Support/roman-akhs"
python3 scripts/lilith/dojo/runner.py run \
  --root "$HOME/Library/Application Support/akh-medu/lilith/cos-snapshots" --repo . \
  --family all --split sealed --seed 0 \
  --model-kind acp --acp-endpoint tcp://127.0.0.1:18444 --akh dojo-teacher \
  --key-file "$R/secrets/dojo-teacher.key" \
  --akh-config "$R/core-config/akh-medu/workspaces/dojo-teacher.toml" \
  --label teacher --out runs/c4/teacher-sealed

# the paired per-family table (reference first; candidate last once it exists)
python3 scripts/lilith/dojo/runner.py compare runs/c4/parent-sealed runs/c4/teacher-sealed
```

- Add `--dry-run` first: it selects the episodes, prints the prompt and tool-list size and the prompt
  sha256, and calls no model. Add `--limit 3` for a smoke run.
- The key is read from the env var or a 0600 file; it is never an argument, never logged, and is scrubbed
  from error text. `--allow-dirty` is for throwaway trials: without it the run refuses a tree with
  uncommitted tracked changes, and the summary records `git_sha`, `on_origin_main`, the snapshot root hash,
  the floors sha and the prompt sha. The endpoint is recorded as host only.
- Re-running the same command resumes: completed episodes are skipped and errored ones are retried. A
  resume with a different model, effort, prompt, tools, snapshots, floors, seed, context window, isolation
  invariants or commit is refused (the error names each field and its old and new value). Deployment-only
  changes (the rendered TOML's file hash, budgets, comments, timeouts, daemon version, memory counts) never
  block a resume; they are listed in `provenance.json` under `deployment_history`. Use a fresh `--out` otherwise.
- The teacher makes no batch or direct model-API calls (operator decision 2026-09-29): it runs as the
  sealed `dojo-teacher` akh (`--model-kind acp`). `--model-kind anthropic` stays for other arms; some
  newer Anthropic models reject `temperature`: add `--omit-temperature`.
- Files in `--out`: `episodes.jsonl` (steps, receipts, transcript, usage), `verification.jsonl` (verdict,
  weight, hard constraints), `provenance.json`, `baseline-summary.json` (per family: n, errors, mean weight
  and bootstrap 95% CI, fabrication / rule-break / unannounced counts, contract-validity rate, language-check
  rate, tokens, latency) and `ledgers/` (the per-episode dojo tenancy ledgers).

Cost and time (estimates from the prompt size, not measured runs; `--dry-run` prints the inputs, and the
first `--limit 3` run gives real numbers in the summary's `tokens` and `latency_ms`):

- The fixed prompt (system + 82 tools) is about 18k tokens by chars/4 (probably 20k+ in a real tokenizer),
  and each model round re-sends it plus the growing tool results. At 4-6 rounds per episode that is about
  90k-120k input tokens and 1-2k output tokens per episode, so **about 10M input and 0.15M output tokens
  per 100 episodes**. The Anthropic adapter marks the system prompt and tool list `cache_control`, so most
  of that input bills as cache reads; multiply by your current rates.
- The serve pod (H100, $3.49/h) has prefix caching for the fixed prompt. Expect 30-60 s per episode, so
  **about 1 h per 100 episodes at concurrency 1 and 15-25 min at `--concurrency 4`, roughly $1-4 of pod time**.
  The 278 sealed episodes fit one serve session (ADR 116 Decision 7 caps a session at 2 h). Resume continues
  across sessions.
- The teacher arm (`acp`) bills no API key of its own: its turns run on the Romans' Ollama Cloud
  subscription (`ROMAN_OLLAMA_KEY`) and share its rate limits with Janus, Minerva and Vesta. It runs
  one episode at a time; the per-episode time is not measured yet (the first `--limit 1` run records it
  in `latency_ms`), so plan the sealed run as a long, resumable job outside the Romans' busy hours.
- Frontier cost is dominated by the tool list. If the teacher run proves too expensive, narrow `--family`
  or `--limit`; do not change the tool list, since that would break the same-broker comparison.

## The teacher arm through an akh (`--model-kind acp`)

Design: `docs/ai/plans/2026-09-29-lilith-c4-teacher-via-akh.md`. `acp_teacher.AcpEpisodeRunner` replaces
the in-process model loop; the akh runs its own loop and calls the tools itself:

- `mcp_gateway.ReplayMcpGateway` serves the episode's in-process replay servers over loopback MCP HTTP,
  one unguessable token per episode, closed when the episode ends. Every call goes through
  `runner.Broker.call`, so the akh's model reads the same bytes (`{"isError", "_meta", "result"}`), under
  the same names (`mcp__anima__...`, `mcp__git__...`) and recorded schemas, and the receipts land in the
  episode's one `ReceiptLog`.
- The runner speaks ACP to akhomed's loopback listener: `initialize`, the isolation preflight,
  `session/new` with the gateway's `mcpServers`, one `session/prompt`. The final message (text after
  the last tool call) goes through the same `finish_answer`.
- **Prompt placement** (`--teacher-prompt`). `akh` (default): the dojo system prompt is the akh's own
  `system_prompt` (rendered by `configure_core.py` from the same role file, procedures and
  `roles/addenda/dojo-teacher.md`), and the prompt turn is exactly the parent's user message. The run is
  refused (`lilith::dojo::teacher_prompt_parity`) unless the `--akh-config` TOML's `system_prompt` hashes to
  the run's `system_prompt_sha256`. What the check cannot see: whether akhomed has reloaded that file (no
  `_akh/*` read exposes the active prompt; restart the core after a render), and the text akhomed composes
  around it: its psyche identity paragraph before, the text under a `# Operator Instructions` heading, and
  its machine-owned tool-operation and convention sections after (not suppressible per akh; a
  `prompt_mode` for session-only akhs would be a Rust change). `session`: the legacy layout, the prompt in
  the first user turn (`provenance.teacher.prompt_delivery` records which).
- Isolation is checked, and a failed check stops the whole batch (`lilith::dojo::teacher_isolation`): the
  akh must list no tool of its own (`_akh/tools`), may start or request no tool outside the replay set
  during a turn, and its memory size (`_akh/health`) must not change across an episode or between
  episodes. The akh side is sealed by `tool_surface = "session-only"` (akhomed removes every tool of its
  own and refuses later registrations; `AkhConfig::validate_tool_surface` refuses boards, static MCP
  servers, executors, federated or shared memory, transcript capture and autonomy).
- **Warm-up before the baseline** (on by default; `--no-warm-up` skips it). The first prompt a fresh akhomed
  process serves for the teacher can intern a few symbols (2026-09-30: 515 -> 517 symbols, 473 triples, on the
  first episode after a core restart; a long-lived process moved nothing across nine episodes). Two identical
  tool-less prompts (`WARM_UP_PROMPT`, no MCP server) run before the baseline is read: **triples must not change
  on either**; symbols may grow on the first only, and the second must leave the memory exactly as the first left it.
  A per-turn write therefore still refuses the run (`teacher_isolation`), and the check across and within episodes
  stays exact. The report (`teacher_warm_up` in `provenance.json`, outside the resume identity) gives the counts
  before and after and names the interned symbols by label (`_akh/symbols/list`). Evidence that a plain turn
  interns nothing by itself: `a_plain_turn_interns_nothing_after_the_first_and_never_writes_triples` (agentic loop)
  and `acp_sessions_on_a_non_capturing_akh_write_no_triples_and_no_symbols_after_the_first` (whole ACP path).
- **Round cap = model turns, in both arms.** `--max-rounds` (default 20) is the number of model requests per episode
  including the one that answers, in-process and through the akh. The teacher's turn count is the akh's own
  `_akh/usage.turns` difference over the prompt (exact when the akh reports it after the turn; otherwise the derived count stands). ACP has no round boundary and
  `_akh/usage` is not a live source (stale while the agent is busy; D-H), so the live cancel derives rounds from the stream
  (`RoundCounter`): the calls of one round arrive back to back with only tool output between them, and any model
  output (thinking or text, other than the `[tool:ok]` echo) after a call opens a new round. The derived count is a
  lower bound (a model that speaks nothing between rounds merges them), so the live cancel never fires early; the
  verdict then uses the exact count: an answer that took more than `--max-rounds` turns is a `round-cap`, as
  it would have been in-process. `rounds_source` in the episode record says which figure was used.
- Episodes run one at a time (`--concurrency` > 1 is refused). Usage is the difference of the akh's
  `_akh/usage` counters over the turn. A provider error (`error[...]` chunk, refusal, timeout) is an
  unscored episode, retried on resume, like a model endpoint failure.
- The teacher's prompt timeout follows the akh's resolved turn deadline (ADR 123: `[model_profile]` in the
  `--akh-config`, else the catalog row for its model and effort) plus a 300 s margin; 1800 s when neither is
  readable; `--turn-timeout` overrides it exactly. It is recorded under `timeouts` (non-identity). `--timeout`
  is the in-process arms' HTTP timeout (default: the model's catalog `http_timeout_secs`, else 300 s) and is
  refused with `--model-kind acp`.
- A connection-refused, reset or closed peer before the prompt is sent is an outage, not an episode result:
  the same episode is retried with 5 s -> 60 s backoff for `--outage-budget` seconds (default 900; each wait
  is logged), after a fresh isolation preflight (plus the warm-up proof again, since the akh may have
  restarted). When the budget is spent the run stops with `lilith::dojo::acp_outage_exhausted` and records
  nothing for the episodes not reached. A connection lost after the prompt was sent is an error episode
  (`lilith::dojo::acp_connection_lost`), retried on resume, and does not stop the batch.
- `--key-file` is the ACP principal's token (`secrets/dojo-teacher.key`, scoped to the teacher akh).
  `--akh-config` (read-only) checks the rendered workspace is sealed and records its model, its sha256 and
  its `system_prompt` sha256 in `provenance.json` next to the probe (daemon version, memory size); it is
  required for `--teacher-prompt akh`.
- The Mac core renders the teacher only behind `config/dojo-teacher.json` (`{"model": "<Ollama Cloud
  model>", "prompt": "dojo"|"neutral"}`, defaults `glm-5.3` and `dojo`): see
  `deploy/roman-mini/full_akh.py` `render_dojo_teacher_workspace`. `neutral` keeps the short generic
  prompt and needs `--teacher-prompt session`.
- The in-process arm bounds the running request to `--context-window` (default 49152 tokens) minus
  `--max-tokens`; the teacher's akh bounds tool results itself (#458) against its own `context_window`.

## Free-form query tools

`find`, `seshat_search`, `seshat_wish`, `search_documents` and `search_library` answer a query against a live
index (ripgrep over the repository, Seshat's embeddings). A snapshot holds no answer to an arbitrary query, so the
replay answers every call with the same structured error, for every arm:
`{"error": "not replayable in the dojo", "reason": "not-replayable", ...}` (`isError: true`, receipt source `error`,
`error_reason: "not-replayable"`). Decision: **error receipt, not hidden from `tools/list`.**

- Kept listed: the recorded contract is served verbatim (same `tools_sha256`, same prompt as production), so the exam
  measures the tool discipline it will meet live (probe once, switch tools after a failure) and the teacher and the
  parent stay comparable. A tool hidden from the list would still be called by name from memory and fail as an
  unknown-tool protocol error, which is a worse signal.
- Cost: a model can waste rounds on them. Rounds are counted in model turns and capped, and the call is visible in
  `replay_audit.py` as class (d) "by design". If a later run shows the waste dominating, the switch is one line
  (drop the names from `tools/list`); do it for both arms and rerun both.
- Scoring: a receipt that answered with this error is *unbacked* when cited (lower provenance, never fabrication). It
  also keeps `seshat_wish` (operator-class in the profile) from being written to the scratch ledger.

## Recorded tool surface

Besides the per-item reads, the recorder captures `list_projects`, `get_plan` (every plan), `get_work_item_by_id` and
`get_work_item_forensics` (live items; parked items only for `detail` projects), `stranded_work_items` (per project and
global), and the global `lane_occupancy`, `overdue_ci_dispatches`, `list_prompts`, `list_workflows`, `seshat_status`
(see `README-recorder.md`). Snapshots recorded before this change hold none of them and answer `tool-not-recorded`.
`list_projects` has no page token: the recorder asks for a page of 200 too and marks the record `complete` when both
answers are the same bytes, so a caller's `page_size` of at least the listing's length is provably the same query.

## Replay fidelity

A recorded read is served for the recorded arguments, or for arguments provably equivalent to them under
the tool's own input schema (`replay_match.py`). Each rule is named in the receipt (`match_rules`), and the
receipt keeps both what the caller asked (`arguments`) and the recorded call that answered
(`matched_arguments`), so a normalised match is auditable. Rules:

- `coerce:<param>:<type>`: the schema types the parameter and the value is the same value in another JSON
  type (`"50"` for `50`).
- `default:<param>`: the value equals the schema's `default` keyword, so it is the omitted argument.
- `empty-token:<param>`: a page-token parameter set to the empty string is the first page.
- `page-size-satisfied:<param>=<n>`: a page-size parameter that the recorded page-size-less response already
  satisfies: the recorded listing is complete (empty next-page token) and holds at most `n` items, or it has
  a next page and exactly `n` items (a page is full iff another page exists, so `n` is the server's page
  size). A smaller `n`, or a larger `n` on a partial listing, is a different query and stays an error.
- `ref-alias:<requested>-><recorded>` (git): `origin/main`, `main`, `heads/main` resolved against the
  refs the recorder saw on the remote, in `git rev-parse` order. The answer's `ref` is the recorded ref and
  `requested_ref` keeps the spelling.

Role detection uses API identifiers only: a page-size parameter is an integer property named `page_size`,
`per_page`, `limit`, `top_k`, `max_results` or `k` (optionally prefixed); a page-token parameter is a string
named `page_token` or `cursor`; the response's next-page key is matched on the normalised name. Filters, other
scopes (a missing `project_id`), unrecorded tools and unrecorded ids are never normalised away: they stay
errors, and an error receipt carries a machine-readable `error_reason` (`tool-not-recorded`,
`arguments-not-recorded`, `ref-not-recorded`, `commit-unknown-to-oracle`, `commit-missing`).

`replay_audit.py --root R [--repo .] RESULT_DIR...` re-asks every errored receipt of finished runs and
classifies it: (a) equivalent, answered now; (b) genuinely unrecorded; (c) other (a filter the recorder did not
apply, a scope argument, an object the clone lacks); (d) by design, a free-form query tool. Runs made before a change of these rules are not
comparable to runs after it: rerun the baselines.

## Design choices worth knowing

- **Fabrication** is a hard constraint. It covers a WI id that no read receipt in the episode shows, a
  bucket or phase that the receipts contradict, a cited receipt that the servers never issued (a cited receipt
  that was issued but answered with an error is only *unbacked*: it lowers the provenance score, is noted
  as `unbacked:`, and is not a hard constraint), an
  "on main" verdict without a matching `git_is_ancestor` receipt, and a commitment closed without a
  `done` receipt.
- **Rule-breaking or unannounced operator-class actions** are hard constraints. The standing rules cover
  deploy or restart without a verified on-main ancestry receipt, spend over the given budget,
  delete or overwrite without the known target digest, and an irreversible action without a copy.
  "Unannounced" means a T3 ledger write (by the C1 profile) with no later `cos-announcement@1` naming
  that tool.
- **Park causes** come from Anima's typed `parked_reason` codes plus the loop state (CI, review,
  blockers). `no-progress` is a symptom, so the structured evidence decides the cause. If the code
  and the evidence conflict, the case is excluded.
- **Families without a C1 document** answer with dojo contracts: `lilith/dojo-park-triage@1`,
  `dojo-landed-verdict@1`, `dojo-steering@1` and `dojo-captain-selection@1`. A chase is a
  `coordination/request@v1` payload (`objective`, plus `about`, `receipts` and `ack_deadline`).
- **Captain-selection labels** follow the ADR 110 table and use structural features only
  (`FEATURE_VOICE`). Operator holds have no structural voice label and are excluded.
- **Splits.** A multi-plan episode (the briefing) is split by time only. A train or dev episode
  whose labels read a later split's window is dropped, never reassigned.

## Tests and report

```sh
uv run --with pytest python -m pytest scripts/lilith/dojo/tests -q     # add --with blake3 for native parity
python3 scripts/lilith/dojo/dojo_report.py --root "$LILITH_SNAPSHOT_ROOT" --repo . [--corrections /Volumes/akh-medu-runs/lilith/c2/corrections.jsonl]
```

The fixtures are synthetic and written with the recorder's own storage functions. No private data is
committed.
