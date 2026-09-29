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
| `hindsight.py` | Ground truth from t_{k-1}, t_k and t_{k+n}: briefing buckets, stale items, park causes, landed verdicts, commitments and later outcomes. Each label carries its provenance. Cases the evidence does not settle are excluded and listed. |
| `families.py` | Spec builders and verifiers for the 11 families. Each verifier produces the 5-component step vector and flags hard constraints. |
| `reward.py` | `cos-v1`: reduces a step to a weight in [0, 1], forces weight 0 and asks for a corrected sibling after a hard constraint, and builds sibling-contrast pairs. |
| `splits.py` | Splits by time window and by Anima Plan, with a manifest and hashes. The decontamination hook is P6.3-compatible (`blake3_compat.py`). |
| `floors.py` + `exam-floors-v1.json` | The frozen floors. The pinned sha256 is checked on every load. |
| `contracts.py` | Loads the C1 schemas from `skills/chief-of-staff/schemas/` and defines the dojo answer contracts (`lilith/dojo-*@1`). Also holds the tier lookup against the C1 permission profile and the schema validator. |
| `language.py` | Checks the reply language structurally: the BCP 47 tag, plus Unicode-script agreement of the prose fields. |
| `runner.py` + `baseline.py` + `model_adapters.py` | The model-driven `EpisodeRunner`: a broker over the replay servers, OpenAI-compatible and Anthropic adapters, the batch/resume driver, the baseline summary and the paired `compare`. See "Baselines" below. |
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

# (b) the Claude teacher. The key comes from an env var the operator provides.
ANTHROPIC_API_KEY=... python3 scripts/lilith/dojo/runner.py run \
  --root "$HOME/Library/Application Support/akh-medu/lilith/cos-snapshots" --repo . \
  --family all --split sealed --seed 0 --concurrency 4 \
  --model-kind anthropic --key-env ANTHROPIC_API_KEY --model <opus id> \
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
  resume with a different model, prompt, snapshots, seed or commit is refused. Use a fresh `--out`.
- The teacher default is Opus 5.5 (`claude-opus-5-5`), the exam plan's teacher. Some newer Anthropic
  models reject `temperature`: add `--omit-temperature`.
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
- Frontier cost is dominated by the tool list. If the teacher run proves too expensive, narrow `--family`
  or `--limit`; do not change the tool list, since that would break the same-broker comparison.

## Design choices worth knowing

- **Fabrication** is a hard constraint. It covers a WI id that no read receipt in the episode shows, a
  bucket or phase that the receipts contradict, a cited receipt that the servers never issued, an
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
