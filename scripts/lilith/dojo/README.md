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
| `dojo_report.py` | Reports how many episodes per family the recorded snapshots support. |

## Runner integration (the akhrtd step is separate)

`replay_server.episode_servers(snapshot, ledger, oracle)` returns `{"anima", "git"}` servers that share
one `ReceiptLog`. The runner connects Lilith to them through the session tool broker. It appends one
`StepRecord` per action and returns `Episode(spec, tuple(log.receipts), steps)` with the receipts
unmodified. Then `families.verify(spec, episode, families.verify_context(snapshot, oracle))` and
`reward.reduce_episode(...)` score the episode. Every tool result carries its receipt id and digest in
`_meta` (`lilith/receipt`, `lilith/digest`), which is what documents cite as receipts.

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
python3 scripts/lilith/dojo/dojo_report.py --root "$LILITH_SNAPSHOT_ROOT" --repo .
```

The fixtures are synthetic and written with the recorder's own storage functions. No private data is
committed.
