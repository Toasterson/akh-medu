# Runbook — Lilith C2 operator sessions (Stage 0)

Slice C2 of [ADR 116](../ai/decisions/116-lilith-chief-of-staff-guided-rl.md) (Anima WI
`01a0e270-314e`, [plan](../ai/plans/2026-09-27-lilith-chief-of-staff-rl-runpod.md)). The exit
condition is **at least five operator sessions of real work**, with the transcripts and corrections
admitted to the NFS share `akh-medu-runs`, baseline notes (missed items, fabrications, autonomy
noise), and the tok/s and spend of the window.

## Before you start (supervisor)

- The core running the `lilith` akh is an `akhomed` built from `origin/main` **after** the C2 change
  (it adds the read-only ACP `_akh/transcripts/list` and puts each tool call's arguments, output and
  duration into the captured turn). An older core answers the harvester with "older than ADR 116 C2"
  and its turns carry no receipts. Deploy per `.claude/rules/deploy-from-main.md` first.
- The serving endpoint must be up (Lilith's turns fail with the backend error until it answers
  `/health`; a cold start takes several minutes, see [lilith-serve.md](lilith-serve.md)).
- The NFS share `/Volumes/akh-medu-runs` is mounted on the Mac (the harvester refuses to write
  otherwise, exit code 3).
- Note the time you hand over to the operator (UTC); it is the `--since` of the harvest, so earlier
  test chats are not counted.

## How to start (operator)

1. Open T3 Code and start a new thread with the provider **"Lilith · Chief of Staff"**
   (`roman-lilith`). Every message you send there is captured (`capture_transcripts = true`).
2. One thread = one session is the natural unit, but it is not required: the harvester splits
   sessions by idle time (a gap of more than 45 minutes starts a new one).
3. Work as you would with a real chief of staff. Do not coach her toward the answer; the point is
   the baseline.

## Five suggested sessions (real work, not exercises)

Each needs at least two of your messages to count. Follow your own thread; the openers are only
prompts.

1. **Morning briefing.** Ask what changed overnight and what needs you today. Then pick one item
   she named and ask her to show you why.
2. **Park triage of the akh-medu lane.** Ask which items are parked, why, and what would unblock
   each. Pick one and ask what she would do (do not let her act on it unless you want to).
3. **illumos guidance on a real illumos work item.** Choose an open illumos item and ask her to
   guide the implementer or PM through it. Correct anything that is wrong about illumos.
4. **Chasing a stalled item.** Pick an item that has not moved, and ask her who to chase, with what
   message, and when to follow up. Ask her to draft the message; you send it.
5. **Commitments review.** Ask what has been promised (by you, by the akhs, by implementers) and
   what is overdue or at risk.

## Marking a correction

When her answer was wrong, start your next message with **✎** (U+270E LOWER RIGHT PENCIL) as the
very first character (spaces before it are fine). It works in any language and means "your previous
answer was wrong; here is the correction". Optionally put the tool she should have used in
backticks right after it:

```
✎ `mcp__anima__list_work_items` you left out the parked items
✎ the commit is not on main yet
```

Type it once and reuse it via macOS Text Replacement (System Settings, Keyboard, Text Replacements:
for example `,,` becomes `✎`), or paste it from Character Viewer (Control-Command-Space, search
"pencil"). The correction always refers to her immediately preceding answer. A `✎` message as the
first message of a session has nothing to correct and is ignored. Write the correction as you would
tell a colleague; the text is kept as the operator's words.

## What the supervisor runs afterwards

`S=~/ws/akh-medu/scripts/lilith` from a clean checkout of `origin/main`. Nothing here opens a store
file; the harvester talks to the running core over ACP with the operator key
(`~/Library/Application Support/roman-akhs/secrets/operator.key`, read by the script, never printed).

```sh
# 1. Harvest sessions and corrections to NFS (idempotent; re-run any time; never deletes).
python3 $S/c2/harvest_sessions.py --since 2026-09-30T08:00:00Z
#    -> /Volumes/akh-medu-runs/lilith/c2/sessions/<session-id>.json, corrections.jsonl,
#       corrections-unmatched.jsonl (no recorder snapshot before the turn), autonomous.jsonl

# 2. Serving spend for the window (report it), then the baseline notes with it.
python3 ~/ws/akh-medu/scripts/lilith/serve/lilith_serve.py ledger
python3 $S/c2/c2_notes.py --tok-per-s <measured> --spend-usd <ledger figure>
#    -> c2-notes.md + c2-notes.json; the first lines say whether the exit condition is met

# 3. Feed the dojo: the correction family now has episodes.
python3 $S/dojo/dojo_report.py --repo ~/ws/akh-medu \
    --corrections /Volumes/akh-medu-runs/lilith/c2/corrections.jsonl
python3 $S/dojo/baseline.py run --family correction --split all --dry-run \
    --corrections /Volumes/akh-medu-runs/lilith/c2/corrections.jsonl \
    --model-kind openai --key-env UNUSED
```

Exit codes of the harvester and the notes: 0 ok, 1 failed, 2 usage, 3 NFS share not mounted (nothing
written), 4 core unreachable, token refused, or core too old.

### Reading the notes

- **Fabricated ids**: a work-item id in her answer that no tool result of the session so far, and
  none of your messages, contains. Ids are matched by shape only (UUID or its first two groups); a
  status claim in prose is not checked. An id whose receipt was cut at the capture cap (4000 bytes)
  or not captured is *unverifiable*, not fabricated; read the session file to rule on it.
- **Missed items**: for a briefing-like turn (her answer names at least three work items of the
  nearest recorder snapshot; name others with `--briefing-turn <session-id>#<turn>`), the items the
  snapshot's briefing buckets require that she left out.
- **Autonomy noise**: turns she ran on her own (outside your sessions) and calls beyond reading and
  her own state (permission tiers t2, t3, outside-grant, unlisted). Whether a call was "requested"
  is your judgement, so each is listed with its turn.
- **tok/s** is not captured per turn by the akh; take it from the serving pod's vLLM metrics.

## What is stored, and where

Private data stays on NFS (`/Volumes/akh-medu-runs/lilith/c2/`); nothing is committed to the repo.
Session files hold the full text of your messages and her answers, her tool calls with arguments and
output (truncated at 1000 and 4000 bytes), per-call durations, the model name, and provenance of the
harvest. A correction record carries the session and turn (`episode_ref`), the recorder snapshot at
or before the corrected turn (`snapshot_id`), the hash of that turn's tool-call sequence
(`corrected_signature`), the expected tool, and the work-item ids in its receipts (`subject_ids`).
