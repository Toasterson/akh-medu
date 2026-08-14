---
name: babysit-lane
description: One full babysit pass over the akh-medu lane in the Anima tracker — report running/draft items, verify CI verdicts, merge greens, fix rebase-only conflicts, queue draft/ready items to the akhs in dependency order, and work the parked pile item by item (the parked items ARE the backlog — resolve each individually, don't bulk-triage). Use when the user says "babysit", "run a lane pass", "how is the lane", or "unstick the queue".
---

# Babysit the akh-medu lane

One complete pass: **observe → report → act**. Everything here is a single
iteration; the user decides cadence (looping via /loop or cron is theirs to arm).

## Constants

- akh-medu project: `019d9844-ce74-7993-a9f0-c953b8f04a4e`
- Anima project (tracker's own repo, in scope only where it unsticks akh-medu):
  `019d9ca9-ca71-7673-912c-09a8749821cb`
- Executors on this Mac: `tecton`, `imhotep` (gateway `~/.config/anima-runner/gateway.env`,
  one session per akh — ADR-027 agent mutex; see memory `mac-mini-imhotep-lane`)
- Anima repo checkout: `/Users/toasty/ws/anima` (Forgejo token embedded in the
  origin URL: `git remote get-url origin | sed -n 's|.*oauth2:\([^@]*\)@.*|\1|p'`)
- anima-server logs: `kubectl -n anima logs deploy/anima-server`
- Solstice orchestrator ground truth: `ssh archibald 'docker logs solstice-orchestrator ...'`
- **Never touch** the illumos project (track-only, unstaffed).
- **Operator-only** (report, never do): rabbitmq broker restart, forger image
  builds, anima-server deploys, anything destructive on live infra.

Load `mcp__anima__*` tools via ToolSearch first if they are deferred.

## 1. Observe

In parallel: `lane_occupancy`, `stranded_work_items`, and `list_work_items` for
akh-medu in phases `working`, `testing`, `ready`, `draft`, `parked`. Also grep
recent server events:

```
kubectl -n anima logs deploy/anima-server --since=2h | grep -aE 'result_handled|review_approved|review_rejected|ci_red|parked|solstice_dispatch"'
```

## 2. Report (always, before acting)

A compact table/list: per executor — current session + WI; akh-medu items by
phase with one-line status each (draft items included: what they're waiting on);
CI verdicts since last pass; anything parked and why. Lead with deltas.

## 3. Act — the human half of the loop

### Merging (testing items with green CI on the CURRENT head)
1. Confirm `ci_runs` has `success=true` + `pinned_to_current=true` for the PR head.
2. Check real mergeability locally — Forgejo's `mergeable` flag is often stale:
   `git fetch origin main <branch> && git merge-tree --write-tree origin/main FETCH_HEAD`.
   Conflict → route WI to `working` with a rebase note; do not merge.
3. Review basis: trivial/mechanical diffs review inline; substantive diffs get a
   delegated sonnet review agent (fetch `<pr>.diff` to /tmp first). REJECT routes
   back to working with the findings posted to the thread.
4. Strip `WIP: ` from the PR title via Forgejo PATCH (draft blocks merge).
5. `force_clear_ai_gate` → `approve_and_merge`. On MCP timeout, verify via
   Forgejo API (`merged: true`) and hand-land: `update_work_item(status=done,
   current_phase=done)` + a thread note with the evidence.
6. Post a merge note (head SHA, CI request id, review verdict, any handoffs).

### CI verdicts the broker ate (WI-534 class)
A dispatched run silent >1h while later dispatches completed = probably lost.
Pull ground truth: `ssh archibald 'docker logs solstice-orchestrator --since 4h'`
grep the request_id. `job finished ... success=true` + failed/absent publish →
hand-apply: merge via Forgejo API with the log line quoted as evidence in the
thread. Never invent a verdict — no finish line, no verdict.
Re-arming a run whose dispatch claim is stuck: push an empty commit
(`git commit-tree "$sha^{tree}" -p "$sha" -m "ci: re-arm"` — use `${new}:refs/heads/$ref`,
zsh eats `$new:r`), `resync_pr_state`, then flip the WI working→testing to fire
the `promote_retest` dispatch.

### Rebase-only fixups (parked `rebase-conflict`, or merge-tree conflicts)
Delegate to a sonnet agent in worktree isolation on `/Users/toasty/ws/anima`:
rebase onto fresh origin/main; prefer main for mechanical both-sides changes;
keep branch feature code; renumber colliding migrations to the next free number;
ABORT (needs-human) on semantic conflicts. Verify with scoped cargo check +
clippy `-D warnings` + the branch's own tests (PWA branches: typecheck/lint/test
instead). Force-push, then `resync_pr_state` + unpark: content-complete branches
→ `testing` (fires CI); branches with known defects → `working` with the finding
in the thread. Only *fix* what is rebase-mechanical — real defects get described,
not patched, and go back to their implementer.

### Queueing draft/ready items to the akhs
- Keep every live executor ≥2 items of runway. Pin via
  `update_work_item(executor=...)` (needs a no-op `status="open"` alongside).
- Draft plan items: promote to `ready` only when their predecessor merged
  (dependency order comes from the plan family — `list_work_items(plan_id=...)`
  — and the WI descriptions' "Blocked by" sections). Never promote a draft whose
  contract/predecessor hasn't landed.
- Match item to lane: well-scoped code fixes → either akh; long grinds and
  design-heavy items → tecton; probes/benches/mechanical ports → imhotep.

### Parked items — the main work of the pass, one item at a time

**This is not a bulk triage step.** `ready` and `draft` are routinely empty, so the
parked pile *is* the backlog: every future item is behind a park. Work them
**individually** — open the item, establish what is actually true about it now,
and resolve that item. `parked_reason` tells you why the machine stopped; it does
not tell you what the item needs. Do not batch by reason, and do not unpark a row
you have not read.

Per item, in this order:

1. **Check whether it already landed.** Parks are stale by construction, and work
   reaches `main` by routes the tracker never sees — a direct push, another WI's
   branch, an operator fix. `git log --oneline --all --grep=<keyword>`, then
   `git branch --contains <sha>` to confirm it is on `main`, then grep for the
   symbol/message in the tree. If it landed: post the commit SHAs and what they
   do, `update_work_item(status=done, current_phase=done)`, close the PR with a
   reason. *Three of five items closed in the 2026-08-14 pass were already fixed
   on main; two were fixed the previous day by the operator.* Do this before any
   other diagnosis — everything downstream is wasted if the work exists.
2. **Re-read the gates as live state.** `exit_bars` and `ci_runs` move after the
   park fires and nothing re-evaluates the queue. WI-267 went green **42 minutes
   after** its auto-park, and sat parked for three days on a verdict that was
   already obsolete. Confirm `pinned_to_current: true` — a green run bound to an
   older head is not a verdict.
3. **Read the branch, not the WI title.** `git diff --stat origin/main...<branch>`
   + `merge-tree`. Decide *content-complete or not* from the diff. A WI asking for
   three measurement tables whose branch is a 223-line harness is not complete,
   however green its CI — CI passes because a harness compiles.
4. **Then resolve by what you found:**
   - Landed → close (step 1).
   - Infra/fake failure (watchdog kill, sandbox misconfiguration, quota, flaky
     socket test) → unpark with a thread note naming the false cause. Genuine
     3-strike code failures stay parked for their implementer.
   - `too-big` that is real scope → **split it**, don't unpark it. That is the
     ceiling's prescribed remedy. Carve out the finished part, close it with
     evidence, and `create_work_item` for the remainder with the parent's
     findings carried across. WI-291 → asks 1+2 closed as landed, ask 3 became
     WI-297.
   - `too-big` that is misconfiguration scar tissue → `reset_in_flight_time` with
     an audit reason, crediting only the hours you can account for. Do not zero
     history you cannot justify.
   - `rebase-conflict` → rebase flow above.
   - `gate-needs-human` → NEVER auto-unpark; surface to the user, one line each.
5. **Before unparking, write the branch's real state into the thread.** Unpark
   dispatches immediately, and if the machine gate is green with the AI gate
   uncleared it dispatches a **review** — even into `working`, even on scaffolding.
   The note is what makes that reviewer adjudicate against the real acceptance
   bar. Name the missing deliverable explicitly.

Blocked sessions >30 min with no progress → `cancel_session` (frees the slot;
corrective dispatch resumes the WI).

### Known traps
- `approve_and_merge` timeout ≠ failure — always verify server-side before retrying.
- Forgejo `mergeable:false` is stale until the draft flag flips or main moves — trust merge-tree.
- A green run bound to a *previous* head is not a verdict for the current head.
- **Unpark dispatches a *review*, not an implementer**, when the machine gate is
  green and the AI gate is uncleared — regardless of phase or whether the branch
  is complete. `cancel_session` does not stop it: the gate condition is unchanged,
  so a new review session appears in ~40s. Don't fight it; state the branch's gaps
  in the thread and let the reject cycle route it back to `working`.
- **An ADR's `Status:` header is a claim about intent, not about the tree.** A
  2026-07-24 review declared ADR 049 dead and recommended retiring WI-180 on that
  basis; the Postgres store had landed five weeks earlier and the status line was
  simply never updated. Grep the code before believing any status field — in a
  review you are reading *or* writing.
- A PR being unmergeable is not the same as its WI being open. Close PRs that are
  superseded, don't compile, or would regress `main`, with the reason on both the
  PR and the thread — otherwise the next pass re-diagnoses them from scratch.
- Turns stamped `Refusal` by the idle watchdog are infrastructure deaths, not
  model refusals — check `/var/log/akhomed.out.log` for `chat turn stalled`.
- The Mac akhs build in `/Users/toasty/ws` — the user's live checkouts. Never
  "clean up" that tree.

## 4. Close the pass

Summarize deltas: merged / closed / split / routed / promoted / unparked /
cancelled, with WI codes and PR numbers, and **name the parked items you worked
and what you found on each** — "already on main as `<sha>`", "split into WI-N",
"scar tissue, credited 24h". A parked item you did not open is not a parked item
you triaged; say which ones you left and why (usually: ran out of pass, or
`gate-needs-human`).

List what is waiting on the user (operator-only items, `gate-needs-human`). If
nothing needed action, say so in one line.
