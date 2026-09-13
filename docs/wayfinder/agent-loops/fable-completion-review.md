# Fable completion review

2026-09-12. Local design review with the requested `claude-fable-5`, through
`/Users/toasty/.local/bin/claude` using `/Users/toasty/.claude`. The keychain/login
issue was resolved; the request returned `is_error=false` with model output.
Fable had tools disabled and reviewed supplied code excerpts. This is design
input, not CI, a PR approval or deployment evidence.

The review supports keeping canonical Change/result verification before closure,
retaining ownership for uncertain PR creation, evaluating the complete follow-up
turn set, and separating central route capacity from physical runner capacity.
It recommends composing outcome, witness, capability revocation, audit and exact
claim release in a single Anima transaction after authoritative Akh cleanup.
Completed model turns must never satisfy review, CI, dependency or merge gates.

Findings checked against the implementation:

- A delayed admission response can regress `observed` to `admitted`. Preserve
  `observed` in that update, matching admission-revocation reconciliation.
- Retirement can stage a new intent on an already closed owner. Treat a closed
  owner as handled without adding another terminal intent.
- Natural completion needs pending-PR and verified-source guards inside its
  canonical commit. The compatibility transport closer alone is insufficient.
  A PR Change uses the source snapshot association; it has no snapshot row of
  its own. Do not require the external PR Change ID to equal the source diff ID.
- Late session inputs require conditional archival against the observed journal
  prefix, including completed follow-ups. `archive-selected` now reserves the
  exact generation/cursor decision and rejects intervening events atomically.
- Fable's proposed project-lock mismatch is not present: staging, canonical
  allocation and dependency writers use the same project advisory key. UUID key
  collisions cause extra serialization, not a missing exclusion boundary.
- The claim currently writes session state, not the work-item fields included in
  the staging digest; native staging/admission tests pass. Keep the strict
  freshness check and retain `pending_prompt` in its authority snapshot because
  an operator's changed instructions must invalidate staged input.
- Akh retirement receipts are already immutable by request identity, with replay
  tests. Do not weaken equality to accommodate hypothetical changing timestamps.
- Current-phase selection remains strict for mutation. Historical verification
  and phase-moved cleanup need distinct handling; removing the current-phase
  check from the shared mutation candidate would weaken PR gate protection.

Natural finalization and scheduler route integration are now implemented in the
Anima worktree. Empty completed turns remain distinct from failures, and stopping
a turn remains resumable rather than silently retiring the session.

## Follow-up review of finalization and scheduling

A second authenticated Fable review inspected the new completion, conditional
close, scheduler adapter and lifecycle code. Its eight findings were checked
against the full implementation:

- Verified deliverables must count as progress even when a later provider turn
  fails. Fixed in selection: result/document disposition takes precedence; the
  terminal provider error remains in the archived checkpoint. The transport
  accepts failed terminal checkpoints with an explicitly verified artifact
  disposition. A full-schema test verifies no failure-budget debit.
- Ready-to-working preparation commits before network staging. This is deliberate
  lifecycle preparation, not allocation evidence. Working items without owners
  remain canonical dispatch candidates. The real operator-tick test now injects
  an S3 failure, verifies no authority/session rows, then successfully retries the
  working item with zero physical runner slots. No compensating backwards phase
  transition is needed or safe after concurrent lifecycle edits.
- `apply_close_in_tx` returns false only for the exact previously committed
  closure. It returns a conflict for a changed/cleared intent. The alleged false
  success for an open superseded owner is absent; exact replay tests cover it.
- The suggested fallback to an older published Change would discard newer
  unverified work. Retain ownership when the latest selected proposal lacks
  successful publication/PR evidence. Operators can inspect publication state,
  repair through the existing session or use attributed retirement; an ordinary
  completion must not silently discard it.
- Canonical `compute_next_action` returns `RedispatchWithoutCharge` only for
  refusal/cancellation. Failed/empty attempts are charged once; dependency waits
  do not erase the already-recorded attempt. Replay tests verify single charging.
- Document completion can record immutable session-authored history after a phase
  moves, while preserving the newer lifecycle. This differs from publishing a
  PR, which mutates current upstream/result state. Added an explanatory comment.
- Revocation cannot normally race a selected admitted close through the supported
  writer, which refuses revocation after admission. Added explicit revocation
  guards to selected archival and final apply as defense in depth.
- Completion preparation now recognizes an already-closed selected completion
  and returns without restaging it. Retiring/unadmitted owners remain pending.
  The full-schema test covers the closed selection retry.

Both reviews are design input from supplied excerpts, not exact-head approval or
CI. Findings that contradict the underlying canonical APIs are documented rather
than applied blindly.
