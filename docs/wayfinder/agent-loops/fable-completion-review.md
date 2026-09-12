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

Natural finalization and scheduler route integration remain implementation work.
Fable's suggestion to call a no-result model turn a failure must be reconciled
with Anima's existing `Empty`/`Failed` distinction. Likewise, a stop-turn remains
resumable; it must not silently become session retirement.
