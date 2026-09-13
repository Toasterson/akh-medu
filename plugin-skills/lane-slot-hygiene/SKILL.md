---
name: lane-slot-hygiene
description: Detect and recover a session stuck "provisioning" forever with an allocated sandbox and no owner-driven progress — the akhsessd/Anima/Archibald failure mode from the 2026-09-13 stuck-session incident. Use when Anima shows provisioning sessions with zero witnesses, cancel_session returns pending_central_retirement that never resolves, or akhsessd logs repeat "lease renewal deferred" / "reconciliation deferred".
version: 0.1.0
---

# Lane slot hygiene

## The invariant

A session that Anima dispatched a work item to must always end up either
making progress (witnesses > 0) or in a terminal status. It must never sit in
`provisioning` indefinitely while holding a sandbox lane on Archibald. akhsessd
now enforces this itself (see below); this skill is for confirming the
enforcement actually ran, and for manual recovery if it hasn't yet (e.g. before
the fix ships, or if the reconciliation interval hasn't elapsed).

## Where the reconciliation lives

- `crates/akh-session-services/src/server/sessions.rs`: `SessionApp::reconcile_progress`
  (called every dispatcher tick, throttled by `reconciliation_interval_ms`,
  default 5 min) force-retires a session quiet longer than
  `session_stall_threshold_ms` (default 15 min), and finalizes any retiring
  session once every turn is terminal — including forcing a turn whose
  cancellation an unreachable runtime never confirmed, after
  `retirement_cancel_grace_ms` (default 2 min).
- `crates/akh-session-services/src/session.rs`: `force_retire`,
  `force_cancel_unreachable`, `finalize_retirement_if_ready`, and the pure
  predicates `is_stalled` / `should_force_cancel` (unit tested there).
- The workspace-lease-expired case is handled inline where it's detected:
  `server/sessions.rs`, the lease renewal branch — `code == "expired"` retires
  immediately instead of warning forever.

## How to detect a stuck session

1. `mcp__anima__lane_occupancy` — look for sessions in `provisioning` with
   `witness_count: 0` and an age over 15 minutes.
2. On Archibald: `kubectl -n akh-workspaces get pods -L akh-owner` — an idle
   pod (only `akhexecd` running, no work) whose owner session matches one from
   step 1 confirms the sandbox is still held.
3. akhsessd logs, module `akh_session_services::server::sessions` — repeated
   `workspace lease renewal deferred code=expired` or `session reconciliation
   deferred code=transport` for the same session id without a following
   `session_retiring` / `session_archived` event is the signature of the bug
   this skill exists for.

## Manual recovery

1. `mcp__anima__cancel_session` on the stuck session id.
2. If it still reports `pending_central_retirement` after
   `retirement_cancel_grace_ms` + `reconciliation_interval_ms` (default
   ~7 minutes) has passed, the fix above did not run — check akhsessd is
   actually up and its dispatcher loop is ticking (readiness probe, `/healthz`).
3. Verify sandbox release: on Archibald, confirm the pod for that session's
   allocation is gone or retained-and-idle, not still running work.
4. If the sandbox is still allocated after the session shows `closed_at`, that
   is a separate akhsbxd-side release bug — file it distinctly; do not assume
   this skill's reconciliation sweep is responsible for akhsbxd's own retries.
