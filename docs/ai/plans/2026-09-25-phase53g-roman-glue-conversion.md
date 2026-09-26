# Plan: Phase 53g step 2 — converting the Roman Mac glue, one PR per script

- **Date:** 2026-09-25
- **Status:** Accepted. The operator asked for implementation, and the recommended defaults for decisions D1–D6 were adopted.
- **ADR:** [113](../decisions/113-full-akh-always.md), decision 10
- **Anima:** WI `01a0d569-1cf1` item 5 (umbrella `01a0d43b-bf90`)
- **Basis:** `origin/main` `c05dc0df` and the live `roman-akhs` deployment, read-only. The Phase 53f cutover is live.

## Conclusions

1. **Eight PRs, lowest risk first:**
   1. retire `t3_activity`;
   2. retire the `laya_service` classifier;
   3. make Vesta's Grafana access a direct MCP server;
   4. add a provider-failure taxonomy to akhomed;
   5. move observer monitoring into an akheventd monitor source;
   6. move the observer Anima scan plus `laya_controller` into an akheventd Anima source, with reactors and a skill;
   7. move `janus_escalation` and `mergiraf_worker` to 53e delegation;
   8. move `vesta_ops` and `vesta_actions` to execd and Kubernetes sandboxes.
2. **Stays deploy tooling** (listed in INVENTORY, no conversion): `daemon_credentials.py`, the credential seam, and `t3_entrypoint.py`, the T3 provider shim.
3. **Retire the Laya model.** Every one of the 52 classifications with complete input ended in `investigate`, with near-uniform per-choice scores (0.10–0.24 against 1/6). Every `product` and `infrastructure` route came from `host_action`'s structural rules, which match Anima machine codes. Those rules move into data-driven route filters.
4. **`provider_failure.py` is re-implemented, not ported.** It matches English prose. The Rust version classifies by HTTP status and `Retry-After`, and matches structured `error.code`/`error.type` values exactly against a lexicon loaded from config (no hardcoded natural-language constants).

## Operator decisions (defaults adopted)

| # | Decision | Adopted default |
|---|---|---|
| D1 | Laya model | Retire it (evidence above) |
| D2 | Grafana credentials for akheventd | A Grafana service-account token file plus the exported CA file, instead of extracting the admin secret with `kubectl` on every start (operator creates the token) |
| D3 | Where Astra/mergiraf workers run | akhsessd native Codex under ADR 113 D2/D3, with a Board `report_thread`; T3 stays as the rollback |
| D4 | mergiraf push authority | Fast-forward pushes to the existing PR source branch only; Janus merges only through Anima's gated path |
| D5 | Vesta's live-cluster and GitOps write powers | Keep them, enforced by an RBAC-scoped ServiceAccount and a `vesta/*`-only deploy key instead of Python argument checks (done last) |
| D6 | `anima-activity.py` | Retire it with PR 1 |

## Live dependency map

| Consumer | What it uses |
|---|---|
| All three akhs | `[[mcp.servers]]` `observer` (18436, `monitoring_snapshot`) and `laya` (18447, `record_fixer_result`) |
| Janus | `janus-escalation` (18438) |
| Vesta | `vesta-ops` (18437) |
| akheventd | Receives the observer outbox: `monitoring.changed` (coalesced) and `laya.<role>.intervention` |
| Observer | Posts the watchdog notice to Board thread `01a0ab49…` |

Restart privileges: akheventd and core are system LaunchDaemons. The core can also be restarted by stopping its process, since KeepAlive brings it back. Every Python service is a LaunchAgent and needs no sudo.

## PRs

### PR 1: retire `t3_activity.py` (risk: very low)

- **Today:** it mirrors akhsessd sessions into T3 and forwards T3 messages back. Its LaunchAgent is not loaded, and CUTOVER step 11 already made it obsolete.
- **Change:** delete the script and its test, drop the activity rendering from `configure_t3.py`, and mark it retired in INVENTORY.
- **On the Mac:** remove the plist and `bin/roman-activity-*`, and boot out `anima-activity` (D6).

### PR 2: retire `laya_service.py` (risk: very low; D1)

- **Change:** `laya_controller.Controller.classify` returns `investigate` with `classifier: retired` and makes no HTTP call. `CHOICES`, `digest` and `encoded` move into the controller. A new `configure_laya.py` renders the controller plist with the system `python3`, removing the dependency on the experiments venv.
- **Keep `laya.json.model_revision` unchanged:** it is part of the decision id, and changing it would re-decide every entity.
- **On the Mac:** boot out `dev.roman-akhs.laya`, then install and kickstart the controller.
- **Verify over 24 h:** decisions keep accumulating, and the action distribution matches the baseline.

### PR 3: Grafana passthrough becomes a direct MCP server (risk: low)

- **Change:** Vesta gets `[[mcp.servers]] name="grafana" url="http://127.0.0.1:18441/mcp"`, with a tool allowlist from `vesta-grafana-tools.json`. The rename rule `mcp__vesta-ops__grafana__X` → `mcp__grafana__X` goes into `full_akh.py`, the addenda and the seeds. `vesta_ops.py` drops its Grafana tools.
- **On the Mac:** re-render, restart the core, then kickstart `vesta-ops`.

### PR 4: provider-failure taxonomy in akhomed (risk: low–medium)

- **Classification:** `src/agent/provider_failure.rs` defines `ProviderFailureClass {Authentication, Quota, RateLimited{retry_after}, Transient, Unknown}`. It maps HTTP 401/403, 402 and 429 plus `Retry-After`, and matches structured error codes exactly from the `[provider]` lexicon.
- **`ProviderGate`:** persisted in engine meta as `provider:gate:v1` and in the snapshot extra. The cooldown ladder comes from config (default 30 m, 1 h, 2 h, 4 h).
- **Reactor effect:** `LlmTurn` and `DjehutyReply` claims are deferred while the gate is closed, and these deferrals do not count toward the trip-wire. An authentication failure stays gated until a turn succeeds.

### PR 5: observer monitoring becomes an akheventd monitor source (risk: medium; D2)

- **Source:** akheventd gains `Config.monitors` (health endpoints, alertmanager, prometheus) and a structural `monitoring_signature`. It emits coalesced `monitoring.changed` events. An unavailable feed is reported as unknown, never as zero alerts. It serves `GET /v1/monitors/{source}`.
- **Tool:** a native read-only `monitoring_snapshot` akh tool replaces `mcp__observer__monitoring_snapshot`.
- **Observer:** keeps only the watchdog, for now.
- **Implemented** (branch `feat/53g-pr5-monitor-source`):
  - **akheventd:** the source lives in `crates/akh-session-services/src/monitor.rs`
    and the shared wire types in `crates/akh-session-types/src/monitor.rs`.
    - The signature is the blake3 digest of a canonical structural
      projection. The event ID is blake3 of the signature plus the
      transition number, so a recurring state is not refused as a reused ID.
    - The snapshot and the event commit in one transaction. The first poll
      is a silent baseline.
    - `Endpoint.anonymous` endpoints send no token (used for the health
      probes).
  - **akhomed:** the tool is `src/agent/tools/monitoring_snapshot.rs`,
    configured by the workspace `[monitoring]` table. It keeps the old
    top-level keys, and the observer-owned status keys come from
    `state/observer-status.json` until PR 6.
  - **Deploy:** the D2 operator action and the Mac-side steps are in
    `deploy/roman-mini/INVENTORY.md` ("PR 5").

### PR 6: Anima source plus `laya_controller` become an akheventd Anima source, reactors and a skill; retire `observer.py` (risk: medium–high)

- **Source:** akheventd gains `Config.anima`, which scans, baselines, hydrates parked items, derives entity ids from content hashes, caps emissions per park episode, and has a shadow mode.
- **Routing:** data-driven `EventRoute.when` filters, using JSON-pointer predicates. The `host_action` table lives in `events.json` and is rendered by `full_akh.py`; none of it is in Rust. The watchdog moves into akheventd.
- **Akh side:**
  - reactor seeds `roman-anima/anima.work_item.changed` for all three Romans;
  - `record_fixer_result` becomes a memory write (`roman-memory` v1.1.0 adds `intervention_outcome` and `intervention_next_event`);
  - a `laya-intervention` skillpack.
- **Rollout:** 48 h in shadow mode, compared against `laya-decisions.sqlite3`, then the cutover.

### PR 7: `janus_escalation` plus `mergiraf_worker` become Janus delegation and a skill (risk: high; D3, D4)

- **Delegation:**
  - per-project NativeSession delegation profiles;
  - a durable incident ledger (`incident_key` dedup, at most 4 new per day);
  - native tools `escalate`, `dispatch_merge_repair` and `escalation_status`, marked Dangerous and Janus-only.
- **Policy** lives in skillpacks loaded by path.
- **Merge repair:** a `git_publication` profile allows fast-forward pushes to PR branches only.
- **Deploy:** services and core go together, readers before writers.
- **Implemented (repository side, 2026-09-26):**
  - `src/agent/delegation/ledger.rs`: `IncidentLedger` (`delegation:incidents:v1`
    in engine meta and `extra.delegation.incidents` in snapshots), dedup by
    validated `IncidentKey`, `[delegation.native] max_new_per_day` (default 4)
    per UTC day, a stable create request id per incident.
  - `src/agent/delegation/escalation.rs` + `src/agent/tools/escalation.rs`:
    `escalate`, `dispatch_merge_repair`, `escalation_status` (Dangerous,
    Mutating, outside the legacy `/v1` scope), registered only when
    `[delegation.native.escalation] enabled`. Per-project worker profiles in
    `[delegation.native.profiles]` (a bare name or `{ workspace_profile,
    runtime_profile }`; the old single `workspace_profile` stays the default).
    A session on another runtime than `runtime_profile` is retired and the
    incident fails closed. Failed turns are classified by fault code against
    the `[provider]` lexicon (PR 4 taxonomy).
  - akhsessd: `session_grants.<profile>.{runtime, instructions}` overrides
    (a worker profile runs Codex under the akh's identity without its
    persona) and `GET /v1/session-requests/{request_id}`, which restart
    reconciliation uses because a replayed create cannot carry the lost grant
    secret. `git_publication` profiles gain `require_existing_branch` (D4:
    no branch creation, no force flag).
  - Worker policy: `deploy/roman-mini/skills/{astra-escalation,mergiraf}`,
    loaded by path on every dispatch and sent as the worker's first input.
  - `full_akh.py`/`configure_core.py` render Janus's `[delegation]` and drop
    `[[mcp.servers]] janus-escalation`.
  - Mac-side steps, verification, rollback and the remaining D4 gap (host
    executor with ambient Git credentials) are in
    [INVENTORY.md](../../../deploy/roman-mini/INVENTORY.md) ("PR 7 Mac-side
    steps"). `janus_escalation.py`, `mergiraf_worker.py` and
    `provider_failure.py` are deleted in a follow-up after the verified cutover.

### PR 8: `vesta_ops` plus `vesta_actions` move to execd and Kubernetes sandboxes (risk: highest; D5)

- **Exec:** Vesta gets `[exec] kind="execd"` with a Kubernetes sandbox pod (a Talos checkout, no credentials).
- **Writes:** a second profile runs under an RBAC-scoped ServiceAccount with a deploy key limited to `vesta/*`.
- **Procedures** move to a `vesta-remediation` skillpack.
- **Order:** last, and only after PR 3.

## Dependencies

- PR 5 needs PR 3.
- PR 6 needs PR 2 and PR 5.
- PR 7 needs PR 4.
- PR 8 needs PR 3.
- PRs 1–4 are independent.

The full per-PR detail (files, verification, rollback) is in the planning notes behind this document and is repeated in each PR description.
