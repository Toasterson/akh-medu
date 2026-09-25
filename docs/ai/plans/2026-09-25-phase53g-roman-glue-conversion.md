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
