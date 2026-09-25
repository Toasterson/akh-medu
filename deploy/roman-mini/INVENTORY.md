# Roman Mac glue inventory (Phase 53g step 1)

Source of truth for this pass was the live deployment at
`~/Library/Application Support/roman-akhs` (read-only; not modified by this
pass) on 2026-09-24. See `docs/ai/phases/phase-53/README.md` ("53g") and
[ADR 113](../../docs/ai/decisions/113-full-akh-always.md) decision 10 for the
mandate: every deployed script lives in this repo and is retired one at a time
as its akh-native replacement lands.

Legend for **Status**:
- `in repo` — matched live byte-for-byte already.
- `drifted → synced` — live had diverged from the repo copy; this pass copied
  live over the repo file (live is the source of truth) and the diff is
  summarized below.
- `added` — not previously tracked; added from live in this pass.
- `installer, unchanged` — one of the `configure_*.py` / `prepare.py` /
  `activate.py` rendering scripts; not part of the drift check (they render
  config, they aren't deployed standalone), confirmed present and current.

## Deployed Python services (`roman-akhs/bin/*.py`)

| Script | Purpose | Inputs / outputs | Launchd label | Status | ADR 113 target |
|---|---|---|---|---|---|
| `daemon_credentials.py` | Resolves 1Password / file-based credentials into the environment for the daemon it `execve`s into (`runtime*`, `sessions`, `events`, now `core`); never writes secrets to disk. | in: `sys.argv[ROLE, ROOT]`, `op-service-account`, `anima-pat`, `secrets/board-*.json`; out: `execve`s `akhrtd`/`akhsessd`/`akheventd`/(soon) `akhomed` with an augmented env. | `dev.roman-akhs.events`, `.sessions`, `.runtime`, `.runtime-minerva`, `.runtime-vesta` (all exec through it) | drifted → synced (see below) | deploy tooling kept — this is exactly the credential-injection seam ADR 113 decision 10 asks the `core` plist to route through at cutover (53f) |
| `janus_escalation.py` | Durable outbox broker: turns a `escalate_to_astra` / `dispatch_mergiraf` MCP tool call into a T3/Astra thread; the delivery survives T3 being offline. | in: MCP `tools/call` over loopback HTTP, `config/janus-escalation.json` (registered T3 project IDs), `secrets/janus-escalation.key`; out: T3 `/api/orchestration` thread/turn commands, `state/*.sqlite3` incident ledger. | `dev.roman-akhs.janus-escalation` | drifted → synced (see below) | Janus delegation (53e grants) + skill, per the plan's step-2 list |
| `laya_controller.py` | Durable event→classification→scoped-fixer controller: reads an observation queue (`state/laya-decisions.sqlite3`), classifies a blocker (product/execution/infrastructure/human/wait/investigate) and routes an intervention to the right role (`ROLE` map) or opens a mergiraf/Astra dispatch. Serves a small results API on `127.0.0.1:18447`. **`classify()` no longer calls the Laya model** (see "`laya_service.py` retired" below) — it returns `{'choice': 'investigate', 'classifier': 'retired', ...}` synchronously, and `CHOICES`/`digest`/`encoded` (formerly imported from `laya_service`) are now defined in this module. `config/laya.json`'s `model_revision` is still read unchanged (it participates in the decision id). Now installed with `configure_laya.py` on the system `python3` instead of the `experiments/` venv. | in: observations from `observer.py` (when `config/laya.json.event_only` is true), `secrets/laya-<role>.key`; out: `laya.<role>.intervention` outbox events, HTTP decisions API. | `dev.roman-akhs.laya-controller` (LaunchAgent) | in repo | Janus reactor + skill (plan's explicit step-2 item; PR 2 of that step retired the classifier first) |
| `laya_service.py` | **Retired 2026-09-25** (Phase 53g step-2 plan, PR 2, operator decision D1). Was a resident, loopback-only classifier (`127.0.0.1:18446`) that scored blocker text against the 6 `CHOICES` with a local model and validated the returned probability simplex. **Rationale:** live decision-log evidence showed every one of the 52 complete-input classifications ended in `investigate` with near-uniform per-choice scores (0.10–0.24 against a uniform 1/6); every real route to `product` or `infrastructure` came from `laya_controller.py`'s `host_action` structural rules (which match Anima's phase/parked_reason/session fields), never from the model's score. Deleted from the repo; its `dev.roman-akhs.laya` LaunchAgent is booted out on the Mac side (see Mac-side steps below), not replaced. | — | `dev.roman-akhs.laya` (LaunchAgent, booted out) | retired | n/a — folded into `laya_controller.py`'s structural router, no successor process |
| `mergiraf_worker.py` | **Not in repo before this pass.** Library (no server, no launchd entry) imported by `janus_escalation.py`; defines the `POLICY` prompt, JSON-schema `TOOLS` (`dispatch_mergiraf`, `mergiraf_status`) and `escalation()` builder for a bounded, typed merge-conflict-resolution job dispatched to an Astra/T3 worker via the same broker. | in: called in-process from `janus_escalation.Broker.submit_merge`; out: a T3 thread prompt + schema-validated job fields (work item, PR, head/base SHAs). | none (imported, not a service) | added | Janus delegation (53e grants) + skill, alongside `janus_escalation.py` |
| `observer.py` | Polls Anima (rotating 8 projects/min), Grafana and VictoriaMetrics; maintains a deterministic watchdog signal, a durable outbox of `schedule.*`/`anima.*.changed`/`monitoring.changed` events, and (new) `laya.<role>.intervention` events when `event_only` mode is on; serves a read-only `monitoring_snapshot` MCP tool on `127.0.0.1:18436`. | in: Anima MCP, Grafana HTTPS, VictoriaMetrics via `127.0.0.1:18440` tunnel, `config/laya.json`; out: `state/observer.sqlite3` outbox/state, POSTs to akhsessd `127.0.0.1:18431`, board watchdog notices. | `dev.roman-akhs.observer` (LaunchAgent) | drifted → synced (see below) | event production → Rust `akheventd` source; monitoring snapshot → a Vesta tool (both named explicitly in the plan's step-2 list) |
| `provider_failure.py` | Classifies a provider HTTP/error response into `authentication`/`quota`/`rate_limit`/`unknown` so `observer.py` and `janus_escalation.py` apply the right cooldown (30m/1h/2h/4h, or operator-repair-only for auth). | in: response text/status from a provider call; out: a classification tag consumed by callers. | none (imported library) | in repo | folds into akhomed's provider error taxonomy (explicit step-2 item) |
| `t3_activity.py` | T3 ACP driver used as the transport for the "autonomous activity" provider instances (`roman-activity-<role>`); also runs as a background `--service` LaunchAgent. | in: T3 CLI invocation or `--service` mode; out: ACP session against akhsessd `127.0.0.1:18431`, reads `secrets/operator.key`. | `dev.roman-akhs.t3-activity` (LaunchAgent) | **retired 2026-09-25** (ADR 113 / 53g plan PR 1 — see below) | done — once T3 speaks ACP directly to akhomed there is no separate "activity" transport to shim |
| `t3_entrypoint.py` | T3 ACP driver used as the transport for the three named Roman provider instances (`roman-janus`/`roman-minerva`/`roman-vesta`); dispatches on `sys.argv[0]`'s basename. | in: T3 CLI invocation, `secrets/t3-<role>.key`; out: `akh-acp` subprocess against `ws://127.0.0.1:18431/acp` (akhsessd) today. | n/a (invoked by T3, not launchd) | drifted → synced (see below); **also changed in this pass** — see "t3_entrypoint.py ACP endpoint" below | akhomed ACP once 53d/53f land; this pass adds the seam, doesn't flip the default |
| `vesta_actions.py` | **PR 3 landed:** the `grafana_rpc` JSON-RPC client to the Grafana MCP tunnel (`127.0.0.1:18441`) is now used only by `configure_vesta_actions.py`'s discovery step (writing `config/vesta-grafana-tools.json`); `vesta_ops.py` no longer calls it to proxy tool calls. Still provides the `gitops_publish`/`homelab_write` action tools for Vesta's sandbox. | in: MCP `tools/call` (action tools), Grafana `tools/list` (discovery only); out: Grafana RPC, GitOps sandbox operations. | none (imported by `vesta_ops.py`/`configure_vesta_actions.py`) | in repo | Vesta skill (homelab remediation stays a skill under the akh-native Vesta, not a separate broker) |
| `vesta_ops.py` | `homelab_inspect` (read-only kubectl get/describe/logs) MCP broker for Vesta, serving on a loopback port; the read-only counterpart to `vesta_actions.py`'s write path. **PR 3 landed:** dropped its `grafana__*` passthrough tools and the `grafana_rpc` import — Vesta now reaches Grafana through a direct `[[mcp.servers]] name = "grafana"` entry pointed at the tunnel (`http://127.0.0.1:18441/mcp`, no auth) instead of this broker. | in: MCP `tools/call`; out: bounded `kubectl` invocations. | `dev.roman-akhs.vesta-ops` (LaunchAgent) | in repo | Vesta skill, same as `vesta_actions.py` |

### Drift summaries (live vs. previous repo copy)

- **`daemon_credentials.py`**: live extended the `runtime` role match to also cover
  `runtime-minerva` and `runtime-vesta` (per-role runtime processes that didn't
  exist when the repo copy was made). One-line change, now synced. This pass
  additionally adds a `core` role (see below) on top of the synced version.
- **`janus_escalation.py`**: live added the mergiraf dispatch path —
  `submit()` gained a `worker` keyword (`"astra"` default, `"mergiraf"` for
  merge jobs), a new `submit_merge()` entry point, a reserved `mergiraf-`
  incident-key prefix, `mergiraf_status`/`dispatch_mergiraf` tool wiring, and
  imports `mergiraf_worker` (previously undeployed to the repo). Now synced.
- **`observer.py`**: live added a `laya_controller` integration behind
  `config/laya.json`'s `event_only` flag: when set, scheduled-wake replay stops
  (occurrences are marked `accepted=2` instead of resubmitted), Anima/monitoring
  changes are pushed into `laya_controller.Queue` instead of the old
  per-role `pending` table, `schedules()` becomes a no-op, and the watchdog
  signal gains `laya_controller_available`/`laya_attention`. This is a real
  architecture shift (event-driven decision queue replacing fixed-interval
  polling) landed live but not yet reflected anywhere in the repo. Now synced.
- **`t3_entrypoint.py`**: live's `models` health-check print string says
  "Mistral Large 3 via Ollama Cloud"; the repo copy said "GLM-5.3 with thinking
  via Ollama Cloud" (stale — the daemon model changed and this cosmetic string
  wasn't updated in the repo copy). Now synced to live's text.
- **`provider_failure.py`, `t3_activity.py`, `vesta_actions.py`,
  `vesta_ops.py`**: byte-identical to live; re-copied only to keep the sync
  mechanical (no behavior change; `t3_activity.py`'s executable bit was
  normalized to match the repo's `100644` convention for the other scripts).

### `daemon_credentials.py`: new `core` role

Added an `elif role == 'core':` branch that injects `ROMAN_OLLAMA_KEY` (same
`op://Homelab/Ollama API Key Akhrtd Command Group/api key` reference the
`runtime*` roles already read) plus the MCP tokens the akh core will need once
`[[mcp.servers]]` (Phase 53i) is wired: `ROMAN_ARCHIVE_TOKEN` (Seshat — reusing
the existing name from the `sessions` role; live's `configure_integrations.py`
also authenticates the Sokar MCP endpoint with this same token today, so there
is no separate Sokar credential to introduce yet), `ROMAN_ANIMA_TOKEN`
(existing name, same `anima-pat get-auth-header` resolution), and
per-akh `ROMAN_BOARD_JANUS`/`ROMAN_BOARD_MINERVA`/`ROMAN_BOARD_VESTA` (existing
per-akh naming from the `sessions`/`events` roles — a single flat
`ROMAN_BOARD_TOKEN` would not fit a process hosting three akh identities at
once). Because akhomed takes its config from `AKH_CORE_CONFIG` and friends in
the environment rather than a `--config` argument, and its binary name is
content-addressed (`bin/akhomed-<sha[:12]>`, see `configure_core.py`), the
`core` branch reads `evidence/core-binary.json` for the executable path and
`execve`s it directly instead of falling into the shared
`root/'bin'/binary --config ...` tail used by the other roles. **Existing
roles (`runtime*`, `sessions`, `events`) are otherwise byte-identical** to the
synced live version; nothing in `core.plist` was changed by this pass — see
follow-ups below.

### `t3_entrypoint.py`: akhomed ACP endpoint seam

Added two environment overrides, both unset by default so behavior is
unchanged until the 53f runbook verifies the target:

- `ROMAN_T3_ACP_ENDPOINT` — overrides the `--endpoint` passed to `akh-acp`
  (default stays `ws://127.0.0.1:18431/acp`, i.e. akhsessd).
- `ROMAN_T3_HEALTH_URL` — overrides the URL the `models` health-check probes
  (default stays `http://127.0.0.1:18431/v1/sessions`).

Resolved in 53d/53f prep: akhomed's TCP ACP does not read `_meta.akh`, so
`akh-acp --akhomed` bridges it. `configure_core.py` renders `config/t3-acp.json`
(`tcp://127.0.0.1:18444`, `/healthz` on 18430, the `akh-acp-main-<sha>` binary);
when that file exists `t3_entrypoint.py` execs `akh-acp --akhomed ... --t3-compat`
without `--workspace-profile`. Deleting the file is the rollback.

## Shell-wrapper "scripts" (`roman-janus`, `roman-minerva`, `roman-vesta`)

These are not independently maintained scripts. `configure_t3.py`
(`shutil.copy2(source / 't3_entrypoint.py', binary)`) installs one physical
copy of `t3_entrypoint.py` per role as `bin/roman-<role>`; the role is
selected at runtime from `sys.argv[0]`'s basename. Confirmed live:

- `roman-janus/minerva/vesta` differ from live `t3_entrypoint.py` by exactly
  the same stale "GLM-5.3" vs. "Mistral Large 3" string noted above — they
  were installed before that repo/live edit and `configure_t3.py` has not been
  re-run since. Re-running `configure_t3.py` (part of the normal
  install/update flow, not part of this pass) refreshes them from the now-synced
  `t3_entrypoint.py`.

No separate files are added to the repo for these; they are covered by keeping
`t3_entrypoint.py` current and re-running `configure_t3.py`.

**Retired 2026-09-25 (ADR 113 / 53g plan PR 1):** `roman-activity-janus`,
`roman-activity-minerva` and `roman-activity-vesta` — the byte-identical
`t3_activity.py` copies installed alongside the three `roman-<role>`
entrypoints above. `configure_t3.py` no longer renders them, their
`providerInstances` entries, or `bin/t3_activity.py`/the `t3-activity`
LaunchAgent plist. See "Mac-side steps for PR 1" below for removing what a
previous `configure_t3.py` run already installed on the live Mac.

## Anima runner glue outside the deployment root (inventoried, not copied)

Lives at `~/.config/anima-runner/roman-chat/`, a separate mechanism from
`roman-akhs` (its own `anima-runner` binary, own token files, own state dir).
Not trivially safe to fold into `deploy/roman-mini` in this pass — it has its
own binary/config lifecycle and no existing renderer here to extend.

**`anima-activity.py` retired 2026-09-25 (D6, ADR 113 / 53g plan PR 1):** it
lives outside this repo (under `~/.config/anima-runner/roman-chat/`, not
`deploy/roman-mini`), so there is no repo-side change for it — this pass
records the retirement decision and the Mac-side step (boot out
`dev.roman-akhs.anima-activity`) in "Mac-side steps for PR 1" below.

| File | Purpose | Launchd label |
|---|---|---|
| `anima-activity.py` | Projects existing Roman automation wakes into Anima as AKH-authored messages (never human prompts); read-only against Akh sessions/tools, with a local outbox for ambiguous deliveries. | `dev.roman-akhs.anima-activity` (LaunchAgent, `gui/<uid>`) — **retired 2026-09-25** |
| `roman-chat.py` | Chat-only Anima connector to the existing Mac Roman akhs; `--runner` mode starts the Anima-side runner, `janus`/`minerva`/`vesta` mode `execv`s `akh-acp` against akhsessd exactly like `t3_entrypoint.py`. Starts no provider/cognitive daemon itself. | none directly (invoked by the `anima-runner` binary) |
| `anima-runner` (binary) | Vendored runner executable (not a script); out of scope for a python-glue sync. | — |
| `activity.json`, `activity-credential.json`, `activity-token`, `runner-token` | Config + credential files (paths/URLs and a narrow PAT the runner authenticates with); **not inspected for values**, only confirmed to be file-based, not embedded in a script. | — |
| `state/`, `logs/`, `proto/` | Runtime state, logs, and gRPC proto defs for the runner. | — |

**Resolved 2026-09-25 (D6):** `anima-activity.py` is retired with PR 1, not
folded into the akheventd-source conversion — see the retirement note above.
`roman-chat.py` (the chat-only Anima connector) is unaffected and stays as is.

## Launchd plists (names / program / env-var **names** only — no values read)

### System LaunchDaemons (`/Library/LaunchDaemons/dev.roman-akhs.*.plist`)

| Label | Program | Env var names | Rendered by |
|---|---|---|---|
| `core` | `bin/akhomed-d563ba34a247` | `AKH_SERVER_PORT`, `AKH_AUTO_START`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `AKH_SYSTEM_PATHS`, `AKH_SERVER_BIND`, `XDG_RUNTIME_DIR`, `DATABASE_URL`, `RUST_LOG`, `AKH_CORE_CONFIG`, `XDG_CACHE_HOME`, `XDG_STATE_HOME` | `configure_core.py` (in repo) |
| `board-tunnel` | `kubectl -n djehuty-board port-forward … 18089:8080` | `KUBECONFIG`, `HOME`, `PATH` | `configure_integrations.py` (in repo) |
| `events` | `python3 daemon_credentials.py events <root>` | `KUBECONFIG`, `HOME`, `PATH` | `activate.py` (in repo) |
| `executor` | `bin/akhexecd --config config/executor.json` | `RUST_LOG` | `prepare.py` (in repo) |
| `metrics-tunnel` | `kubectl -n observability port-forward … vmsingle 18440:8428` | `KUBECONFIG`, `HOME`, `PATH` | `activate.py` (in repo) |
| `otlp-tunnel` | `kubectl -n observability port-forward … alloy 19418:4318` | `KUBECONFIG`, `HOME`, `PATH` | **no renderer in the repo** — follow-up |
| `runtime` | `python3 daemon_credentials.py runtime <root>` | `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_RESOURCE_ATTRIBUTES`, `HOME`, `PATH`, `OTEL_TRACES_SAMPLER`, `AKH_OTEL_CAPTURE_CONTENT`, `RUST_LOG` | `prepare.py` (in repo; OTLP env additions are hand-layered on top — not fully reproduced by any script) |
| `runtime-minerva` | `python3 daemon_credentials.py runtime-minerva <root>` | `PATH`, `HOME`, `RUST_LOG` | **no renderer in the repo** — follow-up |
| `runtime-vesta` | `python3 daemon_credentials.py runtime-vesta <root>` | `PATH`, `HOME`, `RUST_LOG` | **no renderer in the repo** — follow-up |
| `sandbox` | `bin/akhsbxd --config config/sandbox.json` | `RUST_LOG` | `prepare.py` (in repo) |
| `sokar-tunnel` | `kubectl -n akh-medu port-forward … akh-medu-sokard 18439:9400` | `KUBECONFIG`, `HOME`, `PATH` | `configure_integrations.py` (in repo) |

### User LaunchAgents (`~/Library/LaunchAgents/dev.roman-akhs.*.plist`)

| Label | Program | Env var names | Rendered by |
|---|---|---|---|
| `anima-activity` | `python3 anima-activity.py --config activity.json` (outside `roman-akhs`) | `PATH` | not part of `deploy/roman-mini` — **retired 2026-09-25**, see Anima runner glue above |
| `grafana-tunnel` | `kubectl -n observability port-forward … mcp-grafana 18441:80` | `KUBECONFIG`, `HOME`, `PATH` | `configure_vesta_actions.py` (in repo) |
| `janus-escalation` | `python3 janus_escalation.py <root>` | `PATH`, `HOME` | `configure_janus_escalation.py` (in repo) |
| `laya-controller` | `/usr/bin/python3 laya_controller.py <root>` | `TOKENIZERS_PARALLELISM`, `USE_TF`, `HF_HUB_OFFLINE` | `configure_laya.py` (in repo, this pass). Was `<repo>/experiments/typed-decisions/.venv-comparison/bin/python laya_controller.py <root>` — the venv interpreter loaded the retired classifier's torch/transformers dependency, which the controller no longer imports, so this pass moves it to the system Python like every other roman-akhs LaunchAgent. |
| `laya` | *(retired 2026-09-25, see `laya_service.py` above)* | `TOKENIZERS_PARALLELISM`, `USE_TF`, `HF_HUB_OFFLINE` | n/a — no successor plist; the Mac-side steps below boot it out and don't replace it |
| `observer` | `python3 observer.py <root>` | `KUBECONFIG`, `HOME`, `PATH` | `activate.py` (in repo) |
| `sessions` | `python3 daemon_credentials.py sessions <root>` | `PATH`, `HOME`, `RUST_LOG` | `prepare.py` (in repo) |
| `t3-activity` | `python3 t3_activity.py --service` | `PATH`, `HOME` | **retired 2026-09-25** — `configure_t3.py` no longer renders it |
| `vesta-ops` | `python3 vesta_ops.py <root>` | `KUBECONFIG`, `HOME`, `PATH` | `configure_vesta.py` (in repo) |

## Follow-up work items (step 2, filed per the plan's explicit list)

Each of these needs its own work item before it lands (not done in this pass):

1. **`observer.py` event production → Rust `akheventd` source**, and its
   `monitoring_snapshot` MCP tool → a Vesta tool. Also decide whether
   `anima-activity.py` (Anima-runner glue, not `roman-akhs`) folds into the
   same conversion.
2. **`laya_controller.py` → a Janus reactor plus skill.** ~~`laya_service.py`
   (the classifier)~~ **retired 2026-09-25** (PR 2 of this plan, D1 — see the
   `laya_service.py` inventory row above and the Mac-side steps below); the
   remaining conversion is `laya_controller.py`'s event-driven decision-queue
   design (`event_only` mode, now the live default path once
   `config/laya.json` enables it), which maps naturally onto ADR 113's
   "admission belongs to the akh" — this is scheduling logic that belongs
   inside akhomed's reactor, not a standalone controller process. The
   `experiments/.venv-comparison` dependency noted below is resolved by this
   pass's `configure_laya.py` (system `python3`).
3. **`janus_escalation.py` + `mergiraf_worker.py` → Janus delegation (53e
   grants) + skills.** The escalation-outbox pattern (durable, dedup on
   `incident_key`, cooldown-aware) is a good template for how delegated grants
   should behave; `mergiraf_worker.py`'s typed job schema is a natural skill
   input schema.
4. **`provider_failure.py` → akhomed's provider error taxonomy.** Its
   classification rules (429 vs. explicit quota exhaustion, auth failures need
   operator repair and don't expire on a timer, cooldown ladder 30m/1h/2h/4h)
   should become part of the engine's own provider-error handling rather than
   a shared Python import three scripts each read independently.
5. ~~**`t3_activity.py` obsolete after the cutover** — no separate "activity"
   provider transport is needed once T3 speaks ACP directly to akhomed.~~
   **Done 2026-09-25** (ADR 113 / 53g plan PR 1): `t3_activity.py` and
   `test_t3_activity.py` are deleted, `configure_t3.py` no longer renders the
   `roman-activity-<role>` providers or the `t3-activity` plist, and
   `anima-activity.py` (D6) is retired alongside it. See "Mac-side steps for
   PR 1" below for the operator's live-deployment cleanup.
6. **Render the five plists with no repo mechanism today**
   (`otlp-tunnel`, `runtime-minerva`, `runtime-vesta`, `laya-controller`,
   `laya`) — either extend `prepare.py`/`activate.py` or add small
   `configure_*.py` scripts matching the existing pattern, so these are
   reproducible from a clean clone like the others already are.
5. **`t3_activity.py` obsolete after the cutover** — no separate "activity"
   provider transport is needed once T3 speaks ACP directly to akhomed.
6. **Render the remaining plists with no repo mechanism today**
   (`otlp-tunnel`, `runtime-minerva`, `runtime-vesta`) — either extend
   `prepare.py`/`activate.py` or add small `configure_*.py` scripts matching
   the existing pattern, so these are reproducible from a clean clone like the
   others already are. `laya-controller` is now covered by `configure_laya.py`
   (this pass); `laya` has no plist to render — it's retired.
7. ~~**`daemon_credentials.py`'s new `core` role is not yet wired to
   `dev.roman-akhs.core.plist`.**~~ **Done in 53f prep (repository side):**
   `configure_core.py` now renders the core plist as
   `python3 bin/daemon_credentials.py core <root>`; the `core` role sets
   `AKH_REACTOR_ENABLE=1` and execs the provenance-named
   `bin/akhomed-main-<sha>` recorded in `config/core-release.json`. The live
   plist changes only when the operator runs [CUTOVER.md](CUTOVER.md).
8. **Sokar has no credential distinct from `ROMAN_ARCHIVE_TOKEN`.** Confirm
   whether that's intentional (shared token good enough for Sokar's current
   scope) or whether Sokar needs its own 1Password reference before 53i's
   `[[mcp.servers]]` wiring goes live.

## Mac-side steps for PR 1 (operator, after merge)

Repo-side, PR 1 deletes `t3_activity.py`/`test_t3_activity.py` and stops
`configure_t3.py` from rendering the `roman-activity-<role>` providers, the
`bin/t3_activity.py` copy and the `dev.roman-akhs.t3-activity` plist. Nothing
in this PR touches the live `~/Library/Application Support/roman-akhs`
deployment, launchd, or T3 — the operator does the following by hand, after
merging, on the Mac. None of it needs `sudo`: every LaunchAgent here runs in
`gui/501`, not as a system LaunchDaemon.

```sh
# 1. Stop and unregister the t3-activity LaunchAgent, then remove its plist.
launchctl bootout gui/501/dev.roman-akhs.t3-activity 2>/dev/null
rm -f ~/Library/LaunchAgents/dev.roman-akhs.t3-activity.plist

# 2. Stop and unregister the anima-activity LaunchAgent (D6; lives outside
#    roman-akhs, under ~/.config/anima-runner/roman-chat/).
launchctl bootout gui/501/dev.roman-akhs.anima-activity 2>/dev/null
rm -f ~/Library/LaunchAgents/dev.roman-akhs.anima-activity.plist

# 3. Remove the physical activity binaries a previous configure_t3.py run
#    installed (the ROOT below is the roman-akhs deployment root).
ROOT="$HOME/Library/Application Support/roman-akhs"
rm -f "$ROOT/bin/t3_activity.py" "$ROOT"/bin/roman-activity-*
```

4. **In T3**, delete the three `roman-activity-<role>` provider instances
   (Settings → Providers → remove `roman-activity-janus`,
   `roman-activity-minerva`, `roman-activity-vesta`). The three named
   `roman-janus`/`roman-minerva`/`roman-vesta` providers are untouched — they
   keep working exactly as before, rendered by the same, still-present
   `t3_entrypoint.py` path in `configure_t3.py`.
5. Re-running `configure_t3.py ROOT ~/.t3/userdata/settings.json` after these
   steps is safe and idempotent: it only touches the three named
   `roman-<role>` provider entries and no longer knows about the
   `roman-activity-*` ones, so it will not recreate them.


## PR 2 Mac-side steps (`laya_service.py` retirement, D1)

Not done by this repo pass — the architect runs these against the live
deployment at `~/Library/Application Support/roman-akhs` after this PR merges
to `origin/main` (deploy-from-main applies: build/copy only from a clean
checkout of `origin/main`).

1. **Boot out the classifier**, which has no successor process:
   `launchctl bootout gui/501/dev.roman-akhs.laya`.
2. **Render the controller's new plist** from a clean `origin/main` checkout:
   `python3 deploy/roman-mini/configure_laya.py "$HOME/Library/Application Support/roman-akhs"`.
   This copies the updated `laya_controller.py` into `bin/` and writes
   `launchd/dev.roman-akhs.laya-controller.plist` under the deployment root
   (it does not install anything under `~/Library/LaunchAgents` or touch the
   running daemon).
3. **Install the rendered plist**:
   `cp "$HOME/Library/Application Support/roman-akhs/launchd/dev.roman-akhs.laya-controller.plist" ~/Library/LaunchAgents/`.
4. **Reload the controller** — its `ProgramArguments[0]` changed (venv
   interpreter → `/usr/bin/python3`), so `launchctl kickstart` alone will not
   pick that up; boot it out and back in:
   `launchctl bootout gui/501/dev.roman-akhs.laya-controller` then
   `launchctl bootstrap gui/501 ~/Library/LaunchAgents/dev.roman-akhs.laya-controller.plist`.
   (If a later re-render leaves the plist byte-identical to what's already
   loaded, `launchctl kickstart -k gui/501/dev.roman-akhs.laya-controller` is
   enough.)
5. **Verify over 24 h:**
   - `state/laya-decisions.sqlite3`'s `decisions` table keeps accruing new
     rows at the same rate as before (`select count(*), max(created_at) from
     decisions;`) — the queue must not stall.
   - the `action` distribution in new rows matches the pre-retirement
     baseline: still dominated by `investigate` (with `product`/
     `infrastructure` coming from `host_action`'s structural rules exactly as
     before), since the model already contributed nothing but `investigate` in
     the live evidence behind D1.
   - `laya_status` (via the `observer` outbox / `tick()`) keeps reporting
     `ok: true` with no new `unknown`/`intervention_failed` pileup.
   - no process is listening on `127.0.0.1:18446` any more
     (`curl -sf http://127.0.0.1:18446/healthz` fails).
6. **Rollback** (if the controller misbehaves post-cutover): boot out
   `dev.roman-akhs.laya-controller`, restore the previous
   `dev.roman-akhs.laya-controller.plist` (venv interpreter) and the previous
   `bin/laya_controller.py` from the prior release, `launchctl bootstrap
   gui/501` it back, and `launchctl bootstrap gui/501` the `dev.roman-akhs.laya`
   plist again to bring the classifier back. `config/laya.json`'s
   `model_revision` was never changed by this PR, so no decision is re-decided
   by rolling back or rolling forward.


## PR 3 (landed): Grafana passthrough becomes a direct MCP server

Per the [step-2 plan](../../docs/ai/plans/2026-09-25-phase53g-roman-glue-conversion.md)'s
PR 3. Vesta reaches Grafana through its own `[[mcp.servers]] name = "grafana"`
entry (`http://127.0.0.1:18441/mcp`, no auth — the `grafana-tunnel` LaunchAgent
is already loopback-only) instead of the `vesta-ops` broker passthrough.
`vesta_ops.py` dropped the `grafana__*` tools and the `grafana_rpc` import;
`vesta_actions.py`'s `grafana_rpc` now serves only `configure_vesta_actions.py`'s
discovery step (`config/vesta-grafana-tools.json`, still the source of the new
server's `tools` allowlist). `full_akh.py` renders the entry for Vesta only and
rewrites the prose `Grafana MCP tools prefixed grafana__` to `prefixed
mcp__grafana__` (`roles/addenda/vesta.md` updated to match); rendering refuses
with a `RenderError` if `config/vesta-grafana-tools.json` is missing or empty,
rather than silently registering zero or unbounded tools.

Naming before/after: `mcp__vesta-ops__grafana__query_prometheus` (proxied) →
`mcp__grafana__query_prometheus` (direct).

### Mac-side steps (not run by this pass)

1. Re-render from a clean `origin/main` checkout so this PR is on the box:
   `python3 configure_core.py --root ~/Library/Application\ Support/roman-akhs
   --binaries <clean-checkout>/target/release --repo <clean-checkout>`. This
   rewrites `core-config/akh-medu/workspaces/vesta.toml` (adds `[[mcp.servers]]
   name = "grafana"`) and the reactor seed prompts; it does not start anything.
   `config/vesta-grafana-tools.json` must already exist from a prior
   `configure_vesta_actions.py` run — rendering fails closed with a clear
   message if it's missing.
2. Copy the updated `vesta_ops.py` into `bin/`: `cp deploy/roman-mini/vesta_ops.py
   ~/Library/Application\ Support/roman-akhs/bin/vesta_ops.py`.
3. Restart the core so it picks up the new workspace TOML. The core plist has
   `KeepAlive`, so `kill $(pgrep -f akhomed-main-)` is enough — launchd
   restarts it from the (already rewritten) config; no sudo needed.
4. Restart the `vesta-ops` broker so it drops the Grafana passthrough tools:
   `launchctl kickstart -k gui/501/dev.roman-akhs.vesta-ops`.
5. Verify: `GET /v1/akhs/vesta/tools` with the operator token should list
   `mcp__grafana__*` tools and no `mcp__vesta-ops__grafana__*` ones; or check
   `logs/core.out.log`/`logs/core.err.log` for the `akh::mcp` discovery line
   for Vesta showing a `grafana` server with the expected tool count (matching
   `config/vesta-grafana-tools.json`'s length). Confirm `vesta-ops`'s own
   `tools/list` (`curl -s -H "Authorization: Bearer $(cat
   secrets/vesta-ops.key)" -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
   http://127.0.0.1:18437/mcp`) no longer returns any `grafana__*` name.
6. **Rollback:** restore the previous `core-config/akh-medu/workspaces/vesta.toml`
   and `bin/vesta_ops.py` from the pre-render backup (`configure_core.py`
   preserves unrelated keys but does not itself keep a dated backup; take one
   before step 1, e.g. `cp -a core-config/akh-medu/workspaces
   core-config/akh-medu/workspaces.bak-$(date +%s)`), then repeat steps 3–4 to
   restart the core and kickstart `vesta-ops`. Grafana access degrades to
   "no Grafana tools" during either direction of the switch, never to a
   half-registered state — the workspace TOML and `vesta_ops.py` change
   together in the same restart.
