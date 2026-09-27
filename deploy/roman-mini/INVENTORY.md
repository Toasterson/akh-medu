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
| `janus_escalation.py` | Durable outbox broker: turns a `escalate_to_astra` / `dispatch_mergiraf` MCP tool call into a T3/Astra thread; the delivery survives T3 being offline. | in: MCP `tools/call` over loopback HTTP, `config/janus-escalation.json` (registered T3 project IDs), `secrets/janus-escalation.key`; out: T3 `/api/orchestration` thread/turn commands, `state/*.sqlite3` incident ledger. | `dev.roman-akhs.janus-escalation` | drifted → synced (see below); **converted (53g PR 7)** — replaced by the native `escalate` / `escalation_status` tools, the incident ledger and the `skills/astra-escalation` skillpack; kept until the verified Mac cutover ("PR 7 Mac-side steps" below), then deleted | Janus delegation (53e grants) + skill — **done** |
| `laya_controller.py` | **PR 6:** routing moved to data-driven `when` filters on akheventd routes (the `host_action` table, rendered by `full_akh.py`), the queue to the `roman-anima` source, the brief to the `laya-intervention` skillpack and `record_fixer_result` to a `roman-memory` 1.1.0 memory write (see "PR 6" below); retired at the PR 6 cutover, deleted after it is verified. Before PR 6: Durable event→classification→scoped-fixer controller: reads an observation queue (`state/laya-decisions.sqlite3`), classifies a blocker (product/execution/infrastructure/human/wait/investigate) and routes an intervention to the right role (`ROLE` map) or opens a mergiraf/Astra dispatch. Serves a small results API on `127.0.0.1:18447`. **`classify()` no longer calls the Laya model** (see "`laya_service.py` retired" below) — it returns `{'choice': 'investigate', 'classifier': 'retired', ...}` synchronously, and `CHOICES`/`digest`/`encoded` (formerly imported from `laya_service`) are now defined in this module. `config/laya.json`'s `model_revision` is still read unchanged (it participates in the decision id). Now installed with `configure_laya.py` on the system `python3` instead of the `experiments/` venv. | in: observations from `observer.py` (when `config/laya.json.event_only` is true), `secrets/laya-<role>.key`; out: `laya.<role>.intervention` outbox events, HTTP decisions API. | `dev.roman-akhs.laya-controller` (LaunchAgent) | in repo | Janus reactor + skill (plan's explicit step-2 item; PR 2 of that step retired the classifier first) |
| `laya_service.py` | **Retired 2026-09-25** (Phase 53g step-2 plan, PR 2, operator decision D1). Was a resident, loopback-only classifier (`127.0.0.1:18446`) that scored blocker text against the 6 `CHOICES` with a local model and validated the returned probability simplex. **Rationale:** live decision-log evidence showed every one of the 52 complete-input classifications ended in `investigate` with near-uniform per-choice scores (0.10–0.24 against a uniform 1/6); every real route to `product` or `infrastructure` came from `laya_controller.py`'s `host_action` structural rules (which match Anima's phase/parked_reason/session fields), never from the model's score. Deleted from the repo; its `dev.roman-akhs.laya` LaunchAgent is booted out on the Mac side (see Mac-side steps below), not replaced. | — | `dev.roman-akhs.laya` (LaunchAgent, booted out) | retired | n/a — folded into `laya_controller.py`'s structural router, no successor process |
| `mergiraf_worker.py` | **Not in repo before this pass.** Library (no server, no launchd entry) imported by `janus_escalation.py`; defines the `POLICY` prompt, JSON-schema `TOOLS` (`dispatch_mergiraf`, `mergiraf_status`) and `escalation()` builder for a bounded, typed merge-conflict-resolution job dispatched to an Astra/T3 worker via the same broker. | in: called in-process from `janus_escalation.Broker.submit_merge`; out: a T3 thread prompt + schema-validated job fields (work item, PR, head/base SHAs). | none (imported, not a service) | added; **converted (53g PR 7)** — replaced by the native `dispatch_merge_repair` tool and the `skills/mergiraf` skillpack (D4 in its procedures and in the `require_existing_branch` publication profile); kept until the verified Mac cutover, then deleted | Janus delegation (53e grants) + skill — **done** |
| `observer.py` | **PR 6:** the Anima scan and the watchdog moved into akheventd (`roman-anima` source, `watchdog`); retired at the PR 6 cutover, deleted after it is verified. **PR 5:** monitoring (health, Grafana, VictoriaMetrics) and the `monitoring_snapshot` MCP server moved to akheventd's `roman-alerts` monitor source + akhomed's native tool (see "PR 5" below); the watchdog reads `GET /v1/monitors/roman-alerts`. Before PR 5: polls Anima (rotating 8 projects/min), Grafana and VictoriaMetrics; maintains a deterministic watchdog signal, a durable outbox of `schedule.*`/`anima.*.changed`/`monitoring.changed` events, and (new) `laya.<role>.intervention` events when `event_only` mode is on; serves a read-only `monitoring_snapshot` MCP tool on `127.0.0.1:18436`. | in: Anima MCP, Grafana HTTPS, VictoriaMetrics via `127.0.0.1:18440` tunnel, `config/laya.json`; out: `state/observer.sqlite3` outbox/state, POSTs to akhsessd `127.0.0.1:18431`, board watchdog notices. | `dev.roman-akhs.observer` (LaunchAgent) | drifted → synced (see below) | event production → Rust `akheventd` source; monitoring snapshot → a Vesta tool (both named explicitly in the plan's step-2 list) |
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
| `janus-escalation` | `python3 janus_escalation.py <root>` | `PATH`, `HOME` | `configure_janus_escalation.py` (in repo). **Converted (53g PR 7):** booted out by the "PR 7 Mac-side steps" below; plist, script and renderer are deleted in the follow-up after verification |
| `laya-controller` | `/usr/bin/python3 laya_controller.py <root>` | `TOKENIZERS_PARALLELISM`, `USE_TF`, `HF_HUB_OFFLINE` | `configure_laya.py` (in repo, this pass). Was `<repo>/experiments/typed-decisions/.venv-comparison/bin/python laya_controller.py <root>` — the venv interpreter loaded the retired classifier's torch/transformers dependency, which the controller no longer imports, so this pass moves it to the system Python like every other roman-akhs LaunchAgent. |
| `laya` | *(retired 2026-09-25, see `laya_service.py` above)* | `TOKENIZERS_PARALLELISM`, `USE_TF`, `HF_HUB_OFFLINE` | n/a — no successor plist; the Mac-side steps below boot it out and don't replace it |
| `observer` | `python3 observer.py <root>` | `KUBECONFIG`, `HOME`, `PATH` | `activate.py` (in repo) |
| `sessions` | `python3 daemon_credentials.py sessions <root>` | `PATH`, `HOME`, `RUST_LOG` | `prepare.py` (in repo) |
| `t3-activity` | `python3 t3_activity.py --service` | `PATH`, `HOME` | **retired 2026-09-25** — `configure_t3.py` no longer renders it |
| `vesta-ops` | `python3 vesta_ops.py <root>` | `KUBECONFIG`, `HOME`, `PATH` | `configure_vesta.py` (in repo) |

## Follow-up work items (step 2, filed per the plan's explicit list)

Each of these needs its own work item before it lands (not done in this pass):

1. **`observer.py` event production → Rust `akheventd` source**, and its
   `monitoring_snapshot` MCP tool → a Vesta tool. *(Monitoring half done in
   PR 5: akheventd monitor source + native `monitoring_snapshot`; the Anima
   scan half is PR 6.)* Also decide whether
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
3. ~~**`janus_escalation.py` + `mergiraf_worker.py` → Janus delegation (53e
   grants) + skills.**~~ **Converted in 53g PR 7** (repository side): native
   `escalate` / `dispatch_merge_repair` / `escalation_status` tools
   (`src/agent/tools/escalation.rs`), the durable incident ledger
   (`src/agent/delegation/ledger.rs`), per-project akhsessd worker profiles,
   and the worker policy as skillpacks (`deploy/roman-mini/skills/`). The
   scripts stay in the repo until the Mac cutover below is verified, then a
   follow-up deletes `janus_escalation.py`, `mergiraf_worker.py`,
   `test_janus_escalation.py`, `configure_janus_escalation.py` and (with PR 5
   and PR 6, which drop the observer's use of it) `provider_failure.py`.
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


## PR 7 Mac-side steps (`janus_escalation` + `mergiraf_worker` → delegation, D3/D4)

Not done by the repository PR. The architect runs these against the live
deployment at `~/Library/Application Support/roman-akhs` (`$ROOT` below) after
PR 7 is merged to `origin/main`. deploy-from-main applies to every binary:
build only from a clean checkout of `origin/main` and quote the
`merge-base --is-ancestor` line in the deploy report. `$UID_` is `501`.

**What changes.** Janus loses `[[mcp.servers]] janus-escalation` and gains the
native `escalate`, `dispatch_merge_repair` and `escalation_status` tools. Each
admitted incident runs as an akhsessd session in the project's worker profile
(`janus-worker-<project>`) on the `codex-astra` runtime, carrying a
per-incident delegation grant (read-only cognitive tools plus `memory_write`),
briefed with `skills/astra-escalation` or `skills/mergiraf`, and reporting to
the Roman status thread. T3/Astra stays the rollback. The new ledger starts
empty: open incidents of the old broker are **not** migrated.

### 0. Preconditions

1. No escalation or mergiraf job of the old broker is still needed: list them
   with `sqlite3 "$ROOT/state/janus-escalation.sqlite3" 'select key, thread,
   created, delivered, last_error from incidents order by created'`, finish or
   record each (the T3 threads stay readable in T3), and note the keys so
   Janus does not treat a re-escalation under the same key as a duplicate of
   the old one.
2. A Codex CLI is installed for the `toasty` user and logged in (the
   `codex-astra` runtime profile below manages its own isolated state).

### 1. Build from main (services and core together)

Use CUTOVER.md step 1 unchanged (it builds `akhomed` and `akhsessd`,
`akhrtd`, `akhsbxd`, `akhexecd`, `akheventd`, `akh-acp`, `akh-trigger` from
one `origin/main` checkout and records `SHA`). PR 7 changes akhsessd (the
`GET /v1/session-requests/{id}` lookup, the `session_grants` runtime and
instructions override), akhsbxd/akhexecd (the `require_existing_branch`
publication profile) and akhomed (the tools). They must move together.

### 2. `config/runtime.json` (akhrtd): the Codex worker runtime

Add `runtimes.codex-astra` (kind `codex`, `execution_mode: "brokered"`,
`binary` = absolute path of the `codex` CLI, `working_dir` =
`$ROOT/workspaces`, an `isolation.directory` of its own such as
`$ROOT/runtime-isolation/codex-astra` with a `codex/config.toml` managed file
pinning `model = "gpt-6-astra"` and `model_reasoning_effort = "high"`,
`effort: "high"`, `timeout_secs: 14400`, and a `conformance_file`).
`scripts/session-platform/codex-native-wire-smoke.py` shows a working brokered
Codex profile shape. Install the conformance evidence like any other brokered
profile, then check it:

```sh
"$ROOT/bin/akh-runtime-profile" --config "$ROOT/config/runtime.json" --profile codex-astra verify
```

It must print `codex-astra: installed runtime conformance verified`; akhrtd
refuses a brokered profile without it (`conformance_required`).

### 3. `config/sessions.json` (akhsessd)

Do all of this in one edit after `cp -p "$ROOT/config/sessions.json"
"$ROOT/config/sessions.json.pre-pr7"` (and the same for runtime.json,
sandbox.json and executor.json). akhsessd validates the whole file on start
and refuses an invalid one, so a mistake shows up in step 5 as a failed start
with the reason in `logs/sessions.err.log`; restore the `.pre-pr7` copy then.

1. **Principal** for the core's delegation client (the key file is created
   0600 by `configure_core.py`, step 4):

   ```json
   {"id": "janus-delegation", "token_file": "<ROOT>/secrets/sessions-janus-delegation.key",
    "roles": ["client"], "akhs": ["janus"]}
   ```

2. **One worker workspace per registered project.** The registered projects
   are the keys of `config/janus-escalation.json` (today `akh-medu`,
   `eu-tender`, `anima`, `reddwarf`, `board`, `secrets-share`, `papercrab`,
   `Droneworld`, `chat`, `illumos`, `colonysim`). The profile name is
   `janus-worker-<slug>`, the slug being the project lower-cased with every
   run of other characters replaced by `-` (`Droneworld` →
   `janus-worker-droneworld`). For each, add the same entry under
   `workspaces` in **sessions.json, runtime.json, sandbox.json and
   executor.json**, and create the directory (`mkdir -m 700 -p
   "$ROOT/workspaces/janus-worker-<slug>"`):

   ```json
   "janus-worker-<slug>": {"revision": "roman-mini-v1", "site": "armini",
     "provisioner": {"kind": "existing", "executor": "executor",
                     "root": "<ROOT>/workspaces/janus-worker-<slug>"},
     "retain_on_release": true}
   ```

   Only projects that have this workspace **and** appear in
   `akhs.janus.workspace_profiles` are registered by `configure_core.py`; the
   others are listed as warnings.

3. **`akhs.janus.workspace_profiles`**: append every `janus-worker-<slug>`.

4. **Workspace tools for the workers.** Append workspace `ToolSpec`s to
   `akhs.janus.tools` for the operations `read`, `write`, `edit`, `mkdir`,
   `list`, `stat`, `search`, `exec_start`, `exec_poll`, `exec_stdin`,
   `exec_cancel` (destination `{"kind": "workspace", "operation": "<op>"}`,
   object input schemas; see the smoke script above for the shape). Then add
   `akhs.janus.session_grants.janus` = `{"tools": [<every tool name Janus had
   before this step>]}` so Janus's own profile does not widen.

5. **`akhs.janus.session_grants.janus-worker-<slug>`** for each project:

   ```json
   {"tools": [<the workspace tools>, "kg_query", "memory_recall", "memory_write",
              "reason", "similarity_search", "entity_schema", "entity_types",
              "entity_type_filter",
              "anima__list_projects", "anima__resolve_project_by_git_remote",
              "anima__list_project_remotes", "anima__get_project_ci_status",
              "anima__list_work_items", "anima__get_work_item_by_id",
              "anima__get_work_item_detail", "anima__get_work_item", "anima__get_thread",
              "anima__post_message", "anima__list_sessions",
              "anima__get_session_transcript", "anima__get_session_retirement",
              "anima__get_session_publications", "anima__list_changes",
              "anima__get_change", "anima__get_wi_loop_state", "anima__list_artifacts"],
    "runtime": {"runtime": "runtime", "profile": "codex-astra", "revision": "roman-mini-v1",
                "kind": "codex", "model": "gpt-6-astra", "execution_mode": "brokered"},
    "instructions": "You are a delegated worker session. Your task, its procedure and its limits arrive as your first input; follow them and nothing else."}
   ```

   Deliberately **absent** (D4, and "Janus merges only through Anima's gated
   path"): `anima__approve_and_merge`, `anima__force_clear_ai_gate`,
   `anima__force_unclear_ai_gate`, `anima__set_work_item_phase`,
   `anima__promote_work_item`, `anima__unpark_work_item`,
   `anima__cancel_session`, `anima__hold_work_item`, `anima__release_hold`,
   `anima__repin_executor`, `anima__run_scheduler_tick`,
   `anima__reset_in_flight_time`, `anima__rearm_exhausted_ci`,
   `anima__rearm_exhausted_ai_review`, `anima__resync_pr_state`,
   `anima__create_work_item`, `anima__update_work_item` and `kg_mutate`. For
   the workers to post their report to the Roman status thread themselves
   (D3), also grant the `board__*` tool Janus uses to post to a thread;
   without it the worker's `memory_write` report and its final answer (read
   through `escalation_status`) are the report.

6. Leave `endpoints.janus-escalation` in place until step 7 (rollback needs
   it), and do not touch `automations`.

### 4. Render the core

```sh
cp -p "$ROOT/core-config/akh-medu/workspaces/janus.toml" \
      "$ROOT/core-config/akh-medu/workspaces/janus.toml.pre-pr7"
cd "$REPO"   # the clean origin/main checkout from step 1
/opt/homebrew/bin/python3 deploy/roman-mini/configure_core.py --root "$ROOT" \
  --binaries "$REPO/target/release" --repo "$REPO"
```

This installs `bin/akhomed-main-$SHA`, copies `skills/astra-escalation` and
`skills/mergiraf` to `$ROOT/skills/`, creates
`secrets/sessions-janus-delegation.key` (0600) and renders Janus's workspace
TOML **without** `[[mcp.servers]] janus-escalation` and **with**:

```toml
[delegation]
enabled = false            # the generic `delegate` tool stays off

[delegation.native]
enabled = true
service = "akhsessd"
url = "http://127.0.0.1:18431"
token_file = "<ROOT>/secrets/sessions-janus-delegation.key"
workspace_profile = "janus"
core_principal = "sessions"
max_new_per_day = 4

[delegation.native.profiles]
akh-medu = { workspace_profile = "janus-worker-akh-medu", runtime_profile = "codex-astra" }
# ... one line per registered project with a worker profile

[delegation.native.escalation]
enabled = true
report_thread = "01a0ab49-9cc4-7012-8785-1eada26fbd46"
escalation_skill = "<ROOT>/skills/astra-escalation"
merge_repair_skill = "<ROOT>/skills/mergiraf"
```

Read every `warning:` line: a project listed there has no worker profile and
cannot be escalated for. Janus's prompt now names `escalate`,
`dispatch_merge_repair` and `escalation_status` (roles/addenda/janus.md).

### 5. Restart, readers before writers (CUTOVER "Wire compatibility")

```sh
sudo launchctl bootout system/dev.roman-akhs.core
launchctl kickstart -k gui/$UID_/dev.roman-akhs.sessions       # akhsessd (reader)
sudo launchctl kickstart -k system/dev.roman-akhs.runtime        # akhrtd: codex-astra
sudo launchctl kickstart -k system/dev.roman-akhs.sandbox        # akhsbxd (reader)
sudo cp "$ROOT/launchd/dev.roman-akhs.core.plist" /Library/LaunchDaemons/
sudo launchctl bootstrap system /Library/LaunchDaemons/dev.roman-akhs.core.plist
sudo launchctl kickstart -k system/dev.roman-akhs.executor       # akhexecd last (writer)
```

Each service must run the `-main-$SHA` binary from step 1 (check `ps` and the
`*.err.log` start lines). `logs/core.err.log` must show `escalation tools
registered` for janus with the expected `projects`; an `escalation tools not
registered` line carries the diagnostic to fix.

### 6. Boot out the broker

```sh
launchctl bootout gui/$UID_/dev.roman-akhs.janus-escalation
curl -sf http://127.0.0.1:18438/healthz && echo "broker still listening"   # must print nothing
```

Keep `~/Library/LaunchAgents/dev.roman-akhs.janus-escalation.plist`,
`bin/janus_escalation.py`, `bin/mergiraf_worker.py`,
`config/janus-escalation.json` and `state/janus-escalation.sqlite3` until the
verification below has held for 48 h (rollback).

### 7. Verify

For the dry runs, temporarily add `"escalate"` and `"escalation_status"` to
the `operator` principal's `tools.tools` in `config/core.json` and restart the
core; remove them (and restart) afterwards. Call
`POST http://127.0.0.1:18430/v1/tools/call` with the operator key and
`{"request_id": "<uuid>", "session_id": "pr7-verify", "akh": "janus", "tool":
"escalate", "arguments": {...}}`.

1. **One incident, one session.** Escalate a harmless dry incident
   (`incident_key` `pr7-dry-1`, project `akh-medu`, `requested_action`
   "reply with an akh-result and change nothing"). The receipt is `running`
   with a `session_id`; `GET /v1/sessions?compact=true` on akhsessd shows
   exactly one new session, on runtime profile `codex-astra` (`GET
   /v1/sessions/<id>` → `runtime.profile`).
2. **Repeat → duplicate.** The same call again returns `duplicate: true`, the
   same `delegation_id` and `session_id`, and akhsessd shows no new session.
3. **Fifth new incident in a UTC day refused.** Escalate `pr7-dry-2`,
   `pr7-dry-3`, `pr7-dry-4` (with `pr7-dry-1` that is four), then `pr7-dry-5`:
   refused with `akh::delegation::daily_limit`. (The limit is per UTC day, so
   run all five between 00:00 and 23:59 UTC.)
4. **Status.** `escalation_status` for `pr7-dry-1` shows `running` then,
   after the worker finishes, `succeeded` with the worker's `akh-result`, and
   the session is archived.
5. **Delegated provenance.** The worker's `memory_write` records hang off the
   entity `delegate:native:akhsessd/janus-worker-akh-medu`; its provenance is
   `delegated_call` naming the incident's `session_id` (operator `kg_query`
   for that entity, then `_akh/explain` on one of the symbols).
6. **Restart reconciliation.** Escalate `pr7-dry-6` on the next UTC day,
   restart the core while it runs, and check `logs/core.err.log` for
   `incident ledger reconciled` with no second akhsessd session for it.

### Rollback

1. Restore `core-config/akh-medu/workspaces/janus.toml` from
   `janus.toml.pre-pr7` (step 4), which brings back `[[mcp.servers]]
   janus-escalation` and drops `[delegation]`. The PR 7 core reads the old
   TOML fine; roll the binary back too only if the core itself misbehaves
   (CUTOVER "Rollback", readers after writers).
2. `launchctl bootstrap gui/$UID_ ~/Library/LaunchAgents/dev.roman-akhs.janus-escalation.plist`.
3. Restart the core (`sudo launchctl kickstart -k system/dev.roman-akhs.core`).
4. The sessions.json additions are inert without the core's tools and can
   stay; remove them only together with the akhsessd binary rollback. Worker
   sessions already created stay visible in akhsessd and are archived by its
   sweeps; revoke any still-open grant with `_akh/delegation/grant/revoke`.

### D4 enforcement status

- **Enforced:** workers never hold `anima__approve_and_merge` or any gate or
  lifecycle override (step 3.5), so they cannot merge through Anima; the
  controller-owned publication path refuses to create a branch and pushes
  without any force flag when its `git_publication` profile sets
  `require_existing_branch: true` (akhsessd/akhexecd, tested), and every
  profile already refuses a non-fast-forward result. The merge-repair
  profile, in the executor config next to a pinned `git_snapshot`, per
  repository (the prefixes are the repository's PR source-branch namespaces;
  `refs/heads/` alone is refused):

  ```json
  "git_publication": [{"repository": "https://code.aopc.cloud/<owner>/<repo>.git",
                       "branch_prefixes": ["refs/heads/anima/", "refs/heads/feat/", "refs/heads/fix/"],
                       "config_file": "<absolute private git config with the push credential>",
                       "require_existing_branch": true}]
  ```
- **Gap (Mac):** the Mac executor has no `git_snapshot`/`git_publication`
  configured, and the worker workspaces use the `existing` provisioner on the
  host executor, whose `exec_start` commands run as `toasty` with that user's
  Git credentials. A worker could therefore run `git push --force` or push to
  another branch itself; only the `mergiraf` procedures forbid it. Close it by
  (a) enabling server-side branch protection that disables force pushes on
  the repositories' PR branches, and (b) moving merge-repair publication to
  the controller-owned path with a `require_existing_branch` profile once the
  worker workspaces run in a credential-free sandbox (PR 8's Kubernetes
  sandbox work). Until then D4 on the Mac is policy plus tool narrowing, not
  a hard boundary.
- **Gap (provider taxonomy):** akhrtd reports a failed Codex turn as fault
  `unknown_outcome` with prose, which is never parsed, so a worker's quota or
  credential failure classifies as `unknown` unless its fault code is listed
  in Janus's `[provider]` lexicon. A follow-up should surface Codex's
  structured error code as the fault code in `akh-runtime`.


## PR 5: observer monitoring becomes an akheventd monitor source (D2)

Per the [step-2 plan](../../docs/ai/plans/2026-09-25-phase53g-roman-glue-conversion.md)'s
PR 5 (Anima WI `01a0d569-1cf1` item 5).

**What moved.**

- **akheventd:** the `roman-alerts` monitor source (`Config.monitors`,
  `crates/akh-session-services/src/monitor.rs`) replaces `observer.py`'s
  `monitoring()`. It polls every 60 s:
  - the six health endpoints the observer probed (core 18430, sessions 18431,
    runtime 18432, runtime-minerva 18442, runtime-vesta 18443, events 18435),
    as `anonymous` endpoints that receive no token;
  - Grafana's Alertmanager feed (`https://grafana.lab.home/api/alertmanager/grafana/api/v2/alerts`)
    with a Viewer service-account token and the exported lab CA (D2);
  - VictoriaMetrics `ALERTS{alertstate="firing"}` through the `metrics-tunnel`
    (`127.0.0.1:18440`, anonymous).
- **Signature and events:** the signature is structural. It covers health
  up/down/unknown, alert fingerprint, labels and `state`/`silencedBy`/`inhibitedBy`,
  series label sets and feed availability; annotations and latency are
  excluded. It is hashed with blake3. The first poll is a silent baseline.
  After that, each change ingests one `monitoring.changed` event with data
  `{previous, current, previous_signature, signature, transition, observed_at_utc}`
  into the existing `vesta-monitoring.changed` route, which still has
  `coalesce_pending: true`. The Vesta reactor seed is unchanged.
- **Unknown is not zero:** a feed that cannot be read is `unknown`, never zero
  alerts.
- **Monitor reads:** `GET /v1/monitors/roman-alerts` serves the latest reading.
  The new `monitoring-reader` principal (role `operator`, monitor reads only)
  and the observer principal (`event`, granted `roman-alerts`) may read it.
- **akhomed:** a native read-only `monitoring_snapshot` tool replaces
  `mcp__observer__monitoring_snapshot`, and `[[mcp.servers]] observer` is gone
  from all three workspaces. Each workspace gets a `[monitoring]` table
  (`endpoint`, `source`, `token_file = secrets/monitoring-reader.key`,
  `observer_status_file = state/observer-status.json`). The tool returns the
  old top-level keys: `monitoring{local_services, grafana, victoriametrics}`,
  `monitoring_at`, the observer status keys and `read_at(_utc)`, plus
  `observer_status` and `monitor`. `local_services` values are now
  `up`/`down`/`unknown` instead of `up`/`unavailable`. The operator core
  principal may call the tool, so you can verify it.
- **`observer.py`:** `monitoring()`, `monitoring_signature` and the MCP server
  on `127.0.0.1:18436` are deleted.
  - The watchdog reads `GET /v1/monitors/roman-alerts`. It adds
    `monitor_available`, which is false when akheventd is unreachable or the
    reading is more than 5 minutes old, and that counts as degraded.
  - Each tick writes `state/observer-status.json` (mode 0600) for the tool.
  - The Anima scan, laya queue and watchdog post are unchanged. PR 6 retires
    them.
- **Backpressure moved:** monitoring wakes no longer wait for the observer's
  `role_ready('vesta')`. akhomed's reactor coalescing and the PR 4
  `ProviderGate` now bound them.

### Operator action for D2 (do this first; not in this repo)

akheventd reads every endpoint CA file and every principal token file at
start. If `secrets/lab-ca.pem` is missing, akheventd does not start at all,
and **all** Roman event routing goes down, not just monitoring.
`retarget_events.py` refuses to install the new config until both files
exist, and `configure_core.py` prints a warning naming the missing ones.

Set `ROOT="$HOME/Library/Application Support/roman-akhs"`.

1. **Export the lab CA.** It is the `pki-ca-secret` in the `default`
   namespace, the same CA the observer used to load on every start.

   ```sh
   umask 077
   kubectl -n default get secret pki-ca-secret -o jsonpath='{.data.tls\.crt}' \
     | base64 -d > "$ROOT/secrets/lab-ca.pem"
   chmod 600 "$ROOT/secrets/lab-ca.pem"
   openssl x509 -in "$ROOT/secrets/lab-ca.pem" -noout -subject -enddate
   ```

2. **Create a Grafana service account with the Viewer role, plus a token.**
   Use the Grafana UI at <https://grafana.lab.home>:
   1. Go to *Administration → Users and access → Service accounts → Add
      service account*. Name it `roman-akheventd`, give it role **Viewer**,
      and click *Create*.
   2. Click *Add service account token* and name the token `akheventd`. Pick
      an expiry you will track: when the token expires, the alert feed turns
      `unknown` (reason `unauthorized`) and the watchdog posts `blocked`.
      Nothing fails silently.
   3. Copy the token once and write it without echoing it:

      ```sh
      umask 077
      read -rs GRAFANA_SA_TOKEN   # paste, then Enter
      printf '%s' "$GRAFANA_SA_TOKEN" > "$ROOT/secrets/grafana-sa.token"; unset GRAFANA_SA_TOKEN
      chmod 600 "$ROOT/secrets/grafana-sa.token"
      ```

   You can use the API instead of the UI, one time, with the admin secret.
   Nothing stores the admin credential:

   ```sh
   ADMIN="$(kubectl -n observability get secret grafana-admin -o jsonpath='{.data.admin-user}' | base64 -d):$(kubectl -n observability get secret grafana-admin -o jsonpath='{.data.admin-password}' | base64 -d)"
   SA=$(curl -sf --cacert "$ROOT/secrets/lab-ca.pem" -u "$ADMIN" -H 'Content-Type: application/json' \
     -d '{"name":"roman-akheventd","role":"Viewer"}' https://grafana.lab.home/api/serviceaccounts \
     | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
   (umask 077; curl -sf --cacert "$ROOT/secrets/lab-ca.pem" -u "$ADMIN" -H 'Content-Type: application/json' \
     -d '{"name":"akheventd"}' "https://grafana.lab.home/api/serviceaccounts/$SA/tokens" \
     | python3 -c 'import json,sys;print(json.load(sys.stdin)["key"],end="")' > "$ROOT/secrets/grafana-sa.token")
   unset ADMIN SA
   ```

   Also check that Grafana keeps its database across pod restarts. A service
   account in an ephemeral database disappears with the pod, and the feed then
   reads `unauthorized`.

3. **Check both files together.** This prints only an alert count:

   ```sh
   curl -sf --cacert "$ROOT/secrets/lab-ca.pem" \
     -H "Authorization: Bearer $(cat "$ROOT/secrets/grafana-sa.token")" \
     https://grafana.lab.home/api/alertmanager/grafana/api/v2/alerts \
     | python3 -c 'import json,sys;print(len(json.load(sys.stdin)),"alerts")'
   ```

### Mac-side steps (not run by this pass)

Deploy from `origin/main` only (`.claude/rules/deploy-from-main.md`). akheventd
and the core must both come from the PR 5 merge commit or later.

1. **Build from a clean checkout of origin/main.** Quote both provenance lines
   in the deploy report.

   ```sh
   cd "$REPO" && git fetch origin && git checkout --detach origin/main
   test -z "$(git status --porcelain)" && echo clean
   git merge-base --is-ancestor HEAD origin/main && echo "HEAD $(git rev-parse --short=12 HEAD) is on origin/main"
   SHA=$(git rev-parse --short=12 HEAD); export CARGO_TARGET_DIR="$REPO/target"
   cargo build --release --locked --features server --bin akhomed
   cargo build --release --locked -p akh-session-services --features events --bin akheventd
   ```

   Only akheventd changes among the session services. `Endpoint.anonymous` is
   optional and omitted when false, so akhsessd, akhrtd, akhsbxd and akhexecd
   stay as they are.
2. **Back up** everything this PR replaces:

   ```sh
   BK="$ROOT/evidence/pr5-$(date +%s)"; mkdir -p "$BK"
   cp -p "$ROOT/config/events.json" "$ROOT/config/core.json" "$ROOT/bin/observer.py" "$ROOT/bin/akheventd" "$BK/"
   cp -a "$ROOT/core-config/akh-medu/workspaces" "$BK/workspaces"
   ```

   `retarget_events.py --rollback` restores the **pre-Phase-53** file, not this
   one, so roll back from `$BK`.
3. **Do the D2 operator action** above if you have not already.
4. **Render:** `python3 deploy/roman-mini/configure_core.py --root "$ROOT"
   --binaries "$CARGO_TARGET_DIR/release" --repo "$REPO"`. This:
   - installs `bin/akhomed-main-$SHA`;
   - rewrites the three workspace TOMLs: no `observer` server, a new
     `[monitoring]` table, and addenda that name the native tool;
   - rewrites `core.json`, adding `monitoring_snapshot` to the operator
     principal;
   - writes `config/events.akh-target.json` with the monitor, its endpoints
     and the `monitoring-reader` principal;
   - creates `secrets/monitoring-reader.key` (mode 0600).

   There must be no "akheventd would not start" warning.
5. **Install the events config:** `python3 deploy/roman-mini/retarget_events.py
   "$ROOT"`. It refuses if a CA or token file is missing.
6. **Install akheventd.** It is a system LaunchDaemon, so this needs sudo.

   ```sh
   git -C "$REPO" merge-base --is-ancestor "$SHA" origin/main && echo "akheventd $SHA on origin/main"
   sudo launchctl bootout system/dev.roman-akhs.events
   install -m 755 "$CARGO_TARGET_DIR/release/akheventd" "$ROOT/bin/akheventd"
   shasum -a 256 "$ROOT/bin/akheventd" > "$ROOT/evidence/akheventd-main-$SHA.sha256"
   sudo launchctl bootstrap system /Library/LaunchDaemons/dev.roman-akhs.events.plist
   ```

7. **Restart the core** so it loads the new workspaces and the new binary. The
   core's `KeepAlive` brings it back, so `kill $(pgrep -f akhomed-main-)` is
   enough.
8. **Kickstart the observer.** It is a LaunchAgent, so no sudo:
   `cp -p deploy/roman-mini/observer.py "$ROOT/bin/observer.py" &&
   launchctl kickstart -k gui/501/dev.roman-akhs.observer`.
9. **Verify:**
   - **Monitor reading:** run
     `curl -s -H "Authorization: Bearer $(cat "$ROOT/secrets/monitoring-reader.key")" http://127.0.0.1:18435/v1/monitors/roman-alerts | python3 -m json.tool | head -40`.
     `feeds.alertmanager` and `feeds.prometheus` should be `available`, the
     health states should match the observer's last `local_services` (in the
     2026-09-26 smoke test the three `runtime*` probes were already down in
     both), and `transitions` should be 0 (the baseline).
     `logs/events.err.log` should show no `monitor poll deferred` line.
   - **One delivery per change:** create a Grafana silence for one firing
     alert (*Alerting → Silences*, 10 minutes). If nothing is firing, run
     `sudo launchctl bootout system/dev.roman-akhs.metrics-tunnel` instead,
     which turns the Prometheus feed `unknown`. Within about 60 s:
     - `transitions` becomes 1;
     - `GET /v1/deliveries` (operator token) shows exactly one new delivery on
       route `vesta-monitoring.changed`;
     - its `event_admission.schedules_woken` contains
       `roman-vesta-monitoring.changed`.

     Then expire the silence or run `sudo launchctl bootstrap system
     /Library/LaunchDaemons/dev.roman-akhs.metrics-tunnel.plist`. That yields
     the next transition.
   - **Tool shape:** call the tool through the core API:

     ```sh
     curl -s -H "Authorization: Bearer $(cat "$ROOT/secrets/operator.key")" -H 'Content-Type: application/json' \
       -d '{"request_id":"'"$(uuidgen)"'","session_id":"pr5-verify","akh":"vesta","tool":"monitoring_snapshot","arguments":{}}' \
       http://127.0.0.1:18430/v1/tools/call
     ```

     The result must carry `monitoring.{local_services,grafana,victoriametrics}`,
     `monitoring_at`, `coverage`, `anima_status`, `provider_availability`,
     `read_at` and `read_at_utc`, and `observer_status.available` must be
     `true`. `logs/core.out.log` shows `monitoring_snapshot tool registered`
     for all three akhs and no `observer` MCP discovery line.
   - **Watchdog:** it posts once to the status thread, because the new
     `monitor_available` key changes the signal. `state/observer.sqlite3`
     shows `board_watchdog_status.ok = true`. Nothing listens on
     `127.0.0.1:18436` any more.
10. **Rollback.** A partial rollback is not supported: the old core expects the
    observer MCP server, and the new observer no longer serves it.
    1. Restore `$BK/events.json` to `config/events.json`, `$BK/core.json` to
       `config/core.json` and `$BK/workspaces` to
       `core-config/akh-medu/workspaces`.
    2. Point `config/core-release.json` back at the previous `akhomed-main-*`
       binary.
    3. Restore `$BK/akheventd` and `$BK/observer.py`.
    4. Restart the services: `sudo launchctl bootout` and then `bootstrap` for
       `system/dev.roman-akhs.events`, `kill $(pgrep -f akhomed-main-)`, and
       `launchctl kickstart -k gui/501/dev.roman-akhs.observer`.

    The observer then probes Grafana with kubectl again. The D2 files can stay;
    nothing else reads them.

## PR 6: the Anima scan and `laya_controller` become an akheventd source, reactors and a skill

Per the [step-2 plan](../../docs/ai/plans/2026-09-25-phase53g-roman-glue-conversion.md)'s
PR 6 (Anima WI `01a0d569-1cf1` item 5). The rollout has two stages: a
**shadow** period of at least 48 h, then the **cutover**.

**What moved.**

- **akheventd `roman-anima` source** (`Config.anima`,
  `crates/akh-session-services/src/anima_source.rs`; wire types in
  `crates/akh-session-types/src/anima.rs`). It replaces `observer.py`'s
  `scan_anima` and `laya_controller.py`'s queue and hydration.
  - Every 60 s it visits 8 of the `status == active` projects (round robin;
    the project list is refreshed every 900 s). For each project it lists work
    items with cursor pagination (at most 32 pages; a repeated cursor or a
    longer listing fails the pass) and lists plans.
  - Each entity keeps one projection: the observer's `stable_item` fields.
    Parked items are hydrated from their detail, forensics and sessions, and
    get three extra fields:
    - `park_episode`: the newest forensic `park` event, or else the
      `parked_reason` and `phase_entered_at` values;
    - `live_sessions`: the number of `provisioning`, `running` or `blocked`
      sessions;
    - `latest_thread_message_id`: the newest message with
      `sender_type == human`.

    Hydration is repeated when the listing row changes, or after 300 s. At
    most 16 items are hydrated per poll.
  - The first complete pass of a project is a silent baseline. After that,
    each projection change is one `anima.work_item.changed` or
    `anima.plan.changed` event. Its ID is
    `<entity>@blake3(projection, revision)`, and its data is `AnimaChange`
    (`entity`, `previous`, `current`, `park_episode`, `emission`,
    `max_emissions`).
  - **Emission cap:** each (entity, park episode) can emit at most 2 events,
    which is laya's intervention budget. Later changes in the same episode are
    suppressed. The cap state commits in the same store transaction as the
    event and survives restarts. Capped entities appear in
    `GET /v1/anima/roman-anima` (`attention`) and in the watchdog notice.
  - **Shadow mode** (`shadow = true`, the default rendering): the source
    advances projections and caps exactly as in live mode, but it records each
    would-be event, together with the routes that would have delivered it, as
    a `ShadowRecord` instead of ingesting it. The newest 10 000 records are
    kept. They are served at
    `GET /v1/anima/roman-anima/shadow?since_ms=&limit=`, to the
    `monitoring-reader` (operator) principal and to an event principal granted
    `roman-anima`.
- **Routing is data** (`EventRoute.when`,
  `crates/akh-session-services/src/route_filter.rs`). A route can carry a
  JSON-pointer predicate over the event data. The operators are `eq`, `in`,
  `prefix`, `exists`, `not`, `all` and `any`; they are validated with located
  miette diagnostics. A route without `when` behaves exactly as before.
  `full_akh.py` renders `laya_controller.host_action` as filters (the table
  comment above `anima_route_filters`):

  | `host_action` rule | Route |
  |---|---|
  | a plan changed | `minerva-anima.plan.changed` (no filter) |
  | phase `done`/`cancelled`/`draft`, or any phase except `parked` | no route (every work-item route requires `parked`) |
  | `parked_reason` `operator-hold` / `human-parked` | no route (excluded from Janus) |
  | `parked_reason` starting with `gate-needs-human` | no route (excluded from Janus) |
  | `ci-infra-failing`, `ci-toolchain-broken`, `lane-circuit-breaker*` | `vesta-anima.work_item.changed` (ignores live sessions, like `host_action`) |
  | `live_sessions > 0` | no route (Minerva and Janus need `live_sessions == 0` or no count) |
  | `too-big` | `minerva-anima.work_item.changed` |
  | any other parked item | `janus-anima.work_item.changed` |

  Parity: `test_anima_routing.py` runs `host_action` and the filters over
  fixture snapshots that cover every branch. A replay over the live
  `laya-decisions.sqlite3` on 2026-09-26 (read-only) matched all 201
  recorded decisions: 120 to nobody, 63 to Janus, 16 to Minerva and 2 to
  Vesta. The Rust and Python evaluators share
  `fixtures/route_filter_cases.json`.
- **Watchdog** (`Config.watchdog`, `crates/akh-session-services/src/watchdog.rs`).
  It replaces `observer.py`'s `notify_watchdog`. Every 60 s it builds a
  signal from:
  - monitor availability: a reading older than 300 s counts as unavailable;
    health states; and feed availability from PR 5;
  - Anima source availability, mode and capped episodes;
  - routes whose latest settled delivery was refused (anything but
    `superseded`), and routes with a delivery still retrying after 3 attempts.

  On a change it posts one `coordination/status@v1` notice (`blocked` when
  degraded, otherwise `complete`) to the status thread `01a0ab49…`. It posts
  as the observer's watchdog identity: `secrets/board-watchdog.json` becomes
  `ROMAN_BOARD_WATCHDOG` in `daemon_credentials.py events`. A failed post is
  retried with the same idempotency key. It is rendered only at the cutover,
  because the observer keeps its own watchdog during the shadow period.

  Not carried over: the observer's `activity_available`,
  `provider_availability` and turn-level `workflow_failure` keys. PR 4's
  `ProviderGate` in akhomed now owns provider availability. akheventd sees
  deliveries, not turns.
- **Reactors and skill.** `full_akh.py` renders the reactor seeds
  `roman-<role>-anima.work_item.changed` for Janus, Minerva and Vesta, plus
  `roman-minerva-anima.plan.changed`.
  - **Wake prompt:** the role's wake prompt, then the `focus`, the role's
    Laya `scope` and the `instruction` from
    `deploy/roman-mini/skills/laya-intervention/prompts.json`. The prompt text
    lives in the skillpack, not in Rust.
  - **Exit condition and report:** each seed carries the skillpack's
    `exit_condition` and `report_thread` = the status thread.
  - **The skillpack** `laya-intervention` has `skill.json`, `procedures.md`
    (the procedure plus the routing table), `actions.json` (actions with
    preconditions and effects) and `triples.json` (the outcome vocabulary).
    `configure_core.py` copies it into each Roman's engine skills directory
    `core-data/akh-medu/workspaces/<r>/kg/skills/laya-intervention/`.
  - **`record_fixer_result` becomes a memory write:** `memory_write` on the
    event's `entity`, with `intervention_outcome` (`fix_applied`,
    `human_required`, `waiting` or `blocked`) and `intervention_next_event`
    (`human_reply`, `dependency_changed`, `lane_recovered`, `ci_completed`,
    `review_completed`, `work_item_changed` or `none`).
    `schemas/roman-memory.json` 1.1.0 declares both as single-valued and
    temporal.
  - **Wait guard:** the controller's wait guard is now step 2 of the
    procedure (recall the prior outcome, and record a quiet check while the
    awaited event has not happened), together with the human-only
    `latest_thread_message_id`.
- **Addenda:** `roles/addenda/common.md` now has a `{mcp_servers}` placeholder
  that is filled with the MCP servers actually rendered for each role.

**Behaviour differences, on purpose.**

- The per-episode cap replaces laya's `no_verified_progress` and `stale`
  states.
- Parked items emit again only when their projection changes: a listed
  field, a new park episode, a live-session count or a human reply. Laya
  re-decided on any change to the full dossier (CI runs, PR state, akh
  thread posts).
- Plans have no cap. Laya's budget key made each plan's budget a lifetime 2.
- The batched `anima.janus.changed` / `anima.minerva.changed` events were
  already off in the live deployment (`config/laya.json` `event_only`). They
  have no producer after the cutover.

### Mac-side steps (not run by this pass)

Deploy from `origin/main` only (`.claude/rules/deploy-from-main.md`). PR 5 must
already be deployed. Set `ROOT="$HOME/Library/Application Support/roman-akhs"`
and `REPO=<clean origin/main checkout>`.

#### (a) Deploy akheventd and the core from main; enable shadow mode

1. **Build** from a clean checkout of origin/main. Quote both provenance lines
   in the deploy report.

   ```sh
   cd "$REPO" && git fetch origin && git checkout --detach origin/main
   test -z "$(git status --porcelain)" && echo clean
   git merge-base --is-ancestor HEAD origin/main && echo "HEAD $(git rev-parse --short=12 HEAD) is on origin/main"
   SHA=$(git rev-parse --short=12 HEAD); export CARGO_TARGET_DIR="$REPO/target"
   cargo build --release --locked --features server --bin akhomed --bin akh
   cargo build --release --locked -p akh-session-services --features events --bin akheventd
   ```

   The `akh` CLI is the `akhc` of step (c). Only akheventd changes among the
   session services (`EventRoute.when`,
   `Config.anima` and `Config.watchdog` are optional).
2. **Back up** everything this PR touches:

   ```sh
   BK="$ROOT/evidence/pr6-$(date +%s)"; mkdir -p "$BK"
   cp -p "$ROOT/config/events.json" "$ROOT/config/core.json" "$ROOT/config/core-release.json" \
     "$ROOT/bin/akheventd" "$ROOT/bin/daemon_credentials.py" "$BK/"
   cp -a "$ROOT/config/reactor-seeds" "$BK/reactor-seeds"
   cp -a "$ROOT/core-config/akh-medu/workspaces" "$BK/workspaces"
   ```

3. **Check the credentials** the events role now reads. Neither stops
   akheventd if it is missing: the source reports itself unavailable, and the
   watchdog retries.
   - `secrets/anima-token` (mode 0600) already exists for the core
     (53f). akheventd is a LaunchDaemon and cannot reach the keychain, so
     the file is required.
   - `secrets/board-watchdog.json` is the observer's watchdog identity.

   Both are listed without printing them:

   ```sh
   ls -l "$ROOT/secrets/anima-token" "$ROOT/secrets/board-watchdog.json"
   ```

4. **Render in shadow mode** (the default):

   ```sh
   python3 "$REPO/deploy/roman-mini/configure_core.py" --root "$ROOT" \
     --binaries "$CARGO_TARGET_DIR/release" --repo "$REPO"
   ```

   The output must end with `roman-anima source: shadow`. This run:
   - stages `config/events.akh-target.json` with the `roman-anima` source
     (`shadow: true`), the `anima` endpoint (`token_env ROMAN_ANIMA_TOKEN`)
     and the four filtered routes, and keeps the legacy `laya-*` and
     `*-anima.<role>.changed` routes;
   - renders the new reactor seeds into `config/reactor-seeds/`;
   - copies `daemon_credentials.py` into `bin/`;
   - copies the skillpack into each Roman's `kg/skills/`.
5. **Install the events config** with
   `python3 "$REPO/deploy/roman-mini/retarget_events.py" "$ROOT"`. The output
   lists the three `*-anima.work_item.changed` routes and
   `minerva-anima.plan.changed`.
6. **Stage the new seeds** (added disabled; existing ones are skipped):
   `python3 "$REPO/deploy/roman-mini/apply_reactor_seeds.py" "$ROOT"`. Do not
   run `--enable` yet. In shadow mode nothing is delivered, so a disabled
   seed changes nothing.
7. **Install akheventd.** It is a system LaunchDaemon, so this needs sudo:

   ```sh
   git -C "$REPO" merge-base --is-ancestor "$SHA" origin/main && echo "akheventd $SHA on origin/main"
   sudo launchctl bootout system/dev.roman-akhs.events
   install -m 755 "$CARGO_TARGET_DIR/release/akheventd" "$ROOT/bin/akheventd"
   shasum -a 256 "$ROOT/bin/akheventd" > "$ROOT/evidence/akheventd-main-$SHA.sha256"
   sudo launchctl bootstrap system /Library/LaunchDaemons/dev.roman-akhs.events.plist
   ```

8. **Restart the core** (new binary, new addenda text):
   `kill $(pgrep -f akhomed-main-)`. KeepAlive brings it back.
9. **Verify shadow mode.** Within about 5 minutes (27 projects, 8 per poll),
   `GET /v1/anima/roman-anima` shows `ok: true`, `shadow: true` and every
   project `baselined: true`, with `counters.emitted` equal to 0:

   ```sh
   RK="$(cat "$ROOT/secrets/monitoring-reader.key")"
   curl -s -H "Authorization: Bearer $RK" http://127.0.0.1:18435/v1/anima/roman-anima | python3 -m json.tool | head -40
   ```

   `logs/events.err.log` must show no `anima poll deferred` line. The
   observer and laya-controller keep running unchanged.

#### (b) The 48 h comparison

1. **Structural parity**, which needs no waiting:
   `python3 "$REPO/deploy/roman-mini/compare_anima_shadow.py" --replay`. It
   replays the rendered filters over every snapshot Laya recorded and must
   report `N/N` with no `MISMATCH` line (201/201 on 2026-09-26).
2. **Shadow against Laya** after at least 48 h:

   ```sh
   python3 "$REPO/deploy/roman-mini/compare_anima_shadow.py" --hours 48
   ```

   Both databases are opened read-only (`mode=ro`):
   `state/laya-decisions.sqlite3` and `state/events/akheventd.sqlite3`. To
   use a saved `GET …/shadow` body instead of the store, pass
   `--shadow-json FILE`.

   **The gate:**
   - `same roles` equals `routed by both`, and there is no `MISMATCH` line
     (exit status 0).
   - Entities listed as `laya_only` or `shadow_only` come from cadence, not
     routing. Laya re-queues a parked item every 5 minutes and re-decides on
     dossier changes, while the source emits on projection changes and caps
     each episode at 2. Spot-check each listed entity: for `laya_only`, check
     `GET /v1/anima/roman-anima` `attention` or an unchanged projection; for
     `shadow_only`, check whether Laya's queue simply had not reached it.

   Any `MISMATCH` stops the rollout. Fix the filter table in `full_akh.py`,
   regenerate the fixture (`python3 test_anima_routing.py --regenerate`),
   merge, and redeploy the shadow stage.

#### (c) Cutover

Run these in order. The seeds must be enabled before akheventd starts
delivering, because an admitted event with no enabled schedule is a terminal
`no_subscriber`.

1. **Render live:**

   ```sh
   python3 "$REPO/deploy/roman-mini/configure_core.py" --root "$ROOT" \
     --binaries "$CARGO_TARGET_DIR/release" --repo "$REPO" --live
   ```

   The output must end with `roman-anima source: LIVE (cutover)`. This run:
   - sets `shadow: false`;
   - adds the `watchdog` and the `board-watchdog` endpoint
     (`token_env ROMAN_BOARD_WATCHDOG`);
   - drops the `laya-*` and `*-anima.<role>.changed` routes and lists their
     schedules under `remove` in each seeds file;
   - drops `[[mcp.servers]] laya` and the `observer_status_file` key of
     `[monitoring]`.
2. **Install roman-memory 1.1.0** (see CUTOVER.md step 10 for `akhc`):

   ```sh
   akhc schema compile "$REPO/deploy/roman-mini/schemas/roman-memory.json" --out "$BK/roman-memory-1.1.0.akhschema"
   akhc schema show "$BK/roman-memory-1.1.0.akhschema"      # review: intervention_outcome, intervention_next_event
   akhc schema admit "$BK/roman-memory-1.1.0.akhschema" --yes
   for r in janus minerva vesta; do akhc -w $r schema install roman-memory@1.1.0; done
   ```

3. **Enable the new seeds and remove the retired ones:**

   ```sh
   python3 "$REPO/deploy/roman-mini/apply_reactor_seeds.py" "$ROOT"            # adds anything new
   python3 "$REPO/deploy/roman-mini/apply_reactor_seeds.py" "$ROOT" --enable   # includes the anima seeds
   python3 "$REPO/deploy/roman-mini/apply_reactor_seeds.py" "$ROOT" --remove-retired
   ```

   `--remove-retired` removes `roman-laya-<role>` and
   `roman-{janus,minerva}-anima.<role>.changed`.
4. **Stop the observer and the laya-controller.** Both are LaunchAgents, so
   no sudo is needed. Keep their plists as evidence so they do not reload at
   login:

   ```sh
   launchctl bootout gui/501/dev.roman-akhs.observer
   launchctl bootout gui/501/dev.roman-akhs.laya-controller
   mv ~/Library/LaunchAgents/dev.roman-akhs.observer.plist ~/Library/LaunchAgents/dev.roman-akhs.laya-controller.plist "$BK/"
   ```

   `bin/observer.py`, `bin/laya_controller.py` and their databases stay until
   the cutover is verified.
5. **Install the live events config and restart akheventd:**

   ```sh
   python3 "$REPO/deploy/roman-mini/retarget_events.py" "$ROOT"
   sudo launchctl bootout system/dev.roman-akhs.events
   sudo launchctl bootstrap system /Library/LaunchDaemons/dev.roman-akhs.events.plist
   ```

6. **Restart the core** so it loads the workspaces without the `laya` server:
   `kill $(pgrep -f akhomed-main-)`.
7. **Activate the skillpack** in each Roman, then check that it is `Hot`:

   ```sh
   for r in janus minerva vesta; do
     akhc -w $r acp skill activate laya-intervention
     akhc -w $r acp skill inspect laya-intervention
   done
   ```

   If `inspect` says the skill is unknown, check that
   `core-data/akh-medu/workspaces/$r/kg/skills/laya-intervention/skill.json`
   exists, then restart the core again (skills are discovered at start).

#### (d) Verification

The source visits every project within about 4 minutes. Deliveries are read
with the operator key
(`OK="$(cat "$ROOT/secrets/operator.key")"; curl -s -H "Authorization: Bearer $OK" http://127.0.0.1:18435/v1/deliveries`).

1. **A parked `too-big` WI wakes Minerva.** Take a WI that is parked with
   `parked_reason` `too-big` (an `anima-canary` one if there is one) and edit
   its title in Anima. One delivery on route
   `minerva-anima.work_item.changed` should appear, with
   `event_admission.schedules_woken` = `["roman-minerva-anima.work_item.changed"]`.
   Its `request.event.data.current.parked_reason` is `too-big`.
2. **`ci-infra-failing` wakes Vesta.** Do the same with a WI parked
   `ci-infra-failing`: one delivery on `vesta-anima.work_item.changed`, even
   when the WI has a live session.
3. **A held WI wakes nobody.** Edit the title of a WI parked `operator-hold`.
   The event is recorded with no delivery:

   ```sh
   sqlite3 "file:$ROOT/state/events/akheventd.sqlite3?mode=ro" \
     "select json_extract(cast(value as text),'$.event.occurred_at'), json_extract(cast(value as text),'$.deliveries')
      from records_v1 where key like 'ingress/11:roman-anima%'
      and cast(value as text) like '%\"subject\":\"work_item:<wi-id>\"%' order by 1 desc limit 1"
   ```

   The newest row ends in `[]`, and `/v1/deliveries` has nothing new for
   that entity.
4. **The third emission in an episode is suppressed and appears in the
   watchdog.** Edit the title of the `too-big` WI from check 1 twice more,
   waiting for one project visit (about 4 minutes) between edits. The first
   edit (the second emission) delivers. The next one does not.
   `GET /v1/anima/roman-anima` then lists the entity under `attention` with
   `suppressed: 1`. Within 60 s the status thread gets a watchdog notice with
   `state: blocked` whose `anima.roman-anima.attention` contains the entity.
5. **The reactor report reaches the Board.** The Minerva run from check 1
   posts `coordination/status@v1` to the status thread (its `report_thread`),
   and `akhc -w minerva acp query "SELECT ?o WHERE { ?s ?p ?o }"` (narrowed to
   the entity) shows `intervention_outcome` and `intervention_next_event`
   for `work_item:<id>`.
6. **The watchdog is healthy.** `logs/events.err.log` shows no
   `watchdog notice deferred` line, and there is no second watchdog poster on
   the thread (the observer is stopped).

#### (e) Rollback

- **During shadow mode** (before (c)): restore `$BK/events.json` to
  `config/events.json` and `$BK/akheventd` to `bin/akheventd`, then restart
  akheventd (`sudo launchctl bootout` and then `bootstrap`
  `system/dev.roman-akhs.events`). The observer and laya-controller never
  stopped. The staged seeds are disabled and harmless; remove them with
  `reactor_manage remove` if wanted.
- **After the cutover:**
  1. Restore these from `$BK`: `events.json`, `core.json`,
     `core-release.json`, the `workspaces` directory, the `reactor-seeds`
     directory, `akheventd` and `daemon_credentials.py`.
  2. Disable the new seeds. For each role, call `reactor_manage` with
     `disable` `roman-<role>-anima.work_item.changed` (and
     `roman-minerva-anima.plan.changed`) through `/v1/tools/call`.
  3. Re-add and re-enable the legacy seeds from the restored files: run
     `apply_reactor_seeds.py "$ROOT"`, then again with `--enable`.
  4. Put the observer and laya-controller plists back into
     `~/Library/LaunchAgents/`, then run
     `launchctl bootstrap gui/501 ~/Library/LaunchAgents/dev.roman-akhs.observer.plist`
     and the same for `dev.roman-akhs.laya-controller.plist`.
  5. Restart akheventd (sudo) and the core.

  roman-memory 1.1.0 can stay installed: its two predicates are additive.

**After a verified cutover (a follow-up PR, not this one):**

- In the repo, delete `observer.py`, `laya_controller.py`, `configure_laya.py`,
  `test_observer.py` and `test_laya_controller.py`. Also delete the `laya` and
  `changes` route kinds, the `observer` and `laya` entries in
  `SERVER_ENDPOINTS`/`COMMON_SERVERS`, and `OBSERVER_STATUS_FILE` from
  `full_akh.py`.
- On the Mac, remove `bin/observer.py`, `bin/laya_controller.py`,
  `secrets/laya-*.key` and `secrets/observer-tool.key`. Archive
  `state/observer.sqlite3` and `state/laya-decisions.sqlite3` to `evidence/`.
- The `monitoring_snapshot` tool then has no observer status keys. The Anima
  coverage it used to carry is at `GET /v1/anima/roman-anima`. Extending the
  tool to read it is a separate item.

## PR 8: Vesta's side effects move to akhsbxd/akhexecd sandboxes (D5)

Per the [step-2 plan](../../docs/ai/plans/2026-09-25-phase53g-roman-glue-conversion.md)'s
PR 8; discovery in
[docs/ai/research/2026-09-26-vesta-sandbox-execution.md](../../docs/ai/research/2026-09-26-vesta-sandbox-execution.md).
The cluster side is the talos branch `feat/vesta-sandbox-rbac`
(`clusters/talos.wegmueller.it/{agent-sandbox,vesta-sandbox}/`, operator
steps in `vesta-sandbox/README.md`). Rendering only: nothing on the Mac
changes until the operator writes `config/vesta-sandbox.json` (the cutover
switch) and re-renders.

| Former broker tool | After cutover | Where authority is enforced |
|---|---|---|
| `gitops_exec` (`docker exec` into `roman-vesta-gitops`) | `shell_exec` / `file_io` / `edit` / `glob` / `grep` in akhsbxd profile `vesta-gitops` (`[exec] kind = "execd"`) | the pod: no ServiceAccount token, no key in the executor container, Cilium egress DNS + github.com:22 only |
| `homelab_inspect` (host kubectl, cluster-admin) | `sandbox_exec` `ops`: `kubectl get/describe/logs` | ClusterRole `vesta-inspect` |
| `homelab_write` (host kubectl, cluster-admin) | `sandbox_exec` `ops`: `kubectl apply/patch/delete/scale/rollout restart/cordon/uncordon` | ClusterRole `vesta-write` + ValidatingAdmissionPolicies `vesta-*` |
| `gitops_publish` (host git push, unrestricted identity) | `git format-patch` in `vesta-gitops`, `git am` + tree check + non-force `git push` to `vesta/<topic>` in `ops` | deploy key + GitHub rulesets (only `refs/heads/vesta/**`, no force) |

Rendering (`full_akh.py`, `configure_core.py`):

- `config/vesta-sandbox.json` absent: Vesta renders exactly as before
  (`[exec] kind = "none"`, all four `vesta-ops` tools, `roles/addenda/vesta.md`).
- present (`{"token_file": "<root>/secrets/vesta-sandbox.key"}`, optional
  `sandbox_url` (default `http://vesta-sandbox.lab.home`) and
  `allow_insecure_http`): Vesta's `[exec]` becomes `execd` on `vesta-gitops`
  with `[[exec.sandboxes]] name = "ops", profile = "vesta-ops"`; the
  `vesta-ops` MCP allowlist drops `gitops_exec`, `homelab_write` and
  `gitops_publish` (`homelab_inspect` stays until the broker is retired); her
  policy prompt is `skills/vesta-remediation/procedures.md` instead of the
  broker addendum. A malformed file or a missing token file fails rendering
  closed.
- `skills/` is copied to `<root>/skills/` on every render.
- `vesta_ops.py`, `vesta_actions.py`, `configure_vesta*.py` and the
  `dev.roman-akhs.vesta-ops` LaunchAgent stay until the cutover is verified;
  their retirement is a follow-up.

akh-side code: `[[exec.sandboxes]]` (`src/akh_config.rs`) and the native
`sandbox_exec` tool (`src/agent/tools/sandbox_exec.rs`, parameters `sandbox`,
`command`, `timeout` 1..=600 s, `stdin` ≤ 1 MiB), allocated lazily so a
down `ops` sandbox never blocks Vesta from loading.

### Mac-side steps (not run by this pass; after the talos rollout is green)

1. Cluster first: talos `vesta-sandbox/README.md` "Operator steps" 1–7 (1Password
   items, image, deploy keys, rulesets, admission dry run, unsuspend Flux,
   verification). The raw Vesta token goes to
   `~/Library/Application Support/roman-akhs/secrets/vesta-sandbox.key` (0600).
2. Check the LAN route: `curl -s -o /dev/null -w '%{http_code}\n'
   http://vesta-sandbox.lab.home/v1/allocations/probe` answers `401`.
3. Back up: `cp -a core-config/akh-medu/workspaces core-config/akh-medu/workspaces.bak-$(date +%s)`.
4. Write the switch: `printf '{"token_file": "%s"}\n' "$ROOT/secrets/vesta-sandbox.key" > "$ROOT/config/vesta-sandbox.json"; chmod 600 "$ROOT/config/vesta-sandbox.json"`.
5. Re-render from a clean `origin/main` checkout that contains this PR
   (`configure_core.py --root … --binaries … --repo …`) and install the new
   akhomed built from main. It must know `[[exec.sandboxes]]` and
   `sandbox_exec`: an older binary at best reports the key as unknown and
   runs Vesta without her `ops` sandbox.
6. Restart the core (`kill $(pgrep -f akhomed-main-)`; KeepAlive restarts it).
   Vesta's load allocates `vesta-gitops`; a failure stops Vesta's load with
   the execd diagnostic in `logs/core.err.log` (fail loud, no local fallback).

### Verification (parity per former tool)

Run each as a Vesta turn (or through `/v1/tools/call` with the operator
token) and compare with the broker's result for the same input:

| Former call | Parity check |
|---|---|
| `gitops_exec {"command": "git -C talos status"}` | `shell_exec` `git clone /seed/talos talos && git -C talos status`; the receipt's binding says profile `vesta-gitops`, confinement `container:…`; `env \| grep -i -e token -e key` and `ls /run/vesta /var/run/secrets 2>&1` show no credential |
| `homelab_inspect {"action": "get", "resource": "pods", "namespace": "akh-medu"}` | `sandbox_exec {"sandbox": "ops", "command": "kubectl get pods -n akh-medu -o wide"}` returns the same pod list |
| `homelab_inspect` logs | `kubectl logs <pod> -n <ns> --tail=200 --limit-bytes=65536` |
| `homelab_write {"action": "patch", …, "dry_run": true}` | `kubectl patch … --dry-run=server -o name` succeeds in an app namespace and is refused in `flux-system` for a pod-spec change |
| `homelab_write` scale / rollout_restart / cordon | each with `--dry-run=server`: allowed |
| Secret read (must fail) | `sandbox_exec ops "kubectl get secret -n akh-medu"` → `Forbidden`; `kubectl auth can-i get secrets -A` → `no`; `kubectl exec` → `Forbidden` |
| `gitops_publish {"dry_run": true}` | skillpack publish steps with `git push --dry-run … HEAD:refs/heads/vesta/pr8-probe` accepted, and `HEAD:refs/heads/main` rejected by the ruleset |

Then confirm `mcp__vesta-ops__{gitops_exec,homelab_write,gitops_publish}`
are absent from `GET /v1/akhs/vesta/tools` and `sandbox_exec` is present.

### Rollback

1. `rm "$ROOT/config/vesta-sandbox.json"` and re-render (or restore the
   workspace backup from step 3): Vesta returns to `[exec] kind = "none"`,
   the four broker tools and the broker addendum.
2. Restart the core. The broker (`dev.roman-akhs.vesta-ops`, container
   `roman-vesta-gitops`) was never stopped; if it was, re-bootstrap it:
   `launchctl bootstrap gui/501 ~/Library/LaunchAgents/dev.roman-akhs.vesta-ops.plist`
   (plist in `$ROOT/launchd/`), `docker start roman-vesta-gitops`, then
   `launchctl kickstart -k gui/501/dev.roman-akhs.vesta-ops`.
3. Cluster side: suspend the `vesta-sandbox` Flux Kustomization; delete the
   publish deploy key on GitHub to revoke push authority at once.

