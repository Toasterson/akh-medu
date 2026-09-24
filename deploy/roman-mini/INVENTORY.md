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
| `laya_controller.py` | **Not in repo before this pass.** Durable event→classification→scoped-fixer controller: reads an observation queue (`state/laya-decisions.sqlite3`), calls `laya_service` to classify a blocker (product/execution/infrastructure/human/wait/investigate), and routes an intervention to the right role (`ROLE` map) or opens a mergiraf/Astra dispatch. Serves a small results API on `127.0.0.1:18447` and calls `laya_service` at `127.0.0.1:18446/decide`. | in: observations from `observer.py` (when `config/laya.json.event_only` is true), `secrets/laya-<role>.key`; out: `laya.<role>.intervention` outbox events, HTTP decisions API. | `dev.roman-akhs.laya-controller` (LaunchAgent) | added | Janus reactor + skill (plan's explicit step-2 item) |
| `laya_service.py` | **Not in repo before this pass.** Resident, loopback-only classifier (`127.0.0.1:18446`) with no tracker credentials or executors; scores blocker text against the 6 `CHOICES` and validates the returned probability simplex. | in: HTTP `/decide` POST; out: JSON `{choice, probabilities}`. | `dev.roman-akhs.laya` (LaunchAgent) | added | folds into the same Janus reactor/skill as `laya_controller.py` — it's the controller's model call, not a separate capability |
| `mergiraf_worker.py` | **Not in repo before this pass.** Library (no server, no launchd entry) imported by `janus_escalation.py`; defines the `POLICY` prompt, JSON-schema `TOOLS` (`dispatch_mergiraf`, `mergiraf_status`) and `escalation()` builder for a bounded, typed merge-conflict-resolution job dispatched to an Astra/T3 worker via the same broker. | in: called in-process from `janus_escalation.Broker.submit_merge`; out: a T3 thread prompt + schema-validated job fields (work item, PR, head/base SHAs). | none (imported, not a service) | added | Janus delegation (53e grants) + skill, alongside `janus_escalation.py` |
| `observer.py` | Polls Anima (rotating 8 projects/min), Grafana and VictoriaMetrics; maintains a deterministic watchdog signal, a durable outbox of `schedule.*`/`anima.*.changed`/`monitoring.changed` events, and (new) `laya.<role>.intervention` events when `event_only` mode is on; serves a read-only `monitoring_snapshot` MCP tool on `127.0.0.1:18436`. | in: Anima MCP, Grafana HTTPS, VictoriaMetrics via `127.0.0.1:18440` tunnel, `config/laya.json`; out: `state/observer.sqlite3` outbox/state, POSTs to akhsessd `127.0.0.1:18431`, board watchdog notices. | `dev.roman-akhs.observer` (LaunchAgent) | drifted → synced (see below) | event production → Rust `akheventd` source; monitoring snapshot → a Vesta tool (both named explicitly in the plan's step-2 list) |
| `provider_failure.py` | Classifies a provider HTTP/error response into `authentication`/`quota`/`rate_limit`/`unknown` so `observer.py` and `janus_escalation.py` apply the right cooldown (30m/1h/2h/4h, or operator-repair-only for auth). | in: response text/status from a provider call; out: a classification tag consumed by callers. | none (imported library) | in repo | folds into akhomed's provider error taxonomy (explicit step-2 item) |
| `t3_activity.py` | T3 ACP driver used as the transport for the "autonomous activity" provider instances (`roman-activity-<role>`); also runs as a background `--service` LaunchAgent. | in: T3 CLI invocation or `--service` mode; out: ACP session against akhsessd `127.0.0.1:18431`, reads `secrets/operator.key`. | `dev.roman-akhs.t3-activity` (LaunchAgent) | in repo | obsolete after cutover — once T3 speaks ACP directly to akhomed there is no separate "activity" transport to shim |
| `t3_entrypoint.py` | T3 ACP driver used as the transport for the three named Roman provider instances (`roman-janus`/`roman-minerva`/`roman-vesta`); dispatches on `sys.argv[0]`'s basename. | in: T3 CLI invocation, `secrets/t3-<role>.key`; out: `akh-acp` subprocess against `ws://127.0.0.1:18431/acp` (akhsessd) today. | n/a (invoked by T3, not launchd) | drifted → synced (see below); **also changed in this pass** — see "t3_entrypoint.py ACP endpoint" below | akhomed ACP once 53d/53f land; this pass adds the seam, doesn't flip the default |
| `vesta_actions.py` | Grafana MCP passthrough (`grafana__*` tool names) plus the `gitops_publish`/`homelab_write` action tools for Vesta's sandbox; talks to the Grafana MCP tunnel on `127.0.0.1:18441`. | in: MCP `tools/call`; out: Grafana RPC, GitOps sandbox operations. | none (imported by `vesta_ops.py`/`configure_vesta_actions.py`) | in repo | Vesta skill (homelab remediation stays a skill under the akh-native Vesta, not a separate broker) |
| `vesta_ops.py` | `homelab_inspect` (read-only kubectl get/describe/logs) MCP broker for Vesta, serving on a loopback port; the read-only counterpart to `vesta_actions.py`'s write path. | in: MCP `tools/call`; out: bounded `kubectl` invocations. | `dev.roman-akhs.vesta-ops` (LaunchAgent) | in repo | Vesta skill, same as `vesta_actions.py` |

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

At cutover, pointing these at akhomed's ACP/`/v1` listener (today
`127.0.0.1:18430`, per `configure_core.py`) requires first confirming akhomed's
TCP ACP accepts `_meta.akh` for role selection, or adding an `akh-acp
--akhomed` mode (both raised as open questions in 53d/53f — not resolved by
this pass).

## Shell-wrapper "scripts" (`roman-janus`, `roman-minerva`, `roman-vesta`, `roman-activity-*`)

These are not independently maintained scripts. `configure_t3.py`
(`shutil.copy2(source / 't3_entrypoint.py', binary)` /
`.../'t3_activity.py'`) installs one physical copy of `t3_entrypoint.py` per
role as `bin/roman-<role>` and one copy of `t3_activity.py` per role as
`bin/roman-activity-<role>` (plus `bin/t3_activity.py` itself for the
`--service` LaunchAgent); the role is selected at runtime from
`sys.argv[0]`'s basename. Confirmed live:

- `roman-activity-janus/minerva/vesta` are byte-identical to live
  `t3_activity.py` (0-line diff) — current.
- `roman-janus/minerva/vesta` differ from live `t3_entrypoint.py` by exactly
  the same stale "GLM-5.3" vs. "Mistral Large 3" string noted above — they
  were installed before that repo/live edit and `configure_t3.py` has not been
  re-run since. Re-running `configure_t3.py` (part of the normal
  install/update flow, not part of this pass) refreshes them from the now-synced
  `t3_entrypoint.py`.

No separate files are added to the repo for these; they are covered by keeping
`t3_entrypoint.py` / `t3_activity.py` current and re-running `configure_t3.py`.

## Anima runner glue outside the deployment root (inventoried, not copied)

Lives at `~/.config/anima-runner/roman-chat/`, a separate mechanism from
`roman-akhs` (its own `anima-runner` binary, own token files, own state dir).
Not trivially safe to fold into `deploy/roman-mini` in this pass — it has its
own binary/config lifecycle and no existing renderer here to extend.

| File | Purpose | Launchd label |
|---|---|---|
| `anima-activity.py` | Projects existing Roman automation wakes into Anima as AKH-authored messages (never human prompts); read-only against Akh sessions/tools, with a local outbox for ambiguous deliveries. | `dev.roman-akhs.anima-activity` (LaunchAgent, `gui/<uid>`) |
| `roman-chat.py` | Chat-only Anima connector to the existing Mac Roman akhs; `--runner` mode starts the Anima-side runner, `janus`/`minerva`/`vesta` mode `execv`s `akh-acp` against akhsessd exactly like `t3_entrypoint.py`. Starts no provider/cognitive daemon itself. | none directly (invoked by the `anima-runner` binary) |
| `anima-runner` (binary) | Vendored runner executable (not a script); out of scope for a python-glue sync. | — |
| `activity.json`, `activity-credential.json`, `activity-token`, `runner-token` | Config + credential files (paths/URLs and a narrow PAT the runner authenticates with); **not inspected for values**, only confirmed to be file-based, not embedded in a script. | — |
| `state/`, `logs/`, `proto/` | Runtime state, logs, and gRPC proto defs for the runner. | — |

**Follow-up (filed as a to-do, not done in this pass):** decide whether
`anima-activity.py` becomes part of the same akheventd-source conversion as
`observer.py` (it is doing an adjacent job — projecting Roman activity into
Anima) or stays a separate Anima-runner-owned mechanism, and if it moves,
bring `roman-chat.py`/`anima-activity.py` and a renderer for
`dev.roman-akhs.anima-activity.plist` into `deploy/roman-mini`.

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
| `anima-activity` | `python3 anima-activity.py --config activity.json` (outside `roman-akhs`) | `PATH` | not part of `deploy/roman-mini` — see Anima runner glue above |
| `grafana-tunnel` | `kubectl -n observability port-forward … mcp-grafana 18441:80` | `KUBECONFIG`, `HOME`, `PATH` | `configure_vesta_actions.py` (in repo) |
| `janus-escalation` | `python3 janus_escalation.py <root>` | `PATH`, `HOME` | `configure_janus_escalation.py` (in repo) |
| `laya-controller` | `<repo>/experiments/typed-decisions/.venv-comparison/bin/python laya_controller.py <root>` | `TOKENIZERS_PARALLELISM`, `USE_TF`, `HF_HUB_OFFLINE` | **no renderer in the repo** — follow-up. Note: `ProgramArguments[0]` is a venv interpreter path inside this akh-medu checkout's `experiments/` tree, not a system Python — a real dependency worth resolving before this plist is reproducible from a clean clone. |
| `laya` | `<repo>/experiments/typed-decisions/.venv-comparison/bin/python laya_service.py <root>` | `TOKENIZERS_PARALLELISM`, `USE_TF`, `HF_HUB_OFFLINE` | **no renderer in the repo** — follow-up (same venv-path note as `laya-controller`) |
| `observer` | `python3 observer.py <root>` | `KUBECONFIG`, `HOME`, `PATH` | `activate.py` (in repo) |
| `sessions` | `python3 daemon_credentials.py sessions <root>` | `PATH`, `HOME`, `RUST_LOG` | `prepare.py` (in repo) |
| `t3-activity` | `python3 t3_activity.py --service` | `PATH`, `HOME` | `configure_t3.py` (in repo) |
| `vesta-ops` | `python3 vesta_ops.py <root>` | `KUBECONFIG`, `HOME`, `PATH` | `configure_vesta.py` (in repo) |

## Follow-up work items (step 2, filed per the plan's explicit list)

Each of these needs its own work item before it lands (not done in this pass):

1. **`observer.py` event production → Rust `akheventd` source**, and its
   `monitoring_snapshot` MCP tool → a Vesta tool. Also decide whether
   `anima-activity.py` (Anima-runner glue, not `roman-akhs`) folds into the
   same conversion.
2. **`laya_controller.py` (+ `laya_service.py`) → a Janus reactor plus skill.**
   The event-driven decision-queue design (`event_only` mode, now the live
   default path once `config/laya.json` enables it) maps naturally onto
   ADR 113's "admission belongs to the akh" — this is scheduling logic that
   belongs inside akhomed's reactor, not a standalone HTTP classifier +
   controller pair. Also resolve the `experiments/.venv-comparison` dependency
   noted above (it should not be a load-bearing path for a production plist).
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
5. **`t3_activity.py` obsolete after the cutover** — no separate "activity"
   provider transport is needed once T3 speaks ACP directly to akhomed.
6. **Render the five plists with no repo mechanism today**
   (`otlp-tunnel`, `runtime-minerva`, `runtime-vesta`, `laya-controller`,
   `laya`) — either extend `prepare.py`/`activate.py` or add small
   `configure_*.py` scripts matching the existing pattern, so these are
   reproducible from a clean clone like the others already are.
7. **`daemon_credentials.py`'s new `core` role is not yet wired to
   `dev.roman-akhs.core.plist`.** `configure_core.py` still renders
   `ProgramArguments: [str(executable)]` directly. Routing it through
   `daemon_credentials.py core <root>` (so the core process actually receives
   `ROMAN_OLLAMA_KEY` and the MCP tokens) is a 53f runbook step, done together
   with flipping `[exec]`/`[autonomy]`/`[[mcp.servers]]` in the target
   workspace TOMLs — not done in this pass to avoid touching the live plist
   ahead of that verified cutover.
8. **Sokar has no credential distinct from `ROMAN_ARCHIVE_TOKEN`.** Confirm
   whether that's intentional (shared token good enough for Sokar's current
   scope) or whether Sokar needs its own 1Password reference before 53i's
   `[[mcp.servers]]` wiring goes live.
