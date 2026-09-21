# Native Ptah homelab canary deployment audit

Read-only audit on 2026-09-14. No cluster resources, credentials, fleet routing,
or running services were changed. No database queries were used.

## Finding and deployment boundary

Native Ptah can be added without replacing Tecton/Kimi or any other provider.
However, attaching it to the existing canonical session service currently
requires a shared-service restart. `akh-platform/sessions` has one replica and
`strategy: Recreate`; its configuration is loaded once into an `Arc<Config>`.
There is no configuration reload route or SIGHUP handler. A ConfigMap update
alone does not activate a new runtime, principal, or Akh profile.

The shared cognitive core also has one replica/Recreate. Its scoped session
principal excludes `ptah`, and `CoreConfig` is loaded once. A dedicated Ptah
workspace additionally needs an Archibald sandbox profile; akhsbxd uses the same
startup-only configuration mechanism. These restarts affect other fleet users.
The current authority to reactivate/restart local Ptah does not by itself cover
interrupting shared sessions, core, or sandbox admission. Prepare the additive
Git/config/credential changes first; then request a concrete bounded maintenance
window for these shared owners. Do not restart other runtime providers.

Do not start a second sessions daemon against the canonical store as a workaround:
that would introduce competing dispatch ownership. An isolated canary stack would
be a separate test lane, not evidence of canonical session integration.

## Source of truth

Canonical context: `admin@talos.wegmueller.it`; namespace: `akh-platform`.
Both the sessions Deployment and ConfigMap carry Flux labels for Kustomization
`flux-system/akh-platform`. They are not Helm resources.

- GitRepository `flux-system`: `ssh://git@github.com/toasterson/talos`, branch `main`.
- Applied and source artifact revision at audit: `0247a99ad0150d7a0d2ee68dedfadd21d17000ad`.
- Flux path: `clusters/talos.wegmueller.it/akh-platform`; interval 10 minutes,
  `prune: false`, `wait: true`.
- The Kustomization includes generated `resources.json`.
- Generator inputs/code: `clusters/archibald/akh-platform/fleet.json`, `render.py`,
  `home_placement.py`, and `home_sessions.py`.
- A read-only checkout of that revision is at `/tmp/ptah-talos-audit.wDGIwd`.

Do not hand-edit only the generated/live ConfigMap: the next rendering would
erase the change. The renderer assumes ordinary fleet members have a Board UUID,
shared tool catalogs, event automations, and an HTTP-backed or CLI runtime. Add an
explicit external native/manual-canary branch so Ptah obtains a session profile
and endpoint without an unwanted Kubernetes runtime pod or automatic assignments.

The source README explicitly excludes the previously deactivated Hephaistos Akh.
That historical restriction needs updating alongside the newly authorized native
Ptah configuration; it is not a reason to reactivate the previous monolith.

Tecton is independently configured as `runtime-tecton`, model `kimi-k3`, revision
`tecton-output-32768-v2`. Preserve that inventory entry, its endpoint, credentials,
workspace profiles, Deployment, and running jobs byte-for-byte.

## Additive configuration and credentials

Local runtime configuration is [akhrtd.json](../../../examples/native-ptah/akhrtd.json).
Its service ID is `runtime-ptah`, profile `ptah`, revision
`ptah-native-canary-v1`, kind `internal`, model `ptah-qwen38`, brokered execution.
The explicit device/model/tokenizer hashes and physical context cap remain part
of the immutable native profile. The listener is `192.168.1.114:18110`.

1. Add `sessions.endpoints.runtime-ptah` pointing to that listener, with a distinct
   mounted outgoing credential. Plain LAN HTTP requires the explicit
   `allow_insecure_http: true` setting; alternatively stage authenticated TLS and
   its CA. The LAN address is on the same network as the homelab nodes, but actual
   pod-to-listener connectivity must still be measured after the listener exists.
   No NetworkPolicy was found in the namespace; this is not proof of host firewall
   reachability.
2. Add `sessions.akhs.ptah` with the exact runtime binding above, a dedicated
   `ptah-canary-workspace` profile, short instructions, and a minimal tool catalog
   (`read_file`, and only the additional tools required by the canary). The 2048
   token canary cap cannot accommodate the full fleet catalog and instructions.
   Do not inherit another Akh's Board identity or broad assignment tools.
3. Add an incoming session principal `runtime-ptah`, role `runtime`, scoped only
   to `ptah`. For manual ingress, add a distinct `ptah-canary` client principal
   scoped only to Ptah, or explicitly extend the intended authenticated client's
   allowlist. Do not enable Djehuty or project assignment automations merely to
   perform this canary.
4. Use the real `core-mac` logical endpoint, currently `http://core:8080`, and add
   `ptah` to that core's sessions principal. Verify/create the Ptah cognitive
   workspace/config through the core's supported interfaces. The `core-fleet`
   endpoint exists in sessions config but its Kubernetes Service was absent at
   audit; do not choose it. Preserve `akh-medu/akh-medu-seshd`, which is a separate
   knowledge service, and preserve all existing core workspaces/state.
5. Add a Ptah-specific Archibald sandbox profile with executor verification and
   no other Akh's Board credentials. Current fleet templates mount `board-<akh>`;
   sharing `tecton-workspace` would give the wrong identity. A minimal manual
   canary profile can omit Board/Anima credentials entirely. Preserve the existing
   authenticated relay and executor routing.

Credential pairs (generate new values securely; never print or commit them):

| Direction | Sender stores raw token | Receiver stores verifier |
| --- | --- | --- |
| Sessions → native runtime | `sessions-credentials` key `sessions-runtime-ptah` | local `sessions-incoming.sha256` |
| Native runtime → sessions | local `sessions-outgoing.token` | `sessions-credentials` key `runtime-ptah-sessions.sha256` |
| Manual canary client → sessions | operator-controlled Ptah-only token file | new Ptah client verifier in `sessions-credentials` |

Existing `sessions-credentials` is mounted as a Secret and has no Flux ownership
label; it is not among the generated Secret resources. Use the established private
credential staging path and merge only the new keys, preserving every existing
key. Do not regenerate the entire fleet's credentials. Core's existing sessions
capability can retain its token while adding the explicitly scoped Ptah identity.

## Controlled activation and evidence

Stage the local native binary/unit/profile and the additive source changes, validate
rendered JSON, inspect the diff, and verify that no existing provider configuration
changed. Do not deploy or publish these changes during this read-only audit.

After shared restart authority is explicit, use supported session APIs to inspect
active turns and quiesce affected admission before the brief owner restarts.
Do not use SQL. Sessions shutdown drops dispatcher workers and drains HTTP for up
to ten seconds; durable work remains fenced on restart rather than implying that
an interrupted effective operation succeeded. Therefore, do not assume a Recreate
restart is invisible to active callers or runtime tool callbacks.

Activate only the prepared changes, preserving one owner per store. Then measure:

1. A real authenticated message through `https://anima.wegmueller.it/akh` produces
   a Ptah session whose bound runtime/model/revision match the native profile.
2. Ptah generates an actual `read_file` request; the canonical broker admits it,
   the dedicated executor returns a durable receipt, and the final answer reflects
   the file contents. Parser-only or direct backend success does not satisfy this.
3. A second message resumes the exact provider continuation and recalls the first
   exchange. Record session, turn, job, continuation, and receipt IDs, without
   credentials or private prompt contents.
4. Cancel a turn after generation has begun. Verify compute becomes quiescent,
   output is not falsely accepted, and the internal runtime's Unknown/reconciliation
   behavior is preserved. Resolve the observed outcome through the supported
   operator API; then verify a fresh session can use the resident model.
5. Verify Tecton/Kimi and other unchanged providers retain their identities and
   no runtime Deployment was rolled by the additive change.

The local old `akhomed.service` is currently **disabled, inactive/dead**, with
ExecStart `/home/toasty/.local/bin/akhomed` and a localhost port 8200 setting.
Archive its unit file after the native replacement is verified, retaining a
recoverable copy and all model/cognitive/runtime data. Do not delete shared
binaries or local stores, and do not enable the legacy monolith alongside native
akhrtd. No cleanup was performed in this audit.

## Code evidence

- `crates/akh-session-services/src/server.rs`: one startup `Config::load`, immutable
  shared config, SIGTERM/SIGINT shutdown, ten-second HTTP drain.
- `crates/akh-session-services/src/server/sessions.rs`: fixed route table and
  dispatcher ownership; no profile reload route.
- `src/core_service.rs` and `src/bin/akhomed.rs`: scoped CorePrincipal allowlist and
  startup-only CoreConfig.
- `crates/akh-session-services/src/runtime.rs`: internal cancellation remains
  Unknown until explicit reconciliation; no inferred success.
- `crates/akh-session-services/src/broker.rs`: live JSON Schema validation occurs
  before downstream tool effects and durable tool receipts remain session-owned.
