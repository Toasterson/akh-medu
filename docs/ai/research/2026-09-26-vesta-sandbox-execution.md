# Vesta's side effects in akhsbxd/akhexecd sandboxes: discovery (53g PR 8)

- **Date:** 2026-09-26
- **Plan:** [2026-09-25-phase53g-roman-glue-conversion](../plans/2026-09-25-phase53g-roman-glue-conversion.md), PR 8, decision D5
- **ADR:** [113](../decisions/113-full-akh-always.md), decision 3 (side effects through akhsbxd/akhexecd)
- **Anima:** WI `01a0d569-1cf1` item 5 (umbrella `01a0d43b-bf90`)
- **Basis:** `origin/main` `8ab096a3`, talos `origin/main` `f101eb5`, the live `roman-akhs` deployment and the home cluster, all read-only.

## What Vesta does today

`deploy/roman-mini/vesta_ops.py` is an MCP broker on `127.0.0.1:18437`
(LaunchAgent `dev.roman-akhs.vesta-ops`). With `vesta_actions.py` it serves
four tools:

| Tool | Mechanism | Authority actually used |
|---|---|---|
| `gitops_exec` | `docker exec` into the Colima container `roman-vesta-gitops` (no network, read-only root, uid 501, one bind mount: `workspaces/vesta-gitops`, a toasterson/talos checkout) | none; the broker re-verifies the container's confinement before each call |
| `homelab_inspect` | host `kubectl get/describe/logs` over 18 fixed resource kinds; Secrets refused by a Python allowlist | the host kubeconfig, context `admin@talos.wegmueller.it-1` (cluster-admin) |
| `homelab_write` | host `kubectl apply/patch/delete/scale/rollout restart/cordon/uncordon`, argv built in Python, `--context` from `config/vesta-ops.json` | the same cluster-admin context |
| `gitops_publish` | `git bundle` out of the container, host `git push --porcelain` (non-force) of an exact SHA to `git@github.com:toasterson/talos.git` | the host user's SSH identity, i.e. push to any branch including `main` |

So every limit on Vesta was a Python argument check in front of cluster-admin
and an unrestricted push identity. D5 keeps her powers and moves the limits
into the cluster and the Git remote.

## Provisioners: what exists

akhsbxd (`crates/akh-session-services/src/sandbox.rs`, config
`config.rs::Provisioner`) has three provisioner kinds:

- `existing { executor, root }`: a directory on an already-running akhexecd.
  The Mac's `sandbox` LaunchDaemon uses it for all three Romans
  (`workspaces/<role>`), with akhexecd as an ordinary host process. It is not
  a boundary (ExecdBackend reports it unconfined).
- `command { acquire, release }`: an admin JSON stdin/stdout adapter.
- **`kubernetes { api, namespace, resource, template, executor }`**: creates
  a SIG agent-sandbox `Sandbox` (`agents.x-k8s.io/v1beta1`) from an
  admin-owned template, named after the allocation, annotated with the
  session and profile revision; waits for `Ready`; binds the executor
  endpoint, whose URL may contain `{sandbox}`
  (`http://{sandbox}.<ns>.svc.cluster.local:8080`). Release deletes the
  Sandbox with a UID precondition and waits for foreground deletion.
  Lost-executor recovery also runs through it.

**The Kubernetes provisioner exists and is in production on Archibald**:
`clusters/archibald/agent-sandbox/` (upstream v1.0.0 controller + CRD) and
`clusters/archibald/akh-platform/` (akhsbxd Deployment `sandbox`,
ServiceAccount `sandbox` with a Role limited to `sandboxes get/create/delete`
in `akh-workspaces`, 30 workspace profiles rendered by `render.py` from
`fleet.json`). No new provisioner code was needed.

What does **not** exist, and why Archibald's installation cannot simply be
reused for Vesta:

1. The home cluster (`talos.wegmueller.it`, the cluster Vesta operates) has no
   agent-sandbox CRD and no akhsbxd (`kubectl get crd`, `-n akh-platform`,
   read-only). A pod on Archibald can hold a home-cluster credential only as a
   long-lived token Secret; a pod on the home cluster gets the `vesta`
   ServiceAccount's projected token with no Secret at all, and its RBAC is
   evaluated by the cluster it acts on. So Vesta's sandboxes belong on the
   home cluster.
2. The akh side had one execd profile per workspace. `[exec] kind = "execd"`
   binds `shell_exec`/`file_io`/`edit`/`glob`/`grep` to one akhsbxd profile,
   so a credential-free workspace plus a privileged second sandbox was not
   expressible.
3. The fleet workspace image has no `kubectl` or SSH client.

## How a Sandbox pod gets its checkout and credentials (Archibald pattern)

- Workspace: `emptyDir` at `/workspaces`; akhexecd's workspace is
  `/workspaces/<allocation id>`. Repositories are cloned by akhexecd from the
  profile's `repository_bootstrap`, which needs an exact commit id in the
  allocation request. `ExecdBackend` (akhomed) allocates with
  `bootstrap: None`, and akhsbxd rejects a profile with a bootstrap but a
  request without one, so akhomed-driven sandboxes cannot use it today.
- Credentials: Secret volumes mounted `0440` into the executor container:
  token verifiers (`executor-verifiers`), the akh's Board credential, and a
  Git config with HTTPS credentials (`workspace-git`). The executor container
  can read all of them, so on Archibald "sandbox" means "container runtime
  boundary", not "credential-free".
- Tokens: akhsbxd's executor gateway forwards the caller's own bearer token,
  so akhexecd must accept the akh's token (role `session`) and akhsbxd's own
  executor token (role `sandbox`, for attach/renew/release).

## Minimal path (implemented in this PR)

### (a) Credential-free GitOps workspace (replaces `gitops_exec`)

- akhsbxd profile `vesta-gitops`: Kubernetes provisioner, Sandbox pod with
  `automountServiceAccountToken: false`, an init container that clones
  toasterson/talos into an `emptyDir` at `/seed/talos` with a **read-only**
  deploy key mounted only into the init container, and the executor container
  mounting `/seed` read-only and no key. Cilium limits egress to DNS and
  `github.com:22` (needed by the init container; the executor has no key).
- Vesta's workspace `[exec] kind = "execd"`, `profile = "vesta-gitops"`: the
  existing file and shell tools run there unchanged.

### (b) Privileged ops sandbox (replaces `homelab_inspect`, `homelab_write`, `gitops_publish`)

- akhsbxd profile `vesta-ops`: same image, `serviceAccountName: vesta`
  (projected token, in-cluster `kubectl`), the **write** deploy key mounted for
  `git push`. Egress: DNS, kube-apiserver, `github.com:22`.
- ServiceAccount `vesta` with ClusterRoles `vesta-inspect` and `vesta-write`,
  derived verb by verb from `vesta_ops.py`/`vesta_actions.py` (table in the
  talos `vesta-sandbox/README.md`); no Secrets, no `pods/exec`.
- ValidatingAdmissionPolicies close what RBAC cannot express: Secret mounts
  and env references added through workloads, ServiceAccount swaps, host
  namespaces, hostPath and privileged containers, pod-spec edits in controller
  namespaces, and Flux spec edits beyond `suspend` (Flux applies as
  cluster-admin).
- Deploy key: GitHub deploy keys cannot be scoped to branches. Two rulesets do
  it: everything except `refs/heads/vesta/**` is restricted with only the
  repository admin in the bypass list; `vesta/**` blocks force pushes and
  deletions.
- akh side: `[[exec.sandboxes]]` in the execd table names additional
  profiles on the same akhsbxd, and a native `sandbox_exec` tool
  (`sandbox`, `command`, `timeout`, `stdin`) reaches them, allocating lazily
  so a down `ops` sandbox never blocks Vesta from loading.
- Publishing moves from "export a bundle, push with a host identity" to a
  patch hand-off: `git format-patch` in the GitOps workspace, `git am` in the
  ops sandbox on the same base, tree-hash equality check, non-force push to
  `vesta/<topic>`. The procedure is in the `vesta-remediation` skillpack.

### Transport from the Mac

Vesta's akhomed runs on the Mac today. The home-cluster akhsbxd is published
on the internal Gateway as `http://vesta-sandbox.lab.home` (LAN only, the
same trust decision as `seshat.lab.home`/`sokar.lab.home`), with bearer auth
and `allow_insecure_http = true`. akhexecd stays cluster-internal behind
akhsbxd's executor gateway.

## Where the pieces live

| Piece | Location |
|---|---|
| `[[exec.sandboxes]]`, `SandboxName`, `to_execd_sandbox_config` | `src/akh_config.rs` |
| `sandbox_exec` tool | `src/agent/tools/sandbox_exec.rs` (feature `execd-executor`), registered in `src/bin/akhomed.rs` |
| Vesta `[exec]` rendering, broker write-tool removal, cutover switch `config/vesta-sandbox.json` | `deploy/roman-mini/full_akh.py`, `configure_core.py` |
| Procedures | `deploy/roman-mini/skills/vesta-remediation/` (rendered into Vesta's system prompt in place of `roles/addenda/vesta.md` after cutover) |
| Cluster side | talos branch `feat/vesta-sandbox-rbac`: `clusters/talos.wegmueller.it/{agent-sandbox,vesta-sandbox}/`, two suspended Flux Kustomizations |
| Operator steps, verification, rollback | `deploy/roman-mini/INVENTORY.md` "PR 8"; talos `vesta-sandbox/README.md` |

## Known gaps

- The admission policies' CEL was not evaluated by an API server while
  authoring (no server-side apply allowed); the rollout runs a server dry run
  and reads `status.typeChecking` before relying on them.
- The Sandbox image digest is a placeholder until the operator builds it;
  both Flux Kustomizations are committed suspended.
- Sandbox workspaces are `emptyDir`: unpublished GitOps work is lost when the
  akh restarts (the broker's bind mount persisted). The skillpack says to
  publish promptly.
- GitHub rulesets on a private personal repository need GitHub Pro. Without
  them the publish key is not branch-limited and must not be created (talos
  README step 4).
- Residual authority, inherent to live remediation: Vesta can change the image
  or command of an ordinary application workload and so act with that
  workload's existing ServiceAccount and mounted Secrets.
