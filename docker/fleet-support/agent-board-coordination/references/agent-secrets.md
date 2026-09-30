# Cluster credential delivery

Read this before accessing credentials, changing a secret consumer, or handling
missing or expired authentication. Use the identity assigned to this workload.
OpenBao supplies migrated machine secrets through External Secrets Operator;
agents consume the existing environment variables and private mounted files.

## Supplied paths

- **Board, when granted:** use
  `python3 /opt/agent-board-coordination/scripts/board_request.py` in the sandbox.
  It loads the assigned `/usr/local/etc/akh-medu/board.env` without displaying it.
  Legacy runners receive `BOARD_URL` and `BOARD_AGENT_TOKEN` in their environment.
  Keep that identity across requests; fetching another token is unnecessary.
- **Forgejo/Git, when granted:** retain the installed Git credential configuration
  under `/run/akh/git`. Vesta uses its scoped `/run/vesta/git` mounts. Work through
  the supplied repository tools. Coordination and delegated-worker profiles have
  narrower grants; a guide or helper does not add a Board/Git permission.
- **S3/sccache, when granted:** use the existing `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, and `SCCACHE_*` environment. The sandbox target Secret
  is `workspace-sccache`; do not print its values or duplicate a secret fetch in
  each build. Use the configured endpoint and bucket rather than operator keys.
- **Kubernetes/gateway operations, when granted:** Vesta's ops profile mounts
  `/run/vesta/kubeconfig/config`. Other profiles use only their supplied tools;
  a missing kubeconfig is not permission to retrieve a cluster administrator key.
- **Native model runtimes:** the private provider login and supplied MCP/session
  credentials remain managed by the runtime. Use the sandbox's delivered
  machine credentials through its workspace tools. Native runtime access does
  not grant direct access to another agent's files or secrets.

## Missing credentials and offline limits

For a missing field, expired cache or rejected admission, report the failing
identity, resource and operation without values. An operator must repair the
namespace ExternalSecret, scoped identity or enrolled cache. There is no runtime
1Password fallback; administrator certificates, ESO bootstrap keys, host cache
keys and another agent's credentials stay outside the sandbox.

Archibald admission checks a 24-hour deadline since the current ESO generation
last synchronized. A home outage can leave valid local credentials usable until
that deadline. This does not supply network access to a remote API, start home
sessions offline, or revoke values already held by running processes. Keep the
clock and admission policy intact.

## Changing a consumer

Use the existing ExternalSecret to identify the exact namespace store, KV path,
field and target ownership; never infer access from a former item name. Coordinate
infrastructure changes on the Board and take the appropriate lease. The operator
runbooks are `docs/agent-secrets-agent-guide.md` and `docs/openbao-operations.md`
in the homelab repository; if that checkout is unavailable, request the runbook
and the scoped operation rather than broadening access.

Verify authentication through a harmless authorized operation. Report identity,
scope and success. Keep credential values out of output, logs, command arguments
and Git, and preserve TLS verification.
