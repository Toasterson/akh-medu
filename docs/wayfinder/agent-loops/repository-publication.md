# Executor-backed repository publication

`akhsessd` owns a durable repository publication action independently of model
turns. Anima remains responsible for selecting a canonical Change and for PR,
review, CI, witness and outcome transactions. A published branch is not a PR or
an accepted Change.

The owner submits `POST /v1/sessions/{id}/repository/publications` with
`request_id`, `snapshot_request_id` and a full `refs/heads/...` work branch.
The dedicated `repository_publish` role is required. The session derives the
repository, base, executor, workspace generation and exact source operation from
its verified checkout and successful snapshot receipt. The returned
`SessionPublication` contains the immutable operation, dispatch/cancellation
timestamps and an optional authoritative executor receipt. A missing receipt
never proves publication. Exact retries retain the original action; changing
its snapshot or branch conflicts. GET the same route plus `/{request_id}` for a
read-only observation.

The session persists intent and then reserves dispatch in a transaction shared
with retirement. It reads the original executor receipt before considering the
first send. A request already marked dispatched is never resent when that
receipt is absent. A restarted dispatcher reconciles publication even when all
model turns have finished. Archive waits for every publication to settle;
retirement cancels only actions never reserved for sending. Unknown executor
operations also fence workspace mutations and release.

An administrator can cancel an unresolved send with no executor receipt at
`POST .../publications/{request_id}/reconcile`. The existing reconciliation
contract requires the exact operation fingerprint, a unique request ID,
quiescent senders and bounded evidence references. This path cannot declare a
publication successful. An existing unknown executor receipt must instead be
reconciled at that executor with its native audited recovery API.

`akhexecd` exposes `POST /v1/repository-publications`, accepting only the reserved
`git_publish` operation and `repository_publish` role. The generic tool
operation endpoint refuses this operation. Deployment must explicitly configure
`git_snapshot` with an absolute Git executable and its SHA-256, plus
`git_publication` profiles containing an exact repository and narrow
`branch_prefixes`, for example `refs/heads/anima/`. Publication is disabled by
default. An optional absolute `config_file` provides administrator-owned Git
configuration/credential helpers; workspace `.git/config` is never used as
publication configuration. Runtime/tool capabilities do not inherit this role.

The executor revalidates snapshot objects and patch/tree identity in a private
Git view. It observes the exact remote work branch and refuses an existing tip
absent from the selected result. It pushes one explicit ref using an atomic
push and a lease on the observed tip (or absence), and observes the branch again.
An observed matching commit proves availability at that observation time.
`push_acknowledged` separately records a successful push response;
`already_present` identifies a branch that already had the selected commit.
An unconfirmed observation after an attempted push remains unknown. No default
branch, tag, branch deletion or unobserved overwrite is supported.

Native verification on 2026-09-12 used owned temporary SHA-1 and SHA-256 Git
repositories, Git 2.50.1, actual loopback Git servers and the executor/session
HTTP routers. Tests cover concurrent remote advancement, namespace/role/source
fences, exact retry and restart, unknown mutation/release fences, response loss,
request loss, dispatcher recovery without active turns, retirement, audited
lost-send cancellation and archived receipt recovery. An independent compiled
`anima-akh` client also verified the original native snapshot and publication
receipts through the real session HTTP server. Its fault peer separately rejects
substituted identities, destinations, hashes, timestamps and claimed success for
unsettled results.

These checks do not prove a production credential helper, cross-host deployment,
PR creation, canonical Anima publication selection or natural outcome closure.
The Anima client is implemented; those business-lifecycle connections remain
separate work.
