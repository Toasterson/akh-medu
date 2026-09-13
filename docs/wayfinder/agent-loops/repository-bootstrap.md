# Exact repository bootstrap

Repository preparation is a session prerequisite, independent of the model loop.
The session owner selects an allowed repository and a complete lowercase commit
object ID. `akhsbxd` coordinates the checkout; `akhexecd` owns every file and
process effect. The allocator starts configured editor services only after Git
verifies the assigned commit. `akhsessd` requires matching completion evidence
before binding the workspace and dispatching a runtime job.

The session daemon's Akh profile enables this requirement with a nonempty
`bootstrap_repositories` list. Its digest is pinned at session creation. The
workspace profile independently lists the allowed URLs and Git configuration:

```toml
# In the session daemon's configured Akh profile:
[akhs.thoth]
bootstrap_repositories = ["ssh://git@git.example/team/project.git"]

# In the sandbox daemon's configured workspace profile:
[workspaces.rust.repository_bootstrap]
repositories = ["ssh://git@git.example/team/project.git"]
executable = "/usr/bin/git"
config_file = "/private/bootstrap/gitconfig"
```

The executable and optional configuration file are absolute paths **inside the
executor environment**. Install them through the sandbox image and the existing
private configuration/login mechanism. Omit `config_file` for `/dev/null`.
Credential values do not belong in this public bootstrap contract. The private
file can select a credential helper or SSH configuration without placing secrets
in session messages. Its filesystem path alone does not isolate secrets from
arbitrary execution under the same OS identity; use the sandbox's credential
helper and permission boundary for that separation.

The owner needs the `session_bootstrap` role and posts to
`/v1/sessions/{session}/repository`:

```json
{
  "request_id": "persistent-request-id",
  "checkout": {
    "repository": "ssh://git@git.example/team/project.git",
    "revision": "0123456789012345678901234567890123456789"
  }
}
```

The immutable receipt acknowledges intent. It does not claim that Git ran.
Retries retain the same request, repository and commit. A changed actor or intent
is refused; historical acknowledgement does not reopen an archived session.
URLs require HTTPS or SSH without embedded passwords or query parameters. Local
HTTP/Git transports are available only on loopback for native fixtures. Each
daemon also requires an exact configured repository match.

Allocation pins the complete Git bootstrap profile. A configuration change cannot
alter later steps of an existing allocation. Bootstrap uses seven deterministic
executor operation identities:

1. Inspect the workspace and require it to be empty.
2. Initialize Git without copying a template directory.
3. Register the configured origin URL.
4. Fetch the assigned object without automatic tags or submodule recursion.
5. Check that the object is a commit.
6. Check out that commit detached, without forcing away existing files.
7. Verify that `HEAD` equals the assigned object ID.

Git commands use literal argument vectors, disabled hooks and explicit protocol
restrictions. System and ambient user Git configuration are suppressed; only the
configured private file is read. Terminal and askpass prompts are disabled.
Git environment and configuration controls are described in the
[Git manual](https://git-scm.com/docs/git); initialization and fetch options are
defined by [git-init](https://git-scm.com/docs/git-init) and
[git-fetch](https://git-scm.com/docs/git-fetch).

Every step requires its original successful executor receipt. Lost HTTP responses
retry the same operation identity. Unknown execution stops advancement until an
administrator records an inspected result through executor reconciliation. A
failed step preserves the workspace and receipt for inspection or release. The
coordinator renews the executor lease while progressing; a released or expired
workspace cannot be silently reacquired. Completion records the original request,
workspace, generation, verified revision and all seven receipt identities.

Local native tests use a disposable Git daemon, real Git processes and the real
executor HTTP router. They cover exact checkout, lost replies, manager restart,
unknown-operation fencing and audited recovery, repeated allocation and existing
file preservation. Session tests reject wrong owners, sources, revisions and
workspace evidence. Anima's independent client verifies the session API against
a locally built daemon. These checks do not prove private Git authentication,
editor UI interaction, Kubernetes image contents or cross-host deployment.

Anima's staging/admission coordinator now resolves the exact repository/base and
pins this intent with its scoped result capability. Scheduler selection and the
natural result lifecycle remain to be connected. Neither a mutable branch name
nor an `agent_diff` digest is a Git commit.

Result capture is described in [repository snapshots](repository-snapshots.md).
