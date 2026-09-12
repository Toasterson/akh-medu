# Executor Git snapshots

`akhexecd` can capture the exact patch between two local commits. `akhsessd`
exposes this through an owner-scoped API that supplies the assigned base and
workspace itself. This provides contents evidence for result ingestion. It does
not push a branch, create a PR, approve a Change, or release session ownership.

Enable the installation explicitly in the executor configuration:

```toml
[git_snapshot]
executable = "/usr/bin/git"
executable_sha256 = "<SHA-256 of the installed executable>"
```

The installation digest is checked for each new capture. Snapshot arguments
cannot replace the executable or supply environment/configuration/remote URLs.
The configured binary must support `--no-lazy-fetch` and SHA-256 repositories
when used. Native checks currently cover Apple Git 2.50.1 (Apple Git-155).

The owning principal with `session_bootstrap` posts:

```text
POST /v1/sessions/{session}/repository/snapshots
```

```json
{
  "request_id": "persistent-result-capture-id",
  "head_revision": "0123456789012345678901234567890123456789"
}
```

The session manager requires a verified bootstrap. It pins a `git_snapshot`
workspace operation with the bootstrap's base, existing executor/workspace and
generation before contacting the executor. The response is an `OperationReceipt`;
successful `result.data` is the shared `repository::GitSnapshot` contract.
Consumers must check completed/success, the exact request and receipt identities,
and the patch digest. A capture failure is not a successful empty result.

The executor operation ID is SHA-256 over UTF-8
`repository_snapshot_intent/{session-byte-length}:{session}{request-id-byte-length}:{request-id}`.
Retrying an ID with another head conflicts. Lost replies recover the original
receipt before considering another read request. The session manager caches
terminal evidence, including failures. Existing evidence can be recovered after
archival; archival cannot start a fresh capture. Interrupted reads do not claim
success and do not become an unknown workspace mutation.

Capture uses a temporary private bare Git view and the checked local object
directory. It ignores workspace refs, config/includes, index, worktree files,
attributes, grafts and replacement refs. It disables network/lazy fetching,
external diff drivers and text conversion; Git performs full commit-type,
ancestry and strict object checks. The resulting binary-capable patch and
NUL-delimited statistics describe the selected immutable commits. Git's controls
are documented in the [Git manual](https://git-scm.com/docs/git) and
[diff reference](https://git-scm.com/docs/git-diff).

Git commands share a 30-second deadline and bounded output; patches are limited
to 1 MiB. Non-UTF-8 textual diff output is refused without lossy conversion.
Large or corrupt object stores can therefore refuse capture. The current
checkout contract requires a standalone `.git` directory with self-contained
objects; linked worktrees, symlink roots and alternate object stores are not
supported. Like arbitrary executor commands, Git runs under the executor's OS
sandbox and identity. Concurrent editors may keep working: a snapshot asserts
selected commit contents, not a clean/current checkout or exclusive file access.

Native local tests cover SHA-1/SHA-256, applying the returned binary patch to an
independent base checkout and comparing the resulting tree, odd filenames,
packed objects, corrupt/oversized results, configuration and replacement-ref
isolation, exact retry/restart, scope and workspace fences. The real session and
executor routers also pass lost-response recovery after a real Git bootstrap,
including retrieval after archival. These tests do not prove remote publication,
cross-host deployment, or Anima's canonical finalization.
