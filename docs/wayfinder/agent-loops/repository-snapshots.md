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

An optional `expected_patch` string requests semantic verification of a proposed
patch. Git applies those exact bytes to the assigned base in a temporary index
and object store, and must produce the selected commit's tree. The result carries
`expected_patch_sha256` only after that check. The input patch stays in the
operation request; the snapshot's revision request omits it to avoid duplicating
the input again. This accepts equivalent unified/format-patch representations
and binary patches without requiring identical formatting. Anima's
`verify_repository_patch` client checks both the request and this digest.

The executor operation ID is SHA-256 over UTF-8
`repository_snapshot_intent/{session-byte-length}:{session}{request-id-byte-length}:{request-id}`.
Retrying an ID with another head or input patch conflicts. Lost replies recover the original
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

Semantic-patch tests pass unified diffs with different context, format-patch and
binary contents on both object formats. A valid patch producing a different tree
is refused, and the workspace index, refs, object counts and files remain intact.

Anima's independent compatibility client now validates the owning session,
admitted turn, verified checkout, workspace/generation, derived operation ID,
terminal status, commit/tree identities and patch digest before returning typed
snapshot evidence. Its actual native probe passes both capture and semantic-patch
verification against the real Akh routers after archival. Run the cross-repository fixture by building Anima's
`snapshot-probe` example and setting `ANIMA_AKH_SNAPSHOT_PROBE` to that absolute
binary path when running Akh's `repository_bootstrap` test with
`session,executor,sandbox` features. With no explicit probe path, that test proves
the Akh API only.

Anima's relay now verifies the latest eligible session proposal using its
canonical Change ID as the capture request ID. It supplies the immutable
assignment checkout and proposed patch, and selects credentials only through the
exact configured route. Migration 139 stores the full executor evidence in
`akh_session_result_snapshots`; the original producer receipt and Change remain
unchanged. Verification and its audit event commit together after rechecking
session ownership, phase, bootstrap, source receipt and Change under allocation
locks. A changed phase or failed audit rolls back local verification. Exact
retries reuse stored evidence; superseded evidence remains available as history.

The full-schema PostgreSQL test covers these transactions with a fault-injecting
protocol peer, including invalid producer evidence, bad remote evidence, a phase
change during capture, audit rollback, concurrent retries and a verified
successor. Native Git and the real protocol path are covered by the separate
cross-repository probe above. Neither test makes an unverified publication into
an authoritative remote branch. PR/publication selection, canonical outcome and
witness commits, and scheduler route enablement remain integration work.


## Conditional session archival

Canonical completion can post `ArchiveSelection` to
`/v1/sessions/{id}/archive-selected`: a durable `request_id` plus the observed
`generation` and `next_sequence`. The session ledger compares both under the
same transaction that checks settled turns/publications, stores the selection,
closes the session and appends its archive event. Any intervening event refuses
the stale decision without mutation. Only the exact successful selection replays,
including after restart; legacy archival and operator retirement cannot be
mistaken for that selection. Controller/owner authorization still applies.

Anima's independent `completion_checkpoint` reads the full ordered turn set.
Unknown/active turns and a final cancellation remain ineligible; a stopped turn
stays resumable. `archive_checkpoint` verifies the archived turn set, cursor,
session identity and actual lifecycle release before producing cleanup proof.
The proof says nothing about Change acceptance, review or CI. Canonical callers
must still persist their decision, import its archive event and commit their
business outcome/witness/claim release atomically.

Local verification: 19 session-ledger tests pass, including a late completed
follow-up, unchanged state on refusal, owner isolation and exact restart replay.
The native repository fixture also runs the independent Anima `completion-probe`
against actual session/sandbox/executor HTTP, drops the first successful archive
response and verifies actual allocation release. Its terminal provider reports
are injected fixtures, not live model inference. Enable it with
`ANIMA_AKH_COMPLETION_PROBE=/absolute/path/to/completion-probe` when running
`repository_bootstrap`. One client fault test separately covers all-turn
settlement, endpoint drift, changed replies and pending cleanup.
