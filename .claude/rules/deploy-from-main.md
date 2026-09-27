# Deploy only from `main`

Every artifact that runs an akh or its platform daemons is built from a commit
that is already on `origin/main`. This covers all hosts and all mechanisms:
`akhomed`, the `akh` CLI, `akhrtd` / `akhsessd` / `akhexecd` / `akheventd` /
`akhsbxd`, `akh-acp`, `akh-trigger`, container images, Mac LaunchDaemons (including
the `roman-akhs` deployment), hephaistos, archibald and the Talos cluster.

- **Before building a deploy artifact:** `git fetch origin` and build from a clean
  checkout of `origin/main` (or a commit with
  `git merge-base --is-ancestor <sha> origin/main`). Never build from a feature,
  `wip/*`, `codex/*` or agent-worktree branch, and never from a dirty tree.
- **The fix is not on main yet:** open a PR and merge it first, then deploy. Hot
  fixes follow the same rule; merge a small PR fast instead of shipping the branch.
  Do not hand-build "just this once" from a branch.
- **Record provenance:** every deployed binary or image carries the short main SHA
  in its name, tag or evidence record (for example `akhomed-main-<sha>`,
  `:main-<sha>`). Content hashes alone are not provenance; a hash cannot be traced
  back to a commit.
  - **Stable-named binaries** (for example the Roman core's `bin/akhomed`,
    installed and code-signed at a fixed path so its macOS Local Network TCC
    approval survives a redeploy — see `deploy/roman-mini/CUTOVER.md`) carry
    the SHA in an evidence record instead of the filename: the record names
    the commit and the installed file's own verified sha256, and the running
    binary is checked against that sha256 before every start. Tracing such a
    binary to a commit means reading its evidence record and confirming the
    sha256 matches — a stable name with no sha256 check next to it is not
    provenance.
- **Before restarting anything:** verify the artifact's SHA is an ancestor of
  `origin/main` and quote the check in the deploy report.
- **Verifying a running deployment:** a service whose binary cannot be traced to a
  main commit is out of policy. Report it and plan the redeploy from main. Don't
  paper over it by merging whatever that branch contained without review.
- **Worktree caution:** inside a runner or agent worktree, branch commits look like
  "main". Check against `origin/main` from a clean checkout, not the local
  branch name.

Why: deployments built from branches or uncommitted trees drift away from the
reviewed code, can't be rebuilt reproducibly, and hide which capabilities are
really live. Examples: `akhrtd` was hand-built from the unmerged `feat/runtime-otlp`,
and the `roman-akhs` core (`akhomed-d563ba34a247`) was built from an uncommitted
tree five minutes before `6c331fc8` landed, so it's identified only by a content
hash.
