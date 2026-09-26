//! Publish only a verified commit to an explicitly configured work-branch scope.
use crate::{executor::Executor, git_snapshot, types::*, workspace::Workspace};
use publication::{GitPublication, PublishGitSnapshot};
use repository::{CaptureGitSnapshot, GitSnapshot, is_commit_id};

impl Executor {
    pub(crate) fn publish_git(
        &self,
        operation: &WorkspaceOperation,
        ws: &Workspace,
    ) -> Result<GitPublication> {
        let request: PublishGitSnapshot = serde_json::from_value(operation.arguments.clone())
            .map_err(|_| {
                Fault::invalid(
                    "publication requires a snapshot receipt, repository and work-branch ref",
                )
            })?;
        validate_key(&request.snapshot_operation_id)?;
        publication::validate_branch(&request.branch_ref)?;
        let profiles: Vec<_> = self
            .config
            .git_publication
            .iter()
            .filter(|p| p.repository == request.repository)
            .collect();
        if profiles.len() != 1 {
            return Err(Fault::new(
                "forbidden",
                "publication repository is not uniquely configured",
            ));
        }
        let profile = profiles[0].clone();
        profile.validate()?;
        if !profile
            .branch_prefixes
            .iter()
            .any(|prefix| request.branch_ref.starts_with(prefix))
        {
            return Err(Fault::new(
                "forbidden",
                "branch is outside the configured publication namespace",
            ));
        }
        let git = self.config.git_snapshot.as_ref().ok_or_else(|| {
            Fault::new(
                "unsupported",
                "publication requires a pinned Git installation",
            )
        })?;
        let prior = self.receipt(&request.snapshot_operation_id)?;
        if prior.id != request.snapshot_operation_id
            || prior.request.id != prior.id
            || prior.status != TurnStatus::Completed
            || prior.request.operation != "git_snapshot"
            || prior.request.session_id != operation.session_id
            || prior.request.workspace_id != operation.workspace_id
            || prior.request.generation != operation.generation
        {
            return Err(Fault::conflict(
                "publication snapshot belongs to another workspace or did not complete",
            ));
        }
        let result = prior
            .result
            .as_ref()
            .filter(|r| r.success && r.receipt.as_deref() == Some(&prior.id))
            .ok_or_else(|| Fault::conflict("publication needs successful snapshot evidence"))?;
        let snapshot: GitSnapshot = serde_json::from_value(
            result
                .data
                .clone()
                .ok_or_else(|| Fault::conflict("snapshot data missing"))?,
        )
        .map_err(|_| Fault::conflict("snapshot data invalid"))?;
        let captured: CaptureGitSnapshot = serde_json::from_value(prior.request.arguments.clone())
            .map_err(|_| Fault::conflict("snapshot request invalid"))?;
        captured.validate()?;
        if snapshot.schema_version != SCHEMA_VERSION
            || snapshot.request.base_revision != captured.base_revision
            || snapshot.request.head_revision != captured.head_revision
            || snapshot.request.expected_patch.is_some()
            || snapshot.expected_patch_sha256
                != captured
                    .expected_patch
                    .as_ref()
                    .map(|patch| crate::digest(patch.as_bytes()))
            || snapshot.patch_sha256 != crate::digest(snapshot.patch.as_bytes())
            || snapshot.executable_sha256 != git.executable_sha256
        {
            return Err(Fault::conflict(
                "publication snapshot or Git installation differs",
            ));
        }
        let mut view = git_snapshot::open_view(ws, git, captured.head_revision.len())?;
        let policy_sha256 = crate::digest(
            &serde_json::to_vec(&profile)
                .map_err(|_| Fault::invalid("publication policy encoding"))?,
        );
        git_snapshot::run_blocking(async move {
            // Recheck local object integrity before using it as a push source.
            let current = git_snapshot::capture_in(&view, snapshot.request.clone()).await?;
            if current.head_tree != snapshot.head_tree
                || current.base_tree != snapshot.base_tree
                || current.patch_sha256 != snapshot.patch_sha256
            {
                return Err(Fault::conflict("verified repository contents changed"));
            }
            let existing_only = profile.require_existing_branch;
            view.enable_publication(profile);
            let previous = remote_head(&view, &request).await?;
            if existing_only && previous.is_none() {
                // D4 (53g PR 7): a merge-repair worker updates an existing PR
                // source branch; it never creates one.
                return Err(Fault::new(
                    "forbidden",
                    "this publication profile only fast-forwards an existing branch; no push was attempted",
                ));
            }
            let head = &snapshot.request.head_revision;
            let already_present = previous.as_ref() == Some(head);
            if let Some(old) = previous.as_ref().filter(|_| !already_present) {
                if view
                    .run(&["merge-base", "--is-ancestor", old, head], 64)
                    .await
                    .is_err()
                {
                    return Err(Fault::conflict(
                        "work branch has commits absent from the verified result; no push was attempted",
                    ));
                }
            }
            let acknowledged = if already_present {
                false
            } else {
                let lease = format!(
                    "--force-with-lease={}:{}",
                    request.branch_ref,
                    previous.as_deref().unwrap_or("")
                );
                let refspec = format!("{head}:{}", request.branch_ref);
                // Preserve the assigned base and every commit on the observed
                // work branch. Explicit lease refuses a race after this read.
                // An existing-branch-only profile pushes without any force
                // flag: the remote then accepts nothing but a fast-forward.
                let mut arguments = vec!["push", "--porcelain", "--no-verify", "--atomic"];
                if !existing_only {
                    arguments.push(&lease);
                }
                arguments.extend([request.repository.as_str(), refspec.as_str()]);
                let pushed = view.run(&arguments, 64 * 1024).await;
                let observed = remote_head(&view, &request).await;
                if observed.as_ref().ok().and_then(Option::as_ref) != Some(head) {
                    return Err(Fault::new(
                        "unknown_outcome",
                        "publication did not confirm its remote branch; inspect this original operation before any new write",
                    ));
                }
                // A transport failure followed by a successful observation
                // proves current availability, without inventing a push ACK.
                pushed.is_ok()
            };
            Ok(GitPublication {
                schema_version: SCHEMA_VERSION,
                base_revision: snapshot.request.base_revision,
                head_revision: head.clone(),
                observed_revision: head.clone(),
                previous_revision: previous,
                request,
                push_acknowledged: acknowledged,
                already_present,
                policy_sha256,
                at: now_ms(),
            })
        })
    }
}

async fn remote_head(
    view: &git_snapshot::View,
    request: &PublishGitSnapshot,
) -> Result<Option<String>> {
    let bytes = view
        .run(
            &[
                "ls-remote",
                "--refs",
                &request.repository,
                &request.branch_ref,
            ],
            4096,
        )
        .await?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|_| Fault::conflict("invalid remote reference observation"))?;
    if text.is_empty() {
        return Ok(None);
    }
    let mut lines = text.lines();
    let line = lines
        .next()
        .ok_or_else(|| Fault::conflict("missing remote reference"))?;
    let (head, branch) = line
        .split_once('\t')
        .ok_or_else(|| Fault::conflict("invalid remote reference"))?;
    if lines.next().is_some() || branch != request.branch_ref || !is_commit_id(head) {
        return Err(Fault::conflict(
            "remote returned a different or ambiguous branch",
        ));
    }
    Ok(Some(head.into()))
}
