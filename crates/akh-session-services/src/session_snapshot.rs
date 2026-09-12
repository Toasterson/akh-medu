//! Owner-scoped result evidence, with no model-selected execution destination.
use crate::{config::Principal, server::SessionApp, store::key, types::*};
use repository::{CaptureGitSnapshot, GitSnapshot, SnapshotSessionRepository, is_commit_id};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct Intent {
    request: SnapshotSessionRepository,
    binding: WorkspaceBinding,
    operation: WorkspaceOperation,
}

impl SessionApp {
    pub async fn snapshot_repository(
        &self,
        principal: &Principal,
        id: &str,
        request: &SnapshotSessionRepository,
    ) -> Result<OperationReceipt> {
        crate::http::role(principal, "session_bootstrap")?;
        validate_key(&request.request_id)?;
        if !is_commit_id(&request.head_revision) {
            return Err(Fault::invalid("snapshot requires a full nonzero commit ID"));
        }
        self.sessions.get(principal, id)?;
        let intent_key = key("repository_snapshot_intent", &[id, &request.request_id]);
        let receipt_key = key("repository_snapshot_receipt", &[id, &request.request_id]);
        let intent: Intent = self.sessions.store.transaction(|tx| {
            if let Some(prior) = tx.get::<Intent>(&intent_key)? {
                if prior.request != *request {
                    return Err(Fault::conflict(
                        "snapshot request ID already has another head",
                    ));
                }
                return Ok(prior);
            }
            let session: SessionRecord = tx
                .get(&key("session", &[id]))?
                .ok_or_else(|| Fault::missing("session not found"))?;
            if session.closed_at.is_some() || session.retirement.is_some() {
                return Err(Fault::conflict(
                    "session cannot start another repository snapshot",
                ));
            }
            self.sessions.check_repository_bootstrap(&session, true)?;
            let bootstrap = session
                .repository_bootstrap
                .as_ref()
                .and_then(|slot| slot.result.as_ref())
                .ok_or_else(|| Fault::conflict("snapshot requires a verified session checkout"))?;
            let binding = session
                .workspace
                .as_ref()
                .ok_or_else(|| Fault::conflict("workspace is not ready"))?;
            crate::session_bootstrap::validate_result(&session, binding, Some(bootstrap))?;
            let capture = CaptureGitSnapshot {
                base_revision: bootstrap.verified_revision.clone(),
                head_revision: request.head_revision.clone(),
                expected_patch: request.expected_patch.clone(),
            };
            capture.validate()?;
            let operation = WorkspaceOperation {
                id: crate::digest(intent_key.as_bytes()),
                session_id: id.into(),
                turn_id: "repository-snapshot".into(),
                workspace_id: binding.id.clone(),
                generation: binding.generation,
                operation: "git_snapshot".into(),
                arguments: serde_json::to_value(capture)
                    .map_err(|_| Fault::invalid("snapshot encoding"))?,
            };
            let intent = Intent {
                request: request.clone(),
                binding: binding.clone(),
                operation,
            };
            tx.put(&intent_key, &intent)?;
            Ok(intent)
        })?;
        if let Some(receipt) = self.sessions.store.get::<OperationReceipt>(&receipt_key)? {
            return Ok(receipt);
        }
        let client = self.client.in_sandbox(&intent.binding.sandbox_id)?;
        let operation = &intent.operation;
        // Recover a committed receipt before considering a read reissue. This
        // also recovers a lost response after the workspace was archived.
        let receipt: OperationReceipt = match client
            .get(
                &intent.binding.executor,
                &format!("/v1/operations/{}", crate::http::segment(&operation.id)),
            )
            .await
        {
            Ok(receipt) => receipt,
            Err(error) if error.code == "not_found" => {
                let current = self.sessions.get(principal, id)?;
                if current.closed_at.is_some()
                    || current.retirement.is_some()
                    || current.workspace.as_ref() != Some(&intent.binding)
                {
                    return Err(Fault::conflict("snapshot workspace is no longer available"));
                }
                client
                    .post(&intent.binding.executor, "/v1/operations", operation)
                    .await?
            }
            Err(error) => return Err(error),
        };
        validate_receipt(operation, &receipt)?;
        if matches!(
            receipt.status,
            TurnStatus::Completed | TurnStatus::Failed | TurnStatus::Cancelled
        ) {
            self.sessions.store.transaction(|tx| {
                if tx.get::<Intent>(&intent_key)?.as_ref() != Some(&intent) {
                    return Err(Fault::conflict("snapshot intent changed"));
                }
                if let Some(prior) = tx.get::<OperationReceipt>(&receipt_key)? {
                    if prior != receipt {
                        return Err(Fault::conflict("snapshot receipt changed"));
                    }
                } else {
                    tx.put(&receipt_key, &receipt)?;
                }
                Ok(())
            })?;
        }
        Ok(receipt)
    }
}

pub(crate) fn validate_receipt(
    operation: &WorkspaceOperation,
    receipt: &OperationReceipt,
) -> Result<()> {
    if receipt.id != operation.id || receipt.request != *operation {
        return Err(Fault::conflict(
            "executor returned another snapshot operation",
        ));
    }
    let Some(result) = &receipt.result else {
        if receipt.status.terminal() {
            return Err(Fault::conflict("terminal snapshot lacks a result"));
        }
        return Ok(());
    };
    if result.receipt.as_deref() != Some(&receipt.id)
        || (result.success && receipt.status != TurnStatus::Completed)
    {
        return Err(Fault::conflict(
            "snapshot result identity or status mismatch",
        ));
    }
    if !result.success {
        return Ok(());
    }
    let snapshot: GitSnapshot = serde_json::from_value(
        result
            .data
            .clone()
            .ok_or_else(|| Fault::conflict("snapshot data missing"))?,
    )
    .map_err(|_| Fault::conflict("executor returned invalid snapshot data"))?;
    if snapshot.schema_version != SCHEMA_VERSION
        || snapshot.request.base_revision != operation.arguments["base_revision"]
        || snapshot.request.head_revision != operation.arguments["head_revision"]
        || snapshot.request.expected_patch.is_some()
        || snapshot.expected_patch_sha256
            != operation.arguments["expected_patch"]
                .as_str()
                .map(|patch| crate::digest(patch.as_bytes()))
        || snapshot.patch.len() > 1024 * 1024
        || snapshot.patch_sha256 != crate::digest(snapshot.patch.as_bytes())
        || !is_commit_id(&snapshot.base_tree)
        || !is_commit_id(&snapshot.head_tree)
        || snapshot.base_tree.len() != snapshot.request.base_revision.len()
        || snapshot.head_tree.len() != snapshot.request.head_revision.len()
        || snapshot.files_changed < 0
        || snapshot.lines_added < 0
        || snapshot.lines_removed < 0
        || snapshot.binary_files < 0
        || snapshot.binary_files > snapshot.files_changed
        || snapshot.git_version.is_empty()
        || snapshot.git_version.len() > 256
        || snapshot.executable_sha256.len() != 64
        || !snapshot
            .executable_sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || receipt.started_at == 0
        || receipt.finished_at.is_none_or(|at| at < receipt.started_at)
    {
        return Err(Fault::conflict(
            "executor snapshot does not match its exact commit request",
        ));
    }
    Ok(())
}
