//! Durable controller actions: no model turn, ambient repository, or blind push retry.
use crate::{
    config::Principal,
    server::SessionApp,
    session::{Sessions, append_event},
    store::{Transaction, key},
    types::*,
};
use publication::{
    GitPublication, PublishGitSnapshot, PublishSessionRepository, SessionPublication,
};
use repository::{GitSnapshot, is_commit_id};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Serialize, Deserialize, PartialEq)]
struct Intent {
    action: SessionPublication,
    binding: WorkspaceBinding,
    base_revision: String,
    head_revision: String,
}

fn intent_key(session: &str, request: &str) -> String {
    key("repository_publication", &[session, request])
}
fn load(tx: &Transaction<'_>, session: &str, request: &str) -> Result<Intent> {
    tx.get(&intent_key(session, request))?
        .ok_or_else(|| Fault::missing("publication intent not found"))
}
fn session_in(tx: &Transaction<'_>, id: &str) -> Result<SessionRecord> {
    tx.get(&key("session", &[id]))?
        .ok_or_else(|| Fault::missing("session not found"))
}
fn save(tx: &mut Transaction<'_>, intent: &Intent, kind: &str) -> Result<()> {
    let action = &intent.action;
    let id = &action.operation.session_id;
    tx.put(&intent_key(id, &action.request.request_id), intent)?;
    let mut session = session_in(tx, id)?;
    append_event(
        tx,
        &mut session,
        None,
        kind,
        json!({"request_id":action.request.request_id,"operation_id":action.operation.id,
            "status":action.receipt.as_ref().map(|r|r.status),
            "dispatched_at":action.dispatched_at,"cancelled_at":action.cancelled_at}),
    )?;
    Ok(())
}

/// Called in the session retirement transaction. Only actions never reserved
/// for transmission can be cancelled without external quiescence evidence.
pub(crate) fn cancel_unsent(tx: &mut Transaction<'_>, id: &str) -> Result<()> {
    for mut intent in tx.scan::<Intent>(&key("repository_publication", &[id]))? {
        if !intent.action.settled() && intent.action.dispatched_at.is_none() {
            intent.action.cancelled_at = Some(now_ms());
            save(tx, &intent, "repository_publication_cancelled")?;
        }
    }
    Ok(())
}
pub(crate) fn ensure_settled(tx: &Transaction<'_>, id: &str) -> Result<()> {
    if tx
        .scan::<Intent>(&key("repository_publication", &[id]))?
        .iter()
        .any(|intent| !intent.action.settled())
    {
        return Err(Fault::conflict(
            "reconcile outstanding repository publications before archiving",
        ));
    }
    Ok(())
}

impl Sessions {
    pub fn publication(
        &self,
        p: &Principal,
        id: &str,
        request: &str,
    ) -> Result<SessionPublication> {
        crate::http::role(p, "repository_publish")?;
        self.get(p, id)?;
        self.store
            .get::<Intent>(&intent_key(id, request))?
            .map(|intent| intent.action)
            .ok_or_else(|| Fault::missing("publication intent not found"))
    }

    /// Commit the exact snapshot and destination before any remote effect.
    pub fn prepare_publication(
        &self,
        p: &Principal,
        id: &str,
        request: &PublishSessionRepository,
    ) -> Result<SessionPublication> {
        crate::http::role(p, "repository_publish")?;
        self.get(p, id)?;
        validate_key(&request.request_id)?;
        validate_key(&request.snapshot_request_id)?;
        publication::validate_branch(&request.branch_ref)?;
        self.store.transaction(|tx| {
            if let Some(prior) = tx.get::<Intent>(&intent_key(id, &request.request_id))? {
                if prior.action.request != *request {
                    return Err(Fault::conflict(
                        "publication request ID already has another destination or snapshot",
                    ));
                }
                return Ok(prior.action);
            }
            let session = session_in(tx, id)?;
            if session.closed_at.is_some() || session.retirement.is_some() {
                return Err(Fault::conflict(
                    "session cannot start another repository publication",
                ));
            }
            self.check_repository_bootstrap(&session, true)?;
            let binding = session
                .workspace
                .as_ref()
                .ok_or_else(|| Fault::conflict("workspace is not ready"))?;
            let bootstrap = session
                .repository_bootstrap
                .as_ref()
                .and_then(|b| b.result.as_ref())
                .ok_or_else(|| {
                    Fault::conflict("publication requires a verified session checkout")
                })?;
            crate::session_bootstrap::validate_result(&session, binding, Some(bootstrap))?;
            let receipt: OperationReceipt = tx
                .get(&key(
                    "repository_snapshot_receipt",
                    &[id, &request.snapshot_request_id],
                ))?
                .ok_or_else(|| {
                    Fault::conflict("capture and verify the session snapshot before publication")
                })?;
            crate::session_snapshot::validate_receipt(&receipt.request, &receipt)?;
            let result = receipt
                .result
                .as_ref()
                .filter(|r| r.success && receipt.status == TurnStatus::Completed)
                .ok_or_else(|| {
                    Fault::conflict("publication snapshot did not complete successfully")
                })?;
            let snapshot: GitSnapshot =
                serde_json::from_value(result.data.clone().unwrap_or_default())
                    .map_err(|_| Fault::conflict("publication snapshot data invalid"))?;
            if receipt.id
                != crate::digest(
                    key(
                        "repository_snapshot_intent",
                        &[id, &request.snapshot_request_id],
                    )
                    .as_bytes(),
                )
                || receipt.request.operation != "git_snapshot"
                || receipt.request.session_id != id
                || receipt.request.workspace_id != binding.id
                || receipt.request.generation != binding.generation
                || snapshot.request.base_revision != bootstrap.verified_revision
            {
                return Err(Fault::conflict(
                    "publication snapshot differs from this session checkout",
                ));
            }
            let operation = WorkspaceOperation {
                id: crate::digest(intent_key(id, &request.request_id).as_bytes()),
                session_id: id.into(),
                turn_id: "repository-publication".into(),
                workspace_id: binding.id.clone(),
                generation: binding.generation,
                operation: "git_publish".into(),
                arguments: serde_json::to_value(PublishGitSnapshot {
                    snapshot_operation_id: receipt.id,
                    repository: bootstrap.request.checkout.repository.clone(),
                    branch_ref: request.branch_ref.clone(),
                })
                .map_err(|_| Fault::invalid("publication encoding"))?,
            };
            let intent = Intent {
                action: SessionPublication {
                    schema_version: SCHEMA_VERSION,
                    request: request.clone(),
                    operation,
                    dispatched_at: None,
                    cancelled_at: None,
                    receipt: None,
                },
                binding: binding.clone(),
                base_revision: snapshot.request.base_revision,
                head_revision: snapshot.request.head_revision,
            };
            save(tx, &intent, "repository_publication_queued")?;
            Ok(intent.action)
        })
    }
}

impl SessionApp {
    pub async fn publish_repository(
        &self,
        p: &Principal,
        id: &str,
        request: &PublishSessionRepository,
    ) -> Result<SessionPublication> {
        self.sessions.prepare_publication(p, id, request)?;
        self.drive_publication(id, &request.request_id).await
    }

    pub(crate) async fn reconcile_publications(&self, id: &str) -> Result<()> {
        // Runs even when no model turn remains active. One deferred publication
        // must not starve unrelated session actions.
        for intent in self
            .sessions
            .store
            .scan::<Intent>(&key("repository_publication", &[id]))?
        {
            if !intent.action.settled() {
                if let Err(error) = self
                    .drive_publication(id, &intent.action.request.request_id)
                    .await
                {
                    tracing::warn!(session=%id,code=%error.code,"repository publication reconciliation deferred");
                }
            }
        }
        Ok(())
    }

    async fn drive_publication(&self, id: &str, request: &str) -> Result<SessionPublication> {
        let intent: Intent = self
            .sessions
            .store
            .get(&intent_key(id, request))?
            .ok_or_else(|| Fault::missing("publication intent not found"))?;
        if intent.action.settled() {
            return Ok(intent.action);
        }
        let client = self.client.in_sandbox(&intent.binding.sandbox_id)?;
        let receipt: OperationReceipt = match client
            .get(
                &intent.binding.executor,
                &format!(
                    "/v1/operations/{}",
                    crate::http::segment(&intent.action.operation.id)
                ),
            )
            .await
        {
            Ok(receipt) => receipt,
            Err(error) if error.code == "not_found" => {
                let (current, send) = self.sessions.store.transaction(|tx| {
                    let mut current = load(tx, id, request)?;
                    if current.action.settled() || current.action.dispatched_at.is_some() {
                        // A sent request without a receipt is uncertain. Never
                        // infer that an absent receipt authorizes another push.
                        return Ok((current, false));
                    }
                    let session = session_in(tx, id)?;
                    if session.closed_at.is_some() || session.retirement.is_some() {
                        current.action.cancelled_at = Some(now_ms());
                        save(tx, &current, "repository_publication_cancelled")?;
                        return Ok((current, false));
                    }
                    if session.workspace.as_ref() != Some(&current.binding) {
                        return Err(Fault::conflict("publication workspace binding changed"));
                    }
                    current.action.dispatched_at = Some(now_ms());
                    save(tx, &current, "repository_publication_dispatching")?;
                    Ok((current, true))
                })?;
                if !send {
                    return Ok(current.action);
                }
                client
                    .post(
                        &current.binding.executor,
                        "/v1/repository-publications",
                        &current.action.operation,
                    )
                    .await?
            }
            Err(error) => return Err(error),
        };
        validate_receipt(&intent, &receipt)?;
        self.sessions.store.transaction(|tx| {
            let mut current = load(tx, id, request)?;
            if current.action.settled() || current.action.receipt.as_ref() == Some(&receipt) {
                return Ok(current.action);
            }
            if current
                .action
                .receipt
                .as_ref()
                .is_some_and(|r| r.status == TurnStatus::Unknown)
                && !receipt.status.terminal()
            {
                return Ok(current.action);
            }
            current.action.receipt = Some(receipt);
            save(tx, &current, "repository_publication_observed")?;
            Ok(current.action)
        })
    }

    /// Resolve a lost send only after an operator proves all possible senders
    /// quiescent. Existing executor receipts must be resolved at the executor.
    pub async fn reconcile_publication(
        &self,
        p: &Principal,
        id: &str,
        request: &str,
        resolution: &Reconciliation,
    ) -> Result<SessionPublication> {
        crate::http::role(p, "admin")?;
        self.sessions.get(p, id)?;
        let intent: Intent = self
            .sessions
            .store
            .get(&intent_key(id, request))?
            .ok_or_else(|| Fault::missing("publication intent not found"))?;
        crate::recovery::validate(p, &intent.action.operation, resolution)?;
        if resolution.status != TurnStatus::Cancelled
            || resolution.result.is_some()
            || resolution.output.is_some()
        {
            return Err(Fault::invalid(
                "a missing executor receipt can only be cancelled, never declared published",
            ));
        }
        let target = intent_key(id, request);
        if self
            .sessions
            .store
            .transaction(|tx| crate::recovery::already(tx, p, &target, resolution))?
        {
            return Ok(intent.action);
        }
        let client = self.client.in_sandbox(&intent.binding.sandbox_id)?;
        match client
            .get::<OperationReceipt>(
                &intent.binding.executor,
                &format!(
                    "/v1/operations/{}",
                    crate::http::segment(&intent.action.operation.id)
                ),
            )
            .await
        {
            Err(error) if error.code == "not_found" => {}
            Err(error) => return Err(error),
            Ok(_) => {
                return Err(Fault::conflict(
                    "reconcile the existing publication receipt at its executor",
                ));
            }
        }
        self.sessions.store.transaction(|tx| {
            let mut current = load(tx, id, request)?;
            if crate::recovery::already(tx, p, &target, resolution)? {
                return Ok(current.action);
            }
            if current.action.settled()
                || current.action.receipt.is_some()
                || current.action.dispatched_at.is_none()
            {
                return Err(Fault::conflict(
                    "publication does not have an unresolved send",
                ));
            }
            current.action.cancelled_at = Some(now_ms());
            crate::recovery::record(tx, p, &target, resolution)?;
            save(tx, &current, "repository_publication_reconciled")?;
            Ok(current.action)
        })
    }
}

fn validate_receipt(intent: &Intent, receipt: &OperationReceipt) -> Result<()> {
    if receipt.id != intent.action.operation.id || receipt.request != intent.action.operation {
        return Err(Fault::conflict(
            "executor returned another publication operation",
        ));
    }
    let Some(result) = &receipt.result else {
        return if receipt.status.terminal() {
            Err(Fault::conflict("terminal publication lacks a result"))
        } else {
            Ok(())
        };
    };
    if result.receipt.as_deref() != Some(&receipt.id)
        || result.success != (receipt.status == TurnStatus::Completed)
    {
        return Err(Fault::conflict(
            "publication result identity or status mismatch",
        ));
    }
    if !result.success {
        return Ok(());
    }
    let published: GitPublication = serde_json::from_value(result.data.clone().unwrap_or_default())
        .map_err(|_| Fault::conflict("invalid executor publication data"))?;
    let expected: PublishGitSnapshot =
        serde_json::from_value(intent.action.operation.arguments.clone())
            .map_err(|_| Fault::conflict("invalid stored publication request"))?;
    if published.schema_version != SCHEMA_VERSION
        || published.request != expected
        || published.base_revision != intent.base_revision
        || published.head_revision != intent.head_revision
        || published.observed_revision != intent.head_revision
        || published
            .previous_revision
            .as_ref()
            .is_some_and(|r| !is_commit_id(r) || r.len() != intent.head_revision.len())
        || published.already_present
            != (published.previous_revision.as_ref() == Some(&intent.head_revision))
        || (published.already_present && published.push_acknowledged)
        || published.policy_sha256.len() != 64
        || !published
            .policy_sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        || receipt.started_at == 0
        || published.at < receipt.started_at
        || receipt.finished_at.is_none_or(|at| at < published.at)
    {
        return Err(Fault::conflict(
            "publication observation differs from its verified snapshot or destination",
        ));
    }
    Ok(())
}
