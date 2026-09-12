//! Session-owner repository intent; allocation and execution evidence remain separate.
use crate::{
    config::{AkhProfile, Principal, validate_checkout},
    session::Sessions,
    store::key,
    types::*,
};
use bootstrap::*;

pub(crate) fn initial(profile: &AkhProfile) -> Result<Option<SessionRepositoryBootstrap>> {
    if profile.bootstrap_repositories.is_empty() {
        return Ok(None);
    }
    Ok(Some(SessionRepositoryBootstrap {
        policy_sha256: crate::digest(
            &serde_json::to_vec(&profile.bootstrap_repositories)
                .map_err(|_| Fault::invalid("bootstrap policy encoding"))?,
        ),
        receipt: None,
        result: None,
    }))
}

impl Sessions {
    pub fn prepare_repository(
        &self,
        principal: &Principal,
        id: &str,
        request: &BootstrapRepository,
    ) -> Result<RepositoryBootstrapReceipt> {
        crate::http::role(principal, "session_bootstrap")?;
        validate_checkout(request)?;
        self.store.transaction(|tx| {
            let mut session: SessionRecord = tx
                .get(&key("session", &[id]))?
                .ok_or_else(|| Fault::missing("session not found"))?;
            if session.principal != principal.id && !principal.has_role("admin") {
                return Err(Fault::new(
                    "forbidden",
                    "session belongs to another principal",
                ));
            }
            let slot = session
                .repository_bootstrap
                .as_ref()
                .ok_or_else(|| Fault::conflict("session does not require repository bootstrap"))?;
            if let Some(old) = &slot.receipt {
                if old.request != *request || old.principal != principal.id {
                    return Err(Fault::conflict(
                        "repository bootstrap already has another intent",
                    ));
                }
                return Ok(old.clone());
            }
            if session.closed_at.is_some()
                || session.retirement.is_some()
                || session.workspace.is_some()
            {
                return Err(Fault::conflict(
                    "session can no longer prepare a new repository",
                ));
            }
            let profile = self
                .config
                .akhs
                .get(&session.akh)
                .ok_or_else(|| Fault::missing("Akh profile not found"))?;
            if initial(profile)?.as_ref().map(|s| &s.policy_sha256) != Some(&slot.policy_sha256)
                || !profile
                    .bootstrap_repositories
                    .contains(&request.checkout.repository)
            {
                return Err(Fault::new(
                    "forbidden",
                    "repository is outside the session's pinned bootstrap policy",
                ));
            }
            let receipt = RepositoryBootstrapReceipt {
                schema_version: SCHEMA_VERSION,
                session_id: id.into(),
                request: request.clone(),
                principal: principal.id.clone(),
                at: now_ms(),
            };
            session
                .repository_bootstrap
                .as_mut()
                .expect("checked slot")
                .receipt = Some(receipt.clone());
            crate::session::append_event(
                tx,
                &mut session,
                None,
                "repository_bootstrap_prepared",
                serde_json::to_value(&receipt)
                    .map_err(|_| Fault::invalid("bootstrap receipt encoding"))?,
            )?;
            Ok(receipt)
        })
    }

    pub(crate) fn check_repository_bootstrap(
        &self,
        session: &SessionRecord,
        require_ready: bool,
    ) -> Result<Option<BootstrapRepository>> {
        let profile = self
            .config
            .akhs
            .get(&session.akh)
            .ok_or_else(|| Fault::missing("Akh profile not found"))?;
        if initial(profile)?.as_ref().map(|s| &s.policy_sha256)
            != session
                .repository_bootstrap
                .as_ref()
                .map(|s| &s.policy_sha256)
        {
            return Err(Fault::conflict(
                "session repository bootstrap policy changed",
            ));
        }
        let Some(slot) = &session.repository_bootstrap else {
            return Ok(None);
        };
        let receipt = slot.receipt.as_ref().ok_or_else(|| {
            Fault::new(
                "bootstrap_pending",
                "repository bootstrap intent has not been provided",
            )
        })?;
        if require_ready && slot.result.is_none() {
            return Err(Fault::new(
                "bootstrap_pending",
                "executor has not verified the repository checkout",
            ));
        }
        Ok(Some(receipt.request.clone()))
    }
}

pub(crate) fn validate_result(
    session: &SessionRecord,
    binding: &WorkspaceBinding,
    result: Option<&RepositoryBootstrapResult>,
) -> Result<()> {
    match (&session.repository_bootstrap, result) {
        (None, None) => Ok(()),
        (Some(slot), Some(result)) => {
            if slot
                .receipt
                .as_ref()
                .is_none_or(|r| r.request != result.request)
                || result.session_id != session.id
                || result.workspace_id != binding.id
                || result.generation != binding.generation
                || result.verified_revision != result.request.checkout.revision
                || result.at == 0
                || result.operation_receipts.len() != 7
                || result.operation_receipts.iter().any(|id| id.is_empty())
                || result
                    .operation_receipts
                    .iter()
                    .collect::<std::collections::HashSet<_>>()
                    .len()
                    != 7
                || slot.result.as_ref().is_some_and(|old| old != result)
            {
                return Err(Fault::conflict(
                    "bootstrap evidence differs from the assigned checkout or workspace",
                ));
            }
            Ok(())
        }
        _ => Err(Fault::conflict(
            "workspace bootstrap evidence does not match the session requirement",
        )),
    }
}
