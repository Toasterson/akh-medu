//! A resumable checkout coordinator. Git effects belong to akhexecd receipts.
use crate::{
    config::{GitBootstrapProfile, validate_checkout},
    http::Client,
    sandbox::Sandboxes,
    store::key,
    types::*,
};
use bootstrap::{BootstrapRepository, RepositoryBootstrapResult};
use serde_json::{Value, json};

pub(crate) fn profile_fingerprint(
    profile: Option<&GitBootstrapProfile>,
    request: Option<&BootstrapRepository>,
) -> Result<Option<String>> {
    match (profile, request) {
        (None, None) => Ok(None),
        (Some(profile), Some(request)) => {
            validate_checkout(request)?;
            if !profile.repositories.contains(&request.checkout.repository) {
                return Err(Fault::new(
                    "forbidden",
                    "repository is outside the workspace profile",
                ));
            }
            Ok(Some(crate::digest(&serde_json::to_vec(profile).map_err(
                |_| Fault::invalid("bootstrap profile encoding"),
            )?)))
        }
        _ => Err(Fault::conflict(
            "workspace bootstrap request and profile must both be configured",
        )),
    }
}

fn steps(
    profile: &GitBootstrapProfile,
    request: &BootstrapRepository,
) -> Vec<(&'static str, &'static str, Value)> {
    let repository = &request.checkout.repository;
    let revision = &request.checkout.revision;
    let scheme = url::Url::parse(repository)
        .expect("validated repository")
        .scheme()
        .to_owned();
    let prefix = vec![
        profile.executable.to_string_lossy().into_owned(),
        "-c".into(),
        "core.hooksPath=/dev/null".into(),
        "-c".into(),
        "core.fsmonitor=false".into(),
        "-c".into(),
        "submodule.recurse=false".into(),
        "-c".into(),
        "protocol.allow=never".into(),
        "-c".into(),
        "protocol.file.allow=never".into(),
        "-c".into(),
        "protocol.ext.allow=never".into(),
        "-c".into(),
        format!("protocol.{scheme}.allow=always"),
    ];
    let environment = json!({
        "GIT_CONFIG_SYSTEM":"/dev/null",
        "GIT_CONFIG_GLOBAL":profile.config_file.as_ref().map(|p|p.to_string_lossy().into_owned()).unwrap_or_else(||"/dev/null".into()),
        "GIT_TERMINAL_PROMPT":"0", "GIT_ASKPASS":"/bin/false", "SSH_ASKPASS":"/bin/false"
    });
    let object_format = if revision.len() == 64 {
        "--object-format=sha256"
    } else {
        "--object-format=sha1"
    };
    let commands = [
        (
            "initialize",
            vec![
                "init",
                "--template=",
                object_format,
                "--initial-branch=akh-bootstrap",
                ".",
            ],
        ),
        (
            "origin",
            vec!["remote", "add", "origin", repository.as_str()],
        ),
        (
            "fetch",
            vec![
                "fetch",
                "--no-tags",
                "--no-recurse-submodules",
                "origin",
                revision.as_str(),
            ],
        ),
        ("object_type", vec!["cat-file", "-t", revision.as_str()]),
        ("checkout", vec!["checkout", "--detach", revision.as_str()]),
        ("verify", vec!["rev-parse", "--verify", "HEAD"]),
    ];
    let mut steps = vec![("empty", "list", json!({"path":"."}))];
    for (name, arguments) in commands {
        let mut argv = prefix.clone();
        argv.extend(arguments.into_iter().map(str::to_owned));
        steps.push((
            name,
            "exec_start",
            json!({"argv":argv,"env":environment,"timeout_ms":240000}),
        ));
    }
    steps
}

impl Sandboxes {
    pub(crate) async fn bootstrap_repository(
        &self,
        client: &Client,
        allocation: &Allocation,
        binding: &WorkspaceBinding,
        profile: &GitBootstrapProfile,
        request: &BootstrapRepository,
    ) -> Result<RepositoryBootstrapResult> {
        let fingerprint =
            profile_fingerprint(Some(profile), Some(request))?.expect("configured fingerprint");
        if allocation.bootstrap_profile_sha256.as_ref() != Some(&fingerprint) {
            return Err(Fault::conflict("workspace bootstrap profile changed"));
        }
        let complete_key = key("repository_bootstrap", &[&allocation.id]);
        if let Some(result) = self.store.get::<RepositoryBootstrapResult>(&complete_key)? {
            if result.request != *request
                || result.session_id != allocation.request.session_id
                || result.workspace_id != binding.id
                || result.generation != binding.generation
            {
                return Err(Fault::conflict(
                    "bootstrap result belongs to another workspace or checkout",
                ));
            }
            return Ok(result);
        }
        // Long network fetches must not lose their executor lease while the
        // allocation waits for a subsequent reconciliation tick.
        let lease: WorkspaceLease = client
            .post(
                &binding.executor,
                &format!("/v1/workspaces/{}/renew", crate::http::segment(&binding.id)),
                &RenewLease {
                    session_id: allocation.request.session_id.clone(),
                    generation: binding.generation,
                },
            )
            .await?;
        if lease.session_id != allocation.request.session_id
            || lease.workspace_id != binding.id
            || lease.generation != binding.generation
        {
            return Err(Fault::conflict(
                "bootstrap lease belongs to another workspace",
            ));
        }
        let mut receipts = Vec::new();
        for (step, operation, arguments) in steps(profile, request) {
            let operation = WorkspaceOperation {
                id: crate::digest(key("bootstrap_operation", &[&allocation.id, step]).as_bytes()),
                session_id: allocation.request.session_id.clone(),
                turn_id: "repository-bootstrap".into(),
                workspace_id: binding.id.clone(),
                generation: binding.generation,
                operation: operation.into(),
                arguments,
            };
            // Repeating this exact request only retrieves its executor receipt;
            // a lost HTTP response never permits a newly named Git operation.
            let receipt: OperationReceipt = client
                .post(&binding.executor, "/v1/operations", &operation)
                .await?;
            if receipt.id != operation.id || receipt.request != operation {
                return Err(Fault::conflict(
                    "executor returned another bootstrap receipt",
                ));
            }
            if receipt.status == TurnStatus::Unknown {
                return Err(Fault::new(
                    "unknown_outcome",
                    "repository operation is unconfirmed; reconcile its executor receipt",
                ));
            }
            if !receipt.status.terminal() {
                return Err(Fault::new("busy", "repository bootstrap is still running"));
            }
            let result = receipt
                .result
                .as_ref()
                .filter(|r| r.success && r.receipt.as_deref() == Some(&receipt.id))
                .ok_or_else(|| {
                    Fault::new(
                        "bootstrap_failed",
                        format!(
                            "repository {step} did not succeed; inspect executor receipt {}",
                            receipt.id
                        ),
                    )
                })?;
            if receipt.status != TurnStatus::Completed {
                return Err(Fault::conflict(
                    "bootstrap receipt is not a completed execution",
                ));
            }
            let data = result
                .data
                .as_ref()
                .ok_or_else(|| Fault::conflict("bootstrap execution has no structured result"))?;
            if step == "empty" {
                if !data["entries"].as_array().is_some_and(Vec::is_empty) {
                    return Err(Fault::conflict(
                        "repository bootstrap requires an empty workspace; existing files are preserved",
                    ));
                }
            } else {
                if data["exit_code"] != 0
                    || data["cancelled"] != false
                    || data["timed_out"] != false
                {
                    return Err(Fault::conflict(
                        "bootstrap process result is not a successful exit",
                    ));
                }
                if matches!(step, "object_type" | "verify") {
                    let expected = if step == "object_type" {
                        "commit"
                    } else {
                        &request.checkout.revision
                    };
                    if data["stdout"].as_str().map(str::trim) != Some(expected)
                        || data["truncated"] != false
                    {
                        return Err(Fault::conflict(
                            "Git did not verify the assigned commit object",
                        ));
                    }
                }
            }
            receipts.push(receipt.id);
        }
        let result = RepositoryBootstrapResult {
            request: request.clone(),
            session_id: allocation.request.session_id.clone(),
            workspace_id: binding.id.clone(),
            generation: binding.generation,
            verified_revision: request.checkout.revision.clone(),
            operation_receipts: receipts,
            at: now_ms(),
        };
        self.store
            .transaction(|tx| tx.put(&complete_key, &result))?;
        Ok(result)
    }
}
