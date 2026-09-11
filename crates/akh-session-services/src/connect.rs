//! Per-user connection offers. Upstream applications own their saved connections.
use crate::{config::Endpoint, helper::atomic_json, http::segment, types::*};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf, time::Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConnectConfig {
    pub sessions: Endpoint,
    pub state_file: PathBuf,
    pub client_kind: String,
    pub client_profile: String,
    /// Each argument is expanded literally; `{url}` is never passed through a shell.
    pub open_command: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Offer {
    pub descriptor: WorkspaceAccessDescriptor,
    pub client_profile: String,
    pub offered_at: u64,
    pub state: ConnectionState,
}
pub fn identity(descriptor: &WorkspaceAccessDescriptor, client_profile: &str) -> String {
    crate::digest(
        crate::store::key(
            "connection",
            &[
                &descriptor.principal,
                client_profile,
                &descriptor.environment_id,
            ],
        )
        .as_bytes(),
    )
}

pub fn pairing_url(output: &str, endpoint: &str) -> Result<String> {
    let endpoint =
        url::Url::parse(endpoint).map_err(|_| Fault::invalid("invalid advertised endpoint"))?;
    let mut candidates = vec![];
    for word in output.split_whitespace() {
        let Ok(url) = url::Url::parse(word) else {
            continue;
        };
        if !matches!(url.scheme(), "http" | "https") {
            continue;
        }
        let direct = url.origin() == endpoint.origin();
        let hosted = url.origin().ascii_serialization() == "https://app.t3.codes"
            && url.path() == "/pair"
            && url
                .query_pairs()
                .find(|(k, _)| k == "host")
                .and_then(|(_, v)| url::Url::parse(&v).ok())
                .is_some_and(|host| host.origin() == endpoint.origin());
        if !direct && !hosted {
            continue;
        }
        // The upstream hosted flow keeps its one-time secret in the fragment.
        if hosted && (url.fragment().is_none() || url.query_pairs().any(|(k, _)| k == "token")) {
            continue;
        }
        candidates.push(url.to_string());
    }
    candidates.sort();
    candidates.dedup();
    if candidates.len() != 1 {
        return Err(Fault::new(
            "pairing_unavailable",
            "upstream pairing output did not contain exactly one authorized connection link",
        ));
    }
    Ok(candidates.remove(0))
}

pub async fn run(config: ConnectConfig, session: &str, watch: bool, reconnect: bool) -> Result<()> {
    config.sessions.validate()?;
    validate_key(&config.client_profile)?;
    if !config.state_file.is_absolute() {
        return Err(Fault::invalid("connection state_file must be absolute"));
    }
    let parent = config
        .state_file
        .parent()
        .ok_or_else(|| Fault::invalid("state_file needs a parent directory"))?;
    std::fs::create_dir_all(parent).map_err(|e| Fault::invalid(e.to_string()))?;
    use std::os::{fd::AsRawFd, unix::fs::OpenOptionsExt};
    let _lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(config.state_file.with_extension("lock"))
        .map_err(|e| Fault::invalid(e.to_string()))?;
    if unsafe { libc::flock(_lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(Fault::conflict(
            "a connection helper already owns this client profile",
        ));
    }
    let mut visited = std::collections::HashSet::new();
    let client = config.sessions.http_client()?;
    loop {
        let response = client
            .get(
                config
                    .sessions
                    .route(&format!("/v1/sessions/{}/access", segment(session))),
            )
            .bearer_auth(config.sessions.token()?)
            .send()
            .await
            .map_err(|_| Fault::new("transport", "session access request failed"))?;
        if !response.status().is_success() {
            return Err(Fault::new(
                "transport",
                format!(
                    "session access returned HTTP {}",
                    response.status().as_u16()
                ),
            ));
        }
        let descriptors: Vec<WorkspaceAccessDescriptor> = response
            .json()
            .await
            .map_err(|_| Fault::invalid("invalid access descriptor response"))?;
        let mut offers: BTreeMap<String, Offer> = if config.state_file.exists() {
            serde_json::from_slice(
                &std::fs::read(&config.state_file).map_err(|e| Fault::invalid(e.to_string()))?,
            )
            .map_err(|e| Fault::invalid(e.to_string()))?
        } else {
            BTreeMap::new()
        };
        for descriptor in descriptors
            .into_iter()
            .filter(|d| d.client_kind == config.client_kind && d.state == ConnectionState::Ready)
        {
            let key = identity(&descriptor, &config.client_profile);
            let existing = offers.get(&key);
            if visited.contains(&key) || (existing.is_some() && !reconnect) {
                continue;
            }
            let url = if config.client_kind == "t3" && existing.is_none() {
                let response = client
                    .post(config.sessions.route(&format!(
                        "/v1/sessions/{}/services/{}/actions/pair",
                        segment(session),
                        segment(&descriptor.id)
                    )))
                    .bearer_auth(config.sessions.token()?)
                    .send()
                    .await
                    .map_err(|_| Fault::new("transport", "upstream pairing request failed"))?;
                if !response.status().is_success() {
                    return Err(Fault::new(
                        "pairing_unavailable",
                        "upstream could not create a pairing link",
                    ));
                }
                let value: Value = response
                    .json()
                    .await
                    .map_err(|_| Fault::invalid("invalid pairing response"))?;
                pairing_url(
                    value["output"]
                        .as_str()
                        .ok_or_else(|| Fault::invalid("pairing command has no output"))?,
                    &descriptor.endpoint,
                )?
            } else {
                descriptor.endpoint.clone()
            };
            let (command, arguments) = config
                .open_command
                .split_first()
                .ok_or_else(|| Fault::invalid("open_command is required"))?;
            let offer = Offer {
                descriptor: descriptor.clone(),
                client_profile: config.client_profile.clone(),
                offered_at: now_ms(),
                state: ConnectionState::ConnectionOffered,
            };
            // Store only the public descriptor before handing the URL to an upstream client.
            // Pairing secrets never enter this file or the session journal.
            visited.insert(key.clone());
            offers.insert(key, offer);
            atomic_json(&config.state_file, &offers)?;
            let result = tokio::process::Command::new(command)
                .args(arguments.iter().map(|a| {
                    a.replace("{url}", &url)
                        .replace("{workspace}", &descriptor.workspace_id)
                }))
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await
                .map_err(|_| Fault::new("client_launch", "client command could not start"))?;
            if !result.success() {
                return Err(Fault::new(
                    "client_launch",
                    "client command failed; use --reconnect to retry the existing environment",
                ));
            }
            println!(
                "Offered {} to client profile {}. Upstream pairing/connection confirmation remains client-owned.",
                descriptor.environment_id, config.client_profile
            );
        }
        if !watch {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hosted_pairing_keeps_secret_on_the_correct_origin() {
        let good = "https://app.t3.codes/pair?host=https%3A%2F%2Feditor.example#token=one-time";
        assert_eq!(pairing_url(good, "https://editor.example").unwrap(), good);
        assert!(
            pairing_url(
                "https://app.t3.codes/pair?host=https%3A%2F%2Feditor.example&token=secret",
                "https://editor.example"
            )
            .is_err()
        );
        assert!(
            pairing_url(
                "https://attacker.example/pair#token=secret",
                "https://editor.example"
            )
            .is_err()
        );
    }
    #[test]
    fn endpoint_changes_do_not_create_another_environment_offer() {
        let mut descriptor:WorkspaceAccessDescriptor=serde_json::from_value(serde_json::json!({"id":"service","principal":"user","session_id":"session","workspace_id":"workspace","environment_id":"upstream-id","generation":1,"client_kind":"t3","endpoint":"https://one.example","version":"pinned","central_agent_attachment":false,"state":"ready"})).unwrap();
        let first = identity(&descriptor, "desktop");
        descriptor.endpoint = "https://two.example".into();
        assert_eq!(first, identity(&descriptor, "desktop"));
        assert_ne!(first, identity(&descriptor, "browser"));
    }
}
