//! Peer akh-medu instance coordination (Phase 36d).
//!
//! The [`PeerRegistry`] tracks known peer instances, their capabilities,
//! and historical performance metrics. [`delegate_to_peer`] opens an ACP
//! session on the peer's Unix/TCP endpoint (ADR-061: the daemon speaks only
//! ACP — there is no HTTP surface) and drives one prompt turn with the task
//! description, collecting the streamed events into the delegation outcome.
//!
//! Peer discovery is currently explicit (configured endpoints). Future:
//! mDNS/DNS-SD auto-discovery on LAN.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::error::{DelegationError, DelegationResult};
use super::types::{DelegationId, DelegationOutcome, DelegationOutput, DelegationTask};
use crate::agent::delegation::PeerDelegationConfig;

// ---------------------------------------------------------------------------
// PeerInfo
// ---------------------------------------------------------------------------

/// Metadata about a known peer akh-medu instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    /// ACP address: `host:port` for a TCP peer, an absolute socket path (or
    /// `unix:<path>`) for a same-host peer (ADR-061 — no HTTP surface).
    pub endpoint: String,
    /// Capabilities advertised by this peer.
    pub capabilities: Vec<String>,
    /// Skills currently loaded on the peer.
    pub active_skills: Vec<String>,
    /// Unix timestamp of the last advertisement received.
    pub last_advertisement: u64,
    /// Current load fraction (0.0–1.0).
    pub load: f32,
    /// Historical success rate (0.0–1.0).
    pub success_rate: f32,
    /// Average response time in milliseconds.
    pub avg_response_ms: f64,
    /// Number of tasks successfully delegated.
    pub tasks_completed: u64,
    /// Available capacity (from last advertisement).
    pub available_capacity: usize,
}

impl PeerInfo {
    fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            capabilities: Vec::new(),
            active_skills: Vec::new(),
            last_advertisement: 0,
            load: 0.0,
            success_rate: 1.0,
            avg_response_ms: 1000.0,
            tasks_completed: 0,
            available_capacity: 3,
        }
    }

    /// Selection score: higher is better. Combines load, success rate, and
    /// response time using the formula from the plan:
    /// `(1 - load) × success_rate × (1000 / avg_response_ms)`
    pub fn score(&self) -> f64 {
        if self.available_capacity == 0 || self.load >= 0.9 {
            return 0.0;
        }
        (1.0 - self.load as f64) * self.success_rate as f64 * (1000.0 / self.avg_response_ms)
    }

    /// Whether this peer advertises all of the required capabilities.
    pub fn has_capabilities(&self, required: &[String]) -> bool {
        required
            .iter()
            .all(|cap| self.capabilities.iter().any(|c| c == cap))
    }

    /// Update from a `CapabilityAdvertisement` message.
    pub fn update_from_advertisement(
        &mut self,
        capabilities: Vec<String>,
        active_skills: Vec<String>,
        load: f32,
        available_capacity: usize,
    ) {
        self.capabilities = capabilities;
        self.active_skills = active_skills;
        self.load = load;
        self.available_capacity = available_capacity;
        self.last_advertisement = now_secs();
    }

    /// Record a successful task completion and update the average response time.
    pub fn record_success(&mut self, wall_time_ms: u64) {
        self.tasks_completed += 1;
        // Exponential moving average with α = 0.2.
        self.avg_response_ms =
            0.8 * self.avg_response_ms + 0.2 * wall_time_ms as f64;
        // Decay success rate toward 1.0 over time.
        self.success_rate = (self.success_rate * 0.9 + 0.1).min(1.0);
    }

    /// Record a failed task and decrease the success rate.
    pub fn record_failure(&mut self) {
        self.success_rate = (self.success_rate * 0.9).max(0.0);
    }
}

// ---------------------------------------------------------------------------
// PeerRegistry
// ---------------------------------------------------------------------------

/// Registry of known peer akh-medu instances.
pub struct PeerRegistry {
    peers: Mutex<HashMap<String, PeerInfo>>,
}

impl PeerRegistry {
    /// Create a new registry from config, pre-populating configured endpoints.
    pub fn new(config: &PeerDelegationConfig) -> Self {
        let mut peers = HashMap::new();
        for endpoint in &config.endpoints {
            peers.insert(endpoint.clone(), PeerInfo::new(endpoint.clone()));
        }
        Self {
            peers: Mutex::new(peers),
        }
    }

    /// Register or update a peer.
    pub fn upsert(&self, endpoint: impl Into<String>) {
        let ep = endpoint.into();
        let mut peers = self.peers.lock().unwrap();
        peers.entry(ep.clone()).or_insert_with(|| PeerInfo::new(ep));
    }

    /// Select the best peer for the given required capabilities.
    ///
    /// Filters by capabilities, excludes overloaded peers (load ≥ 0.9), and
    /// ranks by score (see [`PeerInfo::score`]).
    pub fn best_peer(&self, required_capabilities: &[String]) -> Option<String> {
        let peers = self.peers.lock().ok()?;
        peers
            .values()
            .filter(|p| p.has_capabilities(required_capabilities) && p.score() > 0.0)
            .max_by(|a, b| {
                a.score()
                    .partial_cmp(&b.score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.endpoint.clone())
    }

    /// List all known peers.
    pub fn list(&self) -> Vec<PeerInfo> {
        self.peers
            .lock()
            .map(|p| p.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Update a peer from a CapabilityAdvertisement.
    pub fn handle_advertisement(
        &self,
        endpoint: &str,
        capabilities: Vec<String>,
        active_skills: Vec<String>,
        load: f32,
        available_capacity: usize,
    ) {
        let mut peers = self.peers.lock().unwrap();
        let info = peers
            .entry(endpoint.to_string())
            .or_insert_with(|| PeerInfo::new(endpoint));
        info.update_from_advertisement(capabilities, active_skills, load, available_capacity);
    }

    /// Record success for a peer delegation.
    pub fn record_success(&self, endpoint: &str, wall_time_ms: u64) {
        let mut peers = self.peers.lock().unwrap();
        if let Some(info) = peers.get_mut(endpoint) {
            info.record_success(wall_time_ms);
        }
    }

    /// Record failure for a peer delegation.
    pub fn record_failure(&self, endpoint: &str) {
        let mut peers = self.peers.lock().unwrap();
        if let Some(info) = peers.get_mut(endpoint) {
            info.record_failure();
        }
    }

    /// Number of known peers.
    pub fn len(&self) -> usize {
        self.peers.lock().map(|p| p.len()).unwrap_or(0)
    }

    /// Whether the registry has no peers.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for PeerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PeerRegistry")
            .field("peer_count", &self.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Delegation via ACP (ADR-061: the daemon speaks only ACP over Unix/TCP)
// ---------------------------------------------------------------------------

/// Bound on one peer delegation turn. Deliberately matches the `AkhMeduPeer`
/// timeout the `DelegationManager` records when dispatching
/// (`DelegationTargetKind::AkhMeduPeer { .. } => 60` in delegation/mod.rs) so
/// the transport gives up at the same horizon the tracker does.
const PEER_TURN_TIMEOUT: Duration = Duration::from_secs(60);

/// The ACP address of a peer, parsed from the configured endpoint string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerAcpAddress {
    /// TCP `host:port` — a remote peer daemon's ACP listener.
    Tcp(String),
    /// Unix domain socket path — a same-host peer (or self-delegation).
    Unix(std::path::PathBuf),
}

/// Parse a configured peer endpoint into an ACP address. Accepted forms:
/// `host:port`, an absolute socket path, `unix:<path>`, and the legacy
/// `http(s)://host:port` shape (scheme stripped, any trailing path dropped).
/// Returns `None` for strings that name no usable transport — the caller
/// reports an unreachable peer rather than guessing.
pub fn peer_acp_address(endpoint: &str) -> Option<PeerAcpAddress> {
    if let Some(path) = endpoint.strip_prefix("unix:") {
        return Some(PeerAcpAddress::Unix(std::path::PathBuf::from(path)));
    }
    for scheme in ["http://", "https://"] {
        if let Some(rest) = endpoint.strip_prefix(scheme) {
            let hostport = rest.split('/').next().unwrap_or(rest);
            return if hostport.contains(':') {
                Some(PeerAcpAddress::Tcp(hostport.to_string()))
            } else {
                None
            };
        }
    }
    if endpoint.starts_with('/') {
        return Some(PeerAcpAddress::Unix(std::path::PathBuf::from(endpoint)));
    }
    if endpoint.contains(':') {
        return Some(PeerAcpAddress::Tcp(endpoint.to_string()));
    }
    None
}

/// Delegate a task to a peer akh-medu instance over ACP.
///
/// Opens an ACP connection to the peer (ADR-061: the daemon speaks only ACP
/// over Unix socket/TCP — there is no HTTP surface), creates a session in the
/// target workspace (`session/new` with the `akh-ws:<name>` cwd sentinel; the
/// fresh session carries its own isolated `SessionContext`, per WI-1), sends
/// one `session/prompt` carrying the task description, and collects the
/// streamed `session/update` events into the outcome's visible text. Pure
/// internal akhomed-to-akhomed ACP — it never touches Anima.
///
/// `task.target_workspace` selects the peer-side workspace; `None` means the
/// peer's default workspace.
#[cfg(feature = "delegation")]
pub async fn delegate_to_peer(
    endpoint: &str,
    task: &DelegationTask,
    id: DelegationId,
) -> DelegationResult<DelegationOutcome> {
    let start = Instant::now();
    let workspace = task.target_workspace.as_deref().unwrap_or("default");
    let client_name = format!("akh-delegation-{}", id.0);
    let prompt = task.description.clone();

    let address = peer_acp_address(endpoint).ok_or_else(|| {
        tracing::warn!(endpoint = %endpoint, "peer endpoint is not a usable ACP address");
        DelegationError::PeerUnreachable { endpoint: endpoint.to_string() }
    })?;

    tracing::info!(
        delegation_id = id.0,
        peer_endpoint = %endpoint,
        workspace = %workspace,
        ?address,
        "dispatching peer delegation over ACP",
    );

    // One bounded turn: connect → session/new → session/prompt → drain the
    // streamed updates → disconnect.
    let turn = match address {
        PeerAcpAddress::Unix(path) => tokio::time::timeout(
            PEER_TURN_TIMEOUT,
            crate::acp::run_prompt_turn_unix(&path, &client_name, workspace, &prompt),
        )
        .await,
        PeerAcpAddress::Tcp(addr) => tokio::time::timeout(
            PEER_TURN_TIMEOUT,
            crate::acp::run_prompt_turn_tcp(&addr, &client_name, workspace, &prompt),
        )
        .await,
    };

    let text = match turn {
        Err(_elapsed) => {
            tracing::warn!(
                delegation_id = id.0,
                endpoint = %endpoint,
                timeout_secs = PEER_TURN_TIMEOUT.as_secs(),
                "peer ACP turn timed out",
            );
            return Err(DelegationError::PeerTurnFailed {
                endpoint: endpoint.to_string(),
                reason: format!("turn exceeded {}s", PEER_TURN_TIMEOUT.as_secs()),
            });
        }
        Ok(Err(e)) => {
            tracing::warn!(
                delegation_id = id.0,
                endpoint = %endpoint,
                error = %e,
                "peer ACP turn failed",
            );
            // Connect-phase failures mean the peer is not there; anything
            // else is a session/turn failure on a reachable peer.
            let unreachable = matches!(
                e,
                crate::acp::AcpError::Connect { .. }
                    | crate::acp::AcpError::ConnectTimeout { .. }
            );
            return Err(if unreachable {
                DelegationError::PeerUnreachable { endpoint: endpoint.to_string() }
            } else {
                DelegationError::PeerTurnFailed {
                    endpoint: endpoint.to_string(),
                    reason: e.to_string(),
                }
            });
        }
        Ok(Ok(text)) => text,
    };

    // WI-221 #4: a turn that produced no visible text is a failure, never a
    // silent success with exit_code 0 and an empty result.
    if text.trim().is_empty() {
        tracing::warn!(
            delegation_id = id.0,
            endpoint = %endpoint,
            "peer ACP turn returned an empty response",
        );
        return Err(DelegationError::PeerTurnFailed {
            endpoint: endpoint.to_string(),
            reason: "peer returned an empty turn response".into(),
        });
    }

    let wall_time = start.elapsed();
    tracing::info!(
        delegation_id = id.0,
        endpoint = %endpoint,
        workspace = %workspace,
        chars = text.len(),
        wall_time_ms = wall_time.as_millis() as u64,
        "peer delegation completed",
    );

    Ok(DelegationOutcome {
        id,
        stdout: text,
        // The peer's knowledge graph is separate; no structured triples flow
        // back over the chat turn.
        structured: None,
        wall_time,
        tokens_consumed: None,
        cost_usd: None,
        exit_code: Some(0),
        goal_id: task.goal_id,
        error: None,
        target: None,
    })
}

/// Not available without the `delegation` feature — the real path above needs
/// the `acp` transport that `delegation` pulls in, and `server` builds
/// include `daemon` but not `delegation`, so this stub keeps the
/// `daemon`-only build compiling. Returns a clean `PeerUnreachable` so the
/// daemon tick records the failure.
#[cfg(not(feature = "delegation"))]
pub async fn delegate_to_peer(
    endpoint: &str,
    _task: &DelegationTask,
    _id: DelegationId,
) -> DelegationResult<DelegationOutcome> {
    Err(DelegationError::PeerUnreachable { endpoint: endpoint.to_string() })
}

// ---------------------------------------------------------------------------
// Inbound delegation handler
// ---------------------------------------------------------------------------

/// Parse a callback payload received from a peer that completed a task.
#[derive(Debug, Deserialize)]
pub struct DelegationCallbackPayload {
    pub task_id: String,
    pub status: String,
    pub summary: Option<String>,
    pub triples: Option<Vec<crate::skills::LabelTriple>>,
    pub files_modified: Option<Vec<String>>,
    pub wall_time_ms: Option<u64>,
    pub error: Option<String>,
}

impl DelegationCallbackPayload {
    /// Convert to a `DelegationOutcome` for processing by the daemon.
    pub fn into_outcome(self, goal_id: u64) -> DelegationOutcome {
        let structured = if self.status == "success" || self.status == "partial" {
            Some(DelegationOutput {
                status: self.status.clone(),
                summary: self.summary.clone().unwrap_or_default(),
                triples: self.triples.clone().unwrap_or_default(),
                files_modified: self.files_modified.clone().unwrap_or_default(),
                follow_up: None,
            })
        } else {
            None
        };

        DelegationOutcome {
            id: DelegationId(0), // Caller resolves from task_id
            stdout: self.summary.unwrap_or_default(),
            structured,
            wall_time: Duration::from_millis(self.wall_time_ms.unwrap_or(0)),
            tokens_consumed: None,
            cost_usd: None,
            exit_code: None,
            goal_id,
            error: if self.status == "failed" {
                self.error
            } else {
                None
            },
            target: None,
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
