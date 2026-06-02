//! Peer akh-medu instance coordination (Phase 36d).
//!
//! The [`PeerRegistry`] tracks known peer instances, their capabilities,
//! and historical performance metrics. [`delegate_to_peer`] sends a
//! `DelegateTask` message and awaits a `TaskResult` response.
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
    /// HTTP endpoint (e.g. "http://research-akh.local:3001").
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
// Delegation via HTTP
// ---------------------------------------------------------------------------

/// Delegate a task to a peer akh-medu instance.
///
/// Sends a delegation request to the peer's `/delegation/incoming` endpoint
/// and polls for the result. The peer is expected to return a
/// `DelegationCallbackPayload` on its result channel.
///
/// **Current status**: HTTP peer delegation requires the peer's akhomed server
/// to expose a delegation API endpoint (planned in the daemon integration).
/// This function provides the scaffold; until both sides are wired, it returns
/// `DelegationError::PeerUnreachable`.
#[cfg(feature = "daemon")]
pub async fn delegate_to_peer(
    endpoint: &str,
    task: &DelegationTask,
    id: DelegationId,
) -> DelegationResult<DelegationOutcome> {
    // Build the delegation request payload.
    let payload = serde_json::json!({
        "task_id": id.0.to_string(),
        "description": task.description,
        "required_capabilities": task.required_capabilities,
        "context_triples": task.context_triples,
        "goal_id": task.goal_id,
    });

    let url = format!("{endpoint}/delegation/incoming");
    let start = Instant::now();

    // Attempt the HTTP call if reqwest is available.
    // The delegation feature doesn't enable reqwest directly; fall back to
    // PeerUnreachable so the manager can try a different target.
    let _ = (payload, url, start, id);
    Err(DelegationError::PeerUnreachable {
        endpoint: endpoint.to_string(),
    })
}

/// Not available in non-daemon builds.
#[cfg(not(feature = "daemon"))]
pub async fn delegate_to_peer(
    endpoint: &str,
    _task: &DelegationTask,
    _id: DelegationId,
) -> DelegationResult<DelegationOutcome> {
    Err(DelegationError::PeerUnreachable {
        endpoint: endpoint.to_string(),
    })
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
