//! HTTP client wrapper around Seshat's ELIZA scripts REST API.
//!
//! `SeshatScriptStore` provides three operations — `load`, `save`, `list` —
//! backed by a 60-second TTL cache.  When Seshat is unreachable, `load`
//! degrades gracefully: it returns a stale cache hit (with a warning) if one
//! exists, or propagates the transport error otherwise.

use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::eliza::cache::ScriptCache;
use crate::eliza::error::{ElizaError, ElizaResult};

const DEFAULT_TIMEOUT_SECS: u64 = 10;

// ── Wire types — mirror Seshat's `/scripts` REST API ───────────────────────

/// Brief metadata for a script stored in Seshat.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScriptInfo {
    pub name: String,
    /// Seshat document UUID; serves as a revision identifier.
    pub revision: String,
    pub saved_at: String,
}

/// Full script content returned by `GET /scripts/{name}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScriptContent {
    pub name: String,
    pub content: String,
    pub revision: String,
    pub saved_at: String,
}

/// Response returned by `POST /scripts/{name}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SaveResult {
    pub revision: String,
    pub saved_at: String,
}

#[derive(Serialize)]
struct SaveRequest<'a> {
    content: &'a str,
}

// ── SeshatScriptStore ───────────────────────────────────────────────────────

/// HTTP client for ELIZA KDL scripts stored in the Seshat document service.
///
/// The store wraps a `ureq::Agent` (sync HTTP) and an in-memory TTL cache.
/// Multiple callers can share one store via `Clone` — clones share the same
/// underlying cache through an `Arc<RwLock<ScriptCache>>`.
pub struct SeshatScriptStore {
    base_url: String,
    cache: Arc<RwLock<ScriptCache>>,
    agent: Arc<ureq::Agent>,
}

impl Clone for SeshatScriptStore {
    fn clone(&self) -> Self {
        Self {
            base_url: self.base_url.clone(),
            cache: Arc::clone(&self.cache),
            agent: Arc::clone(&self.agent),
        }
    }
}

impl SeshatScriptStore {
    /// Create a new store targeting `base_url` (e.g. `"http://seshat.lab.home:9090"`).
    pub fn new(base_url: impl Into<String>) -> Self {
        let agent = Arc::new(
            ureq::AgentBuilder::new()
                .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
                .build(),
        );
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            cache: Arc::new(RwLock::new(ScriptCache::new())),
            agent,
        }
    }

    /// Override the shared cache (useful to wire the reloader and store to the same cache).
    pub fn with_cache(mut self, cache: Arc<RwLock<ScriptCache>>) -> Self {
        self.cache = cache;
        self
    }

    /// Return a clone of the shared cache handle.
    pub fn cache(&self) -> Arc<RwLock<ScriptCache>> {
        Arc::clone(&self.cache)
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Load a script by name.
    ///
    /// Resolution order:
    /// 1. Return from cache if the entry is still fresh (< 60 s).
    /// 2. Fetch from Seshat and refresh the cache.
    /// 3. If Seshat is unreachable and a stale cache entry exists, return it
    ///    with a `warn!` log.
    /// 4. Otherwise propagate the transport error.
    ///
    /// Returns `Ok(None)` when Seshat returns 404.
    pub fn load(&self, name: &str) -> ElizaResult<Option<ScriptContent>> {
        // Fast path: fresh cache.
        {
            let cache = self.cache.read().map_err(|_| ElizaError::CachePoisoned)?;
            if let Some(entry) = cache.get(name) {
                return Ok(Some(ScriptContent {
                    name: name.to_string(),
                    content: entry.content.clone(),
                    revision: entry.revision.clone(),
                    saved_at: String::new(),
                }));
            }
        }

        // Fetch from Seshat.
        let url = format!("{}/scripts/{}", self.base_url, name);
        match self.agent.get(&url).call() {
            Ok(response) => {
                let script: ScriptContent = response
                    .into_json()
                    .map_err(|e| ElizaError::Serde(e.to_string()))?;

                let mut cache = self.cache.write().map_err(|_| ElizaError::CachePoisoned)?;
                cache.insert(
                    script.name.clone(),
                    script.content.clone(),
                    script.revision.clone(),
                );
                Ok(Some(script))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(ureq::Error::Status(code, response)) => {
                let body = response.into_string().unwrap_or_default();
                Err(ElizaError::SeshatStatus { status: code, body })
            }
            Err(ureq::Error::Transport(e)) => {
                // Degraded mode: try stale cache.
                warn!("Seshat unreachable when loading script '{name}': {e}");
                let cache = self.cache.read().map_err(|_| ElizaError::CachePoisoned)?;
                if let Some(entry) = cache.get_stale(name) {
                    warn!(
                        "Using stale cache for script '{name}' (revision {})",
                        entry.revision
                    );
                    return Ok(Some(ScriptContent {
                        name: name.to_string(),
                        content: entry.content.clone(),
                        revision: entry.revision.clone(),
                        saved_at: String::new(),
                    }));
                }
                Err(ElizaError::SeshatHttp(e.to_string()))
            }
        }
    }

    /// Save `content` as script `name` in Seshat, returning the new revision ID.
    ///
    /// Invalidates the local cache entry for `name` so the next `load` fetches
    /// the freshly-saved revision.
    pub fn save(&self, name: &str, content: &str) -> ElizaResult<SaveResult> {
        let url = format!("{}/scripts/{}", self.base_url, name);
        let body = SaveRequest { content };

        let response = self
            .agent
            .post(&url)
            .send_json(&body)
            .map_err(|e| match e {
                ureq::Error::Status(code, resp) => ElizaError::SeshatStatus {
                    status: code,
                    body: resp.into_string().unwrap_or_default(),
                },
                ureq::Error::Transport(te) => ElizaError::SeshatHttp(te.to_string()),
            })?;

        let result: SaveResult = response
            .into_json()
            .map_err(|e| ElizaError::Serde(e.to_string()))?;

        // Invalidate so the next load gets the freshly-saved version.
        if let Ok(mut cache) = self.cache.write() {
            cache.invalidate(name);
        }

        Ok(result)
    }

    /// List all ELIZA scripts available in Seshat.
    pub fn list(&self) -> ElizaResult<Vec<ScriptInfo>> {
        let url = format!("{}/scripts", self.base_url);
        match self.agent.get(&url).call() {
            Ok(response) => response
                .into_json::<Vec<ScriptInfo>>()
                .map_err(|e| ElizaError::Serde(e.to_string())),
            Err(ureq::Error::Status(code, response)) => {
                let body = response.into_string().unwrap_or_default();
                Err(ElizaError::SeshatStatus { status: code, body })
            }
            Err(ureq::Error::Transport(e)) => Err(ElizaError::SeshatHttp(e.to_string())),
        }
    }
}
