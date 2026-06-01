//! Hot-reloader for ELIZA KDL scripts.
//!
//! `ScriptReloader` spawns a background thread that:
//! 1. Polls Seshat every 30 seconds for new script revisions.
//! 2. In Dev mode: also polls the local scripts directory for `.kdl` file
//!    modifications (using `fs::metadata().modified()` timestamps).
//!
//! When a change is detected, all registered callbacks receive
//! `(script_name, new_content)`.
//!
//! ## Precedence rules
//! - **Dev mode**: local files always win.  If a local `.kdl` file exists for
//!   a script that also changed in Seshat, the Seshat revision is silently
//!   skipped and the local version is fired instead.
//! - **Prod mode**: only Seshat revisions are tracked; the local directory is
//!   not polled even if configured.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, SystemTime};

use tracing::{debug, info, warn};

use crate::eliza::seshat::SeshatScriptStore;
use crate::eliza::{ElizaMode, ScriptChangeCallback};

/// How often the reloader polls Seshat for new revisions.
const POLL_INTERVAL: Duration = Duration::from_secs(30);
/// Inner sleep granularity to allow timely response to the stop signal.
const TICK: Duration = Duration::from_secs(1);

/// Builder / owner for the ELIZA hot-reload background thread.
pub struct ScriptReloader {
    store: SeshatScriptStore,
    mode: ElizaMode,
    local_dir: Option<PathBuf>,
    callbacks: Vec<ScriptChangeCallback>,
}

impl ScriptReloader {
    /// Create a new reloader backed by `store`.
    pub fn new(store: SeshatScriptStore, mode: ElizaMode) -> Self {
        Self {
            store,
            mode,
            local_dir: None,
            callbacks: Vec::new(),
        }
    }

    /// Watch local `.kdl` files in `dir` in addition to Seshat revisions.
    /// Has no effect in Prod mode (Seshat is the sole source of truth).
    pub fn with_local_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.local_dir = Some(dir.into());
        self
    }

    /// Register a callback invoked on every detected script change.
    ///
    /// The callback receives `(script_name, new_content)` and must be
    /// `Send + Sync + 'static` because it runs on the reloader thread.
    pub fn on_change(mut self, callback: ScriptChangeCallback) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Start the hot-reload background thread.
    ///
    /// Returns a [`ReloaderHandle`]; dropping the handle signals the thread to
    /// stop (it exits within one `TICK` interval, i.e. ≤ 1 second).
    pub fn start(self) -> ReloaderHandle {
        let (stop_tx, stop_rx) = mpsc::channel::<()>();

        let store = self.store;
        let mode = self.mode;
        let local_dir = self.local_dir;
        let callbacks = Arc::new(self.callbacks);

        let thread = thread::Builder::new()
            .name("eliza-reloader".to_string())
            .spawn(move || {
                run_reloader(store, mode, local_dir, callbacks, stop_rx);
            })
            .expect("failed to spawn eliza-reloader thread");

        ReloaderHandle {
            _stop_tx: stop_tx,
            _thread: thread,
        }
    }
}

/// RAII handle for the hot-reload thread.
///
/// Drop to request a graceful stop; the thread exits within one TICK interval.
pub struct ReloaderHandle {
    /// Sending on drop signals the thread to exit.
    _stop_tx: mpsc::Sender<()>,
    _thread: thread::JoinHandle<()>,
}

// ── Reloader loop ───────────────────────────────────────────────────────────

fn run_reloader(
    store: SeshatScriptStore,
    mode: ElizaMode,
    local_dir: Option<PathBuf>,
    callbacks: Arc<Vec<ScriptChangeCallback>>,
    stop_rx: mpsc::Receiver<()>,
) {
    // Revision strings seen in the last Seshat poll: name → Seshat revision.
    let mut seshat_revisions: HashMap<String, String> = HashMap::new();
    // Modification times for local .kdl files: name → SystemTime.
    let mut local_mtimes: HashMap<String, SystemTime> = HashMap::new();

    info!("eliza-reloader: started (mode={mode})");

    // Initial load on startup.
    poll_seshat(
        &store,
        mode,
        local_dir.as_deref(),
        &mut seshat_revisions,
        &callbacks,
    );
    if let Some(ref dir) = local_dir {
        poll_local(dir, &mut local_mtimes, &callbacks);
    }

    loop {
        // Sleep POLL_INTERVAL in TICK-sized steps so we respond to stop fast.
        let ticks = POLL_INTERVAL.as_secs() / TICK.as_secs();
        for _ in 0..ticks {
            thread::sleep(TICK);
            if stop_rx.try_recv().is_ok() {
                info!("eliza-reloader: stopped");
                return;
            }
        }

        poll_seshat(
            &store,
            mode,
            local_dir.as_deref(),
            &mut seshat_revisions,
            &callbacks,
        );
        if let (ElizaMode::Dev, Some(dir)) = (mode, local_dir.as_deref()) {
            poll_local(dir, &mut local_mtimes, &callbacks);
        }
    }
}

/// Check Seshat for new or changed script revisions and fire callbacks.
fn poll_seshat(
    store: &SeshatScriptStore,
    mode: ElizaMode,
    local_dir: Option<&Path>,
    known: &mut HashMap<String, String>,
    callbacks: &Arc<Vec<ScriptChangeCallback>>,
) {
    let scripts = match store.list() {
        Ok(s) => s,
        Err(e) => {
            warn!("eliza-reloader: Seshat list failed: {e}");
            return;
        }
    };

    for info in scripts {
        let changed = known
            .get(&info.name)
            .map(|rev| rev != &info.revision)
            .unwrap_or(true);

        if !changed {
            continue;
        }

        // In Dev mode, skip if a local file takes precedence.
        if let (ElizaMode::Dev, Some(dir)) = (mode, local_dir) {
            let local_path = dir.join(format!("{}.kdl", info.name));
            if local_path.exists() {
                debug!(
                    "eliza-reloader: skipping Seshat update for '{}' — \
                     local file has precedence in Dev mode",
                    info.name
                );
                known.insert(info.name.clone(), info.revision.clone());
                continue;
            }
        }

        debug!(
            "eliza-reloader: script '{}' new revision {}",
            info.name, info.revision
        );

        match store.load(&info.name) {
            Ok(Some(script)) => {
                fire_callbacks(callbacks, &info.name, &script.content);
                known.insert(info.name.clone(), info.revision.clone());
            }
            Ok(None) => {
                warn!(
                    "eliza-reloader: script '{}' listed but returned 404 on load",
                    info.name
                );
            }
            Err(e) => {
                warn!("eliza-reloader: failed to load script '{}': {e}", info.name);
            }
        }
    }
}

/// Scan a local directory for `.kdl` files whose modification time changed.
fn poll_local(
    dir: &Path,
    mtimes: &mut HashMap<String, SystemTime>,
    callbacks: &Arc<Vec<ScriptChangeCallback>>,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("kdl") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|n| n.to_str()).map(str::to_owned) else {
            continue;
        };
        let Ok(metadata) = path.metadata() else {
            continue;
        };
        let Ok(mtime) = metadata.modified() else {
            continue;
        };

        let changed = mtimes.get(&name).map(|&prev| prev != mtime).unwrap_or(true);
        if !changed {
            continue;
        }

        match std::fs::read_to_string(&path) {
            Ok(content) => {
                info!("eliza-reloader: local script '{name}' changed, hot-reloading");
                fire_callbacks(callbacks, &name, &content);
                mtimes.insert(name, mtime);
            }
            Err(e) => {
                warn!("eliza-reloader: failed to read local script '{name}': {e}");
            }
        }
    }
}

fn fire_callbacks(callbacks: &Arc<Vec<ScriptChangeCallback>>, name: &str, content: &str) {
    for cb in callbacks.as_ref() {
        cb(name, content);
    }
}
