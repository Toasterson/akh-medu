//! ELIZA KDL script subsystem.
//!
//! Provides durable, federated storage for ELIZA dialogue scripts via
//! Seshat's document store, with a 60-second TTL in-memory cache and
//! a hot-reload background thread that reacts to Seshat revision changes
//! and (in Dev mode) local file modifications.
//!
//! ## Quick start — Dev mode
//! ```no_run
//! use akh_medu::eliza::{ElizaMode, SeshatScriptStore};
//! use akh_medu::eliza::reloader::ScriptReloader;
//!
//! let store = SeshatScriptStore::new("http://seshat.lab.home:9090");
//! // store.save("sage", "node { prop = val }").unwrap();
//!
//! let _reloader = ScriptReloader::new(store.clone(), ElizaMode::Dev)
//!     .with_local_dir("scripts/")
//!     .on_change(Box::new(|name, content| {
//!         println!("Script '{name}' changed ({} bytes)", content.len());
//!     }))
//!     .start();
//! ```
//!
//! ## Quick start — Prod mode
//! ```no_run
//! use akh_medu::eliza::{ElizaMode, SeshatScriptStore};
//! use akh_medu::eliza::reloader::ScriptReloader;
//!
//! let store = SeshatScriptStore::new("http://seshat.lab.home:9090");
//!
//! // Fetch all scripts from Seshat on startup.
//! for info in store.list().unwrap() {
//!     let script = store.load(&info.name).unwrap();
//!     println!("Loaded '{}' revision {}", info.name, info.revision);
//! }
//!
//! // Watch for revision changes every 30 s.
//! let _reloader = ScriptReloader::new(store.clone(), ElizaMode::Prod)
//!     .on_change(Box::new(|name, _content| {
//!         println!("Seshat revision updated for '{name}'");
//!     }))
//!     .start();
//! ```

pub mod cache;
pub mod error;
pub mod reloader;
pub mod seshat;

pub use error::{ElizaError, ElizaResult};
pub use seshat::{SaveResult, ScriptContent, ScriptInfo, SeshatScriptStore};

/// Callback signature for script-change notifications.
///
/// Receives `(script_name, new_content)`.  Must be `Send + Sync + 'static`
/// because it is called from the hot-reload background thread.
pub type ScriptChangeCallback = Box<dyn Fn(&str, &str) + Send + Sync + 'static>;

/// ELIZA operating mode — controls script source precedence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ElizaMode {
    /// Local `.kdl` files take precedence over Seshat.
    ///
    /// Workflow: edit locally → instant hot-reload → `store.save()` to persist.
    #[default]
    Dev,

    /// Seshat is the sole source of truth.
    ///
    /// All scripts are fetched from Seshat on startup; the local directory is
    /// not monitored even if [`reloader::ScriptReloader::with_local_dir`] was
    /// called.
    Prod,
}

impl std::fmt::Display for ElizaMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dev => write!(f, "dev"),
            Self::Prod => write!(f, "prod"),
        }
    }
}
