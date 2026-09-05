//! ELIZA dialogue system.
//!
//! This module bundles two complementary pieces, both developed against
//! `src/eliza/`:
//!
//! * **VSA-enhanced pattern-matching engine** (PR #173) — a classic
//!   ELIZA-style dialogue engine ([`engine::ElizaEngine`]) configured from TOML
//!   [`script::Script`]s, extended with Vector Symbolic Architecture (VSA)
//!   hypervectors for fuzzy keyword matching.
//! * **Seshat script store + hot-reloader** (PR #179) — durable, federated
//!   storage for ELIZA dialogue scripts via Seshat's document store
//!   ([`seshat::SeshatScriptStore`]), with a TTL [`cache`] and a hot-reload
//!   background thread ([`reloader`]).
//!
//! ## Usage — VSA engine
//!
//! ```no_run
//! use akh_medu::eliza::{ElizaEngine, Script};
//!
//! let script_toml = r#"
//! [script]
//! name = "Doctor"
//! default_response = "Please go on."
//! initial_greeting = "How do you do. Please tell me your problem."
//!
//! [[rules]]
//! keywords = ["mother", "mom", "mama", "parent", "father", "dad"]
//! similarity_threshold = 0.70
//! responses = ["Tell me more about your family.", "Who else in your family?"]
//! "#;
//!
//! let script = Script::from_toml(script_toml).unwrap();
//! let mut engine = ElizaEngine::new(script).unwrap();
//!
//! if let Some(greeting) = engine.initial_greeting() {
//!     println!("ELIZA: {greeting}");
//! }
//! println!("{}", engine.respond("I miss my mom"));
//! ```
//!
//! ## Usage — Seshat store (Dev mode)
//!
//! ```no_run
//! use akh_medu::eliza::{ElizaMode, SeshatScriptStore};
//! use akh_medu::eliza::reloader::ScriptReloader;
//!
//! let store = SeshatScriptStore::new("http://seshat.lab.home:9090");
//! // store.save("sage", "node { prop = val }").unwrap();
//!
//! let _reloader = ScriptReloader::new(store.clone(), ElizaMode::Dev)
//!     .with_local_dir("scripts/")
//!     .on_change(Box::new(|name: &str, content: &str| {
//!         println!("Script '{name}' changed ({} bytes)", content.len());
//!     }))
//!     .start();
//! ```

// PR #173 — VSA pattern-matching engine.
pub mod engine;
pub mod script;

// PR #174 — KG-grounded response templates (fact insertion). Reconciled as a
// non-colliding capability module the VSA engine and chat path can call.
pub mod kg_template;

// Shared error type (union of both subsystems' variants).
pub mod error;

// PR #179 — Seshat-backed script store + hot-reloader.
pub mod cache;
pub mod reloader;
pub mod seshat;

pub use engine::ElizaEngine;
pub use error::{ElizaError, ElizaResult};
pub use kg_template::{
    ElizaSession, EntityRef, FactQuery, FactQueryKind, Template, format_facts, query_with_timeout,
    triple_to_nl,
};
pub use script::Script;
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
