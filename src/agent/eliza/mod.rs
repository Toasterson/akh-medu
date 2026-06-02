//! ELIZA dialogue engines.
//!
//! Two complementary engines live under this module:
//!
//! * The always-available Weizenbaum-style pattern-matching core ([`engine`] /
//!   [`script`], PR #172) — configured via TOML scripts with weighted keywords,
//!   decomposition patterns, reassembly templates and a topic-memory stack.
//! * A KDL-based hot-reloadable script engine ([`kdl`], PR #178) — gated behind
//!   the `eliza` cargo feature. Scripts are `.kdl` files that can be edited and
//!   reloaded without restarting akhomed.
//!
//! The two engines use disjoint type names (`ElizaEngine`/`Script` vs
//! `ScriptEngine`/`CompiledScript`) so they coexist without conflict.
//!
//! # Quick start (TOML core)
//!
//! ```rust,ignore
//! use crate::agent::eliza::{ElizaEngine, Script};
//!
//! let script = Script::from_toml(include_str!("scripts/doctor.toml")).unwrap();
//! let mut engine = ElizaEngine::new(script);
//! println!("{}", engine.greeting());
//! println!("{}", engine.respond("I am feeling very depressed."));
//! ```

// ── Core TOML-script engine (PR #172) — always available ──────────────────────
pub mod engine;
pub mod script;

pub use engine::ElizaEngine;
pub use script::{Keyword, Rule, Script, ScriptError, ScriptResult};

// ── KDL hot-reload engine (PR #178) — feature-gated ───────────────────────────
// `eliza` pulls in the `knuffel` KDL parser and `notify` filesystem watcher and
// requires `daemon` (see Cargo.toml). The hot-reload watcher lives in its own
// submodule; both are re-exported at the `eliza` root so existing paths such as
// `agent::eliza::ScriptEngine` keep working.
#[cfg(feature = "eliza")]
pub mod kdl;
#[cfg(feature = "eliza")]
pub mod hot_reload;

#[cfg(feature = "eliza")]
pub use kdl::*;
