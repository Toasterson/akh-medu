//! ELIZA pattern-matching engine (Weizenbaum 1966).
//!
//! A lightweight, rule-based dialogue system configured via TOML scripts.
//! Scripts define weighted keywords, decomposition patterns, and reassembly
//! templates. The engine also maintains a memory stack for topic continuity.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use crate::agent::eliza::{ElizaEngine, Script};
//!
//! let script = Script::from_toml(include_str!("scripts/doctor.toml")).unwrap();
//! let mut engine = ElizaEngine::new(script);
//! println!("{}", engine.greeting());
//! println!("{}", engine.respond("I am feeling very depressed."));
//! ```

pub mod engine;
pub mod script;

pub use engine::ElizaEngine;
pub use script::{Keyword, Rule, Script, ScriptError, ScriptResult};
