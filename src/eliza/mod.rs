//! ELIZA dialogue system with VSA-enhanced fuzzy keyword matching.
//!
//! This module provides a classic ELIZA-style pattern-matching dialogue engine
//! extended with Vector Symbolic Architecture (VSA) hypervectors for fuzzy
//! keyword matching.
//!
//! ## Usage
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
//! ## Matching strategy
//!
//! Each rule carries a `keywords` list that serves as a synonym group.  At
//! load time all keywords are encoded as bipolar hypervectors (10 000 dims).
//!
//! For every user turn the engine:
//!
//! 1. Tokenises the input into lowercase words.
//! 2. For each rule (priority-descending):
//!    a. **Exact path** — checks whether any input word is literally in the
//!       keyword list.  This is the primary match path for synonym groups.
//!    b. **VSA path** — encodes each input word as a hypervector and computes
//!       Hamming similarity against each keyword's hypervector.  A word
//!       matches if the similarity meets or exceeds `similarity_threshold`.
//!       With the current hash-based token encoder this path fires for tokens
//!       that are identical to a keyword.  When semantic vectors are loaded
//!       (Phase 26/32) it will generalise to unseen synonyms automatically.
//! 3. Returns the first matching rule's next round-robin response, or the
//!    script's `default_response`.

pub mod engine;
pub mod error;
pub mod script;

pub use engine::ElizaEngine;
pub use error::{ElizaError, ElizaResult};
pub use script::Script;
