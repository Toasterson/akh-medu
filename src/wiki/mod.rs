//! Akhipedia — the agent's personal knowledge wiki (Phase 22).
//!
//! A structured article store where the agent compiles what it knows
//! into readable, citable articles. Articles have typed blocks
//! (Definition, Relationship, Claim, Opinion) and are stored in the
//! KG with Dublin Core metadata.
//!
//! ## Architecture
//!
//! - **ArticleStore** (22a): CRUD + content-addressed storage in redb
//! - **Article tools** (22b): write/cite/respond tools for the OODA loop
//! - Rendering (22c), search (22d), federation (22e), blog (22f): planned

pub mod store;

pub use store::{
    Article, ArticleHeader, ArticleStore, BlockRef, BlockType, Stance, WikiError, WikiResult,
};
