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
//! - **Article tools** (22b): write/cite/respond/search tools for the OODA loop
//! - **Rendering** (22c): comrak + minijinja → static HTML (requires `wiki` feature)
//! - **Full-text search** (22d): Tantivy index (requires `wiki` feature)
//! - **Federation** (22e): ActivityPub Create/Update/Announce activities
//! - **Blog & opinions** (22f): opinion blocks, debate threads, archetype voice

pub mod blog;
pub mod federation;
pub mod render;
pub mod search;
pub mod store;
pub mod tools;

pub use store::{
    Article, ArticleHeader, ArticleStore, BlockRef, BlockType, Stance, WikiError, WikiResult,
};

pub use tools::{CiteArticleTool, RespondToArticleTool, SearchWikiTool, WriteArticleTool};

pub use blog::{ArchetypeVoice, BlogError, BlogPost, BlogResult, DebateThread};

pub use federation::{
    ApActivity, ApArticle, FederationError, FederationResult,
    build_announce_activity, build_create_activity, build_update_activity,
    parse_inbound_activity,
};

#[cfg(feature = "wiki")]
pub use render::{RenderConfig, RenderError, RenderResult, WikiRenderer};

#[cfg(feature = "wiki")]
pub use search::{SearchError, SearchHit, SearchResult, WikiSearch};
