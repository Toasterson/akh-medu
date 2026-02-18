// thiserror's #[error("...{field}...")] format strings reference struct fields,
// but the compiler doesn't see through the derive macro and reports false positives.
#![allow(unused_assignments)]

//! # akh-medu
//!
//! A neuro-symbolic AI engine combining Vector Symbolic Architectures (HDC),
//! knowledge graphs, and symbolic reasoning via e-graphs.
//!
//! ## Architecture
//!
//! - **VSA core** (`vsa`): Hyperdimensional computing with configurable 10,000+ dim vectors
//! - **Knowledge graph** (`graph`): Dual-indexed (petgraph + oxigraph) with SPARQL
//! - **Symbolic reasoning** (`reason`): E-graph equality saturation via `egg`
//! - **SIMD acceleration** (`simd`): Runtime-dispatched AVX2/generic kernels
//! - **Tiered storage** (`store`): Hot (memory) → warm (mmap) → cold (redb)
//!
//! ## Library usage
//!
//! ```no_run
//! use akh_medu::engine::{Engine, EngineConfig};
//! use akh_medu::symbol::SymbolKind;
//! use akh_medu::graph::Triple;
//!
//! let engine = Engine::new(EngineConfig::default()).unwrap();
//! let sun = engine.create_symbol(SymbolKind::Entity, "Sun").unwrap();
//! let is_a = engine.create_symbol(SymbolKind::Relation, "is-a").unwrap();
//! let star = engine.create_symbol(SymbolKind::Entity, "Star").unwrap();
//! engine.add_triple(&Triple::new(sun.id, is_a.id, star.id)).unwrap();
//! ```

pub mod agent;
pub mod argumentation;
pub mod autonomous;
pub mod client;
pub mod compartment;
pub mod dispatch;
pub mod engine;
pub mod error;
pub mod export;
pub mod glyph;
pub mod grammar;
pub mod graph;
pub mod infer;
pub mod library;
pub mod message;
pub mod partition;
pub mod paths;
pub mod pipeline;
pub mod provenance;
pub mod reason;
pub mod registry;
pub mod rule_macro;
pub mod seeds;
pub mod simd;
pub mod skills;
pub mod store;
pub mod symbol;
pub mod temporal;
pub mod tms;
pub mod tui;
pub mod vsa;
pub mod workspace;
