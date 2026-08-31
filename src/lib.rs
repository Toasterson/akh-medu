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

#[cfg(feature = "acp")]
pub mod acp;
pub mod agent;
pub mod akh_config;
pub mod bench_phase41g;
pub mod bench_phase44;
pub mod cluster;
pub mod argumentation;
pub mod audit;
pub mod backend;
pub mod bootstrap;
pub mod chat;
#[cfg(feature = "email")]
pub mod email;
pub mod autonomous;
pub mod client;
pub mod compartment;
#[cfg(feature = "daemon")]
pub mod config;
pub mod dispatch;
pub mod eliza;
pub mod engine;
pub mod extraction;
pub mod error;
pub mod export;
pub mod fs;
pub mod glyph;
pub mod grammar;
pub mod graph;
pub mod infer;
pub mod inference_options;
pub mod library;
#[cfg(feature = "daemon")]
pub mod upstream_client;
pub mod message;
pub mod mutation;
pub mod nlu;
pub mod partition;
pub mod perception;
pub mod paths;
pub mod pipeline;
pub mod provenance;
pub mod reason;
pub mod reasoning;
pub mod registry;
/// Phase 38i: OTLP tracing skeleton — `init_tracing` helper + `TracerShutdown`.
/// The module is always available; the OTLP exporter path is gated behind the
/// `tracing-otel` feature inside the module so the lean `akh` CLI binary can
/// still call `init_tracing` and get the plain-`fmt` fallback.
pub mod observability;

/// Phase 38j.1: typed metric-instrument registry for the akhomed daemon.
/// Available only when the `tracing-otel` feature is enabled (the OTel
/// `Meter` type is required).
#[cfg(feature = "tracing-otel")]
pub mod metrics;
pub mod rule_macro;
/// ADR 081: schema packs — the installable type system for Akh memory
/// (stable triple identity lives in `graph::identity`).
pub mod schema;
pub mod seeds;
pub mod service;
pub mod setup;
pub mod skolem;
pub mod simd;
pub mod skills;
pub mod store;
pub mod symbol;
pub mod temporal;
/// Phase 41: the unified thinking loop (ADR 052).
pub mod thinking;
pub mod tms;
pub mod training;
pub mod tui;
pub mod vsa;
pub mod wiki;
pub mod workspace;
/// Phase 38b: mutation write-ahead log to the Postgres backing store.
#[cfg(feature = "postgres")]
pub mod wal;
/// Phase 38c: snapshot the akh's relational state to Postgres + wake from it.
#[cfg(feature = "postgres")]
pub mod snapshot;
