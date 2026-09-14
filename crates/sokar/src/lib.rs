//! Sokar artifact daemon foundation (ADR 108; Anima umbrella
//! 01a0a00a-d3b0-7713-90cc-99b6fdac941e, docs/ai/decisions/108-sokar-artifact-daemon.md).
#![forbid(unsafe_code)]
pub mod auth;
pub mod authz;
pub mod config;
pub mod delivery;
pub mod error;
pub mod http;
pub mod index;
pub mod service;
pub mod store;
pub use error::{SokarError, SokarResult};
pub use service::Sokar;
