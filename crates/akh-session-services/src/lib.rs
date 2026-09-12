//! The services share wire contracts, not an engine instance or database file.
pub use akh_session_types as types;
#[cfg(feature = "session")]
pub mod a2a;
#[cfg(feature = "session")]
pub mod acp;
#[cfg(feature = "session")]
pub mod broker;
pub mod config;
#[cfg(feature = "client")]
pub mod connect;
#[cfg(feature = "events")]
pub mod djehuty;
#[cfg(feature = "executor")]
pub mod executor;
#[cfg(feature = "executor")]
pub mod executor_helpers;
#[cfg(feature = "helpers")]
pub mod helper;
pub mod http;
#[cfg(feature = "events")]
pub mod ingress;
#[cfg(feature = "session")]
pub mod mcp;
pub mod recovery;
#[cfg(feature = "runtime")]
pub mod runtime;
#[cfg(feature = "sandbox")]
pub mod sandbox;
#[cfg(any(
    feature = "session",
    feature = "runtime",
    feature = "executor",
    feature = "sandbox",
    feature = "events"
))]
pub mod server;
#[cfg(feature = "session")]
pub mod session;
#[cfg(feature = "session")]
pub mod session_credentials;
pub mod store;
#[cfg(feature = "helpers")]
pub mod workspace;

/// Stable digest for durable identities and request fingerprints.
pub fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(any(feature = "helpers", feature = "runtime", feature = "sandbox"))]
pub mod process;

#[cfg(feature = "runtime")]
pub mod runtime_profile;

#[cfg(feature = "runtime")]
pub mod internal;
