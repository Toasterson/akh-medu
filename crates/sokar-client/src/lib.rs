//! Wire types are usable without an HTTP/TLS stack with
//! `default-features = false, features = ["types-only"]`.
#![forbid(unsafe_code)]
mod types;
pub use types::*;
#[cfg(feature = "client")]
mod client;
#[cfg(feature = "client")]
pub use client::Client;

#[derive(Debug, thiserror::Error, miette::Diagnostic)]
pub enum Error {
    #[error("invalid Sokar value: {0}")]
    #[diagnostic(code(sokar_client::invalid), help("Check the typed request fields."))]
    Invalid(String),
    #[error("unsupported manifest schema {actual}; expected {expected}")]
    #[diagnostic(
        code(sokar_client::manifest_schema),
        help("Upgrade the reader before using a newer manifest schema.")
    )]
    ManifestSchema {
        actual: String,
        expected: &'static str,
    },
    #[error("Sokar HTTP {} ({}): {}", .0.status, .0.code, .0.detail)]
    #[diagnostic(
        code(sokar_client::remote),
        help("Inspect the remote problem code and request_id.")
    )]
    Remote(Box<Problem>),
    #[cfg(feature = "client")]
    #[error("Sokar transport failed: {0}")]
    #[diagnostic(
        code(sokar_client::transport),
        help("Check the server URL and network; retry uploads with the same idempotency key.")
    )]
    Transport(#[from] reqwest::Error),
    #[error("Sokar serialization failed: {0}")]
    #[diagnostic(
        code(sokar_client::serde),
        help("Check the request or response wire schema.")
    )]
    Serde(#[from] serde_json::Error),
}
impl Error {
    pub fn diagnostic_code(&self) -> &'static str {
        match self {
            Self::Invalid(_) => "sokar_client::invalid",
            Self::ManifestSchema { .. } => "sokar_client::manifest_schema",
            Self::Remote(_) => "sokar_client::remote",
            #[cfg(feature = "client")]
            Self::Transport(_) => "sokar_client::transport",
            Self::Serde(_) => "sokar_client::serde",
        }
    }

    pub fn problem(&self) -> Option<&Problem> {
        if let Self::Remote(p) = self {
            Some(p)
        } else {
            None
        }
    }
}
