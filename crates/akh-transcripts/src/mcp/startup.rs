//! Startup diagnostics for `akh-transcripts-mcp`.
//!
//! `crate::error::TranscriptError` is owned by a concurrent increment
//! (adapter/derivation-worker work landing on the same branch), so the
//! errors specific to *this* binary's startup path — bearer-token
//! configuration, bind-address parsing, listener binding — get their own
//! small diagnostic enum here instead, following the same
//! miette + thiserror convention: every variant carries a `code` and
//! actionable `help`.

use std::net::SocketAddr;

use miette::Diagnostic;
use thiserror::Error;

#[derive(Debug, Error, Diagnostic)]
pub enum StartupError {
    #[error("no bearer token configured: {0}")]
    #[diagnostic(
        code(akh_transcripts::mcp::bearer_token_missing),
        help(
            "Set AKH_TRANSCRIPTS_BEARER_TOKEN (inline) or AKH_TRANSCRIPTS_BEARER_TOKEN_FILE \
             (a projected Kubernetes Secret path, one token per line) before starting the \
             server. This server refuses to start unauthenticated — there is no \
             unauthenticated mode, by design (read-mcp-surface.md §3)."
        )
    )]
    BearerTokenMissing(String),

    #[error("AKH_TRANSCRIPTS_DATABASE_URL is not set")]
    #[diagnostic(
        code(akh_transcripts::mcp::database_url_missing),
        help(
            "Set AKH_TRANSCRIPTS_DATABASE_URL to a Postgres connection string for the \
             harvest_read role (SELECT-only — this server never writes). See \
             docs/ops/transcript-harvest.md for the mTLS connection shape used in the fleet."
        )
    )]
    DatabaseUrlMissing,

    #[error("invalid AKH_TRANSCRIPTS_BIND value {addr:?}: {source}")]
    #[diagnostic(
        code(akh_transcripts::mcp::invalid_bind_address),
        help("AKH_TRANSCRIPTS_BIND must be a host:port pair, e.g. 0.0.0.0:9500.")
    )]
    InvalidBindAddress {
        addr: String,
        #[source]
        source: std::net::AddrParseError,
    },

    #[error("failed to bind {addr}: {source}")]
    #[diagnostic(
        code(akh_transcripts::mcp::bind_failed),
        help(
            "Confirm nothing else is already listening on this address/port, and that the \
             process has permission to bind it (ports below 1024 need a privileged user or \
             CAP_NET_BIND_SERVICE)."
        )
    )]
    BindFailed {
        addr: SocketAddr,
        #[source]
        source: std::io::Error,
    },
}
