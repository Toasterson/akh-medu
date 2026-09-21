//! Heap profiling, exposed as pprof for Pyroscope and Grafana.
//!
//! The daemons' resident memory is dominated by whatever they hydrate out of
//! Postgres, and container metrics only ever show the total. jemalloc's
//! sampling profiler attributes that total to allocation sites, which is the
//! only practical way to answer "what is holding the 1.3 GB" without guessing.
//!
//! Sampling is off unless the process is started with jemalloc profiling
//! enabled, e.g. `MALLOC_CONF=prof:true,prof_active:true,lg_prof_sample:19`.
//! Without it the endpoint reports 503 rather than pretending, so a missing
//! `MALLOC_CONF` cannot be mistaken for "nothing is allocating".
//!
//! # Exposure
//!
//! This is served on its own listener, never on [`crate::config::Config::listen`].
//! A heap profile contains live strings — session content, tokens, credentials
//! — and the service port is published through an HTTPRoute. The profiling
//! port is deliberately reachable only from inside the cluster, so scraping it
//! is a cluster-local privilege.

use axum::{Router, http::StatusCode, response::IntoResponse, routing::get};
use std::net::SocketAddr;

/// jemalloc, with its profiler compiled in. Without this the process uses the
/// system allocator and there is nothing to profile.
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// Renders the current heap as a pprof profile.
///
/// Returns 503 when the binary was built with profiling but the process was
/// started without `MALLOC_CONF`, which is the common misconfiguration.
async fn heap() -> impl IntoResponse {
    let Some(ctl) = jemalloc_pprof::PROF_CTL.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "jemalloc profiling is not enabled; start with MALLOC_CONF=prof:true,prof_active:true",
        )
            .into_response();
    };
    let mut ctl = ctl.lock().await;
    if !ctl.activated() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "jemalloc profiling is compiled in but not active",
        )
            .into_response();
    }
    match ctl.dump_pprof() {
        Ok(profile) => (
            StatusCode::OK,
            [("content-type", "application/octet-stream")],
            profile,
        )
            .into_response(),
        Err(error) => {
            tracing::error!(%error, "heap profile dump failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "heap profile dump failed",
            )
                .into_response()
        }
    }
}

/// The pprof endpoints, on the path layout Grafana Alloy's `pyroscope.scrape`
/// expects for a `memory` profile.
pub fn router() -> Router {
    Router::new().route("/debug/pprof/heap", get(heap))
}

/// Serves [`router`] on `address` until the process exits.
///
/// Binding is fatal: a profiling listener that silently failed to start is
/// worse than no profiling, because the scrape target just looks idle.
pub async fn serve(address: SocketAddr) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(address).await?;
    tracing::info!(address=%listener.local_addr()?, "heap profiling listening");
    axum::serve(listener, router()).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the real listener rather than the handler in isolation, so a
    /// routing or bind mistake cannot pass. Asserts both outcomes explicitly:
    /// with `MALLOC_CONF` the response must be a non-trivial gzipped pprof,
    /// without it a 503 — never a silent empty 200, which would look like a
    /// healthy scrape target that reports nothing.
    /// Held live across the dump so the sampler has something to attribute.
    /// `inline_never` keeps the frame in the stack, which is what makes this a
    /// real symbolization check rather than a check that bytes came back.
    #[inline(never)]
    fn allocate_profiling_ballast() -> Vec<Vec<u8>> {
        (0..64).map(|_| vec![7u8; 1024 * 1024]).collect()
    }

    #[tokio::test]
    async fn heap_endpoint_serves_pprof_or_says_why_not() {
        // 64 MiB against the default 512 KiB sample interval, so the profile
        // cannot be empty for lack of allocations.
        let ballast = allocate_profiling_ballast();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router()).await });

        let response = reqwest::get(format!("http://{address}/debug/pprof/heap"))
            .await
            .expect("profiling endpoint reachable");
        let status = response.status();
        let body = response.bytes().await.unwrap();
        eprintln!("heap endpoint: status={status} bytes={}", body.len());

        // macOS jemalloc has no `prof` support, so the 200 path is only
        // reachable on Linux. Set AKH_REQUIRE_HEAP_PROFILE=1 where profiling
        // must actually work (the container build does) so that a silent
        // regression to 503 fails instead of passing as "the other branch".
        let required = std::env::var("AKH_REQUIRE_HEAP_PROFILE").is_ok_and(|v| v == "1");
        assert!(
            !required || status == reqwest::StatusCode::OK,
            "AKH_REQUIRE_HEAP_PROFILE=1 but heap profiling returned {status}"
        );
        if status == reqwest::StatusCode::OK {
            assert!(
                body.len() > 128,
                "pprof payload implausibly small: {}",
                body.len()
            );
            // pprof is gzip-wrapped protobuf; check the magic rather than the length alone.
            assert_eq!(&body[..2], &[0x1f, 0x8b], "expected gzip-wrapped pprof");
            // Written out so the container check can assert the profile is
            // symbolized; an unsymbolized dump still passes every check above.
            let _ = std::fs::write("/tmp/heap-test.pprof", &body);
            drop(ballast);
        } else {
            assert_eq!(status, reqwest::StatusCode::SERVICE_UNAVAILABLE);
            assert!(
                String::from_utf8_lossy(&body).contains("profiling"),
                "503 should explain that profiling is off"
            );
        }
    }
}
