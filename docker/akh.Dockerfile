# akh — the akh-medu CLI/TUI client.
#
# Lean single-arch (amd64) image built with the legacy docker builder so it
# works on plain Solstice runner VMs without buildx. Built with the `client`
# feature only (`--no-default-features --features client` = ACP transport, no
# nlu-candle): `akh` is a pure ACP client, so it needs no in-process LLM/candle
# and no onnxruntime. Small + fast to build, and the binary this image carries is
# what the release extracts for the cargo-binstall artifact.
#
# Build:  docker build -f docker/akh.Dockerfile -t akh .
# Run:    docker run --rm -it -v akh-data:/data akh akh --help

ARG RUST_VERSION=1.94.1

FROM rust:${RUST_VERSION}-bookworm AS builder

WORKDIR /build

# Copy manifests first so dep compilation caches as its own layer.
COPY Cargo.toml Cargo.lock rust-toolchain.toml* ./
COPY crates/seshat/Cargo.toml crates/seshat/Cargo.toml
RUN mkdir -p src/bin crates/seshat/src benches && \
    echo 'fn main() {}' > src/main.rs && \
    echo 'fn main() {}' > src/bin/akhomed.rs && \
    touch src/lib.rs && \
    echo 'fn main() {}' > crates/seshat/src/main.rs && \
    touch crates/seshat/src/lib.rs && \
    echo 'fn main() {}' > benches/vsa_ops.rs && \
    echo 'fn main() {}' > benches/phase26.rs

# Pre-build dependencies with the stub entry point.
RUN cargo build --release --no-default-features --features client --bin akh || true

# Copy full source and build the real binary. `touch` defeats COPY's preserved
# mtimes; `rm -f` is belt-and-suspenders against the cached stub artifact.
COPY . .
RUN find src crates benches -name '*.rs' -exec touch {} + && \
    rm -f target/release/akh && \
    cargo build --release --no-default-features --features client --bin akh && \
    cp target/release/akh /build/akh

FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r akh && useradd -r -g akh -d /data -s /sbin/nologin akh && \
    mkdir -p /data && chown akh:akh /data

COPY --from=builder /build/akh /usr/local/bin/akh

ENV RUST_LOG=warn \
    HOME=/data

USER akh
WORKDIR /data

ENTRYPOINT ["tini", "--", "akh"]
CMD ["--help"]
