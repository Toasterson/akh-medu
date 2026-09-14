# sokard — Sokar content-addressed artifact daemon (OpenDAL blobs + Postgres metadata)
#
# Single-arch (amd64). Uses the legacy docker builder so it works on plain
# Solstice runner VMs without buildx. Chart + HelmRelease pin pods to amd64
# nodes, so cross-compile complexity is unnecessary.
#
# Build:  docker build -f docker/sokard.Dockerfile -t sokard .
# Run:    docker run -p 9400:9400 -p 9401:9401 -v sokar-config:/config sokard

ARG RUST_VERSION=1.94.1

# ── Stage 1: Build ────────────────────────────────────────────────────────
FROM rust:${RUST_VERSION}-bookworm AS builder

WORKDIR /build

# Cap rustc parallelism: the Solstice runner VMs are memory-constrained and
# sokar's graph (opendal + sqlx + axum + aws-lc) is heavier than seshd's.
ENV CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0

# Copy manifests first so dep compilation caches as its own layer. sokar
# depends on sokar-client by path, so both manifests are needed up front.
COPY Cargo.toml Cargo.lock rust-toolchain.toml* ./
COPY crates/sokar/Cargo.toml crates/sokar/Cargo.toml
COPY crates/sokar-client/Cargo.toml crates/sokar-client/Cargo.toml
RUN mkdir -p src/bin crates/sokar/src crates/sokar-client/src benches && \
    echo 'fn main() {}' > src/main.rs && \
    echo 'fn main() {}' > src/bin/akhomed.rs && \
    touch src/lib.rs && \
    echo 'fn main() {}' > crates/sokar/src/main.rs && \
    touch crates/sokar/src/lib.rs && \
    touch crates/sokar-client/src/lib.rs && \
    echo 'fn main() {}' > benches/vsa_ops.rs && \
    echo 'fn main() {}' > benches/phase26.rs

# Pre-build dependencies with the stub entry points. `|| true` so the stub
# phase doesn't fail the build if anything goes sideways.
RUN cargo build --release -p sokar --bin sokard || true

# Copy full source and build the real binary.
# `touch` forces cargo to re-evaluate mtimes (docker COPY preserves them and
# can leave cargo thinking the stub sources are still current). `rm -f` on
# the cached stub binary is belt-and-suspenders so we never cp the stub.
COPY . .
RUN find src crates benches -name '*.rs' -exec touch {} + && \
    rm -f target/release/sokard && \
    cargo build --release -p sokar --bin sokard && \
    cp target/release/sokard /build/sokard

# ── Stage 2: Runtime ─────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r -g 999 sokar && useradd -r -u 999 -g sokar -d /data -s /sbin/nologin sokar && \
    mkdir -p /data /config && chown sokar:sokar /data /config

COPY --from=builder /build/sokard /usr/local/bin/sokard

ENV RUST_LOG=sokar=info

EXPOSE 9400 9401

USER sokar
WORKDIR /data

VOLUME ["/data", "/config"]

ENTRYPOINT ["tini", "--"]
CMD ["sokard", "--config", "/config/sokar.toml"]
