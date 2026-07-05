# syntax=docker/dockerfile:1
# akhomed-cloud — a LEAN cloud-akh image (ADR 049/070): the ACP daemon + Postgres
# backing + Ollama chat backing, no Candle/GPU and no ONNX. Cloud akhs (Tecton,
# Thoth, an OpenIndiana-zone akh driven by Anima) call ollama.com, so they don't
# need a local model — keeping the image small and fast to build for both arches.
#
# The truth lives in the shared akh-pg Postgres (mTLS); this container is stateless
# compute that can run anywhere it can reach Postgres + Ollama.
#
# Build (multi-arch, cross-compiled on one amd64 builder — e.g. hephaistos):
#   docker buildx build -f docker/akhomed-cloud.Dockerfile \
#     --platform linux/amd64,linux/arm64 \
#     -t code.aopc.cloud/toasterson/akhomed:cloud --push .
# Run (see docs/ops/akh-cloud-container.md):
#   docker run -d --name akh-thoth \
#     -v $PWD/config:/config:ro -v akh-thoth-data:/data \
#     -p 8201:8201 code.aopc.cloud/toasterson/akhomed:cloud

ARG RUST_VERSION=1.94.1
# Lean feature set: daemon + ACP + Postgres + delegation. No nlu-candle (Candle),
# no nlu-ml (ONNX). sea-orm/sqlx use rustls and ureq uses rustls, so no OpenSSL.
ARG FEATURES="server,delegation"

# ── Stage 1: Build (native amd64 builder, cross-links for arm64) ───────────
FROM --platform=$BUILDPLATFORM rust:${RUST_VERSION}-bookworm AS builder
ARG TARGETPLATFORM
ARG FEATURES

# oxigraph (oxrocksdb-sys) needs a C/C++ compiler + clang; arm64 needs the cross
# toolchain so the amd64 builder can link aarch64 objects without QEMU.
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ clang libclang-dev && \
    case "$TARGETPLATFORM" in \
      linux/arm64) \
        apt-get install -y --no-install-recommends \
          gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross && \
        rustup target add aarch64-unknown-linux-gnu ;; \
      *) rustup target add x86_64-unknown-linux-gnu ;; \
    esac && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Manifests first for dependency caching (mirror the workspace's crate stubs).
COPY Cargo.toml Cargo.lock rust-toolchain.toml* ./
COPY crates/seshat/Cargo.toml crates/seshat/Cargo.toml
COPY crates/akhomed-store/Cargo.toml crates/akhomed-store/Cargo.toml
COPY crates/heka/Cargo.toml crates/heka/Cargo.toml
COPY crates/anima-ai/Cargo.toml crates/anima-ai/Cargo.toml
RUN mkdir -p src/bin crates/seshat/src crates/akhomed-store/src crates/heka/src crates/anima-ai/src benches && \
    echo 'fn main() {}' > src/main.rs && \
    echo 'fn main() {}' > src/bin/akhomed.rs && \
    echo 'fn main() {}' > crates/seshat/src/main.rs && \
    touch src/lib.rs crates/seshat/src/lib.rs crates/akhomed-store/src/lib.rs \
          crates/heka/src/lib.rs crates/anima-ai/src/lib.rs && \
    echo 'fn main() {}' > benches/vsa_ops.rs && \
    echo 'fn main() {}' > benches/phase26.rs

ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
# Resolve the rust target triple once.
RUN case "$TARGETPLATFORM" in \
      linux/arm64) echo aarch64-unknown-linux-gnu ;; \
      *)           echo x86_64-unknown-linux-gnu  ;; \
    esac > /tmp/triple

# Pre-build dependencies (cached per target platform).
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM} \
    --mount=type=cache,target=/build/target,id=cargo-target-akhomed-cloud-${TARGETPLATFORM} \
    cargo build --release --no-default-features --features "${FEATURES}" \
      --bin akhomed --target "$(cat /tmp/triple)" || true

# Real build.
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM} \
    --mount=type=cache,target=/build/target,id=cargo-target-akhomed-cloud-${TARGETPLATFORM} \
    find src crates benches -name '*.rs' -exec touch {} + && \
    T="$(cat /tmp/triple)" && rm -f "target/$T/release/akhomed" && \
    cargo build --release --no-default-features --features "${FEATURES}" \
      --bin akhomed --target "$T" && \
    cp "target/$T/release/akhomed" /build/akhomed

# ── Stage 2: Runtime ───────────────────────────────────────────────────────
FROM --platform=$TARGETPLATFORM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r akh && useradd -r -g akh -d /data -s /sbin/nologin akh && \
    mkdir -p /data /config && chown akh:akh /data /config

COPY --from=builder /build/akhomed /usr/local/bin/akhomed

# Config (config.toml, workspaces/, the mTLS cert bundle) is mounted at /config;
# durable cache/snapshots at /data. ACP is exposed over TCP for Anima to reach.
ENV XDG_CONFIG_HOME=/config \
    XDG_DATA_HOME=/data \
    XDG_STATE_HOME=/data/state \
    XDG_CACHE_HOME=/data/cache \
    XDG_RUNTIME_DIR=/data/run \
    RUST_LOG=akh_medu=info

EXPOSE 8201
USER akh
WORKDIR /data
VOLUME ["/data", "/config"]

# fd headroom: docker's default soft RLIMIT_NOFILE is 1024, which a long-lived
# ACP daemon can exhaust (2026-06 archibald incident). akhomed raises its own
# soft limit to the hard limit at startup, so no --ulimit flag is needed as
# long as the runtime's HARD limit is generous (dockerd default: 524288).

ENTRYPOINT ["tini", "--"]
CMD ["akhomed"]

# ── Stage 3: Implementer runtime (akhomed + coding toolchain) ───────────────
# For implementer akhs that build/test what they edit (Anima worktrees under
# their scratch/): git + Node/npm + a Rust toolchain + sccache, so `cargo
# build`/`npm test` work inside the akh's shell tool. sccache shares the same
# S3 compile cache as the anima-runner fleet — inject SCCACHE_BUCKET/ENDPOINT/
# REGION + AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY and RUSTC_WRAPPER=sccache
# at run time (see anima's docker-compose.runner.yml runner-env for values).
# The default `runtime` stage stays lean; build this variant with
#   docker buildx build -f docker/akhomed-cloud.Dockerfile --target implementer \
#     -t akhomed:cloud-tools --load .
FROM runtime AS implementer
USER root
ARG SCCACHE_VERSION=v0.10.0
# Toolchain lives read-only in /usr/local; cargo's writable state (registry,
# git checkouts) goes to the /data volume at runtime (uid 999 = akh).
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:/usr/local/bin:/usr/bin:/bin
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      git curl build-essential pkg-config libssl-dev nodejs npm && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://sh.rustup.rs | sh -s -- -y \
      --default-toolchain stable --profile minimal -c rustfmt -c clippy && \
    chmod -R a+rX "${RUSTUP_HOME}" "${CARGO_HOME}" && \
    ARCH="$(dpkg --print-architecture)" && \
    case "$ARCH" in \
      arm64) T=aarch64-unknown-linux-musl ;; \
      *)     T=x86_64-unknown-linux-musl  ;; \
    esac && \
    curl -fsSL "https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/sccache-${SCCACHE_VERSION}-${T}.tar.gz" \
      | tar xz -C /tmp && \
    install -m 0755 "/tmp/sccache-${SCCACHE_VERSION}-${T}/sccache" /usr/local/bin/sccache && \
    rm -rf /tmp/sccache-*
ENV CARGO_HOME=/data/cache/cargo
USER akh
