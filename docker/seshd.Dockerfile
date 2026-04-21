# syntax=docker/dockerfile:1
# seshd — Seshat's Archive MCP service (pgvector RAG + synthesis)
#
# Build:  docker buildx build -f docker/seshd.Dockerfile -t seshd .
# Run:    docker run -p 9090:9090 -v seshat-config:/config seshd

ARG RUST_VERSION=1.94.1
ARG FEATURES="embedding,synthesis"

# ── Stage 1: Build ────────────────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM rust:${RUST_VERSION}-bookworm AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG FEATURES

RUN case "$TARGETPLATFORM" in \
      linux/arm64) \
        dpkg --add-architecture arm64 && \
        apt-get update && \
        apt-get install -y gcc-aarch64-linux-gnu libc6-dev-arm64-cross && \
        rustup target add aarch64-unknown-linux-gnu ;; \
      linux/amd64) \
        rustup target add x86_64-unknown-linux-gnu ;; \
    esac

WORKDIR /build

# Copy manifests first for dependency caching.
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

# Pre-build dependencies.
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM} \
    --mount=type=cache,target=/build/target,id=cargo-target-seshd-${TARGETPLATFORM} \
    case "$TARGETPLATFORM" in \
      linux/arm64) \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc && \
        cargo build --release -p seshat --bin seshd --features "${FEATURES}" \
          --target aarch64-unknown-linux-gnu ;; \
      *) \
        cargo build --release -p seshat --bin seshd --features "${FEATURES}" \
          --target x86_64-unknown-linux-gnu ;; \
    esac || true

# Copy full source.
COPY . .

# Build the real binary.
# NOTE: `touch` forces cargo to re-evaluate source mtimes (COPY preserves them,
# which can leave cargo thinking the stub binaries from the previous layer are
# still up-to-date). `rm -f` on the binary is belt-and-suspenders against the
# cache-mount retaining the stub artefact.
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM} \
    --mount=type=cache,target=/build/target,id=cargo-target-seshd-${TARGETPLATFORM} \
    find src crates benches -name '*.rs' -exec touch {} + && \
    case "$TARGETPLATFORM" in \
      linux/arm64) \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc && \
        rm -f target/aarch64-unknown-linux-gnu/release/seshd && \
        cargo build --release -p seshat --bin seshd --features "${FEATURES}" \
          --target aarch64-unknown-linux-gnu && \
        cp target/aarch64-unknown-linux-gnu/release/seshd /build/seshd ;; \
      *) \
        rm -f target/x86_64-unknown-linux-gnu/release/seshd && \
        cargo build --release -p seshat --bin seshd --features "${FEATURES}" \
          --target x86_64-unknown-linux-gnu && \
        cp target/x86_64-unknown-linux-gnu/release/seshd /build/seshd ;; \
    esac

# ── Stage 2: ONNX Runtime (for sentence embedding) ───────────────────────
FROM --platform=$TARGETPLATFORM debian:bookworm-slim AS ort-fetch

ARG TARGETPLATFORM
ARG ORT_VERSION=1.23.2

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN case "$TARGETPLATFORM" in \
      linux/arm64) ORT_ARCH="linux-aarch64" ;; \
      *)           ORT_ARCH="linux-x64" ;; \
    esac && \
    curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_ARCH}-${ORT_VERSION}.tgz" | \
    tar xzf - --strip-components=1 -C /tmp && \
    mkdir -p /opt/ort/lib && \
    cp /tmp/lib/libonnxruntime.so* /opt/ort/lib/

# ── Stage 3: Runtime ─────────────────────────────────────────────────────
FROM --platform=$TARGETPLATFORM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r seshat && useradd -r -g seshat -d /data -s /sbin/nologin seshat && \
    mkdir -p /data /config /models && chown seshat:seshat /data /config /models

COPY --from=builder /build/seshd /usr/local/bin/seshd
COPY --from=ort-fetch /opt/ort/lib/ /usr/local/lib/

RUN ldconfig

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
    SESHAT_CONFIG=/config/seshat.toml \
    SESHAT_MODEL_DIR=/models \
    RUST_LOG=seshat=info

EXPOSE 9090

USER seshat
WORKDIR /data

VOLUME ["/data", "/config", "/models"]

ENTRYPOINT ["tini", "--"]
CMD ["seshd", "/config/seshat.toml"]
