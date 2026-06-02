# seshd — Seshat's Archive MCP service (pgvector RAG + synthesis)
#
# Single-arch (amd64). Uses the legacy docker builder so it works on plain
# Solstice runner VMs without buildx. Chart + HelmRelease pin pods to amd64
# nodes, so cross-compile complexity is unnecessary.
#
# Build:  docker build -f docker/seshd.Dockerfile -t seshd .
# Run:    docker run -p 9090:9090 -v seshat-config:/config seshd

ARG RUST_VERSION=1.94.1
ARG FEATURES="embedding,synthesis"

# ── Stage 1: Build ────────────────────────────────────────────────────────
FROM rust:${RUST_VERSION}-bookworm AS builder

ARG FEATURES

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

# Pre-build dependencies with the stub entry points. `|| true` so the stub
# phase doesn't fail the build if anything goes sideways.
RUN cargo build --release -p seshat --bin seshd --features "${FEATURES}" || true

# Copy full source and build the real binary.
# `touch` forces cargo to re-evaluate mtimes (docker COPY preserves them and
# can leave cargo thinking the stub sources are still current). `rm -f` on
# the cached stub binary is belt-and-suspenders so we never cp the stub.
COPY . .
RUN find src crates benches -name '*.rs' -exec touch {} + && \
    rm -f target/release/seshd && \
    cargo build --release -p seshat --bin seshd --features "${FEATURES}" && \
    cp target/release/seshd /build/seshd

# ── Stage 2: ONNX Runtime (for sentence embedding) ───────────────────────
FROM debian:bookworm-slim AS ort-fetch

ARG ORT_VERSION=1.23.2

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" | \
    tar xzf - --strip-components=1 -C /tmp && \
    mkdir -p /opt/ort/lib && \
    cp /tmp/lib/libonnxruntime.so* /opt/ort/lib/

# ── Stage 3: Runtime ─────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

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
