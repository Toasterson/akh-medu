# syntax=docker/dockerfile:1
# akhomed — akh-medu daemon with full NLU + LLM + MCP
#
# Build:  docker buildx build -f docker/akhomed.Dockerfile -t akhomed .
# Run:    docker run -p 8200:8200 -v akh-data:/data akhomed

ARG RUST_VERSION=1.94.1
ARG FEATURES="server,nlu-ml,nlu-candle,delegation,email,calendar"

# ── Stage 1: Build ────────────────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM rust:${RUST_VERSION}-bookworm AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG FEATURES

# oxigraph (oxrocksdb-sys) requires a C++ compiler and clang for RocksDB.
# Cross-compilation tools for ARM when building on x86.
RUN apt-get update && \
    apt-get install -y --no-install-recommends g++ clang libclang-dev && \
    case "$TARGETPLATFORM" in \
      linux/arm64) \
        apt-get install -y --no-install-recommends \
          gcc-aarch64-linux-gnu g++-aarch64-linux-gnu libc6-dev-arm64-cross && \
        rustup target add aarch64-unknown-linux-gnu ;; \
      linux/amd64) \
        rustup target add x86_64-unknown-linux-gnu && \
        # Cross toolchain, but only when the build host is not already amd64.
        # Building linux/amd64 on an arm64 host (an Apple Silicon Mac, armini)
        # needs an x86_64 linker: several deps have C components (ring, rocksdb
        # via oxigraph) whose cc-rs probe fails with ToolNotFound otherwise.
        # The arm64 branch above has always installed its cross toolchain; this
        # branch did not, which is why multi-arch builds only ever worked from
        # an amd64 host.
        if [ "$BUILDPLATFORM" != "linux/amd64" ]; then \
          dpkg --add-architecture amd64 && apt-get update && \
          apt-get install -y --no-install-recommends \
            gcc-x86-64-linux-gnu g++-x86-64-linux-gnu libc6-dev-amd64-cross \
            libssl-dev:amd64 ; \
        fi ;; \
    esac && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy manifests first for dependency caching.
COPY Cargo.toml Cargo.lock rust-toolchain.toml* ./
COPY crates/seshat/Cargo.toml crates/seshat/Cargo.toml
RUN mkdir -p src/bin crates/seshat/src benches && \
    echo 'fn main() {}' > src/main.rs && \
    echo 'fn main() {}' > src/bin/akhomed.rs && \
    echo 'fn main() {}' > crates/seshat/src/main.rs && \
    touch crates/seshat/src/lib.rs && \
    touch src/lib.rs && \
    echo 'fn main() {}' > benches/vsa_ops.rs && \
    echo 'fn main() {}' > benches/phase26.rs

# Pre-build dependencies (cached per target platform).
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM} \
    --mount=type=cache,target=/build/target,id=cargo-target-akhomed-${TARGETPLATFORM} \
    case "$TARGETPLATFORM" in \
      linux/arm64) \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc && \
        cargo build --release --features "${FEATURES}" --bin akhomed \
          --target aarch64-unknown-linux-gnu ;; \
      *) \
        if command -v x86_64-linux-gnu-gcc >/dev/null 2>&1; then \
          export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc \
                 CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc \
                 CXX_x86_64_unknown_linux_gnu=x86_64-linux-gnu-g++ \
                 PKG_CONFIG_ALLOW_CROSS=1 \
                 PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig \
                 PKG_CONFIG_SYSROOT_DIR=/ ; \
        fi && \
        cargo build --release --features "${FEATURES}" --bin akhomed \
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
    --mount=type=cache,target=/build/target,id=cargo-target-akhomed-${TARGETPLATFORM} \
    find src crates benches -name '*.rs' -exec touch {} + && \
    case "$TARGETPLATFORM" in \
      linux/arm64) \
        export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc && \
        rm -f target/aarch64-unknown-linux-gnu/release/akhomed && \
        cargo build --release --features "${FEATURES}" --bin akhomed \
          --target aarch64-unknown-linux-gnu && \
        cp target/aarch64-unknown-linux-gnu/release/akhomed /build/akhomed ;; \
      *) \
        if command -v x86_64-linux-gnu-gcc >/dev/null 2>&1; then \
          export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc \
                 CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc \
                 CXX_x86_64_unknown_linux_gnu=x86_64-linux-gnu-g++ \
                 PKG_CONFIG_ALLOW_CROSS=1 \
                 PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig \
                 PKG_CONFIG_SYSROOT_DIR=/ ; \
        fi && \
        rm -f target/x86_64-unknown-linux-gnu/release/akhomed && \
        cargo build --release --features "${FEATURES}" --bin akhomed \
          --target x86_64-unknown-linux-gnu && \
        cp target/x86_64-unknown-linux-gnu/release/akhomed /build/akhomed ;; \
    esac

# ── Stage 2: ONNX Runtime ────────────────────────────────────────────────
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
    groupadd -r akh && useradd -r -g akh -d /data -s /sbin/nologin akh && \
    mkdir -p /data /config && chown akh:akh /data /config

COPY --from=builder /build/akhomed /usr/local/bin/akhomed
COPY --from=ort-fetch /opt/ort/lib/ /usr/local/lib/

RUN ldconfig

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
    AKH_SERVER_BIND=0.0.0.0 \
    AKH_SERVER_PORT=8200 \
    AKH_HOME=/data \
    RUST_LOG=info

EXPOSE 8200

USER akh
WORKDIR /data

VOLUME ["/data", "/config"]

ENTRYPOINT ["tini", "--"]
CMD ["akhomed"]
